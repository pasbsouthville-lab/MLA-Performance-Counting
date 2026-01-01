import streamlit as st
import pandas as pd
import calendar
import os
import io
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Pressure Performance Analyzer", layout="wide")
st.title("📈 Multi-Device Pressure Performance Analyzer")
st.markdown("Upload one or more monthly pressure CSV files to view performance by day, week, and month. PDF includes performance chart + colored tables.")

# -----------------------
# Helper: process one CSV (robust timestamp detection)
# -----------------------
def process_file(file):
    """
    Robust CSV loader which:
      - Detects and skips metadata before the real header (looks for line starting with "DATA ID")
      - Loads the CSV table
      - Detects the timestamp column (common names or by attempting to parse columns)
      - Keeps only rows where pressure exists
      - Extracts packetloss and rssi when present
      - Produces daily, weekly, monthly aggregates with Performance (%) and rounding
    Returns: daily_df, weekly_df, monthly_df, missing_dates_list, device_name
    """

    # --- read file content safely
    try:
        raw_text = file.getvalue().decode("utf-8", errors="ignore").splitlines()
    except Exception:
        # fallback: try to let pandas read the file object directly
        try:
            df_try = pd.read_csv(file, on_bad_lines="skip")
            raw_text = None
        except Exception as e:
            st.error(f"❌ Could not read file {file.name}: {e}")
            return None, None, None, None, None

    # --- detect header line (prefer line that starts with "DATA ID") ---
    header_idx = None
    if raw_text is not None:
        for i, line in enumerate(raw_text):
            if line.strip().upper().startswith("DATA ID"):
                header_idx = i
                break

    try:
        if header_idx is not None:
            csv_text = "\n".join(raw_text[header_idx:])
            df = pd.read_csv(pd.io.common.StringIO(csv_text), on_bad_lines="skip")
        else:
            # No DATA ID header found. Try reading whole file (pandas will use first line as header).
            # Use previously attempted df_try if available
            if raw_text is not None:
                csv_text = "\n".join(raw_text)
                df = pd.read_csv(pd.io.common.StringIO(csv_text), on_bad_lines="skip")
            else:
                df = df_try
    except Exception as e:
        st.error(f"❌ Error parsing CSV table in {file.name}: {e}")
        return None, None, None, None, None

    # Normalize column names (strip, remove non-breaking chars)
    df.columns = [str(c).strip().replace("\u00A0", " ") for c in df.columns]

    # --- Detect timestamp column robustly ---
    timestamp_candidates = []
    # common names to check first (in order)
    common_ts_names = ["timestamp", "time", "export time", "export_time", "date", "datetime", "exporttime"]

    cols_lower = [c.lower() for c in df.columns]
    for name in common_ts_names:
        for orig_col, low in zip(df.columns, cols_lower):
            if name == low or name in low:
                timestamp_candidates.append(orig_col)

    # if found candidate(s) by name, pick first
    timestamp_col = timestamp_candidates[0] if timestamp_candidates else None

    # fallback: try to parse each column and pick the one that yields the most non-null datetimes
    if timestamp_col is None:
        best_col = None
        best_count = 0
        nrows = len(df)
        for col in df.columns:
            # skip numeric-only columns early
            try:
                parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            except Exception:
                parsed = pd.Series([pd.NaT] * nrows)
            non_null = parsed.notna().sum()
            if non_null > best_count and non_null >= max(1, int(0.1 * nrows)):  # require at least 10% parse rate
                best_count = non_null
                best_col = col
        timestamp_col = best_col

    if timestamp_col is None:
        st.error(f"❌ Timestamp column not found in {file.name}. Tried common names and parsing each column.")
        return None, None, None, None, None

    # --- Detect pressure column (required) ---
    pressure_col = None
    for c in df.columns:
        if "pressure" in c.lower():
            pressure_col = c
            break
    if pressure_col is None:
        st.error(f"❌ Pressure column (e.g., 'PRESSURE [kPa]') not found in {file.name}.")
        return None, None, None, None, None

    # --- Parse timestamp column to datetime and filter only valid pressure rows ---
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=[timestamp_col, pressure_col]).copy()
    if df.empty:
        st.warning(f"⚠️ No valid rows with both timestamp and pressure in {file.name}.")
        return None, None, None, None, None

    df = df.sort_values(timestamp_col)

    # --- Extract device name from filename (fallback to device column if present) ---
    device_name = os.path.splitext(os.path.basename(file.name))[0].split("_")[0]
    # If there is a device column, prefer it (first match for 'device' in header)
    for c in df.columns:
        if "device" in c.lower():
            try:
                possible = df[c].dropna().astype(str).iloc[0]
                if possible:
                    device_name = possible
                    break
            except Exception:
                pass

    # --- Add date/week/month fields ---
    df["Date_dt"] = df[timestamp_col].dt.floor("D")  # keep as datetime
    df["Date"] = df["Date_dt"].dt.date
    df["Week"] = df[timestamp_col].dt.isocalendar().week
    df["Year"] = df[timestamp_col].dt.isocalendar().year
    df["Month"] = df[timestamp_col].dt.to_period("M").astype(str)

    # choose month metadata from data (first month present)
    month_label = df["Month"].unique()[0]
    year_month = month_label.split("-")
    year = int(year_month[0]); month = int(year_month[1])
    days_in_month = calendar.monthrange(year, month)[1]
    expected_monthly = 144 * days_in_month

    # --- PacketLoss and RSSI columns detection ---
    packet_col = None
    rssi_col = None
    for c in df.columns:
        cl = c.lower()
        if "packetloss" in cl or "packet_loss" in cl or "packet loss" in cl:
            packet_col = c
        if "rssi" in cl:
            rssi_col = c

    # --- DAILY aggregation ---
    # count pressure records per date
    daily = df.groupby("Date_dt").agg(
        Actual_Count=(pressure_col, "count"),
        Avg_Packet_Loss=(packet_col, "mean") if packet_col else (pressure_col, lambda s: pd.NA),
        Avg_RSSI=(rssi_col, "mean") if rssi_col else (pressure_col, lambda s: pd.NA)
    ).reset_index()

    # Round and fill NA
    if "Avg_Packet_Loss" in daily.columns:
        daily["Avg_Packet_Loss"] = daily["Avg_Packet_Loss"].round(2).fillna(0.0)
    else:
        daily["Avg_Packet_Loss"] = 0.0
    if "Avg_RSSI" in daily.columns:
        daily["Avg_RSSI"] = daily["Avg_RSSI"].round(2).fillna(0.0)
    else:
        daily["Avg_RSSI"] = 0.0

    # Build full month day list (1..last day of that month)
    all_days = pd.DataFrame({"Date_dt": pd.date_range(start=f"{year}-{month:02d}-01", periods=days_in_month, freq="D")})
    all_days["Device"] = device_name
    daily["Device"] = device_name

    # Ensure types for merging
    daily["Date_dt"] = pd.to_datetime(daily["Date_dt"])
    all_days["Date_dt"] = pd.to_datetime(all_days["Date_dt"])

    # Merge to ensure every day of the month present
    daily = pd.merge(all_days, daily, on=["Date_dt", "Device"], how="left").fillna({
        "Actual_Count": 0,
        "Avg_Packet_Loss": 0.0,
        "Avg_RSSI": 0.0
    })

    # Expected count per day is 144 (samples) — keep as per your earlier code
    daily["Expected Count"] = 144
    daily["Performance (%)"] = ((daily["Actual_Count"] / daily["Expected Count"]) * 100).round(2)

    # Convert Date for display as dd/mm/yy as you requested
    daily["Date"] = daily["Date_dt"].dt.strftime("%d/%m/%y")

    # Missing dates
    missing_dates = daily.loc[daily["Actual_Count"] == 0, "Date"].tolist()

    # --- WEEKLY aggregation ---
    # Build weekly raw from original df (not from daily - to avoid date formatting issues)
    weekly_raw = df.groupby(["Year", "Week"]).agg(
        Actual_Count=(pressure_col, "count"),
        Avg_Packet_Loss=(packet_col, "mean") if packet_col else (pressure_col, lambda s: pd.NA),
        Avg_RSSI=(rssi_col, "mean") if rssi_col else (pressure_col, lambda s: pd.NA)
    ).reset_index()

    weekly_list = []
    for _, row in weekly_raw.iterrows():
        yr = int(row["Year"])
        wk = int(row["Week"])
        # compute week start (Monday) using ISO calendar
        try:
            week_start = datetime.fromisocalendar(yr, wk, 1)
        except Exception:
            # fallback: use first day of month
            week_start = datetime(year, month, 1)
        week_days = [week_start + timedelta(days=i) for i in range(7)]
        # count days that belong to the target month
        days_in_target_month = [d for d in week_days if d.month == month and d.year == year]
        expected = len(days_in_target_month) * 144
        # if all 7 days in month: expected = 1008
        if len(days_in_target_month) == 7:
            expected = 7 * 144

        avg_pl = round(row["Avg_Packet_Loss"], 2) if pd.notna(row.get("Avg_Packet_Loss")) else 0.0
        avg_rssi = round(row["Avg_RSSI"], 2) if pd.notna(row.get("Avg_RSSI")) else 0.0
        actual = int(row["Actual_Count"])
        perf = round((actual / expected) * 100, 2) if expected > 0 else 0.0

        weekly_list.append({
            "Week_Label": f"{yr}-W{wk}",
            "Device": device_name,
            "Actual Count": actual,
            "Expected Count": int(expected),
            "Performance (%)": perf,
            "Avg Packet Loss (%)": avg_pl,
            "Avg RSSI (dBm)": avg_rssi
        })

    weekly = pd.DataFrame(weekly_list)

    # --- MONTHLY aggregation ---
    avg_pl_month = round(df[packet_col].mean(), 2) if (packet_col and packet_col in df.columns and df[packet_col].notna().any()) else 0.0
    avg_rssi_month = round(df[rssi_col].mean(), 2) if (rssi_col and rssi_col in df.columns and df[rssi_col].notna().any()) else 0.0

    monthly = pd.DataFrame([{
        "Device": device_name,
        "Month": month_label,
        "Actual Count": len(df),
        "Expected Count": expected_monthly,
        "Performance (%)": round((len(df) / expected_monthly) * 100, 2) if expected_monthly > 0 else 0.0,
        "Loss (%)": round(100 - (len(df) / expected_monthly) * 100, 2) if expected_monthly > 0 else 0.0,
        "Avg Packet Loss (%)": avg_pl_month,
        "Avg RSSI (dBm)": avg_rssi_month
    }])

    # Final column tidy: ensure columns exist and have expected names
    # (so PDF generator won't fail on missing names)
    # daily: Date, Device, Actual_Count, Expected Count, Performance (%), Avg_Packet_Loss, Avg_RSSI
    # weekly/monthly: already set

    return daily, weekly, monthly, missing_dates, device_name


# -----------------------
# Helper: DataFrame -> JPG (for Streamlit downloads)
# -----------------------
def dataframe_to_jpg(df, title):
    fig, ax = plt.subplots(figsize=(10, max(2, len(df) * 0.25 + 1)))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    plt.title(title, fontsize=12, pad=10)
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg", bbox_inches="tight", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf


# -----------------------
# Helper: Generate PDF with chart + colored tables
# -----------------------
def generate_pdf(daily_df, weekly_df, monthly_df):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    # Header
    content.append(Paragraph("📊 Pressure Performance Summary Report", styles["Title"]))
    content.append(Spacer(1, 12))

    # -----------------------
    # Legend table (first page)
    # -----------------------
    content.append(Paragraph("<b>Legend / Notes</b>", styles["Heading2"]))
    content.append(Spacer(1, 6))

    legend_data = [
        ["Color / Word", "Meaning"],
        [Paragraph('<font color="lightgreen">Light Green</font>', styles["Normal"]), "Performance ≥ 90%"],
        [Paragraph('<font color="yellow">Yellow</font>', styles["Normal"]), "Performance 70% - 90%"],
        [Paragraph('<font color="salmon">Salmon</font>', styles["Normal"]), "Performance < 70%"],
        [Paragraph('<font color="lightgrey">Grey</font>', styles["Normal"]), "Actual Count = 0 (FDD)"],
        ["FDD", "Full Day Disconnected"],
        ["PDD", "Partial Day Disconnected (Performance < 90%)"]
    ]

    legend_table = Table(legend_data, colWidths=[120, 300])
    legend_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.25, colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('BACKGROUND', (0,1), (0,1), colors.lightgreen),
        ('BACKGROUND', (0,2), (0,2), colors.yellow),
        ('BACKGROUND', (0,3), (0,3), colors.salmon),
        ('BACKGROUND', (0,4), (0,4), colors.lightgrey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    content.append(legend_table)
    content.append(Spacer(1, 12))

    # -----------------------
    # Monthly table
    # -----------------------
    content.append(Paragraph("<b>1. Monthly Summary Comparison</b>", styles["Heading2"]))
    content.append(Spacer(1, 8))

    # Sort monthly table using same logic as chart
    try:
        monthly_sorted_table = monthly_df.copy()
        monthly_sorted_table["DeviceSort"] = monthly_sorted_table["Device"].apply(
            lambda x: int(''.join(filter(str.isdigit, str(x)))) if any(ch.isdigit() for ch in str(x)) else str(x)
        )
        monthly_sorted_table = monthly_sorted_table.sort_values("DeviceSort")
        monthly_sorted_table = monthly_sorted_table.drop(columns=["DeviceSort"])
    except:
        monthly_sorted_table = monthly_df.copy().sort_values("Device")

    # Build table
    table_data = [monthly_sorted_table.columns.tolist()] + monthly_sorted_table.values.tolist()

    t = Table(table_data, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
    ]))

    content.append(t)
    content.append(Spacer(1, 12))

    # -----------------------
    # Monthly performance bar chart (Sorted Low → High by Device ID)
    # -----------------------
    # Make a sorted copy for chart only
    try:
        monthly_sorted = monthly_df.copy()
        # Attempt numeric sorting if device looks like number
        monthly_sorted["DeviceSort"] = monthly_sorted["Device"].apply(
            lambda x: int(''.join(filter(str.isdigit, str(x)))) if any(ch.isdigit() for ch in str(x)) else str(x)
        )
        monthly_sorted = monthly_sorted.sort_values("DeviceSort")
    except:
        monthly_sorted = monthly_df.copy().sort_values("Device")

    # Extract sorted lists
    devices = monthly_sorted["Device"].tolist()
    perf = monthly_sorted["Performance (%)"].tolist()

    # Plot chart
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(devices, perf)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Performance (%)")
    ax.set_title("Monthly Performance (%) by Device")

    for bar, v in zip(bars, perf):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.2f}%", 
                ha="center", va="bottom", fontsize=8)

    chart_buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(chart_buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    chart_buf.seek(0)
    content.append(Image(chart_buf, width=440, height=220))
    content.append(PageBreak())

    # -----------------------
    # Daily & Weekly tables per device
    # -----------------------
    for device in sorted(daily_df["Device"].unique()):
        content.append(Paragraph(f"📟 Device: <b>{device}</b>", styles["Heading3"]))
        content.append(Spacer(1, 6))

        # --- Daily ---
        df_day = daily_df[daily_df["Device"] == device].copy()

        # Auto remark rules
        def compute_remark(row):
            if row["Actual_Count"] == 0:
                return "FDD"
            elif row["Performance (%)"] < 90:
                return "PDD"
            else:
                return ""
        df_day["Remark"] = df_day.apply(compute_remark, axis=1)

        df_day_display = df_day.copy()
        df_day_display["Avg Packet Loss (%)"] = df_day_display["Avg_Packet_Loss"].round(2)
        df_day_display["Avg RSSI (dBm)"] = df_day_display["Avg_RSSI"].round(2)

        df_day_display = df_day_display[[
            "Date", "Actual_Count", "Expected Count", "Performance (%)",
            "Avg Packet Loss (%)", "Avg RSSI (dBm)", "Remark"
        ]]

        day_table_data = [df_day_display.columns.tolist()] + df_day_display.values.tolist()
        day_table = Table(day_table_data, repeatRows=1)

        day_style = TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ])

        for i in range(len(df_day)):
            row_idx = i + 1
            actual = int(df_day.iloc[i].get("Actual_Count", 0))
            perf_val = float(df_day.iloc[i].get("Performance (%)", 0.0))

            if actual == 0:
                day_style.add('BACKGROUND', (0, row_idx), (-1, row_idx), colors.lightgrey)
            else:
                if perf_val >= 90:
                    day_style.add('BACKGROUND', (0, row_idx), (-1, row_idx), colors.lightgreen)
                elif perf_val >= 70:
                    day_style.add('BACKGROUND', (0, row_idx), (-1, row_idx), colors.yellow)
                else:
                    day_style.add('BACKGROUND', (0, row_idx), (-1, row_idx), colors.salmon)

        day_table.setStyle(day_style)
        content.append(day_table)
        content.append(Spacer(1, 12))

        # --- Weekly ---
        df_week = weekly_df[weekly_df["Device"] == device].copy()
        df_week["Avg Packet Loss (%)"] = df_week["Avg Packet Loss (%)"].round(2)
        df_week["Avg RSSI (dBm)"] = df_week["Avg RSSI (dBm)"].round(2)

        week_display = df_week[[
            "Week_Label", "Actual Count", "Expected Count",
            "Performance (%)", "Avg Packet Loss (%)", "Avg RSSI (dBm)"
        ]]

        week_table_data = [week_display.columns.tolist()] + week_display.values.tolist()
        week_table = Table(week_table_data, repeatRows=1)

        week_style = TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ])

        for i in range(len(df_week)):
            row_idx = i + 1
            actual = int(df_week.iloc[i].get("Actual Count", 0))
            perf_val = float(df_week.iloc[i].get("Performance (%)", 0.0))

            if actual == 0:
                week_style.add('BACKGROUND', (0, row_idx), (-1, row_idx), colors.lightgrey)
            else:
                if perf_val >= 90:
                    week_style.add('BACKGROUND', (0, row_idx), (-1, row_idx), colors.lightgreen)
                elif perf_val >= 70:
                    week_style.add('BACKGROUND', (0, row_idx), (-1, row_idx), colors.yellow)
                else:
                    week_style.add('BACKGROUND', (0, row_idx), (-1, row_idx), colors.salmon)

        week_table.setStyle(week_style)
        content.append(Paragraph("<b>Weekly Performance</b>", styles["Heading4"]))
        content.append(Spacer(1, 6))
        content.append(week_table)
        content.append(PageBreak())

    doc.build(content)
    buf.seek(0)
    return buf

# -----------------------
# Streamlit main
# -----------------------
uploaded_files = st.file_uploader("📤 Upload one or more CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    unique = {}
    duplicates = []
    for f in uploaded_files:
        if f.name not in unique:
            unique[f.name] = f
        else:
            duplicates.append(f.name)
    if duplicates:
        st.warning(f"⚠️ Duplicate files ignored: {', '.join(duplicates)}")

    files = list(unique.values())

    all_daily, all_weekly, all_monthly = [], [], []
    missing_map = {}

    for f in files:
        daily, weekly, monthly, missing_dates, device_name = process_file(f)
        if daily is not None:
            all_daily.append(daily)
            all_weekly.append(weekly)
            all_monthly.append(monthly)
            missing_map[device_name] = missing_dates

    if not all_daily:
        st.error("❌ No valid CSV files processed.")
        st.stop()

    df_daily = pd.concat(all_daily, ignore_index=True)
    df_weekly = pd.concat(all_weekly, ignore_index=True)
    df_monthly = pd.concat(all_monthly, ignore_index=True)

    # Make display-friendly column names consistent for Streamlit
    df_daily_display = df_daily.copy()
    df_daily_display = df_daily_display.rename(columns={
        "Actual_Count": "Actual Count",
        "Avg_Packet_Loss": "Avg Packet Loss (%)",
        "Avg_RSSI": "Avg RSSI (dBm)",
        "Expected Count": "Expected Count",
        "Performance (%)": "Performance (%)"
    })

    df_weekly_display = df_weekly.copy()
    df_monthly_display = df_monthly.copy()

    device_list = sorted(df_monthly_display["Device"].unique())
    selected = st.selectbox("🔍 Select device to view details:", device_list)

    st.subheader(f"📅 Daily Performance — {selected}")
    st.dataframe(df_daily_display[df_daily_display["Device"] == selected].reset_index(drop=True), use_container_width=True, height=300)

    st.subheader("📆 Weekly Performance")
    st.dataframe(df_weekly_display[df_weekly_display["Device"] == selected].reset_index(drop=True), use_container_width=True, height=300)

    st.subheader("🗓 Monthly Comparison (all devices)")
    st.dataframe(df_monthly_display.reset_index(drop=True), use_container_width=True, height=250)

    pdf_buf = generate_pdf(df_daily, df_weekly, df_monthly)
    st.download_button("📄 Download PDF Report (with chart + colored tables)", data=pdf_buf, file_name="Performance_Report.pdf", mime="application/pdf")

    # JPG downloads for Streamlit table images
    st.download_button(
        label="🖼️ Download Selected Daily Table (JPG)",
        data=dataframe_to_jpg(df_daily_display[df_daily_display["Device"] == selected], f"{selected} Daily"),
        file_name=f"{selected}_Daily.jpg",
        mime="image/jpeg"
    )
    st.download_button(
        label="🖼️ Download Selected Weekly Table (JPG)",
        data=dataframe_to_jpg(df_weekly_display[df_weekly_display["Device"] == selected], f"{selected} Weekly"),
        file_name=f"{selected}_Weekly.jpg",
        mime="image/jpeg"
    )

else:
    st.info("👆 Upload one or more CSV files to begin analysis.") 

