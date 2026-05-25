# Modules

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Style for better plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# Paths and Parameters

# File paths
project_root = Path(r"C:\Users\Sean Kaylee\OneDrive\Desktop\EDS_4761_MANZANO")

# Data and output folders
data_dir = project_root / "data"
output_dir = project_root / "outputs"

# Create directories if they don't exist
data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Original data file
airquality_file = data_dir / "globalAirQuality.csv"

# Output cleaned file
cleaned_file = data_dir / "dataset_cleaned.csv"

# Define unique filtering parameters
COUNTRY = "US"
CITY = "Los Angeles"
START_DATE = "2025-11-04"
END_DATE = "2025-11-19"
START_HOUR = 0  # Full 24-hour cycle for diurnal analysis
END_HOUR = 23
MIN_O3 = 0  # Minimum valid ozone concentration


# Helper functions


def load_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return None


def filter_by_unique_logic(df):
    # Filter by country and city
    df = df[(df["country"] == COUNTRY) & (df["city"] == CITY)].copy()

    # Parse datetime
    df["DATETIME"] = pd.to_datetime(df["timestamp"])

    # Filter by date range
    df = df[(df["DATETIME"] >= START_DATE) & (df["DATETIME"] <= END_DATE)].copy()

    # Filter by hour range
    df["HOUR"] = df["DATETIME"].dt.hour
    df = df[(df["HOUR"] >= START_HOUR) & (df["HOUR"] <= END_HOUR)].copy()

    # Remove invalid ozone readings
    df = df[df["o3"] > MIN_O3].copy()

    # Extract date for daily grouping
    df["DATE"] = df["DATETIME"].dt.date

    return df


def clean_data(df):
    initial_rows = len(df)

    # Remove duplicates
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)

    # Check for missing values in key columns
    key_cols = ["o3", "no2", "pm25", "temperature", "humidity", "wind_speed"]
    for col in key_cols:
        if col in df.columns and df[col].isnull().any():
            # Fill with forward fill then median
            df[col] = df[col].fillna(method="ffill")
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    return df


def compute_statistics_numpy(df):
    # Overall Ozone statistics
    o3_values = df["o3"].values

    # Separate by time of day (peak vs night)
    peak_hours = df[(df["HOUR"] >= 12) & (df["HOUR"] <= 16)]["o3"].values
    night_hours = df[(df["HOUR"] >= 22) | (df["HOUR"] <= 5)]["o3"].values

    stats_results = {}

    # Overall O3 statistics
    stats_results["o3_mean"] = np.mean(o3_values)
    stats_results["o3_median"] = np.median(o3_values)
    stats_results["o3_std"] = np.std(o3_values)
    stats_results["o3_var"] = np.var(o3_values)
    stats_results["o3_min"] = np.min(o3_values)
    stats_results["o3_max"] = np.max(o3_values)
    stats_results["o3_q25"] = np.percentile(o3_values, 25)
    stats_results["o3_q75"] = np.percentile(o3_values, 75)

    # Peak vs Night comparison
    stats_results["peak_mean"] = np.mean(peak_hours) if len(peak_hours) > 0 else 0
    stats_results["night_mean"] = np.mean(night_hours) if len(night_hours) > 0 else 0
    stats_results["peak_night_ratio"] = (
        stats_results["peak_mean"] / stats_results["night_mean"]
        if stats_results["night_mean"] > 0
        else 0
    )

    # Correlation between O3 and other variables
    if "no2" in df.columns:
        valid_mask = df["o3"].notna() & df["no2"].notna()
        if valid_mask.sum() > 1:
            stats_results["o3_no2_correlation"] = np.corrcoef(
                df["o3"][valid_mask].values, df["no2"][valid_mask].values
            )[0, 1]
        else:
            stats_results["o3_no2_correlation"] = 0

    if "temperature" in df.columns:
        valid_mask = df["o3"].notna() & df["temperature"].notna()
        if valid_mask.sum() > 1:
            stats_results["o3_temp_correlation"] = np.corrcoef(
                df["o3"][valid_mask].values, df["temperature"][valid_mask].values
            )[0, 1]
        else:
            stats_results["o3_temp_correlation"] = 0

    if "humidity" in df.columns:
        valid_mask = df["o3"].notna() & df["humidity"].notna()
        if valid_mask.sum() > 1:
            stats_results["o3_humidity_correlation"] = np.corrcoef(
                df["o3"][valid_mask].values, df["humidity"][valid_mask].values
            )[0, 1]
        else:
            stats_results["o3_humidity_correlation"] = 0

    return stats_results


def calculate_skewness(data):
    n = len(data)
    if n < 2:
        return 0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    skewness = np.sum((data - mean) ** 3) / (n * std**3)
    return skewness


# TABLE 1: Sample Telemetry Records and Extracted Features
def create_sample_telemetry_table(df):
    """
    Generate Table 1: Sample Telemetry Records with 5 rows
    Saves as CSV and prints formatted table
    """
    print("─" * 110)
    print("  GENERATING TABLE 1: SAMPLE TELEMETRY RECORDS")
    print("─" * 110)

    # Select columns for the sample table
    table1_cols = [
        "country",
        "city",
        "DATETIME",
        "HOUR",
        "o3",
        "no2",
        "temperature",
        "humidity",
        "TIME_PERIOD",
    ]

    # Get first 5 rows as sample
    table1 = df[table1_cols].head(5).reset_index(drop=True)

    # Format datetime for better display
    table1["DATETIME"] = pd.to_datetime(table1["DATETIME"]).dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    # Rename columns for publication
    table1.columns = [
        "COUNTRY",
        "CITY",
        "DATETIME",
        "HOUR",
        "O3_ug_m3",
        "NO2_ug_m3",
        "TEMP_C",
        "HUM_pct",
        "TIME_PERIOD",
    ]

    # Save to CSV
    table1.to_csv(output_dir / "Table1_Sample_Records.csv", index=False)

    # Print formatted table
    print("\n  TABLE 1: Sample Telemetry Records and Extracted Features")
    print("  " + "=" * 110)
    print(
        f"  {'COUNTRY':<8} {'CITY':<12} {'DATETIME':<20} {'HOUR':<6} {'O3':<10} {'NO2':<10} {'TEMP':<8} {'HUM':<8} {'TIME PERIOD':<20}"
    )
    print("  " + "-" * 110)

    for _, row in table1.iterrows():
        print(
            f"  {row['COUNTRY']:<8} {row['CITY']:<12} {row['DATETIME']:<20} {row['HOUR']:<6} {row['O3_ug_m3']:<10.2f} {row['NO2_ug_m3']:<10.2f} {row['TEMP_C']:<8.1f} {row['HUM_pct']:<8.1f} {row['TIME_PERIOD']:<20}"
        )

    print("  " + "=" * 110)
    print(f"\n  Note: First 5 records shown. Total records: {len(df)}")

    return table1


# Visualization functions


# Boxplot
def create_diurnal_boxplot(df):
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for each hour
    box_data = []
    hours = range(24)
    for hour in hours:
        hour_data = df[df["HOUR"] == hour]["o3"].dropna().values
        box_data.append(hour_data)

    # Create boxplot
    bp = ax.boxplot(
        box_data,
        positions=hours,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(linewidth=2, color="darkred"),
    )

    # Color boxes by time of day
    for i, box in enumerate(bp["boxes"]):
        if 10 <= i <= 16:  # Peak sunlight hours
            box.set_facecolor("#FF9999")
            box.set_edgecolor("darkred")
        elif 6 <= i <= 9:  # Morning ramp-up
            box.set_facecolor("#FFCC99")
            box.set_edgecolor("orange")
        elif 17 <= i <= 19:  # Evening decline
            box.set_facecolor("#99CCFF")
            box.set_edgecolor("blue")
        else:  # Nighttime
            box.set_facecolor("#CCCCCC")
            box.set_edgecolor("gray")

    # Add WHO guideline
    WHO_GUIDELINE = 100  # WHO 8-hour guideline (µg/m³)
    ax.axhline(
        y=WHO_GUIDELINE,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"WHO Guideline ({WHO_GUIDELINE} µg/m³)",
    )

    ax.set_xlabel("Hour of Day", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ozone Concentration (O₃) - µg/m³", fontsize=12, fontweight="bold")
    ax.set_title(
        "Diurnal Ozone Cycle - Hourly Distribution",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(hours[::2])
    ax.set_xticklabels([f"{h:02d}:00" for h in hours[::2]], rotation=45)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # Add annotation boxes
    ax.text(
        0.02,
        0.98,
        f"Peak Hours (10AM-4PM): Higher O₃ from photochemistry\n"
        f"Nighttime (10PM-5AM): O₃ depletion via NO titration",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure1_Diurnal_Boxplot.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"    [/] Figure1_Diurnal_Boxplot.png")


# Line plot
def create_hourly_avg_line(df):
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate hourly statistics
    hourly_stats = []
    for hour in range(24):
        values = df[df["HOUR"] == hour]["o3"].dropna().values
        if len(values) > 0:
            hourly_stats.append(
                {
                    "hour": hour,
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "sem": np.std(values) / np.sqrt(len(values)),
                    "count": len(values),
                }
            )

    hourly_df = pd.DataFrame(hourly_stats)

    # Plot mean line
    ax.plot(
        hourly_df["hour"],
        hourly_df["mean"],
        "b-o",
        linewidth=2.5,
        markersize=8,
        label="Mean O₃ Concentration",
        color="#1f77b4",
    )

    # Add confidence band (±1 standard deviation)
    ax.fill_between(
        hourly_df["hour"],
        hourly_df["mean"] - hourly_df["std"],
        hourly_df["mean"] + hourly_df["std"],
        alpha=0.25,
        color="blue",
        label="±1 Standard Deviation",
    )

    # Add standard error band (lighter)
    ax.fill_between(
        hourly_df["hour"],
        hourly_df["mean"] - hourly_df["sem"],
        hourly_df["mean"] + hourly_df["sem"],
        alpha=0.4,
        color="lightblue",
        label="±1 Standard Error",
    )

    # Find and mark peak hour
    peak_idx = hourly_df["mean"].idxmax()
    peak_hour = hourly_df.loc[peak_idx, "hour"]
    peak_value = hourly_df.loc[peak_idx, "mean"]

    ax.plot(
        peak_hour, peak_value, "r*", markersize=20, label=f"Peak: {peak_hour:02d}:00"
    )

    # Add vertical line at peak
    ax.axvline(x=peak_hour, color="red", linestyle=":", alpha=0.5, linewidth=1.5)

    # Add WHO guideline
    WHO_GUIDELINE = 100
    ax.axhline(
        y=WHO_GUIDELINE,
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"WHO Guideline ({WHO_GUIDELINE} µg/m³)",
    )

    ax.set_xlabel("Hour of Day", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ozone Concentration (O₃) - µg/m³", fontsize=12, fontweight="bold")
    ax.set_title(
        "Average Diurnal Ozone Cycle with Variability Bands",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.text(
        0.98,
        0.98,
        f"Peak Time: {peak_hour:02d}:00 ({peak_value:.1f} µg/m³)\n"
        f"Morning ramp: 6AM-10AM\n"
        f"Evening decay: 4PM-8PM",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure2_Hourly_Avg_Line.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"    [/] Figure2_Hourly_Avg_Line.png")


# Heatmap
def create_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Select numerical columns for correlation
    corr_cols = ["o3", "no2", "pm25", "temperature", "humidity", "wind_speed"]
    available_cols = [c for c in corr_cols if c in df.columns]

    # Compute correlation matrix using NumPy
    corr_matrix = np.zeros((len(available_cols), len(available_cols)))
    for i, col1 in enumerate(available_cols):
        for j, col2 in enumerate(available_cols):
            valid = df[col1].notna() & df[col2].notna()
            if valid.sum() > 1:
                corr_matrix[i, j] = np.corrcoef(
                    df[col1][valid].values, df[col2][valid].values
                )[0, 1]
            else:
                corr_matrix[i, j] = 0

    # Create heatmap
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation Coefficient", fontsize=11)

    # Set labels
    ax.set_xticks(range(len(available_cols)))
    ax.set_yticks(range(len(available_cols)))
    ax.set_xticklabels(available_cols, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(available_cols, fontsize=10)

    # Add correlation values inside cells
    for i in range(len(available_cols)):
        for j in range(len(available_cols)):
            text_color = "white" if abs(corr_matrix[i, j]) > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title(
        "Correlation Matrix - Pollutants & Meteorology",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure3_Correlation_Heatmap.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"    [/] Figure3_Correlation_Heatmap.png")


# Scatter plot
def create_ozone_vs_no2_scatter(df):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by hour of day
    scatter = ax.scatter(
        df["no2"],
        df["o3"],
        c=df["HOUR"],
        cmap="viridis",
        alpha=0.6,
        s=30,
        edgecolors="black",
        linewidth=0.5,
    )

    # Add colorbar for hours
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Hour of Day", fontsize=11)

    # Add trend line
    valid_mask = df["no2"].notna() & df["o3"].notna()
    if valid_mask.sum() > 1:
        z = np.polyfit(df["no2"][valid_mask].values, df["o3"][valid_mask].values, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df["no2"].min(), df["no2"].max(), 100)
        ax.plot(
            x_trend, p(x_trend), "r--", linewidth=2, label=f"Trend (slope={z[0]:.2f})"
        )

    ax.set_xlabel("Nitrogen Dioxide (NO₂) - µg/m³", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ozone (O₃) - µg/m³", fontsize=12, fontweight="bold")
    ax.set_title(
        "Ozone vs NO₂ - Inverse Relationship from NO₂ Titration",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add annotation explaining the chemistry
    ax.text(
        0.02,
        0.98,
        "Chemical Interpretation:\n"
        "NO₂ + sunlight → NO + O (Ozone formation)\n"
        "NO + O₃ → NO₂ + O₂ (Ozone destruction)\n"
        "Higher NO₂ at night = Lower O₃",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure4_O3_vs_NO2_Scatter.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"    [/] Figure4_O3_vs_NO2_Scatter.png")


# Animation 1
def create_animated_diurnal_evolution(df):
    # Get unique dates
    dates = sorted(df["DATE"].unique())
    n_days = len(dates)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Line plot for current day's diurnal pattern
    (line_curr,) = ax1.plot(
        [], [], "b-o", linewidth=2, markersize=6, label="Current Day"
    )
    (line_avg,) = ax1.plot(
        [], [], "gray", linewidth=1.5, alpha=0.5, label="Multi-Day Average"
    )

    # Fill between for reference
    ax1.set_xlim(0, 23)
    ax1.set_ylim(df["o3"].min() * 0.9, df["o3"].max() * 1.1)
    ax1.set_xlabel("Hour of Day", fontsize=12)
    ax1.set_ylabel("Ozone Concentration (µg/m³)", fontsize=12)
    ax1.set_title(
        "Animation 1 - Daily Ozone Cycle Evolution",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Bar chart for daily peak values
    bars = ax2.bar(
        range(n_days), [0] * n_days, color="steelblue", alpha=0.7, label="Daily Peak O₃"
    )
    (trend_line,) = ax2.plot([], [], "r-", linewidth=2, label="Trend")

    ax2.set_xlabel("Day Index", fontsize=12)
    ax2.set_ylabel("Peak Ozone Concentration (µg/m³)", fontsize=12)
    ax2.set_title("Daily Peak Ozone Values", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, df["o3"].max() * 1.2)

    # Add WHO guideline to both plots
    WHO_GUIDELINE = 100
    ax1.axhline(
        y=WHO_GUIDELINE, color="red", linestyle="--", alpha=0.5, label="WHO Guideline"
    )
    ax2.axhline(y=WHO_GUIDELINE, color="red", linestyle="--", alpha=0.5)

    # Precompute daily averages and peaks
    daily_data = []
    for date in dates:
        day_df = df[df["DATE"] == date]
        hourly_avg = []
        for hour in range(24):
            hour_values = day_df[day_df["HOUR"] == hour]["o3"].values
            hourly_avg.append(np.mean(hour_values) if len(hour_values) > 0 else np.nan)
        daily_data.append(
            {"date": date, "hourly": hourly_avg, "peak": np.nanmax(hourly_avg)}
        )

    # Compute overall average diurnal pattern
    overall_hourly = []
    for hour in range(24):
        all_values = df[df["HOUR"] == hour]["o3"].values
        overall_hourly.append(np.mean(all_values) if len(all_values) > 0 else 0)

    # Initialize average line
    line_avg.set_data(range(24), overall_hourly)

    def init():
        line_curr.set_data([], [])
        for bar in bars:
            bar.set_height(0)
        trend_line.set_data([], [])
        return line_curr, *bars, trend_line

    def update(frame):
        # Update current day's diurnal pattern
        curr_data = daily_data[frame]
        hours_display = list(range(24))
        values_display = curr_data["hourly"]

        # Filter out NaN values
        valid_pairs = [
            (h, v) for h, v in zip(hours_display, values_display) if not np.isnan(v)
        ]
        if valid_pairs:
            hours_valid, values_valid = zip(*valid_pairs)
            line_curr.set_data(hours_valid, values_valid)

        # Update bar chart
        for i, bar in enumerate(bars):
            if i <= frame:
                bar.set_height(daily_data[i]["peak"])

        # Update trend line
        peaks_so_far = [daily_data[i]["peak"] for i in range(frame + 1)]
        trend_line.set_data(range(frame + 1), peaks_so_far)

        # Update title with current date
        ax1.set_title(
            f"Daily Ozone Cycle - {curr_data['date']}",
            fontsize=14,
            fontweight="bold",
        )

        return line_curr, *bars, trend_line

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_days,
        init_func=init,
        interval=500,
        repeat=False,
        blit=False,
    )

    anim.save(
        output_dir / "Animation1_Diurnal_Evolution.gif", writer=PillowWriter(fps=3)
    )
    plt.close()
    print(f"    [/] Animation1_Diurnal_Evolution.gif")


# Animation 2
def create_animated_heatmap_series(df):
    # Prepare pivot table: rows = hour, columns = date, values = O3
    pivot_data = df.pivot_table(
        index="HOUR", columns="DATE", values="o3", aggfunc="mean"
    )

    # Sort columns chronologically
    pivot_data = pivot_data.reindex(sorted(pivot_data.columns), axis=1)
    dates = pivot_data.columns.tolist()
    hours = range(24)

    # Get min and max for consistent color scale
    vmin = pivot_data.min().min()
    vmax = pivot_data.max().max()

    fig, ax = plt.subplots(figsize=(14, 8))

    def animate(frame):
        ax.clear()

        # Show data up to current frame
        current_dates = dates[: frame + 1]
        current_data = pivot_data[current_dates]

        # Create heatmap
        im = ax.imshow(
            current_data.values,
            cmap="YlOrRd",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        # Labels
        ax.set_xticks(range(len(current_dates)))
        ax.set_xticklabels(
            [str(d)[5:10] for d in current_dates], rotation=45, ha="right", fontsize=8
        )
        ax.set_yticks(range(24))
        ax.set_yticklabels([f"{h:02d}:00" for h in hours], fontsize=8)
        ax.set_xlabel("Date (MM-DD)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Hour of Day", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Ozone Concentration Heatmap - Day {frame+1}/{len(dates)}",
            fontsize=14,
            fontweight="bold",
        )

        # Add colorbar on last frame
        if frame == len(dates) - 1:
            plt.colorbar(im, ax=ax, label="O₃ Concentration (µg/m³)", shrink=0.8)

        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(dates),
        interval=400,
        repeat=False,
        blit=False,
    )

    anim.save(output_dir / "Animation2_Heatmap_Series.gif", writer=PillowWriter(fps=2))
    plt.close()
    print(f"    [/] Animation2_Heatmap_Series.gif")


# Applied OOP structure


class DiurnalOzonePipeline:
    def __init__(self, file_path):
        # Encapsulated attributes (private by convention using _)
        self._file_path = file_path
        self._df = None
        self._cleaned_df = None
        self._stats = None
        self._output_dir = output_dir

    # Data Ingestion

    def load_data(self):
        self._df = load_csv_file(self._file_path)

        if self._df is None:
            raise FileNotFoundError("Could not load data file. Check file path.")

        return self  # Enable method chaining

    # Filtering

    def filter_data(self):
        self._df = filter_by_unique_logic(self._df)
        return self

    # Data Cleaning

    def clean_data(self):
        self._df = clean_data(self._df)
        return self

    # Statistical Analysis

    def compute_statistics(self):
        self._stats = compute_statistics_numpy(self._df)
        return self

    # Save Cleaned Data

    def save_cleaned_data(self):
        self._df.to_csv(cleaned_file, index=False)
        return self

    # Visualization

    def visualize(self):
        create_diurnal_boxplot(self._df)
        create_hourly_avg_line(self._df)
        create_correlation_heatmap(self._df)
        create_ozone_vs_no2_scatter(self._df)
        create_animated_diurnal_evolution(self._df)
        create_animated_heatmap_series(self._df)
        return self

    # Generate Table 1
    def generate_table(self):
        create_sample_telemetry_table(self._df)
        return self

    # Getter Methods (Encapsulation)

    def get_data(self):
        return self._df

    def get_statistics(self):
        return self._stats

    # Run Full Pipeline (Method Chaining)

    def run(self):
        return (
            self.load_data()
            .filter_data()
            .clean_data()
            .compute_statistics()
            .save_cleaned_data()
            .generate_table()
            .visualize()
        )


# Main execution function

if __name__ == "__main__":
    print("=" * 110)
    print("  DIURNAL OZONE CYCLES ANALYSIS")
    print("  Air Quality Data Analytics Pipeline")
    print("=" * 110)

    # Load data with cleaner output
    print("Loading data...")
    df = pd.read_csv(airquality_file)
    print(f"    [/] globalAirQuality.csv ({len(df):,} rows)\n")

    # Apply filters
    print("Applying filters...")
    print(f"    [/] Country: {COUNTRY} | City: {CITY}")
    print(f"    [/] Date range: {START_DATE} to {END_DATE}")
    print(f"    [/] Time range: Full 24-hour cycle\n")

    filtered_df = df[(df["country"] == COUNTRY) & (df["city"] == CITY)].copy()
    filtered_df["DATETIME"] = pd.to_datetime(filtered_df["timestamp"])
    filtered_df = filtered_df[
        (filtered_df["DATETIME"] >= START_DATE) & (filtered_df["DATETIME"] <= END_DATE)
    ].copy()
    filtered_df["HOUR"] = filtered_df["DATETIME"].dt.hour
    filtered_df = filtered_df[filtered_df["o3"] > MIN_O3].copy()
    filtered_df["DATE"] = filtered_df["DATETIME"].dt.date

    print(f"    [/] Filtered dataset: {len(filtered_df)} rows")
    print(
        f"    [/] Date range in data: {filtered_df['DATE'].min()} to {filtered_df['DATE'].max()}"
    )
    print(f"    [/] Unique days: {len(filtered_df['DATE'].unique())}\n")

    # Clean data
    print("Cleaning data...")
    filtered_df = filtered_df.drop_duplicates()

    # Fill missing values
    key_cols = ["o3", "no2", "pm25", "temperature", "humidity", "wind_speed"]
    for col in key_cols:
        if col in filtered_df.columns and filtered_df[col].isnull().any():
            filtered_df[col] = filtered_df[col].fillna(method="ffill")
            if filtered_df[col].isnull().any():
                filtered_df[col] = filtered_df[col].fillna(filtered_df[col].median())

    print(f"    [/] Cleaned dataset: {len(filtered_df)} rows\n")

    # Classify by time of day
    print("Classifying time periods...")
    filtered_df["TIME_PERIOD"] = np.where(
        (filtered_df["HOUR"] >= 10) & (filtered_df["HOUR"] <= 16),
        "Peak (10AM-4PM)",
        np.where(
            (filtered_df["HOUR"] >= 22) | (filtered_df["HOUR"] <= 5),
            "Night (10PM-5AM)",
            "Transition",
        ),
    )

    peak_count = len(filtered_df[filtered_df["TIME_PERIOD"] == "Peak (10AM-4PM)"])
    night_count = len(filtered_df[filtered_df["TIME_PERIOD"] == "Night (10PM-5AM)"])
    trans_count = len(filtered_df[filtered_df["TIME_PERIOD"] == "Transition"])
    print(f"    [/] Peak hours: {peak_count} ({peak_count/len(filtered_df)*100:.1f}%)")
    print(
        f"    [/] Night hours: {night_count} ({night_count/len(filtered_df)*100:.1f}%)"
    )
    print(
        f"    [/] Transition: {trans_count} ({trans_count/len(filtered_df)*100:.1f}%)"
    )

    # Compute statistics
    print("-" * 110)
    print("  STATISTICAL RESULTS (NumPy-based)")
    print("-" * 110)

    # Overall O3 statistics
    o3_vals = filtered_df["o3"].values
    print(f"Ozone (O₃) Descriptive Statistics:")
    print(f"    Mean:     {np.mean(o3_vals):8.2f} µg/m³")
    print(f"    Median:   {np.median(o3_vals):8.2f} µg/m³")
    print(f"    Std Dev:  {np.std(o3_vals):8.2f} µg/m³")
    print(f"    Variance: {np.var(o3_vals):8.2f}")
    print(f"    Min:      {np.min(o3_vals):8.2f} µg/m³")
    print(f"    Max:      {np.max(o3_vals):8.2f} µg/m³")
    print(f"    Q25:      {np.percentile(o3_vals, 25):8.2f} µg/m³")
    print(f"    Q75:      {np.percentile(o3_vals, 75):8.2f} µg/m³")

    # Peak vs Night comparison
    peak_vals = filtered_df[filtered_df["TIME_PERIOD"] == "Peak (10AM-4PM)"][
        "o3"
    ].values
    night_vals = filtered_df[filtered_df["TIME_PERIOD"] == "Night (10PM-5AM)"][
        "o3"
    ].values
    print(f"\nPeak vs Night Comparison:")
    print(f"    Peak hours mean:  {np.mean(peak_vals):8.2f} µg/m³")
    print(f"    Night hours mean: {np.mean(night_vals):8.2f} µg/m³")
    print(f"    Ratio (Peak/Night): {np.mean(peak_vals) / np.mean(night_vals):8.2f}x")

    # SKEWNESS CALCULATION
    print("-" * 110)
    print("  DISTRIBUTION ANALYSIS - SKEWNESS")
    print("-" * 110)

    # Calculate skewness for different time periods
    all_skew = calculate_skewness(filtered_df["o3"].values)
    peak_skew = calculate_skewness(peak_vals)
    night_skew = calculate_skewness(night_vals)

    print(f"Ozone Distribution Skewness:")
    print(f"    Full dataset:   {all_skew:.4f}")
    print(f"    Peak hours:     {peak_skew:.4f}")
    print(f"    Night hours:    {night_skew:.4f}")

    # Interpret skewness
    print(f"\nInterpretation:")
    if all_skew > 0.5:
        print(
            f"    The overall O₃ distribution is positively skewed ({all_skew:.4f}) -"
        )
        print("    indicating a tail toward higher concentrations (spikes/events).")
    elif all_skew < -0.5:
        print(f"    The overall O₃ distribution is negatively skewed ({all_skew:.4f})")
    else:
        print(
            f"    The overall O₃ distribution is approximately symmetric ({all_skew:.4f})"
        )

    # OUTLIER DETECTION (IQR method)
    print("-" * 110)
    print("  OUTLIER DETECTION")
    print("-" * 110)

    q1 = np.percentile(o3_vals, 25)
    q3 = np.percentile(o3_vals, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = o3_vals[(o3_vals < lower_bound) | (o3_vals > upper_bound)]

    print(f"Ozone Outlier Analysis (IQR method):")
    print(f"    Q1: {q1:.2f} | Q3: {q3:.2f} | IQR: {iqr:.2f}")
    print(f"    Lower bound: {lower_bound:.2f} | Upper bound: {upper_bound:.2f}")

    # Check if outliers exist before computing min/max
    if len(outliers) > 0:
        print(
            f"    Outliers detected: {len(outliers)} ({len(outliers)/len(o3_vals)*100:.1f}% of data)"
        )
        print(
            f"    Outlier range: {np.min(outliers):.2f} to {np.max(outliers):.2f} µg/m³"
        )
    else:
        print(f"    Outliers detected: 0 (0.0% of data)")
        print(f"    No outliers found - all data points are within the IQR bounds")

    # CORRELATION ANALYSIS
    print("-" * 110)
    print("  CORRELATION ANALYSIS")
    print("-" * 110)

    if "no2" in filtered_df.columns:
        valid = filtered_df["o3"].notna() & filtered_df["no2"].notna()
        o3_no2_corr = np.corrcoef(
            filtered_df["o3"][valid].values, filtered_df["no2"][valid].values
        )[0, 1]
        direction = "negative" if o3_no2_corr < 0 else "positive"
        strength = (
            "strong"
            if abs(o3_no2_corr) > 0.6
            else "moderate" if abs(o3_no2_corr) > 0.3 else "weak"
        )
        print(f"    O₃ vs NO₂:     {o3_no2_corr:.4f} ({direction}, {strength})")

    if "temperature" in filtered_df.columns:
        valid = filtered_df["o3"].notna() & filtered_df["temperature"].notna()
        o3_temp_corr = np.corrcoef(
            filtered_df["o3"][valid].values, filtered_df["temperature"][valid].values
        )[0, 1]
        direction = "positive" if o3_temp_corr > 0 else "negative"
        strength = (
            "strong"
            if abs(o3_temp_corr) > 0.6
            else "moderate" if abs(o3_temp_corr) > 0.3 else "weak"
        )
        print(f"    O₃ vs Temp:    {o3_temp_corr:.4f} ({direction}, {strength})")

    if "humidity" in filtered_df.columns:
        valid = filtered_df["o3"].notna() & filtered_df["humidity"].notna()
        o3_hum_corr = np.corrcoef(
            filtered_df["o3"][valid].values, filtered_df["humidity"][valid].values
        )[0, 1]
        direction = "negative" if o3_hum_corr < 0 else "positive"
        strength = (
            "strong"
            if abs(o3_hum_corr) > 0.6
            else "moderate" if abs(o3_hum_corr) > 0.3 else "weak"
        )
        print(f"    O₃ vs Humidity:{o3_hum_corr:.4f} ({direction}, {strength})")

    # Generate Table 1: Sample Telemetry Records
    create_sample_telemetry_table(filtered_df)

    # Generate visualizations
    print("-" * 110)
    print("  GENERATING FIGURES AND ANIMATIONS")
    print("-" * 110)

    create_diurnal_boxplot(filtered_df)
    create_hourly_avg_line(filtered_df)
    create_correlation_heatmap(filtered_df)
    create_ozone_vs_no2_scatter(filtered_df)
    create_animated_diurnal_evolution(filtered_df)
    create_animated_heatmap_series(filtered_df)

    # Save cleaned data
    filtered_df.to_csv(cleaned_file, index=False)
    print(f"\n      [/] Cleaned data saved to: {cleaned_file}")

    print("\n" + "=" * 110)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print(f"  Processed: {len(filtered_df)} records")
    print(f"  Outputs saved to: {output_dir}")
    print("=" * 110)