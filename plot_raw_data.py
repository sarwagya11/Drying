"""Generate Moisture Ratio vs. Normalized Time plots for raw CSV files.

This script scans the current working directory for CSV files that contain
columns named ``time_min`` and ``X_db``. For each valid CSV it generates a
Moisture Ratio (MR) curve, saving the output to ``raw_data_plots``.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def find_csv_files(patterns: Iterable[str]) -> list[Path]:
    """Return all unique CSV files matching the provided glob patterns."""
    files: list[Path] = []
    seen = set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            candidate = Path(path)
            if candidate.suffix.lower() != ".csv":
                continue
            if candidate.name in seen:
                continue
            seen.add(candidate.name)
            files.append(candidate)
    return sorted(files)


def load_raw_data(csv_path: Path) -> pd.DataFrame | None:
    """Load a CSV file into a DataFrame, handling empty or invalid files."""
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"Skipping {csv_path.name}: file is empty.")
        return None
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Skipping {csv_path.name}: failed to read CSV ({exc}).")
        return None

    if df.empty:
        print(f"Skipping {csv_path.name}: no rows found.")
        return None

    missing = {"time_min", "X_db"} - set(df.columns)
    if missing:
        print(
            f"Skipping {csv_path.name}: missing required columns {sorted(missing)}."
        )
        return None

    return df


def compute_normalized_series(series: pd.Series) -> pd.Series:
    """Return the series normalized by subtracting the first value."""
    values = series.to_numpy(dtype=float, copy=False)
    first_value = values[0]
    return pd.Series(values - first_value, index=series.index)


def compute_moisture_ratio(series: pd.Series) -> pd.Series | None:
    """Compute MR = X_db / X_db[0], guarding against division by zero."""
    values = series.to_numpy(dtype=float, copy=False)
    first_value = values[0]
    if np.isclose(first_value, 0.0):
        print("Cannot compute MR: initial moisture content is zero.")
        return None
    return pd.Series(values / first_value, index=series.index)


def plot_mr_curve(time_series: pd.Series, mr_series: pd.Series, title: str, output_dir: Path) -> None:
    """Create and save the MR curve plot to the output directory."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_series, mr_series, marker="o", linestyle="-")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Moisture Ratio (MR)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    output_path = output_dir / f"{title}_raw_plot.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def main() -> None:
    csv_files = find_csv_files(["*.csv"])

    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    output_dir = Path("raw_data_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_generated = 0

    for csv_path in csv_files:
        df = load_raw_data(csv_path)
        if df is None:
            continue

        try:
            time_series = pd.to_numeric(df["time_min"], errors="raise")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Skipping {csv_path.name}: invalid time data ({exc}).")
            continue

        norm_time = compute_normalized_series(time_series)

        try:
            moisture_series = pd.to_numeric(df["X_db"], errors="raise")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Skipping {csv_path.name}: invalid moisture data ({exc}).")
            continue

        mr_series = compute_moisture_ratio(moisture_series)
        if mr_series is None:
            print(f"Skipping {csv_path.name}: unable to compute moisture ratio.")
            continue

        title = csv_path.stem
        plot_mr_curve(norm_time, mr_series, title, output_dir)
        plots_generated += 1

    print(
        f"Generated {plots_generated} plot{'s' if plots_generated != 1 else ''} "
        f"in '{output_dir}'."
    )


if __name__ == "__main__":
    main()
