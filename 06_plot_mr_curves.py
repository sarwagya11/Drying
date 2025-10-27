"""Plot MR vs time for all runs in a raw drying CSV file.

This utility loads a CSV file containing drying experiment data, computes the
moisture ratio (MR) for each run, and generates a single plot containing the
MR–time curves for every run present in the file.

Example usage::

    python 06_plot_mr_curves.py T50_v1p1_velsweep.csv --output mr_plot.png

If no output path is provided, the plot will be displayed in an interactive
window (when supported by the execution environment).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TIME_COLUMN = "time_min"
DEFAULT_MOISTURE_COLUMN = "X_db"
DEFAULT_RUN_COLUMN = "run_id"


def compute_moisture_ratio(x_db_series: pd.Series) -> np.ndarray:
    """Compute the moisture ratio (MR) from dry basis moisture content data."""

    x_db = x_db_series.to_numpy(dtype=float)
    if x_db.size == 0:
        return x_db
    x0 = x_db[0]
    if x0 == 0:
        raise ValueError("Initial dry basis moisture content cannot be zero.")
    return x_db / x0


def normalise_time(time_series: pd.Series) -> np.ndarray:
    """Normalise time values so each run starts at zero."""

    times = time_series.to_numpy(dtype=float)
    if times.size == 0:
        return times
    return times - times[0]


def plot_mr_curves(
    df: pd.DataFrame,
    *,
    time_column: str = DEFAULT_TIME_COLUMN,
    moisture_column: str = DEFAULT_MOISTURE_COLUMN,
    run_column: str = DEFAULT_RUN_COLUMN,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the MR–time curves for all runs contained in ``df``."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if run_column not in df.columns:
        raise KeyError(f"Column '{run_column}' not found in the input data.")
    if time_column not in df.columns:
        raise KeyError(f"Column '{time_column}' not found in the input data.")
    if moisture_column not in df.columns:
        raise KeyError(f"Column '{moisture_column}' not found in the input data.")

    grouped = df.groupby(run_column)
    if len(grouped) == 0:
        raise ValueError("No runs found in the input data.")

    for run_id, run_df in grouped:
        run_df = run_df.sort_values(time_column)
        t_norm = normalise_time(run_df[time_column])
        mr = compute_moisture_ratio(run_df[moisture_column])
        ax.plot(t_norm, mr, marker="o", linestyle="-", label=str(run_id))

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Moisture Ratio (MR)")
    ax.set_title("MR vs Time for All Runs")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(title="Run ID")

    fig.tight_layout()
    return ax


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the raw drying CSV file.",
    )
    parser.add_argument(
        "--time-column",
        default=DEFAULT_TIME_COLUMN,
        help=f"Name of the time column (default: {DEFAULT_TIME_COLUMN}).",
    )
    parser.add_argument(
        "--moisture-column",
        default=DEFAULT_MOISTURE_COLUMN,
        help=f"Name of the dry-basis moisture column (default: {DEFAULT_MOISTURE_COLUMN}).",
    )
    parser.add_argument(
        "--run-column",
        default=DEFAULT_RUN_COLUMN,
        help=f"Name of the run identifier column (default: {DEFAULT_RUN_COLUMN}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the generated plot as an image (e.g., PNG).",
    )
    return parser.parse_args(args=args)


def main() -> None:
    args = parse_args()

    if not args.input_csv.is_file():
        raise FileNotFoundError(f"CSV file not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    ax = plot_mr_curves(
        df,
        time_column=args.time_column,
        moisture_column=args.moisture_column,
        run_column=args.run_column,
    )

    if args.output:
        ax.figure.savefig(args.output, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
