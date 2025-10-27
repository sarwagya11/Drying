"""Phase 2 correlation builder for Midilli-a parameters.

This script reads the master Phase 1 results dataset, filters the Midilli-a
parameters, and fits correlation models that relate those parameters to the
process conditions. All regressions are performed with ``statsmodels`` to
provide full statistical summaries.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm


R_GAS_CONSTANT = 8.31446261815324  # J/(mol*K)
ABSOLUTE_ZERO_C = 273.15
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "phase1_out" / "phase1_full_results.csv"
MODELS_DIR = ROOT_DIR / "models"


def ensure_models_dir() -> None:
    """Create the models directory if it does not exist."""

    MODELS_DIR.mkdir(exist_ok=True)


def load_phase1_results() -> pd.DataFrame:
    """Load the Phase 1 CSV and validate required Midilli-a columns."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Phase 1 results CSV not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required_columns = {
        "T_C",
        "v_ms",
        "RH_pct",
        "thickness_mm",
        "Midilli-a_a",
        "Midilli-a_k",
        "Midilli-a_n",
        "Midilli-a_b",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    return df


def fit_arrhenius_model(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit ln(k) against Arrhenius-style predictors."""

    subset = df[
        [
            "Midilli-a_k",
            "T_C",
            "v_ms",
            "RH_pct",
            "thickness_mm",
        ]
    ].dropna()

    mask_positive = (
        (subset["Midilli-a_k"] > 0)
        & (subset["v_ms"] > 0)
        & (subset["RH_pct"] > 0)
        & (subset["thickness_mm"] > 0)
    )
    subset = subset.loc[mask_positive].copy()
    if subset.empty:
        raise ValueError("No valid observations available for Arrhenius regression of k.")

    temperature_abs = subset["T_C"].to_numpy(dtype=float) + ABSOLUTE_ZERO_C
    features = pd.DataFrame(
        {
            "inv_T_abs": 1.0 / temperature_abs,
            "ln_v_ms": np.log(subset["v_ms"].to_numpy(dtype=float)),
            "ln_RH_pct": np.log(subset["RH_pct"].to_numpy(dtype=float)),
            "ln_thickness_mm": np.log(subset["thickness_mm"].to_numpy(dtype=float)),
        }
    )
    X = sm.add_constant(features, has_constant="add")
    y = np.log(subset["Midilli-a_k"].to_numpy(dtype=float))

    model = sm.OLS(y, X).fit()
    print(model.summary())

    beta_inv_T = model.params["inv_T_abs"]
    activation_energy_kj_mol = -beta_inv_T * R_GAS_CONSTANT / 1000.0
    print(f"Activation energy (Ea): {activation_energy_kj_mol:.6f} kJ/mol")

    return model


def fit_linear_model(
    df: pd.DataFrame, target: str
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit a linear model for a Midilli-a parameter using process conditions."""

    subset = df[
        [
            target,
            "T_C",
            "v_ms",
            "RH_pct",
            "thickness_mm",
        ]
    ].dropna()

    if subset.empty:
        raise ValueError(f"No valid observations available for regression of {target}.")

    X = sm.add_constant(subset[["T_C", "v_ms", "RH_pct", "thickness_mm"]].astype(float), has_constant="add")
    y = subset[target].to_numpy(dtype=float)

    model = sm.OLS(y, X).fit()
    print(model.summary())

    return model


def save_model(model: sm.regression.linear_model.RegressionResultsWrapper, filename: str) -> None:
    """Persist a fitted statsmodels OLS result."""

    with (MODELS_DIR / filename).open("wb") as handle:
        pickle.dump(model, handle)


def save_process_bounds(df: pd.DataFrame) -> None:
    """Persist min/max bounds for the process-condition features."""

    bounds: Dict[str, Dict[str, float]] = {}
    for column in ["T_C", "v_ms", "RH_pct", "thickness_mm"]:
        values = df[column].dropna().to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError(f"No data available to compute bounds for {column}.")
        bounds[column] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    with (MODELS_DIR / "process_bounds.json").open("w", encoding="utf-8") as handle:
        json.dump(bounds, handle, indent=2)


def main() -> None:
    ensure_models_dir()
    df = load_phase1_results()

    model_k = fit_arrhenius_model(df)
    save_model(model_k, "model_k.pkl")

    model_a = fit_linear_model(df, "Midilli-a_a")
    save_model(model_a, "model_a.pkl")

    model_n = fit_linear_model(df, "Midilli-a_n")
    save_model(model_n, "model_n.pkl")

    model_b = fit_linear_model(df, "Midilli-a_b")
    save_model(model_b, "model_b.pkl")

    save_process_bounds(df)


if __name__ == "__main__":
    main()
