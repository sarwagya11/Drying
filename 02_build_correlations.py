"""Phase 2 correlation builder for Midilli-a parameters using statsmodels."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "phase1_out" / "phase1_full_results.csv"
MODELS_DIR = ROOT_DIR / "models"
ABSOLUTE_ZERO_C = 273.15
R_GAS = 8.314462618  # J/(mol*K)
TEMP_SWEEP_RUNS = ["T_40_v1p1", "T_45_v1p1", "T_50_v1p1_tempsweep"]
MIDILLI_COLUMNS = [
    "run_id",
    "T_C",
    "v_ms",
    "RH_pct",
    "thickness_mm",
    "Midilli-a_a",
    "Midilli-a_k",
    "Midilli-a_n",
    "Midilli-a_b",
]
FEATURE_COLUMNS = ["v_ms", "RH_pct", "thickness_mm"]


def ensure_models_dir() -> None:
    MODELS_DIR.mkdir(exist_ok=True)


def load_midilli_dataframe() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Phase 1 results CSV not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, usecols=MIDILLI_COLUMNS)
    numeric_cols = [col for col in MIDILLI_COLUMNS if col != "run_id"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def fit_arrhenius_model(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    subset = df[df["run_id"].isin(TEMP_SWEEP_RUNS)].copy()
    subset = subset.dropna(subset=["T_C", "Midilli-a_k"])

    if subset.shape[0] != 3:
        raise ValueError(
            "Temperature sweep dataset must contain exactly 3 Midilli-a runs. "
            f"Found {subset.shape[0]} rows."
        )

    if (subset["Midilli-a_k"] <= 0).any():
        raise ValueError("All Midilli-a_k values must be positive for the Arrhenius fit.")

    T_abs = subset["T_C"].to_numpy(dtype=float) + ABSOLUTE_ZERO_C
    ln_k = np.log(subset["Midilli-a_k"].to_numpy(dtype=float))

    X = pd.DataFrame({"inv_T_abs": 1.0 / T_abs})
    X = sm.add_constant(X)

    model = sm.OLS(ln_k, X).fit()

    print("--- Arrhenius Model (T_C Only, 3 data points) ---")
    print(model.summary())

    beta_inv_T = model.params["inv_T_abs"]
    activation_energy_kjmol = -beta_inv_T * R_GAS / 1000.0
    print(f"Activation Energy (Ea): {activation_energy_kjmol:.4f} kJ/mol")

    return model


def fit_linear_model(
    df: pd.DataFrame, target_column: str
) -> sm.regression.linear_model.RegressionResultsWrapper:
    subset = df.dropna(subset=FEATURE_COLUMNS + [target_column]).copy()

    if subset.empty:
        raise ValueError(f"No valid rows available to fit the model for {target_column}.")

    X = sm.add_constant(subset[FEATURE_COLUMNS].astype(float))
    y = subset[target_column].astype(float)

    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model


def save_model(model: sm.regression.linear_model.RegressionResultsWrapper, filename: str) -> None:
    path = MODELS_DIR / filename
    with path.open("wb") as f:
        pickle.dump(model, f)


def save_process_bounds(df: pd.DataFrame) -> None:
    bounds = {}
    for column in ["T_C"] + FEATURE_COLUMNS:
        series = df[column].dropna().astype(float)
        bounds[column] = {"min": float(series.min()), "max": float(series.max())}

    path = MODELS_DIR / "process_bounds.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(bounds, f, indent=2)


def main() -> None:
    ensure_models_dir()
    df = load_midilli_dataframe()

    model_k = fit_arrhenius_model(df)
    save_model(model_k, "model_k.pkl")

    print("--- Model for Midilli-a_a ---")
    model_a = fit_linear_model(df, "Midilli-a_a")
    save_model(model_a, "model_a.pkl")

    print("--- Model for Midilli-a_n ---")
    model_n = fit_linear_model(df, "Midilli-a_n")
    save_model(model_n, "model_n.pkl")

    print("--- Model for Midilli-a_b ---")
    model_b = fit_linear_model(df, "Midilli-a_b")
    save_model(model_b, "model_b.pkl")

    save_process_bounds(df)


if __name__ == "__main__":
    main()
