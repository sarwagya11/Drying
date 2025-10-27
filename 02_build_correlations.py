"""Phase 2 correlation builder for Midilli-a parameters.

This script loads the master Phase 1 results CSV, filters for the
Midilli-a model parameters, and fits correlation models linking the
parameters (a, k, n, b) to the process conditions. The ``k`` parameter is
modelled using a linearised Arrhenius expression, while ``a``, ``n`` and
``b`` employ standard multiple linear regression. The fitted models and
the process-condition bounds are saved in the ``models`` directory.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "phase1_out" / "phase1_full_results.csv"
MODELS_DIR = ROOT_DIR / "models"
R_GAS = 8.314462618  # J/(mol*K)
ABSOLUTE_ZERO_C = 273.15


def ensure_models_dir() -> None:
    MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class LinearModel:
    """Simple container for a linear regression model."""

    target: str
    feature_names: List[str]
    intercept: float
    coefficients: Dict[str, float]
    notes: Dict[str, str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "target": self.target,
            "feature_names": self.feature_names,
            "intercept": self.intercept,
            "coefficients": self.coefficients,
            "notes": self.notes,
        }


def load_phase1_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Phase 1 results CSV not found: {path}")
    df = pd.read_csv(path)
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
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in input data: {missing_list}")
    return df


def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    if X.ndim != 2:
        raise ValueError("Feature matrix must be 2-D.")
    if y.ndim != 1:
        raise ValueError("Target vector must be 1-D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of rows in X must match size of y.")
    # Add intercept term
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    intercept = float(coeffs[0])
    slopes = coeffs[1:]
    return intercept, slopes


def build_arrhenius_model(df: pd.DataFrame) -> LinearModel:
    subset = df[[
        "Midilli-a_k",
        "T_C",
        "v_ms",
        "RH_pct",
        "thickness_mm",
    ]].dropna()

    # Ensure positive values for logarithmic transforms
    positive_mask = (
        (subset["Midilli-a_k"] > 0)
        & (subset["v_ms"] > 0)
        & (subset["RH_pct"] > 0)
        & (subset["thickness_mm"] > 0)
    )
    subset = subset.loc[positive_mask].copy()
    if subset.empty:
        raise ValueError("No valid rows available to fit the Arrhenius model for k.")

    T_abs = subset["T_C"].to_numpy(dtype=float) + ABSOLUTE_ZERO_C
    features = np.column_stack([
        1.0 / T_abs,
        np.log(subset["v_ms"].to_numpy(dtype=float)),
        np.log(subset["RH_pct"].to_numpy(dtype=float)),
        np.log(subset["thickness_mm"].to_numpy(dtype=float)),
    ])
    y = np.log(subset["Midilli-a_k"].to_numpy(dtype=float))
    feature_names = [
        "inv_T_abs",
        "ln_v_ms",
        "ln_RH_pct",
        "ln_thickness_mm",
    ]

    intercept, slopes = fit_linear_regression(features, y)
    coefficients = {
        name: float(value) for name, value in zip(feature_names, slopes)
    }
    model = LinearModel(
        target="Midilli-a_k",
        feature_names=feature_names,
        intercept=intercept,
        coefficients=coefficients,
        notes={
            "transforms": (
                "ln(k) = intercept + inv_T_abs + ln_v_ms + ln_RH_pct + ln_thickness_mm"
            ),
            "T_abs_definition": "T_abs = T_C + 273.15",
        },
    )

    # Compute activation energy from the inverse-temperature coefficient
    beta_inv_T = coefficients["inv_T_abs"]
    activation_energy_kjmol = -beta_inv_T * R_GAS / 1000.0
    print(f"Activation energy (Ea): {activation_energy_kjmol:.4f} kJ/mol")

    return model


def build_linear_model(df: pd.DataFrame, target_column: str) -> LinearModel:
    subset = df[[
        target_column,
        "T_C",
        "v_ms",
        "RH_pct",
        "thickness_mm",
    ]].dropna()

    if subset.empty:
        raise ValueError(f"No valid rows available to fit the model for {target_column}.")

    features = subset[["T_C", "v_ms", "RH_pct", "thickness_mm"]].to_numpy(dtype=float)
    y = subset[target_column].to_numpy(dtype=float)
    feature_names = ["T_C", "v_ms", "RH_pct", "thickness_mm"]

    intercept, slopes = fit_linear_regression(features, y)
    coefficients = {
        name: float(value) for name, value in zip(feature_names, slopes)
    }
    model = LinearModel(
        target=target_column,
        feature_names=feature_names,
        intercept=intercept,
        coefficients=coefficients,
        notes={"transforms": "linear"},
    )
    return model


def save_model(model: LinearModel, filename: str) -> None:
    path = MODELS_DIR / filename
    with path.open("wb") as fh:
        pickle.dump(model.as_dict(), fh)


def save_process_bounds(df: pd.DataFrame) -> None:
    bounds = {}
    for column in ["T_C", "v_ms", "RH_pct", "thickness_mm"]:
        series = df[column].dropna().to_numpy(dtype=float)
        if series.size == 0:
            raise ValueError(f"No data available to compute bounds for {column}.")
        bounds[column] = {
            "min": float(np.min(series)),
            "max": float(np.max(series)),
        }

    path = MODELS_DIR / "process_bounds.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(bounds, fh, indent=2)


def main() -> None:
    ensure_models_dir()
    df = load_phase1_results(DATA_PATH)

    model_k = build_arrhenius_model(df)
    save_model(model_k, "model_k.pkl")

    for target, filename in [
        ("Midilli-a_a", "model_a.pkl"),
        ("Midilli-a_n", "model_n.pkl"),
        ("Midilli-a_b", "model_b.pkl"),
    ]:
        model = build_linear_model(df, target)
        save_model(model, filename)

    save_process_bounds(df)


if __name__ == "__main__":
    main()
