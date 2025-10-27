"""Phase 3 prediction engine for Midilli-a drying model.

This script loads the hybrid correlation models generated in Phase 2 and
provides helper utilities to predict Midilli-a parameters along with the time
required to reach a target moisture ratio.  An example prediction and drying
curve plot are produced when the script is executed directly.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve


ABSOLUTE_ZERO_C = 273.15
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
PLOT_PATH = ROOT_DIR / "phase1_out" / "predicted_curve.png"


# ----------------------------------------------------------------------------
# Model Loading
# ----------------------------------------------------------------------------

def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required model file not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


MODEL_K = _load_pickle(MODELS_DIR / "model_k.pkl")
MODEL_A = _load_pickle(MODELS_DIR / "model_a.pkl")
MODEL_N = _load_pickle(MODELS_DIR / "model_n.pkl")
MODEL_B = _load_pickle(MODELS_DIR / "model_b.pkl")

PROCESS_BOUNDS_PATH = MODELS_DIR / "process_bounds.json"
if not PROCESS_BOUNDS_PATH.exists():
    raise FileNotFoundError(
        "Process bounds file not found. Expected at " f"{PROCESS_BOUNDS_PATH}"
    )
with PROCESS_BOUNDS_PATH.open("r", encoding="utf-8") as f:
    PROCESS_BOUNDS = json.load(f)


# ----------------------------------------------------------------------------
# Prediction Helpers
# ----------------------------------------------------------------------------

def _check_bounds(values: Dict[str, float]) -> None:
    """Print a warning if any provided values are outside training bounds."""

    out_of_bounds = []
    for key, value in values.items():
        bounds = PROCESS_BOUNDS.get(key)
        if not bounds:
            continue
        lower = bounds.get("min", -math.inf)
        upper = bounds.get("max", math.inf)
        if value < lower or value > upper:
            out_of_bounds.append(key)

    if out_of_bounds:
        labels = ", ".join(out_of_bounds)
        print(
            "WARNING: Input values for "
            f"{labels} are outside the training bounds; prediction is an extrapolation."
        )


def _linear_model_predict(model, features: Dict[str, float]) -> float:
    """Predict a response for a statsmodels OLS model using named features."""

    prediction = 0.0
    params = model.params
    # Statsmodels always includes the intercept as "const" when sm.add_constant is used.
    intercept = params.get("const")
    if intercept is not None:
        prediction += float(intercept)

    for name, value in features.items():
        if name not in params:
            continue
        prediction += float(params[name]) * value

    return prediction


def predict_parameters(
    T_C: float,
    v_ms: float,
    RH_pct: float,
    thickness_mm: float,
) -> Dict[str, float]:
    """Predict Midilli-a parameters using the hybrid correlation models."""

    _check_bounds({
        "T_C": T_C,
        "v_ms": v_ms,
        "RH_pct": RH_pct,
        "thickness_mm": thickness_mm,
    })

    T_abs = T_C + ABSOLUTE_ZERO_C
    if T_abs <= 0:
        raise ValueError("Absolute temperature must be positive.")

    inv_T_abs = 1.0 / T_abs
    ln_k = _linear_model_predict(MODEL_K, {"inv_T_abs": inv_T_abs})
    k_value = math.exp(ln_k)

    shared_features = {
        "v_ms": v_ms,
        "RH_pct": RH_pct,
        "thickness_mm": thickness_mm,
    }
    a_value = _linear_model_predict(MODEL_A, shared_features)
    n_value = _linear_model_predict(MODEL_N, shared_features)
    b_value = _linear_model_predict(MODEL_B, shared_features)

    return {
        "a": a_value,
        "k": k_value,
        "n": n_value,
        "b": b_value,
    }


def predict_time_to_mr(mr_target: float, params: Dict[str, float]) -> float:
    """Solve for the time needed to reach a target moisture ratio."""

    a = params["a"]
    k = params["k"]
    n = params["n"]
    b = params["b"]

    def midilli_residual(t: float) -> float:
        return a * math.exp(-k * t**n) + b * t - mr_target

    # Use a heuristic initial guess based on the intercept and slope terms.
    initial_guess = max(mr_target / max(b, 1e-6), 1.0)

    solution = fsolve(lambda t: midilli_residual(float(t)), x0=[initial_guess])
    time_value = float(solution[0])
    return max(time_value, 0.0)


# ----------------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------------

def _run_example() -> None:
    example_conditions = {
        "T_C": 45.0,
        "v_ms": 1.0,
        "RH_pct": 40.0,
        "thickness_mm": 6.0,
    }
    params = predict_parameters(**example_conditions)
    print("Predicted Midilli-a parameters:")
    for name in ["a", "k", "n", "b"]:
        print(f"  {name} = {params[name]:.6f}")

    target_mr = 0.1
    time_to_target = predict_time_to_mr(target_mr, params)
    print(f"\nPredicted time to reach MR={target_mr}: {time_to_target:.2f} min")

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t_values = np.linspace(0.0, time_to_target * 1.2 if time_to_target > 0 else 60.0, 200)
    mr_values = params["a"] * np.exp(-params["k"] * t_values**params["n"]) + params["b"] * t_values

    plt.figure(figsize=(8, 5))
    plt.plot(t_values, mr_values, label="Predicted MR")
    plt.axhline(target_mr, color="red", linestyle="--", label=f"MR={target_mr}")
    plt.xlabel("Time (min)")
    plt.ylabel("Moisture Ratio (MR)")
    plt.title("Predicted Drying Curve (Midilli-a Model)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()
    print(f"Drying curve plot saved to: {PLOT_PATH}")


if __name__ == "__main__":
    _run_example()
