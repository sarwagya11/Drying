"""Compare Midilli-a model predictions to experimental drying data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    import pickle

    class _JoblibFallback:
        @staticmethod
        def load(filename):
            with open(filename, "rb") as fh:
                return pickle.load(fh)

    joblib = _JoblibFallback()


ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "phase1_out"


MODEL_FILES = {
    "k": "model_k.pkl",
    "a": "model_a.pkl",
    "n": "model_n.pkl",
    "b": "model_b.pkl",
}

PROCESS_BOUNDS_FILE = MODELS_DIR / "process_bounds.json"


def load_model(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required model file not found: {path}")
    return joblib.load(path)


def load_bounds(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Process bounds file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def midilli_a_model(t: np.ndarray, a: float, k: float, n: float, b: float) -> np.ndarray:
    return a * np.exp(-k * np.power(t, n)) + b * t


def warn_out_of_bounds(values: Dict[str, float], bounds: Dict[str, Dict[str, float]]) -> None:
    out_of_bounds: list[str] = []
    for key, value in values.items():
        limits = bounds.get(key)
        if not limits:
            continue
        lower = limits.get("min", -np.inf)
        upper = limits.get("max", np.inf)
        if value < lower or value > upper:
            out_of_bounds.append(f"{key} ({value})")
    if out_of_bounds:
        joined = ", ".join(out_of_bounds)
        print(
            "WARNING: The following inputs are outside the training bounds and may "
            f"require extrapolation: {joined}"
        )


def linear_model_predict(model, features: Dict[str, float]) -> float:
    prediction = 0.0
    params = model.params
    intercept = params.get("const")
    if intercept is not None:
        prediction += float(intercept)
    for name, value in features.items():
        if name not in params:
            continue
        prediction += float(params[name]) * value
    return prediction


def predict_parameters(
    models: Dict[str, object],
    bounds: Dict[str, Dict[str, float]],
    T_C: float,
    v_ms: float,
    RH_pct: float,
    thickness_mm: float,
) -> Dict[str, float]:
    conditions = {
        "T_C": T_C,
        "v_ms": v_ms,
        "RH_pct": RH_pct,
        "thickness_mm": thickness_mm,
    }
    warn_out_of_bounds(conditions, bounds)

    T_abs = T_C + 273.15
    if T_abs <= 0:
        raise ValueError("Absolute temperature must be positive.")

    ln_k = linear_model_predict(models["k"], {"inv_T_abs": 1.0 / T_abs})
    k_value = float(np.exp(ln_k))

    shared = {"v_ms": v_ms, "RH_pct": RH_pct, "thickness_mm": thickness_mm}
    a_value = float(linear_model_predict(models["a"], shared))
    n_value = float(linear_model_predict(models["n"], shared))
    b_value = float(linear_model_predict(models["b"], shared))

    return {"a": a_value, "k": k_value, "n": n_value, "b": b_value}


def main() -> None:
    models = {name: load_model(filename) for name, filename in MODEL_FILES.items()}
    process_bounds = load_bounds(PROCESS_BOUNDS_FILE)

    target_experiment_file = "T50_v0p60.csv"
    target_path = ROOT_DIR / target_experiment_file
    if not target_path.exists():
        raise FileNotFoundError(f"Target experiment file not found: {target_path}")

    exp_df = pd.read_csv(target_path)

    time_raw = exp_df["time_min"].to_numpy(dtype=float)
    time_norm = time_raw - time_raw[0]
    x_db = exp_df["X_db"].to_numpy(dtype=float)
    mr_exp = x_db / x_db[0]

    conditions = {
        "T_C": float(exp_df["T_C"].iloc[0]),
        "v_ms": float(exp_df["v_ms"].iloc[0]),
        "RH_pct": float(exp_df["RH_pct"].iloc[0]),
        "thickness_mm": float(exp_df["thickness_mm"].iloc[0]),
    }

    params = predict_parameters(models, process_bounds, **conditions)
    print("Predicted Midilli-a parameters for", target_experiment_file)
    for name in ["a", "k", "n", "b"]:
        print(f"  {name} = {params[name]:.6f}")

    t_max = float(time_norm.max(initial=0.0))
    t_range = np.linspace(0.0, t_max, 300)
    mr_pred = midilli_a_model(t_range, params["a"], params["k"], params["n"], params["b"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_name = Path(target_experiment_file).stem
    plot_path = OUTPUT_DIR / f"comparison_plot_{experiment_name}.png"

    plt.figure(figsize=(8, 5))
    plt.plot(t_range, mr_pred, color="green", label="Predicted Curve")
    plt.scatter(time_norm, mr_exp, color="blue", label="Experimental Data", zorder=3)
    plt.xlabel("Time (min)")
    plt.ylabel("Moisture Ratio")
    plt.title(f"Midilli-a Prediction vs Experimental Data: {experiment_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
