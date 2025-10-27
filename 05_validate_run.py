"""Validate Master Model predictions against a specific experimental run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - optional fallback
    import pickle

    class _JoblibFallback:
        @staticmethod
        def load(filename):
            with open(filename, "rb") as fh:
                return pickle.load(fh)

    joblib = _JoblibFallback()


TARGET_EXPERIMENT_FILE = "T50_v0p60.csv"
TARGET_MR = 0.1

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
PHASE1_RESULTS_PATH = ROOT_DIR / "phase1_out" / "phase1_full_results.csv"
OUTPUT_DIR = ROOT_DIR / "phase1_out"


def _load_model(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required model file not found: {path}")
    return joblib.load(path)


MODEL_K = _load_model("model_k.pkl")
MODEL_A = _load_model("model_a.pkl")
MODEL_N = _load_model("model_n.pkl")
MODEL_B = _load_model("model_b.pkl")

PROCESS_BOUNDS_PATH = MODELS_DIR / "process_bounds.json"
if not PROCESS_BOUNDS_PATH.exists():
    raise FileNotFoundError(
        "Process bounds file not found. Expected at " f"{PROCESS_BOUNDS_PATH}"
    )
with PROCESS_BOUNDS_PATH.open("r", encoding="utf-8") as fh:
    PROCESS_BOUNDS = json.load(fh)

ABSOLUTE_ZERO_C = 273.15


def _check_bounds(values: Dict[str, float]) -> None:
    out_of_bounds = []
    for key, value in values.items():
        bounds = PROCESS_BOUNDS.get(key)
        if not bounds:
            continue
        lower = bounds.get("min", -np.inf)
        upper = bounds.get("max", np.inf)
        if value < lower or value > upper:
            out_of_bounds.append(key)

    if out_of_bounds:
        labels = ", ".join(out_of_bounds)
        print(
            "WARNING: Input values for "
            f"{labels} are outside the training bounds; prediction is an extrapolation."
        )


def _linear_model_predict(model, features: Dict[str, float]) -> float:
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
    T_C: float,
    v_ms: float,
    RH_pct: float,
    thickness_mm: float,
) -> Dict[str, float]:
    _check_bounds(
        {
            "T_C": T_C,
            "v_ms": v_ms,
            "RH_pct": RH_pct,
            "thickness_mm": thickness_mm,
        }
    )

    T_abs = T_C + ABSOLUTE_ZERO_C
    if T_abs <= 0:
        raise ValueError("Absolute temperature must be positive.")

    ln_k = _linear_model_predict(MODEL_K, {"inv_T_abs": 1.0 / T_abs})
    k_value = float(np.exp(ln_k))

    shared_features = {
        "v_ms": v_ms,
        "RH_pct": RH_pct,
        "thickness_mm": thickness_mm,
    }
    a_value = float(_linear_model_predict(MODEL_A, shared_features))
    n_value = float(_linear_model_predict(MODEL_N, shared_features))
    b_value = float(_linear_model_predict(MODEL_B, shared_features))

    return {
        "a": a_value,
        "k": k_value,
        "n": n_value,
        "b": b_value,
        "tau": 0.0,
    }


def predict_time_to_mr(mr_target: float, params: Dict[str, float]) -> float:
    from scipy.optimize import fsolve

    a = float(params["a"])
    k = float(params["k"])
    n = float(params["n"])
    b = float(params["b"])

    def residual(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        t_nonneg = np.maximum(0.0, t)
        mr_prediction = a * np.exp(-k * t_nonneg**n) + b * t
        return mr_prediction - mr_target

    initial_guess = np.array([max(mr_target / max(b, 1e-6), 1.0)])
    solution = fsolve(residual, x0=initial_guess)
    time_value = float(solution[0])
    return max(time_value, 0.0)


def load_phase1_results() -> pd.DataFrame:
    if not PHASE1_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Phase 1 full results CSV not found at " f"{PHASE1_RESULTS_PATH}"
        )
    return pd.read_csv(PHASE1_RESULTS_PATH)


def load_experimental_data(filename: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    experiment_path = ROOT_DIR / filename
    if not experiment_path.exists():
        raise FileNotFoundError(
            "Experimental data file not found at " f"{experiment_path}"
        )

    df = pd.read_csv(experiment_path)
    if df.empty:
        raise ValueError(f"Experimental dataset {filename} is empty.")

    df = df.sort_values("time_min").reset_index(drop=True)
    first_row = df.iloc[0]
    conditions = {
        "T_C": float(first_row["T_C"]),
        "v_ms": float(first_row["v_ms"]),
        "RH_pct": float(first_row["RH_pct"]),
        "thickness_mm": float(first_row["thickness_mm"]),
    }

    return df, conditions


def compute_mr_and_time(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_raw = df["time_min"].to_numpy(dtype=float)
    time_norm = time_raw - time_raw[0]
    x_db = df["X_db"].to_numpy(dtype=float)
    mr = x_db / x_db[0]
    return time_raw, time_norm, mr


def compute_time_to_target(time_values: np.ndarray, mr: np.ndarray, target: float) -> float:
    if np.any(np.isnan(mr)):
        raise ValueError("Experimental data contains NaN moisture ratio values.")

    below = np.nonzero(mr <= target)[0]
    if len(below) == 0:
        return float("nan")

    idx = int(below[0])
    if idx == 0 or mr[idx] == target:
        return float(time_values[idx])

    prev_idx = idx - 1
    t0, t1 = float(time_values[prev_idx]), float(time_values[idx])
    mr0, mr1 = float(mr[prev_idx]), float(mr[idx])

    if mr0 == mr1:
        return t1

    return float(np.interp(target, [mr1, mr0], [t1, t0]))


def get_phase1_parameters(df_phase1: pd.DataFrame, run_id: str) -> Dict[str, float]:
    matching = df_phase1[df_phase1["run_id"] == run_id]
    if matching.empty:
        raise KeyError(f"Run ID '{run_id}' not found in Phase 1 results.")

    row = matching.iloc[0]
    return {
        "a": float(row["Midilli-a_a"]),
        "k": float(row["Midilli-a_k"]),
        "n": float(row["Midilli-a_n"]),
        "b": float(row["Midilli-a_b"]),
    }


def build_prediction_curve(params: Dict[str, float], max_time: float) -> Tuple[np.ndarray, np.ndarray]:
    time_values = np.linspace(0.0, max(max_time, 1e-6), 300)
    mr_values = (
        params["a"] * np.exp(-params["k"] * time_values**params["n"]) + params["b"] * time_values
    )
    return time_values, mr_values


def save_plot(
    experiment_name: str,
    time_norm: np.ndarray,
    mr_exp: np.ndarray,
    pred_time: np.ndarray,
    pred_mr: np.ndarray,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / f"validation_plot_{experiment_name}.png"

    plt.figure(figsize=(8, 5))
    plt.scatter(time_norm, mr_exp, color="tab:blue", label="Experimental data")
    plt.plot(pred_time, pred_mr, color="tab:orange", label="Predicted curve")
    plt.axhline(y=TARGET_MR, color="gray", linestyle="--", linewidth=1, label=f"Target MR={TARGET_MR}")
    plt.xlabel("Time (min, normalized)")
    plt.ylabel("Moisture Ratio (MR)")
    plt.title(f"Validation of Master Model for {experiment_name}")
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path


def save_summary(
    experiment_name: str,
    run_id: str,
    time_experimental: float,
    time_predicted: float,
    params_experimental: Dict[str, float],
    params_predicted: Dict[str, float],
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / f"validation_summary_{experiment_name}.csv"

    summary_df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "target_MR": TARGET_MR,
                "time_experimental": time_experimental,
                "time_predicted": time_predicted,
                "a_experimental": params_experimental["a"],
                "k_experimental": params_experimental["k"],
                "n_experimental": params_experimental["n"],
                "b_experimental": params_experimental["b"],
                "a_predicted": params_predicted["a"],
                "k_predicted": params_predicted["k"],
                "n_predicted": params_predicted["n"],
                "b_predicted": params_predicted["b"],
            }
        ]
    )

    summary_df.to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    experiment_name = Path(TARGET_EXPERIMENT_FILE).stem
    run_id = experiment_name

    df_phase1_results = load_phase1_results()
    params_experimental = get_phase1_parameters(df_phase1_results, run_id)

    df_experiment, conditions = load_experimental_data(TARGET_EXPERIMENT_FILE)
    time_raw, time_norm, mr_exp = compute_mr_and_time(df_experiment)
    time_experimental = compute_time_to_target(time_raw, mr_exp, TARGET_MR)

    params_predicted = predict_parameters(**conditions)
    time_predicted = predict_time_to_mr(TARGET_MR, params_predicted)

    max_time_for_plot = max(
        np.nanmax(time_norm) if len(time_norm) else 0.0,
        time_predicted if np.isfinite(time_predicted) else 0.0,
    )
    pred_time, pred_mr = build_prediction_curve(params_predicted, max_time_for_plot)

    plot_path = save_plot(experiment_name, time_norm, mr_exp, pred_time, pred_mr)
    summary_path = save_summary(
        experiment_name,
        run_id,
        time_experimental,
        time_predicted,
        params_experimental,
        params_predicted,
    )

    print("Validation complete.")
    print(f"Plot saved to: {plot_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
