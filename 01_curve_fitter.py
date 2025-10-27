"""Phase 1 drying curve fitter.

This script loads all CSV drying experiments in the repository root,
performs time normalisation, Moisture Ratio (MR) calculation, isotonic
regression smoothing, fits three drying models (Page, Midilli, Midilli-a)
with an optimised time delay, and generates plots plus a master summary
CSV in the ``phase1_out`` directory.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "phase1_out"
PLOTS_DIR = OUTPUT_DIR / "plots"


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)


def load_csv_files() -> List[Path]:
    return sorted(ROOT_DIR.glob("*.csv"))


def normalise_time(time_series: pd.Series) -> np.ndarray:
    t_raw = time_series.to_numpy(dtype=float)
    if t_raw.size == 0:
        return t_raw
    return t_raw - t_raw[0]


def compute_moisture_ratio(x_db_series: pd.Series) -> np.ndarray:
    x_db = x_db_series.to_numpy(dtype=float)
    if x_db.size == 0:
        return x_db
    x0 = x_db[0]
    if x0 == 0:
        raise ValueError("Initial dry basis moisture content cannot be zero.")
    return x_db / x0


def pava_non_decreasing(values: Iterable[float]) -> np.ndarray:
    """Pool Adjacent Violators Algorithm for a non-decreasing fit."""

    y = np.asarray(list(values), dtype=float)
    n = y.size
    if n == 0:
        return y

    g = y.copy()
    w = np.ones(n, dtype=float)

    i = 0
    while i < n - 1:
        if g[i] > g[i + 1]:
            total_weight = w[i] + w[i + 1]
            total_level = (g[i] * w[i] + g[i + 1] * w[i + 1]) / total_weight
            g[i] = g[i + 1] = total_level
            w[i] = w[i + 1] = total_weight

            j = i
            while j > 0 and g[j - 1] > g[j]:
                total_weight = w[j - 1] + w[j]
                total_level = (
                    g[j - 1] * w[j - 1] + g[j] * w[j]
                ) / total_weight
                g[j - 1] = g[j] = total_level
                w[j - 1] = w[j] = total_weight
                j -= 1
        i += 1

    # Expand block averages
    fitted = g.copy()
    i = 0
    while i < n:
        j = i + 1
        while j < n and math.isclose(g[j], g[i], rel_tol=1e-12, abs_tol=1e-12):
            j += 1
        fitted[i:j] = np.mean(g[i:j])
        i = j

    return fitted


def pava_monotone_decreasing(values: Iterable[float]) -> np.ndarray:
    return -pava_non_decreasing(-np.asarray(list(values), dtype=float))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def aicc(n: int, rss: float, num_params: int) -> float:
    if n <= num_params + 1:
        return float("inf")
    if rss <= 0:
        rss = 1e-12
    aic = n * math.log(rss / n) + 2 * num_params
    correction = (2 * num_params * (num_params + 1)) / (n - num_params - 1)
    return aic + correction


def build_tau_candidates(t: np.ndarray, num: int = 40) -> np.ndarray:
    max_tau = float(np.max(t)) if t.size else 0.0
    if max_tau <= 0:
        return np.array([0.0])
    upper = 0.5 * max_tau
    if upper <= 0:
        upper = max_tau
    candidates = np.linspace(0.0, upper, num=num)
    # refine around zero for early-time sensitivity
    fine = np.linspace(0.0, min(upper, max_tau * 0.1), num=max(5, num // 4))
    return np.unique(np.concatenate([candidates, fine]))


def effective_time(t_norm: np.ndarray, tau: float) -> np.ndarray:
    return np.clip(t_norm - tau, a_min=0.0, a_max=None)


ModelFunc = Callable[[np.ndarray, float, float], np.ndarray]


def page_model(t: np.ndarray, k: float, n: float) -> np.ndarray:
    return np.exp(-k * np.power(t, n))


def midilli_model(t: np.ndarray, k: float, n: float, b: float) -> np.ndarray:
    return np.exp(-k * np.power(t, n)) + b * t


def midilli_a_model(t: np.ndarray, a: float, k: float, n: float, b: float) -> np.ndarray:
    return a * np.exp(-k * np.power(t, n)) + b * t


@dataclass
class ModelConfig:
    name: str
    func: Callable
    param_names: Tuple[str, ...]
    initial_guess: Callable[[np.ndarray, np.ndarray], Tuple[float, ...]]
    bounds: Tuple[Tuple[float, ...], Tuple[float, ...]]


def default_initial_guess(t_eff: np.ndarray, mr: np.ndarray, extra: int = 0) -> Tuple[float, ...]:
    k0 = 0.01 if t_eff.max(initial=0) == 0 else 0.5 / (t_eff.max() + 1e-12)
    n0 = 1.0
    guesses = [k0, n0]
    if extra >= 1:
        guesses.append(0.0)
    if extra >= 2:
        guesses.insert(0, 1.0)  # a parameter
    return tuple(guesses)


MODELS: Dict[str, ModelConfig] = {
    "page": ModelConfig(
        name="Page",
        func=page_model,
        param_names=("k", "n"),
        initial_guess=lambda t, y: default_initial_guess(t, y, extra=0),
        bounds=((1e-8, 0.1), (10.0, 3.0)),
    ),
    "midilli": ModelConfig(
        name="Midilli",
        func=midilli_model,
        param_names=("k", "n", "b"),
        initial_guess=lambda t, y: default_initial_guess(t, y, extra=1),
        bounds=((1e-8, 0.1, -1.0), (10.0, 3.0, 1.0)),
    ),
    "midilli_a": ModelConfig(
        name="Midilli-a",
        func=midilli_a_model,
        param_names=("a", "k", "n", "b"),
        initial_guess=lambda t, y: default_initial_guess(t, y, extra=2),
        bounds=((0.3, 1e-8, 0.1, -1.0), (1.7, 10.0, 3.0, 1.0)),
    ),
}


@dataclass
class FitResult:
    model_key: str
    params: Tuple[float, ...]
    tau: float
    rmse: float
    aicc: float


def fit_model(
    config: ModelConfig,
    t_norm: np.ndarray,
    mr_raw: np.ndarray,
    mr_iso: np.ndarray,
) -> FitResult:
    n_samples = mr_raw.size
    best_result: FitResult | None = None

    tau_candidates = build_tau_candidates(t_norm)

    for tau in tau_candidates:
        t_eff = effective_time(t_norm, tau)
        if np.allclose(t_eff, 0.0):
            continue

        try:
            initial = config.initial_guess(t_eff, mr_iso)
            params, _ = curve_fit(
                config.func,
                t_eff,
                mr_iso,
                p0=initial,
                bounds=config.bounds,
                maxfev=20000,
            )
        except Exception:
            continue

        predictions = config.func(t_eff, *params)
        residuals = mr_raw - predictions
        rss = float(np.sum(residuals**2))
        rmse_val = rmse(mr_raw, predictions)
        aicc_val = aicc(n_samples, rss, len(params) + 1)  # include tau

        if not math.isfinite(aicc_val):
            continue

        if best_result is None or aicc_val < best_result.aicc:
            best_result = FitResult(
                model_key=config.name,
                params=tuple(float(p) for p in params),
                tau=float(tau),
                rmse=float(rmse_val),
                aicc=float(aicc_val),
            )

    if best_result is None:
        best_result = FitResult(
            model_key=config.name,
            params=tuple(math.nan for _ in config.param_names),
            tau=float("nan"),
            rmse=float("nan"),
            aicc=float("inf"),
        )

    return best_result


def create_plot(
    csv_path: Path,
    t_norm: np.ndarray,
    mr_raw: np.ndarray,
    results: Dict[str, FitResult],
) -> None:
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

    ax_curve = fig.add_subplot(gs[0])
    ax_residual = fig.add_subplot(gs[1], sharex=ax_curve)

    ax_curve.scatter(t_norm, mr_raw, color="black", label="Raw MR", s=20, alpha=0.7)

    t_grid = np.linspace(0, float(np.max(t_norm)) if t_norm.size else 0.0, 300)

    colors = {
        "Page": "tab:blue",
        "Midilli": "tab:orange",
        "Midilli-a": "tab:green",
    }

    for model_key, result in results.items():
        if not all(math.isfinite(p) for p in result.params) or not math.isfinite(result.tau):
            continue
        t_eff_grid = effective_time(t_grid, result.tau)
        pred_grid = MODELS[model_key.lower().replace("-", "_")].func(t_eff_grid, *result.params)
        ax_curve.plot(
            t_grid,
            pred_grid,
            label=f"{result.model_key} (AICc={result.aicc:.2f})",
            color=colors.get(result.model_key, None),
        )

    best_model_key = min(results, key=lambda k: results[k].aicc)
    best_result = results[best_model_key]
    if all(math.isfinite(p) for p in best_result.params) and math.isfinite(best_result.tau):
        t_eff_best = effective_time(t_norm, best_result.tau)
        predictions = MODELS[best_model_key.lower().replace("-", "_")].func(
            t_eff_best, *best_result.params
        )
        residuals = mr_raw - predictions
        ax_residual.scatter(t_norm, residuals, color="tab:red", s=20)
        ax_residual.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax_residual.set_ylabel("Residual")
        ax_residual.set_xlabel("Time (min, normalised)")

    ax_curve.set_xlabel("Time (min, normalised)")
    ax_curve.set_ylabel("Moisture Ratio (MR)")
    ax_curve.set_title(csv_path.stem)
    ax_curve.legend()

    plot_path = PLOTS_DIR / f"{csv_path.stem}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)


def process_file(csv_path: Path) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV file {csv_path} is empty.")

    time_col = None
    for candidate in ("t", "time", "time_min"):
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        raise KeyError(f"No time column found in {csv_path}.")

    t_norm = normalise_time(df[time_col])
    mr_raw = compute_moisture_ratio(df["X_db"])
    mr_iso = pava_monotone_decreasing(mr_raw)

    results: Dict[str, FitResult] = {}
    for key, config in MODELS.items():
        result = fit_model(config, t_norm, mr_raw, mr_iso)
        results[key] = result

    create_plot(csv_path, t_norm, mr_raw, results)

    best_key = min(results, key=lambda k: results[k].aicc)
    best_result = results[best_key]

    meta = df.iloc[0]
    summary = {
        "run_id": meta.get("run_id", csv_path.stem),
        "T_C": meta.get("T_C", np.nan),
        "v_ms": meta.get("v_ms", np.nan),
        "RH_pct": meta.get("RH_pct", np.nan),
        "thickness_mm": meta.get("thickness_mm", np.nan),
        "best_model": best_result.model_key,
        "param_a": np.nan,
        "param_k": np.nan,
        "param_n": np.nan,
        "param_b": np.nan,
        "param_tau": best_result.tau,
        "RMSE": best_result.rmse,
        "AICc": best_result.aicc,
    }

    param_map = dict(zip(MODELS[best_key].param_names, best_result.params))
    if "a" in param_map:
        summary["param_a"] = param_map["a"]
    if "k" in param_map:
        summary["param_k"] = param_map["k"]
    if "n" in param_map:
        summary["param_n"] = param_map["n"]
    if "b" in param_map:
        summary["param_b"] = param_map["b"]

    return summary


def main() -> None:
    ensure_output_dirs()
    csv_files = load_csv_files()

    if not csv_files:
        raise FileNotFoundError("No CSV files were found in the repository root.")

    summaries: List[Dict[str, object]] = []
    for csv_path in csv_files:
        print(f"Processing {csv_path.name}...")
        summary = process_file(csv_path)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.sort_values("run_id", inplace=True)
    summary_path = OUTPUT_DIR / "phase1_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

