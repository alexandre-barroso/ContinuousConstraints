#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, minimize
from scipy.special import logsumexp
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler


EPS = 1e-12
NUMERIC_EPS = 1e-6


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", quotechar='"', header=0)
    required = {"F1", "F2", "Vogal", "Falante"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in data file: {sorted(missing)}")
    df = df[(df["F1"] != "NA") & (df["F2"] != "NA")].copy()
    df["F1"] = pd.to_numeric(df["F1"])
    df["F2"] = pd.to_numeric(df["F2"])
    return df


def subset_data(df: pd.DataFrame, vowels: Sequence[str], speakers: Optional[Sequence[int]]) -> pd.DataFrame:
    out = df[df["Vogal"].isin(vowels)].copy()
    if speakers is not None:
        out = out[out["Falante"].isin(speakers)].copy()
    return out[["F1", "F2", "Vogal", "Falante"]].reset_index(drop=True)


def make_reference_params(df_subset: pd.DataFrame, base_params: Dict) -> Dict:
    p = dict(base_params)
    p["a_F1"] = float(df_subset["F1"].min())
    p["b_F1"] = float(df_subset["F1"].max())
    p["a_F2"] = float(df_subset["F2"].min())
    p["b_F2"] = float(df_subset["F2"].max())
    p["min_F1"] = float(df_subset["F1"].min())
    p["max_F1"] = float(df_subset["F1"].max())
    p["min_F2"] = float(df_subset["F2"].min())
    p["max_F2"] = float(df_subset["F2"].max())
    return p


def z_pair(scaler: StandardScaler, f1: float, f2: float) -> Tuple[float, float]:
    z = scaler.transform([[f1, f2]])[0]
    return float(z[0]), float(z[1])


def build_z_params(params_ref: Dict, scaler: StandardScaler) -> Dict:
    alvo_F1_z, _ = z_pair(scaler, params_ref["alvo_F1"], params_ref["alvo_F2"])
    _, alvo_F2_z = z_pair(scaler, params_ref["alvo_F1"], params_ref["alvo_F2"])

    limiar_1_z, _ = z_pair(scaler, params_ref["limiar_1"], params_ref["alvo_F2"])
    limiar_2_z, _ = z_pair(scaler, params_ref["limiar_2"], params_ref["alvo_F2"])

    neutro_F1_z, _ = z_pair(scaler, params_ref["neutro_F1"], params_ref["neutro_F2"])
    _, neutro_F2_z = z_pair(scaler, params_ref["neutro_F1"], params_ref["neutro_F2"])

    a_F1_z, _ = z_pair(scaler, params_ref["a_F1"], params_ref["alvo_F2"])
    b_F1_z, _ = z_pair(scaler, params_ref["b_F1"], params_ref["alvo_F2"])
    min_F1_z, _ = z_pair(scaler, params_ref["min_F1"], params_ref["alvo_F2"])
    max_F1_z, _ = z_pair(scaler, params_ref["max_F1"], params_ref["alvo_F2"])

    _, a_F2_z = z_pair(scaler, params_ref["alvo_F1"], params_ref["a_F2"])
    _, b_F2_z = z_pair(scaler, params_ref["alvo_F1"], params_ref["b_F2"])
    _, min_F2_z = z_pair(scaler, params_ref["alvo_F1"], params_ref["min_F2"])
    _, max_F2_z = z_pair(scaler, params_ref["alvo_F1"], params_ref["max_F2"])

    return {
        "alvo_F1": alvo_F1_z,
        "alvo_F2": alvo_F2_z,
        "limiar_1": limiar_1_z,
        "limiar_2": limiar_2_z,
        "neutro_F1": neutro_F1_z,
        "neutro_F2": neutro_F2_z,
        "a_F1": a_F1_z,
        "b_F1": b_F1_z,
        "a_F2": a_F2_z,
        "b_F2": b_F2_z,
        "min_F1": min_F1_z,
        "max_F1": max_F1_z,
        "min_F2": min_F2_z,
        "max_F2": max_F2_z,
        "L": float(params_ref["L"]),
        "k_1": float(params_ref["k_1"]),
        "k_2": float(params_ref["k_2"]),
    }


def auto_params_for_subset(df_subset: pd.DataFrame, template_params: Dict) -> Dict:
    p = dict(template_params)
    f1_min = float(df_subset["F1"].min())
    f1_max = float(df_subset["F1"].max())
    f1_span = max(f1_max - f1_min, 1.0)

    p["alvo_F1"] = float(df_subset["F1"].median())
    p["alvo_F2"] = float(df_subset["F2"].median())
    p["limiar_2"] = float(df_subset["F1"].quantile(0.15))
    p["limiar_1"] = float(f1_max + 0.20 * f1_span)
    p["neutro_F1"] = float(f1_max + 0.20 * f1_span)
    p["neutro_F2"] = float(df_subset["F2"].max())
    p["L"] = template_params["L"]
    p["k_1"] = template_params["k_1"]
    p["k_2"] = template_params["k_2"]
    return p


def fit_kde_and_scaler(df_subset: pd.DataFrame, bandwidth: float) -> Tuple[gaussian_kde, StandardScaler]:
    scaler = StandardScaler()
    values = df_subset[["F1", "F2"]].to_numpy()
    values_z = scaler.fit_transform(values)
    kde = gaussian_kde(values_z.T, bw_method=bandwidth)
    return kde, scaler


def perceptual_constraint(F1: np.ndarray, z_params: Dict) -> np.ndarray:
    limiar_1 = z_params["limiar_1"]
    limiar_2 = z_params["limiar_2"]
    L = z_params["L"]
    k_1 = z_params["k_1"]
    k_2 = z_params["k_2"]
    produto = (L ** 2) / (
        (1.0 + np.exp(k_1 * (F1 - limiar_1))) *
        (1.0 + np.exp(-k_2 * (F1 - limiar_2)))
    )
    return L - produto


def articulatory_constraint(F1: np.ndarray, F2: np.ndarray, z_params: Dict) -> np.ndarray:
    alvo_F1 = z_params["alvo_F1"]
    alvo_F2 = z_params["alvo_F2"]
    neutro_F1 = z_params["neutro_F1"]
    neutro_F2 = z_params["neutro_F2"]

    esforco_alvo = np.sqrt((alvo_F1 - neutro_F1) ** 2 + (alvo_F2 - neutro_F2) ** 2)
    esforco_producao = np.sqrt((F1 - neutro_F1) ** 2 + (F2 - neutro_F2) ** 2)
    distancia = np.sqrt((F1 - alvo_F1) ** 2 + (F2 - alvo_F2) ** 2)
    dif_esforco = (esforco_producao + NUMERIC_EPS) / (esforco_alvo + NUMERIC_EPS)
    return distancia * dif_esforco


@dataclass
class PrecomputedModel:
    p_mass: np.ndarray
    V_A: np.ndarray
    V_P: np.ndarray
    E_p_VA: float
    E_p_VP: float
    n_grid: int


def precompute_for_subset(
    df_subset: pd.DataFrame,
    params_base: Dict,
    bandwidth: float,
    resolution: int,
) -> PrecomputedModel:
    if len(df_subset) < 10:
        raise ValueError("Subset too small for stable KDE-based fitting (<10 tokens).")

    params_ref = make_reference_params(df_subset, params_base)
    kde, scaler = fit_kde_and_scaler(df_subset, bandwidth)
    z_params = build_z_params(params_ref, scaler)

    F1 = np.linspace(z_params["min_F1"], z_params["max_F1"], resolution)
    F2 = np.linspace(z_params["min_F2"], z_params["max_F2"], resolution)
    F1_mesh, F2_mesh = np.meshgrid(F1, F2)
    pts = np.vstack([F1_mesh.ravel(), F2_mesh.ravel()])

    kde_vals = kde(pts).reshape(F1_mesh.shape)
    p_mass = kde_vals.ravel().astype(np.float64)
    p_mass /= max(p_mass.sum(), EPS)

    p_matrix = p_mass.reshape(F1_mesh.shape)
    marginal_f1 = p_matrix.sum(axis=0)
    marginal_f1 = np.clip(marginal_f1, EPS, None)
    marginal_f1 /= marginal_f1.sum()

    C_P = perceptual_constraint(F1, z_params)
    V_P_1d = np.exp(C_P - marginal_f1)
    V_P = np.tile(V_P_1d, (resolution, 1)).ravel().astype(np.float64)

    C_A = articulatory_constraint(F1_mesh, F2_mesh, z_params)
    kde_safe = np.clip(p_matrix, EPS, None)
    V_A = np.exp(C_A - kde_safe).ravel().astype(np.float64)

    E_p_VA = float(np.dot(p_mass, V_A))
    E_p_VP = float(np.dot(p_mass, V_P))

    return PrecomputedModel(
        p_mass=p_mass,
        V_A=V_A,
        V_P=V_P,
        E_p_VA=E_p_VA,
        E_p_VP=E_p_VP,
        n_grid=resolution * resolution,
    )


def objective_and_grad(lambdas: np.ndarray, model: PrecomputedModel) -> Tuple[float, np.ndarray]:
    lambda_RA = float(lambdas[0])
    lambda_RP = float(lambdas[1])

    logits = -lambda_RA * model.V_A - lambda_RP * model.V_P
    logZ = logsumexp(logits)
    q_mass = np.exp(logits - logZ)

    obj = lambda_RA * model.E_p_VA + lambda_RP * model.E_p_VP + logZ
    grad = np.array([
        model.E_p_VA - float(np.dot(q_mass, model.V_A)),
        model.E_p_VP - float(np.dot(q_mass, model.V_P)),
    ], dtype=np.float64)
    return float(obj), grad


def kl_from_lambdas(lambdas: Sequence[float], model: PrecomputedModel) -> float:
    lambda_RA = float(lambdas[0])
    lambda_RP = float(lambdas[1])
    logits = -lambda_RA * model.V_A - lambda_RP * model.V_P
    logZ = logsumexp(logits)
    q_mass = np.exp(logits - logZ)
    return float(np.sum(model.p_mass * (np.log(np.clip(model.p_mass, EPS, None)) - np.log(np.clip(q_mass, EPS, None)))))


def fit_weights(
    model: PrecomputedModel,
    init_lambdas: Tuple[float, float],
    bounds_2d: Tuple[Tuple[float, float], Tuple[float, float]],
) -> Dict:
    def fun(x):
        return objective_and_grad(x, model)

    result = minimize(
        fun,
        x0=np.array(init_lambdas, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=Bounds(
            [bounds_2d[0][0], bounds_2d[1][0]],
            [bounds_2d[0][1], bounds_2d[1][1]],
        ),
        options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-8},
    )

    fitted = np.array(result.x, dtype=np.float64)
    baseline = np.array(init_lambdas, dtype=np.float64)

    baseline_kl = kl_from_lambdas(baseline, model)
    fitted_kl = kl_from_lambdas(fitted, model)

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "lambda_RA": float(fitted[0]),
        "lambda_RP": float(fitted[1]),
        "baseline_kl": baseline_kl,
        "fitted_kl": fitted_kl,
        "improvement": baseline_kl - fitted_kl,
        "nit": int(getattr(result, "nit", -1)),
    }


def fit_subset_fast(
    df_subset: pd.DataFrame,
    params_template: Dict,
    bandwidth: float,
    coarse_resolution: int,
    fine_resolution: int,
    init_lambdas: Tuple[float, float],
    bounds_2d,
    auto_parametrize: bool = False,
) -> Dict:
    t0 = perf_counter()

    params_base = auto_params_for_subset(df_subset, params_template) if auto_parametrize else dict(params_template)

    coarse = precompute_for_subset(df_subset, params_base, bandwidth, coarse_resolution)
    coarse_fit = fit_weights(coarse, init_lambdas, bounds_2d)

    fine = precompute_for_subset(df_subset, params_base, bandwidth, fine_resolution)
    fine_fit = fit_weights(fine, (coarse_fit["lambda_RA"], coarse_fit["lambda_RP"]), bounds_2d)

    elapsed = perf_counter() - t0
    out = dict(fine_fit)
    out["coarse_lambda_RA"] = coarse_fit["lambda_RA"]
    out["coarse_lambda_RP"] = coarse_fit["lambda_RP"]
    out["coarse_fitted_kl"] = coarse_fit["fitted_kl"]
    out["n_tokens"] = int(len(df_subset))
    out["n_speakers"] = int(df_subset["Falante"].nunique())
    out["vowels"] = ",".join(sorted(df_subset["Vogal"].astype(str).unique()))
    out["speakers"] = ",".join(map(str, sorted(df_subset["Falante"].unique())))
    out["elapsed_seconds"] = float(elapsed)
    return out


def bootstrap_tasks(df_focus: pd.DataFrame, n_boot: int, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    n = len(df_focus)
    tasks = []
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = df_focus.iloc[idx].reset_index(drop=True)
        tasks.append({"kind": "bootstrap", "iteration": i + 1, "subset": sample})
    return tasks


def split_half_tasks(df_focus: pd.DataFrame, n_splits: int, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    n = len(df_focus)
    idx = np.arange(n)
    tasks = []
    for i in range(n_splits):
        rng.shuffle(idx)
        half = n // 2
        left = df_focus.iloc[idx[:half]].reset_index(drop=True)
        right = df_focus.iloc[idx[half:]].reset_index(drop=True)
        if len(left) >= 10:
            tasks.append({"kind": "split_half", "iteration": i + 1, "split": "half_A", "subset": left})
        if len(right) >= 10:
            tasks.append({"kind": "split_half", "iteration": i + 1, "split": "half_B", "subset": right})
    return tasks


def speaker_tasks(df_focus: pd.DataFrame) -> List[Dict]:
    tasks = []
    speakers = sorted(df_focus["Falante"].unique())
    for spk in speakers:
        left_out = df_focus[df_focus["Falante"] != spk].reset_index(drop=True)
        only_spk = df_focus[df_focus["Falante"] == spk].reset_index(drop=True)
        if len(left_out) >= 10:
            tasks.append({"kind": "speaker_holdout", "condition": "leave_one_speaker_out", "speaker": int(spk), "subset": left_out})
        if len(only_spk) >= 10:
            tasks.append({"kind": "speaker_holdout", "condition": "single_speaker_only", "speaker": int(spk), "subset": only_spk})
    return tasks


def transfer_tasks(df_full: pd.DataFrame, min_tokens_transfer: int) -> List[Dict]:
    tasks = []
    for vowel in sorted(df_full["Vogal"].astype(str).unique()):
        subset = df_full[df_full["Vogal"].astype(str) == vowel].reset_index(drop=True)
        if len(subset) >= min_tokens_transfer:
            tasks.append({"kind": "transfer", "transfer_vowel": vowel, "subset": subset})
    return tasks


def worker_run(
    task: Dict,
    params_template: Dict,
    bandwidth: float,
    coarse_resolution: int,
    fine_resolution: int,
    init_lambdas: Tuple[float, float],
    bounds_2d,
) -> Dict:
    subset = task["subset"]
    auto = task["kind"] == "transfer"
    try:
        res = fit_subset_fast(
            subset,
            params_template,
            bandwidth,
            coarse_resolution,
            fine_resolution,
            init_lambdas,
            bounds_2d,
            auto_parametrize=auto,
        )
        res["task_success"] = True
    except Exception as exc:
        res = {
            "task_success": False,
            "message": f"{type(exc).__name__}: {exc}",
            "n_tokens": int(len(subset)),
            "n_speakers": int(subset["Falante"].nunique()),
            "vowels": ",".join(sorted(subset["Vogal"].astype(str).unique())),
            "speakers": ",".join(map(str, sorted(subset["Falante"].unique()))),
        }
    for key, value in task.items():
        if key != "subset":
            res[key] = value
    return res


def summarize_series(series: pd.Series) -> Dict[str, float]:
    s = series.dropna()
    if len(s) == 0:
        return {"mean": np.nan, "sd": np.nan, "min": np.nan, "max": np.nan, "median": np.nan}
    return {
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
        "median": float(s.median()),
    }


def write_summary(
    path: Path,
    baseline: Dict,
    bootstrap_df: pd.DataFrame,
    split_df: pd.DataFrame,
    speaker_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    metadata: Dict,
) -> None:
    lines: List[str] = []

    lines.append("FAST ROBUSTNESS SUITE SUMMARY")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Methodological note")
    lines.append("-" * 90)
    lines.append(
        "These checks use the same model architecture as the paper, but implement KL and normalization "
        "on regular grids as discrete probability masses. This is a computational acceleration, not a "
        "change in the adopted theory."
    )
    lines.append("")
    lines.append("Timing configuration")
    lines.append("-" * 90)
    lines.append(f"coarse_resolution: {metadata['coarse_resolution']}")
    lines.append(f"fine_resolution: {metadata['fine_resolution']}")
    lines.append(f"bandwidth: {metadata['bandwidth']}")
    lines.append(f"workers: {metadata['workers']}")
    lines.append("")

    lines.append("1. Baseline fit on focal dataset")
    lines.append("-" * 90)
    for key in [
        "task_success", "success", "message", "n_tokens", "n_speakers", "vowels", "speakers",
        "baseline_kl", "fitted_kl", "improvement", "coarse_fitted_kl",
        "coarse_lambda_RA", "coarse_lambda_RP", "lambda_RA", "lambda_RP", "elapsed_seconds",
    ]:
        if key in baseline:
            lines.append(f"{key}: {baseline[key]}")
    lines.append("")

    def section(title: str, df: pd.DataFrame, extra_group: Optional[str] = None):
        lines.append(title)
        lines.append("-" * 90)
        if df.empty:
            lines.append("No results.")
            lines.append("")
            return

        ok = df[df["task_success"] == True].copy()
        lines.append(f"successful tasks: {len(ok)}/{len(df)}")
        if ok.empty:
            lines.append("")
            return

        for col in ["fitted_kl", "improvement", "lambda_RA", "lambda_RP", "elapsed_seconds"]:
            stats = summarize_series(ok[col])
            lines.append(
                f"{col}: mean={stats['mean']:.6f}, sd={stats['sd']:.6f}, "
                f"min={stats['min']:.6f}, max={stats['max']:.6f}"
            )

        if extra_group and extra_group in ok.columns:
            lines.append("")
            lines.append(f"By {extra_group}:")
            for val, part in ok.groupby(extra_group):
                stats = summarize_series(part["fitted_kl"])
                lines.append(
                    f"  {extra_group}={val}: fitted_kl mean={stats['mean']:.6f}, "
                    f"sd={stats['sd']:.6f}, n={len(part)}"
                )
        lines.append("")

    section("2. Bootstrap robustness", bootstrap_df)
    section("3. Split-half robustness", split_df, extra_group="split")
    section("4. Speaker robustness", speaker_df, extra_group="condition")
    section("5. Transfer checks", transfer_df, extra_group="transfer_vowel")

    lines.append("Interpretive note")
    lines.append("-" * 90)
    lines.append(
        "For the paper, the main quantities to report are the stability of fitted KL and of the "
        "two shape-relevant weights (lambda_RA and lambda_RP) across bootstrap, split-half, "
        "speaker holdout, and transfer tasks."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.txt-like file")
    parser.add_argument("--output_dir", default="robustness_fast_output")
    parser.add_argument("--vogais", default="e")
    parser.add_argument("--entrevistados", default="1,3,5")
    parser.add_argument("--alvo_F1", type=float, default=421)
    parser.add_argument("--alvo_F2", type=float, default=1887)
    parser.add_argument("--limiar_1", type=float, default=600)
    parser.add_argument("--limiar_2", type=float, default=345)
    parser.add_argument("--neutro_F1", type=float, default=610)
    parser.add_argument("--neutro_F2", type=float, default=1900)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--k_1", type=float, default=1.0)
    parser.add_argument("--k_2", type=float, default=7.0)
    parser.add_argument("--bandwidth", type=float, default=0.2)
    parser.add_argument("--coarse_resolution", type=int, default=80)
    parser.add_argument("--fine_resolution", type=int, default=160)
    parser.add_argument("--init_lambdas", default="1,1", help="lambda_RA,lambda_RP")
    parser.add_argument("--lambda_lower", type=float, default=1e-4)
    parser.add_argument("--lambda_upper", type=float, default=10.0)
    parser.add_argument("--n_boot", type=int, default=24)
    parser.add_argument("--n_splits", type=int, default=12)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--min_tokens_transfer", type=int, default=30)
    parser.add_argument("--skip_transfer", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_full = load_data(args.data)
    vowels = parse_str_list(args.vogais)
    speakers = parse_int_list(args.entrevistados)
    df_focus = subset_data(df_full, vowels, speakers)
    if len(df_focus) < 10:
        raise ValueError("Focal subset has fewer than 10 tokens; cannot run robustness checks.")

    init_vals = [float(x.strip()) for x in args.init_lambdas.split(",")]
    if len(init_vals) != 2:
        raise ValueError("--init_lambdas must contain exactly two comma-separated values: lambda_RA,lambda_RP")
    init_lambdas = (init_vals[0], init_vals[1])

    bounds_2d = (
        (args.lambda_lower, args.lambda_upper),
        (args.lambda_lower, args.lambda_upper),
    )

    params_template = {
        "alvo_F1": args.alvo_F1,
        "alvo_F2": args.alvo_F2,
        "limiar_1": args.limiar_1,
        "limiar_2": args.limiar_2,
        "neutro_F1": args.neutro_F1,
        "neutro_F2": args.neutro_F2,
        "L": args.L,
        "k_1": args.k_1,
        "k_2": args.k_2,
    }

    t0 = perf_counter()
    baseline = fit_subset_fast(
        df_focus,
        params_template,
        args.bandwidth,
        args.coarse_resolution,
        args.fine_resolution,
        init_lambdas,
        bounds_2d,
        auto_parametrize=False,
    )

    tasks = []
    tasks.extend(bootstrap_tasks(df_focus, args.n_boot, args.seed))
    tasks.extend(split_half_tasks(df_focus, args.n_splits, args.seed))
    tasks.extend(speaker_tasks(df_focus))
    if not args.skip_transfer:
        tasks.extend(transfer_tasks(df_full, args.min_tokens_transfer))

    bootstrap_rows = []
    split_rows = []
    speaker_rows = []
    transfer_rows = []

    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(
                    worker_run,
                    task,
                    params_template,
                    args.bandwidth,
                    args.coarse_resolution,
                    args.fine_resolution,
                    init_lambdas,
                    bounds_2d,
                )
                for task in tasks
            ]
            for fut in as_completed(futures):
                row = fut.result()
                kind = row.get("kind")
                if kind == "bootstrap":
                    bootstrap_rows.append(row)
                elif kind == "split_half":
                    split_rows.append(row)
                elif kind == "speaker_holdout":
                    speaker_rows.append(row)
                elif kind == "transfer":
                    transfer_rows.append(row)

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    split_df = pd.DataFrame(split_rows)
    speaker_df = pd.DataFrame(speaker_rows)
    transfer_df = pd.DataFrame(transfer_rows)

    bootstrap_df.to_csv(outdir / "bootstrap_results.csv", index=False)
    split_df.to_csv(outdir / "split_half_results.csv", index=False)
    speaker_df.to_csv(outdir / "speaker_holdout_results.csv", index=False)
    transfer_df.to_csv(outdir / "transfer_results.csv", index=False)

    total_elapsed = perf_counter() - t0
    metadata = {
        "data": os.path.abspath(args.data),
        "focus_vowels": vowels,
        "focus_speakers": speakers,
        "n_focus_tokens": int(len(df_focus)),
        "bandwidth": args.bandwidth,
        "coarse_resolution": args.coarse_resolution,
        "fine_resolution": args.fine_resolution,
        "init_lambdas": init_lambdas,
        "n_boot": args.n_boot,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "min_tokens_transfer": args.min_tokens_transfer,
        "skip_transfer": bool(args.skip_transfer),
        "workers": args.workers,
        "template_params": params_template,
        "baseline": baseline,
        "total_elapsed_seconds": total_elapsed,
    }

    (outdir / "robustness_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_summary(
        outdir / "robustness_summary.txt",
        baseline,
        bootstrap_df,
        split_df,
        speaker_df,
        transfer_df,
        metadata,
    )

    print(f"Wrote outputs to: {outdir.resolve()}")
    print(f"Total elapsed seconds: {total_elapsed:.3f}")


if __name__ == "__main__":
    main()
