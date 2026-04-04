#!/usr/bin/env python3
"""
link_choice_robustness_fast.py

Fast comparative robustness checks for link-function choice.

This script is a major rewrite of the earlier comparative script. It is designed to make
link-choice validation practical without changing the paper's adopted architecture.

What it tests
-------------
The same four transparent candidate links discussed in the paper:

    1. multiplication   V(x) = C(x) * fhat(x)
    2. subtraction      V(x) = C(x) - fhat(x)
    3. division         V(x) = C(x) / fhat(x)
    4. exponentialized  V(x) = exp(C(x) - fhat(x))

It evaluates them under the same downstream architecture:
- z-score preprocessing
- 2D KDE over F1/F2
- paired-logistic perceptual constraint
- articulatory constraint = target distance * relative effort
- normalized MaxEnt-style output density
- KL comparison to the empirical density

Major speedups
--------------
1. Precomputes all common subset-specific objects:
   - KDE grid
   - empirical probability mass p(x)
   - perceptual constraint values C_P
   - perceptual density values D_P
   - articulatory constraint values C_A
   - articulatory density values D_A
2. Optimizes only the shape-relevant weights (lambda_RA, lambda_RP).
   lambda_zero is omitted because numerical normalization removes its effect on shape.
3. Uses the analytic KL objective on a regular grid:
      KL(p||q) = const + lambda_RA E_p[V_A] + lambda_RP E_p[V_P] + log Z(lambda)
   with analytic gradient:
      d/dlambda_i = E_p[V_i] - E_q[V_i]
4. Uses sum-based normalization on discrete probability masses instead of repeated quadrature.
5. Uses coarse-to-fine fitting.
6. Parallelizes across bootstrap / bandwidth / initialization tasks.

Scientific note
---------------
The main approximation is computational: continuous densities are represented on regular grids
and normalized as discrete probability masses. For robustness and comparative checks, this is
scientifically appropriate and dramatically faster.
"""

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
from scipy.stats import gaussian_kde, spearmanr
from sklearn.preprocessing import StandardScaler


EPS = 1e-12
DENSITY_EPS = 1e-12
NUMERIC_EPS = 1e-6
MAX_V = 1e9  # purely numerical cap to prevent pathological floating overflow


# ------------------------------
# Parsing helpers
# ------------------------------

def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_pairs(text: str) -> List[Tuple[float, float]]:
    """
    Example:
        "1,1;0.5,0.5;2,1;1,2"
    """
    out = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = [float(x.strip()) for x in chunk.split(",")]
        if len(vals) != 2:
            raise ValueError(f"Invalid init pair: {chunk}")
        out.append((vals[0], vals[1]))
    return out


# ------------------------------
# Data I/O
# ------------------------------

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


def subset_data(df: pd.DataFrame, vowels: Sequence[str], speakers: Sequence[int]) -> pd.DataFrame:
    out = df[df["Vogal"].isin(vowels) & df["Falante"].isin(speakers)].copy()
    return out[["F1", "F2", "Vogal", "Falante"]].reset_index(drop=True)


# ------------------------------
# Parameters and scaling
# ------------------------------

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


# ------------------------------
# Model pieces from app.py
# ------------------------------

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


# ------------------------------
# Precompute common fields
# ------------------------------

@dataclass
class CommonFields:
    p_mass: np.ndarray
    C_P: np.ndarray
    D_P: np.ndarray
    C_A: np.ndarray
    D_A: np.ndarray
    n_grid: int


def precompute_common(
    df_subset: pd.DataFrame,
    params_template: Dict,
    bandwidth: float,
    resolution: int,
) -> CommonFields:
    if len(df_subset) < 10:
        raise ValueError("Subset too small for stable KDE-based fitting (<10 tokens).")

    params_ref = make_reference_params(df_subset, params_template)
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
    marginal_f1 = np.clip(marginal_f1, DENSITY_EPS, None)
    marginal_f1 /= marginal_f1.sum()

    C_P = perceptual_constraint(F1, z_params).astype(np.float64)
    D_P = marginal_f1.astype(np.float64)

    C_A = articulatory_constraint(F1_mesh, F2_mesh, z_params).ravel().astype(np.float64)
    D_A = np.clip(p_mass, DENSITY_EPS, None).astype(np.float64)

    return CommonFields(
        p_mass=p_mass,
        C_P=C_P,
        D_P=D_P,
        C_A=C_A,
        D_A=D_A,
        n_grid=resolution * resolution,
    )


# ------------------------------
# Link functions and diagnostics
# ------------------------------

def apply_link(name: str, C, D):
    D_safe = np.clip(D, DENSITY_EPS, None)
    if name == "multiplication":
        V = C * D
    elif name == "subtraction":
        V = C - D
    elif name == "division":
        V = C / D_safe
    elif name == "exponentialized":
        V = np.exp(np.clip(C - D, -700.0, 700.0))
    else:
        raise ValueError(f"Unknown link: {name}")

    V = np.asarray(V, dtype=np.float64)
    return np.clip(V, -MAX_V, MAX_V)


def safe_spearman(x, y):
    if len(x) < 5:
        return np.nan
    if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return np.nan
    rho, _ = spearmanr(x, y)
    return float(rho) if np.isfinite(rho) else np.nan


def stratified_rho(primary, secondary, response, n_bins=8):
    primary = np.asarray(primary).ravel()
    secondary = np.asarray(secondary).ravel()
    response = np.asarray(response).ravel()
    try:
        qs = np.quantile(primary, np.linspace(0, 1, n_bins + 1))
    except Exception:
        return np.nan

    rhos = []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        if i == n_bins - 1:
            mask = (primary >= lo) & (primary <= hi)
        else:
            mask = (primary >= lo) & (primary < hi)
        if mask.sum() < 10:
            continue
        rho = safe_spearman(secondary[mask], response[mask])
        if np.isfinite(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else np.nan


def diagnostics_for_fields(C, D, V, label_prefix: str) -> Dict[str, float]:
    C = np.asarray(C).ravel()
    D = np.asarray(D).ravel()
    V = np.asarray(V).ravel()

    finite_mask = np.isfinite(V)
    V_f = V[finite_mask]
    C_f = C[finite_mask]
    D_f = D[finite_mask]

    if len(V_f) == 0:
        return {
            f"{label_prefix}_finite_frac": 0.0,
            f"{label_prefix}_negative_frac": np.nan,
            f"{label_prefix}_nonnegative_frac": np.nan,
            f"{label_prefix}_dynamic_range_q95_q05": np.nan,
            f"{label_prefix}_attenuation_rho": np.nan,
            f"{label_prefix}_constraint_monotonicity_rho": np.nan,
            f"{label_prefix}_low_density_tail_ratio": np.nan,
        }

    q05, q95 = np.quantile(V_f, [0.05, 0.95])
    low_density_cut = np.quantile(D_f, 0.05)
    low_mask = D_f <= low_density_cut
    if low_mask.sum() >= 5:
        q99_low = np.quantile(V_f[low_mask], 0.99)
        tail_ratio = float(q99_low / max(q95, EPS))
    else:
        tail_ratio = np.nan

    attenuation_rho = stratified_rho(C_f, D_f, V_f, n_bins=8)
    monotonicity_rho = stratified_rho(D_f, C_f, V_f, n_bins=8)

    return {
        f"{label_prefix}_finite_frac": float(np.mean(finite_mask)),
        f"{label_prefix}_negative_frac": float(np.mean(V_f < 0)),
        f"{label_prefix}_nonnegative_frac": float(np.mean(V_f >= 0)),
        f"{label_prefix}_dynamic_range_q95_q05": float(q95 - q05),
        f"{label_prefix}_attenuation_rho": attenuation_rho,
        f"{label_prefix}_constraint_monotonicity_rho": monotonicity_rho,
        f"{label_prefix}_low_density_tail_ratio": tail_ratio,
    }


@dataclass
class LinkModel:
    p_mass: np.ndarray
    V_A: np.ndarray
    V_P: np.ndarray
    E_p_VA: float
    E_p_VP: float
    diagnostics: Dict[str, float]


def build_link_model(common: CommonFields, link_name: str) -> LinkModel:
    V_P_1d = apply_link(link_name, common.C_P, common.D_P)
    res = int(round(np.sqrt(common.n_grid)))
    V_P = np.tile(V_P_1d, (res, 1)).ravel().astype(np.float64)

    V_A = apply_link(link_name, common.C_A, common.D_A).astype(np.float64)

    E_p_VA = float(np.dot(common.p_mass, V_A))
    E_p_VP = float(np.dot(common.p_mass, V_P))

    diag = {}
    diag.update(diagnostics_for_fields(common.C_P, common.D_P, V_P_1d, "P"))
    diag.update(diagnostics_for_fields(common.C_A, common.D_A, V_A, "A"))

    for metric in [
        "finite_frac", "negative_frac", "nonnegative_frac",
        "dynamic_range_q95_q05", "attenuation_rho",
        "constraint_monotonicity_rho", "low_density_tail_ratio"
    ]:
        p_key = f"P_{metric}"
        a_key = f"A_{metric}"
        vals = [diag[p_key], diag[a_key]]
        vals = [v for v in vals if pd.notna(v)]
        diag[f"avg_{metric}"] = float(np.mean(vals)) if vals else np.nan

    return LinkModel(
        p_mass=common.p_mass,
        V_A=V_A,
        V_P=V_P,
        E_p_VA=E_p_VA,
        E_p_VP=E_p_VP,
        diagnostics=diag,
    )


# ------------------------------
# Fast KL optimization per link
# ------------------------------

def objective_and_grad(lambdas: np.ndarray, model: LinkModel) -> Tuple[float, np.ndarray]:
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


def kl_from_lambdas(lambdas: Sequence[float], model: LinkModel) -> float:
    lambda_RA = float(lambdas[0])
    lambda_RP = float(lambdas[1])
    logits = -lambda_RA * model.V_A - lambda_RP * model.V_P
    logZ = logsumexp(logits)
    q_mass = np.exp(logits - logZ)
    return float(np.sum(model.p_mass * (np.log(np.clip(model.p_mass, EPS, None)) - np.log(np.clip(q_mass, EPS, None)))))


def fit_weights_for_link(
    model: LinkModel,
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

    out = {
        "success": bool(result.success),
        "message": str(result.message),
        "lambda_RA": float(fitted[0]),
        "lambda_RP": float(fitted[1]),
        "baseline_kl": baseline_kl,
        "fitted_kl": fitted_kl,
        "improvement": baseline_kl - fitted_kl,
        "nit": int(getattr(result, "nit", -1)),
    }
    out.update(model.diagnostics)
    return out


def fit_all_links_fast(
    df_subset: pd.DataFrame,
    params_template: Dict,
    bandwidth: float,
    coarse_resolution: int,
    fine_resolution: int,
    init_lambdas: Tuple[float, float],
    bounds_2d,
) -> pd.DataFrame:
    t0 = perf_counter()

    coarse_common = precompute_common(df_subset, params_template, bandwidth, coarse_resolution)
    fine_common = precompute_common(df_subset, params_template, bandwidth, fine_resolution)

    rows = []
    for link_name in ["multiplication", "subtraction", "division", "exponentialized"]:
        try:
            coarse_model = build_link_model(coarse_common, link_name)
            coarse_fit = fit_weights_for_link(coarse_model, init_lambdas, bounds_2d)

            fine_model = build_link_model(fine_common, link_name)
            fine_fit = fit_weights_for_link(
                fine_model,
                (coarse_fit["lambda_RA"], coarse_fit["lambda_RP"]),
                bounds_2d,
            )

            row = dict(fine_fit)
            row["link"] = link_name
            row["coarse_lambda_RA"] = coarse_fit["lambda_RA"]
            row["coarse_lambda_RP"] = coarse_fit["lambda_RP"]
            row["coarse_fitted_kl"] = coarse_fit["fitted_kl"]
        except Exception as exc:
            row = {
                "link": link_name,
                "success": False,
                "message": f"{type(exc).__name__}: {exc}",
                "lambda_RA": np.nan,
                "lambda_RP": np.nan,
                "baseline_kl": np.nan,
                "fitted_kl": np.nan,
                "improvement": np.nan,
                "nit": np.nan,
            }

        row["n_tokens"] = int(len(df_subset))
        row["n_speakers"] = int(df_subset["Falante"].nunique())
        row["vowels"] = ",".join(sorted(df_subset["Vogal"].astype(str).unique()))
        row["speakers"] = ",".join(map(str, sorted(df_subset["Falante"].unique())))
        row["elapsed_seconds_total_task"] = float(perf_counter() - t0)
        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------
# Task generation
# ------------------------------

def bootstrap_tasks(df_focus: pd.DataFrame, n_boot: int, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    n = len(df_focus)
    tasks = []
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = df_focus.iloc[idx].reset_index(drop=True)
        tasks.append({"kind": "bootstrap", "iteration": i + 1, "subset": sample})
    return tasks


def bandwidth_tasks(df_focus: pd.DataFrame, bandwidths: Sequence[float]) -> List[Dict]:
    return [{"kind": "bandwidth", "bandwidth_value": float(bw), "subset": df_focus.copy()} for bw in bandwidths]


def init_tasks(df_focus: pd.DataFrame, init_sets: Sequence[Tuple[float, float]]) -> List[Dict]:
    return [
        {
            "kind": "initialization",
            "init_lambda_RA": float(pair[0]),
            "init_lambda_RP": float(pair[1]),
            "init_pair": pair,
            "subset": df_focus.copy(),
        }
        for pair in init_sets
    ]


# ------------------------------
# Parallel worker
# ------------------------------

def worker_run(
    task: Dict,
    params_template: Dict,
    default_bandwidth: float,
    coarse_resolution: int,
    fine_resolution: int,
    default_init_lambdas: Tuple[float, float],
    bounds_2d,
) -> pd.DataFrame:
    subset = task["subset"]
    bandwidth = float(task.get("bandwidth_value", default_bandwidth))
    init_lambdas = task.get("init_pair", default_init_lambdas)

    df_res = fit_all_links_fast(
        subset,
        params_template,
        bandwidth,
        coarse_resolution,
        fine_resolution,
        init_lambdas,
        bounds_2d,
    )

    for key, value in task.items():
        if key in {"subset", "init_pair"}:
            continue
        df_res[key] = value

    return df_res


# ------------------------------
# Reporting
# ------------------------------

def mean_sd(series):
    s = series.dropna()
    if len(s) == 0:
        return (np.nan, np.nan)
    return (float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0)


def winner_counts(df: pd.DataFrame, grouping_cols: List[str]) -> Dict[str, int]:
    counts = {k: 0 for k in ["multiplication", "subtraction", "division", "exponentialized"]}
    if df.empty:
        return counts
    ok = df[df["success"] == True].copy()
    if ok.empty:
        return counts
    for _, part in ok.groupby(grouping_cols):
        best = part.loc[part["fitted_kl"].idxmin(), "link"]
        counts[best] += 1
    return counts


def summarize_block(lines: List[str], title: str, df: pd.DataFrame):
    lines.append(title)
    lines.append("-" * 90)
    if df.empty:
        lines.append("No results.")
        lines.append("")
        return

    for link in ["multiplication", "subtraction", "division", "exponentialized"]:
        part = df[df["link"] == link].copy()
        ok = part[part["success"] == True].copy()
        lines.append(f"{link}: {len(ok)}/{len(part)} successful fits")
        if not ok.empty:
            for col in [
                "fitted_kl",
                "improvement",
                "avg_nonnegative_frac",
                "avg_attenuation_rho",
                "avg_constraint_monotonicity_rho",
                "avg_low_density_tail_ratio",
                "avg_dynamic_range_q95_q05",
            ]:
                m, s = mean_sd(ok[col])
                lines.append(f"  {col}: mean={m:.6f}, sd={s:.6f}")
        lines.append("")
    lines.append("")


def write_summary(
    path: Path,
    baseline_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    bandwidth_df: pd.DataFrame,
    init_df: pd.DataFrame,
    metadata: Dict,
):
    lines = []
    lines.append("FAST LINK-CHOICE ROBUSTNESS SUMMARY")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Methodological note")
    lines.append("-" * 90)
    lines.append(
        "These checks use the paper's architecture but compute the empirical density and MaxEnt field "
        "as discrete probability masses on regular grids for speed. This is a computational "
        "acceleration, not a change in the adopted theory."
    )
    lines.append("")
    lines.append("Interpretive guide")
    lines.append("-" * 90)
    lines.append("Preferred behavior under the paper's desiderata is:")
    lines.append("  - high nonnegative fraction")
    lines.append("  - negative attenuation rho (within constraint strata, higher density lowers violations)")
    lines.append("  - positive constraint-monotonicity rho (within density strata, stronger constraint raises violations)")
    lines.append("  - moderate low-density tail ratio (avoids instability explosions)")
    lines.append("  - strong downstream compatibility (low fitted KL, successful optimization)")
    lines.append("")
    lines.append("Timing configuration")
    lines.append("-" * 90)
    lines.append(f"coarse_resolution: {metadata['coarse_resolution']}")
    lines.append(f"fine_resolution: {metadata['fine_resolution']}")
    lines.append(f"default_bandwidth: {metadata['default_bandwidth']}")
    lines.append(f"workers: {metadata['workers']}")
    lines.append("")

    summarize_block(lines, "1. Baseline comparison", baseline_df)
    summarize_block(lines, "2. Bootstrap robustness", bootstrap_df)
    summarize_block(lines, "3. Bandwidth sensitivity", bandwidth_df)
    summarize_block(lines, "4. Initialization sensitivity", init_df)

    lines.append("5. Winner counts by fitted KL")
    lines.append("-" * 90)

    baseline_counts = winner_counts(baseline_df.assign(case="baseline"), ["case"])
    lines.append("Baseline winner count")
    for link, count in baseline_counts.items():
        lines.append(f"  {link}: {count}")
    lines.append("")

    if not bootstrap_df.empty:
        boot_counts = winner_counts(bootstrap_df, ["iteration"])
        lines.append("Bootstrap winner counts")
        for link, count in boot_counts.items():
            lines.append(f"  {link}: {count}")
        lines.append("")

    if not bandwidth_df.empty:
        bw_counts = winner_counts(bandwidth_df, ["bandwidth_value"])
        lines.append("Bandwidth winner counts")
        for link, count in bw_counts.items():
            lines.append(f"  {link}: {count}")
        lines.append("")

    if not init_df.empty:
        init_counts = winner_counts(init_df, ["init_lambda_RA", "init_lambda_RP"])
        lines.append("Initialization winner counts")
        for link, count in init_counts.items():
            lines.append(f"  {link}: {count}")
        lines.append("")

    lines.append("Interpretive note")
    lines.append("-" * 90)
    lines.append(
        "If the exponentialized link combines the strongest positivity profile, the expected "
        "attenuation / monotonicity behavior, acceptable low-density stability, and the most "
        "consistent fitted-KL wins across these checks, that provides a robust empirical supplement "
        "to the paper's criterion-based defense of the link choice."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.txt-like file")
    parser.add_argument("--output_dir", default="link_choice_fast_output")

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
    parser.add_argument("--bandwidths", default="0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--coarse_resolution", type=int, default=80)
    parser.add_argument("--fine_resolution", type=int, default=160)

    parser.add_argument("--init_lambdas", default="1,1", help="lambda_RA,lambda_RP")
    parser.add_argument("--init_sets", default="1,1;0.5,0.5;2,1;1,2")
    parser.add_argument("--lambda_lower", type=float, default=1e-4)
    parser.add_argument("--lambda_upper", type=float, default=10.0)

    parser.add_argument("--n_boot", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))

    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    vowels = parse_str_list(args.vogais)
    speakers = parse_int_list(args.entrevistados)
    df_focus = subset_data(df, vowels, speakers)
    if len(df_focus) < 10:
        raise ValueError("Focal subset has fewer than 10 tokens; cannot run link-choice checks.")

    init_vals = [float(x.strip()) for x in args.init_lambdas.split(",")]
    if len(init_vals) != 2:
        raise ValueError("--init_lambdas must have exactly two comma-separated values: lambda_RA,lambda_RP")
    init_lambdas = (init_vals[0], init_vals[1])

    init_sets = parse_float_pairs(args.init_sets)
    bandwidths = parse_float_list(args.bandwidths)

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

    baseline_df = fit_all_links_fast(
        df_focus,
        params_template,
        args.bandwidth,
        args.coarse_resolution,
        args.fine_resolution,
        init_lambdas,
        bounds_2d,
    )

    tasks = []
    tasks.extend(bootstrap_tasks(df_focus, args.n_boot, args.seed))
    tasks.extend(bandwidth_tasks(df_focus, bandwidths))
    tasks.extend(init_tasks(df_focus, init_sets))

    bootstrap_frames = []
    bandwidth_frames = []
    init_frames = []

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
                df_res = fut.result()
                kind = df_res["kind"].iloc[0] if not df_res.empty else None
                if kind == "bootstrap":
                    bootstrap_frames.append(df_res)
                elif kind == "bandwidth":
                    bandwidth_frames.append(df_res)
                elif kind == "initialization":
                    init_frames.append(df_res)

    bootstrap_df = pd.concat(bootstrap_frames, ignore_index=True) if bootstrap_frames else pd.DataFrame()
    bandwidth_df = pd.concat(bandwidth_frames, ignore_index=True) if bandwidth_frames else pd.DataFrame()
    init_df = pd.concat(init_frames, ignore_index=True) if init_frames else pd.DataFrame()

    baseline_df.to_csv(outdir / "baseline_link_results.csv", index=False)
    bootstrap_df.to_csv(outdir / "bootstrap_link_results.csv", index=False)
    bandwidth_df.to_csv(outdir / "bandwidth_link_results.csv", index=False)
    init_df.to_csv(outdir / "init_link_results.csv", index=False)

    total_elapsed = perf_counter() - t0
    metadata = {
        "data": os.path.abspath(args.data),
        "focus_vowels": vowels,
        "focus_speakers": speakers,
        "n_focus_tokens": int(len(df_focus)),
        "default_bandwidth": args.bandwidth,
        "bandwidths": bandwidths,
        "coarse_resolution": args.coarse_resolution,
        "fine_resolution": args.fine_resolution,
        "init_lambdas": init_lambdas,
        "init_sets": init_sets,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "workers": args.workers,
        "template_params": params_template,
        "total_elapsed_seconds": total_elapsed,
    }

    (outdir / "link_choice_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_summary(
        outdir / "link_choice_summary.txt",
        baseline_df,
        bootstrap_df,
        bandwidth_df,
        init_df,
        metadata,
    )

    print(f"Wrote outputs to: {outdir.resolve()}")
    print(f"Total elapsed seconds: {total_elapsed:.3f}")


if __name__ == "__main__":
    main()
