#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import Bounds, minimize
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler


EPS = 1e-10
DIVISION_EPS = 1e-6
DEFAULT_RESOLUTION = 250
DEFAULT_MAXITER = 200


@dataclass
class NormalizedParams:
    resolucao: int
    alvo_F1: float
    alvo_F2: float
    limiar_1: float
    limiar_2: float
    neutro_F1: float
    neutro_F2: float
    L: float
    k_1: float
    k_2: float
    a_F1: float
    b_F1: float
    a_F2: float
    b_F2: float
    min_F1: float
    max_F1: float
    min_F2: float
    max_F2: float


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation suite for the continuous-candidate model")
    parser.add_argument("--data", required=True, help="Path to the space-separated data file")
    parser.add_argument("--output_dir", default="validation_output", help="Directory for report files")
    parser.add_argument("--vogais", default="e", help="Comma-separated vowels to include")
    parser.add_argument("--entrevistados", default="1,3,5", help="Comma-separated speaker IDs")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--bandwidths", default="0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--init_sets", default="1,1,1;0.5,0.5,0.5;2,1,1;1,2,1;1,1,2;2,0.5,0.5")

    # Original-scale model settings from app.py / paper
    parser.add_argument("--alvo_F1", type=float, default=421)
    parser.add_argument("--alvo_F2", type=float, default=1887)
    parser.add_argument("--limiar_1", type=float, default=600)
    parser.add_argument("--limiar_2", type=float, default=345)
    parser.add_argument("--neutro_F1", type=float, default=610)
    parser.add_argument("--neutro_F2", type=float, default=1900)
    parser.add_argument("--L", type=float, default=1)
    parser.add_argument("--k_1", type=float, default=1)
    parser.add_argument("--k_2", type=float, default=7)

    return parser.parse_args()


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(path: str, vogais: Iterable[str], entrevistados: Iterable[int]) -> pd.DataFrame:
    df = pd.read_csv(path, sep=" ", quotechar='"', header=0)
    df = df[(df["F1"] != "NA") & (df["F2"] != "NA")].copy()
    df["F1"] = pd.to_numeric(df["F1"])
    df["F2"] = pd.to_numeric(df["F2"])
    subset = df[df["Vogal"].isin(vogais) & df["Falante"].isin(entrevistados)].copy()
    if subset.empty:
        raise ValueError("No rows remain after filtering by vowel and speaker.")
    return subset[["F1", "F2"]]


def build_scaler_and_kde(data: pd.DataFrame, bandwidth: float) -> Tuple[StandardScaler, gaussian_kde]:
    scaler = StandardScaler()
    values = data[["F1", "F2"]].to_numpy(dtype=float)
    scaled = scaler.fit_transform(values)
    kde = gaussian_kde(scaled.T, bw_method=bandwidth)
    return scaler, kde


def normalize_params(args: argparse.Namespace, data: pd.DataFrame, scaler: StandardScaler) -> NormalizedParams:
    f1_min = float(data["F1"].min())
    f1_max = float(data["F1"].max())
    f2_min = float(data["F2"].min())
    f2_max = float(data["F2"].max())

    return NormalizedParams(
        resolucao=args.resolution,
        alvo_F1=float(scaler.transform([[args.alvo_F1, 0]])[0][0]),
        alvo_F2=float(scaler.transform([[0, args.alvo_F2]])[0][1]),
        limiar_1=float(scaler.transform([[args.limiar_1, 0]])[0][0]),
        limiar_2=float(scaler.transform([[args.limiar_2, 0]])[0][0]),
        neutro_F1=float(scaler.transform([[args.neutro_F1, 0]])[0][0]),
        neutro_F2=float(scaler.transform([[0, args.neutro_F2]])[0][1]),
        L=float(args.L),
        k_1=float(args.k_1),
        k_2=float(args.k_2),
        a_F1=float(scaler.transform([[f1_min, 0]])[0][0]),
        b_F1=float(scaler.transform([[f1_max, 0]])[0][0]),
        a_F2=float(scaler.transform([[0, f2_min]])[0][1]),
        b_F2=float(scaler.transform([[0, f2_max]])[0][1]),
        min_F1=float(scaler.transform([[f1_min, 0]])[0][0]),
        max_F1=float(scaler.transform([[f1_max, 0]])[0][0]),
        min_F2=float(scaler.transform([[0, f2_min]])[0][1]),
        max_F2=float(scaler.transform([[0, f2_max]])[0][1]),
    )


def make_grid(params: NormalizedParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    grid_f1 = np.linspace(params.min_F1, params.max_F1, params.resolucao)
    grid_f2 = np.linspace(params.min_F2, params.max_F2, params.resolucao)
    mesh_f1, mesh_f2 = np.meshgrid(grid_f1, grid_f2, indexing="xy")
    points = np.vstack([mesh_f1.ravel(), mesh_f2.ravel()])
    dx = abs(grid_f1[1] - grid_f1[0]) if len(grid_f1) > 1 else 1.0
    dy = abs(grid_f2[1] - grid_f2[0]) if len(grid_f2) > 1 else 1.0
    return grid_f1, grid_f2, mesh_f1, mesh_f2, dx, dy


def compute_kde_grid(kde: gaussian_kde, mesh_f1: np.ndarray, mesh_f2: np.ndarray) -> np.ndarray:
    points = np.vstack([mesh_f1.ravel(), mesh_f2.ravel()])
    return kde(points).reshape(mesh_f1.shape)


def create_f1_marginal(kde_grid: np.ndarray, grid_f1: np.ndarray, grid_f2: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    dy = abs(grid_f2[1] - grid_f2[0]) if len(grid_f2) > 1 else 1.0
    dx = abs(grid_f1[1] - grid_f1[0]) if len(grid_f1) > 1 else 1.0
    # Sum over F2 dimension to obtain the F1 marginal on the bounded analysis domain.
    marginal = np.sum(kde_grid, axis=0) * dy
    integral = np.sum(marginal) * dx
    if integral <= 0:
        raise ValueError("Marginal integral is non-positive.")
    marginal = marginal / integral
    return interp1d(grid_f1, marginal, kind="cubic", fill_value="extrapolate")


def perceptual_constraint(f1: np.ndarray, params: NormalizedParams) -> np.ndarray:
    product = (
        params.L ** 2
        / ((1 + np.exp(params.k_1 * (f1 - params.limiar_1))) * (1 + np.exp(-params.k_2 * (f1 - params.limiar_2))))
    )
    return params.L - product


def articulatory_constraint(f1: np.ndarray, f2: np.ndarray, params: NormalizedParams) -> np.ndarray:
    effort_target = np.sqrt((params.alvo_F1 - params.neutro_F1) ** 2 + (params.alvo_F2 - params.neutro_F2) ** 2)
    effort_production = np.sqrt((f1 - params.neutro_F1) ** 2 + (f2 - params.neutro_F2) ** 2)
    distance_target = np.sqrt((f1 - params.alvo_F1) ** 2 + (f2 - params.alvo_F2) ** 2)
    effort_ratio = (effort_production + 1e-6) / (effort_target + 1e-6)
    return distance_target * effort_ratio


def link_multiplication(c: np.ndarray, f: np.ndarray) -> np.ndarray:
    return c * f


def link_subtraction(c: np.ndarray, f: np.ndarray) -> np.ndarray:
    return c - f


def link_division(c: np.ndarray, f: np.ndarray) -> np.ndarray:
    return c / np.maximum(f, DIVISION_EPS)


def link_exponential(c: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.exp(c - f)


LINK_FUNCTIONS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "multiplication": link_multiplication,
    "subtraction": link_subtraction,
    "division_eps": link_division,
    "exponential": link_exponential,
}


def normalize_density(values: np.ndarray, dx: float, dy: float) -> np.ndarray:
    integral = np.trapz(np.trapz(values, dx=dx, axis=1), dx=dy)
    if integral <= 0 or not np.isfinite(integral):
        raise ValueError(f"Cannot normalize density: integral={integral}")
    return values / integral


def kl_divergence(p: np.ndarray, q: np.ndarray, dx: float, dy: float) -> float:
    p = normalize_density(np.clip(p, EPS, None), dx, dy)
    q = normalize_density(np.clip(q, EPS, None), dx, dy)
    integrand = p * (np.log(p) - np.log(q))
    return float(np.trapz(np.trapz(integrand, dx=dx, axis=1), dx=dy))


def build_violation_surfaces(
    params: NormalizedParams,
    kde_grid: np.ndarray,
    marginal_f1: Callable[[np.ndarray], np.ndarray],
    mesh_f1: np.ndarray,
    mesh_f2: np.ndarray,
    link_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    link = LINK_FUNCTIONS[link_name]
    cp = perceptual_constraint(mesh_f1, params)
    ca = articulatory_constraint(mesh_f1, mesh_f2, params)
    f1_density = marginal_f1(mesh_f1)
    vp = link(cp, f1_density)
    va = link(ca, kde_grid)
    return vp, va


def optimize_weights(
    params: NormalizedParams,
    kde_grid: np.ndarray,
    vp: np.ndarray,
    va: np.ndarray,
    dx: float,
    dy: float,
    initial: Tuple[float, float, float],
    maxiter: int,
) -> Dict[str, float]:
    target = normalize_density(np.clip(kde_grid, EPS, None), dx, dy)

    def objective(lambdas: np.ndarray) -> float:
        lambda_zero, lambda_a, lambda_p = lambdas
        raw = np.exp(-1.0 - lambda_zero - lambda_a * va - lambda_p * vp)
        try:
            model = normalize_density(np.clip(raw, EPS, None), dx, dy)
        except ValueError:
            return 1e12
        return kl_divergence(target, model, dx, dy)

    bounds = Bounds([1e-8, 1e-8, 1e-8], [np.inf, np.inf, np.inf])
    result = minimize(objective, np.array(initial, dtype=float), method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
    lambda_zero, lambda_a, lambda_p = result.x
    raw = np.exp(-1.0 - lambda_zero - lambda_a * va - lambda_p * vp)
    model = normalize_density(np.clip(raw, EPS, None), dx, dy)
    kl = kl_divergence(target, model, dx, dy)
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "lambda_zero": float(lambda_zero),
        "lambda_A": float(lambda_a),
        "lambda_P": float(lambda_p),
        "kl": float(kl),
        "iterations": int(getattr(result, "nit", -1)),
    }


def surface_diagnostics(surface: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(surface)
    negative = np.sum(surface < 0)
    zeros = np.sum(np.isclose(surface, 0.0))
    return {
        "min": float(np.nanmin(surface)),
        "max": float(np.nanmax(surface)),
        "mean": float(np.nanmean(surface)),
        "std": float(np.nanstd(surface)),
        "finite_share": float(np.mean(finite)),
        "negative_share": float(negative / surface.size),
        "zero_share": float(zeros / surface.size),
    }


def run_link_comparison(
    params: NormalizedParams,
    kde_grid: np.ndarray,
    marginal_f1: Callable[[np.ndarray], np.ndarray],
    mesh_f1: np.ndarray,
    mesh_f2: np.ndarray,
    dx: float,
    dy: float,
    maxiter: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for link_name in LINK_FUNCTIONS:
        vp, va = build_violation_surfaces(params, kde_grid, marginal_f1, mesh_f1, mesh_f2, link_name)
        diag_vp = surface_diagnostics(vp)
        diag_va = surface_diagnostics(va)
        fit = optimize_weights(params, kde_grid, vp, va, dx, dy, initial=(1.0, 1.0, 1.0), maxiter=maxiter)
        row = {
            "link": link_name,
            "success": fit["success"],
            "kl": fit["kl"],
            "lambda_zero": fit["lambda_zero"],
            "lambda_A": fit["lambda_A"],
            "lambda_P": fit["lambda_P"],
            "vp_min": diag_vp["min"],
            "vp_max": diag_vp["max"],
            "vp_negative_share": diag_vp["negative_share"],
            "va_min": diag_va["min"],
            "va_max": diag_va["max"],
            "va_negative_share": diag_va["negative_share"],
            "message": fit["message"],
        }
        rows.append(row)
    return rows


def run_bandwidth_sensitivity(
    data: pd.DataFrame,
    args: argparse.Namespace,
    bandwidths: List[float],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for bw in bandwidths:
        scaler, kde = build_scaler_and_kde(data, bw)
        params = normalize_params(args, data, scaler)
        grid_f1, grid_f2, mesh_f1, mesh_f2, dx, dy = make_grid(params)
        kde_grid = compute_kde_grid(kde, mesh_f1, mesh_f2)
        marginal_f1 = create_f1_marginal(kde_grid, grid_f1, grid_f2)
        vp, va = build_violation_surfaces(params, kde_grid, marginal_f1, mesh_f1, mesh_f2, "exponential")
        fit = optimize_weights(params, kde_grid, vp, va, dx, dy, initial=(1.0, 1.0, 1.0), maxiter=args.maxiter)
        rows.append({
            "bandwidth": bw,
            "kl": fit["kl"],
            "lambda_zero": fit["lambda_zero"],
            "lambda_A": fit["lambda_A"],
            "lambda_P": fit["lambda_P"],
            "success": fit["success"],
            "message": fit["message"],
        })
    return rows


def parse_init_sets(raw: str) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for chunk in raw.split(";"):
        pieces = [float(x.strip()) for x in chunk.split(",") if x.strip()]
        if len(pieces) != 3:
            raise ValueError(f"Invalid initialization triple: {chunk!r}")
        out.append((pieces[0], pieces[1], pieces[2]))
    return out


def run_initialization_sensitivity(
    params: NormalizedParams,
    kde_grid: np.ndarray,
    marginal_f1: Callable[[np.ndarray], np.ndarray],
    mesh_f1: np.ndarray,
    mesh_f2: np.ndarray,
    dx: float,
    dy: float,
    init_sets: List[Tuple[float, float, float]],
    maxiter: int,
) -> List[Dict[str, float]]:
    vp, va = build_violation_surfaces(params, kde_grid, marginal_f1, mesh_f1, mesh_f2, "exponential")
    rows: List[Dict[str, float]] = []
    for init in init_sets:
        fit = optimize_weights(params, kde_grid, vp, va, dx, dy, initial=init, maxiter=maxiter)
        rows.append({
            "init_lambda_zero": init[0],
            "init_lambda_A": init[1],
            "init_lambda_P": init[2],
            "kl": fit["kl"],
            "lambda_zero": fit["lambda_zero"],
            "lambda_A": fit["lambda_A"],
            "lambda_P": fit["lambda_P"],
            "success": fit["success"],
            "message": fit["message"],
        })
    return rows


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_link_rows(rows: List[Dict[str, float]]) -> str:
    lines = ["Link-function comparison:"]
    lines.append("  link | KL | negative VP share | negative VA share | success")
    for row in rows:
        lines.append(
            f"  {row['link']} | {row['kl']:.6f} | {row['vp_negative_share']:.4f} | {row['va_negative_share']:.4f} | {row['success']}"
        )
    best = min(rows, key=lambda x: x["kl"])
    lines.append(f"Best KL under the shared protocol: {best['link']} ({best['kl']:.6f}).")
    return "\n".join(lines)


def summarize_bandwidth_rows(rows: List[Dict[str, float]]) -> str:
    kls = [row["kl"] for row in rows]
    lines = ["Bandwidth sensitivity (exponential link):"]
    for row in rows:
        lines.append(
            f"  bw={row['bandwidth']:.2f} | KL={row['kl']:.6f} | lambda_P={row['lambda_P']:.6f} | lambda_A={row['lambda_A']:.6f}"
        )
    lines.append(
        f"KL range across bandwidths: min={min(kls):.6f}, max={max(kls):.6f}, mean={np.mean(kls):.6f}, sd={np.std(kls):.6f}."
    )
    return "\n".join(lines)


def summarize_init_rows(rows: List[Dict[str, float]]) -> str:
    kls = [row["kl"] for row in rows]
    lambda_p = [row["lambda_P"] for row in rows]
    lambda_a = [row["lambda_A"] for row in rows]
    lines = ["Initialization sensitivity (exponential link):"]
    for row in rows:
        lines.append(
            f"  init=({row['init_lambda_zero']:.3f},{row['init_lambda_A']:.3f},{row['init_lambda_P']:.3f}) | KL={row['kl']:.6f} | fitted=({row['lambda_zero']:.6f},{row['lambda_A']:.6f},{row['lambda_P']:.6f})"
        )
    lines.append(
        "Fitted-parameter spread across initializations: "
        f"lambda_A mean={np.mean(lambda_a):.6f}, sd={np.std(lambda_a):.6f}; "
        f"lambda_P mean={np.mean(lambda_p):.6f}, sd={np.std(lambda_p):.6f}; "
        f"KL mean={np.mean(kls):.6f}, sd={np.std(kls):.6f}."
    )
    return "\n".join(lines)


def write_text_report(
    path: str,
    args: argparse.Namespace,
    data: pd.DataFrame,
    link_rows: List[Dict[str, float]],
    bw_rows: List[Dict[str, float]],
    init_rows: List[Dict[str, float]],
) -> None:
    best_link = min(link_rows, key=lambda x: x["kl"])
    best_bw = min(bw_rows, key=lambda x: x["kl"])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Validation report for the continuous-candidate model\n")
        handle.write("=" * 72 + "\n\n")
        handle.write(f"Data file: {args.data}\n")
        handle.write(f"Tokens after filtering: {len(data)}\n")
        handle.write(f"Vowel(s): {args.vogais}\n")
        handle.write(f"Speaker(s): {args.entrevistados}\n")
        handle.write(f"Resolution: {args.resolution}\n")
        handle.write(f"Max iterations per optimization: {args.maxiter}\n\n")

        handle.write(summarize_link_rows(link_rows))
        handle.write("\n\n")
        handle.write(summarize_bandwidth_rows(bw_rows))
        handle.write("\n\n")
        handle.write(summarize_init_rows(init_rows))
        handle.write("\n\n")

        handle.write("Suggested prose-ready findings\n")
        handle.write("-" * 72 + "\n")
        handle.write(
            f"Under a shared optimization protocol, the best-performing link was {best_link['link']} "
            f"(KL = {best_link['kl']:.6f}). "
            f"The subtraction link produced a negative perceptual-violation share of "
            f"{next(row['vp_negative_share'] for row in link_rows if row['link'] == 'subtraction'):.4f}, "
            "which quantifies the positivity problem discussed in the paper.\n"
        )
        handle.write(
            f"Across the tested KDE bandwidths, the exponentialized model's KL values ranged from "
            f"{min(row['kl'] for row in bw_rows):.6f} to {max(row['kl'] for row in bw_rows):.6f}, "
            f"with the best fit at bandwidth {best_bw['bandwidth']:.2f}.\n"
        )
        handle.write(
            f"Across the tested optimization initializations, the exponentialized model yielded a KL mean of "
            f"{np.mean([row['kl'] for row in init_rows]):.6f} with standard deviation "
            f"{np.std([row['kl'] for row in init_rows]):.6f}, indicating "
            "the degree of initialization sensitivity in the fitted solution.\n"
        )


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    vogais = [x.strip() for x in args.vogais.split(",") if x.strip()]
    entrevistados = parse_int_list(args.entrevistados)
    bandwidths = parse_float_list(args.bandwidths)
    init_sets = parse_init_sets(args.init_sets)

    data = load_data(args.data, vogais, entrevistados)

    # Main configuration uses the paper's bandwidth of 0.20 for link comparison and init sensitivity.
    scaler, kde = build_scaler_and_kde(data, 0.20)
    params = normalize_params(args, data, scaler)
    grid_f1, grid_f2, mesh_f1, mesh_f2, dx, dy = make_grid(params)
    kde_grid = compute_kde_grid(kde, mesh_f1, mesh_f2)
    marginal_f1 = create_f1_marginal(kde_grid, grid_f1, grid_f2)

    link_rows = run_link_comparison(params, kde_grid, marginal_f1, mesh_f1, mesh_f2, dx, dy, args.maxiter)
    bw_rows = run_bandwidth_sensitivity(data, args, bandwidths)
    init_rows = run_initialization_sensitivity(params, kde_grid, marginal_f1, mesh_f1, mesh_f2, dx, dy, init_sets, args.maxiter)

    write_csv(os.path.join(args.output_dir, "validation_link_comparison.csv"), link_rows)
    write_csv(os.path.join(args.output_dir, "validation_bandwidth_sensitivity.csv"), bw_rows)
    write_csv(os.path.join(args.output_dir, "validation_initialization_sensitivity.csv"), init_rows)

    metadata = {
        "data": args.data,
        "vogais": vogais,
        "entrevistados": entrevistados,
        "resolution": args.resolution,
        "maxiter": args.maxiter,
        "bandwidths": bandwidths,
        "init_sets": init_sets,
        "tokens": int(len(data)),
    }
    with open(os.path.join(args.output_dir, "validation_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    write_text_report(
        os.path.join(args.output_dir, "validation_summary.txt"),
        args,
        data,
        link_rows,
        bw_rows,
        init_rows,
    )

    print("Validation suite completed successfully.")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
