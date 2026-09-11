#!/usr/bin/env python3
"""Reproduce the controlled perturbation test for family discrimination."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_core import (
    BASELINE_DATASET,
    DEFAULT_DATA_ROOT,
    FamilyResponses,
    evaluate_family,
    ln_parameter_grid,
    load_database,
    mpl_parameter_grid,
    subset_responses,
)


DEFAULT_SEED = 20260911
NOISE_LEVELS = (0.01, 0.03, 0.05)
POOL_SIZES = (700, 1000)
TRUTH_CASES = (
    ("MPL-discriminated", "MPL", np.array([3.6, 0.2])),
    ("LN-near-degenerate", "LN", np.array([0.15, 1.5])),
)


@dataclass(frozen=True)
class Observations:
    cext: np.ndarray
    g: np.ndarray
    f11: np.ndarray
    dolp: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--realizations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def append_truth(pool: FamilyResponses, truth: FamilyResponses) -> FamilyResponses:
    return FamilyResponses(
        family=pool.family,
        params=np.vstack([pool.params, truth.params]),
        cext=np.r_[pool.cext, truth.cext],
        csca=np.r_[pool.csca, truth.csca],
        g=np.r_[pool.g, truth.g],
        f11=np.vstack([pool.f11, truth.f11]),
        dolp=np.vstack([pool.dolp, truth.dolp]),
    )


def scalar_distance(truth: FamilyResponses, candidates: FamilyResponses) -> np.ndarray:
    return np.sqrt(np.log(candidates.cext / truth.cext[0]) ** 2 + (candidates.g - truth.g[0]) ** 2)


def candidate_pool(
    truth: FamilyResponses,
    candidates: FamilyResponses,
    pool_size: int,
    include_exact_truth: bool,
) -> FamilyResponses:
    order = np.argsort(scalar_distance(truth, candidates), kind="stable")[:pool_size]
    pool = subset_responses(candidates, order)
    return append_truth(pool, truth) if include_exact_truth else pool


def perturb(truth: FamilyResponses, eta: float, realizations: int, rng: np.random.Generator) -> Observations:
    if len(truth.params) != 1:
        raise ValueError("The perturbation routine expects one truth response")
    cext = truth.cext[0] * np.exp(eta * rng.standard_normal(realizations) - 0.5 * eta**2)
    f11 = truth.f11[0] * np.exp(
        eta * rng.standard_normal((realizations, truth.f11.shape[1])) - 0.5 * eta**2
    )
    g = np.clip(truth.g[0] + eta * rng.standard_normal(realizations), -1.0, 1.0)
    dolp = np.clip(
        truth.dolp[0] + eta * rng.standard_normal((realizations, truth.dolp.shape[1])),
        -1.0,
        1.0,
    )
    return Observations(cext=cext, g=g, f11=f11, dolp=dolp)


def minimum_joint_discrepancy(
    candidates: FamilyResponses,
    observations: Observations,
    angle_mask: np.ndarray,
    candidate_chunk: int = 256,
) -> np.ndarray:
    """Return the exact minimum over the supplied finite candidate pool."""

    n_realizations = len(observations.cext)
    minimum_squared = np.full(n_realizations, np.inf)
    obs_log_cext = np.log(observations.cext)
    obs_log_f11 = np.log(np.maximum(observations.f11[:, angle_mask], 1e-300))
    obs_dolp = observations.dolp[:, angle_mask]
    inv_angles = 1.0 / angle_mask.sum()

    obs_log_f11_sq = np.sum(obs_log_f11**2, axis=1) * inv_angles
    obs_dolp_sq = np.sum(obs_dolp**2, axis=1) * inv_angles
    for start in range(0, len(candidates.params), candidate_chunk):
        stop = min(start + candidate_chunk, len(candidates.params))
        log_cext = np.log(candidates.cext[start:stop])
        scalar_squared = (
            (log_cext[:, None] - obs_log_cext[None, :]) ** 2
            + (candidates.g[start:stop, None] - observations.g[None, :]) ** 2
        )

        candidate_log_f11 = np.log(np.maximum(candidates.f11[start:stop, angle_mask], 1e-300))
        phase_squared = (
            np.sum(candidate_log_f11**2, axis=1)[:, None] * inv_angles
            + obs_log_f11_sq[None, :]
            - 2.0 * (candidate_log_f11 @ obs_log_f11.T) * inv_angles
        )

        candidate_dolp = candidates.dolp[start:stop, angle_mask]
        dolp_squared = (
            np.sum(candidate_dolp**2, axis=1)[:, None] * inv_angles
            + obs_dolp_sq[None, :]
            - 2.0 * (candidate_dolp @ obs_dolp.T) * inv_angles
        )
        total_squared = scalar_squared + np.maximum(phase_squared, 0.0) + np.maximum(dolp_squared, 0.0)
        minimum_squared = np.minimum(minimum_squared, np.min(total_squared, axis=0))
    return np.sqrt(np.maximum(minimum_squared, 0.0))


def save_figure(summary: pd.DataFrame, output: Path) -> None:
    main = summary[summary["pool_size"] == max(POOL_SIZES)].copy()
    colors = {"MPL-discriminated": "#0072B2", "LN-near-degenerate": "#D55E00"}
    labels = {
        "MPL-discriminated": r"MPL truth: $\alpha=3.6$",
        "LN-near-degenerate": r"LN truth: $x_g=0.15$, $\sigma_g=1.5$",
    }
    fig, axes = plt.subplots(2, 1, figsize=(6.7, 6.0), sharex=True)
    for case_name in colors:
        part = main[main["truth_case"] == case_name].sort_values("noise_pct")
        x = part["noise_pct"].to_numpy(float)
        axes[0].plot(
            x,
            100.0 * part["correct_selection_fraction"],
            marker="o" if case_name == "MPL-discriminated" else "s",
            ls="-" if case_name == "MPL-discriminated" else "--",
            color=colors[case_name],
            label=labels[case_name],
        )
        axes[1].plot(
            x,
            part["median_margin"],
            marker="o" if case_name == "MPL-discriminated" else "s",
            ls="-" if case_name == "MPL-discriminated" else "--",
            color=colors[case_name],
            label=labels[case_name],
        )
        axes[1].fill_between(
            x,
            part["p05_margin"].to_numpy(float),
            part["p95_margin"].to_numpy(float),
            color=colors[case_name],
            alpha=0.16,
        )
    axes[0].set_ylabel("Correct-family selection (%)")
    axes[0].set_ylim(0, 105)
    axes[1].axhline(0.0, color="0.35", ls=":", lw=1.0)
    axes[1].set_xlabel("Controlled perturbation level (%)")
    axes[1].set_ylabel(r"Family-separation margin $\Delta E$")
    for axis in axes:
        axis.grid(True, alpha=0.22)
        axis.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output / "fig6_noise_robustness.pdf", bbox_inches="tight")
    fig.savefig(output / "fig6_noise_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    database = load_database(args.data_root, BASELINE_DATASET)
    print("Evaluating frozen LN and MPL grids for perturbation pools ...", flush=True)
    all_responses = {
        "LN": evaluate_family(database, "LN", ln_parameter_grid(), args.batch_size),
        "MPL": evaluate_family(database, "MPL", mpl_parameter_grid(), args.batch_size),
    }

    seed_sequence = np.random.SeedSequence(args.seed)
    child_seeds = seed_sequence.spawn(len(TRUTH_CASES) * len(NOISE_LEVELS))
    summary_rows: list[dict[str, float | int | str]] = []
    realization_rows: list[pd.DataFrame] = []
    seed_index = 0

    for case_name, truth_family, truth_params in TRUTH_CASES:
        wrong_family = "LN" if truth_family == "MPL" else "MPL"
        truth = evaluate_family(database, truth_family, truth_params)
        for eta in NOISE_LEVELS:
            child_seed = child_seeds[seed_index]
            seed_index += 1
            rng = np.random.default_rng(child_seed)
            observations = perturb(truth, eta, args.realizations, rng)
            for pool_size in POOL_SIZES:
                correct_pool = candidate_pool(truth, all_responses[truth_family], pool_size, True)
                wrong_pool = candidate_pool(truth, all_responses[wrong_family], pool_size, False)
                correct_min = minimum_joint_discrepancy(correct_pool, observations, database.angle_mask)
                wrong_min = minimum_joint_discrepancy(wrong_pool, observations, database.angle_mask)
                margin = wrong_min - correct_min
                summary_rows.append(
                    {
                        "truth_case": case_name,
                        "truth_family": truth_family,
                        "truth_param_1": truth_params[0],
                        "truth_param_2": truth_params[1],
                        "noise_fraction": eta,
                        "noise_pct": 100.0 * eta,
                        "realizations": args.realizations,
                        "master_seed": args.seed,
                        "spawn_key": ".".join(map(str, child_seed.spawn_key)),
                        "pool_size": pool_size,
                        "correct_pool_count_including_truth": len(correct_pool.params),
                        "wrong_pool_count": len(wrong_pool.params),
                        "correct_selection_fraction": float(np.mean(margin > 0.0)),
                        "median_margin": float(np.median(margin)),
                        "p05_margin": float(np.quantile(margin, 0.05)),
                        "p95_margin": float(np.quantile(margin, 0.95)),
                    }
                )
                if pool_size == max(POOL_SIZES):
                    realization_rows.append(
                        pd.DataFrame(
                            {
                                "truth_case": case_name,
                                "truth_family": truth_family,
                                "noise_fraction": eta,
                                "noise_pct": 100.0 * eta,
                                "realization": np.arange(args.realizations),
                                "correct_min_E_joint": correct_min,
                                "wrong_min_E_joint": wrong_min,
                                "margin": margin,
                                "correct_selected": margin > 0.0,
                                "master_seed": args.seed,
                                "spawn_key": ".".join(map(str, child_seed.spawn_key)),
                                "pool_size": pool_size,
                            }
                        )
                    )
            print(f"Completed {case_name}, perturbation={100.0 * eta:.0f}%", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "noise_robustness_summary.csv", index=False)
    realizations = pd.concat(realization_rows, ignore_index=True)
    realizations.to_csv(output / "noise_robustness_realizations.csv", index=False)

    sensitivity = summary.pivot_table(
        index=["truth_case", "noise_pct"],
        columns="pool_size",
        values=["correct_selection_fraction", "median_margin", "p05_margin", "p95_margin"],
    )
    sensitivity.columns = [f"{metric}_pool{pool}" for metric, pool in sensitivity.columns]
    sensitivity = sensitivity.reset_index()
    for metric in ("correct_selection_fraction", "median_margin", "p05_margin", "p95_margin"):
        sensitivity[f"delta_{metric}_1000_minus_700"] = sensitivity[f"{metric}_pool1000"] - sensitivity[f"{metric}_pool700"]
    sensitivity.to_csv(output / "noise_pool_sensitivity.csv", index=False)

    save_figure(summary, output)
    print("\nPrimary 1000-candidate-pool summary")
    print(
        summary[summary["pool_size"] == max(POOL_SIZES)][
            ["truth_case", "noise_pct", "correct_selection_fraction", "median_margin", "p05_margin", "p95_margin"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
