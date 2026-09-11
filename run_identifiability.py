#!/usr/bin/env python3
"""Reproduce the deterministic reciprocal cross-family retrieval experiment.

This portable version evaluates the complete frozen parameter grids for the
joint criterion and repeats every reported solution with an independent
orientation-quadrature database.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_core import (
    BASELINE_DATASET,
    CHECK_DATASET,
    DEFAULT_DATA_ROOT,
    LN_TRUTHS,
    MPL_TRUTHS,
    FamilyResponses,
    evaluate_family,
    ln_parameter_grid,
    load_database,
    metric_components,
    mpl_parameter_grid,
    response_row,
)


COLORS = {
    "truth": "#202124",
    "scalar": "#D55E00",
    "joint": "#0072B2",
    "dolp": "#009E73",
    "phase": "#CC79A7",
}
TRUTH_CACHE: dict[tuple[str, tuple[float, float]], FamilyResponses] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing adaptive_seed_hardcase and hardcase_ndgs3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for CSV and figure outputs",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def configure_plots() -> None:
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


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def number_distribution(family: str, params: np.ndarray, x: np.ndarray) -> np.ndarray:
    if family == "MPL":
        alpha, xb = params
        n = np.ones_like(x)
        mask = x > xb
        n[mask] = (x[mask] / xb) ** (-alpha)
    else:
        xg, sigma = params
        n = np.zeros_like(x)
        mask = x > 0
        n[mask] = np.exp(-0.5 * (np.log(x[mask] / xg) / np.log(sigma)) ** 2) / x[mask]
    return n


def result_record(
    truth_family: str,
    truth_params: np.ndarray,
    retrievals: FamilyResponses,
    metrics: dict[str, np.ndarray],
    criterion: str,
    index: int,
) -> dict[str, float | str | int | bool]:
    truth = TRUTH_CACHE[(truth_family, tuple(truth_params))]
    candidate = response_row(retrievals, index)
    record: dict[str, float | str | int | bool] = {
        "truth_family": truth_family,
        "retrieval_family": retrievals.family,
        "criterion": criterion,
        "E_scalar": float(metrics["E_scalar"][index]),
        "E_phase": float(metrics["E_phase"][index]),
        "E_DoLP": float(metrics["E_DoLP"][index]),
        "E_joint": float(metrics["E_joint"][index]),
        "truth_cext": float(truth.cext[0]),
        "retr_cext": float(candidate.cext[0]),
        "truth_g": float(truth.g[0]),
        "retr_g": float(candidate.g[0]),
        "cext_rel_pct": float(100.0 * (candidate.cext[0] / truth.cext[0] - 1.0)),
        "g_abs": float(abs(candidate.g[0] - truth.g[0])),
        "grid_candidates": int(len(retrievals.params)),
    }
    if truth_family == "MPL":
        record.update({"truth_alpha": float(truth_params[0]), "truth_xb": float(truth_params[1])})
    else:
        record.update({"truth_xg": float(truth_params[0]), "truth_sigma_g": float(truth_params[1])})
    if retrievals.family == "LN":
        xg, sigma = candidate.params[0]
        record.update(
            {
                "retr_xg": float(xg),
                "retr_sigma_g": float(sigma),
                "retrieval_on_boundary": bool(
                    np.isclose(xg, retrievals.params[:, 0].min())
                    or np.isclose(xg, retrievals.params[:, 0].max())
                    or np.isclose(sigma, retrievals.params[:, 1].min())
                    or np.isclose(sigma, retrievals.params[:, 1].max())
                ),
            }
        )
    else:
        alpha, xb = candidate.params[0]
        record.update(
            {
                "retr_alpha": float(alpha),
                "retr_xb": float(xb),
                "retrieval_on_boundary": bool(
                    np.isclose(alpha, retrievals.params[:, 0].min())
                    or np.isclose(alpha, retrievals.params[:, 0].max())
                    or np.isclose(xb, retrievals.params[:, 1].min())
                    or np.isclose(xb, retrievals.params[:, 1].max())
                ),
            }
        )
    return record


def same_candidate(a: FamilyResponses, b: FamilyResponses) -> bool:
    return bool(np.allclose(a.params[0], b.params[0], rtol=0.0, atol=1e-14))


def plot_fit(
    angles: np.ndarray,
    truth: FamilyResponses,
    scalar: FamilyResponses,
    joint: FamilyResponses,
    truth_label: str,
    candidate_family: str,
    output: Path,
    stem: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.7, 6.0), sharex=True)
    axes[0].semilogy(angles, truth.f11[0], color=COLORS["truth"], label=truth_label)
    axes[1].plot(angles, truth.dolp[0], color=COLORS["truth"], label=truth_label)
    if same_candidate(scalar, joint):
        label = f"{candidate_family} best scalar = joint"
        axes[0].semilogy(angles, joint.f11[0], "--", color=COLORS["joint"], label=label)
        axes[1].plot(angles, joint.dolp[0], "--", color=COLORS["joint"], label=label)
    else:
        axes[0].semilogy(angles, scalar.f11[0], "--", color=COLORS["scalar"], label=f"{candidate_family} scalar fit")
        axes[0].semilogy(angles, joint.f11[0], ":", color=COLORS["joint"], label=f"{candidate_family} joint fit")
        axes[1].plot(angles, scalar.dolp[0], "--", color=COLORS["scalar"], label=f"{candidate_family} scalar fit")
        axes[1].plot(angles, joint.dolp[0], ":", color=COLORS["joint"], label=f"{candidate_family} joint fit")
    axes[0].set_ylabel(r"$F_{11}$")
    axes[1].set_xlabel(r"Scattering angle $\theta$ (deg)")
    axes[1].set_ylabel(r"$P_{\mathrm{L}}=-F_{12}/F_{11}$")
    for axis in axes:
        axis.grid(True, alpha=0.22)
        axis.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, output, stem)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_plots()

    baseline = load_database(args.data_root, BASELINE_DATASET)
    check = load_database(args.data_root, CHECK_DATASET)
    if not np.array_equal(baseline.manifest["x"].to_numpy(float), check.manifest["x"].to_numpy(float)):
        raise ValueError("The NDGS=2 and NDGS=3 databases do not share the same 60 size nodes")
    if not np.array_equal(baseline.angles, check.angles):
        raise ValueError("The NDGS=2 and NDGS=3 databases do not share the same angular grid")

    ln_grid = ln_parameter_grid()
    mpl_grid = mpl_parameter_grid()
    print(f"Evaluating all {len(ln_grid)} LN candidates ...", flush=True)
    ln_responses = evaluate_family(baseline, "LN", ln_grid, args.batch_size)
    print(f"Evaluating all {len(mpl_grid)} MPL candidates ...", flush=True)
    mpl_responses = evaluate_family(baseline, "MPL", mpl_grid, args.batch_size)

    for params in MPL_TRUTHS:
        TRUTH_CACHE[("MPL", tuple(params))] = evaluate_family(baseline, "MPL", params)
    for params in LN_TRUTHS:
        TRUTH_CACHE[("LN", tuple(params))] = evaluate_family(baseline, "LN", params)

    records: list[dict[str, float | str | int | bool]] = []
    selections: dict[tuple[str, tuple[float, float]], dict[str, int]] = {}
    metric_cache: dict[tuple[str, tuple[float, float]], dict[str, np.ndarray]] = {}

    for truth_family, truths, retrievals in (
        ("MPL", MPL_TRUTHS, ln_responses),
        ("LN", LN_TRUTHS, mpl_responses),
    ):
        for params in truths:
            key = (truth_family, tuple(params))
            metrics = metric_components(TRUTH_CACHE[key], retrievals, baseline.angle_mask)
            scalar_index = int(np.argmin(metrics["E_scalar"]))
            joint_index = int(np.argmin(metrics["E_joint"]))
            selections[key] = {"scalar": scalar_index, "joint": joint_index}
            metric_cache[key] = metrics
            records.append(result_record(truth_family, params, retrievals, metrics, "joint", joint_index))
            records.append(result_record(truth_family, params, retrievals, metrics, "scalar", scalar_index))

    result_columns = [
        "truth_family", "truth_alpha", "truth_xb", "truth_xg", "truth_sigma_g",
        "retrieval_family", "criterion", "retr_xg", "retr_sigma_g", "retr_alpha", "retr_xb",
        "E_scalar", "E_phase", "E_DoLP", "E_joint", "truth_cext", "retr_cext",
        "truth_g", "retr_g", "cext_rel_pct", "g_abs", "grid_candidates", "retrieval_on_boundary",
    ]
    results = pd.DataFrame(records).reindex(columns=result_columns)
    results.to_csv(output / "cross_family_best_fits.csv", index=False)

    search_audit = results.loc[
        :, ["truth_family", "truth_alpha", "truth_xg", "truth_sigma_g", "criterion", "retrieval_family", "grid_candidates", "retrieval_on_boundary", "E_scalar", "E_joint"]
    ].copy()
    search_audit["joint_search_scope"] = "complete frozen grid"
    search_audit.to_csv(output / "deterministic_search_audit.csv", index=False)

    check_rows = []
    for record in records:
        if record["truth_family"] == "MPL":
            truth_params = np.array([record["truth_alpha"], record["truth_xb"]], dtype=float)
        else:
            truth_params = np.array([record["truth_xg"], record["truth_sigma_g"]], dtype=float)
        retrieval_family = str(record["retrieval_family"])
        if retrieval_family == "LN":
            candidate_params = np.array([record["retr_xg"], record["retr_sigma_g"]], dtype=float)
        else:
            candidate_params = np.array([record["retr_alpha"], record["retr_xb"]], dtype=float)
        base_truth = evaluate_family(baseline, str(record["truth_family"]), truth_params)
        base_candidate = evaluate_family(baseline, retrieval_family, candidate_params)
        check_truth = evaluate_family(check, str(record["truth_family"]), truth_params)
        check_candidate = evaluate_family(check, retrieval_family, candidate_params)
        base_metrics = metric_components(base_truth, base_candidate, baseline.angle_mask)
        check_metrics = metric_components(check_truth, check_candidate, check.angle_mask)
        check_rows.append(
            {
                "truth_family": record["truth_family"],
                "truth_param_1": truth_params[0],
                "truth_param_2": truth_params[1],
                "criterion": record["criterion"],
                "retrieval_family": retrieval_family,
                "candidate_param_1": candidate_params[0],
                "candidate_param_2": candidate_params[1],
                "baseline_E_joint": base_metrics["E_joint"][0],
                "check_E_joint": check_metrics["E_joint"][0],
                "abs_delta_E_joint": abs(check_metrics["E_joint"][0] - base_metrics["E_joint"][0]),
                "truth_cext_rel_delta": check_truth.cext[0] / base_truth.cext[0] - 1.0,
                "candidate_cext_rel_delta": check_candidate.cext[0] / base_candidate.cext[0] - 1.0,
                "truth_g_abs_delta": abs(check_truth.g[0] - base_truth.g[0]),
                "candidate_g_abs_delta": abs(check_candidate.g[0] - base_candidate.g[0]),
                "truth_phase_rms_log_delta": np.sqrt(np.mean(np.log(check_truth.f11[0, check.angle_mask] / base_truth.f11[0, baseline.angle_mask]) ** 2)),
                "truth_DoLP_rms_delta": np.sqrt(np.mean((check_truth.dolp[0, check.angle_mask] - base_truth.dolp[0, baseline.angle_mask]) ** 2)),
            }
        )
    consistency = pd.DataFrame(check_rows)
    consistency.to_csv(output / "orientation_quadrature_consistency.csv", index=False)

    validation = pd.DataFrame(
        [
            ("baseline_node_count", len(baseline.manifest)),
            ("check_node_count", len(check.manifest)),
            ("angular_sample_count", len(baseline.angles)),
            ("dense_integration_point_count", len(baseline.dense_x)),
            ("LN_grid_candidate_count", len(ln_grid)),
            ("MPL_grid_candidate_count", len(mpl_grid)),
            ("max_abs_orientation_delta_E_joint", consistency["abs_delta_E_joint"].max()),
            ("max_abs_orientation_cext_relative_delta", max(consistency["truth_cext_rel_delta"].abs().max(), consistency["candidate_cext_rel_delta"].abs().max())),
            ("max_abs_orientation_g_delta", max(consistency["truth_g_abs_delta"].max(), consistency["candidate_g_abs_delta"].max())),
            ("max_orientation_phase_rms_log_delta", consistency["truth_phase_rms_log_delta"].max()),
            ("max_orientation_DoLP_rms_delta", consistency["truth_DoLP_rms_delta"].max()),
        ], columns=["check", "value"]
    )
    validation.to_csv(output / "validation_summary.csv", index=False)

    fig, axis = plt.subplots(figsize=(6.7, 4.25))
    for color, marker, alpha in zip(
        ("#0072B2", "#009E73", "#D55E00"),
        ("o", "s", "^"),
        (3.0, 3.6, 4.2),
    ):
        key = ("MPL", (alpha, 0.2))
        ln_params = ln_responses.params[selections[key]["joint"]]
        n_mpl = number_distribution("MPL", np.array([alpha, 0.2]), baseline.dense_x)
        n_ln = number_distribution("LN", ln_params, baseline.dense_x)
        n_mpl /= np.sum(baseline.quadrature * n_mpl)
        n_ln /= np.sum(baseline.quadrature * n_ln)
        axis.loglog(
            baseline.dense_x[1:], n_mpl[1:], color=color, marker=marker,
            markevery=350, ms=3.5, mfc="white", mec=color,
            label=rf"MPL $\alpha={alpha:g}$",
        )
        axis.loglog(
            baseline.dense_x[1:], n_ln[1:], "--", color=color, marker=marker,
            markevery=350, ms=3.5, mfc=color, mec=color,
            label=rf"LN fit: $x_g={ln_params[0]:.3g}$, $\sigma_g={ln_params[1]:.2f}$",
        )
    axis.axvline(0.2, color="0.35", ls=":", lw=1.0)
    axis.set_xlabel("Size parameter $x$")
    axis.set_ylabel("Normalized number density $n(x)$")
    axis.set_xlim(0.01, 127.5)
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    save_figure(fig, output, "fig1_distributions")

    key = ("MPL", (3.6, 0.2))
    plot_fit(
        baseline.angles, TRUTH_CACHE[key],
        response_row(ln_responses, selections[key]["scalar"]),
        response_row(ln_responses, selections[key]["joint"]),
        r"MPL truth $\alpha=3.6$", "LN", output, "fig2_angular_fit_alpha36",
    )

    metrics = metric_cache[key]
    frame = pd.DataFrame({"xg": ln_responses.params[:, 0], "sigma_g": ln_responses.params[:, 1], "E_scalar": metrics["E_scalar"]})
    pivot = frame.pivot(index="sigma_g", columns="xg", values="E_scalar").sort_index()
    fig, axis = plt.subplots(figsize=(6.7, 4.6))
    image = axis.pcolormesh(pivot.columns.to_numpy(), pivot.index.to_numpy(), np.log10(np.maximum(pivot.to_numpy(), 1e-12)), shading="auto", cmap="viridis")
    axis.set_xscale("log")
    axis.set_xlabel(r"Lognormal $x_g$")
    axis.set_ylabel(r"Lognormal $\sigma_g$")
    fig.colorbar(image, ax=axis).set_label(r"$\log_{10}E_{\mathrm{scalar}}$")
    scalar_candidate = response_row(ln_responses, selections[key]["scalar"])
    joint_candidate = response_row(ln_responses, selections[key]["joint"])
    if same_candidate(scalar_candidate, joint_candidate):
        axis.plot(joint_candidate.params[0, 0], joint_candidate.params[0, 1], marker="*", color="#D55E00", ms=10, label="best scalar = joint")
    else:
        axis.plot(*scalar_candidate.params[0], marker="o", color="#D55E00", ms=6, label="best scalar")
        axis.plot(*joint_candidate.params[0], marker="s", color="#0072B2", ms=6, label="best joint")
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output, "fig3_scalar_mismatch_map_alpha36")

    joint_mpl = results[(results["truth_family"] == "MPL") & (results["criterion"] == "joint")].sort_values("truth_alpha")
    fig, axis = plt.subplots(figsize=(6.7, 4.25))
    axis.semilogy(joint_mpl["truth_alpha"], joint_mpl["E_scalar"], "o-", color=COLORS["joint"], label=r"$E_{\mathrm{scalar}}$")
    axis.semilogy(joint_mpl["truth_alpha"], joint_mpl["E_phase"], "s--", color=COLORS["phase"], label=r"$E_{\mathrm{phase}}$")
    axis.semilogy(joint_mpl["truth_alpha"], joint_mpl["E_DoLP"], "^-.", color=COLORS["dolp"], label=r"$E_{\mathrm{DoLP}}$")
    axis.set_xlabel(r"MPL truth exponent $\alpha$")
    axis.set_ylabel("Mismatch")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output, "fig4_bestfit_metrics")

    key = ("LN", (0.4, 2.1))
    plot_fit(
        baseline.angles, TRUTH_CACHE[key],
        response_row(mpl_responses, selections[key]["scalar"]),
        response_row(mpl_responses, selections[key]["joint"]),
        r"LN truth $x_g=0.40$, $\sigma_g=2.10$", "MPL", output, "fig5_reciprocal_LN_truth",
    )

    print("\nMPL truth -> LN joint best")
    print(joint_mpl[["truth_alpha", "retr_xg", "retr_sigma_g", "cext_rel_pct", "g_abs", "E_scalar", "E_phase", "E_DoLP", "E_joint"]].to_string(index=False))
    joint_ln = results[(results["truth_family"] == "LN") & (results["criterion"] == "joint")]
    print("\nLN truth -> MPL joint best")
    print(joint_ln[["truth_xg", "truth_sigma_g", "retr_alpha", "retr_xb", "cext_rel_pct", "g_abs", "E_scalar", "E_phase", "E_DoLP", "E_joint"]].to_string(index=False))


if __name__ == "__main__":
    main()
