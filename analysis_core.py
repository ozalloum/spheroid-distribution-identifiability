#!/usr/bin/env python3
"""Shared numerical core for the cross-family identifiability study.

The routines in this module reproduce the quadrature used in the original
analysis while making the data paths portable.  All input data are resolved
relative to the package unless an explicit data root is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


Family = Literal["LN", "MPL"]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PACKAGE_ROOT / "validation_data" / "monodisperse_database"
BASELINE_DATASET = "adaptive_seed_hardcase"
CHECK_DATASET = "hardcase_ndgs3"

MPL_TRUTHS = np.array([[3.0, 0.2], [3.3, 0.2], [3.6, 0.2], [3.9, 0.2], [4.2, 0.2]])
LN_TRUTHS = np.array([[0.15, 1.5], [0.25, 1.8], [0.40, 2.1], [0.65, 2.4], [1.00, 2.7]])


@dataclass(frozen=True)
class Database:
    """Validated monodisperse database and exact dense-grid integration maps."""

    dataset_name: str
    manifest: pd.DataFrame
    angles: np.ndarray
    angle_mask: np.ndarray
    dense_x: np.ndarray
    quadrature: np.ndarray
    linear_basis: csr_matrix
    product_basis: csr_matrix
    cext_nodes: np.ndarray
    csca_nodes: np.ndarray
    g_nodes: np.ndarray
    csca_g_coeff: np.ndarray
    csca_f11_coeff: np.ndarray
    csca_f12_coeff: np.ndarray


@dataclass(frozen=True)
class FamilyResponses:
    """Optical responses evaluated for a finite parameter grid."""

    family: Family
    params: np.ndarray
    cext: np.ndarray
    csca: np.ndarray
    g: np.ndarray
    f11: np.ndarray
    dolp: np.ndarray


def _resolve_angular_file(dataset_dir: Path, run_id: str, x_value: float) -> Path:
    matches = sorted((dataset_dir / "runs").glob(f"*{run_id}*/angular.csv"))
    if len(matches) != 1:
        fallback = sorted((dataset_dir / "runs").glob(f"*_x{x_value:g}_eps*/angular.csv"))
        matches = fallback
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one angular.csv for x={x_value:g}, run_id={run_id}; found {len(matches)}"
        )
    return matches[0]


def _trapezoid_coefficients(x: np.ndarray) -> np.ndarray:
    q = np.empty_like(x)
    q[0] = 0.5 * (x[1] - x[0])
    q[-1] = 0.5 * (x[-1] - x[-2])
    q[1:-1] = 0.5 * (x[2:] - x[:-2])
    return q


def _interpolation_bases(dense_x: np.ndarray, nodes: np.ndarray) -> tuple[csr_matrix, csr_matrix]:
    """Return sparse bases for linear values and products of linear values.

    The original analysis interpolated every monodisperse quantity linearly on
    a dense grid and then multiplied the interpolants.  The second sparse basis
    preserves that product exactly on the same dense grid without constructing
    a large dense angle-by-size array for every candidate distribution.
    """

    right = np.searchsorted(nodes, dense_x, side="right")
    right = np.clip(right, 1, len(nodes) - 1)
    left = right - 1
    span = nodes[right] - nodes[left]
    t = (dense_x - nodes[left]) / span
    one_minus_t = 1.0 - t
    rows = np.arange(len(dense_x))

    linear = csr_matrix(
        (
            np.concatenate([one_minus_t, t]),
            (np.concatenate([rows, rows]), np.concatenate([left, right])),
        ),
        shape=(len(dense_x), len(nodes)),
    )

    interval = left
    product = csr_matrix(
        (
            np.concatenate([one_minus_t**2, one_minus_t * t, t**2]),
            (
                np.concatenate([rows, rows, rows]),
                np.concatenate([3 * interval, 3 * interval + 1, 3 * interval + 2]),
            ),
        ),
        shape=(len(dense_x), 3 * (len(nodes) - 1)),
    )
    return linear, product


def _product_coefficients(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Coefficients for the product of two piecewise-linear interpolants."""

    b2 = np.asarray(b, dtype=float)
    scalar = b2.ndim == 1
    if scalar:
        b2 = b2[:, None]
    a2 = np.asarray(a, dtype=float)[:, None]
    c0 = a2[:-1] * b2[:-1]
    c1 = a2[:-1] * b2[1:] + a2[1:] * b2[:-1]
    c2 = a2[1:] * b2[1:]
    coeff = np.stack([c0, c1, c2], axis=1).reshape(-1, b2.shape[1])
    return coeff[:, 0] if scalar else coeff


def load_database(data_root: Path = DEFAULT_DATA_ROOT, dataset_name: str = BASELINE_DATASET) -> Database:
    dataset_dir = Path(data_root).resolve() / dataset_name
    manifest_path = dataset_dir / "grid_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing monodisperse manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path).sort_values("x").reset_index(drop=True)
    if len(manifest) != 60:
        raise ValueError(f"Expected 60 monodisperse nodes in {dataset_name}; found {len(manifest)}")
    if manifest["x"].duplicated().any() or not np.all(np.diff(manifest["x"]) > 0):
        raise ValueError(f"The size grid in {dataset_name} is not strictly increasing and unique")
    if set(manifest["status"].astype(str)) != {"ok"}:
        raise ValueError(f"At least one monodisperse run in {dataset_name} is not marked 'ok'")
    if not np.isclose(manifest["x"].iloc[0], 0.01) or not np.isclose(manifest["x"].iloc[-1], 127.5):
        raise ValueError(f"Unexpected size range in {dataset_name}")
    for column, expected in (("eps", 0.6666667), ("mrr", 1.07), ("mri", 0.0), ("npna", 181)):
        values = manifest[column].to_numpy(float)
        if not np.allclose(values, expected, rtol=0.0, atol=1e-10):
            raise ValueError(f"Unexpected {column} values in {dataset_name}")

    angles: np.ndarray | None = None
    f11_rows: list[np.ndarray] = []
    f12_rows: list[np.ndarray] = []
    for row in manifest.itertuples(index=False):
        angular = pd.read_csv(_resolve_angular_file(dataset_dir, str(row.run_id), float(row.x)))
        theta = angular["theta_deg"].to_numpy(float)
        if angles is None:
            angles = theta
        elif not np.array_equal(theta, angles):
            raise ValueError(f"Inconsistent angular grid at x={row.x:g} in {dataset_name}")
        f11_rows.append(angular["F11"].to_numpy(float))
        f12_rows.append(angular["F12"].to_numpy(float))
    assert angles is not None
    if len(angles) != 181 or not np.isclose(angles[0], 0.0) or not np.isclose(angles[-1], 180.0):
        raise ValueError(f"Unexpected angular grid in {dataset_name}")

    x_nodes = np.r_[0.0, manifest["x"].to_numpy(float)]
    cext_nodes = np.r_[0.0, manifest["cext"].to_numpy(float)]
    csca_nodes = np.r_[0.0, manifest["csca"].to_numpy(float)]
    g_nodes = np.r_[0.0, manifest["asymmetry_g"].to_numpy(float)]
    f11_nodes = np.vstack([f11_rows[0], np.asarray(f11_rows)])
    f12_nodes = np.vstack([f12_rows[0], np.asarray(f12_rows)])

    dense_x = np.unique(
        np.r_[
            np.linspace(0.0, 0.5, 501),
            np.linspace(0.5, 10.0, 1901),
            np.linspace(10.0, 127.5, 2351),
            manifest["x"].to_numpy(float),
            0.2,
        ]
    )
    quadrature = _trapezoid_coefficients(dense_x)
    linear_basis, product_basis = _interpolation_bases(dense_x, x_nodes)
    angle_mask = (angles >= 5.0) & (angles <= 175.0)

    return Database(
        dataset_name=dataset_name,
        manifest=manifest,
        angles=angles,
        angle_mask=angle_mask,
        dense_x=dense_x,
        quadrature=quadrature,
        linear_basis=linear_basis,
        product_basis=product_basis,
        cext_nodes=cext_nodes,
        csca_nodes=csca_nodes,
        g_nodes=g_nodes,
        csca_g_coeff=_product_coefficients(csca_nodes, g_nodes),
        csca_f11_coeff=_product_coefficients(csca_nodes, f11_nodes),
        csca_f12_coeff=_product_coefficients(csca_nodes, f12_nodes),
    )


def ln_parameter_grid() -> np.ndarray:
    xg = np.geomspace(0.005, 5.0, 101)
    sigma = np.linspace(1.10, 6.00, 99)
    return np.asarray([(xg_value, sigma_value) for sigma_value in sigma for xg_value in xg])


def mpl_parameter_grid() -> np.ndarray:
    alpha = np.linspace(2.2, 8.0, 117)
    xb = np.geomspace(0.01, 3.0, 101)
    return np.asarray([(alpha_value, xb_value) for alpha_value in alpha for xb_value in xb])


def _number_distributions(family: Family, params: np.ndarray, x: np.ndarray) -> np.ndarray:
    params = np.atleast_2d(np.asarray(params, dtype=float))
    if family == "MPL":
        alpha = params[:, 0, None]
        xb = params[:, 1, None]
        ratio = np.divide(x[None, :], xb, out=np.zeros((len(params), len(x))), where=xb > 0)
        n = np.ones_like(ratio)
        mask = x[None, :] > xb
        n[mask] = ratio[mask] ** (-np.broadcast_to(alpha, ratio.shape)[mask])
        return n
    if family == "LN":
        xg = params[:, 0, None]
        sigma = params[:, 1, None]
        if np.any(xg <= 0) or np.any(sigma <= 1):
            raise ValueError("Lognormal candidates require x_g > 0 and sigma_g > 1")
        n = np.zeros((len(params), len(x)), dtype=float)
        positive = x > 0
        log_ratio = np.log(x[positive][None, :] / xg)
        n[:, positive] = np.exp(-0.5 * (log_ratio / np.log(sigma)) ** 2) / x[positive][None, :]
        return n
    raise ValueError(f"Unknown family: {family}")


def evaluate_family(
    database: Database,
    family: Family,
    params: np.ndarray,
    batch_size: int = 512,
) -> FamilyResponses:
    params = np.atleast_2d(np.asarray(params, dtype=float))
    count = len(params)
    nangles = len(database.angles)
    cext = np.empty(count)
    csca = np.empty(count)
    asymmetry = np.empty(count)
    f11 = np.empty((count, nangles))
    dolp = np.empty((count, nangles))

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        n = _number_distributions(family, params[start:stop], database.dense_x)
        weighted_n = n * database.quadrature[None, :]
        number_integral = weighted_n.sum(axis=1)
        linear_moments = np.asarray(weighted_n @ database.linear_basis)
        product_moments = np.asarray(weighted_n @ database.product_basis)

        cext_integral = linear_moments @ database.cext_nodes
        csca_integral = linear_moments @ database.csca_nodes
        g_integral = product_moments @ database.csca_g_coeff
        f11_integral = product_moments @ database.csca_f11_coeff
        f12_integral = product_moments @ database.csca_f12_coeff

        if np.any(number_integral <= 0) or np.any(csca_integral <= 0) or np.any(f11_integral <= 0):
            raise FloatingPointError(f"Non-positive normalization encountered for {family} candidates")
        cext[start:stop] = cext_integral / number_integral
        csca[start:stop] = csca_integral / number_integral
        asymmetry[start:stop] = g_integral / csca_integral
        f11[start:stop] = f11_integral / csca_integral[:, None]
        dolp[start:stop] = -f12_integral / f11_integral

    return FamilyResponses(family, params, cext, csca, asymmetry, f11, dolp)


def subset_responses(responses: FamilyResponses, indices: np.ndarray) -> FamilyResponses:
    idx = np.asarray(indices)
    return FamilyResponses(
        responses.family,
        responses.params[idx],
        responses.cext[idx],
        responses.csca[idx],
        responses.g[idx],
        responses.f11[idx],
        responses.dolp[idx],
    )


def metric_components(truth: FamilyResponses, candidates: FamilyResponses, mask: np.ndarray) -> dict[str, np.ndarray]:
    if len(truth.params) != 1:
        raise ValueError("metric_components expects exactly one truth response")
    log_cext = np.log(candidates.cext / truth.cext[0])
    delta_g = candidates.g - truth.g[0]
    e_scalar = np.sqrt(log_cext**2 + delta_g**2)
    log_ratio = np.log(np.maximum(candidates.f11[:, mask], 1e-300) / np.maximum(truth.f11[0, mask], 1e-300))
    e_phase = np.sqrt(np.mean(log_ratio**2, axis=1))
    e_dolp = np.sqrt(np.mean((candidates.dolp[:, mask] - truth.dolp[0, mask]) ** 2, axis=1))
    e_joint = np.sqrt(e_scalar**2 + e_phase**2 + e_dolp**2)
    return {"E_scalar": e_scalar, "E_phase": e_phase, "E_DoLP": e_dolp, "E_joint": e_joint}


def response_row(responses: FamilyResponses, index: int) -> FamilyResponses:
    return subset_responses(responses, np.asarray([index]))

