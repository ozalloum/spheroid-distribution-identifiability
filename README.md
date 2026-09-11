# Supplementary reproducibility package

## Associated manuscript

**Title:** Cross-Family Non-Uniqueness in Polarization-Resolved Optical Characterization of Polydisperse Spheroids  
**Author:** Othman H. Y. Zalloum  
**Package date:** 11 September 2026

The source repository is available at:

<https://github.com/ozalloum/spheroid-distribution-identifiability>

## Purpose

This archive reproduces the ensemble post-processing, reciprocal finite-grid
retrievals, controlled perturbation experiment, machine-readable tables, and
Figs. 1--6 reported in the associated manuscript. It contains the two archived
60-node monodisperse databases used in the study, the analysis code, reference
outputs, validation records, and the verified software environment.

The calculations address size-distribution-family identifiability for one
fixed, randomly oriented prolate spheroid with `EPS=0.6666667`, refractive
index `m=1.07+0i`, and wavelength `0.412 micrometres`. The baseline database
uses `NDGS=2`; the second database uses `NDGS=3` only as an independent
orientation-quadrature consistency check.

## Important scope limitation

The original T-matrix solver executable and source code are not included.
Accordingly, this archive reproduces the reported ensemble analyses from the
archived monodisperse database, but it does not regenerate that database from
Maxwell-equation inputs. The per-run metadata, angular tables, grid manifests,
and validation records preserve the numerical inputs used by the reported
post-processing.

## Quick start

The verified environment uses Python 3.12. From the extracted package root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python run_identifiability.py --output-dir reproduced_outputs
python run_noise_robustness.py --output-dir reproduced_outputs
```

On Windows PowerShell, use:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_identifiability.py --output-dir reproduced_outputs
python run_noise_robustness.py --output-dir reproduced_outputs
```

The deterministic calculation takes approximately 15 seconds and the
perturbation calculation approximately 8 seconds on the verified CPU
environment; runtime varies with hardware. The scripts are CPU-only.

## Frozen numerical protocol

- 60 monodisperse size nodes in each orientation-quadrature database
- 181 scattering angles from 0 to 180 degrees
- 4763-point dense ensemble-integration grid
- 9999 lognormal candidates
- 11817 modified-power-law candidates
- 1000 perturbation realizations per truth/noise pair
- master random seed `20260911`
- candidate-pool sensitivity check at 700 and 1000 candidates per family

## Package contents

### Analysis code

- `analysis_core.py` -- database loading, interpolation, ensemble averaging,
  retrieval metrics, and shared utilities
- `run_identifiability.py` -- deterministic reciprocal searches, validation
  summaries, and Figs. 1--5
- `run_noise_robustness.py` -- controlled perturbation experiment, pool-size
  sensitivity analysis, and Fig. 6

### Archived input data

```text
validation_data/monodisperse_database/
  adaptive_seed_hardcase/   baseline NDGS=2 database, 60 nodes
  hardcase_ndgs3/            independent NDGS=3 check, 60 nodes
```

Each database includes `grid_manifest.csv`, archival metadata, and one run
directory per size node. Each run directory contains `angular.csv` and
`meta.json`. Paths recorded in historical manifest fields are provenance
records; the Python code resolves the supplied data using package-relative
paths.

### Reference machine-readable outputs

- `cross_family_best_fits.csv`
- `deterministic_search_audit.csv`
- `orientation_quadrature_consistency.csv`
- `validation_summary.csv`
- `noise_robustness_summary.csv`
- `noise_robustness_realizations.csv`
- `noise_pool_sensitivity.csv`

### Reference figures

Figs. 1--6 are supplied in vector PDF and 300-dpi PNG formats. Running the
scripts with `--output-dir reproduced_outputs` creates a separate regenerated
set without overwriting the supplied reference files.

### Environment and integrity

- `requirements.txt` -- pinned Python dependencies
- `ENVIRONMENT.txt` -- verified runtime details
- CHECKSUMS.sha256

On Linux or macOS, verify the extracted package with:

```bash
sha256sum -c CHECKSUMS.sha256
```

## Expected headline checks

- The representative MPL truth with `alpha=3.6` is correctly classified in
  all 1000 realizations at each 1%, 3%, and 5% perturbation level.
- Correct-family selection for the near-degenerate LN truth is 76.7% at 1%,
  51.8% at 3%, and 58.5% at 5%.
- Replacing `NDGS=2` with `NDGS=3` changes the reported joint discrepancies by
  at most approximately `1.12e-12` across the audited cases.

Small floating-point differences in the final digits may occur across Python,
library, or platform versions. The pinned environment records the versions
used to generate the reference outputs.

## Reuse

No separate software or data license is included in this archive. For reuse
beyond peer review and verification of the reported results, contact the
corresponding author at `ozalloum@ppu.edu`.
