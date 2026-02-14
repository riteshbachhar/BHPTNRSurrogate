# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BHPTNRSurrogate is a Python package providing gravitational waveform surrogate models built on point-particle black hole perturbation theory (ppBHPT), calibrated to numerical relativity. It contains two models:

- **BHPTNRSur1dq1e4**: Non-spinning, 1D parameter space (mass ratio q ∈ [2.5, 10000]), 50 modes up to ℓ=10, uses spline fitting
- **BHPTNRSur2dq1e3**: Spinning, 2D parameter space (q ∈ [3, 1000], χ₁ ∈ [-0.8, 0.8]), 16 modes up to ℓ=4, uses Gaussian Process Regression (GPR)

## Build & Development Commands

```bash
# Install in development mode
pip install -e .

# Install with dev dependencies (pytest)
pip install -e ".[dev]"

# Run tests
pytest

# Run a single test
pytest tests/test_foo.py::test_name -v

# Download data files (required, ~1.7GB total)
wget -P data/ https://zenodo.org/records/13340319/BHPTNRSur1dq1e4.h5
wget -P data/ https://zenodo.org/records/13340319/BHPTNRSur2dq1e3.h5
```

No linter or formatter is currently configured.

## Architecture

### Waveform Generation Pipeline

User calls `generate_surrogate(q, spin1, modes, ...)` on a model wrapper → **input validation** (`check_inputs.py`) → **load HDF5 data** (`load_surrogates.py`) → **evaluate fits at EIM nodes** (`fits.py`) → **reconstruct waveform via basis matrices** (h = B.T · fit_values) → **NR calibration** (alpha-beta scaling, `nr_calibration.py`) → **post-processing** (frame transforms, unit conversion, `utils.py`)

### Key Structural Pattern

The two models share an abstract interface but differ in fitting method:
- `load_splines.py` → spline interpolation for 1D model
- `load_GPRs.py` → scikit-learn GPR for 2D model
- `eval_pysur/evaluate_fit.py` → vendored GPR predictor (from pySurrogate)

Both are unified through `fits.py` which dispatches to the appropriate evaluator, and `eval_surrogates.py` which orchestrates the full pipeline.

### Module Layout

- `bhptnrsurrogate/BHPTNRSur1dq1e4.py`, `BHPTNRSur2dq1e3.py` — thin model wrappers exposing `generate_surrogate()`
- `bhptnrsurrogate/model_utils/` — loading HDF5 data and evaluating the surrogate pipeline
- `bhptnrsurrogate/common_utils/` — shared code: fitting, calibration, waveform processing, input validation
- `data/` — HDF5 surrogate data files (not committed, downloaded from Zenodo)
- `tutorials/` — Jupyter notebooks demonstrating each model

### Data Structures

- Waveforms are dictionaries keyed by `(\ell, m)` tuples with complex numpy array values
- Fit data is packed into `fit_data_dict_1` / `fit_data_dict_2` abstractions (spline coefficients or GPR models)
- Basis matrices `B_dict_1`, `B_dict_2` store the EIM decomposition per mode

## Conventions

- All imports are relative (e.g., `from ..common_utils import utils`)
- Private functions prefixed with underscore (e.g., `_evaluate_splines_at_EIM_nodes`)
- `@docs.copy_doc` decorator combines model-specific and generic docstrings
- Data files are auto-downloaded from Zenodo on first import if missing, verified via MD5 hash
- Python ≥ 3.9 required; tested on 3.9–3.12
