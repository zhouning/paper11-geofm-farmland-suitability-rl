# Reproduction Guide

This guide documents the lightweight reviewer path for the standalone Paper11 repository.

## 1. Environment

Create a Python environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The core smoke check requires only Python and NumPy. Full RL training uses PyTorch, Gymnasium, Stable-Baselines3, and sb3-contrib. Embedding extraction from Google Earth Engine requires a configured Earth Engine account.

## 2. Verify Repository Integrity

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests
```

Expected outcome:

- required design, runtime, sample data, and reproducibility files exist;
- `data/bishan_alphaearth_sample/metadata.json` reports years 2017-2024 and embedding dimension 64;
- each included Bishan embedding array is readable and has final dimension 64.

## 3. Inspect the Paper11 Design

Read these files in order:

```text
paper/design/01_design_synthesis.md
paper/design/02_system_design.md
paper/design/03_experiment_plan.md
paper/design/04_manuscript_outline.md
paper/design/05_risks_and_boundaries.md
```

The design intentionally keeps Paper11 within current-state suitability representation and DRL layout optimization.

## 4. Inspect Runtime Code

Important copied runtime files:

```text
experiments/geofm_runtime/embedding_space_env.py
experiments/geofm_runtime/train_embedding_rl.py
experiments/geofm_runtime/dual_rep_env.py
experiments/geofm_runtime/train_dual_rep.py
src/legacy_runtime/county_env.py
src/legacy_runtime/parcel_scoring_policy.py
```

These scripts preserve the original Paper58/Paper8 development code. Some scripts still contain historical path assumptions from the source workspace, especially for Google Earth Engine helpers, model weights, or `data` directories. The smoke check is the stable reviewer entry point for this repository snapshot.

## 5. Regenerate Embeddings

The included Bishan arrays are cached samples from:

```text
GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
```

To regenerate or extend them, inspect:

```text
experiments/geofm_runtime/extract_bishan_embeddings.py
experiments/geofm_runtime/extract_village_embeddings.py
```

These extraction scripts require Google Earth Engine authentication and may need local path edits for the target machine.

## 6. Large Data and Weights

Large arrays, model weights, and intervention transition files are not included in ordinary Git. See `DATA_MANIFEST.md` for what was deliberately included and excluded.

For a full artifact release, place large files in Git LFS or an external archive and record checksums in the data manifest.
