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

## 3. Run the Phase 1 Bishan Baseline

Run:

```powershell
python experiments\phase1_bishan_baseline\run_phase1.py
```

Expected outcome:

- `experiments/phase1_bishan_baseline/outputs/region_features.csv` is created;
- `experiments/phase1_bishan_baseline/outputs/summary.json` is created;
- the summary reports 25 deterministic grid regions by default;
- `claim_boundary` states that `suitability_proxy` is derived from latent remote-sensing embeddings and does not directly measure soil, fertility, or irrigation.

The default outputs are ignored by Git. Use a custom output directory for controlled verification:

```powershell
python experiments\phase1_bishan_baseline\run_phase1.py --output-dir D:\tmp\paper11_phase1_outputs
```

## 4. Run the Phase 2 Block Feature Assembly Baseline

Run:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py
```

Expected outcome:

- `experiments/phase2_block_geofm_features/outputs/block_geofm_features.csv` is created;
- `experiments/phase2_block_geofm_features/outputs/summary.json` is created;
- the summary reports 25 generated grid-derived blocks by default;
- `feature_readiness` reports B0/B1/B2/B3 readiness and marks explicit-feature-dependent variants incomplete when explicit planning features are absent;
- `claim_boundary` keeps the same remote-sensing proxy boundary used in Phase 1.

The default Phase 2 path derives a deterministic block-to-pixel mapping from the included Bishan sample. It is a lightweight feature-assembly smoke test, not real block-level DRL evidence and not a substitute for parcel/block geometry, explicit planning features, or weak-label validation.

Use a custom output directory for controlled verification:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --output-dir D:\tmp\paper11_phase2_outputs
```

To run Phase 2 with real planning units, provide a mapping table:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv D:\tmp\block_pixel_mapping.csv --output-dir D:\tmp\paper11_phase2_outputs
```

The mapping CSV schema is:

```text
block_id,row,col,weight
```

`weight` is optional. If omitted, every mapping row receives weight `1.0`.

To join explicit block features or weak labels, add an attributes CSV:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv D:\tmp\block_pixel_mapping.csv --attributes-csv D:\tmp\block_attributes.csv --output-dir D:\tmp\paper11_phase2_outputs
```

The attributes CSV must be keyed by `block_id`. For B0/B1/B2/B3 readiness, include all columns from `explicit_feature_00` through `explicit_feature_16`. Optional weak-label columns include `stable_farmland_label` and `high_standard_farmland_label`; optional split metadata can use `split`.

When weak-label columns are present, Phase 2 also writes `weak_label_validation.json`. This file is a diagnostic proxy check that compares `suitability_proxy` distributions against the available weak labels; it is not proof of agronomic validity or direct measurement of soil quality, fertility, or irrigation access.

The repository includes a tiny CSV fixture for this path:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir D:\tmp\paper11_phase2_csv_fixture_outputs
```

Expected outcome:

- the summary reports `mapping_mode` as `mapping_csv`;
- `n_blocks` is `4`;
- B3 readiness is `true` because the fixture includes all 17 explicit feature columns plus GeoFM embeddings and `suitability_proxy`;
- `weak_label_validation.json` is created because the fixture includes `stable_farmland_label` and `high_standard_farmland_label`.

## 5. Inspect the Paper11 Design

Read these files in order:

```text
paper/design/01_design_synthesis.md
paper/design/02_system_design.md
paper/design/03_experiment_plan.md
paper/design/04_manuscript_outline.md
paper/design/05_risks_and_boundaries.md
```

The design intentionally keeps Paper11 within current-state suitability representation and DRL layout optimization.

## 6. Inspect Runtime Code

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

Phase 1 executable files:

```text
experiments/phase1_bishan_baseline/run_phase1.py
src/paper11_geofm/
```

Phase 2 executable files:

```text
experiments/phase2_block_geofm_features/run_phase2.py
src/paper11_geofm/block_mapping.py
src/paper11_geofm/block_features.py
src/paper11_geofm/block_schema.py
```

The Phase 1 and Phase 2 baselines are deterministic and do not require internet, GPU, Earth Engine, or full DRL training.

## 7. Regenerate Embeddings

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

## 8. Large Data and Weights

Large arrays, model weights, and intervention transition files are not included in ordinary Git. See `DATA_MANIFEST.md` for what was deliberately included and excluded.

For a full artifact release, place large files in Git LFS or an external archive and record checksums in the data manifest.
