# Paper11: GeoFM-Enhanced Farmland Suitability RL

This repository is the standalone reviewer package for Paper11:

**GeoFM-enhanced current-state farmland suitability representation and DRL spatial layout optimization.**

The repository separates Paper11 from the broader Paper58 workspace so reviewers can inspect the design, copied runtime code, lightweight sample data, and reproduction entry points without needing the original experiment directory.

## Scope Boundary

Paper11 uses AlphaEarth or other GeoFM embeddings as latent remote-sensing proxies for land-surface and environmental conditions related to farmland suitability. It does not claim that AlphaEarth directly measures soil quality, fertility, or irrigation access.

Paper11 is also distinct from future-aware planning work. The target framing is current-state farmland suitability representation plus reinforcement-learning layout optimization, not prediction-then-optimization.

## Repository Layout

- `paper/design/`: Paper11 design package, including system design, experiment plan, manuscript outline, and risk boundaries.
- `paper/phase1_results/`: interpretation of the executable Phase 1 baseline and the next experiment matrix.
- `docs/source_notes/`: original design notes used to derive the Paper11 package.
- `experiments/geofm_runtime/`: copied GeoFM and embedding-space experiment scripts from the source Paper58 workspace.
- `experiments/phase1_bishan_baseline/`: executable Phase 1 Bishan GeoFM representation baseline.
- `experiments/phase2_block_geofm_features/`: executable Phase 2 block-level GeoFM feature assembly baseline.
- `src/paper11_geofm/`: focused utilities for sample loading, deterministic region aggregation, block feature assembly, suitability proxy scoring, and artifact export.
- `src/legacy_runtime/`: copied legacy county/block RL runtime files imported by the experiment scripts.
- `data/bishan_alphaearth_sample/`: lightweight Bishan AlphaEarth embedding sample for smoke tests and reviewer inspection.
- `reproducibility/`: reproduction guide, data manifest, and file manifest.
- `scripts/smoke_check.py`: verifies repository layout and sample-data readability.
- `tests/`: pytest checks for layout and smoke-check behavior.

## Quick Start

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the lightweight checks:

```powershell
python scripts\smoke_check.py
python -m pytest tests
```

The smoke check reads the included Bishan sample arrays and verifies metadata. It does not run training, contact Google Earth Engine, or require GPU access.

Run the Phase 1 Bishan GeoFM representation baseline:

```powershell
python experiments\phase1_bishan_baseline\run_phase1.py
```

The command writes `region_features.csv` and `summary.json` under `experiments/phase1_bishan_baseline/outputs/`. These outputs are generated artifacts and are ignored by Git.

Run the Phase 2 block-level GeoFM feature assembly baseline:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py
```

The default Phase 2 path uses a generated grid-derived block-to-pixel mapping from the included Bishan sample. It writes `block_geofm_features.csv`, `summary.json`, and `experiment_variants.json` under `experiments/phase2_block_geofm_features/outputs/`. The variant manifest defines B0/B1/B2/B3 feature-table contracts for later DRL experiments; it is not trained-policy evidence. Because the generated-grid path does not include explicit planning features, the default manifest leaves variant `feature_table` values unset and does not create `variant_B*_features.csv` files. This is a feature-assembly smoke test, not real block-level DRL evidence.

To use real planning units, pass a block-to-pixel mapping CSV:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv path\to\block_pixel_mapping.csv
```

The mapping CSV must include `block_id`, `row`, and `col`, with optional `weight`. Optional block attributes can be joined by `block_id`:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv path\to\block_pixel_mapping.csv --attributes-csv path\to\block_attributes.csv
```

The attributes CSV can include `explicit_feature_00` through `explicit_feature_16`, weak labels such as `stable_farmland_label`, and split labels such as `split`. When a variant has all required columns, Phase 2 also writes a ready-only variant table: `variant_B0_features.csv`, `variant_B1_features.csv`, `variant_B2_features.csv`, or `variant_B3_features.csv`. B0 contains explicit planning features, B1 adds GeoFM embedding means, B2 adds the latent suitability proxy, and B3 combines explicit planning features, GeoFM embedding means, and the suitability proxy. When weak-label columns are present, the runner also writes `weak_label_validation.json` as a diagnostic proxy check comparing `suitability_proxy` against available weak labels.

This repository includes a tiny CSV fixture for checking that path directly:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv
```

The included fixture has the required explicit feature columns, so it creates `variant_B0_features.csv` through `variant_B3_features.csv` and records those filenames in `experiment_variants.json`.

## Key Entry Points

- Design synthesis: `paper/design/01_design_synthesis.md`
- System design: `paper/design/02_system_design.md`
- Experiment plan: `paper/design/03_experiment_plan.md`
- Phase 1 result interpretation: `paper/phase1_results/01_phase1_result_interpretation.md`
- Next experiment matrix: `paper/phase1_results/02_next_experiment_matrix.md`
- Main embedding environment: `experiments/geofm_runtime/embedding_space_env.py`
- Phase 1 Bishan baseline runner: `experiments/phase1_bishan_baseline/run_phase1.py`
- Phase 2 block feature assembly runner: `experiments/phase2_block_geofm_features/run_phase2.py`
- Phase 1 utility package: `src/paper11_geofm/`
- Embedding RL training script: `experiments/geofm_runtime/train_embedding_rl.py`
- Dual-representation environment: `experiments/geofm_runtime/dual_rep_env.py`
- Legacy county environment: `src/legacy_runtime/county_env.py`

## Data Policy

This repository includes only the lightweight Bishan sample needed for reviewer smoke tests. Larger village and Heping embedding arrays, model checkpoints, and intervention transition files are documented in `reproducibility/DATA_MANIFEST.md` but are not included in ordinary Git.

Use Git LFS or an external archival repository such as Zenodo or OSF before distributing large derived arrays or trained weights.

## Recommended Journal Framing

The strongest Paper11 framing is for applied remote sensing and geospatial optimization venues: GeoFM representations improve current-state farmland suitability representation and DRL spatial planning, with careful claim boundaries around what remote-sensing embeddings can and cannot measure.
