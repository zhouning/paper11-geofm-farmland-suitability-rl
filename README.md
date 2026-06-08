# Paper11: GeoFM-Enhanced Farmland Suitability RL

This repository is the standalone reviewer package for Paper11:

**GeoFM-enhanced current-state farmland suitability representation and DRL spatial layout optimization.**

The repository separates Paper11 from the broader Paper58 workspace so reviewers can inspect the design, copied runtime code, lightweight sample data, and reproduction entry points without needing the original experiment directory.

## Scope Boundary

Paper11 uses AlphaEarth or other GeoFM embeddings as latent remote-sensing proxies for land-surface and environmental conditions related to farmland suitability. It does not claim that AlphaEarth directly measures soil quality, fertility, or irrigation access.

Paper11 is also distinct from future-aware planning work. The target framing is current-state farmland suitability representation plus reinforcement-learning layout optimization, not prediction-then-optimization.

## Repository Layout

- `paper/design/`: Paper11 design package, including system design, experiment plan, manuscript outline, and risk boundaries.
- `docs/source_notes/`: original design notes used to derive the Paper11 package.
- `experiments/geofm_runtime/`: copied GeoFM and embedding-space experiment scripts from the source Paper58 workspace.
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

## Key Entry Points

- Design synthesis: `paper/design/01_design_synthesis.md`
- System design: `paper/design/02_system_design.md`
- Experiment plan: `paper/design/03_experiment_plan.md`
- Main embedding environment: `experiments/geofm_runtime/embedding_space_env.py`
- Embedding RL training script: `experiments/geofm_runtime/train_embedding_rl.py`
- Dual-representation environment: `experiments/geofm_runtime/dual_rep_env.py`
- Legacy county environment: `src/legacy_runtime/county_env.py`

## Data Policy

This repository includes only the lightweight Bishan sample needed for reviewer smoke tests. Larger village and Heping embedding arrays, model checkpoints, and intervention transition files are documented in `reproducibility/DATA_MANIFEST.md` but are not included in ordinary Git.

Use Git LFS or an external archival repository such as Zenodo or OSF before distributing large derived arrays or trained weights.

## Recommended Journal Framing

The strongest Paper11 framing is for applied remote sensing and geospatial optimization venues: GeoFM representations improve current-state farmland suitability representation and DRL spatial planning, with careful claim boundaries around what remote-sensing embeddings can and cannot measure.
