# macOS 128GB Transfer and Main-Run Guide

This guide documents how to move the non-Git large Paper11 real-data artifacts
to a macOS 128GB machine and run the Phase 25/26 empirical path.

## Platform Decision

The macOS 128GB machine can run the current Paper11 Phase 25/26 route.

Reason: Phase 25 uses `MaskablePPO` with `device="cpu"` in
`src/paper11_geofm/padded_heldout_policy.py`. CUDA is not required by the
current implementation. Colab Pro+ remains a useful backup for long wall-clock
runs, but it is not mandatory for the current B0/B1 main run.

## What GitHub Contains

GitHub contains code, tests, lightweight sample data, and documentation.

It deliberately does not contain:

- `*.gpkg` real geospatial sources;
- generated `experiments/**/outputs/`;
- model weights;
- large arrays or main-run result folders.

Those files should be transferred through Google Drive or a proper data
archive, not ordinary Git.

## Transfer Bundle Created On Windows

The Windows workstation has a prepared transfer folder:

```text
D:\test\paper11_macos_transfer
```

Recommended Google Drive upload: upload the whole folder as
`paper11_macos_transfer`.

It contains:

```text
paper11_macos_transfer/
  README_TRANSFER.md
  TRANSFER_MANIFEST.csv
  DLTB_with_slope.gpkg
  experiments/
    phase11_bishan_dltb_real/
      outputs/
        adapter/
        phase2_real/
        phase9_real/
        phase10_real/
    phase13_tiled_real_contract/
      outputs/
        real_bishan/
```

Current payload summary, excluding transfer helper text files:

| Item | Value |
|---|---:|
| Payload file count | 15 |
| Payload bytes | 481,836,291 |
| Largest raw source | `DLTB_with_slope.gpkg` |
| Required for Phase 25/26 | `experiments/phase11.../phase2_real/` and `experiments/phase13.../phase13_tile_index.csv` |

`DLTB_with_slope.gpkg` is needed only if regenerating Phase 11 from the raw
GeoPackage. For Phase 25/26, the copied generated outputs are enough.

## Restore On macOS

Clone or pull the repository:

```bash
git clone <repo-url>
cd paper11-geofm-farmland-suitability-rl
git status --short --branch
```

Download `paper11_macos_transfer` from Google Drive, then merge its experiment
outputs into the repository root:

```bash
rsync -av /path/to/paper11_macos_transfer/experiments/ ./experiments/
```

Optionally keep the raw GeoPackage outside the cloned Git repository:

```bash
mkdir -p ../paper11_local_data
cp /path/to/paper11_macos_transfer/DLTB_with_slope.gpkg ../paper11_local_data/DLTB_with_slope.gpkg
```

Minimal path checks:

```bash
test -f experiments/phase11_bishan_dltb_real/outputs/phase2_real/variant_B0_features.csv
test -f experiments/phase11_bishan_dltb_real/outputs/phase2_real/variant_B1_features.csv
test -f experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv
```

Optional checksum check:

```bash
shasum -a 256 experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv
```

Compare against `TRANSFER_MANIFEST.csv` in the Google Drive transfer bundle.

## Install Dependencies On macOS

Use a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run repository checks:

```bash
python scripts/smoke_check.py
python -m pytest tests/test_phase25_padded_heldout_policy.py -q
python -m pytest tests/test_phase26_main_experiment.py -q
```

## First Timing Probe

Before a longer main run, run a short Phase 26 `run-and-analyze` probe:

```bash
python experiments/phase26_main_experiment/run_phase26_main_experiment.py \
  --mode run-and-analyze \
  --phase25-output-dir experiments/phase26_main_experiment/outputs/macos_timing_probe/phase25_run \
  --output-dir experiments/phase26_main_experiment/outputs/macos_timing_probe/phase26_analysis \
  --phase2-output-dir experiments/phase11_bishan_dltb_real/outputs/phase2_real \
  --tile-index-csv experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv \
  --variants B0,B1 \
  --total-timesteps 128 \
  --eval-max-steps 4 \
  --seeds 0 \
  --max-eval-tiles 1
```

This checks dependency compatibility, file paths, and wall-clock cost on the Mac.

## Main Phase 25 Run

If the probe is stable, run the B0/B1 main experiment:

```bash
python experiments/phase25_padded_heldout_policy/run_phase25_padded_heldout_policy.py \
  --phase2-output-dir experiments/phase11_bishan_dltb_real/outputs/phase2_real \
  --tile-index-csv experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv \
  --variants B0,B1 \
  --total-timesteps 1024 \
  --eval-max-steps 8 \
  --seeds 0,1,2 \
  --max-eval-tiles 3 \
  --output-dir experiments/phase26_main_experiment/outputs/macos_main/phase25_run
```

If runtime is acceptable, repeat with `--total-timesteps 4096` under a separate
output directory, for example:

```text
experiments/phase26_main_experiment/outputs/macos_main_4096/phase25_run
```

## Phase 26 Analysis

Analyze the Phase 25 outputs:

```bash
python experiments/phase26_main_experiment/run_phase26_main_experiment.py \
  --mode analyze-only \
  --phase25-output-dir experiments/phase26_main_experiment/outputs/macos_main/phase25_run \
  --output-dir experiments/phase26_main_experiment/outputs/macos_main/phase26_analysis
```

Expected analysis artifacts:

- `phase26_main_summary.csv`
- `phase26_tile_seed_delta_table.csv`
- `phase26_main_comparison.json`
- `phase26_claim_readiness.md`

## Claim Boundaries

Keep this main empirical run restricted to:

- B0/B1 only;
- deterministic `base_planning_reward`;
- held-out Bishan tiles;
- Phase 26 analysis of Phase 25 outputs.

Do not enable B2/B3 or suitability reward in this run. Phase 10 currently keeps
the suitability reward blocked, so B2/B3 remain outside the current main
empirical claim.
