# Phase 27 Current Progress Handoff

Last updated: 2026-06-17.

## Repository State

- Repository: `D:\test\paper11-geofm-farmland-suitability-rl`
- Branch: `main`
- Remote: `origin/main`
- Current pushed head before this handoff: `1ee4fac docs: add macOS transfer guide`
- Status before this handoff: `main...origin/main`

## Completed Work

The macOS 128GB transfer path for the Paper11 Phase 25/26 real empirical run
has been prepared.

Committed and pushed in `1ee4fac`:

- `reproducibility/MACOS_128GB_TRANSFER_GUIDE.md`
- updated `reproducibility/FILE_MANIFEST.tsv`
- updated `.gitignore` to ignore `.pytest_tmp*`

The macOS guide records that the 128GB Mac can run the current Phase 25/26
route because Phase 25 uses `MaskablePPO` with `device="cpu"`. Colab Pro+
remains a backup for long wall-clock runs, but it is not mandatory for the
current B0/B1 main run.

## External Transfer Folder

Large/generated files that should not go through ordinary Git were organized in:

```text
D:\test\paper11_macos_transfer
```

Contents:

- `README_TRANSFER.md`
- `TRANSFER_MANIFEST.csv`
- `DLTB_with_slope.gpkg`
- `experiments/phase11_bishan_dltb_real/outputs/`
- `experiments/phase13_tiled_real_contract/outputs/`

Payload summary, excluding `README_TRANSFER.md` and `TRANSFER_MANIFEST.csv`:

- payload files: `15`
- payload bytes: `481,836,291`

Key payload paths:

```text
D:\test\paper11_macos_transfer\experiments\phase11_bishan_dltb_real\outputs\phase2_real
D:\test\paper11_macos_transfer\experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv
```

`DLTB_with_slope.gpkg` is included for completeness and for possible Phase 11
regeneration. Phase 25/26 can run from the copied Phase 11 and Phase 13 outputs
without re-reading the GeoPackage.

## What The User Should Do Next

Upload the whole folder to Google Drive:

```text
D:\test\paper11_macos_transfer
```

On macOS:

```bash
git clone <repo-url>
cd paper11-geofm-farmland-suitability-rl
git status --short --branch
rsync -av /path/to/paper11_macos_transfer/experiments/ ./experiments/
```

Optional raw GeoPackage placement outside the cloned repository:

```bash
mkdir -p ../paper11_local_data
cp /path/to/paper11_macos_transfer/DLTB_with_slope.gpkg ../paper11_local_data/DLTB_with_slope.gpkg
```

Then follow:

```text
reproducibility/MACOS_128GB_TRANSFER_GUIDE.md
```

## First macOS Verification Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/smoke_check.py
python -m pytest tests/test_phase25_padded_heldout_policy.py -q
python -m pytest tests/test_phase26_main_experiment.py -q
```

Minimal data path checks:

```bash
test -f experiments/phase11_bishan_dltb_real/outputs/phase2_real/variant_B0_features.csv
test -f experiments/phase11_bishan_dltb_real/outputs/phase2_real/variant_B1_features.csv
test -f experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv
```

## First macOS Timing Probe

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

## Main Empirical Run Boundary

The next real empirical run remains restricted to:

- B0/B1 only;
- deterministic `base_planning_reward`;
- held-out Bishan tiles;
- Phase 26 analysis of Phase 25 outputs.

Do not enable B2/B3 or suitability reward yet. Phase 10 still blocks the
suitability reward route, so B2/B3 are outside the current main empirical claim.

## Suggested Resume Commands On Windows

```powershell
cd D:\test\paper11-geofm-farmland-suitability-rl
git status --short --branch
git log --oneline --max-count=5
Get-Content docs\superpowers\phase27_current_progress_handoff.md
Get-ChildItem -Force D:\test\paper11_macos_transfer
```

If the user has uploaded the transfer folder and moved to macOS, continue from
`reproducibility/MACOS_128GB_TRANSFER_GUIDE.md`.
