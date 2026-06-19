# Phase 30 Current Progress Handoff

Last updated: 2026-06-19.

## Repository State

- Repository: `D:\test\paper11-geofm-farmland-suitability-rl`
- Branch: `main`
- Remote: `origin/main`
- Current pushed head before this handoff: `79502fe feat: add Phase 30 normalized-B1 ablation`
- Status before this handoff: `main...origin/main`

## Completed Work

Phase 29 and Phase 30 are implemented, verified, committed, and pushed.

Recent key commits:

- `386b5cf feat: add Phase 29 representation-scale diagnosis`
- `79502fe feat: add Phase 30 normalized-B1 ablation`

Phase 30 added the normalized-B1 follow-up branch:

- `N1Z`: explicit features plus column-centered and z-scored B1 embeddings
- `N1ZR`: explicit features plus z-scored and row-L2-normalized B1 embeddings

New tracked Phase 30 files already in GitHub:

- `docs/superpowers/specs/2026-06-19-phase30-normalized-b1-ablation-design.md`
- `docs/superpowers/plans/2026-06-19-phase30-normalized-b1-ablation.md`
- `src/paper11_geofm/phase30_normalized_b1_ablation.py`
- `experiments/phase30_normalized_b1_ablation/run_phase30_normalized_b1_ablation.py`
- `tests/test_phase30_normalized_b1_ablation.py`
- `paper/phase28_results/04_phase30_normalized_b1_ablation.md`

## Verified Phase 30 Result

Phase 30 was re-verified before push with:

```powershell
python -m pytest tests\test_phase30_normalized_b1_ablation.py tests\test_phase25_padded_heldout_policy.py tests\test_phase28_representation_controls.py tests\test_phase29_representation_scale_diagnosis.py -q --basetemp=.pytest_tmp_phase30_final -p no:cacheprovider
python scripts\smoke_check.py
git diff --check
```

Verification status at the time of the Phase 30 push:

- `30 passed`
- `Paper11 smoke check passed.`
- `git diff --check`: clean

## Current Scientific Position

The current bounded representation-side conclusion is:

1. raw `B1` underperforms badly in the existing held-out Bishan protocol;
2. normalization materially improves `B1`;
3. `N1Z` and `N1ZR` both recover the mean gap to `B0`;
4. normalized `B1` still trails `D4P8` and `D4P16`.

Conservative interpretation:

- Phase 29's optimization/scale hypothesis has partial support.
- Raw embedding scale is a real factor in the `B1` failure mode.
- Scale correction alone does not explain the full compressed-control advantage.

Do not claim:

- suitability reward readiness;
- `B2/B3` superiority;
- cross-region transfer;
- submission-level planning-performance success;
- that normalized `B1` is generally better than the compressed controls.

## Important Local Non-Git Artifacts

The real Phase 30 generated outputs are local-only and intentionally not committed:

```text
D:\test\paper11-geofm-farmland-suitability-rl\experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental
```

Top-level files currently present there:

- `phase30_normalized_b1_comparison.json`
- `phase30_normalized_b1_delta_table.csv`
- `phase30_normalized_b1_readiness.md`
- `phase30_normalized_b1_summary.csv`
- `phase30_normalized_b1_traces.json`
- `derived_normalized_controls/`

Current top-level file byte sum excluding the subdirectory:

- bytes: `421355`

These outputs are reproducible from the committed runner plus the required real
input data. They should not be added to ordinary Git history.

## External Transfer Folder Status

The macOS transfer folder still exists locally:

```text
D:\test\paper11_macos_transfer
```

Current visible contents:

- `DLTB_with_slope.gpkg`
- `README_TRANSFER.md`
- `TRANSFER_MANIFEST.csv`
- `experiments/`

It remains the correct folder to upload through Google Drive when a macOS
machine needs the real data and intermediate outputs.

## Recommended Next Research Step

Paper 11 is still not at manuscript-output stage. The next rigorous step should
stay conservative and bounded.

Recommended immediate continuation:

1. keep the current representation conclusion fixed as a bounded diagnostic;
2. investigate why `D4P8/D4P16` still beat `N1Z/N1ZR`;
3. prioritize case-map and action-selection diagnostics on representative
   held-out tiles;
4. in parallel, resume the blocked suitability-proxy validation path instead of
   jumping to `B2/B3` claims.

In practical terms, the next window should first read:

- `docs/superpowers/phase30_current_progress_handoff.md`
- `paper/phase28_results/04_phase30_normalized_b1_ablation.md`
- `paper/phase26_results/02_next_experiment_matrix.md`

## Suggested Resume Commands On Windows

```powershell
cd D:\test\paper11-geofm-farmland-suitability-rl
git status --short --branch
git log --oneline --max-count=8
Get-Content docs\superpowers\phase30_current_progress_handoff.md
Get-Content paper\phase28_results\04_phase30_normalized_b1_ablation.md
Get-ChildItem -Force experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental
Get-ChildItem -Force D:\test\paper11_macos_transfer
```

## Reproduction Command For The Latest Real Result

```powershell
python experiments\phase30_normalized_b1_ablation\run_phase30_normalized_b1_ablation.py --mode run-and-analyze --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --existing-control-summary-csv experiments\phase28_representation_controls\outputs\real_bishan_4096\phase28_representation_control_summary.csv --variants B0,B1,N1Z,N1ZR,D2,D3,D4P8,D4P16 --total-timesteps 4096 --eval-max-steps 8 --seeds 0,1,2 --max-eval-tiles 3 --output-dir experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental
```

## Resume Boundary

The next session should continue from the current bounded-diagnostics state.
Do not restart from early Phase 25/26 setup unless the real artifacts are lost.
