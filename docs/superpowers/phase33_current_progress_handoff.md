# Phase 33 Current Progress Handoff

## Current State

Paper11 remains experiment-first. The newest Phase 33 work extends the earlier
single-tile `5120` pilot into a bounded `3 tiles x 3 seeds` matched robustness
check over the Phase 30 normalized-B1 branch.

Repository state when this handoff was written:

- branch: `main`
- remote relation before committing this handoff: `main...origin/main [ahead 3]`
- latest committed code before this handoff: `91b8afc feat: add Phase 33 budget robustness experiment`
- current tracked edits: README, Phase 26 next matrix, Phase 33 result note,
  and this handoff

## What Was Run This Window

The earlier Phase 33 state only had a positive local pilot:

- `tile_r002_c003`, `seed=0`, `5120` train timesteps
- local status: `budget_closes_compressed_gap`

This window completed the remaining bounded coverage for the current Phase 30
held-out set:

- tiles: `tile_r002_c003`, `tile_r005_c003`, `tile_r005_c004`
- seeds: `0,1,2`
- train timesteps: `5120`
- eval max steps: `8`
- trained variants: `N1Z`, `N1ZR`
- reused matched comparators: `B1`, `D4P8`, `D4P16` from the existing `4096`
  Phase 28/30 artifacts

The single-run outputs are under:

```text
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r002_c003_seed0_matched
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r002_c003_seed1_matched
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r002_c003_seed2_matched
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c003_seed0_matched
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c003_seed1_matched
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c003_seed2_matched
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c004_seed0_matched
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c004_seed1_matched
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c004_seed2_matched
```

The aggregate outputs are under:

```text
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r002_c003_3seed_aggregate
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c003_3seed_aggregate
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c004_3seed_aggregate
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tiles_r002c003_r005c003_6run_aggregate
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_3tiles_3seeds_9run_aggregate
```

## Main Result

The current full bounded Phase 33 status is:

```text
budget_not_explanatory
```

The full `3 tiles x 3 seeds` aggregate is:

```text
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_3tiles_3seeds_9run_aggregate/phase33_budget_robustness.json
```

Key focal aggregate deltas:

| Gap | 4096 mean delta | 5120 mean delta | 5120 positive count |
|---|---:|---:|---:|
| `N1Z - B1` | `0.3008963657` | `-0.1185345091` | `5 / 9` |
| `N1Z - D4P16` | `-0.3402976578` | `-0.7597285327` | `4 / 9` |
| `N1Z - D4P8` | `-0.0759554690` | `-0.4953863438` | `3 / 9` |
| `N1ZR - B1` | `0.2266106233` | `-0.0246975093` | `5 / 9` |
| `N1ZR - D4P16` | `-0.4145834002` | `-0.6658915329` | `3 / 9` |
| `N1ZR - D4P8` | `-0.1502412114` | `-0.4015493440` | `4 / 9` |

Tilewise aggregate statuses:

- `tile_r002_c003`: `budget_closes_compressed_gap`
- `tile_r005_c003`: `budget_not_explanatory`
- `tile_r005_c004`: `budget_not_explanatory`

Interpretation: the positive `tile_r002_c003 seed0` pilot was real but local.
It does not support a general budget-rescue claim for normalized B1. The
broader bounded result weakens the Phase 29 optimization-budget explanation and
pushes the next experimental priority toward spatial/action diagnostics and
suitability-proxy validation.

## Tracked Docs Updated

Updated this window:

```text
README.md
paper/phase26_results/02_next_experiment_matrix.md
paper/phase28_results/07_phase33_budget_robustness.md
docs/superpowers/phase33_current_progress_handoff.md
```

Do not describe Phase 33 as generally `budget_closes_compressed_gap`. That is
only true for the `tile_r002_c003` three-seed aggregate, not for the complete
bounded aggregate.

## Verification Run

Fresh verification completed after the full 9-run aggregate:

```powershell
python -m pytest tests\test_phase33_budget_robustness.py tests\test_phase32_action_order_diagnostics.py tests\test_phase30_normalized_b1_ablation.py -q --basetemp=.pytest_tmp_phase33_full_cover -p no:cacheprovider
```

Result:

```text
13 passed in 0.61s
```

Smoke check:

```powershell
python scripts\smoke_check.py
```

Result:

```text
Paper11 smoke check passed.
Sample years: [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
Embedding shape: (67, 70, 64)
```

## Next Experimental Priority

Do not move directly to manuscript claims. Next work should stay experiment-first:

1. Build spatial case maps for `tile_r002_c003` versus `tile_r005_c003` at
   matched seeds, focusing on selected blocks from `N1Z`, `N1ZR`, `D4P8`, and
   `D4P16`.
2. Extend Phase 31/32 action-overlap diagnostics to the new Phase 33 `5120`
   outputs, especially the contrast between `tile_r002_c003` positive flips and
   `tile_r005_c003` negative/stable failures.
3. Resume suitability-proxy validation before any B2/B3 reward integration.
4. Treat `8192` only as a later, chunked/resumable experiment if the case-map
   diagnostics show a reason to test longer budgets. The previous `8192`
   attempts did not finish within the execution window and are not evidence.

## Useful Commands For Next Window

```powershell
cd D:\test\paper11-geofm-farmland-suitability-rl
git status --short --branch
Get-Content docs\superpowers\phase33_current_progress_handoff.md
Get-Content paper\phase28_results\07_phase33_budget_robustness.md
Import-Csv experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_3tiles_3seeds_9run_aggregate\phase33_focal_gap_transition.csv | Format-Table -AutoSize
Import-Csv experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_3tiles_3seeds_9run_aggregate\phase33_tile_seed_stability.csv | Group-Object stability_class | Select-Object Name,Count
python -m pytest tests\test_phase33_budget_robustness.py tests\test_phase32_action_order_diagnostics.py tests\test_phase30_normalized_b1_ablation.py -q --basetemp=.pytest_tmp_phase33_resume -p no:cacheprovider
python scripts\smoke_check.py
```
