# Phase 33 Current Progress Handoff

## Current State

Paper11 remains experiment-first. Phase 33 has completed the bounded
`3 tiles x 3 seeds` matched robustness check over the Phase 30 normalized-B1
branch. The current continuation adds Phase 34: a read-only case-map diagnostic
over the completed Phase 33 matched pilot outputs.

Repository state when the Phase 34 continuation started:

- branch: `main`
- remote relation: `main...origin/main`
- latest commit before Phase 34 continuation: `6f94881 docs: record Phase 33 full-grid budget result`
- starting tracked edits: none

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

## Phase 34 Case-Map Continuation

Phase 34 completed the first next-step priority from the previous handoff. It
is read-only and uses the nine existing Phase 33 matched pilot directories; it
does not run new policy training.

Local ignored Phase 34 output:

```text
experiments/phase34_case_map_diagnostics/outputs/real_bishan_5120_phase33_9run
```

Artifacts:

```text
phase34_case_map_cases.csv
phase34_case_map_blocks.csv
phase34_case_map_diagnostics.json
phase34_case_map_diagnostics.md
```

Status and row counts:

- status: `case_map_diagnostics_ready`
- case rows: `54`
- case-map block rows: `857`
- selected Phase 2 feature rows: `294`
- Phase 33 matched directories: `9`

Main diagnostic result:

- positive Phase 33 cases: `24`, mean higher delta `0.5822555613`, mean
  base-reward gap `0.0727819453`, all classified as
  `variant_selects_higher_base_reward_blocks`
- failure Phase 33 cases: `30`, mean higher delta `-1.2055407806`, mean
  base-reward gap `-0.1506925976`, all classified as
  `variant_selects_lower_base_reward_blocks`
- `tile_r002_c003`: `15` higher / `3` lower spatial-pattern split
- `tile_r005_c003`: `0` higher / `18` lower; clearest negative spatial
  counterexample
- `tile_r005_c004`: `9` higher / `9` lower

Interpretation: Phase 34 supports the existing bounded Phase 33 conclusion and
makes the spatial split concrete. Positive cases are associated with selected
block sets that have higher deterministic base-planning reward than their
comparators; negative cases are associated with lower base-reward selected
block sets. This still does not support a general budget-rescue claim for
normalized B1.

## Tracked Docs Updated

Updated in the Phase 33 window:

```text
README.md
paper/phase26_results/02_next_experiment_matrix.md
paper/phase28_results/07_phase33_budget_robustness.md
docs/superpowers/phase33_current_progress_handoff.md
```

Updated in the Phase 34 continuation:

```text
README.md
paper/phase28_results/README.md
paper/phase28_results/08_phase34_case_map_diagnostics.md
reproducibility/FILE_MANIFEST.tsv
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

1. Extend Phase 31/32 action-overlap diagnostics to the new Phase 33 `5120`
   outputs, especially the contrast between `tile_r002_c003` positive flips and
   `tile_r005_c003` negative/stable failures.
2. Resume suitability-proxy validation before any B2/B3 reward integration.
3. Use the Phase 34 case-map outputs as spatial diagnostic support only; do not
   convert them into manuscript-level performance claims.
4. Treat `8192` only as a later, chunked/resumable experiment if the Phase 34
   and action-overlap diagnostics show a reason to test longer budgets. The
   previous `8192` attempts did not finish within the execution window and are
   not evidence.

## Useful Commands For Next Window

```powershell
cd D:\test\paper11-geofm-farmland-suitability-rl
git status --short --branch
Get-Content docs\superpowers\phase33_current_progress_handoff.md
Get-Content paper\phase28_results\07_phase33_budget_robustness.md
Get-Content paper\phase28_results\08_phase34_case_map_diagnostics.md
Get-ChildItem -Force experiments\phase34_case_map_diagnostics\outputs\real_bishan_5120_phase33_9run
Import-Csv experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_3tiles_3seeds_9run_aggregate\phase33_focal_gap_transition.csv | Format-Table -AutoSize
Import-Csv experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_3tiles_3seeds_9run_aggregate\phase33_tile_seed_stability.csv | Group-Object stability_class | Select-Object Name,Count
Import-Csv experiments\phase34_case_map_diagnostics\outputs\real_bishan_5120_phase33_9run\phase34_case_map_cases.csv | Group-Object eval_tile_id,spatial_pattern | Select-Object Name,Count
python -m pytest tests\test_phase34_case_map_diagnostics.py tests\test_phase33_budget_robustness.py tests\test_phase32_action_order_diagnostics.py tests\test_phase30_normalized_b1_ablation.py -q --basetemp=.pytest_tmp_phase34_resume -p no:cacheprovider
python scripts\smoke_check.py
```
