# Phase 35 Phase 33 Action-Overlap Diagnostics

## One-Sentence Argument

Phase 35 is a read-only action-overlap follow-up to Phase 33 and Phase 34: it
tests whether the completed `5120` matched normalized-B1 cases differ from
their matched comparators mainly by action order within similar selected block
sets or by selecting almost different block sets.

## Current Experiment Snapshot

Phase 35 reads the same nine completed Phase 33 matched pilot directories used
by Phase 34:

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

It writes the local ignored output:

```text
experiments/phase35_phase33_action_overlap_diagnostics/outputs/real_bishan_5120_phase33_9run
```

The output contains:

```text
phase35_action_overlap_cases.csv
phase35_action_overlap_steps.csv
phase35_action_overlap_diagnostics.json
phase35_action_overlap_diagnostics.md
```

The real run status is:

```text
action_overlap_diagnostics_ready
```

Row counts:

- case rows: `54`
- step rows: `432`
- Phase 33 matched directories: `9`

## Main Result

Phase 35 strengthens the Phase 34 interpretation. The `5120` Phase 33 cases are
not mainly "same selected blocks, different order" cases. They are almost
entirely different selected-block sets.

| Action-overlap pattern | Count |
|---|---:|
| `disjoint_positive_gap` | `20` |
| `partial_overlap_positive_gap` | `4` |
| `disjoint_negative_gap` | `27` |
| `partial_overlap_negative_gap` | `3` |

By case role:

| Group | Count | Mean selected-block Jaccard | Mean summary reward gap | Nonzero-overlap cases |
|---|---:|---:|---:|---:|
| Phase 33 positive cases | `24` | `0.0111111111` | `0.5822555613` | `4` |
| Phase 33 failure cases | `30` | `0.0066666667` | `-1.2055407806` | `3` |

By tile:

| Tile | Case rows | Mean selected-block Jaccard | Mean summary reward gap | Nonzero-overlap cases |
|---|---:|---:|---:|---:|
| `tile_r002_c003` | `18` | `0.0074074074` | `0.4638019194` | `2` |
| `tile_r005_c003` | `18` | `0.0074074074` | `-1.3403609350` | `2` |
| `tile_r005_c004` | `18` | `0.0111111111` | `-0.3563348703` | `3` |

By stability class:

| Stability class | Count | Mean selected-block Jaccard | Mean summary reward gap | Nonzero-overlap cases |
|---|---:|---:|---:|---:|
| `stable_positive` | `11` | `0.0242424243` | `0.7219712548` | `4` |
| `flip_to_positive` | `13` | `0.0000000000` | `0.4640345899` | `0` |
| `stable_negative` | `18` | `0.0037037037` | `-1.0630149499` | `0` |
| `flip_to_negative` | `12` | `0.0111111111` | `-1.4193295266` | `2` |

Interpretation: Phase 35 supports the bounded negative Phase 33 conclusion and
the Phase 34 spatial-composition diagnosis. Positive and negative outcomes are
associated with selecting nearly disjoint block sets from their comparators,
not with small action-order perturbations over a shared selected set. The
clearest negative counterexample remains `tile_r005_c003`, where the mean
summary gap is `-1.3403609350` and the mean selected-block Jaccard is only
`0.0074074074`.

## Trace Coverage Boundary

For the current Phase 33 matched pilot artifacts, `N1Z` and `N1ZR` have
high-budget trace steps, but matched comparators `B1`, `D4P8`, and `D4P16` are
available through summary `selected_block_ids` rather than complete comparator
trace rewards. Phase 35 therefore uses trace order when present and falls back
to summary selected-block order otherwise. This supports selected-block overlap
and summary-gap diagnosis, but it does not support a full comparator step-reward
trajectory claim.

## Reproduction Command

```text
python experiments/phase35_phase33_action_overlap_diagnostics/run_phase35_phase33_action_overlap_diagnostics.py --phase33-output-dirs experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r002_c003_seed0_matched experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r002_c003_seed1_matched experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r002_c003_seed2_matched experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c003_seed0_matched experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c003_seed1_matched experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c003_seed2_matched experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c004_seed0_matched experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c004_seed1_matched experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c004_seed2_matched --output-dir experiments/phase35_phase33_action_overlap_diagnostics/outputs/real_bishan_5120_phase33_9run --variants N1Z,N1ZR --comparators B1,D4P8,D4P16
```

## Claim Boundary

Phase 35 is a read-only action-overlap diagnostic over existing Phase 33
matched pilot artifacts. It does not run new policy training, does not alter
rewards, does not enable suitability reward, does not test B2/B3, and does not
support final submission-level planning-performance claims.
