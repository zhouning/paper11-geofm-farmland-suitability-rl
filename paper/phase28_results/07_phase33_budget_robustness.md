# Phase 33 Budget Robustness

## One-Sentence Argument

Phase 33 is a bounded budget-robustness follow-up to Phase 30 that tests
whether a modestly higher training budget can reduce the remaining normalized-B1
versus compressed-control gap under the same held-out Bishan base-reward
protocol.

## Current Experiment Snapshot

The current completed Phase 33 result set is no longer only the original
single-tile, single-seed pilot. The project has now completed bounded
`5120`-step matched pilots for the three held-out Phase 30 evaluation tiles
(`tile_r002_c003`, `tile_r005_c003`, `tile_r005_c004`) across seeds `0,1,2`,
always training only `N1Z` and `N1ZR` while reusing the existing `4096`
comparators `B1`, `D4P8`, and `D4P16`.

This fuller `3 tiles x 3 seeds` Phase 33 aggregate changes the conclusion.
The complete bounded aggregate is:

- output: `experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_3tiles_3seeds_9run_aggregate`
- status: `budget_not_explanatory`

The aggregate `4096 -> 5120` focal transitions are:

| Variant-gap | 4096 mean delta | 5120 mean delta | 5120 positive tile-seed count |
|---|---:|---:|---:|
| `N1Z - B1` | `0.3008963657` | `-0.1185345091` | `5 / 9` |
| `N1Z - D4P16` | `-0.3402976578` | `-0.7597285327` | `4 / 9` |
| `N1Z - D4P8` | `-0.0759554690` | `-0.4953863438` | `3 / 9` |
| `N1ZR - B1` | `0.2266106233` | `-0.0246975093` | `5 / 9` |
| `N1ZR - D4P16` | `-0.4145834002` | `-0.6658915329` | `3 / 9` |
| `N1ZR - D4P8` | `-0.1502412114` | `-0.4015493440` | `4 / 9` |

The positive single-tile pilot on `tile_r002_c003` remains useful as a local
counterexample, but it does not survive broader bounded coverage. The tilewise
aggregates now split clearly:

- `tile_r002_c003` 3-seed aggregate: `budget_closes_compressed_gap`;
- `tile_r005_c003` 3-seed aggregate: `budget_not_explanatory`;
- `tile_r005_c004` 3-seed aggregate: `budget_not_explanatory`.

## Important Boundary

This is still not a full Phase 33 conclusion in the manuscript sense. The
current evidence remains Bishan-only, bounded to `5120` train timesteps, and
restricted to the normalized-versus-compressed representation branch. Earlier
attempts to run a broader `8192`-step Phase 33 window did not finish within the
current execution window, so the project should treat the new aggregate as a
stronger bounded negative result, not as a final budget-robustness endpoint.

## Reproduction Command

```text
python experiments/phase33_budget_robustness/run_phase33_budget_robustness.py --mode run-and-analyze --baseline-phase30-comparison-json experiments/phase30_normalized_b1_ablation/outputs/real_bishan_4096_incremental/phase30_normalized_b1_comparison.json --baseline-control-summary-csv experiments/phase28_representation_controls/outputs/real_bishan_4096/phase28_representation_control_summary.csv --phase2-output-dir experiments/phase11_bishan_dltb_real/outputs/phase2_real --phase8-output-dir experiments/phase8_ablation_controls/outputs/real_bishan_controls --tile-index-csv experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv --variants B1,N1Z,N1ZR,D4P8,D4P16 --total-timesteps 5120 --eval-max-steps 8 --seeds 0,1,2 --eval-tile-ids tile_r002_c003,tile_r005_c003,tile_r005_c004 --output-dir experiments/phase33_budget_robustness/outputs/real_bishan_5120_full_grid_placeholder
```

In the current local run history, the equivalent bounded coverage was executed
as nine separate matched pilots and then aggregated into:

```text
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_3tiles_3seeds_9run_aggregate
```

## Claim Boundary

Phase 33 is a bounded budget-robustness follow-up over Phase 30 normalized-B1
and compressed-control artifacts. It does not enable suitability reward, does
not test B2/B3, does not test cross-region transfer, and does not support
final submission-level planning-performance claims.
