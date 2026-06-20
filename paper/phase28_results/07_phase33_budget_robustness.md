# Phase 33 Budget Robustness

## One-Sentence Argument

Phase 33 is a bounded budget-robustness follow-up to Phase 30 that tests
whether a modestly higher training budget can reduce the remaining normalized-B1
versus compressed-control gap under the same held-out Bishan base-reward
protocol.

## Current Experiment Snapshot

The current completed Phase 33 result is a matched pilot, not a full-budget
grid:

- high-budget run: `5120` train timesteps;
- coverage: `tile_r002_c003`, `seed=0`;
- trained variants: `N1Z`, `N1ZR`;
- reused comparators from existing `4096` artifacts: `B1`, `D4P8`, `D4P16`.

The matched `4096 -> 5120` transition on this single tile-seed pair is:

| Variant-gap | 4096 delta | 5120 delta | Change |
|---|---:|---:|---:|
| `N1Z - D4P16` | `-1.0605926275` | `0.1019858154` | `+1.1625784429` |
| `N1Z - D4P8` | `-0.7502553086` | `0.4123231343` | `+1.1625784429` |
| `N1ZR - D4P16` | `-0.4330059982` | `0.3081516791` | `+0.7411576773` |
| `N1ZR - D4P8` | `-0.1226686793` | `0.6184889980` | `+0.7411576773` |

Under this matched pilot, all four normalized-versus-compressed gaps flip from
negative to positive, so the local Phase 33 status is
`budget_closes_compressed_gap`.

## Important Boundary

This is not yet a full Phase 33 conclusion. The completed run is only a
single-tile, single-seed matched pilot. Earlier attempts to run a broader
`8192`-step Phase 33 window did not finish within the current execution window,
so the project should treat the current evidence as a pilot budget signal
rather than a stable budget-robustness claim.

## Reproduction Command

```text
python experiments/phase33_budget_robustness/run_phase33_budget_robustness.py --mode run-and-analyze --baseline-phase30-comparison-json experiments/phase30_normalized_b1_ablation/outputs/real_bishan_4096_incremental/phase30_normalized_b1_comparison.json --baseline-control-summary-csv experiments/phase28_representation_controls/outputs/real_bishan_4096/phase28_representation_control_summary.csv --phase2-output-dir experiments/phase11_bishan_dltb_real/outputs/phase2_real --phase8-output-dir experiments/phase8_ablation_controls/outputs/real_bishan_controls --tile-index-csv experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv --variants B1,N1Z,N1ZR,D4P8,D4P16 --total-timesteps 5120 --eval-max-steps 8 --seeds 0 --eval-tile-ids tile_r002_c003 --output-dir experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r002_c003_seed0_matched
```

## Claim Boundary

Phase 33 is a bounded budget-robustness follow-up over Phase 30 normalized-B1
and compressed-control artifacts. It does not enable suitability reward, does
not test B2/B3, does not test cross-region transfer, and does not support
final submission-level planning-performance claims.
