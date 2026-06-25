# Phase 34 Case-Map Diagnostics

## One-Sentence Argument

Phase 34 is a read-only spatial case-map follow-up to Phase 33: it joins the
completed `5120` matched pilot outputs to Phase 2 block features and Phase 13
tile metadata to inspect whether positive and negative Phase 33 cases are
associated with different selected-block spatial and base-reward composition.

## Current Experiment Snapshot

Phase 34 reads the nine completed Phase 33 matched pilot directories:

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
experiments/phase34_case_map_diagnostics/outputs/real_bishan_5120_phase33_9run
```

The output contains:

```text
phase34_case_map_cases.csv
phase34_case_map_blocks.csv
phase34_case_map_diagnostics.json
phase34_case_map_diagnostics.md
```

The real run status is:

```text
case_map_diagnostics_ready
```

Row counts:

- case rows: `54`
- case-map block rows: `857`
- selected feature rows joined from Phase 2: `294`
- Phase 33 matched directories: `9`

## Main Result

The case-map diagnostic gives a simple spatial-composition explanation for the
Phase 33 split. Every positive case falls into
`variant_selects_higher_base_reward_blocks`, and every negative case falls into
`variant_selects_lower_base_reward_blocks`.

| Group | Count | Mean higher delta | Mean base-reward gap | Mean selected-block Jaccard |
|---|---:|---:|---:|---:|
| Phase 33 positive cases | `24` | `0.5822555613` | `0.0727819453` | `0.0111111111` |
| Phase 33 failure cases | `30` | `-1.2055407806` | `-0.1506925976` | `0.0066666667` |

By stability class:

| Stability class | Count | Mean higher delta | Mean base-reward gap | Mean Jaccard |
|---|---:|---:|---:|---:|
| `stable_positive` | `11` | `0.7219712548` | `0.0902464070` | `0.0242424243` |
| `flip_to_positive` | `13` | `0.4640345899` | `0.0580043239` | `0.0000000000` |
| `stable_negative` | `18` | `-1.0630149499` | `-0.1328768688` | `0.0037037037` |
| `flip_to_negative` | `12` | `-1.4193295266` | `-0.1774161907` | `0.0111111111` |

By tile:

| Tile | Case rows | Mean higher delta | Mean base-reward gap | Spatial pattern split |
|---|---:|---:|---:|---|
| `tile_r002_c003` | `18` | `0.4638019194` | `0.0579752401` | `15` higher / `3` lower |
| `tile_r005_c003` | `18` | `-1.3403609350` | `-0.1675451169` | `0` higher / `18` lower |
| `tile_r005_c004` | `18` | `-0.3563348703` | `-0.0445418588` | `9` higher / `9` lower |

This supports the Phase 33 interpretation that the positive `tile_r002_c003`
budget pilot was real but spatially local. It also identifies
`tile_r005_c003` as the clearest negative spatial counterexample: all 18
variant-comparator case rows on that tile select lower mean base-reward block
sets than their comparators.

## Reproduction Command

```text
python experiments/phase34_case_map_diagnostics/run_phase34_case_map_diagnostics.py --phase33-output-dirs experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r002_c003_seed0_matched experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r002_c003_seed1_matched experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r002_c003_seed2_matched experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c003_seed0_matched experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c003_seed1_matched experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c003_seed2_matched experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c004_seed0_matched experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c004_seed1_matched experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c004_seed2_matched --phase2-features-csv experiments/phase11_bishan_dltb_real/outputs/phase2_real/block_geofm_features.csv --tile-index-csv experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv --output-dir experiments/phase34_case_map_diagnostics/outputs/real_bishan_5120_phase33_9run --variants N1Z,N1ZR --comparators B1,D4P8,D4P16
```

## Claim Boundary

Phase 34 is a read-only case-map diagnostic over existing Phase 33 matched
pilot artifacts. It does not run new policy training, does not alter rewards,
does not enable suitability reward, does not test B2/B3, and does not support
final submission-level planning-performance claims.
