# Phase 31 Case Diagnostics

## One-Sentence Argument

Phase 31 is the experiment-first follow-up to Phase 30: it does not train a new
policy, but ranks informative Phase 30 tile-seed cases and summarizes selected
blocks and tile geometry for spatial inspection.

## What This Phase Does

Phase 31 reads existing Phase 30 artifacts:

```text
phase30_normalized_b1_summary.csv
phase30_normalized_b1_traces.json
```

It joins them with:

```text
experiments/phase11_bishan_dltb_real/outputs/phase2_real/block_geofm_features.csv
experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv
experiments/phase11_bishan_dltb_real/outputs/adapter/block_pixel_mapping.csv
```

The resulting artifacts are:

```text
phase31_ranked_cases.csv
phase31_selected_blocks.csv
phase31_tile_geometry.csv
phase31_case_diagnostics.json
phase31_case_diagnostics.md
```

## Current Real Run Snapshot

The local real Bishan run with `--top-k 6` reports:

| Rank | Case | Role | N1ZR - B1 | Selected-block Jaccard |
|---:|---|---|---:|---:|
| 1 | `tile_r005_c004|1|N1ZR|B1` | strong_positive | `1.6253422876` | `0.6` |
| 2 | `tile_r002_c003|0|N1ZR|B1` | strong_positive | `0.9750187978` | `0.3333333333` |
| 3 | `tile_r005_c003|2|N1ZR|B1` | strong_positive | `0.6445820376` | `0.6` |
| 4 | `tile_r005_c003|0|N1ZR|B1` | failure_case | `-0.4077216427` | `0.2307692308` |
| 5 | `tile_r002_c003|2|N1ZR|B1` | failure_case | `-0.3638798802` | `0.7777777778` |
| 6 | `tile_r005_c003|1|N1ZR|B1` | failure_case | `-0.3223425560` | `0.3333333333` |

The selected-block summaries give a concrete follow-up target. In the strongest
positive case, `tile_r005_c004|1`, N1ZR selected eight blocks with mean
`low_slope_farmland_label = 0.375` and mean base planning reward
`0.2344417950`, while B1 selected eight blocks with mean
`low_slope_farmland_label = 0.125` and mean base planning reward
`0.0312740091`. In the carried-forward failure case,
`tile_r002_c003|2`, N1ZR remains below B1 even with high selected-block overlap
(`0.7777777778`), so the next diagnostic should inspect action order and local
block composition rather than only set membership.

## Reproduction Command

```text
python experiments/phase31_case_diagnostics/run_phase31_case_diagnostics.py --summary-csv experiments/phase30_normalized_b1_ablation/outputs/real_bishan_4096_incremental/phase30_normalized_b1_summary.csv --traces-json experiments/phase30_normalized_b1_ablation/outputs/real_bishan_4096_incremental/phase30_normalized_b1_traces.json --phase2-features-csv experiments/phase11_bishan_dltb_real/outputs/phase2_real/block_geofm_features.csv --tile-index-csv experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv --block-mapping-csv experiments/phase11_bishan_dltb_real/outputs/adapter/block_pixel_mapping.csv --output-dir experiments/phase31_case_diagnostics/outputs/real_bishan_4096 --top-k 6
```

## Claim Boundary

Phase 31 is a read-only case diagnostic over existing Phase 30 artifacts. It
does not run new policy training, does not alter rewards, does not enable
suitability reward, does not test B2/B3, and does not support final
submission-level planning-performance claims.
