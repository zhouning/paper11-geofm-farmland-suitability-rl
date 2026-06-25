# Phase 28 Results Package

This folder records the current Phase 28 representation-control evidence for
Paper11. Phase 28 tests whether the raw B1 GeoFM representation is
distinguishable from explicit-only B0 and from random, shuffled, and
PCA-compressed representation controls under the same padded held-out Bishan
base-reward protocol.

## Files

- `01_phase28_representation_control_diagnosis.md`: reviewer-facing
  interpretation of the 1024-step and 4096-step Phase 28 representation-control
  diagnostics.
- `02_phase28_compression_diagnosis.md`: read-only follow-up diagnosis of why
  D4P8/D4P16 can exceed raw B1 in the current 4096-step run.
- `03_phase29_representation_scale_diagnosis.md`: read-only follow-up diagnosis
  of raw-B1 scale, normalization profiles, and PCA redundancy after the Phase
  28 compressed-control result.
- `04_phase30_normalized_b1_ablation.md`: bounded held-out follow-up testing
  whether normalized B1 variants recover raw-B1 optimization losses under the
  same protocol.
- `05_phase31_case_diagnostics.md`: read-only case diagnostic ranking
  informative Phase 30 tile-seed pairs and summarizing selected blocks plus
  tile geometry for spatial inspection.
- `06_phase32_action_order_diagnostics.md`: read-only action-order diagnostic
  comparing focal/comparator trace order, cumulative rewards, and local tile
  block-pool composition for Phase 31 cases.
- `07_phase33_budget_robustness.md`: bounded matched-pilot budget robustness
  follow-up comparing the existing 4096-step Phase 30 result with completed
  5120-step normalized-B1 reruns over three tiles and three seeds.
- `08_phase34_case_map_diagnostics.md`: read-only case-map diagnostic joining
  completed Phase 33 matched pilots to selected-block spatial and base-reward
  composition.

## Reproduction Link

Run the diagnostic from the repository root after the Phase 11/13 real Bishan
outputs and Phase 8 D-control feature tables exist:

```powershell
python experiments\phase28_representation_controls\run_phase28_representation_controls.py --mode run-and-analyze --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1,D2,D3,D4P8,D4P16 --total-timesteps 4096 --eval-max-steps 8 --seeds 0,1,2 --max-eval-tiles 3 --output-dir experiments\phase28_representation_controls\outputs\real_bishan_4096
```

The generated artifacts are ignored by Git, but the interpretation here is
based on the current generated Phase 28 1024-step and 4096-step comparison
JSON files.

The compression diagnosis is read-only. It recomputes selected-block overlap,
base-reward component means, and PCA variance summaries from the generated
4096-step Phase 28 summary CSV plus the existing Phase 2/8 feature tables. It
does not run additional policy training.

Run the reproducible compression follow-up from the repository root:

```powershell
python experiments\phase28_compression_diagnosis\run_phase28_compression_diagnosis.py --summary-csv experiments\phase28_representation_controls\outputs\real_bishan_4096\phase28_representation_control_summary.csv --phase2-b1-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\variant_B1_features.csv --d4p8-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P8_features.csv --d4p16-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P16_features.csv --output-dir experiments\phase28_compression_diagnosis\outputs\real_bishan_4096
```

Expected local read-only artifacts:

- `phase28_compression_overlap.csv`
- `phase28_compression_reward_components.csv`
- `phase28_compression_diagnosis.json`
- `phase28_compression_diagnosis.md`

The Phase 29 representation-scale follow-up is also read-only. It inspects
the same B1/D4 feature tables and the Phase 13 tile index without running
new policy training:

```powershell
python experiments\phase29_representation_scale_diagnosis\run_phase29_representation_scale_diagnosis.py --phase2-b1-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\variant_B1_features.csv --d4p8-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P8_features.csv --d4p16-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P16_features.csv --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --phase28-summary-csv experiments\phase28_representation_controls\outputs\real_bishan_4096\phase28_representation_control_summary.csv --output-dir experiments\phase29_representation_scale_diagnosis\outputs\real_bishan_4096
```

Expected local Phase 29 artifacts:

- `phase29_variant_scale_summary.csv`
- `phase29_tile_scale_summary.csv`
- `phase29_b1_normalization_profiles.csv`
- `phase29_representation_scale_diagnosis.json`
- `phase29_representation_scale_diagnosis.md`

The Phase 30 normalized-B1 follow-up is a bounded training experiment, but the
recommended path reuses the existing Phase 28 control summary and trains only
the two new normalized variants:

```powershell
python experiments\phase30_normalized_b1_ablation\run_phase30_normalized_b1_ablation.py --mode run-and-analyze --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --existing-control-summary-csv experiments\phase28_representation_controls\outputs\real_bishan_4096\phase28_representation_control_summary.csv --variants B0,B1,N1Z,N1ZR,D2,D3,D4P8,D4P16 --total-timesteps 4096 --eval-max-steps 8 --seeds 0,1,2 --max-eval-tiles 3 --output-dir experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental
```

Expected local Phase 30 artifacts:

- `derived_normalized_controls/`
- `phase30_normalized_b1_summary.csv`
- `phase30_normalized_b1_traces.json`
- `phase30_normalized_b1_comparison.json`
- `phase30_normalized_b1_delta_table.csv`
- `phase30_normalized_b1_readiness.md`

The Phase 31 case diagnostic is read-only. It ranks representative positive
and failure tile-seed cases from the Phase 30 summary and trace outputs, then
joins selected blocks to Phase 2 features, the Phase 13 tile index, and the
Phase 11 block-pixel mapping:

```powershell
python experiments\phase31_case_diagnostics\run_phase31_case_diagnostics.py --summary-csv experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental\phase30_normalized_b1_summary.csv --traces-json experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental\phase30_normalized_b1_traces.json --phase2-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\block_geofm_features.csv --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --block-mapping-csv experiments\phase11_bishan_dltb_real\outputs\adapter\block_pixel_mapping.csv --output-dir experiments\phase31_case_diagnostics\outputs\real_bishan_4096 --top-k 6
```

Expected local Phase 31 artifacts:

- `phase31_ranked_cases.csv`
- `phase31_selected_blocks.csv`
- `phase31_tile_geometry.csv`
- `phase31_case_diagnostics.json`
- `phase31_case_diagnostics.md`

The Phase 32 action-order diagnostic is also read-only. It uses Phase 31 ranked
cases, Phase 30 N1ZR traces, Phase 28 B1 traces, Phase 2 features, and the
Phase 13 tile index to compare action order and local block composition:

```powershell
python experiments\phase32_action_order_diagnostics\run_phase32_action_order_diagnostics.py --ranked-cases-csv experiments\phase31_case_diagnostics\outputs\real_bishan_4096\phase31_ranked_cases.csv --focal-traces-json experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental\phase30_normalized_b1_traces.json --comparator-traces-json experiments\phase28_representation_controls\outputs\real_bishan_4096\phase28_representation_control_traces.json --phase2-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\block_geofm_features.csv --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --output-dir experiments\phase32_action_order_diagnostics\outputs\real_bishan_4096 --top-k 6
```

Expected local Phase 32 artifacts:

- `phase32_step_alignment.csv`
- `phase32_case_summary.csv`
- `phase32_tile_pool_composition.csv`
- `phase32_action_order_diagnostics.json`
- `phase32_action_order_diagnostics.md`

The Phase 33 budget-robustness follow-up is a bounded matched pilot. It reuses
the existing 4096-step Phase 30 comparison and 4096-step Phase 28 control
summary, trains only the normalized variants at a modestly higher budget, and
then compares the matched tile-seed/comparator coverage:

```powershell
python experiments\phase33_budget_robustness\run_phase33_budget_robustness.py --mode run-and-analyze --baseline-phase30-comparison-json experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental\phase30_normalized_b1_comparison.json --baseline-control-summary-csv experiments\phase28_representation_controls\outputs\real_bishan_4096\phase28_representation_control_summary.csv --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B1,N1Z,N1ZR,D4P8,D4P16 --total-timesteps 5120 --eval-max-steps 8 --seeds 0 --eval-tile-ids tile_r002_c003 --output-dir experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r002_c003_seed0_matched
```

Expected local Phase 33 artifacts:

- `phase33_matched_baseline_comparison.json`
- `phase33_budget_transition.csv`
- `phase33_focal_gap_transition.csv`
- `phase33_tile_seed_stability.csv`
- `phase33_budget_robustness.json`
- `phase33_budget_robustness.md`
- `phase30_high_budget/`

The Phase 34 case-map diagnostic is read-only. It joins the completed Phase 33
matched pilot summaries and traces to Phase 2 block features and the Phase 13
tile index:

```powershell
python experiments\phase34_case_map_diagnostics\run_phase34_case_map_diagnostics.py --phase33-output-dirs experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r002_c003_seed0_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r002_c003_seed1_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r002_c003_seed2_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c003_seed0_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c003_seed1_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c003_seed2_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c004_seed0_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c004_seed1_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c004_seed2_matched --phase2-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\block_geofm_features.csv --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --output-dir experiments\phase34_case_map_diagnostics\outputs\real_bishan_5120_phase33_9run --variants N1Z,N1ZR --comparators B1,D4P8,D4P16
```

Expected local Phase 34 artifacts:

- `phase34_case_map_cases.csv`
- `phase34_case_map_blocks.csv`
- `phase34_case_map_diagnostics.json`
- `phase34_case_map_diagnostics.md`

## Claim Boundary

These representation-branch follow-ups are diagnostic only. They do not enable
suitability reward, do not test B2/B3, do not test cross-region transfer, and
do not support final submission-level planning-performance claims.
