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
- `09_phase35_phase33_action_overlap_diagnostics.md`: read-only action-overlap
  diagnostic over completed Phase 33 matched pilots, testing whether selected
  blocks are shared/reordered or nearly disjoint.
- `10_phase36_suitability_proxy_validation.md`: read-only weak-label
  suitability-proxy validation over existing Phase 2, Phase 8, and Phase 30
  feature tables before any B2/B3 reward work.
- `11_phase37_decision_alignment.md`: read-only decision-alignment audit
  joining Phase 34, Phase 35, and Phase 36 artifacts to test whether Phase 33
  decisions separate in available proxy, slope, and weak farmland diagnostics.
- `12_phase38_proxy_rebuild.md`: leakage-aware proxy-rebuild diagnostic
  over existing Phase 2, Phase 8, and Phase 30 feature tables before any B2/B3 reward integration.
- `13_phase39_independent_label_audit.md`: independent-label inventory and
  readiness audit over the Phase 2 real feature table before any Phase 38 rerun
  with non-leakage labels.
- `14_phase40_independent_label_gate.md`: hard go/no-go independent-label gate
  that blocks Phase 38 rerun, B2/B3 reward integration, and positive
  suitability claims unless an external non-leakage label registry passes.
- `15_phase41_geofm_suitability_prior.md`: independent-label-calibrated GeoFM
  suitability-prior gate that blocks low-dimensional prior export unless GeoFM
  clears baseline, control, fold-stability, and calibration checks.
- `16_phase42_local_label_source_audit.md`: local candidate-label source
  audit showing that available DLTB/slope weak labels remain diagnostic-only
  and that no Phase 40-passing independent label is present locally.
- `17_phase48_compressed_geofm_rescue.md`: read-only rescue audit showing that D4P8/D4P16 are supported compressed GeoFM candidate routes under the existing base-reward held-out protocol.
- `18_phase49_compressed_route_robustness.md`: read-only robustness audit showing that the Phase 48 compressed route survives pooled sign-test, bootstrap, and leave-one sensitivity checks.
- `19_phase50_cluster_level_robustness.md`: conservative tile-seed cluster-level audit showing directional support with `7 / 9` positive clusters and p `0.08984375`.
- `20_phase51_cluster_magnitude_support.md`: exact signed-rank cluster magnitude audit showing positive rank sum `40 / 45` and p `0.01953125`.
- `21_phase52_expanded_cluster_replication.md`: expanded five-tile, three-seed
  replication showing that compressed GeoFM routes remain mean-positive against
  B0, raw B1, random D2, and shuffled D3, with pooled row-level p
  `0.0066881634` and cluster signed-rank p `0.0206298828`.
- `22_phase53_cluster_mean_support.md`: exact sign-flip, bootstrap, and
  leave-one influence audit showing `cluster_mean_support` for the expanded
  Phase 52 cluster mean.
- `23_phase54_artifact_lineage_consistency.md`: read-only lineage audit showing
  `artifact_lineage_consistent` for the formal Phase 52/53 artifact chain.
- `24_phase57_compressed_representation_mechanism.md`: read-only
  representation-geometry audit showing that D4P8/D4P16 retain most raw GeoFM
  variance while reducing effective rank and conditioning burden relative to
  raw B1.
- `25_phase59_matched_dimension_controls.md`: matched-dimension control audit
  showing that D4P8/D4P16 do not clearly outperform same-dimension random or
  shuffled controls under the expanded Bishan base-reward protocol.
- `26_phase60_information_optimization_attribution.md`: read-only
  information-vs-optimization attribution audit reconciling Phase 48/52,
  Phase 53, Phase 57, and Phase 59 evidence and narrowing the mechanism claim
  to a bounded low-dimensional compressed state route.
- `27_phase61_d6_geofm_projection_controls.md`: D6 GeoFM projection-control
  feature and geometry audit preparing raw-B1 random and PCA projection
  controls for later matched training.
- `28_phase62_d4_d6_matched_ppo.md`: matched PPO training evaluation showing
  that D4P8/D4P16 do not outperform D6R8/D6R16 raw-B1 random orthonormal
  projection controls under the Phase 52 five-tile, three-seed protocol.
- `29_phase63_set_policy_oracle_pretraining.md`: set-policy oracle-pretraining
  experiment showing that explicit per-block scoring plus deterministic
  behavior cloning improves base-reward block selection over the flattened PPO
  route, while GeoFM-derived variants are not distinguished from B0.
- `30_phase64_set_policy_error_diagnosis.md`: set-policy error diagnosis and
  standardization gate using Phase 63 behavior-cloned rollout, oracle summary,
  training-history, and feature-scale artifacts to decide whether standardized
  set-policy BC is the next justified algorithm experiment.
- `34_phase68_external_independent_label_package.md`: external independent-label
  package and preflight contract showing that templates are ready, but no
  external independent label has been supplied yet.
- `35_phase69_label_free_evidence_synthesis_gate.md`: read-only label-free
  synthesis gate showing that the strongest current algorithm claim must remain
  narrowed to a bounded low-dimensional compressed state route.
- `36_phase70_standardized_set_policy_rerun.md`: standardized set-policy rerun testing whether train-tile-fitted model-input standardization strengthens the bounded low-dimensional set-policy route.

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


The Phase 35 action-overlap diagnostic is read-only. It uses the same Phase
33 matched pilot directories and compares high-budget normalized-B1 selected
blocks against matched comparator selected blocks:

```powershell
python experiments\phase35_phase33_action_overlap_diagnostics\run_phase35_phase33_action_overlap_diagnostics.py --phase33-output-dirs experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r002_c003_seed0_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r002_c003_seed1_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r002_c003_seed2_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c003_seed0_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c003_seed1_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c003_seed2_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c004_seed0_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c004_seed1_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c004_seed2_matched --output-dir experiments\phase35_phase33_action_overlap_diagnostics\outputs\real_bishan_5120_phase33_9run --variants N1Z,N1ZR --comparators B1,D4P8,D4P16
```

Expected local Phase 35 artifacts:

- `phase35_action_overlap_cases.csv`
- `phase35_action_overlap_steps.csv`
- `phase35_action_overlap_diagnostics.json`
- `phase35_action_overlap_diagnostics.md`

The Phase 36 suitability-proxy validation is read-only. It tests whether
GeoFM-derived feature families add weak-label signal beyond explicit planning
features and diagnostic controls before any suitability-reward integration:

```powershell
python experiments\phase36_suitability_proxy_validation\run_phase36_suitability_proxy_validation.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --normalized-controls-dir experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental\derived_normalized_controls --output-dir experiments\phase36_suitability_proxy_validation\outputs\real_bishan --label-columns current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label
```

Expected local Phase 36 artifacts:

- `phase36_label_summary.csv`
- `phase36_model_summary.csv`
- `phase36_suitability_proxy_validation.json`
- `phase36_suitability_proxy_validation.md`

The current status is `proxy_signal_not_supported`: the available weak labels
are usable, but they are DLTB/slope-derived and flagged for explicit-feature
leakage risk. The current scalar `suitability_proxy` is not supported as a
reward term; the next branch should use independent labels or rebuild a
supervised/semi-supervised suitability proxy.


The Phase 37 decision-alignment audit is read-only. It joins Phase 34 case-map
rows, Phase 34 selected-block rows, Phase 35 action-overlap cases, and the
Phase 36 diagnosis before any B2/B3 reward integration:

```powershell
python experiments\phase37_decision_alignment\run_phase37_decision_alignment.py --phase34-cases-csv experiments\phase34_case_map_diagnostics\outputs\real_bishan_5120_phase33_9run\phase34_case_map_cases.csv --phase34-blocks-csv experiments\phase34_case_map_diagnostics\outputs\real_bishan_5120_phase33_9run\phase34_case_map_blocks.csv --phase35-cases-csv experiments\phase35_phase33_action_overlap_diagnostics\outputs\real_bishan_5120_phase33_9run\phase35_action_overlap_cases.csv --phase36-diagnosis-json experiments\phase36_suitability_proxy_validation\outputs\real_bishan\phase36_suitability_proxy_validation.json --output-dir experiments\phase37_decision_alignment\outputs\real_bishan_5120_phase33_9run
```

Expected local Phase 37 artifacts:

- `phase37_decision_alignment_cases.csv`
- `phase37_decision_alignment_summary.csv`
- `phase37_decision_alignment.json`
- `phase37_decision_alignment.md`

The current Phase 37 status is `decision_alignment_not_supported` with `54`
joined case rows and `37` summary rows. Phase 36 remains
`proxy_signal_not_supported`, so B2/B3 suitability reward remains blocked.

The Phase 38 proxy-rebuild diagnostic is read-only with respect to rewards and
policy training. It rebuilds diagnostic proxy classifiers from existing real
feature tables and writes full per-block rebuilt scores to CSV:

```powershell
python experiments\phase38_proxy_rebuild\run_phase38_proxy_rebuild.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --normalized-controls-dir experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental\derived_normalized_controls --output-dir experiments\phase38_proxy_rebuild\outputs\real_bishan --label-columns current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label --model-families logistic_elastic_net,random_forest,hist_gradient_boosting
```

Expected local Phase 38 artifacts:

- `phase38_label_summary.csv`
- `phase38_model_summary.csv`
- `phase38_rebuilt_proxy_scores.csv`
- `phase38_proxy_rebuild.json`
- `phase38_proxy_rebuild.md`

The current Phase 38 status is `proxy_rebuild_diagnostic_only` with `64984`
block rows, `99` model rows, and `6433416` rebuilt proxy score rows. All
current labels are classified as `explicit_label_leakage_risk`, so B2/B3
suitability reward remains blocked.

The Phase 39 independent-label audit is read-only with respect to rewards and
policy training. It inventories available label-like columns in the Phase 2
real feature table and writes a registry template for future independent label
sources:

```powershell
python experiments\phase39_independent_label_audit\run_phase39_independent_label_audit.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase39_independent_label_audit\outputs\real_bishan
```

Expected local Phase 39 artifacts:

- `phase39_label_inventory.csv`
- `phase39_label_readiness.csv`
- `phase39_label_registry_template.csv`
- `phase39_independent_label_audit.json`
- `phase39_independent_label_audit.md`

The current Phase 39 status is `independent_label_inputs_missing` with `64984`
block rows, `7` label inventory rows, `7` label readiness rows, and `0`
registry rows. The audited default labels are `current_farmland_label`,
`farmland_or_orchard_label`, `low_slope_farmland_label`, `source_bsm`,
`source_category`, `source_dlbm`, and `source_dlmc`.

Phase 39 does not run PPO, alter rewards, enable B2/B3, prove agronomic
validity, or support planning-performance claims. Phase 38 cannot yet be rerun
with a stronger non-leakage label and B2/B3 remains blocked.

The Phase 40 independent-label gate is the hard decision point after Phase 39.
Run it from the repository root after the Phase 2 real feature table exists:

```powershell
python experiments\phase40_independent_label_gate\run_phase40_independent_label_gate.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase40_independent_label_gate\outputs\real_bishan_no_registry
```

Expected local Phase 40 artifacts:

- `phase40_label_gate_summary.csv`
- `phase40_independent_label_gate.json`
- `phase40_independent_label_gate.md`

The current Phase 40 status is `independent_label_inputs_missing`. This is a
hard stop for suitability reward, Phase 38 rerun, and B2/B3 until an external
independent label registry is supplied and passes the gate.

The Phase 41 GeoFM suitability-prior gate tests the revised route in which
GeoFM is used only as an independent-label-calibrated low-dimensional prior,
not as raw 64-dimensional policy state. Run it from the repository root after
the Phase 2 real feature table exists:

```powershell
python experiments\phase41_geofm_suitability_prior\run_phase41_geofm_suitability_prior.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase41_geofm_suitability_prior\outputs\real_bishan_no_registry
```

Expected local Phase 41 artifacts:

- `phase41_geofm_prior_summary.csv`
- `phase41_geofm_prior_metrics.csv`
- `phase41_geofm_prior.json`
- `phase41_geofm_prior.md`

The current Phase 41 status is `phase41_independent_label_inputs_missing`.
Phase 41 therefore does not produce a calibrated GeoFM suitability prior for
the real Bishan run, and B2/B3 remains blocked.

Phase 42 audits local candidate label sources after Phase 41. It records that
available DLTB/slope-derived labels are diagnostic-only and that unrelated
Paper10/Paper58 labels cannot be registered as Paper11 Bishan suitability
labels. A diagnostic registry check returns Phase 40
`independent_label_gate_diagnostic_only` and Phase 41
`phase41_independent_label_inputs_missing`.

Phase 48 reinterprets the compressed D4 controls as candidate GeoFM routes. It
runs read-only over the frozen Phase 28 `4096`-step summary CSV:

```powershell
python experiments\phase48_compressed_geofm_rescue\run_phase48_compressed_geofm_rescue.py --existing-summary-csv experiments\phase28_representation_controls\outputs\real_bishan_4096\phase28_representation_control_summary.csv --output-dir experiments\phase48_compressed_geofm_rescue\outputs\real_bishan_4096
```

The current Phase 48 status is `compressed_geofm_route_supported`. D4P8 and
D4P16 exceed B0, raw B1, random D2, and shuffled D3 on mean learned-policy
reward. The pooled compressed-control delta is `0.4673011499` with `48 / 72`
positive comparisons. This changes the representation conclusion: raw B1 direct
injection remains unsupported, but compressed GeoFM state routes are supported
under the current base-reward held-out Bishan protocol. Suitability reward and
B2/B3 remain blocked by the Phase 40/41 independent-label gates.

Phase 49 tests the Phase 48 delta table with pooled sign-test, bootstrap, and
leave-one sensitivity checks. The current status is
`compressed_route_statistically_robust`, with pooled mean delta `0.4673011499`,
`48 / 72` positive comparisons, one-sided sign-test p `0.0031549137`, bootstrap
CI95 `[0.2827829983, 0.6639974489]`, minimum leave-one-tile mean
`0.1613586660`, and minimum leave-one-seed mean `0.3644002401`.

Phase 50 then aggregates the same Phase 48 deltas to tile-seed clusters. The
current status is `cluster_directional_support`: mean cluster delta
`0.4673011499`, `7 / 9` positive clusters, and one-sided cluster sign-test p
`0.08984375`. This is a conservative wording boundary rather than a reversal of
Phase 48/49. Phase 51 then tests cluster magnitude with an exact one-sided
signed-rank test and reports `cluster_magnitude_support`, positive rank sum
`40 / 45`, and p `0.01953125`.

Phase 52 expands the same six-variant protocol to five held-out tiles and
three seeds. Reusing the existing read-only Phase 48-51 analyzers, the expanded
summary reports `compressed_geofm_route_supported`, pooled compressed-control
delta `0.2921767818`, `74 / 120` positive row-level comparisons, one-sided
row-level sign-test p `0.0066881634`, bootstrap CI95 `[0.1623326461,
0.4323997354]`, `10 / 15` positive tile-seed clusters with sign-only p
`0.1508789062`, and exact cluster signed-rank p `0.0206298828`. This
strengthens the compressed-route evidence while preserving the cluster
sign-only wording boundary.

Phase 53 then audits the expanded Phase 52 cluster mean directly. The current
status is `cluster_mean_support`: mean cluster delta `0.2921767818`, exact
one-sided sign-flip mean p `0.0196838379`, bootstrap CI95 `[0.0570820445,
0.5823557658]`, minimum leave-one-cluster mean `0.2060081575`, minimum
leave-one-tile mean `0.0954244478`, and minimum leave-one-seed mean
`0.2083797951`. This supports the conclusion that the expanded compressed-route
cluster mean is not driven only by one favorable cluster, tile, or seed.

Phase 54 audits artifact lineage for the formal Phase 52/53 evidence chain. The
current status is `artifact_lineage_consistent`: recomputing Phase 50 cluster
means from the authoritative Phase 48 delta table, then recomputing Phase 51 and
Phase 53 statistics from the authoritative cluster CSV, reproduces the formal
values used in the manuscript. Historical `real_bishan_4096_5tiles*` generated
outputs are therefore not used as the formal evidence source for this conclusion.

Phase 57 audits the compressed-route mechanism without retraining policies. Run
it from the repository root after the Phase 2/8 feature tables, Phase 13 tile
index, and expanded Phase 52 delta table exist:

```powershell
python experiments\phase57_compressed_representation_mechanism\run_phase57_compressed_representation_mechanism.py --b1-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\variant_B1_features.csv --d4p8-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P8_features.csv --d4p16-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P16_features.csv --delta-csv experiments\phase52_expanded_cluster_replication\outputs\phase52_full5_seed3_phase48_compressed_rescue\phase48_compressed_geofm_rescue_delta_table.csv --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --output-dir experiments\phase57_compressed_representation_mechanism\outputs\phase52_full5_seed3
```

The current Phase 57 status is `compressed_geometry_consistent`. All `64,984`
blocks align across B1, D4P8, and D4P16 feature tables. D4P8 retains
`85.87823898%` of raw GeoFM variance while reducing effective rank from
`9.4947211626` to `5.1322783588` and condition number from `6658.9542931381`
to `16.2704676982`. D4P16 retains `94.96006154%` of raw GeoFM variance while
reducing effective rank to `7.3009059917` and condition number to
`53.6978527088`. The expanded-replication reward gains remain positive for
both compressed variants.

Phase 59 then tests the rival explanation that the D4P8/D4P16 gains are mainly
a same-dimension optimization effect rather than a GeoFM-specific compressed
signal. The current Phase 59 status is `matched_dimension_geofm_not_supported`:
D4P8 - D5R8 mean `-0.0107871307` (`5 / 15` positive), D4P8 - D5S8 mean
`0.0003232239` (`7 / 15` positive), D4P16 - D5R16 mean `-0.1193811247`
(`2 / 15` positive), and D4P16 - D5S16 mean `0.060921975` (`8 / 15` positive).
The pooled matched-control mean is `-0.0172307641`, with `22 / 60` positive
rows, bootstrap CI95 `[-0.1081223337, 0.0751760409]`, cluster sign-test p `0.5`,
and signed-rank p `0.6192321777`. This narrows the mechanism wording: the
earlier compressed-route evidence remains valid against B0/B1/D2/D3, but the
current run does not support a stronger GeoFM-specific advantage over matched
low-dimensional controls.

Phase 60 reconciles the Phase 48/52, Phase 53, Phase 57, and Phase 59 evidence
as a read-only information-vs-optimization attribution audit:

```powershell
python experiments\phase60_information_optimization_attribution\run_phase60_information_optimization_attribution.py --phase48-json experiments\phase52_expanded_cluster_replication\outputs\phase52_full5_seed3_phase48_compressed_rescue\phase48_compressed_geofm_rescue_comparison.json --phase53-json experiments\phase53_cluster_mean_support\outputs\phase52_full5_seed3\phase53_cluster_mean_support.json --phase57-json experiments\phase57_compressed_representation_mechanism\outputs\phase52_full5_seed3\phase57_compressed_representation_mechanism.json --phase59-json experiments\phase59_matched_dimension_controls\outputs\phase52_full5_seed3\phase59_matched_dimension_controls.json --output-dir experiments\phase60_information_optimization_attribution\outputs\phase52_full5_seed3
```

The current Phase 60 status is `mechanism_claim_narrowed`. The compressed-route
performance axis, cluster-level robustness axis, and compressed-geometry axis
are supported, but the GeoFM-specific matched-dimension axis is not supported
because Phase 59 reports pooled matched-control mean delta `-0.0172307641`.
The resulting claim boundary is a bounded low-dimensional compressed state
route under the Bishan base-reward protocol, not a proven unique advantage of
the current PCA-compressed GeoFM coordinates over same-dimension controls.
Before a stronger mechanism claim, the next optional experiment is a D6-style
GeoFM projection-control audit.

Phase 61 builds that D6-style projection-control set without training PPO
policies:

```powershell
python experiments\phase61_d6_geofm_projection_controls\run_phase61_d6_geofm_projection_controls.py --b0-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\variant_B0_features.csv --b1-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\variant_B1_features.csv --d4p8-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P8_features.csv --d4p16-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P16_features.csv --output-dir experiments\phase61_d6_geofm_projection_controls\outputs\phase52_full5_seed3 --dimensions 8,16 --seed 61
```

The current Phase 61 status is `d6_projection_controls_ready_for_training`.
All four generated D6 controls have `64,984` aligned rows and valid projection
variance. D6P8 and D6P16 reproduce D4P8/D4P16 exactly by column correlation
(`1.0`), while D6R8 and D6R16 provide distinct raw-B1 random orthonormal
projection controls with D4 similarities `0.13402939` and `0.1827507258`. This
prepares the next matched training experiment but does not itself support a
learned-policy performance claim.

Phase 62 then trains the primary D4/D6R comparison under the same Phase 52
five-tile, three-seed base-reward protocol:

```powershell
python experiments\phase62_d4_d6_matched_ppo\run_phase62_d4_d6_matched_ppo.py --mode run-and-analyze --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --phase61-output-dir experiments\phase61_d6_geofm_projection_controls\outputs\phase52_full5_seed3 --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants D4P8,D4P16,D6R8,D6R16 --eval-tile-ids tile_r002_c003,tile_r005_c004,tile_r005_c003,tile_r000_c004,tile_r001_c004 --total-timesteps 4096 --eval-max-steps 8 --seeds 0,1,2 --bootstrap-iterations 5000 --seed 62 --output-dir experiments\phase62_d4_d6_matched_ppo\outputs\phase52_full5_seed3
```

The current Phase 62 status is `d6_random_projection_advantage`. D4P8 - D6R8
has mean delta `-0.0279096981` (`5 / 15` positive), D4P16 - D6R16 has mean
`-0.1004704046` (`1 / 15` positive), and the pooled primary D4-D6R mean is
`-0.0641900514` (`6 / 30` positive), with bootstrap CI95 `[-0.1459486743,
0.01836613]`, cluster mean `-0.0641900514`, positive clusters `5 / 15`, and
signed-rank p `0.8793945312`. Phase 62 therefore does not support a
PCA-specific D4 advantage over raw-B1 random orthonormal projections; it
preserves the Phase 60 narrowed claim rather than strengthening it.

Phase 63 then switches from representation-only diagnosis to algorithm/model
work. It trains a set-style block scorer from deterministic base-reward oracle
trajectories and rolls the behavior-cloned policy out on the same five held-out
tiles and three seeds. The current status is
`architecture_improves_but_geofm_not_distinguished`: the architecture improves
over the flattened PPO baseline with mean delta `4.4387176072` and `75 / 75`
positive rows, but D4/B0 mean delta is `-0.0677835004` (`10 / 30` positive)
and D4/D6 mean delta is `-0.0479468867` (`7 / 30` positive). This supports
continuing algorithm/model work before manuscript revision, without reviving a
GeoFM-specific or PCA-optimality claim.

Phase 64 keeps the work in algorithm/experiment mode. It does not retrain a
policy. It reads Phase 63 set-policy artifacts, diagnoses selected-block
overlap, oracle-rank gaps, training convergence, and feature scale/effective
rank, then records a standardization gate decision in
`30_phase64_set_policy_error_diagnosis.md`.
## Claim Boundary

These representation-branch follow-ups are diagnostic only. They do not enable
suitability reward, do not validate the current scalar suitability proxy as a
reward term, do not test B2/B3, do not test cross-region transfer, and do not
support final submission-level planning-performance claims.
