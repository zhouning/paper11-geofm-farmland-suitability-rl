# Reproduction Guide

This guide documents the lightweight reviewer path for the standalone Paper11 repository.

## 1. Environment

Create a Python environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The core smoke check requires only Python and NumPy. Full RL training uses PyTorch, Gymnasium, Stable-Baselines3, and sb3-contrib. Embedding extraction from Google Earth Engine requires a configured Earth Engine account.

## 2. Verify Repository Integrity

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests
```

Expected outcome:

- required design, runtime, sample data, and reproducibility files exist;
- `data/bishan_alphaearth_sample/metadata.json` reports years 2017-2024 and embedding dimension 64;
- each included Bishan embedding array is readable and has final dimension 64.

## 3. Run the Phase 1 Bishan Baseline

Run:

```powershell
python experiments\phase1_bishan_baseline\run_phase1.py
```

Expected outcome:

- `experiments/phase1_bishan_baseline/outputs/region_features.csv` is created;
- `experiments/phase1_bishan_baseline/outputs/summary.json` is created;
- the summary reports 25 deterministic grid regions by default;
- `claim_boundary` states that `suitability_proxy` is derived from latent remote-sensing embeddings and does not directly measure soil, fertility, or irrigation.

The default outputs are ignored by Git. Use a custom output directory for controlled verification:

```powershell
python experiments\phase1_bishan_baseline\run_phase1.py --output-dir D:\tmp\paper11_phase1_outputs
```

## 4. Run the Phase 2 Block Feature Assembly Baseline

Run:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py
```

Expected outcome:

- `experiments/phase2_block_geofm_features/outputs/block_geofm_features.csv` is created;
- `experiments/phase2_block_geofm_features/outputs/summary.json` is created;
- `experiments/phase2_block_geofm_features/outputs/experiment_variants.json` is created;
- the summary reports 25 generated grid-derived blocks by default;
- `feature_readiness` reports B0/B1/B2/B3 readiness and marks explicit-feature-dependent variants incomplete when explicit planning features are absent;
- no `variant_B*_features.csv` files are created by default because the generated-grid path lacks explicit planning features;
- incomplete variants in `experiment_variants.json` have `feature_table` set to `null` and `row_count` set to `0`;
- `claim_boundary` keeps the same remote-sensing proxy boundary used in Phase 1.

The default Phase 2 path derives a deterministic block-to-pixel mapping from the included Bishan sample. It is a lightweight feature-assembly smoke test, not real block-level DRL evidence and not a substitute for parcel/block geometry, explicit planning features, or weak-label validation. The `experiment_variants.json` file defines B0/B1/B2/B3 feature-table contracts for later DRL experiments; it does not report trained-policy performance. Ready variants are exported as CSV inputs only when all required columns are present.

Use a custom output directory for controlled verification:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --output-dir D:\tmp\paper11_phase2_outputs
```

To run Phase 2 with real planning units, provide a mapping table:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv D:\tmp\block_pixel_mapping.csv --output-dir D:\tmp\paper11_phase2_outputs
```

The mapping CSV schema is:

```text
block_id,row,col,weight
```

`weight` is optional. If omitted, every mapping row receives weight `1.0`.

To join explicit block features or weak labels, add an attributes CSV:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv D:\tmp\block_pixel_mapping.csv --attributes-csv D:\tmp\block_attributes.csv --output-dir D:\tmp\paper11_phase2_outputs
```

The attributes CSV must be keyed by `block_id`. For B0/B1/B2/B3 readiness, include all columns from `explicit_feature_00` through `explicit_feature_16`. Optional weak-label columns include `stable_farmland_label` and `high_standard_farmland_label`; optional split metadata can use `split`.

When weak-label columns are present, Phase 2 also writes `weak_label_validation.json`. This file is a diagnostic proxy check that compares `suitability_proxy` distributions against the available weak labels; it is not proof of agronomic validity or direct measurement of soil quality, fertility, or irrigation access.

The repository includes a tiny CSV fixture for this path:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir D:\tmp\paper11_phase2_csv_fixture_outputs
```

Expected outcome:

- the summary reports `mapping_mode` as `mapping_csv`;
- `n_blocks` is `4`;
- B3 readiness is `true` because the fixture includes all 17 explicit feature columns plus GeoFM embeddings and `suitability_proxy`;
- `experiment_variants.json` marks B0/B1/B2/B3 ready for the fixture;
- `variant_B0_features.csv`, `variant_B1_features.csv`, `variant_B2_features.csv`, and `variant_B3_features.csv` are created;
- the fixture `experiment_variants.json` points each ready variant to its feature table and reports `row_count` as `4`;
- `weak_label_validation.json` is created because the fixture includes `stable_farmland_label` and `high_standard_farmland_label`.

## 5. Inspect Phase 3 DRL Input Contracts

Run Phase 2 with the included fixture, then inspect B3 as a later-DRL input matrix:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir .pytest_tmp\phase3_drl_adapter_fixture
python experiments\phase3_drl_input_adapter\inspect_variant_inputs.py --phase2-output-dir .pytest_tmp\phase3_drl_adapter_fixture --variant B3
```

Expected outcome:

- the command reports variant `B3`;
- row count is `4` for the fixture;
- feature count is `82`;
- matrix shape is `4 x 82`;
- reward mode is `base_plus_suitability_reward`;
- the claim boundary states that no DRL policy is trained or evaluated.

The Phase 3 inspection command validates the input contract between Phase 2 feature assembly and later DRL experiments. It does not create a Gymnasium environment, run Stable-Baselines3, simulate actions, or report planning performance.

## 6. Run the Phase 4 DRL Smoke Environment

Run Phase 2 with the included fixture, then run one reset/step cycle for B3:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir .pytest_tmp\phase4_drl_smoke_fixture
python experiments\phase4_drl_smoke_env\run_phase4_smoke.py --phase2-output-dir .pytest_tmp\phase4_drl_smoke_fixture --variant B3
```

Expected outcome:

- the command reports variant `B3`;
- observation shape is `331` for the 4-row fixture, computed as `4 * 82 + 3`;
- action space is `Discrete(4)`;
- initial valid action count is `4`;
- selected block is `sample_block_00`;
- reward mode is `base_plus_suitability_reward`;
- the claim boundary states that Phase 4 is a DRL input-contract smoke environment.

The Phase 4 command consumes the Phase 3 `VariantInput` contract and exposes Gymnasium-compatible observation, action, action-mask, reset, and step wiring. It does not train a policy, evaluate a policy, run Stable-Baselines3 learning, simulate parcel transitions, or report planning performance.

## 7. Run the Phase 5 Rollout Protocol Smoke Check

Run Phase 2 with the included fixture, then run the deterministic masked-rollout protocol across B0/B1/B2/B3:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase5_rollout_protocol\outputs\phase2_fixture
python experiments\phase5_rollout_protocol\run_phase5_rollout.py --phase2-output-dir experiments\phase5_rollout_protocol\outputs\phase2_fixture --output-dir experiments\phase5_rollout_protocol\outputs\phase5_protocol --variants B0,B1,B2,B3
```

Expected outcome:

- the command prints one summary line for each requested variant;
- fixture feature counts are B0 = `17`, B1 = `81`, B2 = `18`, and B3 = `82`;
- B0 and B1 report zero contract reward because their reward mode is `base_planning_reward`;
- B2 and B3 report positive contract reward from `suitability_proxy` wiring;
- `phase5_rollout_summary.csv` is written;
- `phase5_rollout_steps.json` is written;
- the claim boundary states that Phase 5 is a deterministic rollout-protocol smoke check.

The Phase 5 command standardizes episode-summary and action-mask accounting across ready variants. It does not train a policy, evaluate a policy, compute planning metrics, simulate land-use transitions, or report planning performance.

## 8. Run the Phase 6 Masked Baseline Evaluator

Run Phase 2 with the included fixture, then run the non-learning masked baseline evaluator across B0/B1/B2/B3:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase6_masked_baselines\outputs\phase2_fixture
python experiments\phase6_masked_baselines\run_phase6_baselines.py --phase2-output-dir experiments\phase6_masked_baselines\outputs\phase2_fixture --output-dir experiments\phase6_masked_baselines\outputs\phase6_baselines --variants B0,B1,B2,B3 --policies first_valid,seeded_random --seed 0
```

Expected outcome:

- the command prints one summary line for each requested policy and variant;
- default policies are `first_valid` and `seeded_random`;
- fixture feature counts are B0 = `17`, B1 = `81`, B2 = `18`, and B3 = `82`;
- `seeded_random` is deterministic for the same seed;
- B0 and B1 report zero contract reward because their reward mode is `base_planning_reward`;
- B2 and B3 report positive contract reward from `suitability_proxy` wiring;
- `phase6_baseline_summary.csv` is written;
- `phase6_baseline_traces.json` is written;
- the claim boundary states that Phase 6 is a non-learning masked baseline evaluator.

The Phase 6 command checks masked baseline-evaluator plumbing only. It does not train a policy, evaluate a DRL policy, compute planning metrics, simulate land-use transitions, or report planning performance.

## 9. Run the Phase 7 MaskablePPO Compatibility Smoke Check

Run Phase 2 with the included fixture, then run the MaskablePPO compatibility smoke check for B3:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase7_maskableppo_smoke\outputs\phase2_fixture
python experiments\phase7_maskableppo_smoke\run_phase7_maskableppo_smoke.py --phase2-output-dir experiments\phase7_maskableppo_smoke\outputs\phase2_fixture --output-dir experiments\phase7_maskableppo_smoke\outputs\phase7_smoke --variant B3 --total-timesteps 8 --seed 0
```

Expected outcome:

- the command reports variant `B3`;
- observation shape is `331` for the 4-row fixture, computed as `4 * 82 + 3`;
- action space is `Discrete(4)`;
- masking support is `True`;
- predicted action validity is `True`;
- `phase7_maskableppo_smoke.json` is written;
- the claim boundary states that Phase 7 is a MaskablePPO compatibility smoke check.

This command requires the `stable-baselines3` and `sb3-contrib` dependencies listed in `requirements.txt`. It runs a tiny CPU-only `MaskablePPO.learn()` call and one masked `predict()` call to check library integration. It does not train, tune, evaluate, compare variants, report reward, or report a useful DRL policy.

## 10. Run the Phase 8 Ablation-Control Feature-Table Generator

Run Phase 2 with the included fixture, then generate deterministic D-control feature tables:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase8_ablation_controls\outputs\phase2_fixture
python experiments\phase8_ablation_controls\run_phase8_ablation_controls.py --phase2-output-dir experiments\phase8_ablation_controls\outputs\phase2_fixture --output-dir experiments\phase8_ablation_controls\outputs\phase8_controls --seed 0 --pca-dimensions 8,16
```

Expected outcome:

- the command reports generated variants `D2,D3,D4P8,D4P16`;
- D2 and D3 each report `81` features;
- D4P8 reports `25` features;
- D4P16 reports `33` features;
- `variant_D2_features.csv`, `variant_D3_features.csv`, `variant_D4P8_features.csv`, and `variant_D4P16_features.csv` are written;
- `experiment_variants.json` is written and can be consumed by the Phase 3 input loader;
- `phase8_ablation_control_summary.json` is written;
- the claim boundary states that Phase 8 builds diagnostic ablation-control feature tables.

The seed controls deterministic random and shuffled controls. The command creates table-readiness artifacts for later ablation design only. It does not train a policy, evaluate a policy, compare policy results, compute planning metrics, or report planning performance.

## 11. Run the Phase 9 Weak-Label Proxy-Validation Report

Run Phase 2 with the included fixture, then build the Phase 9 validation report:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase9_proxy_validation\outputs\phase2_fixture
python experiments\phase9_proxy_validation\run_phase9_proxy_validation.py --phase2-output-dir experiments\phase9_proxy_validation\outputs\phase2_fixture --output-dir experiments\phase9_proxy_validation\outputs\phase9_report --label-columns stable_farmland_label,high_standard_farmland_label
```

Expected outcome:

- the command reports `4` blocks for the fixture;
- `stable_farmland_label` and `high_standard_farmland_label` are listed as available labels;
- per-label rank AUC, mean difference, and interpretation are printed;
- `phase9_proxy_validation_report.json` is written;
- the report includes suitability min, max, mean, standard deviation, quartiles, and label-specific suitability quantiles;
- the claim boundary states that Phase 9 does not prove agronomic validity, train a policy, evaluate a policy, or report planning performance.

The report is a diagnostic check for directional alignment between `suitability_proxy` and available weak labels. Missing labels, one-class labels, or weak alignment are reported explicitly and should not be reframed as policy or planning-performance evidence.

## 12. Run the Phase 10 Suitability Reward-Readiness Gate

Run Phase 2 with the included fixture, build the Phase 9 report, then build the
Phase 10 gate:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase10_reward_readiness\outputs\phase2_fixture
python experiments\phase9_proxy_validation\run_phase9_proxy_validation.py --phase2-output-dir experiments\phase10_reward_readiness\outputs\phase2_fixture --output-dir experiments\phase10_reward_readiness\outputs\phase9_report --label-columns stable_farmland_label,high_standard_farmland_label
python experiments\phase10_reward_readiness\run_phase10_reward_readiness.py --phase9-report experiments\phase10_reward_readiness\outputs\phase9_report\phase9_proxy_validation_report.json --output-dir experiments\phase10_reward_readiness\outputs\phase10_gate --required-labels stable_farmland_label,high_standard_farmland_label
```

Expected outcome for the included tiny fixture:

- `phase10_reward_readiness_gate.json` is written;
- status is `not_ready_for_suitability_reward`;
- recommendation is `do_not_enable_suitability_reward`;
- both included weak labels fail the gate because Phase 9 reports `negative_or_no_alignment`;
- the claim boundary states that Phase 10 does not train, tune, evaluate, or report a DRL policy and does not prove agronomic validity.

The Phase 10 gate is an executable guardrail for later reward experiments. It
does not make a planning-performance claim; it records whether current weak
label diagnostics are strong enough to permit later bounded suitability-reward
smoke tests.

## 13. Run the Phase 11 Real Bishan DLTB Adapter

This path uses the local real Bishan DLTB-with-slope GeoPackage from the
previous paper workspace. The file is not committed because it is a large
external geospatial artifact with separate provenance.

Run the adapter:

```powershell
python experiments\phase11_bishan_dltb_real\run_phase11_bishan_dltb_adapter.py --dltb-path D:\test\dem_slope_analysis\output\DLTB_with_slope.gpkg --output-dir experiments\phase11_bishan_dltb_real\outputs\adapter
```

Expected local outcome for the current Bishan source:

- `block_pixel_mapping.csv` is written;
- `block_attributes.csv` is written;
- `phase11_bishan_dltb_adapter_summary.json` is written;
- `rows_read_in_bbox` is `65146`;
- `rows_exported` is `64984`;
- category counts are `Other = 27988`, `Farmland = 25359`, `Forest = 8717`, and `Orchard = 2920`;
- weak-label positive counts are `current_farmland_label = 25359`, `low_slope_farmland_label = 7443`, and `farmland_or_orchard_label = 28279`.

Then run the real-data Phase 2, Phase 9, and Phase 10 chain:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv experiments\phase11_bishan_dltb_real\outputs\adapter\block_pixel_mapping.csv --attributes-csv experiments\phase11_bishan_dltb_real\outputs\adapter\block_attributes.csv --output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real
python experiments\phase9_proxy_validation\run_phase9_proxy_validation.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase11_bishan_dltb_real\outputs\phase9_real --label-columns current_farmland_label,low_slope_farmland_label,farmland_or_orchard_label
python experiments\phase10_reward_readiness\run_phase10_reward_readiness.py --phase9-report experiments\phase11_bishan_dltb_real\outputs\phase9_real\phase9_proxy_validation_report.json --output-dir experiments\phase11_bishan_dltb_real\outputs\phase10_real --required-labels current_farmland_label,low_slope_farmland_label,farmland_or_orchard_label
```

Observed local Phase 9/10 outcome:

- `current_farmland_label` reports weak positive proxy alignment;
- `farmland_or_orchard_label` reports weak positive proxy alignment;
- `low_slope_farmland_label` reports `negative_or_no_alignment`;
- Phase 10 status is `not_ready_for_suitability_reward`;
- Phase 10 recommendation is `do_not_enable_suitability_reward`.

This confirms the Bishan DLTB data can drive real Phase 2 feature-table
experiments. It does not enable a suitability reward under the current strict
all-label gate, and it does not train, evaluate, or report a DRL policy.

## 14. Run the Phase 12 Real DLTB Scale Audit

After the Phase 11 real-data chain has produced Phase 11, Phase 2, Phase 9,
and Phase 10 artifacts, run:

```powershell
python experiments\phase12_real_scale_audit\run_phase12_real_scale_audit.py --phase11-summary experiments\phase11_bishan_dltb_real\outputs\adapter\phase11_bishan_dltb_adapter_summary.json --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase9-report experiments\phase11_bishan_dltb_real\outputs\phase9_real\phase9_proxy_validation_report.json --phase10-gate experiments\phase11_bishan_dltb_real\outputs\phase10_real\phase10_reward_readiness_gate.json --output-dir experiments\phase12_real_scale_audit\outputs\real_bishan
```

Expected current Bishan outcome:

- `phase12_real_dltb_scale_audit.json` is written;
- blocks are `64984`;
- maximum flat observation dimension is `5328691`;
- `real_feature_tables_ready` is `true`;
- `representation_only_smoke_allowed` is `true`;
- `suitability_reward_allowed` is `false`;
- `flat_full_scale_training_ready` is `false`;
- `requires_tiled_or_hierarchical_env` is `true`.

Phase 12 is a downstream-readiness audit. It preserves the Phase 10 reward
gate, does not read the large DLTB GeoPackage, does not train a policy, and
does not report planning performance.

## 15. Run the Phase 13 Tiled Real-Data Contract Builder

After Phase 11 has produced `block_pixel_mapping.csv` and Phase 2 has produced
`experiment_variants.json`, run:

```powershell
python experiments\phase13_tiled_real_contract\run_phase13_tiled_real_contract.py --mapping-csv experiments\phase11_bishan_dltb_real\outputs\adapter\block_pixel_mapping.csv --variant-manifest experiments\phase11_bishan_dltb_real\outputs\phase2_real\experiment_variants.json --output-dir experiments\phase13_tiled_real_contract\outputs\real_bishan --tile-rows 8 --tile-cols 8
```

Expected current Bishan outcome:

- `phase13_tile_index.csv` is written;
- `phase13_tiled_real_contract.json` is written;
- total blocks are `64984`;
- non-empty tiles are `54`;
- maximum blocks per tile are `2234`;
- B3 maximum tiled observation dimension is `183191`;
- `all_tiles_within_observation_threshold` is `true`;
- `tiled_contract_ready` is `true`.

Phase 13 answers the scale issue raised by Phase 12 by creating tractable
tile-level episode metadata. It does not train a policy, enable suitability
reward, or report planning performance.

## 16. Run the Phase 14 Tile-Level Smoke Environment

After Phase 13 has produced `phase13_tile_index.csv`, run the largest current
Bishan tile with the B1 representation-only variant:

```powershell
python experiments\phase14_tiled_smoke_env\run_phase14_tiled_smoke.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --tile-id tile_r003_c003 --variant B1 --output-dir experiments\phase14_tiled_smoke_env\outputs\real_bishan_largest_tile
```

Expected current Bishan outcome:

- `phase14_tiled_smoke_summary.json` is written;
- tile ID is `tile_r003_c003`;
- variant is `B1`;
- rows are `2234`;
- features are `81`;
- observation shape is `180957`;
- action space is `Discrete(2234)`;
- reward mode is `base_planning_reward`;
- one-step reward for the first selected block is `-0.197259`.

Phase 14 verifies that a real Phase 13 tile can be loaded into the existing
Phase 4 environment contract and that the Phase 19 base planning reward can be
computed on real B1 features. It does not train a policy, enable suitability
reward, or report planning performance.

## 17. Run the Phase 15 All-Tile Batch Smoke Check

After Phase 13 has produced the real tile index, run B1 across all tiles:

```powershell
python experiments\phase15_tiled_batch_smoke\run_phase15_tiled_batch_smoke.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variant B1 --output-dir experiments\phase15_tiled_batch_smoke\outputs\real_bishan_all_tiles
```

Expected current Bishan outcome:

- `phase15_tiled_batch_smoke_summary.csv` is written;
- `phase15_tiled_batch_smoke_report.json` is written;
- tiles processed are `54`;
- total blocks are `64984`;
- maximum observation shape is `180957`;
- all tile smoke checks pass.

Phase 15 verifies that every real Phase 13 tile can run the representation-only
B1 input-contract smoke step. It does not train a policy, enable suitability
reward, or report planning performance.

## 18. Run the Phase 16 Tiled Non-Learning Baseline Protocol

After Phase 15 has confirmed all real tiles pass the input contract, run short
deterministic masked rollouts over the same real tile index:

```powershell
python experiments\phase16_tiled_baseline_protocol\run_phase16_tiled_baselines.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variant B1 --policies first_valid,seeded_random --max-steps 4 --seed 0 --output-dir experiments\phase16_tiled_baseline_protocol\outputs\real_bishan_b1
```

Expected current Bishan outcome:

- `phase16_tiled_baseline_summary.csv` is written;
- `phase16_tiled_baseline_traces.json` is written;
- tiles processed are `54`;
- policies processed are `2`;
- summary rows are `108`;
- total blocks are `64984`;
- maximum observation shape is `180957`;
- all rollouts complete;
- B1 total contract reward accumulates the deterministic base planning reward.

Phase 16 verifies deterministic non-learning masked rollouts across every real
Phase 13 tile. It does not train, tune, evaluate, or compare a DRL policy, does
not enable suitability reward, and does not report planning performance.

## 19. Run the Phase 17 Tiled MaskablePPO Readiness Smoke Check

After Phase 16 has confirmed deterministic non-learning rollouts on real tiles,
run one tiny MaskablePPO compatibility smoke check on the largest real tile:

```powershell
python experiments\phase17_tiled_maskableppo_readiness\run_phase17_tiled_maskableppo_readiness.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variant B1 --tile-selection largest --total-timesteps 8 --seed 0 --output-dir experiments\phase17_tiled_maskableppo_readiness\outputs\real_bishan_largest_tile
```

Expected current Bishan outcome:

- `phase17_tiled_maskableppo_readiness.json` is written;
- selected tile is `tile_r003_c003`;
- variant is `B1`;
- blocks are `2234`;
- features are `81`;
- observation shape is `180957`;
- action space is `Discrete(2234)`;
- masking support is `true`;
- predicted action is valid;
- readiness status is `passed_tiled_maskableppo_smoke`.

Phase 17 verifies that the real tiled B1 contract can be consumed by the
MaskablePPO API for a tiny CPU-only compatibility smoke. It does not train,
tune, evaluate, or compare a useful DRL policy, does not enable suitability
reward, and does not report planning performance.

## 20. Run the Phase 18 Planning-Reward Readiness Gate

After Phase 17 has produced the optional tiled MaskablePPO readiness artifact,
run the planning-reward readiness gate:

```powershell
python experiments\phase18_planning_reward_readiness\run_phase18_planning_reward_readiness.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase10-gate experiments\phase11_bishan_dltb_real\outputs\phase10_real\phase10_reward_readiness_gate.json --phase12-audit experiments\phase12_real_scale_audit\outputs\real_bishan\phase12_real_dltb_scale_audit.json --phase17-readiness experiments\phase17_tiled_maskableppo_readiness\outputs\real_bishan_largest_tile\phase17_tiled_maskableppo_readiness.json --output-dir experiments\phase18_planning_reward_readiness\outputs\real_bishan
```

Expected current Bishan outcome:

- `phase18_planning_reward_readiness.json` is written;
- base planning reward implemented is `True`;
- suitability reward allowed is `False`;
- flat full-scale training ready is `False`;
- tiled MaskablePPO API ready is `True` when the Phase 17 artifact is supplied;
- performance experiment ready is `False`;
- recommendation is `resolve_suitability_reward_gate_before_suitability_reward_experiments`.

Phase 18 records that the current repository is not ready for true planning
performance experiments. After Phase 19, B0/B1 have a deterministic
`base_planning_reward`, but suitability reward remains disabled and full-scale
flat training remains blocked. Phase 18 does not train, tune, evaluate, or
compare a DRL policy, enable suitability reward, or report planning
performance.

## 21. Inspect the Phase 19 Base Planning Reward

The base planning reward implementation is:

```text
src/paper11_geofm/planning_reward.py
```

It is a bounded weighted score over explicit Phase 11 planning features: area,
mean slope, maximum slope, current farmland/orchard indicators, low-slope
indicators, built-up, and water. It is the first executable base reward for B0
and B1 contracts. It is not a calibrated policy objective and is not planning
performance evidence.

## 22. Run the Phase 20 Bounded Same-Tile B0/B1 Training Pilot

After Phase 13 has produced the real tile index and Phase 19 has provided the
deterministic base planning reward, run the bounded B0/B1 same-tile pilot:

```powershell
python experiments\phase20_bounded_tiled_training\run_phase20_bounded_tiled_training.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --total-timesteps 8 --eval-max-steps 4 --seed 0 --output-dir experiments\phase20_bounded_tiled_training\outputs\real_bishan_pilot
```

Expected current Bishan outcome:

- `phase20_bounded_tiled_training_summary.csv` is written;
- `phase20_bounded_tiled_training_traces.json` is written;
- train tile is `tile_r003_c003`;
- evaluation tile is also `tile_r003_c003`;
- cross-tile learned-policy status is `blocked_variable_observation_shape`;
- variants are `B0` and `B1`;
- summary rows are `6`;
- training completed is `True`;
- all evaluations completed is `True`.

Phase 20 verifies that a bounded B0/B1 MaskablePPO training/evaluation protocol
can execute under the deterministic base planning reward on a real tiled
episode. It is a same-tile pilot only. The current flat observation and action
spaces vary with tile block count, so learned-policy cross-tile evaluation is
blocked until a variable-size, padded, or per-block policy design exists.
Phase 20 does not enable suitability reward, test transfer, compare final
policy performance, or support submission-level planning-performance claims.

## 23. Run the Phase 21 Cross-Tile Per-Block Scorer Pilot

After Phase 20 has recorded the flat-observation cross-tile blocker, run the
dimension-agnostic per-block scorer pilot:

```powershell
python experiments\phase21_cross_tile_block_scorer\run_phase21_cross_tile_block_scorer.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --ridge-alpha 1e-6 --eval-max-steps 4 --seed 0 --output-dir experiments\phase21_cross_tile_block_scorer\outputs\real_bishan_pilot
```

Expected current Bishan outcome:

- `phase21_cross_tile_block_scorer_summary.csv` is written;
- `phase21_cross_tile_block_scorer_traces.json` is written;
- train tile is `tile_r003_c003`;
- evaluation tile is the distinct tile `tile_r002_c003`;
- cross-tile learned-policy status is `executed_distinct_tile`;
- variants are `B0` and `B1`;
- summary rows are `6`;
- all evaluations completed is `True`.

Phase 21 trains a standardized ridge-linear per-block scorer on the train tile
and evaluates it on a distinct evaluation tile. It verifies that the learned
policy interface can be made independent of tile block count. It does not
enable suitability reward, run PPO training, prove cross-region transfer,
compare final policy performance, or support submission-level planning claims.

## 24. Run the Phase 22 Multi-Tile Multi-Seed Scorer Evaluation

After Phase 21 has verified a one-train-tile and one-evaluation-tile per-block
scorer interface, run the bounded multi-tile, multi-seed evaluation pilot:

```powershell
python experiments\phase22_multi_tile_scorer_eval\run_phase22_multi_tile_scorer_eval.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --ridge-alpha 1e-6 --eval-max-steps 4 --seeds 0,1 --max-eval-tiles 2 --output-dir experiments\phase22_multi_tile_scorer_eval\outputs\real_bishan_pilot
```

Expected current Bishan outcome:

- `phase22_multi_tile_scorer_eval_summary.csv` is written;
- `phase22_multi_tile_scorer_eval_traces.json` is written;
- train tile is `tile_r003_c003`;
- evaluation tiles are `tile_r002_c003` and `tile_r005_c004`;
- variants are `B0` and `B1`;
- seeds are `0` and `1`;
- summary rows are `24`;
- all evaluations completed is `True`.

Phase 22 fits the standardized ridge-linear per-block scorer once per variant
on the selected train tile, then evaluates learned-scorer, first-valid, and
seeded-random policies across selected evaluation tiles and seeds. It broadens
the Phase 21 interface pilot, but it does not enable suitability reward, run
PPO training, prove transfer, compare final policy performance, or support
submission-level planning claims.

## 25. Run the Phase 23 Multi-Seed Same-Tile B0/B1 Training Pilot

After Phase 22 has broadened the cross-tile scorer interface pilot, run the
bounded multi-seed same-tile B0/B1 MaskablePPO training pilot:

```powershell
python experiments\phase23_multi_seed_training\run_phase23_multi_seed_training.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --total-timesteps 8 --eval-max-steps 4 --seeds 0,1,2 --output-dir experiments\phase23_multi_seed_training\outputs\real_bishan_pilot
```

Expected current Bishan outcome:

- `phase23_multi_seed_training_summary.csv` is written;
- `phase23_multi_seed_training_traces.json` is written;
- `phase23_multi_seed_training_comparison.json` is written;
- train tile is `tile_r003_c003`;
- evaluation tile is also `tile_r003_c003`;
- seeds are `0`, `1`, and `2`;
- variants are `B0` and `B1`;
- summary rows are `18`;
- all evaluations completed is `True`;
- B1-B0 learned-policy mean reward delta is `0.4273019432` for the short pilot.

Phase 23 strengthens the learned-policy evidence by repeating the bounded
same-tile B0/B1 MaskablePPO protocol across multiple seeds. It remains a pilot:
the current flat observation/action spaces still block cross-tile learned-policy
evaluation, suitability reward remains disabled, and final planning-performance
or transfer claims still require longer training, ablations, spatial maps, and
held-out-region tests.

## 26. Build the Phase 24 IJAEOG Evidence Package

After Phase 22 and Phase 23 outputs exist, build the IJAEOG claim-readiness
evidence package:

```powershell
python experiments\phase24_ijaeog_evidence_package\run_phase24_ijaeog_evidence_package.py --phase22-summary-csv experiments\phase22_multi_tile_scorer_eval\outputs\real_bishan_pilot\phase22_multi_tile_scorer_eval_summary.csv --phase23-summary-csv experiments\phase23_multi_seed_training\outputs\real_bishan_pilot\phase23_multi_seed_training_summary.csv --phase23-comparison-json experiments\phase23_multi_seed_training\outputs\real_bishan_pilot\phase23_multi_seed_training_comparison.json --output-dir experiments\phase24_ijaeog_evidence_package\outputs\real_bishan
```

Expected current Bishan outcome:

- `phase24_ijaeog_evidence_table.csv` is written;
- `phase24_ijaeog_evidence_summary.json` is written;
- `phase24_ijaeog_claim_readiness.md` is written;
- Phase 22 summary rows are `24`;
- Phase 23 summary rows are `18`;
- B1-B0 learned-policy mean reward delta is `0.4273019432`;
- same-tile B0/B1 training pilot readiness is `pilot_supported`;
- multi-tile scorer interface readiness is `pilot_supported`;
- suitability reward readiness is `not_ready`;
- transfer readiness is `not_ready`;
- submission readiness is `not_ready`.

Phase 24 is a synthesis and claim-readiness package. It summarizes current
Phase 22/23 pilot evidence and remaining gaps, but it does not create new
policy-performance, transfer, or suitability-reward evidence.

## 27. Run the Phase 25 Padded Held-Out Policy Pilot

After Phase 13 has produced the real tile index and Phase 19 has provided the
deterministic base planning reward, run the Windows smoke command:

```powershell
python experiments\phase25_padded_heldout_policy\run_phase25_padded_heldout_policy.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --total-timesteps 32 --eval-max-steps 4 --seeds 0 --max-eval-tiles 1 --output-dir experiments\phase25_padded_heldout_policy\outputs\real_bishan_smoke
```

Expected current Bishan smoke outcome:

- `phase25_padded_heldout_policy_summary.csv` is written;
- `phase25_padded_heldout_policy_traces.json` is written;
- `phase25_padded_heldout_policy_comparison.json` is written;
- train tile is `tile_r003_c003`;
- held-out evaluation tile is `tile_r002_c003`;
- padded max blocks is `2234`;
- variants are `B0` and `B1`;
- seed is `0`;
- total timesteps is `32`;
- evaluation max steps is `4`;
- summary rows are `6`;
- all evaluations completed is `True`;
- B1-B0 held-out learned-policy mean reward delta is `1.3314600457`;
- pilot result status is `B1_improves_B0`;
- the comparison JSON keeps `suitability_reward_validation_before_B2_B3` in
  the remaining evidence gaps.

For the Colab Pro+ main run, use:

```bash
python experiments/phase25_padded_heldout_policy/run_phase25_padded_heldout_policy.py --phase2-output-dir experiments/phase11_bishan_dltb_real/outputs/phase2_real --tile-index-csv experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv --variants B0,B1 --total-timesteps 1024 --eval-max-steps 8 --seeds 0,1,2 --max-eval-tiles 3 --output-dir experiments/phase25_padded_heldout_policy/outputs/real_bishan_colab_main
```

Windows is the implementation, test, schema-check, and short-smoke platform.
Colab Pro+ is the intended platform for the multi-seed, multi-held-out-tile
main run. A timing probe can adjust `--total-timesteps` within `512` to `4096`
before the main run budget is fixed.

Phase 25 uses `base_planning_reward` only. Do not enable B2/B3 or suitability
reward in Phase 25; B2/B3 remain blocked until Phase 10 suitability reward
readiness is resolved. The padded variable-size contract removes the Phase
20/23 flat observation/action shape blocker for held-out Bishan tile evaluation,
but Phase 25 remains a bounded B0/B1 pilot and does not support cross-region
transfer or submission-level planning-performance claims.

## 28. Run the Phase 26 Main Empirical Analysis Package

Phase 26 turns Phase 25 padded held-out B0/B1 outputs into manuscript-facing
multi-seed, multi-held-out-tile tables. Unit tests use synthetic fixtures and
do not run long RL training.

For a local Windows timing probe, use:

```powershell
python experiments\phase26_main_experiment\run_phase26_main_experiment.py --mode run-and-analyze --phase25-output-dir experiments\phase26_main_experiment\outputs\windows_timing_probe\phase25_run --output-dir experiments\phase26_main_experiment\outputs\windows_timing_probe\phase26_analysis --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --total-timesteps 128 --eval-max-steps 4 --seeds 0 --max-eval-tiles 1
```

The timing probe requires existing real Phase 11/13 outputs. In the isolated
implementation worktree used for Phase 26, those generated real-data outputs
were not present, so the local timing probe was skipped and only synthetic
tests were run.

For the Colab Pro+ main Phase 25 run, use:

```bash
python experiments/phase25_padded_heldout_policy/run_phase25_padded_heldout_policy.py --phase2-output-dir experiments/phase11_bishan_dltb_real/outputs/phase2_real --tile-index-csv experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv --variants B0,B1 --total-timesteps 1024 --eval-max-steps 8 --seeds 0,1,2 --max-eval-tiles 3 --output-dir experiments/phase26_main_experiment/outputs/colab_main/phase25_run
```

If the timing probe is stable and runtime allows, repeat the Colab command with
`--total-timesteps 4096`.

After Phase 25 outputs have been produced or copied back, run analysis-only:

```powershell
python experiments\phase26_main_experiment\run_phase26_main_experiment.py --mode analyze-only --phase25-output-dir experiments\phase26_main_experiment\outputs\colab_main\phase25_run --output-dir experiments\phase26_main_experiment\outputs\colab_main\phase26_analysis
```

Expected Phase 26 artifacts:

- `phase26_main_summary.csv`;
- `phase26_tile_seed_delta_table.csv`;
- `phase26_main_comparison.json`;
- `phase26_claim_readiness.md`.

The current macOS Phase 26 result artifacts are available under
`experiments/phase26_main_experiment/outputs/macos_main/` and
`experiments/phase26_main_experiment/outputs/macos_main_4096/`. The 1024-step
result reports a B1-B0 learned-policy mean reward delta of `-0.4329022862`
with `4 / 9` positive tile-seed pairs. The 4096-step result reports
`-0.1318712688` with `3 / 9` positive tile-seed pairs. Both have claim status
`not_supported`.

Phase 26 therefore reports B1-B0 learned-policy deltas across held-out Bishan
tiles and random seeds under the deterministic base planning reward, but the
current evidence does not support a positive B1-over-B0 learned-policy claim.
It does not enable suitability reward, test B2/B3, demonstrate cross-region
transfer, or support submission-level planning-performance claims.

## 29. Run the Phase 27 B0/B1 Stability Diagnosis

Phase 27 is a read-only diagnosis over the existing Phase 26 comparison JSON
artifacts. It does not run RL training.

Run:

```powershell
python experiments\phase27_stability_diagnosis\run_phase27_stability_diagnosis.py --phase26-comparison-json experiments\phase26_main_experiment\outputs\macos_main\phase26_analysis\phase26_main_comparison.json --phase26-comparison-json experiments\phase26_main_experiment\outputs\macos_main_4096\phase26_analysis\phase26_main_comparison.json --output-dir experiments\phase27_stability_diagnosis\outputs\macos_1024_vs_4096
```

Expected Phase 27 artifacts:

- `phase27_budget_transition_table.csv`;
- `phase27_tile_seed_stability.csv`;
- `phase27_diagnostic_summary.json`;
- `phase27_diagnostic_readiness.md`.

Current observed outcome:

- diagnostic status: `budget_not_explanatory`;
- 1024-step B1-B0 mean delta: `-0.4329022862`;
- 4096-step B1-B0 mean delta: `-0.1318712688`;
- mean delta change: `0.3010310174`;
- positive tile-seed count change: `-1`;
- stability counts: stable-positive `1`, stable-negative `3`,
  flip-to-positive `2`, flip-to-negative `3`, incomplete `0`.

Phase 27 confirms that budget sensitivity exists, but budget alone does not
explain or resolve the current negative B1 evidence. The next recommended work
is representation controls and repeated/intermediate budget stability, not a
positive learned-policy claim.

## 30. Run the Phase 28 Representation-Control Evaluation

Phase 28 evaluates whether B1 differs from B0 and D2/D3/D4 representation
controls under the same padded held-out base-reward protocol. It consumes
Phase 2 B0/B1 outputs, Phase 8 D-control outputs, and the Phase 13 tile index.

Run-and-analyze mode:

```powershell
python experiments\phase28_representation_controls\run_phase28_representation_controls.py --mode run-and-analyze --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1,D2,D3,D4P8,D4P16 --total-timesteps 1024 --eval-max-steps 8 --seeds 0,1,2 --max-eval-tiles 3 --output-dir experiments\phase28_representation_controls\outputs\real_bishan_main
```

Analyze-only mode:

```powershell
python experiments\phase28_representation_controls\run_phase28_representation_controls.py --mode analyze-only --existing-summary-csv experiments\phase28_representation_controls\outputs\real_bishan_main\phase28_representation_control_summary.csv --output-dir experiments\phase28_representation_controls\outputs\real_bishan_analysis
```

Expected Phase 28 artifacts:

- `phase28_representation_control_summary.csv`;
- `phase28_representation_control_traces.json`;
- `phase28_representation_control_comparison.json`;
- `phase28_tile_seed_delta_table.csv`;
- `phase28_control_readiness.md`.

`run-and-analyze` requires explicit `--variants`, `--total-timesteps`,
`--eval-max-steps`, `--seeds`, and either `--max-eval-tiles` or
`--eval-tile-ids`. Phase 28 is diagnostic only: it does not enable suitability
reward, test B2/B3, test cross-region transfer, or support final
submission-level planning-performance claims.

## 31. Inspect the Paper11 Design

Read these files in order:

```text
paper/design/01_design_synthesis.md
paper/design/02_system_design.md
paper/design/03_experiment_plan.md
paper/design/04_manuscript_outline.md
paper/design/05_risks_and_boundaries.md
```

The design intentionally keeps Paper11 within current-state suitability representation and DRL layout optimization.

## 32. Inspect Runtime Code

Important copied runtime files:

```text
experiments/geofm_runtime/embedding_space_env.py
experiments/geofm_runtime/train_embedding_rl.py
experiments/geofm_runtime/dual_rep_env.py
experiments/geofm_runtime/train_dual_rep.py
src/legacy_runtime/county_env.py
src/legacy_runtime/parcel_scoring_policy.py
```

These scripts preserve the original Paper58/Paper8 development code. Some scripts still contain historical path assumptions from the source workspace, especially for Google Earth Engine helpers, model weights, or `data` directories. The smoke check is the stable reviewer entry point for this repository snapshot.

Phase 1 executable files:

```text
experiments/phase1_bishan_baseline/run_phase1.py
src/paper11_geofm/
```

Phase 2 executable files:

```text
experiments/phase2_block_geofm_features/run_phase2.py
src/paper11_geofm/block_mapping.py
src/paper11_geofm/block_features.py
src/paper11_geofm/block_schema.py
```

Phase 3 executable files:

```text
experiments/phase3_drl_input_adapter/inspect_variant_inputs.py
src/paper11_geofm/drl_inputs.py
```

Phase 4 executable files:

```text
experiments/phase4_drl_smoke_env/run_phase4_smoke.py
src/paper11_geofm/drl_smoke_env.py
```

Phase 5 executable files:

```text
experiments/phase5_rollout_protocol/run_phase5_rollout.py
src/paper11_geofm/rollout_smoke.py
```

Phase 6 executable files:

```text
experiments/phase6_masked_baselines/run_phase6_baselines.py
src/paper11_geofm/baseline_eval.py
```

Phase 7 executable files:

```text
experiments/phase7_maskableppo_smoke/run_phase7_maskableppo_smoke.py
src/paper11_geofm/maskableppo_smoke.py
```

Phase 8 executable files:

```text
experiments/phase8_ablation_controls/run_phase8_ablation_controls.py
src/paper11_geofm/ablation_controls.py
```

Phase 9 executable files:

```text
experiments/phase9_proxy_validation/run_phase9_proxy_validation.py
src/paper11_geofm/proxy_validation.py
```

Phase 10 executable files:

```text
experiments/phase10_reward_readiness/run_phase10_reward_readiness.py
src/paper11_geofm/reward_readiness.py
```

Phase 11 executable files:

```text
experiments/phase11_bishan_dltb_real/run_phase11_bishan_dltb_adapter.py
src/paper11_geofm/dltb_adapter.py
```

Phase 12 executable files:

```text
experiments/phase12_real_scale_audit/run_phase12_real_scale_audit.py
src/paper11_geofm/real_scale_audit.py
```

Phase 13 executable files:

```text
experiments/phase13_tiled_real_contract/run_phase13_tiled_real_contract.py
src/paper11_geofm/tiled_contract.py
```

Phase 14 executable files:

```text
experiments/phase14_tiled_smoke_env/run_phase14_tiled_smoke.py
src/paper11_geofm/tiled_inputs.py
```

Phase 15 executable files:

```text
experiments/phase15_tiled_batch_smoke/run_phase15_tiled_batch_smoke.py
src/paper11_geofm/tiled_batch_smoke.py
```

Phase 16 executable files:

```text
experiments/phase16_tiled_baseline_protocol/run_phase16_tiled_baselines.py
src/paper11_geofm/tiled_baseline_protocol.py
```

Phase 17 executable files:

```text
experiments/phase17_tiled_maskableppo_readiness/run_phase17_tiled_maskableppo_readiness.py
src/paper11_geofm/tiled_maskableppo_readiness.py
```

Phase 18 executable files:

```text
experiments/phase18_planning_reward_readiness/run_phase18_planning_reward_readiness.py
src/paper11_geofm/planning_reward_readiness.py
```

Phase 19 executable files:

```text
src/paper11_geofm/planning_reward.py
```

Phase 20 executable files:

```text
experiments/phase20_bounded_tiled_training/run_phase20_bounded_tiled_training.py
src/paper11_geofm/bounded_tiled_training.py
```

Phase 21 executable files:

```text
experiments/phase21_cross_tile_block_scorer/run_phase21_cross_tile_block_scorer.py
src/paper11_geofm/cross_tile_block_scorer.py
```

Phase 22 executable files:

```text
experiments/phase22_multi_tile_scorer_eval/run_phase22_multi_tile_scorer_eval.py
src/paper11_geofm/multi_tile_scorer_eval.py
```

Phase 23 executable files:

```text
experiments/phase23_multi_seed_training/run_phase23_multi_seed_training.py
src/paper11_geofm/multi_seed_training.py
```

Phase 24 executable files:

```text
experiments/phase24_ijaeog_evidence_package/run_phase24_ijaeog_evidence_package.py
src/paper11_geofm/ijaeog_evidence_package.py
```

Phase 25 executable files:

```text
experiments/phase25_padded_heldout_policy/run_phase25_padded_heldout_policy.py
src/paper11_geofm/padded_heldout_policy.py
```

Phase 26 executable files:

```text
experiments/phase26_main_experiment/run_phase26_main_experiment.py
src/paper11_geofm/phase26_main_experiment.py
```

Phase 27 executable files:

```text
experiments/phase27_stability_diagnosis/run_phase27_stability_diagnosis.py
src/paper11_geofm/phase27_stability_diagnosis.py
```

Phase 28 executable files:

```text
experiments/phase28_representation_controls/run_phase28_representation_controls.py
src/paper11_geofm/phase28_representation_controls.py
```

The Phase 1, Phase 2, Phase 3, Phase 4, Phase 5, Phase 6, Phase 7, Phase 8, Phase 9, Phase 10, Phase 11, Phase 12, Phase 13, Phase 14, Phase 15, Phase 16, Phase 17, Phase 18, Phase 19, Phase 20, Phase 21, Phase 22, Phase 23, Phase 24, Phase 25, Phase 26 analysis-only, Phase 27, and Phase 28 analyze-only reviewer paths are deterministic and do not require internet, GPU, Earth Engine, or full-scale DRL training. Phase 11 additionally requires a local copy of the external Bishan DLTB GeoPackage, and Phase 26/28 run-and-analyze modes require the real Phase 11/13 outputs plus the RL stack.

## 33. Regenerate Embeddings

The included Bishan arrays are cached samples from:

```text
GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
```

To regenerate or extend them, inspect:

```text
experiments/geofm_runtime/extract_bishan_embeddings.py
experiments/geofm_runtime/extract_village_embeddings.py
```

These extraction scripts require Google Earth Engine authentication and may need local path edits for the target machine.

## 34. Large Data and Weights

Large arrays, model weights, and intervention transition files are not included in ordinary Git. See `DATA_MANIFEST.md` for what was deliberately included and excluded.

For a full artifact release, place large files in Git LFS or an external archive and record checksums in the data manifest.
