# Paper11: GeoFM-Enhanced Farmland Suitability RL

This repository is the standalone reviewer package for Paper11:

**GeoFM-enhanced current-state farmland suitability representation and DRL spatial layout optimization.**

The repository separates Paper11 from the broader Paper58 workspace so reviewers can inspect the design, copied runtime code, lightweight sample data, and reproduction entry points without needing the original experiment directory.

## Scope Boundary

Paper11 uses AlphaEarth or other GeoFM embeddings as latent remote-sensing proxies for land-surface and environmental conditions related to farmland suitability. It does not claim that AlphaEarth directly measures soil quality, fertility, or irrigation access.

Paper11 is also distinct from future-aware planning work. The target framing is current-state farmland suitability representation plus reinforcement-learning layout optimization, not prediction-then-optimization.

## Repository Layout

- `paper/design/`: Paper11 design package, including system design, experiment plan, manuscript outline, and risk boundaries.
- `paper/phase1_results/`: interpretation of the executable Phase 1 baseline and the next experiment matrix.
- `paper/phase26_results/`: interpretation of the current Phase 26 empirical package and the next diagnostic matrix.
- `paper/phase27_results/`: interpretation of the Phase 27 B0/B1 budget and tile-seed stability diagnosis.
- `paper/phase28_results/`: interpretation of the Phase 28 B0/B1/D2/D3/D4 representation-control diagnosis.
- `paper/submission/`: IJAEOG submission-readiness audit and guarded submission text drafts.
- `docs/source_notes/`: original design notes used to derive the Paper11 package.
- `experiments/geofm_runtime/`: copied GeoFM and embedding-space experiment scripts from the source Paper58 workspace.
- `experiments/phase1_bishan_baseline/`: executable Phase 1 Bishan GeoFM representation baseline.
- `experiments/phase2_block_geofm_features/`: executable Phase 2 block-level GeoFM feature assembly baseline.
- `experiments/phase3_drl_input_adapter/`: executable Phase 3 DRL input-contract inspection runner.
- `experiments/phase4_drl_smoke_env/`: executable Phase 4 one-step DRL input-contract smoke runner.
- `experiments/phase5_rollout_protocol/`: executable Phase 5 deterministic rollout protocol smoke runner.
- `experiments/phase6_masked_baselines/`: executable Phase 6 non-learning masked baseline evaluator.
- `experiments/phase7_maskableppo_smoke/`: executable Phase 7 MaskablePPO compatibility smoke runner.
- `experiments/phase8_ablation_controls/`: executable Phase 8 ablation-control feature-table generator.
- `experiments/phase9_proxy_validation/`: executable Phase 9 weak-label proxy-validation report runner.
- `experiments/phase10_reward_readiness/`: executable Phase 10 suitability reward-readiness gate runner.
- `experiments/phase11_bishan_dltb_real/`: executable Phase 11 real Bishan DLTB adapter and local real-data workflow.
- `experiments/phase12_real_scale_audit/`: executable Phase 12 real DLTB scale audit and downstream-readiness gate.
- `experiments/phase13_tiled_real_contract/`: executable Phase 13 tiled real-data contract builder.
- `experiments/phase14_tiled_smoke_env/`: executable Phase 14 tile-level one-step smoke environment.
- `experiments/phase15_tiled_batch_smoke/`: executable Phase 15 all-tile batch smoke runner.
- `experiments/phase16_tiled_baseline_protocol/`: executable Phase 16 tiled non-learning baseline protocol runner.
- `experiments/phase17_tiled_maskableppo_readiness/`: executable Phase 17 tiled MaskablePPO readiness smoke runner.
- `experiments/phase18_planning_reward_readiness/`: executable Phase 18 planning-reward readiness gate runner.
- `experiments/phase20_bounded_tiled_training/`: executable Phase 20 bounded same-tile B0/B1 training pilot runner.
- `experiments/phase21_cross_tile_block_scorer/`: executable Phase 21 cross-tile per-block scorer pilot runner.
- `experiments/phase22_multi_tile_scorer_eval/`: executable Phase 22 multi-tile, multi-seed per-block scorer evaluation runner.
- `experiments/phase23_multi_seed_training/`: executable Phase 23 multi-seed same-tile B0/B1 training pilot runner.
- `experiments/phase24_ijaeog_evidence_package/`: executable Phase 24 IJAEOG evidence-package and claim-readiness runner.
- `experiments/phase25_padded_heldout_policy/`: executable Phase 25 padded variable-size held-out-tile B0/B1 policy pilot runner.
- `experiments/phase26_main_experiment/`: executable Phase 26 main empirical analysis runner.
- `experiments/phase27_stability_diagnosis/`: executable Phase 27 B0/B1 stability diagnosis runner.
- `experiments/phase28_representation_controls/`: executable Phase 28 representation-control runner for B0/B1/D2/D3/D4 padded held-out diagnostics.
- `experiments/phase28_compression_diagnosis/`: executable read-only Phase 28 compression diagnosis runner for existing 4096-step representation-control outputs.
- `experiments/phase29_representation_scale_diagnosis/`: executable read-only Phase 29 representation-scale diagnosis runner for B1/D4 feature tables.
- `experiments/phase30_normalized_b1_ablation/`: executable Phase 30 normalized-B1 held-out ablation runner with optional reuse of Phase 28 control summaries.
- `experiments/phase33_budget_robustness/`: executable Phase 33 bounded budget-robustness follow-up over existing Phase 30 normalized-B1 artifacts.
- `experiments/phase34_case_map_diagnostics/`: executable read-only Phase 34 case-map diagnostic runner over existing Phase 33 matched pilot artifacts.
- `src/paper11_geofm/`: focused utilities for sample loading, deterministic region aggregation, block feature assembly, suitability proxy scoring, base planning reward scoring, artifact export, proxy validation, reward-readiness gating, bounded tiled training pilots, cross-tile block-scorer pilots, multi-tile scorer evaluation pilots, multi-seed training pilots, IJAEOG evidence packaging, padded held-out policy pilots, Phase 26 empirical analysis, Phase 27 stability diagnosis, Phase 28 representation-control diagnostics, Phase 28 compression diagnostics, Phase 29 representation-scale diagnostics, Phase 30 normalized-B1 ablations, Phase 33 budget-robustness analysis, and Phase 34 case-map diagnostics.
- `src/legacy_runtime/`: copied legacy county/block RL runtime files imported by the experiment scripts.
- `data/bishan_alphaearth_sample/`: lightweight Bishan AlphaEarth embedding sample for smoke tests and reviewer inspection.
- `reproducibility/`: reproduction guide, data manifest, and file manifest.
- `scripts/smoke_check.py`: verifies repository layout and sample-data readability.
- `tests/`: pytest checks for layout and smoke-check behavior.

## Quick Start

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the lightweight checks:

```powershell
python scripts\smoke_check.py
python -m pytest tests
```

The smoke check reads the included Bishan sample arrays and verifies metadata. It does not run training, contact Google Earth Engine, or require GPU access.

Run the Phase 1 Bishan GeoFM representation baseline:

```powershell
python experiments\phase1_bishan_baseline\run_phase1.py
```

The command writes `region_features.csv` and `summary.json` under `experiments/phase1_bishan_baseline/outputs/`. These outputs are generated artifacts and are ignored by Git.

Run the Phase 2 block-level GeoFM feature assembly baseline:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py
```

The default Phase 2 path uses a generated grid-derived block-to-pixel mapping from the included Bishan sample. It writes `block_geofm_features.csv`, `summary.json`, and `experiment_variants.json` under `experiments/phase2_block_geofm_features/outputs/`. The variant manifest defines B0/B1/B2/B3 feature-table contracts for later DRL experiments; it is not trained-policy evidence. Because the generated-grid path does not include explicit planning features, the default manifest leaves variant `feature_table` values unset and does not create `variant_B*_features.csv` files. This is a feature-assembly smoke test, not real block-level DRL evidence.

To use real planning units, pass a block-to-pixel mapping CSV:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv path\to\block_pixel_mapping.csv
```

The mapping CSV must include `block_id`, `row`, and `col`, with optional `weight`. Optional block attributes can be joined by `block_id`:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv path\to\block_pixel_mapping.csv --attributes-csv path\to\block_attributes.csv
```

The attributes CSV can include `explicit_feature_00` through `explicit_feature_16`, weak labels such as `stable_farmland_label`, and split labels such as `split`. When a variant has all required columns, Phase 2 also writes a ready-only variant table: `variant_B0_features.csv`, `variant_B1_features.csv`, `variant_B2_features.csv`, or `variant_B3_features.csv`. B0 contains explicit planning features, B1 adds GeoFM embedding means, B2 adds the latent suitability proxy, and B3 combines explicit planning features, GeoFM embedding means, and the suitability proxy. When weak-label columns are present, the runner also writes `weak_label_validation.json` as a diagnostic proxy check comparing `suitability_proxy` against available weak labels.

This repository includes a tiny CSV fixture for checking that path directly:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir .pytest_tmp\phase2_variant_csv_exports
```

The included fixture has the required explicit feature columns, so it creates `variant_B0_features.csv` through `variant_B3_features.csv` and records those filenames in `experiment_variants.json`.

After a Phase 2 run has produced ready variant CSVs, inspect the DRL input matrix without training a policy:

```powershell
python experiments\phase3_drl_input_adapter\inspect_variant_inputs.py --phase2-output-dir .pytest_tmp\phase2_variant_csv_exports --variant B3
```

This command validates the `experiment_variants.json` contract, loads the requested variant CSV into a numeric matrix, and reports shape and reward metadata only. It does not train or evaluate a DRL policy.

After Phase 2 has produced ready variant CSVs, run the Phase 4 one-step Gymnasium contract smoke environment:

```powershell
python experiments\phase4_drl_smoke_env\run_phase4_smoke.py --phase2-output-dir .pytest_tmp\phase2_variant_csv_exports --variant B3
```

This command wraps the Phase 3 input matrix as a Gymnasium-compatible observation/action/mask contract, takes the first valid action once, and prints reward wiring metadata. It does not train a policy, evaluate a policy, or simulate planning outcomes.

Run the Phase 5 deterministic masked-rollout protocol smoke check:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase5_rollout_protocol\outputs\phase2_fixture
python experiments\phase5_rollout_protocol\run_phase5_rollout.py --phase2-output-dir experiments\phase5_rollout_protocol\outputs\phase2_fixture --output-dir experiments\phase5_rollout_protocol\outputs\phase5_protocol --variants B0,B1,B2,B3
```

This command runs the same deterministic masked rollout protocol across ready variants and writes `phase5_rollout_summary.csv` and `phase5_rollout_steps.json`. It does not train or evaluate a policy and does not report planning performance.

Run the Phase 6 non-learning masked baseline evaluator:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase6_masked_baselines\outputs\phase2_fixture
python experiments\phase6_masked_baselines\run_phase6_baselines.py --phase2-output-dir experiments\phase6_masked_baselines\outputs\phase2_fixture --output-dir experiments\phase6_masked_baselines\outputs\phase6_baselines --variants B0,B1,B2,B3 --policies first_valid,seeded_random --seed 0
```

This command runs `first_valid` and `seeded_random` masked action selectors across ready variants and writes `phase6_baseline_summary.csv` and `phase6_baseline_traces.json`. It does not train or evaluate a DRL policy and does not report planning performance.

Run the Phase 7 MaskablePPO compatibility smoke check:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase7_maskableppo_smoke\outputs\phase2_fixture
python experiments\phase7_maskableppo_smoke\run_phase7_maskableppo_smoke.py --phase2-output-dir experiments\phase7_maskableppo_smoke\outputs\phase2_fixture --output-dir experiments\phase7_maskableppo_smoke\outputs\phase7_smoke --variant B3 --total-timesteps 8 --seed 0
```

This command checks that `sb3-contrib` MaskablePPO can consume the Phase 4 action-mask environment, run a tiny CPU-only `learn()` call, make a masked prediction, and write `phase7_maskableppo_smoke.json`. It does not train, tune, evaluate, or report a useful DRL policy.

Run the Phase 8 ablation-control feature-table generator:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase8_ablation_controls\outputs\phase2_fixture
python experiments\phase8_ablation_controls\run_phase8_ablation_controls.py --phase2-output-dir experiments\phase8_ablation_controls\outputs\phase2_fixture --output-dir experiments\phase8_ablation_controls\outputs\phase8_controls --seed 0 --pca-dimensions 8,16
```

This command derives D2/D3/D4P8/D4P16 diagnostic control feature tables, writes a Phase 3-compatible `experiment_variants.json`, and writes `phase8_ablation_control_summary.json`. It does not train, tune, evaluate, compare, or report a useful DRL policy.

Run the Phase 9 weak-label proxy-validation report:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase9_proxy_validation\outputs\phase2_fixture
python experiments\phase9_proxy_validation\run_phase9_proxy_validation.py --phase2-output-dir experiments\phase9_proxy_validation\outputs\phase2_fixture --output-dir experiments\phase9_proxy_validation\outputs\phase9_report --label-columns stable_farmland_label,high_standard_farmland_label
```

This command writes `phase9_proxy_validation_report.json` with suitability distribution and weak-label alignment diagnostics. It does not prove agronomic validity, train a policy, evaluate a policy, or report planning performance.

Run the Phase 10 suitability reward-readiness gate:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase10_reward_readiness\outputs\phase2_fixture
python experiments\phase9_proxy_validation\run_phase9_proxy_validation.py --phase2-output-dir experiments\phase10_reward_readiness\outputs\phase2_fixture --output-dir experiments\phase10_reward_readiness\outputs\phase9_report --label-columns stable_farmland_label,high_standard_farmland_label
python experiments\phase10_reward_readiness\run_phase10_reward_readiness.py --phase9-report experiments\phase10_reward_readiness\outputs\phase9_report\phase9_proxy_validation_report.json --output-dir experiments\phase10_reward_readiness\outputs\phase10_gate --required-labels stable_farmland_label,high_standard_farmland_label
```

This command writes `phase10_reward_readiness_gate.json`. For the included tiny fixture, the expected status is `not_ready_for_suitability_reward` because Phase 9 reports `negative_or_no_alignment` for both weak labels. It does not train, tune, evaluate, or report a DRL policy.

Run the Phase 11 real Bishan DLTB adapter if the local DLTB GeoPackage is available:

```powershell
python experiments\phase11_bishan_dltb_real\run_phase11_bishan_dltb_adapter.py --dltb-path D:\test\dem_slope_analysis\output\DLTB_with_slope.gpkg --output-dir experiments\phase11_bishan_dltb_real\outputs\adapter
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv experiments\phase11_bishan_dltb_real\outputs\adapter\block_pixel_mapping.csv --attributes-csv experiments\phase11_bishan_dltb_real\outputs\adapter\block_attributes.csv --output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real
python experiments\phase9_proxy_validation\run_phase9_proxy_validation.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase11_bishan_dltb_real\outputs\phase9_real --label-columns current_farmland_label,low_slope_farmland_label,farmland_or_orchard_label
python experiments\phase10_reward_readiness\run_phase10_reward_readiness.py --phase9-report experiments\phase11_bishan_dltb_real\outputs\phase9_real\phase9_proxy_validation_report.json --output-dir experiments\phase11_bishan_dltb_real\outputs\phase10_real --required-labels current_farmland_label,low_slope_farmland_label,farmland_or_orchard_label
```

In the local Bishan run, the adapter exported 64,984 DLTB polygons into Phase 2-compatible inputs. Phase 9 found weak positive alignment for `current_farmland_label` and `farmland_or_orchard_label`, but not for `low_slope_farmland_label`; Phase 10 therefore reported `not_ready_for_suitability_reward`. This is real-data feature-table evidence, not DRL policy-performance evidence.

Run the Phase 12 real DLTB scale audit after the Phase 11 real-data chain:

```powershell
python experiments\phase12_real_scale_audit\run_phase12_real_scale_audit.py --phase11-summary experiments\phase11_bishan_dltb_real\outputs\adapter\phase11_bishan_dltb_adapter_summary.json --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase9-report experiments\phase11_bishan_dltb_real\outputs\phase9_real\phase9_proxy_validation_report.json --phase10-gate experiments\phase11_bishan_dltb_real\outputs\phase10_real\phase10_reward_readiness_gate.json --output-dir experiments\phase12_real_scale_audit\outputs\real_bishan
```

For the current real Bishan artifacts, Phase 12 reports that real feature tables are ready and representation-only smoke checks are allowed, but suitability reward is not allowed and full-scale flat DRL training is not ready. The maximum flat B3 observation dimension is 5,328,691, so the next training-oriented design should be tiled or hierarchical.

Run the Phase 13 tiled real-data contract builder after Phase 11/12 artifacts exist:

```powershell
python experiments\phase13_tiled_real_contract\run_phase13_tiled_real_contract.py --mapping-csv experiments\phase11_bishan_dltb_real\outputs\adapter\block_pixel_mapping.csv --variant-manifest experiments\phase11_bishan_dltb_real\outputs\phase2_real\experiment_variants.json --output-dir experiments\phase13_tiled_real_contract\outputs\real_bishan --tile-rows 8 --tile-cols 8
```

For the current real Bishan mapping, Phase 13 produces 54 non-empty tiles. The largest tile has 2,234 blocks, and the B3 maximum tiled observation dimension is 183,191, which is below the 1,000,000 threshold. This establishes a tractable tiled contract for later environment design, not policy performance.

Run the Phase 14 tile-level smoke environment on the largest real Bishan tile:

```powershell
python experiments\phase14_tiled_smoke_env\run_phase14_tiled_smoke.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --tile-id tile_r003_c003 --variant B1 --output-dir experiments\phase14_tiled_smoke_env\outputs\real_bishan_largest_tile
```

For the current real Bishan artifacts, Phase 14 loads 2,234 blocks from `tile_r003_c003` into the B1 representation-only contract. The observation shape is 180,957, the action space is `Discrete(2234)`, and the first selected block receives a deterministic base planning reward of `-0.197259`. This is a tiled input-contract smoke check, not training or planning-performance evidence.

Run the Phase 15 all-tile batch smoke check:

```powershell
python experiments\phase15_tiled_batch_smoke\run_phase15_tiled_batch_smoke.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variant B1 --output-dir experiments\phase15_tiled_batch_smoke\outputs\real_bishan_all_tiles
```

For the current real Bishan artifacts, Phase 15 processes all 54 tiles, covers 64,984 blocks, and reports `All passed: True` with maximum B1 observation shape 180,957. It is still a representation-only input-contract smoke check, not policy training or evaluation.

Run the Phase 16 tiled non-learning baseline protocol:

```powershell
python experiments\phase16_tiled_baseline_protocol\run_phase16_tiled_baselines.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variant B1 --policies first_valid,seeded_random --max-steps 4 --seed 0 --output-dir experiments\phase16_tiled_baseline_protocol\outputs\real_bishan_b1
```

For the current real Bishan artifacts, Phase 16 processes all 54 tiles with two deterministic non-learning policies, writes 108 summary rows, covers 64,984 blocks, and accumulates the deterministic B1 base planning reward. It is a tiled masked-rollout protocol check only; it does not train, evaluate, or compare a DRL policy and does not enable suitability reward.

Run the Phase 17 tiled MaskablePPO readiness smoke check:

```powershell
python experiments\phase17_tiled_maskableppo_readiness\run_phase17_tiled_maskableppo_readiness.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variant B1 --tile-selection largest --total-timesteps 8 --seed 0 --output-dir experiments\phase17_tiled_maskableppo_readiness\outputs\real_bishan_largest_tile
```

For the current real Bishan artifacts, Phase 17 selects the largest tile, `tile_r003_c003`, with 2,234 blocks and B1 observation shape 180,957. It verifies MaskablePPO action-mask compatibility, a tiny CPU-only `learn()` call, and a valid masked prediction. It does not train, tune, evaluate, or compare a useful DRL policy and does not enable suitability reward.

Run the Phase 18 planning-reward readiness gate:

```powershell
python experiments\phase18_planning_reward_readiness\run_phase18_planning_reward_readiness.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase10-gate experiments\phase11_bishan_dltb_real\outputs\phase10_real\phase10_reward_readiness_gate.json --phase12-audit experiments\phase12_real_scale_audit\outputs\real_bishan\phase12_real_dltb_scale_audit.json --phase17-readiness experiments\phase17_tiled_maskableppo_readiness\outputs\real_bishan_largest_tile\phase17_tiled_maskableppo_readiness.json --output-dir experiments\phase18_planning_reward_readiness\outputs\real_bishan
```

For the current real Bishan artifacts after Phase 19, Phase 18 reports that `base_planning_reward` is implemented and the tiled MaskablePPO API path is ready, but true planning-performance experiments are still not ready. Phase 10/12 keep suitability reward disabled and flat full-scale training not ready, so the current artifacts remain readiness evidence rather than policy-performance evidence.

Run the Phase 20 bounded same-tile B0/B1 training pilot:

```powershell
python experiments\phase20_bounded_tiled_training\run_phase20_bounded_tiled_training.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --total-timesteps 8 --eval-max-steps 4 --seed 0 --output-dir experiments\phase20_bounded_tiled_training\outputs\real_bishan_pilot
```

For the current real Bishan artifacts, Phase 20 selects `tile_r003_c003` as both the train tile and the same-tile learned-policy evaluation tile, writes six B0/B1 trained-policy and baseline summary rows, and records `blocked_variable_observation_shape` for cross-tile learned-policy evaluation. It verifies that a bounded B0/B1 MaskablePPO training/evaluation protocol can execute under the deterministic base planning reward, but it does not enable suitability reward, support cross-tile transfer, or provide final planning-performance evidence.

Run the Phase 21 cross-tile per-block scorer pilot:

```powershell
python experiments\phase21_cross_tile_block_scorer\run_phase21_cross_tile_block_scorer.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --ridge-alpha 1e-6 --eval-max-steps 4 --seed 0 --output-dir experiments\phase21_cross_tile_block_scorer\outputs\real_bishan_pilot
```

For the current real Bishan artifacts, Phase 21 trains a standardized ridge-linear per-block scorer on `tile_r003_c003` and evaluates the learned scorer on the distinct tile `tile_r002_c003`. It writes six B0/B1 learned-scorer and baseline summary rows and reports `executed_distinct_tile`. This verifies a variable-block-count cross-tile policy interface, but it does not enable suitability reward, prove cross-region transfer, or provide final planning-performance evidence.

Run the Phase 22 multi-tile, multi-seed per-block scorer evaluation pilot:

```powershell
python experiments\phase22_multi_tile_scorer_eval\run_phase22_multi_tile_scorer_eval.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --ridge-alpha 1e-6 --eval-max-steps 4 --seeds 0,1 --max-eval-tiles 2 --output-dir experiments\phase22_multi_tile_scorer_eval\outputs\real_bishan_pilot
```

For the current real Bishan artifacts, Phase 22 trains the same standardized ridge-linear per-block scorer once per B0/B1 variant on `tile_r003_c003` and evaluates learned-scorer, first-valid, and seeded-random policies across `tile_r002_c003` and `tile_r005_c004` with seeds `0` and `1`. It writes 24 summary rows. This broadens the Phase 21 interface pilot, but it does not enable suitability reward, run PPO training, prove transfer, or provide final planning-performance evidence.

Run the Phase 23 multi-seed same-tile B0/B1 training pilot:

```powershell
python experiments\phase23_multi_seed_training\run_phase23_multi_seed_training.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --total-timesteps 8 --eval-max-steps 4 --seeds 0,1,2 --output-dir experiments\phase23_multi_seed_training\outputs\real_bishan_pilot
```

For the current real Bishan artifacts, Phase 23 repeats the bounded same-tile B0/B1 MaskablePPO pilot on `tile_r003_c003` across seeds `0`, `1`, and `2`. It writes 18 summary rows and an aggregate comparison report; the observed B1-B0 learned-policy mean reward delta is `0.4273019432` under the short pilot budget. This strengthens multi-seed learned-policy execution evidence, but it does not solve cross-tile learned-policy evaluation, enable suitability reward, prove transfer, or provide final planning-performance evidence.

Build the Phase 24 IJAEOG evidence package from Phase 22 and Phase 23 pilot outputs:

```powershell
python experiments\phase24_ijaeog_evidence_package\run_phase24_ijaeog_evidence_package.py --phase22-summary-csv experiments\phase22_multi_tile_scorer_eval\outputs\real_bishan_pilot\phase22_multi_tile_scorer_eval_summary.csv --phase23-summary-csv experiments\phase23_multi_seed_training\outputs\real_bishan_pilot\phase23_multi_seed_training_summary.csv --phase23-comparison-json experiments\phase23_multi_seed_training\outputs\real_bishan_pilot\phase23_multi_seed_training_comparison.json --output-dir experiments\phase24_ijaeog_evidence_package\outputs\real_bishan
```

For the current real Bishan artifacts, Phase 24 reports 24 Phase 22 rows, 18 Phase 23 rows, a B1-B0 learned-policy mean reward delta of `0.4273019432`, and `submission_ready: not_ready`. It writes `phase24_ijaeog_evidence_table.csv`, `phase24_ijaeog_evidence_summary.json`, and `phase24_ijaeog_claim_readiness.md`. This is a synthesis and claim-readiness package only; it does not create new policy-performance, transfer, or suitability-reward evidence.

Run the Phase 25 padded held-out-tile B0/B1 policy pilot smoke:

```powershell
python experiments\phase25_padded_heldout_policy\run_phase25_padded_heldout_policy.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --total-timesteps 32 --eval-max-steps 4 --seeds 0 --max-eval-tiles 1 --output-dir experiments\phase25_padded_heldout_policy\outputs\real_bishan_smoke
```

For the current real Bishan artifacts, Phase 25 trains on `tile_r003_c003` and evaluates on the distinct held-out tile `tile_r002_c003` with padded maximum blocks `2234`, variants `B0,B1`, seed `0`, total timesteps `32`, and evaluation max steps `4`. It writes six summary rows, reports `all_evaluations_completed: True`, records a B1-B0 held-out learned-policy mean reward delta of `1.3314600457`, and sets `pilot_result_status: B1_improves_B0`. Expected artifacts are `phase25_padded_heldout_policy_summary.csv`, `phase25_padded_heldout_policy_traces.json`, and `phase25_padded_heldout_policy_comparison.json`; the comparison JSON keeps `suitability_reward_validation_before_B2_B3` in the remaining evidence gaps.

Phase 25 adds a padded variable-size held-out-tile MaskablePPO pilot for B0/B1 under the deterministic `base_planning_reward`. It removes the Phase 20/23 flat observation/action shape blocker for held-out Bishan tile evaluation, but it remains a bounded pilot: no suitability reward, no B2/B3, no cross-region transfer, and no submission-level planning-performance claims.

Run the Phase 26 main empirical analysis package after Phase 25 outputs exist:

```powershell
python experiments\phase26_main_experiment\run_phase26_main_experiment.py --mode analyze-only --phase25-output-dir experiments\phase26_main_experiment\outputs\colab_main\phase25_run --output-dir experiments\phase26_main_experiment\outputs\colab_main\phase26_analysis
```

Phase 26 analyzes Phase 25 padded held-out B0/B1 outputs into manuscript-facing empirical tables. It reports B1-B0 learned-policy deltas by held-out tile and seed, assigns a conservative claim status, and keeps suitability reward, B2/B3, and cross-region transfer out of scope.

Run the Phase 27 B0/B1 stability diagnosis after the current Phase 26 macOS
artifacts exist:

```powershell
python experiments\phase27_stability_diagnosis\run_phase27_stability_diagnosis.py --phase26-comparison-json experiments\phase26_main_experiment\outputs\macos_main\phase26_analysis\phase26_main_comparison.json --phase26-comparison-json experiments\phase26_main_experiment\outputs\macos_main_4096\phase26_analysis\phase26_main_comparison.json --output-dir experiments\phase27_stability_diagnosis\outputs\macos_1024_vs_4096
```

Phase 27 is read-only. It compares the 1024-step and 4096-step Phase 26
B1-B0 learned-policy deltas and reports `budget_not_explanatory`: the mean
delta improves by `0.3010310174`, but the higher-budget result is still
negative and the positive tile-seed count falls from `4 / 9` to `3 / 9`.

Run the Phase 28 representation-control package after Phase 2 B0/B1 outputs,
Phase 8 D-control outputs, and the Phase 13 tile index are available:

```powershell
python experiments\phase28_representation_controls\run_phase28_representation_controls.py --mode run-and-analyze --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1,D2,D3,D4P8,D4P16 --total-timesteps 1024 --eval-max-steps 8 --seeds 0,1,2 --max-eval-tiles 3 --output-dir experiments\phase28_representation_controls\outputs\real_bishan_main
```

For existing Phase 28 summary rows, use analyze-only mode:

```powershell
python experiments\phase28_representation_controls\run_phase28_representation_controls.py --mode analyze-only --existing-summary-csv experiments\phase28_representation_controls\outputs\real_bishan_main\phase28_representation_control_summary.csv --output-dir experiments\phase28_representation_controls\outputs\real_bishan_analysis
```

Phase 28 writes `phase28_representation_control_summary.csv`,
`phase28_representation_control_traces.json`,
`phase28_representation_control_comparison.json`,
`phase28_tile_seed_delta_table.csv`, and `phase28_control_readiness.md`.
The current 1024-step and 4096-step real Bishan runs both report
`compression_matches_raw`. The 4096-step run records B1-B0 mean delta
`-0.1318712688` and compressed controls D4P8/D4P16 above B1, so Phase 28
does not support a positive raw-B1 representation claim. It is diagnostic only:
it does not enable suitability reward, test B2/B3, test cross-region transfer,
or support final submission-level planning-performance claims.

For the read-only 4096-step compression follow-up, run:

```powershell
python experiments\phase28_compression_diagnosis\run_phase28_compression_diagnosis.py --summary-csv experiments\phase28_representation_controls\outputs\real_bishan_4096\phase28_representation_control_summary.csv --phase2-b1-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\variant_B1_features.csv --d4p8-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P8_features.csv --d4p16-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P16_features.csv --output-dir experiments\phase28_compression_diagnosis\outputs\real_bishan_4096
```

This command writes `phase28_compression_overlap.csv`,
`phase28_compression_reward_components.csv`,
`phase28_compression_diagnosis.json`, and
`phase28_compression_diagnosis.md`. For the current 4096-step artifacts, it
reports `compressed_controls_select_distinct_higher_reward_blocks`: D4P8 and
D4P16 exceed raw B1 while selecting almost disjoint block sets with better
explicit base-reward components. This remains a read-only association, not a
causal claim that PCA is intrinsically superior.

Run the read-only Phase 29 representation-scale follow-up after the Phase 2
B1 feature table, Phase 8 D4 feature tables, Phase 13 tile index, and optional
Phase 28 summary CSV are available:

```powershell
python experiments\phase29_representation_scale_diagnosis\run_phase29_representation_scale_diagnosis.py --phase2-b1-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\variant_B1_features.csv --d4p8-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P8_features.csv --d4p16-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P16_features.csv --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --phase28-summary-csv experiments\phase28_representation_controls\outputs\real_bishan_4096\phase28_representation_control_summary.csv --output-dir experiments\phase29_representation_scale_diagnosis\outputs\real_bishan_4096
```

This command writes `phase29_variant_scale_summary.csv`,
`phase29_tile_scale_summary.csv`,
`phase29_b1_normalization_profiles.csv`,
`phase29_representation_scale_diagnosis.json`, and
`phase29_representation_scale_diagnosis.md`. For the current real Bishan
artifacts, it reports `raw_b1_scale_may_affect_optimization`: raw B1 has mean
column standard deviation `0.0377917339`, D4P8/D4P16 have larger component
standard deviations, and the raw embedding effective rank is `5.2467650861`.
This is a diagnostic optimization hypothesis, not proof that PCA or
normalization improves PPO performance.

Run the bounded Phase 30 normalized-B1 follow-up after the Phase 28 4096-step
control summary exists. The recommended incremental path reuses the frozen
`B0,B1,D2,D3,D4P8,D4P16` Phase 28 rows and trains only `N1Z` and `N1ZR`:

```powershell
python experiments\phase30_normalized_b1_ablation\run_phase30_normalized_b1_ablation.py --mode run-and-analyze --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --existing-control-summary-csv experiments\phase28_representation_controls\outputs\real_bishan_4096\phase28_representation_control_summary.csv --variants B0,B1,N1Z,N1ZR,D2,D3,D4P8,D4P16 --total-timesteps 4096 --eval-max-steps 8 --seeds 0,1,2 --max-eval-tiles 3 --output-dir experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental
```

This command writes `phase30_normalized_b1_summary.csv`,
`phase30_normalized_b1_traces.json`,
`phase30_normalized_b1_comparison.json`,
`phase30_normalized_b1_delta_table.csv`,
`phase30_normalized_b1_readiness.md`, and a derived
`derived_normalized_controls/` directory containing `variant_N1Z_features.csv`
and `variant_N1ZR_features.csv`. For the current real Bishan artifacts, it
reports `normalized_b1_recovers_b0_gap`: `N1Z` mean learned-policy reward is
`0.6515323140`, `N1ZR` is `0.5772465716`, both exceed raw `B1`
(`0.3506359482`) and `B0` (`0.4825072170`) on mean reward, but both remain
below `D4P8` (`0.7274877829`) and `D4P16` (`0.9918299718`). This is partial
support for the Phase 29 optimization hypothesis, not a manuscript-ready
GeoFM planning-performance claim.

Run the bounded Phase 33 budget-robustness follow-up after the Phase 30
incremental output exists. The current local execution history used nine
separate matched `5120`-step pilots across the three held-out Phase 30 tiles
and seeds `0,1,2`, then aggregated those pilots into a single bounded
Phase 33 verdict:

```powershell
python experiments\phase33_budget_robustness\run_phase33_budget_robustness.py --mode analyze-only --phase30-comparison-json experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_3tiles_3seeds_9run_aggregate\phase33_matched_baseline_comparison.json --phase30-comparison-json experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_3tiles_3seeds_9run_aggregate\phase30_aggregate.json --output-dir experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_3tiles_3seeds_9run_aggregate
```

The current bounded Phase 33 aggregate reports `budget_not_explanatory`.
Across the completed `3 tiles x 3 seeds` coverage, all six tracked focal gaps
are negative on mean delta at `5120` steps. The four compressed-control gaps
are:

- `N1Z - D4P16`: `-0.7597285327`
- `N1Z - D4P8`: `-0.4953863438`
- `N1ZR - D4P16`: `-0.6658915329`
- `N1ZR - D4P8`: `-0.4015493440`

The earlier positive `tile_r002_c003` seed-0 pilot remains in
`experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r002_c003_seed0_matched`,
but it does not survive broader bounded coverage. The tilewise three-seed
aggregates now split as:

- `tile_r002_c003`: `budget_closes_compressed_gap`
- `tile_r005_c003`: `budget_not_explanatory`
- `tile_r005_c004`: `budget_not_explanatory`

This keeps Phase 33 in the diagnostic bucket: it narrows the representation
branch, but it does not justify a stable budget-based rescue claim for
normalized B1.

Run the read-only Phase 34 case-map diagnostic after the completed Phase 33
matched pilot directories exist:

```powershell
python experiments\phase34_case_map_diagnostics\run_phase34_case_map_diagnostics.py --phase33-output-dirs experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r002_c003_seed0_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r002_c003_seed1_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r002_c003_seed2_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c003_seed0_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c003_seed1_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c003_seed2_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c004_seed0_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c004_seed1_matched experiments\phase33_budget_robustness\outputs\real_bishan_5120_pilot_tile_r005_c004_seed2_matched --phase2-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\block_geofm_features.csv --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --output-dir experiments\phase34_case_map_diagnostics\outputs\real_bishan_5120_phase33_9run --variants N1Z,N1ZR --comparators B1,D4P8,D4P16
```

The current Phase 34 output reports `case_map_diagnostics_ready` with `54`
case rows and `857` selected-block rows. Positive Phase 33 cases select higher
mean base-reward block sets, while negative cases select lower mean base-reward
block sets. `tile_r005_c003` is the clearest negative spatial counterexample.
This remains a read-only diagnostic and does not change the Phase 33 aggregate
status of `budget_not_explanatory`.

## Key Entry Points

- Design synthesis: `paper/design/01_design_synthesis.md`
- System design: `paper/design/02_system_design.md`
- Experiment plan: `paper/design/03_experiment_plan.md`
- Phase 1 result interpretation: `paper/phase1_results/01_phase1_result_interpretation.md`
- Next experiment matrix: `paper/phase1_results/02_next_experiment_matrix.md`
- Phase 26 result interpretation: `paper/phase26_results/01_phase26_result_interpretation.md`
- Phase 26 next experiment matrix: `paper/phase26_results/02_next_experiment_matrix.md`
- Phase 26 budget comparison: `paper/phase26_results/03_phase26_budget_comparison.md`
- Phase 27 stability diagnosis: `paper/phase27_results/01_phase27_stability_diagnosis.md`
- Phase 28 representation-control diagnosis: `paper/phase28_results/01_phase28_representation_control_diagnosis.md`
- Phase 28 compression diagnosis: `paper/phase28_results/02_phase28_compression_diagnosis.md`
- Phase 29 representation-scale diagnosis: `paper/phase28_results/03_phase29_representation_scale_diagnosis.md`
- Phase 30 normalized-B1 ablation: `paper/phase28_results/04_phase30_normalized_b1_ablation.md`
- Phase 33 budget robustness: `paper/phase28_results/07_phase33_budget_robustness.md`
- Phase 34 case-map diagnostics: `paper/phase28_results/08_phase34_case_map_diagnostics.md`
- Main embedding environment: `experiments/geofm_runtime/embedding_space_env.py`
- Phase 1 Bishan baseline runner: `experiments/phase1_bishan_baseline/run_phase1.py`
- Phase 2 block feature assembly runner: `experiments/phase2_block_geofm_features/run_phase2.py`
- Phase 3 DRL input inspection runner: `experiments/phase3_drl_input_adapter/inspect_variant_inputs.py`
- Phase 4 DRL smoke environment runner: `experiments/phase4_drl_smoke_env/run_phase4_smoke.py`
- Phase 5 rollout protocol smoke runner: `experiments/phase5_rollout_protocol/run_phase5_rollout.py`
- Phase 6 masked baseline evaluator runner: `experiments/phase6_masked_baselines/run_phase6_baselines.py`
- Phase 7 MaskablePPO compatibility smoke runner: `experiments/phase7_maskableppo_smoke/run_phase7_maskableppo_smoke.py`
- Phase 8 ablation-control feature-table runner: `experiments/phase8_ablation_controls/run_phase8_ablation_controls.py`
- Phase 9 proxy-validation report runner: `experiments/phase9_proxy_validation/run_phase9_proxy_validation.py`
- Phase 10 reward-readiness gate runner: `experiments/phase10_reward_readiness/run_phase10_reward_readiness.py`
- Phase 11 Bishan DLTB real-data adapter runner: `experiments/phase11_bishan_dltb_real/run_phase11_bishan_dltb_adapter.py`
- Phase 12 real DLTB scale audit runner: `experiments/phase12_real_scale_audit/run_phase12_real_scale_audit.py`
- Phase 13 tiled real-data contract runner: `experiments/phase13_tiled_real_contract/run_phase13_tiled_real_contract.py`
- Phase 14 tiled smoke environment runner: `experiments/phase14_tiled_smoke_env/run_phase14_tiled_smoke.py`
- Phase 15 tiled batch smoke runner: `experiments/phase15_tiled_batch_smoke/run_phase15_tiled_batch_smoke.py`
- Phase 16 tiled baseline protocol runner: `experiments/phase16_tiled_baseline_protocol/run_phase16_tiled_baselines.py`
- Phase 17 tiled MaskablePPO readiness runner: `experiments/phase17_tiled_maskableppo_readiness/run_phase17_tiled_maskableppo_readiness.py`
- Phase 18 planning-reward readiness gate runner: `experiments/phase18_planning_reward_readiness/run_phase18_planning_reward_readiness.py`
- Phase 19 base planning reward module: `src/paper11_geofm/planning_reward.py`
- Phase 20 bounded same-tile B0/B1 training runner: `experiments/phase20_bounded_tiled_training/run_phase20_bounded_tiled_training.py`
- Phase 20 bounded training module: `src/paper11_geofm/bounded_tiled_training.py`
- Phase 21 cross-tile per-block scorer runner: `experiments/phase21_cross_tile_block_scorer/run_phase21_cross_tile_block_scorer.py`
- Phase 21 cross-tile scorer module: `src/paper11_geofm/cross_tile_block_scorer.py`
- Phase 22 multi-tile scorer evaluation runner: `experiments/phase22_multi_tile_scorer_eval/run_phase22_multi_tile_scorer_eval.py`
- Phase 22 multi-tile scorer evaluation module: `src/paper11_geofm/multi_tile_scorer_eval.py`
- Phase 23 multi-seed training runner: `experiments/phase23_multi_seed_training/run_phase23_multi_seed_training.py`
- Phase 23 multi-seed training module: `src/paper11_geofm/multi_seed_training.py`
- Phase 24 IJAEOG evidence-package runner: `experiments/phase24_ijaeog_evidence_package/run_phase24_ijaeog_evidence_package.py`
- Phase 24 IJAEOG evidence-package module: `src/paper11_geofm/ijaeog_evidence_package.py`
- Phase 25 padded held-out policy runner: `experiments/phase25_padded_heldout_policy/run_phase25_padded_heldout_policy.py`
- Phase 25 padded held-out policy module: `src/paper11_geofm/padded_heldout_policy.py`
- Phase 26 main empirical analysis runner: `experiments/phase26_main_experiment/run_phase26_main_experiment.py`
- Phase 26 main empirical analysis module: `src/paper11_geofm/phase26_main_experiment.py`
- Phase 27 stability diagnosis runner: `experiments/phase27_stability_diagnosis/run_phase27_stability_diagnosis.py`
- Phase 27 stability diagnosis module: `src/paper11_geofm/phase27_stability_diagnosis.py`
- Phase 28 representation-control runner: `experiments/phase28_representation_controls/run_phase28_representation_controls.py`
- Phase 28 representation-control module: `src/paper11_geofm/phase28_representation_controls.py`
- Phase 28 compression diagnosis runner: `experiments/phase28_compression_diagnosis/run_phase28_compression_diagnosis.py`
- Phase 28 compression diagnosis module: `src/paper11_geofm/phase28_compression_diagnosis.py`
- Phase 29 representation-scale diagnosis runner: `experiments/phase29_representation_scale_diagnosis/run_phase29_representation_scale_diagnosis.py`
- Phase 29 representation-scale diagnosis module: `src/paper11_geofm/phase29_representation_scale_diagnosis.py`
- Phase 30 normalized-B1 ablation runner: `experiments/phase30_normalized_b1_ablation/run_phase30_normalized_b1_ablation.py`
- Phase 30 normalized-B1 ablation module: `src/paper11_geofm/phase30_normalized_b1_ablation.py`
- Phase 33 budget-robustness runner: `experiments/phase33_budget_robustness/run_phase33_budget_robustness.py`
- Phase 33 budget-robustness module: `src/paper11_geofm/phase33_budget_robustness.py`
- Phase 34 case-map diagnostics runner: `experiments/phase34_case_map_diagnostics/run_phase34_case_map_diagnostics.py`
- Phase 34 case-map diagnostics module: `src/paper11_geofm/phase34_case_map_diagnostics.py`
- Phase 1 utility package: `src/paper11_geofm/`
- Embedding RL training script: `experiments/geofm_runtime/train_embedding_rl.py`
- Dual-representation environment: `experiments/geofm_runtime/dual_rep_env.py`
- Legacy county environment: `src/legacy_runtime/county_env.py`

## Data Policy

This repository includes only the lightweight Bishan sample needed for reviewer smoke tests. Larger village and Heping embedding arrays, model checkpoints, and intervention transition files are documented in `reproducibility/DATA_MANIFEST.md` but are not included in ordinary Git.

The real Bishan DLTB-with-slope GeoPackage used by Phase 11 is an external local source documented in `reproducibility/DATA_MANIFEST.md`. It is not committed to Git; generated Phase 11 outputs under `experiments/phase11_bishan_dltb_real/outputs/` are also ignored.

Use Git LFS or an external archival repository such as Zenodo or OSF before distributing large derived arrays or trained weights.

## Recommended Journal Framing

The strongest Paper11 framing is for applied remote sensing and geospatial optimization venues: GeoFM representations improve current-state farmland suitability representation and DRL spatial planning, with careful claim boundaries around what remote-sensing embeddings can and cannot measure.
