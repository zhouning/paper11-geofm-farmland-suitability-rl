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

## 12. Inspect the Paper11 Design

Read these files in order:

```text
paper/design/01_design_synthesis.md
paper/design/02_system_design.md
paper/design/03_experiment_plan.md
paper/design/04_manuscript_outline.md
paper/design/05_risks_and_boundaries.md
```

The design intentionally keeps Paper11 within current-state suitability representation and DRL layout optimization.

## 13. Inspect Runtime Code

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

The Phase 1, Phase 2, Phase 3, Phase 4, Phase 5, Phase 6, Phase 7, Phase 8, and Phase 9 reviewer paths are deterministic and do not require internet, GPU, Earth Engine, or full DRL training.

## 14. Regenerate Embeddings

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

## 15. Large Data and Weights

Large arrays, model weights, and intervention transition files are not included in ordinary Git. See `DATA_MANIFEST.md` for what was deliberately included and excluded.

For a full artifact release, place large files in Git LFS or an external archive and record checksums in the data manifest.
