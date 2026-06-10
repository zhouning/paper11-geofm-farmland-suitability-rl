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
- `src/paper11_geofm/`: focused utilities for sample loading, deterministic region aggregation, block feature assembly, suitability proxy scoring, artifact export, proxy validation, and reward-readiness gating.
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

## Key Entry Points

- Design synthesis: `paper/design/01_design_synthesis.md`
- System design: `paper/design/02_system_design.md`
- Experiment plan: `paper/design/03_experiment_plan.md`
- Phase 1 result interpretation: `paper/phase1_results/01_phase1_result_interpretation.md`
- Next experiment matrix: `paper/phase1_results/02_next_experiment_matrix.md`
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
