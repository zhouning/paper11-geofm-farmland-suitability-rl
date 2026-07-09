# Phase 67 Candidate Reward/Label Target Audit

Status: independent_label_required_before_reward_redesign

Candidate count: 0
Best candidate target IDs: []

Claim boundary:
Phase 67 is a read-only candidate reward/label target audit. It inventories diagnostic targets and checks leakage, gate status, and explicit-versus-GeoFM information gain. It does not train a policy, modify rewards, enable suitability reward, create B2/B3 variants, prove GeoFM advantage, or justify formal submission-level claims.

## Key Evidence

- Audited block count: 64,984.
- Phase 67 status: `independent_label_required_before_reward_redesign`.
- Candidate diagnostic target count passing the gate: 0.
- Strongest GeoFM-minus-explicit proxy R2: 0.0125211988 for `residual_geofm_norm_embedding_mean_after_explicit` on `D4P16`, below the 0.05 diagnostic gate threshold.
- `geofm_norm_embedding_mean` on `D4P16` reached GeoFM proxy R2 0.0125499002 with all-explicit proxy R2 0.0007164117, but it is self-referential and diagnostic-only.
- Weak label residuals for `current_farmland_label` and `farmland_or_orchard_label` had zero residual variance after explicit-feature fitting.
- No candidate target can currently support reward redesign without independent non-leakage labels.

## Reproduction

Run Phase 67 from the repository root after Phase 66 artifacts exist:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase67_candidate_reward_label_target_audit\run_phase67_candidate_reward_label_target_audit.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --phase61-output-dir experiments\phase61_d6_geofm_projection_controls\outputs\phase52_full5_seed3 --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --phase10-json experiments\phase11_bishan_dltb_real\outputs\phase10_real\phase10_reward_readiness_gate.json --phase18-json experiments\phase18_planning_reward_readiness\outputs\real_bishan\phase18_planning_reward_readiness.json --phase66-json experiments\phase66_reward_label_representation_audit\outputs\phase52_full5_seed3\phase66_reward_label_representation_audit.json --variants B0,D4P8,D4P16,D6R8,D6R16 --label-columns current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label --top-k-values 8,16,32 --output-dir experiments\phase67_candidate_reward_label_target_audit\outputs\phase52_full5_seed3
```

## Boundary

No formal manuscript files were changed in this phase.
