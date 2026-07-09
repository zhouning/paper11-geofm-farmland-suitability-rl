# Phase 70 Standardized Set-Policy Rerun

Status: standardization_not_sufficient

## Key Evidence

- Phase 70 reran the Phase 63 set-policy route with train-tile-fitted feature standardization for model inputs.
- The rerun preserved original unstandardized feature matrices for base-reward scoring and oracle ranking.
- Phase 70 minus Phase 63 reward summary: {"max_delta":1.0127905299,"mean_delta":-0.0076333409,"min_delta":-1.4410704099,"positive_count":48,"total_count":75}.
- D4 versus B0 standardized delta summary: {"max_delta":0.5918017425,"mean_delta":-0.381099,"min_delta":-1.4393657461,"positive_count":5,"total_count":30}.
- D4 versus D6 standardized delta summary: {"max_delta":1.3190410962,"mean_delta":-0.0841856126,"min_delta":-1.3783311504,"positive_count":15,"total_count":30}.
- Recommended next step: Treat standardization as insufficient and design a different algorithm route.
- Phase 70 does not alter rewards, enable B2/B3, validate suitability, prove PCA optimality, or revise formal manuscript files.

## Reproduction

Run from the repository root:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase70_standardized_set_policy_rerun\run_phase70_standardized_set_policy_rerun.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --phase61-output-dir experiments\phase61_d6_geofm_projection_controls\outputs\phase52_full5_seed3 --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --phase63-rollout-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_bc_rollout_summary.csv --variants B0,D4P8,D4P16,D6R8,D6R16 --eval-tile-ids tile_r002_c003,tile_r005_c004,tile_r005_c003,tile_r000_c004,tile_r001_c004 --eval-max-steps 8 --seeds 0,1,2 --bc-epochs 80 --learning-rate 0.001 --hidden-dim 64 --top-k 3 --output-dir experiments\phase70_standardized_set_policy_rerun\outputs\phase52_full5_seed3
```

## Boundary

Phase 70 is a standardized set-policy algorithm experiment under the existing Bishan base-reward protocol. It does not alter rewards, enable B2/B3, validate suitability, prove independent agronomic value, prove PCA optimality, or justify formal submission-level claims.