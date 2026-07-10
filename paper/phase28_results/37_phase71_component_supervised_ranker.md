# Phase 71 Component-Supervised Listwise Ranker

Status: ranker_improves_but_target_masks_geofm

## Key Evidence

- Phase 71 trained a component-supervised listwise ranker under the existing Bishan base-reward protocol.
- The experiment preserved original feature matrices for reward and oracle scoring.
- Phase 71 minus Phase 63 reward summary: {"max_delta":1.1167472976,"mean_delta":0.42645,"min_delta":-0.0312008374,"positive_count":74,"total_count":75}.
- Phase 71 minus Phase 70 reward summary: {"max_delta":1.8612068485,"mean_delta":0.4340833829,"min_delta":-0.0995773531,"positive_count":53,"total_count":75}.
- D4 versus B0 Phase 71 delta summary: {"max_delta":0.0314642968,"mean_delta":-0.0498759068,"min_delta":-0.2071761723,"positive_count":5,"total_count":30}.
- D4 versus D6 Phase 71 delta summary: {"max_delta":0.0392837765,"mean_delta":-0.0116453458,"min_delta":-0.1027135046,"positive_count":16,"total_count":30}.
- Recommended next step: Keep Phase 71 as a stronger decision-learning baseline, but treat the explicit base target as masking GeoFM-specific value.
- Phase 71 does not alter rewards, enable B2/B3, validate suitability, prove GeoFM superiority, prove PCA optimality, or revise formal manuscript files.

## Reproduction

Run from the repository root:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase71_component_supervised_ranker\run_phase71_component_supervised_ranker.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --phase61-output-dir experiments\phase61_d6_geofm_projection_controls\outputs\phase52_full5_seed3 --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --phase63-rollout-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_bc_rollout_summary.csv --phase70-rollout-csv experiments\phase70_standardized_set_policy_rerun\outputs\phase52_full5_seed3\phase70_standardized_bc_rollout_summary.csv --variants B0,D4P8,D4P16,D6R8,D6R16 --eval-tile-ids tile_r002_c003,tile_r005_c004,tile_r005_c003,tile_r000_c004,tile_r001_c004 --eval-max-steps 8 --seeds 0,1,2 --ranker-epochs 80 --learning-rate 0.001 --hidden-dim 64 --component-weight 0.05 --top-k 3 --output-dir experiments\phase71_component_supervised_ranker\outputs\phase52_full5_seed3
```

## Boundary

Phase 71 is an algorithm/model experiment under the existing Bishan base-reward protocol. It does not alter rewards, enable B2/B3, validate suitability, prove independent agronomic value, prove GeoFM superiority, prove PCA optimality, test transfer, or justify formal submission-level claims.
