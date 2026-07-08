# Phase 63 Set-Policy Oracle Pretraining

Status: architecture_improves_but_geofm_not_distinguished

Conclusion:
Phase 63 supports the set-policy architecture route, but does not separate GeoFM-derived variants from B0.

Mean behavior-cloned reward by variant:
- B0: 4.9556965601
- D4P16: 4.8822289094
- D4P8: 4.8935972062
- D6R16: 4.9244544652
- D6R8: 4.9472654273

Mean oracle reward by variant:
- B0: 5.3920694097
- D4P16: 5.3920694097
- D4P8: 5.3920694097
- D6R16: 5.3920694097
- D6R8: 5.3920694097

Architecture delta summary: {'mean_delta': 4.4387176072, 'positive_count': 75, 'total_count': 75, 'min_delta': 2.3454884885, 'max_delta': 6.0659705018}
D4/B0 delta summary: {'mean_delta': -0.0677835004, 'positive_count': 10, 'total_count': 30, 'min_delta': -0.5905151188, 'max_delta': 0.1363046136}
D4/D6 delta summary: {'mean_delta': -0.0479468867, 'positive_count': 7, 'total_count': 30, 'min_delta': -0.574438559, 'max_delta': 0.0775225301}
Oracle gap summary: {'mean_delta': 0.0882844088, 'positive_count': 75, 'total_count': 75, 'min_delta': 0.0177772236, 'max_delta': 0.2099909286}

Claim boundary:
Phase 63 is a base-reward set-policy oracle-pretraining experiment. It tests whether task-aware block scoring and deterministic oracle behavior cloning improve candidate-block selection under existing Bishan tile inputs. It does not enable suitability reward, does not test B2/B3, does not test cross-region transfer, does not prove independent agronomic suitability, does not prove PCA optimality, and does not justify final submission-level planning-performance claims.

## Reproduction

Run Phase 63 from the repository root after the Phase 2 B0 table, Phase 8 D4
tables, Phase 61 D6 tables, Phase 13 tile index, Phase 52 flattened PPO
summary, and Phase 62 D4/D6 flattened PPO summary exist. The current local run
used `D:\adk\.venv\Scripts\python.exe`; a stable Python environment with
PyTorch can replace that executable.

```powershell
python experiments\phase63_set_policy_oracle_pretraining\run_phase63_set_policy_oracle_pretraining.py --mode run-and-analyze --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --phase61-output-dir experiments\phase61_d6_geofm_projection_controls\outputs\phase52_full5_seed3 --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,D4P8,D4P16,D6R8,D6R16 --eval-tile-ids tile_r002_c003,tile_r005_c004,tile_r005_c003,tile_r000_c004,tile_r001_c004 --eval-max-steps 8 --seeds 0,1,2 --bc-epochs 80 --learning-rate 0.001 --hidden-dim 64 --top-k 3 --existing-flattened-summary-csvs experiments\phase52_expanded_cluster_replication\outputs\real_bishan_4096_5tiles\phase28_representation_control_summary.csv,experiments\phase62_d4_d6_matched_ppo\outputs\phase52_full5_seed3\phase62_d4_d6_matched_ppo_summary.csv --output-dir experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3
```

## Boundary

Phase 63 is algorithm/model evidence under the existing deterministic
base-planning reward. It does not enable suitability reward, does not test
B2/B3, does not test cross-region transfer, does not prove independent
agronomic suitability, does not prove PCA optimality, and does not justify
formal submission-level planning-performance claims. No formal manuscript files
were changed in this phase.
