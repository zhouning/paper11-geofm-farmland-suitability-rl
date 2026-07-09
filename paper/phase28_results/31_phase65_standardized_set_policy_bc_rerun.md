# Phase 65 Standardized Set-Policy BC Rerun

Status: standardization_not_helpful

Mean standardized BC reward by variant:
- B0: 5.0730631557
- D4P16: 4.4798692838
- D4P8: 5.0088517682
- D6R16: 4.8323460998
- D6R8: 4.8531476548

Overall standardized-minus-unstandardized summary: {'mean_delta': -0.071192921, 'positive_count': 45, 'total_count': 75, 'min_delta': -1.6501980738, 'max_delta': 1.0127905299}
D4 standardized-minus-unstandardized summary: {'mean_delta': -0.1435525318, 'positive_count': 16, 'total_count': 30, 'min_delta': -1.4410704099, 'max_delta': 1.0127905299}
D4/B0 delta summary after standardization: {'mean_delta': -0.3287026297, 'positive_count': 6, 'total_count': 30, 'min_delta': -1.7110408087, 'max_delta': 0.9502730653}
D4/D6 delta summary after standardization: {'mean_delta': -0.0983863509, 'positive_count': 15, 'total_count': 30, 'min_delta': -1.3783311504, 'max_delta': 1.3190410962}
Oracle gap summary after standardization: {'mean_delta': 0.102231062, 'positive_count': 74, 'total_count': 75, 'min_delta': 0.0, 'max_delta': 0.403338}

Claim boundary:
Phase 65 is a base-reward train-tile-fitted standardized set-policy behavior-cloning rerun. Standardization is applied only to policy model inputs; oracle targets and rollout rewards remain computed from raw unstandardized feature matrices. It does not enable suitability reward, does not test B2/B3, does not test transfer, does not prove GeoFM advantage or PCA optimality, and does not justify formal submission-level claims.

## Reproduction

Run Phase 65 from the repository root after Phase 63 full-run artifacts exist:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase65_standardized_set_policy_bc_rerun\run_phase65_standardized_set_policy_bc_rerun.py --phase63-comparison-json experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_set_policy_comparison.json --phase63-rollout-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_bc_rollout_summary.csv --existing-flattened-summary-csvs experiments\phase52_expanded_cluster_replication\outputs\real_bishan_4096_5tiles\phase28_representation_control_summary.csv,experiments\phase62_d4_d6_matched_ppo\outputs\phase52_full5_seed3\phase62_d4_d6_matched_ppo_summary.csv --output-dir experiments\phase65_standardized_set_policy_bc_rerun\outputs\phase52_full5_seed3
```

## Boundary

No formal manuscript files were changed in this phase.
