# Phase 64 Set-Policy Error Diagnosis

Status: standardization_route_supported

Recommendation: standardized rerun = True

Reason: D4/D6 underperformance coincides with feature scale, shift, or rank flags.

Gate evidence:
- mean best top-1 accuracy: 0.9916666667
- mean best top-k hit rate: 1.0
- D4 underperformance: True
- scale flag count: 24
- shift flag count: 0
- rank flag count: 24

Failure case rows:
- 12

Claim boundary:
Phase 64 is a read-only set-policy error-diagnosis and standardization-gate phase. It uses Phase 63 base-reward artifacts to diagnose behavior-cloned set-policy errors and decide whether a train-tile-fitted standardization rerun is justified. It does not enable suitability reward, does not test B2/B3, does not test transfer, does not prove GeoFM advantage or PCA optimality, and does not justify formal submission-level claims.

## Reproduction

Run Phase 64 from the repository root after Phase 63 full-run artifacts exist:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase64_set_policy_error_diagnosis\run_phase64_set_policy_error_diagnosis.py --phase63-comparison-json experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_set_policy_comparison.json --phase63-rollout-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_bc_rollout_summary.csv --phase63-history-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_bc_training_history.csv --phase63-oracle-summary-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_oracle_summary.csv --output-dir experiments\phase64_set_policy_error_diagnosis\outputs\phase52_full5_seed3
```

## Boundary

Phase 64 is diagnostic evidence under the existing deterministic base-planning
reward. It does not run a new policy training phase, does not enable
suitability reward, does not test B2/B3, does not test cross-region transfer,
does not prove GeoFM advantage, does not prove PCA optimality, and does not
justify formal submission-level planning-performance claims. No formal
manuscript files were changed in this phase.
