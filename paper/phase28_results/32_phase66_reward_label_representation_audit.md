# Phase 66 Reward-Label Representation Audit

Status: base_reward_target_masks_geofm_signal

Alignment advantage: {'b0_explicit_proxy_r2_mean': 0.9973990529, 'geofm_explicit_proxy_r2_mean': 0.9973990529, 'geofm_representation_proxy_r2_mean': 0.029462, 'representation_minus_b0_proxy_r2': -0.9679370606, 'representation_minus_explicit_proxy_r2': -0.9679370606, 'representation_minus_explicit_topk': -0.08125}
Failure mode counts: {'near_oracle_reward_equivalent': 0, 'misses_explicit_reward_components': 75, 'representation_not_aligned_with_base_reward': 60, 'standardization_hurts_rank_geometry': 30, 'tile_specific_instability': 2, 'seed_instability': 3}

Claim boundary:
Phase 66 is a read-only reward-label and representation attribution audit over existing Phase 63, Phase 64, and Phase 65 artifacts plus raw tiled feature matrices. It does not train or fine-tune a policy, does not change the base reward, does not enable suitability reward, does not test B2/B3 or transfer, does not prove GeoFM advantage or PCA optimality, and does not justify formal submission-level claims.

## Reproduction

Run Phase 66 from the repository root after Phase 63, Phase 64, and Phase 65 full-run artifacts exist:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase66_reward_label_representation_audit\run_phase66_reward_label_representation_audit.py --phase63-comparison-json experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_set_policy_comparison.json --phase63-rollout-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_bc_rollout_summary.csv --phase63-oracle-summary-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_oracle_summary.csv --phase64-failure-cases-csv experiments\phase64_set_policy_error_diagnosis\outputs\phase52_full5_seed3\phase64_failure_cases.csv --phase64-feature-effective-rank-csv experiments\phase64_set_policy_error_diagnosis\outputs\phase52_full5_seed3\phase64_feature_effective_rank.csv --phase65-comparison-json experiments\phase65_standardized_set_policy_bc_rerun\outputs\phase52_full5_seed3\phase65_set_policy_comparison.json --phase65-rollout-csv experiments\phase65_standardized_set_policy_bc_rerun\outputs\phase52_full5_seed3\phase65_bc_rollout_summary.csv --phase65-pairwise-delta-csv experiments\phase65_standardized_set_policy_bc_rerun\outputs\phase52_full5_seed3\phase65_standardization_pairwise_delta.csv --phase10-reward-readiness-json experiments\phase11_bishan_dltb_real\outputs\phase10_real\phase10_reward_readiness_gate.json --output-dir experiments\phase66_reward_label_representation_audit\outputs\phase52_full5_seed3
```

## Boundary

No formal manuscript files were changed in this phase.
