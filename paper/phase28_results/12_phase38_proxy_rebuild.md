# Phase 38 Proxy-Rebuild Diagnostic

Phase 38 rebuilds suitability-proxy classifiers under leakage-aware spatial held-out validation before any B2/B3 reward integration.

## Experiment Snapshot

Inputs:

- Phase 2 real Bishan block features: `experiments/phase11_bishan_dltb_real/outputs/phase2_real`
- Phase 8 representation controls: `experiments/phase8_ablation_controls/outputs/real_bishan_controls`
- Phase 30 normalized controls: `experiments/phase30_normalized_b1_ablation/outputs/real_bishan_4096_incremental/derived_normalized_controls`

Output directory:

```text
experiments/phase38_proxy_rebuild/outputs/real_bishan
```

The real run evaluated three label columns, eleven feature families, and three lightweight classifier families (`logistic_elastic_net`, `random_forest`, and `hist_gradient_boosting`). The full rebuilt proxy scores are stored in `phase38_rebuilt_proxy_scores.csv`; the diagnosis JSON stores a bounded preview plus the total row count.

## Main Result

Status: `proxy_rebuild_diagnostic_only`

Row counts from `phase38_proxy_rebuild.json`:

- block rows: `64984`
- feature families: `11`
- label summaries: `3`
- model rows: `99`
- rebuilt proxy score rows: `6433416`

Interpretation: Phase 38 remains diagnostic only: either evaluated labels were explicit leakage risks or GeoFM-derived rebuilt proxies did not clear the control thresholds.

## Label Boundary

All three real labels are usable but classified as `explicit_label_leakage_risk`:

- `current_farmland_label`: `64984` valid labels, positive rate `0.3902345193`
- `farmland_or_orchard_label`: `64984` valid labels, positive rate `0.4351686569`
- `low_slope_farmland_label`: `64984` valid labels, positive rate `0.1145358858`

Because these labels are DLTB/slope-derived, they can validate that the proxy-rebuild pipeline runs on real tables, but they cannot unlock B2/B3 reward use. The perfect `explicit_only` scores in the model summary are a leakage warning, not evidence of independent agronomic validity.

## Reproduction

Run from the repository root after the Phase 2, Phase 8, and Phase 30 generated inputs exist:

```powershell
python experiments\phase38_proxy_rebuild\run_phase38_proxy_rebuild.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --normalized-controls-dir experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental\derived_normalized_controls --output-dir experiments\phase38_proxy_rebuild\outputs\real_bishan --label-columns current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label --model-families logistic_elastic_net,random_forest,hist_gradient_boosting
```

Expected local artifacts:

- `phase38_label_summary.csv`
- `phase38_model_summary.csv`
- `phase38_rebuilt_proxy_scores.csv`
- `phase38_proxy_rebuild.json`
- `phase38_proxy_rebuild.md`

## Claim Boundary

Phase 38 is diagnostic. It does not run PPO, does not alter rewards, does not enable B2/B3 by default, does not prove GeoFM agronomic validity, and does not support final planning-performance claims. With the current real-label set, B2/B3 suitability reward remains blocked rather than allowed even as a bounded smoke claim.