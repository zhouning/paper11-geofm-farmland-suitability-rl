# Phase 36 Suitability-Proxy Validation Design

## Goal

Phase 36 tests whether current GeoFM-derived feature families contain
weak-label suitability signal before Paper11 enables suitability reward or
starts B2/B3 planning-performance experiments.

## One-Sentence Argument

In the current Bishan real-data pipeline, Phase 36 evaluates whether
GeoFM-based representations add weak-label predictive signal beyond explicit
planning features under spatial held-out validation, while treating all labels
as proxy labels rather than agronomic ground truth.

## Scope

Phase 36 is read-only. It consumes existing feature tables and writes
diagnostic artifacts. It does not run PPO, does not alter rewards, does not
enable B2/B3, and does not support final planning-performance claims.

Inputs:

- Phase 2 real block features: `block_geofm_features.csv`.
- Phase 2 variant tables for B0, B1, B2, and B3 when available.
- Optional Phase 8 controls for D2, D3, D4P8, and D4P16.
- Optional Phase 30 normalized controls for N1Z and N1ZR.
- Label columns such as `current_farmland_label`,
  `farmland_or_orchard_label`, and `low_slope_farmland_label`.

Outputs:

- feature-family validation CSV;
- label summary CSV;
- JSON diagnosis;
- Markdown interpretation;
- updated result/readme documentation after the real run.

## Feature Families

The first implementation evaluates these families when inputs exist:

- `explicit_only`: B0 explicit planning features.
- `raw_geofm_only`: B1 embedding columns only.
- `explicit_plus_raw_geofm`: B1 full state.
- `suitability_proxy_only`: the current scalar suitability proxy only.
- `explicit_plus_suitability_proxy`: B2 full state.
- `explicit_plus_random_geofm`: D2 full state.
- `explicit_plus_shuffled_geofm`: D3 full state.
- `explicit_plus_pca8_geofm`: D4P8 full state.
- `explicit_plus_pca16_geofm`: D4P16 full state.
- `explicit_plus_normalized_geofm_zscore`: N1Z full state, if Phase 30
  normalized controls are supplied.
- `explicit_plus_normalized_geofm_zscore_row_l2`: N1ZR full state, if Phase
  30 normalized controls are supplied.

## Validation Design

Phase 36 uses spatial held-out validation by default. If a `split` column is
available, rows with `split=train` train the model and rows with
`split=test` or `split=val` form the evaluation set. If no split is available,
the runner can use deterministic k-fold validation.

For each label and feature family:

- drop rows with missing or non-binary labels;
- require both positive and negative labels in train and evaluation sets;
- train a small logistic-regression classifier with standard scaling;
- report ROC AUC, average precision, balanced accuracy, accuracy, positive
  rate, train/eval counts, and feature count;
- report the top absolute linear coefficients as a diagnostic, not as causal
  attribution.

## Leakage Boundary

The current Bishan weak labels are partly derived from DLTB class and slope.
Some explicit planning features directly encode those same concepts, for
example `explicit_feature_04`, `explicit_feature_07`,
`explicit_feature_13`, and `explicit_feature_16`. Therefore high explicit-only
scores on `current_farmland_label`, `farmland_or_orchard_label`, or
`low_slope_farmland_label` are expected and do not prove suitability validity.

Phase 36 must flag these labels as `explicit_label_leakage_risk` and treat the
result as a proxy-readiness diagnostic only. A stronger future result requires
external high-standard farmland, yield, irrigation, soil, or independent
retention labels.

## Status Rule

The aggregate status is conservative:

- `proxy_signal_supported_for_bounded_reward_smoke`: at least one usable label
  shows `explicit_plus_raw_geofm` or a normalized GeoFM family improving both
  explicit-only and random/shuffled controls by at least `0.02` ROC AUC and
  average precision without split failures.
- `proxy_signal_not_supported`: usable labels exist, but GeoFM families do not
  beat explicit-only and controls by the threshold.
- `insufficient_proxy_labels`: no label has usable binary train/eval coverage.

Even the supported status only allows a bounded B2/B3 reward smoke experiment.
It does not authorize final suitability-reward or planning-performance claims.

## Claim Boundary

Phase 36 may support this guarded statement:

> Under weak-label spatial held-out validation, the current feature tables show
> whether GeoFM-derived features add proxy-label signal beyond explicit
> planning features and diagnostic controls.

Phase 36 may not support:

> GeoFM measures farmland suitability, soil quality, irrigation quality, or
> final planning performance.
