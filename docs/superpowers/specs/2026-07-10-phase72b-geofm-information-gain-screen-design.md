# Phase 72B GeoFM Information-Gain Screen Design

Date: 2026-07-10

Status: approved for implementation planning

## Objective

Phase 72B tests whether AlphaEarth representations contain practically useful
information about one-year farmland conversion beyond strong, temporally valid
explicit GIS and land-cover-history baselines. It is a low-cost, falsifiable
screen before any GeoFM-STaR temporal neural model or planning reward is built.

The primary question is:

```text
Does explicit GIS plus temporally summarized GeoFM improve future conversion
prediction beyond the strongest explicit-only model and strict representation
controls under locked temporal, buffered spatial, and cross-region validation?
```

Phase 72B does not alter the current planning reward, train PPO, implement the
deep GeoFM-STaR architecture, claim causal land-use effects, or revise
`paper/submission/final/*`.

## Inputs and Claim Boundary

The required label and embedding input is the passed Phase 72A package:

```text
experiments/phase72a_temporal_label_package/outputs/
  bishan_dongxing_esri_2017_2024/
```

Its primary endpoint is converted to the minority-class target:

```text
conversion_1y = 1 - y_1y
persistence_1y = y_1y
```

Two-year persistence and continuous two-year persistence are secondary
endpoints. Product labels remain independent annual land-cover product labels,
not manual ground truth, agronomic suitability, policy outcomes, or causal
effects.

Phase 72A used 2023-to-2024 labels for integrity and aggregate-rate auditing.
Phase 72B therefore describes the 2023-origin evaluation as an
analysis-protocol-locked test, not as a completely unopened or blinded test.
No model-specific 2023 results may be inspected before the Phase 72B protocol
and selected model manifest are frozen.

## Public Explicit GIS Package

### Terrain Source

The common terrain source is Copernicus DEM GLO-30 through Google Earth Engine:

```text
collection: COPERNICUS/DEM/GLO30
elevation band: DEM
```

Terrain is static and independent of the ESRI outcome labels, DLTB/base reward,
and AlphaEarth embeddings. Elevation is mosaicked for each tracked region,
slope is derived from the native DEM, and the following grid features are
aggregated to the Phase 72A 500 m grids:

- elevation mean, standard deviation, minimum, and maximum;
- slope mean, standard deviation, and maximum;
- local relief, defined as elevation maximum minus elevation minimum.

Expected arrays are exactly `67 x 70` for Bishan and `91 x 99` for Dongxing.
Every terrain asset records source, band or derivation, scale, bbox, shape,
dtype, path, and SHA256. A shape mismatch, missing band, missing year-independent
terrain asset, or unreadable file returns `phase72b_inputs_not_ready`. The
pipeline never crops, pads, interpolates, or silently resamples an unexpected
array.

### Explicit Feature Groups

All explicit features must be available no later than prediction origin year
`t`. The primary explicit feature groups are:

1. **Terrain:** the eight Copernicus DEM features above.
2. **Position:** cell-center longitude and latitude, normalized row and column,
   and region indicator for pooled models.
3. **Time:** origin year and history length.
4. **Cell LULC history through `t`:** previous class, previous-year crop flag,
   crop fraction across observed history, number of crop/non-crop transitions,
   years since the most recent non-crop observation, and one-hot counts for
   ESRI classes `1, 2, 4, 5, 7, 8, 9, 10, 11`, plus an unknown class bucket.
5. **Current neighborhood:** class proportions and crop fraction in centered
   `3 x 3` and `5 x 5` windows using the product label at `t`.
6. **Historical neighborhood:** mean and linear trend of the `3 x 3` and
   `5 x 5` crop fractions using labels only from 2017 through `t`.

Because the primary cohort contains cells classified as crop at `t`, current
cell class alone is constant and cannot serve as the strong baseline. Historical
and neighborhood features prevent the explicit comparator from being
deliberately weak.

DLTB parcel attributes are excluded from the primary cross-region comparator
because the existing Bishan and Dongxing sources have different schemas and
spatial units. A separately audited DLTB-to-grid aggregation may be added later
as a within-region sensitivity analysis; it cannot replace the common public
GIS baseline.

## GeoFM Feature Variants

Let the available AlphaEarth history for a sample be `z_2017, ..., z_t`, with
each vector having 64 dimensions. Padded future positions remain zero and are
excluded by the Phase 72A history mask.

The low-cost GeoFM feature variants are:

- `geofm_current`: `z_t`, 64 dimensions;
- `geofm_temporal_mean`: masked historical mean, 64 dimensions;
- `geofm_temporal_full`: concatenation of current value, masked mean, masked
  standard deviation, latest-minus-earliest delta, and per-dimension linear
  trend, 320 dimensions.

The required model variants are:

```text
explicit_static
explicit_history
geofm_current_only
geofm_temporal_mean_only
explicit_plus_geofm_current
explicit_plus_geofm_temporal_full
explicit_plus_temporal_order_shuffle
explicit_plus_spatial_shuffle
explicit_plus_random_projection
```

The primary explicit comparator is the best validation-approved model using
`explicit_history`. The primary GeoFM candidate is
`explicit_plus_geofm_temporal_full`.

## Strict Representation Controls

All controls use the same rows, targets, explicit features, model family,
hyperparameter search budget, and output dimensionality as the full GeoFM
candidate.

### Temporal-Order Shuffle

For each sample, the current embedding `z_t` remains fixed. Earlier embeddings
are deterministically permuted, then the 320-dimensional temporal summary is
recomputed. This preserves current state and the marginal set of historical
embeddings while destroying earlier temporal order where at least two earlier
years exist.

### Spatial Shuffle

Complete masked embedding histories are permuted across units within each
`region_id x origin_year` stratum. Targets, explicit features, history lengths,
and stratum sizes remain unchanged.

### Same-Dimension Random Projection

The zero-padded masked history is flattened from `8 x 64` to 512 dimensions and
multiplied by a fixed orthonormal random projection to 320 dimensions. The
projection uses no labels and is generated before model fitting.

Control seeds are fixed to `72, 73, 74, 75, 76`. The strongest validation
control seed is frozen for confirmation, and the complete five-seed control
distribution is reported. The full GeoFM candidate must exceed the strongest
frozen control, not only an average weak control.

## Models and Preprocessing

Every feature variant is evaluated with two model families:

1. L2-regularized logistic regression.
2. `HistGradientBoostingClassifier`.

The small validation-only search space is frozen as follows:

```text
Logistic C: 0.01, 0.1, 1.0, 10.0
Logistic class_weight: none, balanced
HGB learning_rate: 0.03, 0.08
HGB max_leaf_nodes: 15, 31
HGB min_samples_leaf: 20, 50
HGB max_iter: 200
HGB l2_regularization: 0.0, 1.0
```

Continuous explicit and GeoFM features are standardized for logistic
regression. The strongest explicit model is selected by validation conversion
AP, with Brier score as the first tie-breaker and ECE as the second. The same
selection rule applies to each GeoFM/control variant.

Calibration candidates are `none`, sigmoid scaling, and isotonic regression.
They are fitted on 2022 validation predictions only. The method with the lowest
validation Brier score is frozen; ECE is the tie-breaker. A calibrator cannot be
selected or refitted using 2023 labels or a target region's zero-shot test
labels.

All imputation, scaling, feature filtering, and any learned transformation are
fitted on training rows only. No test-year or target-region test rows may
influence preprocessing, model selection, class weighting, calibration,
thresholds, or feature definitions.

## Locked Validation Axes

### Pooled Temporal Test

The primary pooled analysis uses both regions:

```text
training origins: 2017-2021
validation and calibration origin: 2022
locked test origin: 2023
```

The selected model remains fitted on 2017-2021 rows and its calibrator remains
fitted on 2022 predictions. It is evaluated once on 2023 rows without refitting
or changing any choice.

### Buffered Spatial Test

Each region uses deterministic 8-cell spatial blocks inherited from Phase 72A.
Five spatial folds are assigned by a stable hash. For each fold, all directly
adjacent block rings are removed from training. Models train on 2017-2021
origins from remaining blocks, validate on 2022 remaining blocks, and test on
2023 held blocks. Invalid folds with missing class support are reported and not
silently replaced.

### Bidirectional Zero-Shot Transfer

Required transfer directions are:

```text
Bishan 2017-2021 train, Bishan 2022 validation, Dongxing 2023 test
Dongxing 2017-2021 train, Dongxing 2022 validation, Bishan 2023 test
```

The source-region model remains fitted on source-region 2017-2021 rows and its
calibrator remains fitted on source-region 2022 predictions. Target-region rows
are never used for preprocessing, calibration, feature selection,
hyperparameter selection, or refitting.

## Three-Stage Freeze Workflow

### Stage 1: `prepare`

This stage fetches and audits terrain, assembles explicit and GeoFM/control
feature matrices, validates all hashes and shapes, and writes the complete
split and metric protocol before fitting a model.

Required outputs include:

- terrain manifest and arrays;
- feature manifest and feature-group registry;
- row-alignment and leakage audit;
- frozen evaluation protocol JSON and SHA256.

### Stage 2: `fit-freeze`

This stage loads only training and validation outcomes, evaluates the frozen
candidate search, selects model family, hyperparameters, calibrator, and control
seed for each variant, and writes a selected-model manifest with its SHA256.
It does not compute 2023 model metrics.

### Stage 3: `confirm`

This stage verifies both protocol hashes, loads the frozen fitted models and
calibrators, evaluates 2023 and zero-shot outcomes once, and writes metrics,
paired deltas, uncertainty intervals, and the final gate.
Any hash mismatch or undeclared configuration returns
`phase72b_inputs_not_ready` and refuses confirmation.

## Metrics and Statistical Analysis

The primary target is one-year conversion. Core metrics are:

- conversion Average Precision;
- Brier score;
- Expected Calibration Error with fixed equal-frequency bins;
- conversion capture and net benefit at fixed 10% and 20% risk budgets.

ROC AUC, F1, balanced accuracy, and the two-year endpoints are secondary.
Threshold-dependent metrics use thresholds frozen from 2022 validation data.

All primary model comparisons use paired spatial-block bootstrap deltas with
2,000 replicates and seed 72. Pooled bootstrap sampling is stratified by region;
transfer bootstrap samples target-region blocks. Confidence intervals are the
2.5th and 97.5th percentiles. Rows are never treated as independent bootstrap
units.

## Practical and Statistical Gates

Relative to the strongest explicit comparator, the pooled locked test must
achieve at least two of:

```text
AP absolute increase >= 0.015
Brier absolute decrease >= 0.005
ECE absolute decrease >= 0.010
```

At least one of AP or Brier must also have a paired block-bootstrap 95% interval
entirely in the favorable direction.

The primary GeoFM candidate must exceed the strongest frozen temporal-shuffle,
spatial-shuffle, and random-projection controls. For control comparison, the
required practical margins are:

```text
AP increase over every frozen control >= 0.005
Brier decrease versus every frozen control >= 0.002
```

Each zero-shot direction must improve at least one of:

```text
AP increase >= 0.005
Brier decrease >= 0.002
```

and must not show either material degradation:

```text
AP decrease > 0.005
Brier increase > 0.002
```

Buffered spatial results must preserve the pooled improvement direction and
must not be driven by one region or one valid spatial fold. Budget-specific
conversion capture and net benefit are reported as decision-relevance evidence
but do not replace the core metric gate.

## Result Statuses

Phase 72B emits exactly one of:

- `phase72b_inputs_not_ready`: terrain, label, embedding, split, hash,
  calibration, class-support, or leakage audit failed.
- `geofm_information_not_supported`: the pooled practical/statistical gate or
  strict control gate failed.
- `geofm_information_mixed`: pooled evidence passed but spatial or cross-region
  evidence was materially heterogeneous.
- `geofm_information_supported`: pooled practical/statistical gate, all strict
  controls, buffered spatial direction, and both zero-shot transfer gates
  passed.

Only `geofm_information_supported` permits Phase 72C GeoFM-STaR development.
`geofm_information_mixed` permits only the predeclared heterogeneity audit; it
cannot support a positive multi-region GeoFM claim.

## Artifacts

Stable Phase 72B outputs include:

```text
phase72b_terrain_manifest.csv
phase72b_feature_manifest.csv
phase72b_feature_registry.json
phase72b_row_alignment_audit.csv
phase72b_leakage_audit.json
phase72b_frozen_protocol.json
phase72b_frozen_protocol.sha256
phase72b_selected_models.json
phase72b_selected_models.sha256
phase72b_metrics.csv
phase72b_predictions.csv
phase72b_calibration.csv
phase72b_bootstrap_deltas.csv
phase72b_control_comparison.csv
phase72b_transfer_summary.csv
phase72b_information_gain_screen.json
phase72b_information_gain_screen.md
```

Generated terrain arrays, feature matrices, model objects, and confirmation
outputs remain ignored. Tracked code, contracts, tests, result documentation,
and hashes must be sufficient to reproduce and audit the run.

## Error Handling

- Missing or unexpected terrain/label/embedding assets block the run.
- Unknown feature groups, duplicate sample indexes, row-order changes, or
  non-contiguous indexes block the run.
- A feature timestamp later than its origin year blocks the run.
- A preprocessing fit that includes validation, test-year, or target-region
  test rows blocks the run.
- A selected-model or protocol hash mismatch blocks confirmation.
- Missing outcome classes invalidate the affected fold and are reported.
- Failed controls are not dropped from the gate.
- No fallback proxy label, post hoc threshold, region deletion, or metric
  substitution is allowed.

## Testing Strategy

Implementation follows red-green-refactor TDD and includes:

- terrain contract, shape, source, and hash tests;
- hand-computable explicit LULC-history and neighborhood feature tests;
- history-mask and no-future-feature tests;
- deterministic temporal, spatial, and random-projection control tests;
- split, spatial-buffer, and cross-region isolation tests;
- train-only preprocessing and calibration tests;
- protocol and selected-model hash refusal tests;
- hand-computable AP, Brier, ECE, bootstrap, and gate tests;
- `prepare`, `fit-freeze`, and `confirm` CLI fixture tests;
- real Bishan-Dongxing run, artifact inspection, adjacent regressions, smoke
  check, and formal-manuscript zero-diff check.

## Completion Criteria

Phase 72B is complete when:

1. the public terrain and feature packages pass for both regions;
2. the frozen protocol and selected-model hashes are written before
   confirmation;
3. every required model and control is evaluated on the locked axes;
4. paired spatial-block uncertainty is complete;
5. the final status is assigned without post hoc changes;
6. a measured result note and handoff update record the exact evidence and
   claim boundary;
7. the formal manuscript remains unchanged.

The next phase is conditional. Phase 72C may begin only if the final Phase 72B
status is `geofm_information_supported`.
