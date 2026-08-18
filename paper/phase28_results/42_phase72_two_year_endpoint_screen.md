# Phase 72 Two-Year Endpoint Information-Gain Screen

Status: `two_year_geofm_information_not_supported`

## Question

The official Phase 72B screen evaluated one-year farmland conversion and did
not support representation-specific, transferable, or spatially stable GeoFM
information. This separate Phase 72 exhaustion experiment tested whether the
same conclusion changed at a two-year horizon. It did not enter Phase 72C,
alter a planning reward, or modify the formal manuscript.

## Frozen Design

The tracked protocol defined two endpoints before confirmation:

- `conversion_2y = 1 - y_2y`, indicating non-farmland status two years after
  the origin year;
- `noncontinuous_persistence_2y = 1 - y_continuous_2y`, indicating failure to
  remain farmland in both subsequent annual observations.

Both endpoints had to pass every frozen gate for an overall positive result.
The temporal split used origins 2017--2020 for training, 2021 for validation,
and 2022 for locked confirmation. The experiment retained the Phase 72B seeds
`72--76`, temporal-order shuffle, spatial shuffle, same-dimension random
projection, bidirectional transfer, five buffered spatial folds per region,
2,000 paired block-bootstrap replicates, and all practical and no-harm
thresholds.

An initial full candidate-grid run was stopped before its first model
checkpoint after the wall-clock cost showed that it would not remain a
low-cost screen. No validation metrics were inspected and confirmation targets
had not been opened. Commit `98824fc` then froze a computational amendment:
reuse the candidate configurations already bound by the official Phase 72B
selected-model hash, but refit every estimator and calibration on the new
two-year development labels. All five control seeds and all validation axes
remained in the experiment.

## Integrity and Coverage

The prepared package contained `28,586` eligible samples. Development and
locked confirmation contained `24,690` and `3,896` samples, respectively.
`conversion_2y` had `9,759` positives overall and `1,100` in confirmation;
`noncontinuous_persistence_2y` had `12,126` positives overall and `1,755` in
confirmation.

The fit produced `142` independently refitted model bundles: `71` per
endpoint. All bundle hashes matched the selected-model manifest. Confirmation
produced `142` metric rows and `280,512` prediction rows. The receipt covered
all eight result artifacts, its canonical-JSON sidecar matched, and no artifact
hash differed.

Frozen identities:

```text
prepared package: 4e71071037a636d85c8b9ead1819c769faf610c5f078d59df31f9ba9bd241531
selected models:  cb1941b40d2982b16738c559e73f476bc906466f357a8305aa2818a2d9be574e
confirmation:     f5a3dcc99e828ae6558d175ca9b162d4198ecd5c452fbb804fb0fe570da00d1d
```

## Results

For `conversion_2y`, the full temporal GeoFM model underperformed the explicit
history baseline in pooled average precision (AP delta `-0.018988632052`). Its
Brier improvement was only `+0.000410629917`, and its ECE improvement was
`+0.008331641770`; none reached the practical thresholds. The paired block
bootstrap also failed: AP delta mean `-0.018943853422` with 95% interval
`[-0.039968587920, 0.001790106574]`, and Brier delta mean
`+0.000448097976` with interval `[-0.003979819475, 0.005026152130]`.
Temporal-order shuffle failed the strict control gate, both transfer directions
failed, and the weighted spatial deltas were negative in both Bishan and
Dongxing.

For `noncontinuous_persistence_2y`, pooled AP was effectively unchanged (delta
`+0.000266220113`), while Brier and ECE worsened by `-0.005385588416` and
`-0.040125539855`. The AP bootstrap interval
`[-0.016824098439, 0.018144439200]` and Brier interval
`[-0.011278298266, 0.000573798335]` both crossed zero in the favorable
direction. Temporal-order and spatial-shuffle controls failed. Only one
transfer direction passed, and both regional spatial summaries were negative.

| Endpoint | Practical | Statistical | Controls | Transfer | Spatial | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `conversion_2y` | fail | fail | fail | fail | fail | `geofm_information_not_supported` |
| `noncontinuous_persistence_2y` | fail | fail | fail | fail | fail | `geofm_information_not_supported` |

## Interpretation

The two-year horizon does not rescue the original GeoFM-specific hypothesis in
the current Bishan--Dongxing product-label experiment. The negative result is
stronger than a single pooled miss: neither endpoint passed the practical or
statistical gate, strict controls did not establish representation specificity,
bidirectional transfer was absent, and spatial direction was unfavorable in
both regions.

This result is not a universal claim that GeoFM can never carry useful land-use
information. It is bounded to the available ESRI annual labels, the two study
regions, the explicit-history comparator, the Phase 72B model configurations,
and the frozen thresholds. It also does not test agronomic suitability,
causality, policy learning, or constrained planning outcomes.

## Reproduction

Tracked protocol and runner:

```text
experiments/phase72_two_year_endpoint_screen/phase72_two_year_protocol.json
experiments/phase72_two_year_endpoint_screen/run_phase72_two_year_endpoint_screen.py
```

Ignored real outputs:

```text
experiments/phase72_two_year_endpoint_screen/outputs/prepared_fixed_configs
experiments/phase72_two_year_endpoint_screen/outputs/frozen_fixed_configs
experiments/phase72_two_year_endpoint_screen/outputs/confirmation_fixed_configs
```

## Claim Boundary

This Phase 72 exhaustion experiment tests two-year product-label prediction
with the frozen Phase 72B controls and gates. It does not enter Phase 72C,
establish agronomic suitability, run constrained planning, revise the formal
manuscript, or override the official one-year Phase 72B result.
