# Phase 72 Temporal Neural Exhaustion Screen

Status: `temporal_neural_information_not_supported`

Phase 72C allowed: `false`

## Scientific Question

This frozen exhaustion experiment tested whether a compact temporal neural
encoder could convert annual GeoFM history into robust one-year conversion
information beyond a strong explicit-history baseline. It targeted
`conversion_1y`, the Phase 72 primary endpoint and the only endpoint with a
surviving pooled signal in the earlier explicit residual screen.

This was not Phase 72C. The experiment excluded a two-year neural extension,
decision-aware ranking, domain regularization, ensembles, uncertainty bounds,
reward changes, and planning. Its purpose was to evaluate the remaining
`temporal_neural_model` exhaustion criterion without adding post hoc rescue
components.

## Frozen Model and Integrity

For every validation axis, five-fold `spatial_block_id` cross-fitting generated
explicit training logits. A 1,057-parameter, bias-free gated temporal branch
then encoded the masked annual `64`-channel GeoFM history and added a residual
logit to the explicit model. Channel standardization used valid training
history entries only. Model fitting used AdamW with one fixed seed;
early-stopping epoch and calibration were selected on validation rows only.

The screen reused the original Phase 72B thresholds, both transfer directions,
and ten buffered spatial folds. Pooled controls comprised temporal-order
shuffle, partition-local spatial shuffle, and a same-shape `64 x 64` random
orthogonal channel projection under seeds 72-76.

Confirmation was opened only after three GitHub checkpoints:

```text
preregistered implementation: 3375044
baseline-checkpoint fix: 2a6a549
confirmation-freeze receipt: e5bfa48
```

Execution identities and audits:

```text
prepared SHA256: a50f5bca4b8ffff4c0233e5de545cd06a309657a1f6b57310e8a7930187bdb1f
selected-model SHA256: 76ed030d5fd115b70e6aca2f1c0f256101c4295f1a216a43ebfb4f0f6aa27fcf
confirmation-receipt SHA256: 4020b79dda5e70db4c8eefee16b7ddf267df40102b2dc046f500455acda08802
bundles: 41 total; 28 neural; 13 explicit
confirmation metrics / predictions: 41 / 63,861
bundle hash mismatches: 0 / 41
invalid cross-fit audits: 0 / 28
```

## Confirmation Result

The temporal neural screen returned
`temporal_neural_information_not_supported`. Relative to the explicit-history
baseline on the pooled confirmation axis, the neural model changed the three
primary metrics as follows:

```text
AP delta:     +0.020692254076
Brier delta:  -0.001747246378
ECE delta:    -0.009887323960
AP CI95:      [+0.010568362016, +0.033106428075]
Brier CI95:   [-0.003383805202, -0.000064016120]
```

Positive Brier and ECE deltas denote lower error. The AP improvement was
positive and its block-bootstrap interval excluded zero, but Brier and ECE
worsened. The model therefore passed only one of the three practical metrics,
whereas the frozen practical gate required at least two. The statistical check
passed through AP, but the practical and strict-control checks failed.

The primary model did not exceed any strict control by the required AP and
Brier margins:

| Control | AP delta | Brier delta | Gate |
| --- | ---: | ---: | --- |
| Temporal-order shuffle | +0.000040 | -0.000042 | failed |
| Spatial shuffle | -0.003606 | +0.001128 | failed |
| Random orthogonal projection | -0.006599 | -0.001294 | failed |

Transfer also failed in both directions. Bishan-to-Dongxing AP improved by
`+0.031997`, but Brier worsened by `-0.025233`; Dongxing-to-Bishan AP and Brier
changed by `-0.000794` and `-0.002781`. Fold-level spatial directions remained
heterogeneous: four of ten folds had negative AP deltas, and five of ten had
negative Brier deltas. Although row-weighted regional AP deltas were positive,
the frozen spatial gate required stable fold-level directions and therefore
failed.

## Scientific Interpretation

The compact temporal neural encoder changed pooled ranking performance, but it
did not recover a calibrated, representation-specific, transferable, or
spatially stable GeoFM advantage. Its pooled AP signal was nearly unchanged by
temporal-order shuffle and was weaker than the selected spatial-shuffle and
random-projection controls on AP. This pattern is incompatible with the broad
claim that learned temporal GeoFM structure provides a robust future-stability
signal beyond explicit history.

Combined with the earlier evidence, the original Paper11 future-aware target
is not established. The explicit residual screen retains a narrower mixed
observation: endpoint- and region-dependent short-horizon residual information
may exist. The temporal neural result does not strengthen that observation into
a general prediction or planning claim.

The refreshed Phase 72 exhaustion audit marks `temporal_neural_model` as
`evaluated_negative`. Seven of eleven criteria are now negative or mixed and
three remain unresolved: a second independent annual product, label
disagreement/noise sensitivity, and constrained planning outcomes. Complete
scientific exhaustion is therefore not claimed, but the current evidence does
not support proceeding to Phase 72C or promoting GeoFM future-stability or
planning advantage.

## Verification

```text
focused temporal-neural, exhaustion, residual, and two-year tests: 46 passed
full repository: 599 passed, 84 existing sklearn warnings
smoke check: passed
git diff --check: passed
paper/submission/final/*: unchanged
```

## Local Artifacts

Ignored real-run outputs remain under:

```text
experiments/phase72_temporal_neural_screen/outputs/prepared
experiments/phase72_temporal_neural_screen/outputs/benchmark
experiments/phase72_temporal_neural_screen/outputs/frozen
experiments/phase72_temporal_neural_screen/outputs/confirmation
experiments/phase72_exhaustion_analysis/outputs/real_bishan_dongxing_residual_neural
```

## Claim Boundary

This result does not establish agronomic suitability, a transferable GeoFM
prediction advantage, a planning benefit, or a reason to revise
`paper/submission/final/*`. Do not enter Phase 72C or add a post hoc two-year
neural extension from this result.
