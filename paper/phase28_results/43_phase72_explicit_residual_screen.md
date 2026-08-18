# Phase 72 Explicit Residual Exhaustion Screen

Status: `explicit_residual_information_mixed`

Phase 72C allowed: `false`

## Scientific Question

This frozen exhaustion experiment tested whether temporal GeoFM history adds
future land-cover information after a strong explicit-history model has already
explained the predictable risk. It did not train Phase 72C, alter rewards, run
planning, or modify the formal submission.

For every endpoint and validation axis, five-fold spatial-block cross-fitting
generated explicit training logits. A no-intercept, L2-regularized GeoFM
residual was then added to those logits. The full explicit model and residual
head were refitted on training rows, with model and calibration choices made on
the frozen validation year only. Confirmation labels were opened only after
the implementation and all model hashes were committed to GitHub.

## Frozen Scope and Integrity

The screen evaluated:

- `conversion_1y` with 2023 confirmation labels;
- `conversion_2y` with 2022 confirmation labels;
- `noncontinuous_persistence_2y` with 2022 confirmation labels;
- pooled temporal evaluation, both transfer directions, and ten buffered
  spatial folds;
- temporal-order shuffle, spatial shuffle, and same-dimension random
  projection controls under seeds 72-76.

Execution identities:

```text
preregistered implementation commit: e3b2144ae906349f6a6d520200b17e16359c64c6
confirmation-freeze commit: 8fae9c4f9b6aa8e01501360589392527b711259f
prepared SHA256: 184ade17e02aa86aac2cd3ccb372d1d245be5bd149b734a88fb3df1a9235f396
selected-model SHA256: d49d4e0c57fcf75b668d3a30c1177e2b2600ca02195697ad991a4afbf4762628
confirmation receipt SHA256: c945c4c534c76f81697f06fc4dbc064a473e2da5d9400fcaed2f0f0384f54b81
bundles: 123 total; 84 residual
confirmation metric rows: 123
confirmation prediction rows: 227,493
receipt artifact mismatches: 0 / 8
invalid cross-fit audits: 0 / 84
```

## Endpoint Results

| Endpoint | Gate status | AP delta | Brier delta | ECE delta | AP CI95 | Brier CI95 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `conversion_1y` | `geofm_information_mixed` | +0.023960 | +0.021967 | +0.084490 | [-0.001086, +0.049586] | [+0.015270, +0.029178] |
| `conversion_2y` | `geofm_information_not_supported` | -0.015573 | -0.053778 | -0.144707 | [-0.037439, +0.004749] | [-0.062058, -0.044832] |
| `noncontinuous_persistence_2y` | `geofm_information_not_supported` | +0.004532 | -0.006331 | -0.060283 | [-0.009875, +0.018964] | [-0.012482, -0.000094] |

Positive Brier and ECE deltas mean lower error for the residual model. The
one-year pooled comparison passed all three practical checks, the Brier
bootstrap check, and all three strict controls. Its AP bootstrap interval still
crossed zero.

The one-year result did not survive the predeclared generalization gate:

- Bishan-to-Dongxing AP improved by `+0.026444`, but Brier worsened by
  `-0.027786`;
- Dongxing-to-Bishan AP worsened by `-0.009069`;
- aggregated Bishan spatial-fold AP/Brier deltas were `-0.016509/-0.066282`;
- aggregated Dongxing spatial-fold AP/Brier deltas were
  `-0.033181/-0.033151`.

Both two-year endpoints failed pooled practical/statistical/control gates,
both transfer directions, and spatial stability.

## Scientific Decision

The explicit residual criterion is now evaluated, but its result is mixed, not
positive. There is evidence of a pooled one-year conversion signal beyond the
explicit-history baseline and beyond the three matched controls. That signal
is not spatially stable and does not transfer reliably between Bishan and
Dongxing; it also disappears or reverses at both two-year endpoints.

Therefore the original broad claim remains unsupported: the available evidence
does not establish a robust, transferable, future-stability advantage from
GeoFM. The result does justify retaining a narrower scientific observation:
GeoFM may contain endpoint- and region-dependent short-horizon residual
information that requires independent replication before it can support a
prediction or planning claim.

The remaining unresolved exhaustion criteria are a second independent annual
product, a temporal neural model, label disagreement/noise sensitivity, and
constrained planning outcomes. Do not enter Phase 72C or modify
`paper/submission/final/*` from this result.

## Verification

```text
focused explicit-residual, two-year, and exhaustion tests: 27 passed
full repository: 580 passed, 84 existing sklearn warnings
smoke check: passed
git diff --check: passed
paper/submission/final/*: unchanged
```

## Local Artifacts

Ignored real-run outputs are under:

```text
experiments/phase72_explicit_residual_screen/outputs/prepared
experiments/phase72_explicit_residual_screen/outputs/frozen_v2
experiments/phase72_explicit_residual_screen/outputs/confirmation_v2
experiments/phase72_exhaustion_analysis/outputs/real_bishan_dongxing_residual
```

## Claim Boundary

This is a Phase 72 exhaustion experiment. It does not establish agronomic
suitability, a transferable prediction advantage, a planning benefit, or a
reason to revise the formal submission.
