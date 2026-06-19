# Phase 29 Representation-Scale Diagnosis

## One-Sentence Argument

Phase 29 supports a conservative optimization-difficulty hypothesis for the
current raw B1 design: the 64-dimensional B1 embedding matrix is highly
redundant and has smaller per-dimension scale than the PCA-compressed controls,
but this read-only evidence does not prove that PCA or normalization would
improve PPO performance.

## Terminology Ledger

| Canonical term | Definition | Boundary |
|---|---|---|
| Phase 29 | read-only diagnosis of representation scale, row norms, and PCA concentration | no new training, no reward change |
| raw B1 | explicit planning features plus raw 64-dimensional GeoFM embeddings | current B1 state design only |
| column z-score | column-centered and standard-deviation-scaled B1 embedding matrix | diagnostic transform only |
| row L2 profile | row-normalized B1 embedding matrix | not a trained-policy result |
| optimization-difficulty hypothesis | the current raw representation may be less convenient for PPO than compressed controls | not a causal proof |

## What Was Used

This read-only diagnosis used:

```text
experiments/phase11_bishan_dltb_real/outputs/phase2_real/variant_B1_features.csv
experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P8_features.csv
experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P16_features.csv
experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv
experiments/phase28_representation_controls/outputs/real_bishan_4096/phase28_representation_control_summary.csv
```

Reproduce the diagnosis with:

```text
python experiments/phase29_representation_scale_diagnosis/run_phase29_representation_scale_diagnosis.py --phase2-b1-features-csv experiments/phase11_bishan_dltb_real/outputs/phase2_real/variant_B1_features.csv --d4p8-features-csv experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P8_features.csv --d4p16-features-csv experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P16_features.csv --tile-index-csv experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv --phase28-summary-csv experiments/phase28_representation_controls/outputs/real_bishan_4096/phase28_representation_control_summary.csv --output-dir experiments/phase29_representation_scale_diagnosis/outputs/real_bishan_4096
```

The generated artifacts are:

```text
phase29_variant_scale_summary.csv
phase29_tile_scale_summary.csv
phase29_b1_normalization_profiles.csv
phase29_representation_scale_diagnosis.json
phase29_representation_scale_diagnosis.md
```

## Global Scale Result

| Variant | Dimensions | Mean row L2 | Row-L2 std | Mean column std |
|---|---:|---:|---:|---:|
| B1 | `64` | `1.0000582045` | `0.0020220234` | `0.0377917339` |
| D4P8 | `8` | `0.2551868696` | `0.1384478335` | `0.0912456150` |
| D4P16 | `16` | `0.2723406399` | `0.1379579026` | `0.0620769890` |

Raw B1 rows are already close to unit length, but the per-dimension variation
is small. D4P8 and D4P16 expose fewer dimensions with larger per-dimension
standard deviations. This is consistent with a compressed state arm that may
present stronger per-coordinate gradients to the current policy network.

## Normalization Profiles

| Profile | Mean row L2 | Row-L2 std | Mean column std |
|---|---:|---:|---:|
| raw | `1.0000582045` | `0.0020220234` | `0.0377917339` |
| column z-score | `7.2744760531` | `3.3289635254` | `1.0000000000` |
| row L2 | `1.0000000000` | `0.0000000000` | `0.0377890144` |
| column z-score + row L2 | `1.0000000000` | `0.0000000000` | `0.1231352816` |

This result is useful because it separates two issues. The raw B1 row norms are
not unstable, so the observed Phase 28 result is unlikely to be explained by
large raw row-norm variation. However, column-level standardization reveals
substantial block-to-block variation that is hidden by the small raw
per-coordinate scale. A normalized B1 ablation is therefore a plausible next
test, but Phase 29 does not show whether such a policy would perform better.

## PCA Concentration Result

| Diagnostic | Value |
|---|---:|
| numerical rank above `1e-12` | `64` |
| effective rank | `5.2467650861` |
| top-8 PCA variance ratio | `0.8587823898` |
| top-16 PCA variance ratio | `0.9496006154` |
| mean raw embedding std | `0.0377917339` |
| mean D4P8 component std | `0.0912456150` |
| mean D4P16 component std | `0.0620769890` |

The effective rank is much lower than the nominal 64 dimensions. The first 8
principal components retain about `85.9%` of the observed variance, and the
first 16 retain about `95.0%`. This supports the narrower conclusion that the
current B1 state contains heavy redundancy under the Bishan feature table.

## Interpretation

Phase 29 strengthens the Phase 28 compression diagnosis by identifying a
specific, testable mechanism: the current raw B1 input may be a poorly scaled
and redundant representation for the small padded MaskablePPO setup. The
compressed controls are not necessarily more semantically meaningful, but they
expose concentrated low-dimensional variation with larger coordinate scale.

The current status is:

```text
raw_b1_scale_may_affect_optimization
```

This should be read as a diagnostic hypothesis, not a positive model result.

## What This Supports

This diagnosis supports three conservative claims:

1. B1 raw embeddings are highly compressible in the real Bishan feature table.
2. D4P8/D4P16 expose larger per-coordinate variation than raw B1.
3. A normalized or compressed B1 follow-up is justified before making any
   positive GeoFM learned-policy claim.

## What This Does Not Support Yet

This diagnosis does not support:

- the claim that PCA is intrinsically superior to raw GeoFM semantics;
- the claim that row or column normalization will improve PPO;
- a positive B1-over-B0 planning-performance claim;
- suitability reward readiness, B2/B3 superiority, or transfer conclusions.

## Recommended Next Step

The next rigorous experiment should be a bounded normalization ablation, not a
claim expansion. A useful design would compare raw B1 against a column-centered
and scaled B1 variant and a column-centered plus row-normalized B1 variant
under the exact Phase 28 held-out protocol, while retaining B0, D2, D3, D4P8,
and D4P16 controls.

## Claim Boundary

Phase 29 is a read-only feature-table diagnosis. It does not run new policy
training, does not alter rewards, does not test B2/B3, does not prove that PCA
is intrinsically superior, and does not prove that normalization would improve
PPO performance.
