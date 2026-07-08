# Phase 60 Information-vs-Optimization Attribution Design

## Purpose

Phase 60 is a read-only attribution audit for the current Paper11 compressed
state-route evidence. It reconciles the positive Phase 52/53 compressed-route
replication, the Phase 57 geometry audit, and the negative Phase 59
matched-dimension control result before any further manuscript revision.

The phase remains algorithm-and-experiment first. It does not revise the formal
manuscript and does not add new policy training by default.

## Scientific Question

Do the current Paper11 experiments support a GeoFM-specific information claim,
or only a narrower low-dimensional optimization/representation claim?

Phase 60 should answer this with a claim-boundary matrix rather than a single
positive/negative label. The expected current interpretation is that
D4P8/D4P16 remain supported compressed state routes against B0, raw B1, random
D2, and shuffled D3, while Phase 59 does not support a stronger claim that the
current PCA-compressed GeoFM coordinates outperform same-dimension random or
shuffled controls.

## Inputs

Phase 60 uses existing artifacts only:

- `experiments/phase52_expanded_cluster_replication/outputs/phase52_full5_seed3_phase48_compressed_rescue/phase48_compressed_geofm_rescue_comparison.json`
- `experiments/phase53_cluster_mean_support/outputs/phase52_full5_seed3/phase53_cluster_mean_support.json`
- `experiments/phase57_compressed_representation_mechanism/outputs/phase52_full5_seed3/phase57_compressed_representation_mechanism.json`
- `experiments/phase59_matched_dimension_controls/outputs/phase52_full5_seed3/phase59_matched_dimension_controls.json`

The runner should accept paths for these files so future reruns can use updated
artifacts.

## Attribution Axes

Phase 60 reports four explicit axes.

### 1. Compressed-route performance

Evidence source: Phase 48 analysis over the expanded Phase 52 five-tile,
three-seed summary.

Supported when:

- `phase48_compressed_geofm_status == "compressed_geofm_route_supported"`;
- pooled compressed-control mean delta is positive;
- pooled positive fraction is at least `0.5`.

This axis supports only the statement that D4P8/D4P16 outperform B0, raw B1,
random D2, and shuffled D3 under the existing base-reward held-out protocol.

### 2. Cluster-level robustness

Evidence source: Phase 53 cluster mean support JSON.

Supported when:

- `phase53_cluster_mean_status == "cluster_mean_support"`;
- cluster mean delta is positive;
- exact sign-flip or equivalent cluster-mean p-value is below `0.05` when
  present.

This axis supports the statement that the expanded compressed-route mean is not
only a row-level artifact.

### 3. Compressed geometry consistency

Evidence source: Phase 57 mechanism JSON.

Supported when:

- `phase57_mechanism_status == "compressed_geometry_consistent"`;
- D4P8 and D4P16 have lower effective rank than raw B1;
- D4P8 and D4P16 retain nonzero raw GeoFM variance;
- reward-gain rows for both compressed variants have positive mean deltas.

This axis supports the statement that the compressed route is geometrically
consistent with lower-rank, better-conditioned state representations.

### 4. GeoFM-specific matched-dimension advantage

Evidence source: Phase 59 matched-dimension controls JSON.

Supported only when:

- `phase59_matched_dimension_status == "matched_dimension_geofm_supported"`;
- all four matched comparisons have positive mean deltas;
- pooled matched-control mean delta is positive;
- pooled positive fraction is at least `0.5`;
- coverage issues are empty after ignoring historical non-Phase-59 variants.

When this axis is not supported, Phase 60 must explicitly narrow the mechanism
wording away from a GeoFM-specific same-dimension advantage.

## Status Rule

Phase 60 reports `mechanism_claim_narrowed` when:

- compressed-route performance is supported;
- cluster-level robustness is supported;
- compressed geometry consistency is supported;
- GeoFM-specific matched-dimension advantage is not supported.

It reports `geofm_specific_information_supported` only when all four axes are
supported.

It reports `low_dimensional_route_uncertain` when any of the first three axes
is unsupported or insufficient, regardless of Phase 59.

It reports `insufficient` when any required input artifact is missing critical
fields, has missing coverage, or cannot be interpreted consistently.

## Output Artifacts

Expected implementation and evidence files:

- `src/paper11_geofm/phase60_information_optimization_attribution.py`
- `experiments/phase60_information_optimization_attribution/run_phase60_information_optimization_attribution.py`
- `tests/test_phase60_information_optimization_attribution.py`
- `paper/phase28_results/26_phase60_information_optimization_attribution.md`
- `experiments/phase60_information_optimization_attribution/outputs/phase52_full5_seed3/phase60_information_optimization_attribution.json`
- `experiments/phase60_information_optimization_attribution/outputs/phase52_full5_seed3/phase60_attribution_axes.csv`
- `experiments/phase60_information_optimization_attribution/outputs/phase52_full5_seed3/phase60_information_optimization_attribution.md`

The JSON should include:

- `phase60_attribution_status`;
- one row per attribution axis;
- source artifact paths;
- core numeric evidence copied from the source artifacts;
- claim-boundary recommendations;
- optional next-experiment recommendation.

## Optional Phase 60B Follow-up

Phase 60 should reserve wording for a later Phase 60B or Phase 61 but should
not implement it by default. If the attribution audit determines that stronger
evidence is needed, the next experiment should add new same-dimension GeoFM
controls, such as random projections or orthogonal projections from raw B1,
then train them under the same Phase 52 five-tile, three-seed base-reward
protocol.

That follow-up would test whether any GeoFM-derived low-dimensional projection
beats the current random/shuffled same-dimension controls. It is not required
for Phase 60 read-only attribution.

## Claim Boundary

Phase 60 does not prove PCA optimality, does not enable B2/B3, does not add a
suitability reward, does not test transfer, and does not validate independent
agronomic suitability. It is a claim-boundary audit for algorithm and
experiment interpretation before manuscript revision.

## Verification

Unit tests should cover:

- the expected `mechanism_claim_narrowed` status from synthetic supported
  Phase 48/53/57 evidence plus not-supported Phase 59 evidence;
- the stronger `geofm_specific_information_supported` status when Phase 59 is
  supported;
- `low_dimensional_route_uncertain` when any of Phase 48/53/57 is unsupported;
- `insufficient` for missing critical fields;
- artifact writing for JSON, CSV, and Markdown outputs;
- CLI parsing and required input validation.

Real-run verification should include:

- Phase 60 targeted pytest;
- Phase 48, Phase 57, and Phase 59 regression tests;
- `python scripts\smoke_check.py`;
- `git diff --check`.
