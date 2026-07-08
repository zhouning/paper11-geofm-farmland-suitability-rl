# Phase 61 D6 GeoFM Projection Controls Design

## Purpose

Phase 61 is the next algorithm-and-experiment step after Phase 60 narrowed the
mechanism claim. It designs D6-style GeoFM projection controls that separate
GeoFM-derived low-dimensional information from generic low-dimensional
optimization convenience before any new manuscript revision.

Phase 61 is deliberately staged. The first deliverable is read-only with
respect to PPO: build trainable D6 feature tables and audit their representation
geometry. A later training phase can consume those tables if the controls pass
basic geometry and lineage checks.

## Scientific Question

Can we construct same-dimension controls that are derived from raw B1 GeoFM
embeddings rather than from random noise or row shuffling, so that a later policy
run can test whether the compressed-route advantage is GeoFM-information-specific
rather than only a low-dimensional optimization effect?

Phase 59 showed that D4P8/D4P16 do not clearly outperform same-dimension random
or shuffled controls. Phase 61 therefore should not assume D4 is special. It
should create D6 controls that preserve explicit planning features, preserve row
alignment, and expose low-dimensional raw-B1 projections suitable for the same
Phase 52 five-tile, three-seed base-reward protocol.

## Proposed Controls

Phase 61 creates four D6 variants:

- `D6R8`: explicit planning features plus an 8-dimensional seeded Gaussian
  random orthonormal projection of centered raw B1 GeoFM features.
- `D6R16`: explicit planning features plus a 16-dimensional seeded Gaussian
  random orthonormal projection of centered raw B1 GeoFM features.
- `D6P8`: explicit planning features plus the top-8 PCA projection recomputed
  directly from raw B1 GeoFM features.
- `D6P16`: explicit planning features plus the top-16 PCA projection recomputed
  directly from raw B1 GeoFM features.

`D6P8` and `D6P16` are expected to be close to existing D4P8/D4P16 if the D4
feature tables are exactly raw-B1 PCA coordinates. Phase 61 must quantify that
relationship rather than assume it. `D6R8` and `D6R16` are GeoFM-derived random
projection controls: they keep raw-B1 row information and low-dimensional
linear mixing, but not the PCA variance ordering.

## Inputs

Required real inputs:

- `experiments/phase11_bishan_dltb_real/outputs/phase2_real/variant_B0_features.csv`
- `experiments/phase11_bishan_dltb_real/outputs/phase2_real/variant_B1_features.csv`
- `experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P8_features.csv`
- `experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P16_features.csv`

The implementation should accept paths for all four files so fixture tests and
future reruns can use alternative inputs.

## Outputs

Feature-table output directory:

- `experiments/phase61_d6_geofm_projection_controls/outputs/phase52_full5_seed3/d6_projection_features/`

Expected generated artifacts:

- `variant_D6R8_features.csv`
- `variant_D6R16_features.csv`
- `variant_D6P8_features.csv`
- `variant_D6P16_features.csv`
- `experiment_variants.json`
- `phase61_d6_projection_feature_summary.json`

Geometry-audit output directory:

- `experiments/phase61_d6_geofm_projection_controls/outputs/phase52_full5_seed3/`

Expected audit artifacts:

- `phase61_d6_projection_geometry.json`
- `phase61_d6_projection_geometry.csv`
- `phase61_d6_projection_similarity.csv`
- `phase61_d6_projection_controls.md`

Generated output directories are ignored by Git. The implementation, tests, and
paper result note are versioned.

## Geometry and Similarity Audit

Phase 61 reports these checks:

1. Row lineage: B0, B1, D4P8, and D4P16 must have aligned `block_id` order.
2. Feature shape: D6R8 and D6P8 must have 8 projection columns; D6R16 and D6P16
   must have 16 projection columns.
3. Explicit feature preservation: every D6 table must copy the explicit planning
   columns from B0 exactly within numeric tolerance.
4. Projection variance: D6 projection columns must have nonzero centered
   variance.
5. PCA retention: D6P8 and D6P16 should report retained raw-B1 variance ratios.
6. D4 similarity: D6P8 versus D4P8 and D6P16 versus D4P16 should report
   subspace-aligned correlation or normalized reconstruction similarity.
7. D6R/D6P separation: random-projection controls should be reported separately
   from PCA-projection controls; no performance claim is made from geometry
   alone.

## Status Rules

Phase 61 reports `d6_projection_controls_ready_for_training` when all row
lineage, explicit preservation, dimension, and nonzero-variance checks pass.

It reports `d6_projection_controls_partial` when feature tables are generated
but one of the similarity or retention diagnostics is weak while row lineage and
basic feature-table validity still pass.

It reports `d6_projection_controls_blocked` when row alignment fails, required
columns are missing, projection dimensions are invalid, or any generated D6
feature table is not trainable under the existing manifest convention.

## Claim Boundary

Phase 61 does not train PPO policies, does not compare learned rewards, does not
enable suitability reward, does not test B2/B3, does not test transfer, does
not prove PCA optimality, and does not validate independent agronomic
suitability. It prepares and audits D6 controls for a later matched training
experiment.

## Testing Requirements

Unit tests must cover:

- deterministic D6 control generation for small synthetic aligned rows;
- row-alignment failure when B0/B1/D4 rows differ;
- manifest and CSV writing for all four D6 variants;
- geometry status `d6_projection_controls_ready_for_training` for valid fixture
  controls;
- CLI build-and-audit execution against temporary fixture CSVs.

Real-run verification should include:

- targeted Phase 61 pytest;
- Phase 59 and Phase 60 regression tests;
- `python scripts\smoke_check.py`;
- `git diff --check`.