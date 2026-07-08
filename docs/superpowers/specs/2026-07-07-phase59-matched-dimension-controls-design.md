# Phase 59 Matched-Dimension Control Audit Design

## Purpose

Phase 59 adds a matched-dimension control audit for the current positive
Paper11 conclusion. The audit tests whether the D4P8 and D4P16 compressed
GeoFM gains are attributable to GeoFM-derived low-dimensional signal rather
than only to lower-dimensional and better-conditioned policy inputs.

The phase remains algorithm-and-experiment first. It does not revise the
manuscript until the new evidence is generated and interpreted.

## Scientific Question

Do PCA-compressed GeoFM state routes outperform control features with the same
8- or 16-dimensional state budget under the same Bishan base-reward held-out
policy protocol?

The answer determines whether the current mechanism wording can remain a
GeoFM-compression claim or must be narrowed to a generic low-dimensional
representation claim.

## Inputs

- `experiments/phase11_bishan_dltb_real/outputs/phase2_real/variant_B0_features.csv`
- `experiments/phase11_bishan_dltb_real/outputs/phase2_real/variant_B1_features.csv`
- `experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P8_features.csv`
- `experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P16_features.csv`
- `experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv`
- The Phase 52 expanded protocol settings: five held-out Bishan tiles, seeds
  `0,1,2`, `4096` training timesteps, and `8` evaluation steps.

## Matched-Dimension Variants

Phase 59 creates new explicit-plus-control feature tables that are dimension
matched to D4P8 and D4P16:

- `D5R8`: explicit planning features plus 8-dimensional random controls whose
  columns match the D4P8 column means and standard deviations.
- `D5R16`: explicit planning features plus 16-dimensional random controls whose
  columns match the D4P16 column means and standard deviations.
- `D5S8`: explicit planning features plus block-shuffled D4P8 PCA scores. This
  preserves the D4P8 column distribution and cross-column geometry but breaks
  block-to-feature alignment.
- `D5S16`: explicit planning features plus block-shuffled D4P16 PCA scores.
  This preserves the D4P16 column distribution and cross-column geometry but
  breaks block-to-feature alignment.

The existing supported candidates remain:

- `D4P8`: explicit planning features plus 8 PCA-compressed GeoFM components.
- `D4P16`: explicit planning features plus 16 PCA-compressed GeoFM components.

## Method

The implementation should be a new Phase 59 module and runner rather than a
mutation of Phase 28 constants. It may reuse Phase 28 low-level tiled loading,
padded policy environment, MaskablePPO training, evaluation, trace writing, and
summary helpers where that keeps behavior aligned.

The phase has two steps:

1. Build matched-dimension control feature tables and an
   `experiment_variants.json` manifest for `D5R8`, `D5R16`, `D5S8`, and
   `D5S16`.
2. Run the expanded held-out training/evaluation protocol for
   `D4P8,D4P16,D5R8,D5S8,D5R16,D5S16`, then analyze matched comparisons:
   `D4P8 - D5R8`, `D4P8 - D5S8`, `D4P16 - D5R16`, and `D4P16 - D5S16`.

The analysis should report:

- mean reward by variant;
- matched row-level deltas and positive counts;
- pooled matched-control delta for all four comparisons;
- bootstrap confidence interval for the pooled matched-control delta;
- tile-seed cluster means;
- sign-only and magnitude-sensitive cluster support where feasible with 15
  clusters;
- coverage issues for missing, duplicate, or unexpected tile-seed-variant rows.

## Status Rule

The audit reports `matched_dimension_geofm_supported` only when all required
tile-seed-variant rows are present, all four matched comparison mean deltas are
positive, the pooled matched-control mean delta is positive, and the pooled
matched-control positive fraction is at least `0.5`.

It reports `matched_dimension_geofm_partial` when at least one compressed route
beats both of its dimension-matched controls but the full rule is not met.

It reports `matched_dimension_geofm_not_supported` when neither compressed
route beats both of its matched controls.

It reports `insufficient` if required rows are missing, duplicated, or otherwise
not comparable.

## Claim Boundary

Phase 59 can support or weaken the interpretation that D4P8/D4P16 work because
they preserve GeoFM-derived signal in a compact state representation. It does
not prove that PCA is optimal, does not introduce a suitability reward, does
not enable B2/B3, does not test cross-region transfer, and does not validate
independent agronomic suitability.

No manuscript wording should be updated until the real Phase 59 run and
verification are complete.

## Outputs

Expected implementation and evidence files:

- `src/paper11_geofm/phase59_matched_dimension_controls.py`
- `experiments/phase59_matched_dimension_controls/run_phase59_matched_dimension_controls.py`
- `tests/test_phase59_matched_dimension_controls.py`
- `paper/phase28_results/25_phase59_matched_dimension_controls.md`
- `experiments/phase59_matched_dimension_controls/outputs/phase52_full5_seed3/phase59_matched_dimension_control_summary.csv`
- `experiments/phase59_matched_dimension_controls/outputs/phase52_full5_seed3/phase59_matched_dimension_delta_table.csv`
- `experiments/phase59_matched_dimension_controls/outputs/phase52_full5_seed3/phase59_matched_dimension_controls.json`
- `experiments/phase59_matched_dimension_controls/outputs/phase52_full5_seed3/phase59_matched_dimension_controls.md`

## Verification

Unit tests should cover:

- deterministic construction of random and shuffled matched-dimension controls;
- exact block alignment between D4P and D5 feature tables;
- manifest compatibility with tiled variant loading;
- coverage issue detection;
- status assignment for supported, partial, not-supported, and insufficient
  synthetic cases;
- artifact writing for CSV, JSON, and Markdown outputs.

Real-run verification should include:

- the Phase 59 targeted pytest file;
- relevant Phase 28/48 regression tests if Phase 59 reuses their helpers;
- `python scripts\smoke_check.py`;
- `git diff --check`.
