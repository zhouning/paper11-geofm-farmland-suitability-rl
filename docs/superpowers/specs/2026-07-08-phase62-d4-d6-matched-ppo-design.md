# Phase 62 D4/D6 Matched PPO Evaluation Design

## Purpose

Phase 62 is the bounded training follow-up to Phase 61. Phase 61 prepared D6
raw-B1 projection controls and showed that D6P8/D6P16 reproduce the existing
D4P8/D4P16 PCA controls, while D6R8/D6R16 provide distinct raw-B1 random
orthonormal projection controls. Phase 62 now tests whether the D4 PCA-compressed
route outperforms those GeoFM-derived random projection controls under the same
base-reward held-out PPO protocol used for the expanded Phase 52 evidence.

Phase 62 remains algorithm-and-experiment work. It does not revise the formal
manuscript by default.

## Scientific Question

When dimensionality and raw-B1 provenance are both controlled, do D4P8/D4P16
PCA projections outperform D6R8/D6R16 raw-B1 random orthonormal projections?

A positive Phase 62 result would support a narrower PCA-ordering mechanism for
D4 over raw-B1 random projection controls. A negative or mixed result would
keep the Phase 60 conclusion narrowed: D4 remains a low-dimensional compressed
state route, but not a proven GeoFM-specific or PCA-specific mechanism.

## Default Scope

The default real run should use the Phase 52 held-out protocol:

- train tile: selected by the existing tile-selection helper unless explicitly
  supplied;
- eval tiles: `tile_r002_c003,tile_r005_c004,tile_r005_c003,tile_r000_c004,tile_r001_c004`;
- seeds: `0,1,2`;
- total timesteps: `4096`;
- eval max steps: `8`;
- primary variants: `D4P8,D4P16,D6R8,D6R16`.

`D6P8,D6P16` can be included as optional lineage sanity variants, but the
primary D4-vs-D6R status should not require them. Because D6P8/D6P16 are column
correlation `1.0` reproductions of D4P8/D4P16, they are not independent
mechanism controls.

## Inputs

Required real inputs:

- Phase 8/D4 feature directory:
  `experiments/phase8_ablation_controls/outputs/real_bishan_controls`
- Phase 61/D6 feature directory:
  `experiments/phase61_d6_geofm_projection_controls/outputs/phase52_full5_seed3`
- tile index:
  `experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv`

The runner must support two modes:

- `run-and-analyze`: train/evaluate requested variants and write outputs;
- `analyze-only`: analyze an existing summary CSV without retraining.

`run-and-analyze` should support an optional `--existing-summary-csv` so a
partial run can be merged with new variant rows, following the Phase 59 pattern.

## Outputs

Output directory:

- `experiments/phase62_d4_d6_matched_ppo/outputs/phase52_full5_seed3/`

Expected generated artifacts:

- `phase62_d4_d6_matched_ppo_summary.csv`
- `phase62_d4_d6_matched_ppo_traces.json`
- `phase62_d4_d6_delta_table.csv`
- `phase62_d4_d6_cluster_summary.csv`
- `phase62_d4_d6_matched_ppo.json`
- `phase62_d4_d6_matched_ppo.md`

Generated output directories are ignored by Git. The implementation, tests, and
paper result note are versioned.

## Comparisons

Primary comparisons:

- `D4P8 - D6R8`
- `D4P16 - D6R16`

Optional lineage comparisons when D6P variants are present:

- `D4P8 - D6P8`
- `D4P16 - D6P16`

Phase 62 should report:

- mean delta for each comparison;
- positive tile-seed count and fraction;
- pooled D4-D6R mean delta;
- pooled D4-D6R positive count and fraction;
- cluster-level mean deltas by eval tile and seed;
- one-sided sign-test and signed-rank summaries where applicable;
- coverage issues for missing, duplicate, or unexpected trained-policy rows.

## Status Rules

Report `d4_pca_advantage_over_d6_supported` when:

- no coverage issues exist for primary variants;
- both primary comparison mean deltas are positive;
- pooled primary mean delta is positive;
- pooled primary positive fraction is at least `0.5`.

Report `d6_random_projection_advantage` when:

- no coverage issues exist for primary variants;
- both primary comparison mean deltas are negative;
- pooled primary mean delta is negative.

Report `d4_d6_not_distinguishable` when coverage is complete but neither of the
above rules holds.

Report `insufficient` when required primary variant rows are missing, duplicated,
unexpected, or cannot be interpreted.

## Claim Boundary

Phase 62 tests base-reward learned-policy differences between D4 PCA compressed
states and D6 raw-B1 random projection controls. It does not enable suitability
reward, does not test B2/B3, does not test transfer, does not prove independent
agronomic suitability, and does not by itself justify final submission-level
performance claims. It should not modify formal manuscript files.

## Testing Requirements

Unit tests must cover:

- contract routing for D4 variants to the Phase 8 directory and D6 variants to
  the Phase 61 directory;
- analysis status `d4_pca_advantage_over_d6_supported` for synthetic complete
  rows where D4 beats D6R;
- analysis status `d6_random_projection_advantage` for synthetic complete rows
  where D6R beats D4;
- analysis status `d4_d6_not_distinguishable` for mixed rows;
- `insufficient` for missing primary variant rows;
- writer outputs for summary, traces, JSON, delta CSV, cluster CSV, and Markdown;
- CLI analyze-only behavior against a temporary summary CSV.

Real-run verification should include:

- targeted Phase 62 pytest;
- Phase 61 and Phase 59 regression tests;
- `python scripts\smoke_check.py`;
- `git diff --check`.