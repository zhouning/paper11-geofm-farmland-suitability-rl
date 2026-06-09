# Phase 2 Block GeoFM Feature Assembly Design

Saved: 2026-06-09

## Objective

Build the design for a block-level GeoFM feature assembly pipeline that replaces the Phase 1 deterministic grid regions with real planning units. The output should be a reviewer-inspectable block feature table that can feed the Paper11 B0/B1/B2/B3 experiments.

Phase 2 is not a DRL training phase. It is the feature-assembly bridge between the Phase 1 representation baseline and the later suitability-aware DRL environment.

## Scientific Role

Phase 1 proved that the repository can load the included Bishan AlphaEarth sample, aggregate 64-dimensional embeddings over deterministic regions, and export a conservative `suitability_proxy`.

Phase 2 should prove that the same representation logic can be attached to real farmland planning units:

```text
block or parcel units
  + explicit planning features
  + AlphaEarth annual embedding grids
  + optional weak labels and constraints
  -> B0/B1/B2/B3-ready block feature table
```

This phase creates the data interface needed for policy integration. It should not yet claim that GeoFM improves DRL performance.

## Scope

In scope:

- define accepted block/parcel input formats;
- define a pixel-to-block mapping contract;
- aggregate AlphaEarth 64-dimensional embeddings per block;
- compute block-level dispersion and temporal-stability diagnostics;
- compute a conservative `suitability_proxy` per block;
- join optional explicit planning features;
- export block-level CSV and JSON artifacts;
- make the output schema directly usable for B0, B1, B2, and B3 experiments;
- define validation checks for geometry coverage, embedding shape, score bounds, and missing values.

Out of scope:

- DRL policy training;
- reward implementation;
- future land-use prediction;
- scenario action modeling;
- Google Earth Engine extraction;
- direct soil, fertility, productivity, or irrigation claims;
- committing large full-region arrays to ordinary Git;
- requiring reviewers to have proprietary planning data for the lightweight smoke path.

## Inputs

The Phase 2 implementation should accept three input groups.

### Required Planning Unit Input

At least one of the following must be provided:

```text
block polygons or parcel polygons
block-to-pixel mapping table
```

For polygon input, each geometry must have a stable `block_id`. Parcel-level data may be dissolved or grouped to blocks if a `block_id` column is available.

For mapping-table input, each row should map one AlphaEarth grid pixel to one block:

```text
block_id
row
col
optional weight
```

The mapping-table mode is important for reproducibility because it lets the pipeline run without heavy GIS dependencies once the geometry overlay has already been computed.

### Required GeoFM Input

Use the AlphaEarth annual embedding grid format already present in the repository:

```text
metadata.json
bishan_emb_2017.npy
...
bishan_emb_2024.npy
```

The initial implementation should assume:

```text
embedding_dim = 64
grid_shape = metadata["grid_shape"]
years = metadata["years"]
base_year = configurable, default 2020
```

If the planning base year is unavailable, choose the nearest available embedding year and record the mismatch in `summary.json`.

### Optional Planning Features and Labels

Optional tabular inputs may include:

```text
block_id
explicit_feature_00 ... explicit_feature_16
area
slope_mean
slope_quantile
land_use_share_*
planning_mask
stable_farmland_label
high_standard_farmland_label
split
```

The implementation should not assume that all optional columns exist. It should record which groups are present and which are absent.

## Outputs

The primary Phase 2 output directory should be:

```text
experiments/phase2_block_geofm_features/outputs/
```

Required artifacts:

```text
block_geofm_features.csv
summary.json
```

Optional artifacts:

```text
mapping_diagnostics.csv
feature_missingness.csv
weak_label_validation.json
```

## Block Feature Schema

`block_geofm_features.csv` should include at least:

```text
block_id
pixel_count
pixel_weight_sum
row_min
row_max
col_min
col_max
embedding_mean_00 ... embedding_mean_63
embedding_std_mean
temporal_stability
suitability_proxy
```

When explicit planning features are available, preserve them with stable names:

```text
explicit_feature_00 ... explicit_feature_16
```

When weak labels or splits are available, preserve them as metadata columns:

```text
stable_farmland_label
high_standard_farmland_label
split
```

`summary.json` should include:

```text
metadata_source
base_year_requested
base_year_used
years
grid_shape
embedding_dim
n_blocks
mapping_mode
feature_groups_present
missing_feature_groups
block_table
suitability_min
suitability_max
suitability_mean
claim_boundary
```

The `claim_boundary` field must state that AlphaEarth embeddings and `suitability_proxy` are latent remote-sensing proxies. They do not directly measure soil quality, fertility, irrigation access, or crop productivity.

## Feature Variants for B0/B1/B2/B3

Phase 2 should not train policies, but it must make the experiment variants unambiguous.

| ID | Feature set | Output contract |
|---|---|---|
| B0 | explicit planning features only | `explicit_feature_*` plus block metadata |
| B1 | explicit planning features + AlphaEarth 64d | B0 columns plus `embedding_mean_00..63` |
| B2 | explicit planning features + suitability proxy | B0 columns plus `suitability_proxy` |
| B3 | explicit planning features + AlphaEarth 64d + suitability proxy | B0, B1, and B2 columns together |

If explicit planning features are unavailable in the lightweight sample path, the pipeline should still export GeoFM block features and mark B0-dependent variants as incomplete in `summary.json`.

## Architecture

Add a focused Phase 2 package only after this design is approved:

```text
src/paper11_geofm/
  block_mapping.py
  block_features.py
  block_schema.py
```

Expected responsibilities:

- `block_mapping.py`: validate polygon-derived or table-derived pixel mappings.
- `block_features.py`: aggregate embeddings and optional explicit features by `block_id`.
- `block_schema.py`: define required output columns and feature-group metadata.

Add an experiment entry point:

```text
experiments/phase2_block_geofm_features/run_phase2.py
```

The default lightweight command should run on a small reproducible mapping or generated mapping derived from the included Bishan sample:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py
```

The full-data command can accept external block polygons or a mapping table through CLI arguments.

## Data Flow

```text
AlphaEarth metadata and annual embeddings
  -> validate years, grid shape, and embedding dimension
  -> load or derive block-to-pixel mapping
  -> aggregate base-year 64d embeddings per block
  -> compute dispersion and temporal stability
  -> compute bounded suitability_proxy
  -> join optional explicit planning features and weak labels
  -> export block_geofm_features.csv and summary.json
```

## Mapping Rules

Each block must contain at least one mapped pixel. Blocks with zero mapped pixels should be excluded from the feature table and reported in diagnostics.

If a pixel overlaps multiple blocks, the implementation should support weighted aggregation:

```text
weighted_mean = sum(weight_i * embedding_i) / sum(weight_i)
```

If weights are absent, treat every mapping row as weight 1.

The implementation should reject mappings whose row or column values fall outside `metadata["grid_shape"]`.

## Suitability Proxy

Use the conservative Phase 1 interpretation:

```text
suitability_proxy in [0, 1]
```

The score may combine:

- similarity to a stable latent remote-sensing centroid;
- lower within-block embedding dispersion;
- temporal stability across annual embeddings;
- optional explicit constraints such as slope, if available.

The initial version should keep DEM slope as an explicit constraint and should not treat AlphaEarth channels as slope, soil, fertility, or irrigation measurements.

## Validation

Required checks:

- metadata years and embedding dimension match expectations;
- every mapping row points to a valid grid cell;
- every exported block has `pixel_count > 0`;
- all `embedding_mean_*` columns are finite;
- `suitability_proxy` is finite and bounded in `[0, 1]`;
- `block_id` is unique in the exported table;
- optional explicit feature columns are either all present or clearly marked missing;
- B0/B1/B2/B3 readiness is recorded in `summary.json`.

Optional validation when weak labels exist:

- compare `suitability_proxy` distributions for positive and negative weak labels;
- report AUC or simple rank statistics;
- summarize suitability by slope quantile;
- record that validation is unavailable when weak labels are absent.

## Testing

Implementation tests should be written before Phase 2 code. They should cover:

- mapping validation rejects out-of-range pixels;
- mapping validation aggregates repeated or weighted pixels deterministically;
- block feature table has one row per block and 64 embedding columns;
- explicit planning features are preserved when supplied;
- missing explicit planning features are reported without breaking GeoFM-only output;
- suitability scores are finite and bounded;
- experiment runner writes both required artifacts to a caller-provided output directory.

Tests must not require internet, GPU, Google Earth Engine, or full DRL dependencies.

## Acceptance Criteria

Phase 2 implementation is complete when:

- `python experiments\phase2_block_geofm_features\run_phase2.py` exits with code 0 on the lightweight sample path;
- `block_geofm_features.csv` and `summary.json` are created;
- `summary.json` reports B0/B1/B2/B3 readiness;
- `python scripts\smoke_check.py` passes;
- `python -m pytest tests` passes;
- `reproducibility/FILE_MANIFEST.tsv` lists the new Phase 2 files;
- documentation clearly states whether explicit planning features and weak labels were present.

## Risks

- Real block geometry may not align exactly with the included AlphaEarth sample grid. The mapping contract must make coverage and dropped units explicit.
- If only a synthetic or grid-derived block mapping is available, Phase 2 is still a feature-assembly smoke test, not real planning evidence.
- Without weak labels, `suitability_proxy` remains an unvalidated latent proxy.
- Without explicit planning features, B0 cannot be evaluated and DRL comparison should not start.
- Large full-region embedding arrays should stay outside ordinary Git unless Git LFS is deliberately enabled.

## Recommended Next Step

After this spec is reviewed, write an implementation plan for the Phase 2 feature assembly pipeline. The first implementation should prefer the mapping-table path because it creates a lightweight, testable bridge before introducing full GIS overlay dependencies.
