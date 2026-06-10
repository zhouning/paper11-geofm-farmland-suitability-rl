# Phase 13 Tiled Real Contract Design

## Goal

Build an executable tiled contract builder for the real Bishan DLTB workflow. Phase 13 converts the Phase 11 block-to-pixel mapping and Phase 2 variant manifest into tile-level episode metadata so later DRL work can avoid the full flat 64,984-block action space.

## Problem

Phase 12 shows that the real flat B3 observation dimension is 5,328,691. That makes direct flat full-scale training an unsuitable next claim. However, the same real DLTB data can be partitioned into spatial tiles based on AlphaEarth grid rows and columns.

With an 8 x 8 AlphaEarth-grid tile size on the current Bishan mapping, the local real run has 54 non-empty tiles. The largest tile has 2,234 DLTB blocks. For B3 with 82 features, the largest tiled observation dimension is `2234 * 82 + 3 = 183191`, far below the Phase 12 flat threshold of 1,000,000.

## Inputs

Phase 13 consumes:

- Phase 11 `block_pixel_mapping.csv` with `block_id,row,col,weight`;
- Phase 2 `experiment_variants.json` with B0/B1/B2/B3 readiness and feature-column counts.

It does not read the large DLTB GeoPackage and does not read the full variant feature CSVs.

## Outputs

The runner writes:

- `phase13_tile_index.csv`;
- `phase13_tiled_real_contract.json`.

The tile index CSV contains:

- `tile_id`;
- `tile_row`;
- `tile_col`;
- `n_blocks`;
- `min_grid_row`;
- `max_grid_row`;
- `min_grid_col`;
- `max_grid_col`;
- `block_ids`.

The JSON report contains:

- mapping and manifest paths;
- requested tile row/column size;
- total blocks;
- tile count;
- min/max/mean blocks per tile;
- per-variant feature counts and maximum tiled observation dimension;
- `all_tiles_within_observation_threshold`;
- `tiled_contract_ready`;
- recommendation and claim boundary.

## Decision Rules

`tiled_contract_ready` is true when:

- the mapping contains at least one block;
- every mapping row has `block_id`, `row`, and `col`;
- all requested variants are ready in the Phase 2 manifest;
- every non-empty tile has a maximum observation dimension at or below the configured threshold.

Default tile size is 8 rows x 8 columns. Default observation threshold is 1,000,000 float32 values, matching Phase 12.

Variant observation dimension for a tile is:

```text
n_blocks_in_tile * n_features_for_variant + 3
```

## Claim Boundary

Phase 13 builds tile-level contract metadata only. It does not train, tune, evaluate, or compare a DRL policy. It does not enable suitability reward. It does not prove planning performance. It only demonstrates that real Bishan DLTB blocks can be partitioned into tractable spatial episodes for later tiled or hierarchical environment design.

## Implementation Units

- `src/paper11_geofm/tiled_contract.py`: pure CSV/JSON reader, tile builder, contract summary, and artifact writer.
- `experiments/phase13_tiled_real_contract/run_phase13_tiled_real_contract.py`: CLI runner.
- `tests/test_phase13_tiled_contract.py`: synthetic mapping and manifest tests for tiling, threshold decisions, writer output, and CLI output.
- Documentation updates in `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`, and `reproducibility/FILE_MANIFEST.tsv`.

## Real Bishan Expected Result

For the current local Bishan mapping with 8 x 8 tiles:

- total blocks: `64984`;
- non-empty tiles: `54`;
- minimum blocks per tile: `4`;
- maximum blocks per tile: `2234`;
- mean blocks per tile: about `1203.407`;
- B3 maximum tiled observation dimension: `183191`;
- `all_tiles_within_observation_threshold = true`;
- `tiled_contract_ready = true`.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: all computations use Phase 11 mapping and Phase 2 manifest fields that already exist.
- Scope check: the phase is limited to tile contract metadata and does not add training.
- Ambiguity check: tile size, threshold, observation formula, outputs, and claim boundary are explicit.
