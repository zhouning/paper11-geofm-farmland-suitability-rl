# Phase 15 Tiled Batch Smoke Design

## Goal

Build a batch smoke runner that executes the Phase 14 one-step tile-level input contract across every tile in a Phase 13 tile index. This verifies that the full real Bishan tiled partition can be consumed tile by tile with a representation-only variant.

## Problem

Phase 14 proves the largest tile can be loaded and stepped once. A reviewer-facing tiled workflow should also prove that all real tiles are valid, not only a hand-picked maximum tile. Phase 15 provides that batch check while keeping the same claim boundary: no training, no policy evaluation, no suitability reward.

## Inputs

Phase 15 consumes:

- Phase 2 output directory with ready variant CSVs;
- Phase 13 `phase13_tile_index.csv`;
- variant ID, default `B1`;
- optional maximum number of tiles for quick smoke checks.

## Outputs

The runner writes:

- `phase15_tiled_batch_smoke_summary.csv`;
- `phase15_tiled_batch_smoke_report.json`.

The CSV contains one row per tile:

- `tile_id`;
- `variant_id`;
- `n_blocks`;
- `n_features`;
- `observation_shape`;
- `action_space_n`;
- `selected_block_id`;
- `step_reward`;
- `reward_mode`;
- `status`.

The JSON report contains aggregate counts, min/max/mean rows per tile, max observation shape, all-pass flag, recommendation, and claim boundary.

## Behavior

Phase 15 loads the requested Phase 2 variant once, reads all tile IDs and block IDs from the Phase 13 tile index, then iterates over tiles. For each tile it builds a tile subset, wraps it in the existing Phase 4 environment contract, resets, takes the first valid action, and records the result.

Reward variants with `base_plus_suitability_reward` are rejected by default, matching Phase 14 and the Phase 10 gate. The default variant is `B1`.

## Claim Boundary

Phase 15 is a batch input-contract smoke check over real tiled artifacts. It does not train, tune, evaluate, or compare a DRL policy. It does not enable suitability reward. It does not report planning performance.

## Implementation Units

- `src/paper11_geofm/tiled_batch_smoke.py`: tile index reader, batch smoke runner, CSV/JSON writer.
- `experiments/phase15_tiled_batch_smoke/run_phase15_tiled_batch_smoke.py`: CLI runner.
- `tests/test_phase15_tiled_batch_smoke.py`: synthetic tests for all-tile batch summaries, max-tile cap, reward-variant rejection, writer, and CLI.
- Documentation updates in `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`, and `reproducibility/FILE_MANIFEST.tsv`.

## Real Bishan Expected Result

For the current real Bishan Phase 13 tile index with B1:

- tiles processed: `54`;
- total blocks: `64984`;
- minimum blocks per tile: `4`;
- maximum blocks per tile: `2234`;
- maximum observation shape: `180957`;
- all tile smoke checks pass;
- every step reward is `0.0` because B1 uses `base_planning_reward`.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: Phase 15 uses Phase 13 tile metadata and Phase 14 environment-contract behavior.
- Scope check: this phase is limited to batch smoke checks and does not train.
- Ambiguity check: default variant, reward-mode rejection, output artifacts, and claim boundary are explicit.
