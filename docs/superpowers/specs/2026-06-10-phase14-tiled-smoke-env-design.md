# Phase 14 Tiled Smoke Environment Design

## Goal

Build an executable tile-level input-contract smoke path on top of Phase 13. Phase 14 loads one real tile's block IDs, subsets a ready Phase 2 variant feature table, wraps the subset in the existing Phase 4 smoke environment, and runs one deterministic step.

## Problem

Phase 13 proves that the real Bishan DLTB mapping can be split into tractable tiles. That still leaves one missing executable link: loading a tile-specific feature matrix and proving that the existing DRL input-contract environment can run on the tile rather than the full 64,984-block table.

The current reward gate still says suitability reward is not ready. Therefore Phase 14 should default to representation-only variants, especially B1. It must not silently run B2/B3 suitability-reward variants as if reward training were allowed.

## Inputs

Phase 14 consumes:

- Phase 2 output directory with `experiment_variants.json` and ready variant CSVs;
- Phase 13 `phase13_tile_index.csv`;
- a `tile_id`;
- a variant ID, defaulting to `B1`.

## Outputs

The runner writes one JSON artifact:

- `phase14_tiled_smoke_summary.json`.

The summary includes:

- `tile_id`;
- `variant_id`;
- tile block count;
- feature count;
- observation dimension;
- action-space size;
- reward mode;
- selected first valid block;
- one-step reward;
- termination flags;
- claim boundary.

## Behavior

`load_tiled_variant_input(...)` reads the tile index, extracts the semicolon-separated block IDs for the requested tile, loads the requested Phase 2 variant with the existing `load_variant_input(...)`, and filters the matrix to the tile block order.

`run_phase14_tiled_smoke(...)` creates the existing `Phase4InputContractEnv` from that tile subset, resets it, selects the first valid action, steps once, and returns a summary.

By default, Phase 14 rejects variants whose reward mode is `base_plus_suitability_reward`. This keeps Phase 14 aligned with the Phase 10 reward gate. A future phase can add an explicit bounded reward override only after the weak-label gate changes.

## Claim Boundary

Phase 14 is a tile-level input-contract smoke check. It does not train, tune, evaluate, or compare a DRL policy. It does not enable suitability reward. It does not report planning performance. It only verifies that a real Phase 13 tile can be loaded into the existing environment contract.

## Implementation Units

- `src/paper11_geofm/tiled_inputs.py`: tile-index reader, tile subset loader, one-step tiled smoke runner, JSON writer.
- `experiments/phase14_tiled_smoke_env/run_phase14_tiled_smoke.py`: CLI runner.
- `tests/test_phase14_tiled_smoke.py`: synthetic Phase 2/tile-index tests for subset loading, reward-variant rejection, writer output, and CLI output.
- Documentation updates in `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`, and `reproducibility/FILE_MANIFEST.tsv`.

## Real Bishan Expected Result

For the current Phase 13 largest tile:

- tile ID: `tile_r003_c003`;
- B1 tile blocks: `2234`;
- B1 features: `81`;
- B1 tiled observation dimension: `180957`;
- reward mode: `base_planning_reward`;
- one-step reward: `0.0`;
- selected block is the first block listed in the Phase 13 tile index.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: Phase 14 reuses existing Phase 2 and Phase 4 contracts.
- Scope check: this phase is limited to one tile-level smoke step and does not train.
- Ambiguity check: default variant, reward-mode rejection, outputs, and claim boundary are explicit.
