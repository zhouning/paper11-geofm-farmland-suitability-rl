# Phase 17 Tiled MaskablePPO Readiness Design

## Goal

Build a tiled MaskablePPO readiness smoke check on real Phase 13 tile inputs. Phase 17 verifies that the real tiled B1 contract can be consumed by `sb3-contrib` MaskablePPO for a tiny CPU-only `learn()` and masked `predict()` call.

## Problem

Phase 16 proves deterministic non-learning masked rollouts across every real tile. It still does not prove that the tiled real environment contract is compatible with the actual MaskablePPO training API. Phase 7 proved MaskablePPO compatibility only on small Phase 2 fixture/full-table inputs, not on real tiled inputs.

Phase 17 closes that compatibility gap without claiming policy quality. It is a readiness smoke check, not a training experiment.

## Inputs

Phase 17 consumes:

- Phase 2 output directory with ready variant CSVs;
- Phase 13 `phase13_tile_index.csv`;
- variant ID, default `B1`;
- optional tile ID;
- tile-selection mode, default `largest`;
- total timesteps, default `8`;
- seed, default `0`;
- output directory.

When no tile ID is supplied, Phase 17 selects the largest tile by parsed block count. For current real Bishan artifacts, this is `tile_r003_c003`.

## Outputs

The runner writes:

- `phase17_tiled_maskableppo_readiness.json`.

The JSON contains:

- selected tile ID;
- selection mode;
- variant ID;
- seed;
- total timesteps;
- number of blocks;
- number of features;
- observation shape;
- action-space size;
- reward mode;
- initial valid actions;
- masking support flag;
- predicted action;
- predicted action validity;
- selected block ID for the predicted action;
- dependency metadata for `stable-baselines3` and `sb3-contrib`;
- readiness status;
- recommendation;
- claim boundary.

## Behavior

Phase 17 loads one real tiled B1 environment using the existing tiled-input and Phase 4 action-mask contract. It rejects `base_plus_suitability_reward` variants by default. It then:

1. checks that action masking is supported by `sb3-contrib`;
2. instantiates `MaskablePPO` on CPU with tiny hyperparameters;
3. runs `learn(total_timesteps=8)` by default;
4. resets the environment;
5. runs masked deterministic `predict()`;
6. records whether the predicted action is valid.

The readiness status is `passed_tiled_maskableppo_smoke` only when masking is supported, the tiny learn call returns, and the predicted action is valid. Otherwise the runner fails with an explicit error.

## Claim Boundary

Phase 17 is a tiled MaskablePPO readiness smoke check. It does not train, tune, evaluate, or compare a useful DRL policy. It does not enable suitability reward. It does not report planning performance.

## Implementation Units

- `src/paper11_geofm/tiled_maskableppo_readiness.py`: tile selection, tiled env creation, MaskablePPO smoke runner, JSON writer.
- `experiments/phase17_tiled_maskableppo_readiness/run_phase17_tiled_maskableppo_readiness.py`: CLI runner.
- `tests/test_phase17_tiled_maskableppo_readiness.py`: tests for largest-tile selection, tile override, reward-variant rejection, artifact writing, and CLI output.
- Documentation updates in `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`, and `reproducibility/FILE_MANIFEST.tsv`.

## Real Bishan Expected Result

For current real Bishan Phase 13 tiles with B1, largest-tile selection, seed `0`, and `total_timesteps=8`:

- selected tile ID is `tile_r003_c003`;
- blocks are `2234`;
- features are `81`;
- observation shape is `180957`;
- action space is `Discrete(2234)`;
- reward mode is `base_planning_reward`;
- initial valid actions are `2234`;
- masking support is `true`;
- predicted action is valid;
- readiness status is `passed_tiled_maskableppo_smoke`.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: Phase 17 reuses Phase 14 tile loading, Phase 4 action masks, and Phase 7 MaskablePPO smoke patterns.
- Scope check: this phase is limited to a single-tile compatibility/readiness smoke and does not perform policy evaluation.
- Ambiguity check: default variant, tile selection, timesteps, output artifact, readiness status, and claim boundary are explicit.
