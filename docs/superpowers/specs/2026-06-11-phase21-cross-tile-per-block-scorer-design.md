# Phase 21 Cross-Tile Per-Block Scorer Design

## Goal

Add the first bounded cross-tile learned-policy pilot for Paper11 B0/B1 tiled
experiments. Phase 21 should verify that a learned scorer trained on one real
tile can evaluate a distinct real tile without depending on the flat
tile-specific observation shape that blocked Phase 20.

## Problem

Phase 20 proved that short B0/B1 MaskablePPO training can execute on a real
tile, but the trained policy cannot be evaluated on a different tile when the
tile block counts differ. The current `Phase4InputContractEnv` flattens the
entire tile into one fixed-size observation and uses `Discrete(n_blocks)` as
the action space. Stable-Baselines3 therefore binds the model to the train
tile's observation and action dimensions.

Paper11 already identifies a dimension-agnostic remedy: score each block with a
shared per-block policy and use the scores as masked action logits. Phase 21
implements the smallest reproducible version of that idea as a deterministic
linear block scorer trained from the Phase 19 base planning reward. This tests
the cross-tile data flow without claiming final DRL performance.

## Inputs

Phase 21 consumes:

- Phase 2 output directory containing ready B0/B1 feature tables;
- Phase 13 tile index CSV;
- train-tile selection, defaulting to the largest tile;
- evaluation-tile selection, defaulting to the next largest distinct tile;
- variants, restricted to B0/B1;
- ridge regularization strength;
- evaluation max steps;
- random seed for deterministic baselines;
- output directory.

Suitability reward variants B2/B3 must be rejected by default.

## Protocol

For each requested variant:

1. Load the train tile and evaluation tile as `TiledVariantInput` objects.
2. Compute the deterministic base planning reward for each train-tile block.
3. Fit a shared standardized ridge-linear scorer from per-block features to
   train-tile base rewards.
4. Evaluate the scorer on the distinct evaluation tile. At each step, score all
   unselected evaluation blocks and choose the valid block with the highest
   predicted score.
5. Run the same evaluation tile with `first_valid` and `seeded_random`
   deterministic baselines.
6. Write variant-level summary rows and per-step action traces.

The learned scorer is not a replacement for the later MaskablePPO shared
scorer. It is a bounded architectural pilot that proves the cross-tile policy
interface can be made independent of tile block count.

## Outputs

Phase 21 writes:

- `phase21_cross_tile_block_scorer_summary.csv`;
- `phase21_cross_tile_block_scorer_traces.json`.

Each summary row should include:

- row type (`learned_block_scorer`, `first_valid`, `seeded_random`);
- variant ID;
- train tile ID;
- evaluation tile ID;
- seed;
- ridge alpha;
- evaluation max steps;
- train and evaluation block counts;
- per-block feature dimension;
- flat evaluation observation shape for traceability;
- action-space size;
- episode steps;
- total base planning reward;
- selected block IDs;
- claim boundary.

The JSON trace should preserve the full configuration, tile selections, model
metadata, summaries, and per-step actions.

## Readiness Rules

Phase 21 can report that a cross-tile per-block scorer pilot executed only when:

- B0 and B1 requested variants are ready;
- selected train and evaluation tiles are non-empty and distinct;
- train and evaluation inputs for each variant have the same per-block feature
  dimension;
- the train features contain the explicit columns required for the Phase 19
  base planning reward;
- all learned-scorer and baseline evaluation rollouts finish without invalid
  actions;
- JSON and CSV artifacts are written.

Phase 21 must not set full planning-performance readiness to true. It does not
enable suitability reward, tune RL hyperparameters, run multi-seed evaluation,
or compare final policy performance. It only removes the Phase 20 shape blocker
at the pilot interface level.

## Claim Boundary

Phase 21 is a bounded cross-tile per-block scorer pilot for B0/B1 under the
deterministic base planning reward. It demonstrates that learned block scoring
can be trained on one tile and evaluated on another tile with a different block
count. It does not prove GeoFM superiority, suitability-reward benefit,
cross-region transfer, or submission-level planning performance.

## Real Bishan Expected Result

For current real Bishan artifacts, the default run should:

- select `tile_r003_c003` as the largest train tile;
- select a distinct non-empty evaluation tile;
- run B0 and B1 with a deterministic ridge-linear scorer;
- write both Phase 21 artifacts;
- report completed learned-scorer and baseline rows for each variant;
- keep the claim boundary explicit.

Numerical rewards are pilot diagnostics only. They should not be interpreted as
final performance evidence without later multi-seed, multi-tile, ablation, and
held-out-region evaluation.

## Implementation Units

- `src/paper11_geofm/cross_tile_block_scorer.py`: tile selection, B0/B1
  validation, ridge-linear per-block scorer, cross-tile evaluation rollouts,
  baseline rollouts, artifact writers, and model metadata.
- `experiments/phase21_cross_tile_block_scorer/run_phase21_cross_tile_block_scorer.py`:
  CLI runner.
- `tests/test_phase21_cross_tile_block_scorer.py`: tests for tile selection,
  variant rejection, scorer fitting, cross-tile rollout, artifact writing, and
  CLI output.
- Documentation updates in `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`,
  `reproducibility/FILE_MANIFEST.tsv`, and
  `paper/submission/01_ijaeog_submission_readiness.md`.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: Phase 21 directly addresses the Phase 20
  `blocked_variable_observation_shape` finding without changing the Phase 4
  flat environment.
- Scope check: this phase is limited to B0/B1 cross-tile per-block scorer
  pilots; B2/B3 suitability reward remains disabled.
- Ambiguity check: tile defaults, outputs, readiness rules, and claim boundary
  are explicit.
