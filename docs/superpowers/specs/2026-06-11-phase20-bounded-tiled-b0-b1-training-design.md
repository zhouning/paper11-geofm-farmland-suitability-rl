# Phase 20 Bounded Same-Tile B0/B1 Training Design

## Goal

Add the first bounded real-data training and evaluation protocol for Paper11
B0/B1 tiled experiments. Phase 20 should test whether the already verified
tiled MaskablePPO path can run a small, reproducible same-tile training loop
and produce machine-readable pilot evidence for explicit-only and
GeoFM-enhanced base-reward variants.

## Problem

Phase 17 proves that MaskablePPO can consume a real tiled B1 environment, but it
only performs an API readiness smoke check. Phase 18 still reports that true
planning-performance experiments are not ready. Phase 19 removes one blocker by
implementing a deterministic `base_planning_reward`, but there is still no
bounded protocol that trains and evaluates B0/B1 under the same base reward on
a real tiled episode.

Phase 20 should create that first protocol without enabling suitability reward,
without running full-scale training, and without claiming final policy
superiority. The first attempted distinct-tile learned-policy rollout exposed
an architectural blocker: the current flat observation and action spaces depend
on tile block count, so a MaskablePPO model trained on one tile cannot safely
predict on a tile with a different shape.

## Inputs

Phase 20 consumes:

- Phase 2 output directory containing `experiment_variants.json` and ready
  B0/B1 feature tables;
- Phase 13 tile index CSV;
- train-tile selection, defaulting to the largest tile;
- evaluation-tile selection, defaulting to the same tile as training;
- variants, restricted to B0/B1 by default;
- total training timesteps;
- evaluation max steps;
- random seed;
- output directory.

Suitability reward variants B2/B3 must be rejected by default. Explicitly
requesting a different evaluation tile must also be rejected in Phase 20, with
the cross-tile learned-policy blocker recorded in the contract.

## Protocol

For each requested variant:

1. Load the selected train tile into `Phase4InputContractEnv`.
2. Train `sb3_contrib.MaskablePPO` on CPU for a bounded timestep budget.
3. Load the same selected tile into a fresh evaluation environment.
4. Roll out the trained policy deterministically with action masks for
   `eval_max_steps` or until the episode terminates.
5. Run the same tile with deterministic non-learning baselines:
   `first_valid` and `seeded_random`.
6. Write variant-level summary rows and full action traces.

The trained-model rollout is evidence that the pipeline can execute a bounded
learned-policy pilot. It is not final evidence that GeoFM improves planning.

## Outputs

Phase 20 writes:

- `phase20_bounded_tiled_training_summary.csv`;
- `phase20_bounded_tiled_training_traces.json`.

Each summary row should include:

- row type (`trained_policy`, `first_valid`, `seeded_random`);
- variant ID;
- train tile ID;
- evaluation tile ID;
- seed;
- training timesteps;
- evaluation max steps;
- number of blocks and features;
- observation shape;
- action-space size;
- episode steps;
- total base planning reward;
- selected block IDs;
- claim boundary.

The JSON trace should preserve the full configuration, selected train/eval tile
metadata, dependency metadata, summaries, per-step actions, same-tile
evaluation scope, and the cross-tile learned-policy blocker.

## Readiness Rules

Phase 20 can report that a bounded B0/B1 pilot executed only when:

- B0 and B1 requested variants are ready;
- selected train and evaluation tiles are non-empty and identical for the
  learned-policy pilot;
- the train/evaluation environments expose action masks;
- all trained-policy and baseline evaluation rollouts finish without invalid
  actions;
- JSON and CSV artifacts are written.

It must not set `performance_experiment_ready` to true for the whole paper.
Phase 18 remains the conservative readiness gate until a later phase defines
multi-seed, multi-tile, and transfer-evaluation thresholds. Cross-tile
learned-policy evaluation remains blocked until a variable-size, padded, or
per-block policy design replaces the current flat tile-specific observation
contract.

## Claim Boundary

Phase 20 is a bounded same-tile training pilot for B0/B1 under the
deterministic base planning reward. It trains and evaluates short MaskablePPO
runs only to verify the controlled training/evaluation protocol. It does not
tune hyperparameters, enable suitability reward, test cross-tile transfer,
compare final policy performance, prove GeoFM superiority, or support
submission-level planning-performance claims.

## Real Bishan Expected Result

For current real Bishan artifacts, the default run should:

- select `tile_r003_c003` as the largest train tile;
- use `tile_r003_c003` again as the same-tile learned-policy evaluation tile;
- run B0 and B1 with a small CPU timestep budget;
- write both Phase 20 artifacts;
- report completed trained-policy and baseline rows for each variant;
- record `blocked_variable_observation_shape` for cross-tile learned-policy
  evaluation;
- keep the claim boundary explicit.

The numerical rewards are pilot diagnostics only. They should be recorded but
not interpreted as final performance evidence without later multi-seed,
multi-tile, and held-out-region evaluation under an architecture that supports
variable-size transfer.

## Implementation Units

- `src/paper11_geofm/bounded_tiled_training.py`: tile selection, MaskablePPO
  training, deterministic evaluation rollouts, baseline rollouts, artifact
  writers, and dependency metadata.
- `experiments/phase20_bounded_tiled_training/run_phase20_bounded_tiled_training.py`:
  CLI runner.
- `tests/test_phase20_bounded_tiled_training.py`: tests for same-tile tile
  selection, cross-tile guardrails, variant rejection, tiny
  training/evaluation, artifact writing, and CLI output.
- Documentation updates in `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`,
  `reproducibility/FILE_MANIFEST.tsv`, and
  `paper/submission/01_ijaeog_submission_readiness.md`.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: Phase 20 follows Phase 17/18/19 and does not change the
  Phase 10 suitability gate.
- Scope check: this phase is limited to B0/B1 bounded same-tile pilot training
  and evaluation; B2/B3 suitability reward remains disabled.
- Ambiguity check: default tile selection, outputs, readiness rules, and claim
  boundary are explicit.
