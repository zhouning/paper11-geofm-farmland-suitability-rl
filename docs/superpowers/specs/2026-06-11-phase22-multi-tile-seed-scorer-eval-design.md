# Phase 22 Multi-Tile Multi-Seed Scorer Evaluation Design

## Goal

Add a bounded multi-tile, multi-seed evaluation protocol for the Phase 21
cross-tile per-block scorer. Phase 22 should evaluate whether the
variable-block-count scorer interface can run reproducibly across several
distinct real tiles and seeds under the deterministic base planning reward.

## Problem

Phase 21 removed the Phase 20 flat-observation shape blocker at the interface
level by training a shared per-block scorer on one tile and evaluating it on one
distinct tile. That is still a single train/evaluation split and one seed. The
submission-readiness audit still needs broader pilot evidence before any
policy-performance or transfer claim can be considered.

Phase 22 expands the Phase 21 protocol across multiple evaluation tiles and
multiple seeds. It remains a protocol-readiness pilot. It does not introduce
PPO training, suitability reward, hyperparameter tuning, or final performance
claims.

## Inputs

Phase 22 consumes:

- Phase 2 output directory containing ready B0/B1 feature tables;
- Phase 13 tile index CSV;
- train-tile selection, defaulting to the largest tile;
- evaluation-tile IDs, or a maximum number of largest distinct evaluation tiles;
- variants, restricted to B0/B1;
- ridge regularization strength;
- evaluation max steps;
- one or more integer seeds;
- output directory.

Suitability reward variants B2/B3 must be rejected by default.

## Protocol

For each requested variant:

1. Load the train tile once.
2. Fit the standardized ridge-linear per-block scorer from Phase 21 once per
   variant.
3. Select distinct evaluation tiles.
4. For each evaluation tile and seed, run:
   - `learned_block_scorer`;
   - `first_valid`;
   - `seeded_random`.
5. Write one summary row and one action trace for every
   variant/evaluation-tile/seed/policy combination.

The learned scorer and baselines use the same deterministic base planning
reward. The seed affects seeded-random baselines and is recorded for every row.

## Outputs

Phase 22 writes:

- `phase22_multi_tile_scorer_eval_summary.csv`;
- `phase22_multi_tile_scorer_eval_traces.json`.

Each summary row should include all Phase 21 fields plus `eval_tile_rank`.
The JSON trace should preserve configuration, selected tiles, seeds, model
metadata, summaries, and per-step actions.

## Readiness Rules

Phase 22 can report that a multi-tile scorer-evaluation pilot executed only
when:

- B0 and B1 requested variants are ready;
- the train tile is non-empty;
- at least one distinct evaluation tile is selected;
- train and evaluation inputs for each variant have the same per-block feature
  dimension;
- the train features contain the explicit columns required for the Phase 19
  base planning reward;
- every requested variant/evaluation-tile/seed/policy rollout finishes without
  invalid actions;
- JSON and CSV artifacts are written.

Phase 22 must not set full planning-performance readiness to true. It does not
enable suitability reward, run PPO training, claim cross-region transfer, or
compare final policy performance.

## Claim Boundary

Phase 22 is a bounded multi-tile, multi-seed per-block scorer evaluation pilot
for B0/B1 under the deterministic base planning reward. It demonstrates that
the Phase 21 variable-block-count scorer interface can be executed across
several evaluation tiles and seeds. It does not prove GeoFM superiority,
suitability-reward benefit, cross-region transfer, or submission-level planning
performance.

## Real Bishan Expected Result

For current real Bishan artifacts, the default run should:

- select `tile_r003_c003` as the largest train tile;
- select the largest distinct evaluation tiles;
- run B0 and B1 with a deterministic ridge-linear scorer;
- evaluate each selected tile across the requested seeds;
- write both Phase 22 artifacts;
- keep the claim boundary explicit.

Numerical rewards are pilot diagnostics only. They should not be interpreted as
final performance evidence without later PPO-compatible policy training,
suitability-reward resolution, ablations, and held-out-region evaluation.

## Implementation Units

- `src/paper11_geofm/multi_tile_scorer_eval.py`: Phase 22 contract, tile and
  seed selection, Phase 21 scorer reuse, rollout aggregation, artifact writing.
- `experiments/phase22_multi_tile_scorer_eval/run_phase22_multi_tile_scorer_eval.py`:
  CLI runner.
- `tests/test_phase22_multi_tile_scorer_eval.py`: tests for tile/seed
  selection, variant rejection, multi-tile rollout aggregation, artifact
  writing, and CLI output.
- Documentation updates in `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`,
  `reproducibility/FILE_MANIFEST.tsv`, and submission-readiness materials.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: Phase 22 extends Phase 21 without changing the Phase 4
  flat environment or enabling suitability reward.
- Scope check: this phase is limited to B0/B1 multi-tile, multi-seed scorer
  evaluation pilots.
- Ambiguity check: tile defaults, seed handling, outputs, readiness rules, and
  claim boundary are explicit.
