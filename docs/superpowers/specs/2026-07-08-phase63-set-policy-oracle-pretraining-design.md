# Phase 63 Set-Policy Oracle Pretraining Design

## Purpose

Phase 63 starts a stronger algorithm route after Phase 62 showed that the
existing D4 PCA controls do not outperform D6 raw-B1 random orthonormal
projection controls under the current padded MLP PPO protocol. The working
hypothesis is that the main bottleneck is no longer only representation choice,
but the policy architecture and training signal used for large candidate-block
selection.

Phase 63 is algorithm/model work. It does not revise formal manuscript files.

## Scientific Question

Can a task-aware set-style block-selection policy, initialized from deterministic
oracle trajectories and then optionally fine-tuned with PPO, extract stronger
performance from explicit and GeoFM-derived low-dimensional state variants than
the current flattened padded MLP PPO route?

A positive Phase 63 result would support the claim that Paper11 needs a policy
architecture matched to unordered tile/block sets before drawing conclusions
about GeoFM representation utility. A negative result would indicate that the
base-reward task itself, the current features, or the reward definition may be
the limiting factor.

## Why Phase 63 Is Needed

The current `Phase25PaddedTileEnv` flattens a variable-size tile into one large
padded vector and trains `MaskablePPO` with an MLP policy. This is workable as a
smoke-tested baseline, but it is weak for tiles with thousands of candidate
blocks:

- block identity is represented mostly through padded position in a long vector;
- the policy must infer per-block desirability indirectly from a global flat
  observation;
- PPO starts from a random policy despite the base reward being deterministic
  and oracle actions being cheap to compute;
- representation comparisons can be dominated by optimization difficulty rather
  than feature information.

Phase 63 therefore tests a task-matched set-policy learning setup before deciding that
GeoFM-derived state is not useful.

## Default Scope

The first Phase 63 implementation should remain bounded and reproducible:

- reward: existing `base_planning_reward` only;
- train/eval split: same Phase 52 five-tile, three-seed protocol where possible;
- variants: `B0,D4P8,D4P16,D6R8,D6R16` for the first run;
- no suitability reward, no B2/B3, no transfer claim, no formal manuscript edits;
- generated outputs remain under ignored `experiments/**/outputs/` directories.

Raw B1, D2/D3, and D5 controls can be added after the first Phase 63 contract is
stable.

## Proposed Architecture

### 1. Oracle Trajectory Builder

Create deterministic oracle trajectories for each tiled variant input. For the
base reward, each block has a computable reward independent of selection order,
so the default oracle can greedily select available blocks by descending
per-block `base_planning_reward` until `eval_max_steps` or block exhaustion.

Outputs should include:

- selected action indices and block IDs;
- per-step oracle reward;
- total oracle reward;
- per-tile top-k reward ceiling for comparison;
- deterministic tie-breaking by reward, then block ID or original index.

This oracle is not a manuscript claim by itself. It is a training and diagnostic
instrument.

### 2. Set-Style Block Scorer

Implement a policy module that treats a tile as a set of candidate blocks:

- input: block feature matrix, valid/action mask, selected mask, and small global
  context features;
- per-block encoder: shared MLP over block features;
- context encoder: pooled statistics over valid and selected blocks, such as
  mean/max/count fractions;
- scorer head: per-block logits conditioned on each encoded block plus context;
- action selection: masked softmax over unselected valid blocks.

The key difference from Phase 25 is that each block gets an explicit score. The
model no longer has to recover action-level preferences from a single flattened
observation vector.

### 3. Behavior Cloning Pretraining

Train the set-style scorer to imitate oracle actions before any PPO fine-tuning.
For each oracle step, optimize cross-entropy over the masked valid actions.
Report:

- top-1 oracle action accuracy;
- top-k hit rate;
- imitation total reward when rolled out greedily;
- gap to oracle total reward;
- gap to existing Phase 62 trained-policy totals.

This stage can be fully supervised and fast enough for tight iteration.

### 4. Optional PPO Fine-Tuning

After behavior cloning passes a minimum quality gate, wrap the scorer into a
policy-compatible environment or custom rollout loop and fine-tune under the
same base reward. PPO fine-tuning is optional for the first milestone; a strong
BC rollout result is already useful evidence that architecture/training signal
was the bottleneck.

If custom PPO integration is too expensive, Phase 63 can stop at BC rollout and
record the decision explicitly before a later Phase 64 RL fine-tuning phase.

## Comparisons and Status Rules

Primary Phase 63 comparisons should be reported separately for:

- oracle ceiling by variant;
- behavior-cloned rollout by variant;
- optional PPO-fine-tuned rollout by variant;
- existing Phase 62 flattened-PPO baseline where comparable.

Suggested statuses:

- `set_policy_route_supported`: BC or fine-tuned set policy improves over the
  comparable flattened-PPO baseline on pooled mean and has at least half positive
  tile-seed comparisons without coverage issues.
- `geofm_set_policy_advantage`: GeoFM-derived variants improve over B0 under the
  set-policy protocol with complete coverage.
- `architecture_improves_but_geofm_not_distinguished`: set policy improves over
  flattened PPO, but representation variants are not separable.
- `set_policy_route_not_supported`: complete coverage exists but no meaningful
  improvement over flattened PPO or oracle gap remains large.
- `insufficient`: missing, duplicate, or uninterpretable rows prevent a decision.

Do not use Phase 63 to resurrect unsupported PCA-specific claims unless D4 also
beats D6R controls under the new architecture.

## Outputs

Suggested implementation files:

- `src/paper11_geofm/phase63_set_policy_oracle_pretraining.py`
- `experiments/phase63_set_policy_oracle_pretraining/run_phase63_set_policy_oracle_pretraining.py`
- `tests/test_phase63_set_policy_oracle_pretraining.py`
- `paper/phase28_results/29_phase63_set_policy_oracle_pretraining.md`

Suggested generated artifacts:

- `phase63_oracle_trajectories.json`
- `phase63_oracle_summary.csv`
- `phase63_bc_training_history.csv`
- `phase63_bc_rollout_summary.csv`
- `phase63_set_policy_comparison.json`
- `phase63_set_policy_delta_table.csv`
- `phase63_set_policy_oracle_pretraining.md`

## Testing Requirements

Unit tests should cover:

- deterministic oracle ranking and tie-breaking;
- oracle trajectory termination at `eval_max_steps`;
- set-policy scorer output shape and mask behavior;
- behavior-cloning loss decreases on a tiny synthetic tile;
- greedy BC rollout never selects invalid or repeated actions;
- writer outputs JSON, CSV, and Markdown artifacts;
- CLI analyze-only or rollout-only behavior on tiny synthetic fixtures.

Real-run verification should include targeted Phase 63 tests, relevant Phase 62
regression tests, `python scripts\smoke_check.py`, and `git diff --check`.

## Claim Boundary

Phase 63 tests whether a task-aware policy architecture and oracle-pretrained
training signal improve base-reward block selection. It does not enable
suitability reward, does not test B2/B3, does not test transfer, does not prove
independent agronomic suitability, and does not justify formal submission-level
performance claims. Formal manuscript files should remain untouched until the
algorithm and experiment evidence is stable.
