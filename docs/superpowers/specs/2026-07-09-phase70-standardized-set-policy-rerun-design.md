# Phase 70 Standardized Set-Policy Rerun Design

## Purpose

Phase 70 follows the Phase 63 set-policy oracle-pretraining experiment, the
Phase 64 set-policy error diagnosis, and the Phase 69 label-free synthesis
boundary gate.

Phase 63 showed that the set-policy architecture strongly improves over the
flattened PPO route under the Bishan base-reward protocol, but it did not
separate GeoFM-derived D4 variants from B0 or D6 controls. Phase 64 then found
that D4/D6 underperformance coincides with feature scale and rank flags while
behavior cloning convergence is adequate. Phase 69 kept the current defensible
claim narrowed to a bounded low-dimensional route and blocked suitability,
reward-redesign, and agronomic claims.

Phase 70 should test the concrete algorithm hypothesis produced by Phase 64:
train-tile-fitted feature standardization may remove scale-driven set-policy
failure and reveal whether GeoFM-derived low-dimensional variants improve under
the set-policy route.

## Scientific Question

Does train-tile-fitted standardization improve set-policy behavior-cloned
selection enough to strengthen the current bounded low-dimensional algorithm
claim, especially for D4P8 and D4P16 relative to B0 and D6 controls?

The expected outcome is not assumed to be positive. Phase 70 should produce a
machine-checkable decision separating three cases:

- standardization improves GeoFM-derived set-policy evidence;
- standardization improves the set-policy architecture but still does not
  distinguish GeoFM-derived variants;
- standardization is not sufficient and the next algorithm route must change.

## Why Phase 70 Is Needed

The current evidence chain has a specific algorithm gap:

- Phase 63 supports the set-policy architecture route, with a large architecture
  delta over flattened PPO.
- Phase 63 does not support GeoFM-derived D4 variants over B0 or D6 controls.
- Phase 64 reports `standardization_route_supported`, with adequate behavior
  cloning convergence and feature scale/rank flags for D4/D6 variants.
- Phase 69 shows that Paper11 cannot broaden claims through manuscript writing;
  the next progress must come from a stronger algorithm experiment or external
  independent labels.

A standardized rerun is therefore the smallest high-leverage experiment that
can test a diagnosed failure mode instead of adding uncontrolled variants.

## Scope

Phase 70 is an algorithm and experiment phase. It should run a standardized
set-policy rerun under the existing Bishan base-reward protocol and compare the
result with the unstandardized Phase 63 baseline.

Included:

- add a train-tile-fitted standardization transform for set-policy inputs;
- reuse the Phase 63 set-policy scorer, oracle trajectory, behavior cloning,
  greedy rollout, and analysis conventions where possible;
- run B0, D4P8, D4P16, D6R8, and D6R16 on the same five evaluation tiles and
  three seeds used by Phase 63;
- compare Phase 70 standardized rewards with Phase 63 unstandardized rewards;
- write JSON, CSV, and Markdown artifacts;
- record a paper-facing Phase 70 result note under `paper/phase28_results`.

Excluded:

- changing the reward definition;
- enabling B2/B3;
- using external labels or weak labels as independent labels;
- changing tile selection, seed selection, or eval max steps for the primary
  real run;
- claiming suitability reward readiness, independent agronomic validity,
  cross-region transfer, PCA optimality, or formal submission readiness;
- modifying `paper/submission/final/*`.

## Standardization Protocol

Phase 70 should use a leakage-safe train-tile-fitted transform.

For each variant independently:

1. Load the train tile using the same Phase 63 contract resolution.
2. Compute feature-wise mean and standard deviation from the train tile state
   matrix only.
3. Replace zero, non-finite, or near-zero standard deviations with `1.0`.
4. Transform the train tile and every evaluation tile with the same variant-level
   train-tile parameters.
5. Preserve block ids, feature column names, reward mode, state groups, and
   source-table metadata.
6. Keep a separate unstandardized reward matrix for base planning reward and
   oracle construction. Standardization must affect model inputs only; it must
   not change the reward target or the oracle ranking.

The primary design decision is a dual-matrix protocol: standardized features are
used by the behavior-cloned set-policy scorer, while original unstandardized
features are used for base-reward scoring and oracle trajectories. The
comparison target remains the same base-reward selection task as Phase 63.

## Architecture

Add one focused module:

`src/paper11_geofm/phase70_standardized_set_policy_rerun.py`

Responsibilities:

- own the Phase 70 claim boundary;
- define a small standardization parameter structure;
- fit standardization parameters from train-tile inputs only;
- apply the transform to train and eval model-input matrices without changing
  identity metadata;
- run standardized behavior cloning and rollout with Phase 63 scorer/model
  components while preserving original unstandardized matrices for reward and
  oracle computation;
- build a Phase 70 comparison against Phase 63 baseline artifacts;
- compute the top-level Phase 70 status;
- write stable CSV, JSON, and Markdown artifacts.

Add one thin runner:

`experiments/phase70_standardized_set_policy_rerun/run_phase70_standardized_set_policy_rerun.py`

Responsibilities:

- parse Phase 2, Phase 8, Phase 61, tile index, Phase 63 comparison, Phase 63
  rollout, and output paths;
- call the Phase 70 module;
- print status, artifact paths, recommended next step, and claim boundary.

Add focused tests:

`tests/test_phase70_standardized_set_policy_rerun.py`

Responsibilities:

- standardization parameters are fitted from train tile only;
- zero-variance and non-finite scale columns are handled safely;
- transformed tiled inputs preserve block ids and metadata;
- standardized set-policy analysis can run on a small fixture;
- comparison gate distinguishes GeoFM improvement, architecture-only
  improvement, and not-sufficient cases;
- writer produces stable artifacts;
- CLI parser and fixture run work.

## Data Flow

1. The runner receives the same real Bishan input roots used by Phase 63 plus
   Phase 63 baseline artifacts.
2. The module builds a Phase 63-compatible contract for the requested variants,
   tiles, seeds, and training hyperparameters.
3. For each variant, Phase 70 loads the train tiled input and fits feature mean
   and standard deviation from that train tile.
4. Phase 70 trains the set-policy behavior cloner on standardized train model
   inputs while using original train features to define the oracle target order.
5. For each eval tile and seed, Phase 70 applies the same train-fitted transform
   to model inputs, rolls out the greedy policy, scores selected blocks with the
   original unstandardized feature matrix, and records reward, oracle gap,
   selected blocks, and convergence history.
6. Phase 70 analyzes standardized rollout rows using Phase 63 comparison logic
   where possible.
7. Phase 70 joins standardized rows to Phase 63 baseline rows by variant, eval
   tile, and seed, then computes standardized-minus-baseline deltas.
8. Phase 70 writes artifacts and a paper-facing result note.

## Status Model

Phase 70 should return one top-level status:

- `standardization_improves_geofm_set_policy_route`: standardized D4P8/D4P16
  improve relative to Phase 63 baseline and close or reverse their mean gap to
  B0 and D6 controls.
- `standardization_improves_architecture_not_geofm`: standardized rerun improves
  aggregate set-policy rewards or oracle gaps, but D4P8/D4P16 remain behind B0
  and D6 controls.
- `standardization_not_sufficient`: standardized rerun does not improve the
  architecture route or does not reduce the relevant D4 gaps.
- `standardized_rerun_incomplete`: required rows, variants, tiles, seeds, or
  baseline joins are missing.

The current expected result is unknown. Phase 70 must not assume that
standardization will help.

## Comparison Rules

Phase 70 should compute at least these summaries:

- mean standardized BC reward by variant;
- mean Phase 70 minus Phase 63 reward delta by variant;
- D4P8/D4P16 versus B0 mean reward delta under standardized rerun;
- D4P8/D4P16 versus D6R8/D6R16 mean reward delta under standardized rerun;
- oracle gap and oracle gap fraction summaries;
- coverage report for expected variant, tile, and seed rows;
- selected-block overlap or row-level delta table if available from Phase 70
  and Phase 63 artifacts.

A positive algorithm result requires more than architecture improvement. The
strongest useful positive finding is that standardized D4 variants improve their
relative position against B0 and D6 controls under the same base-reward protocol.

## Artifact Outputs

The real run should write under:

`experiments/phase70_standardized_set_policy_rerun/outputs/phase52_full5_seed3`

Expected artifacts:

- `phase70_standardization_parameters.csv`;
- `phase70_standardized_bc_training_history.csv`;
- `phase70_standardized_bc_rollout_summary.csv`;
- `phase70_standardized_oracle_summary.csv`;
- `phase70_standardized_delta_table.csv`;
- `phase70_standardized_set_policy_comparison.json`;
- `phase70_standardized_set_policy_rerun.md`.

The paper-facing result note should be:

`paper/phase28_results/36_phase70_standardized_set_policy_rerun.md`

## Error Handling

Hard errors:

- missing required input roots or artifacts;
- invalid or incomplete Phase 63 baseline artifacts;
- missing expected variant feature tables;
- missing expected tile ids;
- output directory cannot be written.

Recoverable evidence problems represented in outputs:

- a variant/tile/seed row is incomplete after a run;
- a baseline join is missing for a row;
- standardization produces finite but non-improving results;
- D4 variants improve in absolute reward but remain behind controls.

## Testing And Verification

Targeted tests:

- standardization fit uses only train-tile rows;
- applying standardization to eval tiles uses train-fitted parameters;
- zero or invalid standard deviations are replaced safely;
- transformed model-input views preserve block ids, feature columns, reward mode,
  state groups, source table, tile index metadata, and access to original reward
  matrices;
- fixture rerun produces standardized history, rollout, comparison, and delta
  rows;
- status model covers all four Phase 70 statuses;
- writer emits all expected artifact names;
- CLI runner accepts required inputs and succeeds on fixtures.

Regression checks:

- Phase 70 tests;
- Phase 64 set-policy error diagnosis tests;
- Phase 63 set-policy oracle-pretraining tests;
- Phase 69 label-free synthesis gate tests;
- smoke check;
- `git diff --check`;
- formal manuscript diff check under `paper/submission/final`.

## Expected Result Note

The Phase 70 result note should record:

- top-level Phase 70 status;
- standardized mean reward by variant;
- Phase 70 minus Phase 63 deltas;
- D4 versus B0 and D4 versus D6 summaries;
- whether standardization strengthens, narrows, or fails the current algorithm
  route;
- reproduction command;
- statement that no formal manuscript files changed.

## Claim Boundary

Phase 70 may claim only a standardized set-policy algorithm result under the
existing Bishan base-reward protocol.

Phase 70 must not claim suitability reward readiness, B2/B3 readiness,
independent agronomic suitability, external-label validation, cross-region
transfer, PCA optimality, GeoFM-specific matched-dimension superiority beyond
what the standardized set-policy comparison directly supports, or formal
submission readiness.