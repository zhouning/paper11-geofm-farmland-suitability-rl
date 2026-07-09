# Phase 67 Candidate Reward/Label Target Audit Design

## Purpose

Phase 67 follows the Phase 66 reward-label representation audit. Phase 66 found
that the current deterministic `base_planning_reward` target is dominated by
the explicit planning features that directly define it, while GeoFM-derived
extra representation columns add little independent ranking signal under that
same target.

The next step should not be another policy-training tweak and should not be an
immediate reward rewrite. Phase 67 should first audit candidate reward or label
targets in a read-only way, under the existing Phase 10, Phase 18, Phase 39,
and Phase 40 gate constraints. Its job is to determine whether a defensible
diagnostic target exists for a later experiment, or whether Paper11 still
requires independent non-leakage labels before any reward-redesign route is
scientifically credible.

Phase 67 is algorithm and experiment evidence work. It does not revise formal
submission files and it does not make manuscript-level claims.

## Scientific Question

Can the existing Paper11 artifacts support a candidate diagnostic target that
is less explicitly determined than `base_planning_reward`, more aligned with
GeoFM-derived representation signal, and still bounded by the current
suitability-reward and independent-label gates?

The expected output is a conservative decision about the next algorithm route:
run a later diagnostic-training experiment on a candidate target, collect
independent labels first, or stop reward-redesign work because only explicit or
leakage-risk targets are available.

## Why Phase 67 Is Needed

Phase 66 produced the current blocking evidence:

- status: `base_reward_target_masks_geofm_signal`;
- reward-component attribution rows: `2720`;
- selected-block atlas rows: `75`;
- representation-rank alignment rows: `75`;
- B0 explicit proxy `R2`: `0.9973990529`;
- GeoFM explicit proxy `R2`: `0.9973990529`;
- GeoFM extra representation proxy `R2`: `0.029462`;
- GeoFM representation minus B0 explicit proxy `R2`: `-0.9679370606`;
- GeoFM representation minus explicit top-k enrichment: `-0.08125`;
- `misses_explicit_reward_components`: `75`;
- `representation_not_aligned_with_base_reward`: `60`;
- `standardization_hurts_rank_geometry`: `30`.

This means the current base-reward set-policy route is answering a target that
GeoFM was not designed to uniquely improve. Continuing to tune policy models
against that same target is unlikely to create a defensible GeoFM advantage.

At the same time, suitability reward remains blocked:

- Phase 10 real gate: `not_ready_for_suitability_reward`;
- Phase 10 recommendation: `do_not_enable_suitability_reward`;
- Phase 18: `suitability_reward_allowed: false`;
- Phase 39/40: independent non-leakage label inputs remain missing unless a
  registry is supplied.

Therefore Phase 67 must not simply invent a new reward. It should audit
candidate targets and label sources before any training or reward integration.

## Scope

Phase 67 should be read-only and reproducible from existing artifacts:

- Phase 2 real Bishan feature tables for B0/B1 and available weak labels;
- Phase 8 D4P8/D4P16 feature tables;
- Phase 61 D6R8/D6R16 projection-control feature tables;
- Phase 10 reward-readiness gate output;
- Phase 18 planning-reward readiness output;
- Phase 39 independent-label audit output where available;
- Phase 40 independent-label gate output where available;
- Phase 66 reward-label representation audit output;
- tile index metadata from Phase 13.

The default audit should use the current Phase 52/63/66 variant set where
available:

- variants: `B0,D4P8,D4P16,D6R8,D6R16`;
- train tile: `tile_r003_c003`;
- eval tiles:
  `tile_r002_c003,tile_r005_c004,tile_r005_c003,tile_r000_c004,tile_r001_c004`;
- candidate top-k values: `8,16,32`.

## Non-Goals

Phase 67 should not:

- train or fine-tune any PPO, BC, set-policy, or scorer model;
- modify `planning_reward.py`;
- enable suitability reward;
- create B2/B3 reward variants;
- claim that any candidate target is an agronomic ground truth;
- claim GeoFM advantage, PCA optimality, transfer, or submission readiness;
- edit `paper/submission/final/*`.

## Candidate Target Families

Phase 67 should audit candidate targets as diagnostic targets only. A candidate
target is a vector of per-block scores or labels, not a reward implementation.

### 1. Current Base-Reward Target

Include `base_planning_reward` as the control target. It should reproduce the
Phase 66 finding that explicit planning columns nearly fully explain the
target. This provides the baseline for deciding whether any other target is
meaningfully different.

### 2. Weak DLTB/Slope-Derived Labels

Audit existing weak labels such as:

- `current_farmland_label`;
- `farmland_or_orchard_label`;
- `low_slope_farmland_label`.

These labels are known leakage-risk or explicit-feature-derived labels. Phase
67 may quantify their GeoFM alignment, but the gate must not promote them to
reward-ready or independent-label-ready status.

### 3. GeoFM-Derived Representation Scores

Construct diagnostic-only candidate scores from representation geometry, for
example:

- D4/D6 representation norm or robust norm;
- distance to high-base-reward or high-low-slope block centroids;
- first few representation dimensions or principal coordinates;
- representation similarity to top-k base-reward blocks.

These targets may be useful for diagnosing whether GeoFM organizes blocks in a
meaningful way, but they are self-referential and must not be treated as
external agronomic labels.

### 4. Explicit-Residual Targets

Construct residual diagnostic targets by regressing or fitting simple explicit
feature proxies to:

- base reward;
- weak labels;
- representation-derived scores.

The target of interest is the residual component not explained by explicit
planning columns. Phase 67 should test whether GeoFM extra columns explain this
residual better than explicit columns or D6 controls.

Residual targets remain diagnostic only unless they can be tied to an
independent non-leakage label source.

## Proposed Diagnostic Modules

### 1. Candidate Target Inventory

Create a structured inventory of all candidate targets with:

- target ID;
- target family;
- source artifacts;
- row count and non-missing count;
- score range, variance, and unique count;
- whether it is continuous, binary, or ordinal;
- whether higher values mean better suitability;
- whether it directly reuses explicit planning features;
- whether it depends on GeoFM representation columns;
- whether it is self-referential, weak-label-derived, or independent-label-like.

Targets with zero variance, missing alignment to block IDs, or unclear
direction should be marked unusable.

### 2. Leakage And Gate Audit

For each candidate target, classify gate risk:

- `explicit_reward_defined`: target is directly defined by current
  base-reward explicit columns;
- `explicit_label_leakage_risk`: target is derived from DLTB, slope, or other
  explicit planning labels;
- `geofm_self_reference`: target is built from GeoFM representation values;
- `independent_label_missing`: target is not backed by a registered independent
  label;
- `diagnostic_only_allowed`: target is allowed only for read-only analysis or a
  later diagnostic training experiment with strict claim boundaries;
- `reward_training_blocked`: target must not be used as a reward.

This module should import the current Phase 10/18/39/40 statuses into every
target row so the reason for blocking or diagnostic-only use is explicit.

### 3. Explicit-Versus-GeoFM Information Gain

For each candidate target and tile/variant group, estimate how much of the
target is explained by:

- required explicit base-reward columns;
- all explicit columns;
- GeoFM extra representation columns;
- D6 random projection controls where available;
- combined explicit plus representation columns.

Suggested diagnostics:

- Spearman correlation;
- top-k enrichment for `k=8,16,32`;
- simple OLS or ridge proxy `R2`;
- residual `R2` after fitting explicit columns first;
- D4/D6 representation advantage relative to D6 controls;
- B0/D4/D6 top-k target overlap.

The key quantity is not raw correlation alone, but whether GeoFM extra columns
explain target variation that explicit columns do not already explain.

### 4. Candidate Target Gate

Reduce the audit to one conservative status:

- `candidate_target_found_for_diagnostic_training`: at least one target has
  nontrivial residual variation beyond explicit columns, GeoFM-derived columns
  explain that residual better than controls, and the target is allowed only
  for a later diagnostic-training phase with strict claim boundaries.
- `only_leakage_or_explicit_targets_found`: all usable targets are either
  explicit-feature-defined, DLTB/slope leakage-risk labels, or fully explained
  by explicit columns.
- `independent_label_required_before_reward_redesign`: no defensible candidate
  target can support reward redesign without an external or registered
  independent non-leakage label.
- `insufficient`: required artifacts are missing, block IDs cannot be aligned,
  or diagnostics conflict without supporting a single next step.

The gate should include numeric evidence, not only a status label.

## Artifacts

Suggested implementation files for the later implementation plan:

- `src/paper11_geofm/phase67_candidate_reward_label_target_audit.py`;
- `experiments/phase67_candidate_reward_label_target_audit/run_phase67_candidate_reward_label_target_audit.py`;
- `tests/test_phase67_candidate_reward_label_target_audit.py`;
- `paper/phase28_results/33_phase67_candidate_reward_label_target_audit.md`.

Generated artifacts should remain under ignored experiment output directories:

- `phase67_candidate_target_inventory.csv`;
- `phase67_candidate_target_gate_audit.csv`;
- `phase67_candidate_target_information_gain.csv`;
- `phase67_candidate_target_summary.csv`;
- `phase67_candidate_reward_label_target_audit.json`;
- `phase67_candidate_reward_label_target_audit.md`.

## Error Handling

The implementation should fail clearly when:

- required Phase 2, Phase 8, Phase 10, Phase 13, Phase 18, Phase 61, or Phase
  66 artifacts are missing;
- optional Phase 39 or Phase 40 artifacts are requested but missing;
- block IDs cannot be aligned across B0, D4, D6, labels, and tile metadata;
- candidate targets have no usable non-missing rows;
- a target direction cannot be inferred or specified;
- explicit, GeoFM, and D6 feature groups cannot be separated;
- a target would be promoted as reward-ready despite gate-blocking evidence.

Missing optional independent-label gate artifacts should downgrade the gate
toward `independent_label_required_before_reward_redesign`, not silently create
positive target claims.

## Testing Requirements

Unit tests should cover:

- candidate target inventory for continuous, binary, zero-variance, and
  missing-value targets;
- leakage/gate classification for base reward, weak labels, GeoFM-derived
  self-reference targets, and residual targets;
- explicit versus GeoFM feature-group separation;
- Spearman/top-k/proxy `R2` metrics with ties and constant columns;
- residual target construction after explicit-only fitting;
- information-gain summaries where GeoFM helps, where explicit features fully
  explain the target, and where controls match GeoFM;
- all four candidate target gate statuses;
- writer outputs JSON, CSV, and Markdown artifacts;
- CLI parser accepts all required artifact paths and optional Phase 39/40
  paths.

Verification for the later implementation should include:

- targeted Phase 67 unit tests;
- Phase 66, Phase 65, Phase 64, and Phase 63 regression tests;
- `D:\adk\.venv\Scripts\python.exe scripts\smoke_check.py`;
- `git diff --check`;
- a check that `paper/submission/final/*` remains unchanged.

## Claim Boundary

Phase 67 may report whether existing artifacts contain a candidate diagnostic
target that is less explicitly determined and more GeoFM-aligned than the
current base reward. It may not claim that any candidate target is an
independent agronomic label, that suitability reward is ready, that B2/B3 can
be enabled, that GeoFM improves real farmland suitability planning, or that the
current manuscript is ready for submission.

The manuscript remains downstream of algorithm and experiment evidence.
