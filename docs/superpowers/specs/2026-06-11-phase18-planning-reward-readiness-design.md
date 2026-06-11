# Phase 18 Planning Reward Readiness Design

## Goal

Build an executable readiness gate that states whether Paper11 can begin true
planning-performance DRL experiments on the real Bishan tiled artifacts.

## Problem

Phase 17 proves that a real tiled B1 contract can be consumed by the
MaskablePPO API for a tiny action-mask compatibility smoke. It does not prove
that useful planning training is possible. The current B0/B1 reward contract is
named `base_planning_reward`, but the Phase 4 environment returns `0.0` for all
non-suitability reward modes. B2/B3 suitability reward variants are also still
blocked by the Phase 10 real-data gate.

Without an explicit gate, it is easy to misread "MaskablePPO can run" as
"Paper11 has started real policy evaluation." Phase 18 closes that claim gap by
turning the current limitations into a machine-readable audit.

## Inputs

Phase 18 consumes:

- Phase 2 output directory containing `experiment_variants.json`;
- Phase 10 reward-readiness gate JSON;
- Phase 12 real scale audit JSON;
- optional Phase 17 tiled MaskablePPO readiness JSON;
- output directory.

The optional Phase 17 artifact records whether the tiled action-mask API smoke
passed. It does not override missing reward implementation or reward-readiness
failures.

## Outputs

The runner writes:

- `phase18_planning_reward_readiness.json`.

The JSON contains:

- B0/B1 base reward modes from the Phase 2 manifest;
- whether the base planning reward is implemented;
- concrete evidence for the current base reward status;
- Phase 10 suitability reward status and recommendation;
- Phase 12 real scale gates;
- optional Phase 17 tiled MaskablePPO API status;
- blocked reasons;
- `performance_experiment_ready`;
- recommended next step;
- claim boundary.

## Behavior

Phase 18 reads the current artifact chain and applies conservative readiness
rules:

1. B0 and B1 must be ready in Phase 2.
2. B0 and B1 must use `base_planning_reward`.
3. The current environment must have an implemented non-zero base planning
   reward before performance experiments can start.
4. Suitability reward is allowed only when Phase 10 and Phase 12 allow it.
5. Full flat training readiness remains blocked when Phase 12 reports that flat
   full-scale training is not ready.
6. Phase 17 API readiness is recorded as useful infrastructure evidence, but it
   is not sufficient for performance readiness.

For the current codebase, Phase 18 should report
`base_planning_reward_implemented: false` because
`Phase4InputContractEnv._contract_reward()` returns `0.0` for non-suitability
reward modes. Therefore `performance_experiment_ready` must be `false`.

## Claim Boundary

Phase 18 is a readiness audit. It does not implement a planning reward, train a
policy, tune hyperparameters, compare policies, enable suitability reward, or
report planning performance.

## Implementation Units

- `src/paper11_geofm/planning_reward_readiness.py`: JSON readers, artifact
  reduction logic, reward-implementation evidence, readiness decision, and JSON
  writer.
- `experiments/phase18_planning_reward_readiness/run_phase18_planning_reward_readiness.py`:
  CLI runner.
- `tests/test_phase18_planning_reward_readiness.py`: tests for blocked
  readiness, optional Phase 17 status handling, writer behavior, and CLI output.
- Documentation updates in `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`,
  and `reproducibility/FILE_MANIFEST.tsv`.

## Real Bishan Expected Result

For current real Bishan artifacts, Phase 18 should report:

- `base_planning_reward_implemented` is `false`;
- `suitability_reward_allowed` is `false`;
- `flat_full_scale_training_ready` is `false`;
- `tiled_maskableppo_api_ready` is `true` when the Phase 17 artifact is supplied;
- `performance_experiment_ready` is `false`;
- recommendation is
  `implement_real_tiled_planning_reward_before_policy_evaluation`.

This result means the real data pipeline and tiled MaskablePPO API path are
useful infrastructure, but Paper11 is not yet ready for claims about planning
performance or learned policy superiority.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: Phase 18 reuses Phase 2, Phase 10, Phase 12, and optional
  Phase 17 artifacts without changing their semantics.
- Scope check: this phase is limited to a readiness gate and does not implement
  the missing reward function.
- Ambiguity check: performance readiness, blocked reasons, expected real Bishan
  result, and claim boundary are explicit.
