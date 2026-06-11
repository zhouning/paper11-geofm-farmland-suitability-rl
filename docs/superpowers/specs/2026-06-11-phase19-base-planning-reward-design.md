# Phase 19 Base Planning Reward Design

## Goal

Implement a minimal, explainable `base_planning_reward` so B0/B1 real tiled
experiments no longer use a constant zero reward.

## Problem

Phase 18 showed that Paper11 cannot start true planning-performance DRL
experiments because B0/B1 use `base_planning_reward`, while the current Phase 4
environment returns `0.0` for all non-suitability reward modes. Phase 17 proved
that MaskablePPO can call the real tiled environment API, but the learned signal
is still empty.

Phase 19 should remove only that blocker. It should not enable B2/B3 suitability
reward, claim trained-policy quality, or report planning performance.

## Reward Definition

The base planning reward is a weighted, bounded, per-selected-block score built
from explicit planning features already exported by Phase 11:

- positive contribution from low-slope farmland/orchard indicators;
- positive contribution from current farmland/orchard indicators;
- mild positive contribution from area;
- negative contribution from mean and maximum slope;
- negative contribution from built-up and water indicators.

The default formula is:

```text
reward =
  0.35 * explicit_feature_16
+ 0.20 * max(explicit_feature_04, explicit_feature_07)
+ 0.10 * explicit_feature_13
+ 0.10 * clipped_area_score
- 0.15 * clipped_mean_slope_score
- 0.05 * clipped_max_slope_score
- 0.10 * explicit_feature_09
- 0.10 * explicit_feature_10
```

Where:

- `explicit_feature_16` is low-slope farmland-or-orchard;
- `explicit_feature_04` is current farmland;
- `explicit_feature_07` is orchard;
- `explicit_feature_13` is low-slope;
- `explicit_feature_00` is area in hectares;
- `explicit_feature_01` is mean slope;
- `explicit_feature_02` is maximum slope;
- `explicit_feature_09` is built-up;
- `explicit_feature_10` is water;
- area is clipped to `[0, 1]` by dividing hectares by `5.0`;
- mean slope is clipped to `[0, 1]` by dividing degrees by `25.0`;
- maximum slope is clipped to `[0, 1]` by dividing degrees by `35.0`.

This is a first executable base reward, not a final calibrated policy objective.
It is intentionally simple, deterministic, and auditable.

## Behavior

Phase 19 changes `Phase4InputContractEnv._contract_reward()` so:

1. `base_planning_reward` calls the explicit-feature reward function.
2. `base_plus_suitability_reward` keeps the existing suitability proxy term and
   adds the base planning reward when explicit features are available.
3. Missing explicit feature columns produce a `ValueError` for
   `base_planning_reward`, instead of silently returning `0.0`.
4. The reward readiness evidence in Phase 18 detects the base planning reward as
   implemented once the zero fallback is no longer used for base reward modes.

Suitability reward variants remain disabled by tiled loaders unless explicitly
allowed by existing guardrails.

## Outputs

Phase 19 should produce:

- a new `src/paper11_geofm/planning_reward.py` module;
- updated Phase 4 reward behavior;
- updated Phase 18 readiness evidence;
- a real Bishan Phase 14/18 check showing B1 base reward is non-zero and
  `base_planning_reward_implemented` is `true`;
- documentation describing the reward as a bounded first implementation, not a
  policy-performance result.

## Claim Boundary

Phase 19 implements a first base planning reward and updates readiness evidence.
It does not train, tune, evaluate, or compare a DRL policy. It does not enable
the suitability reward gate. It does not report planning performance or learned
policy superiority.

## Implementation Units

- `src/paper11_geofm/planning_reward.py`: pure reward formula, feature lookup,
  clipped scoring helpers, and explanation metadata.
- `src/paper11_geofm/drl_smoke_env.py`: calls the reward function for
  `base_planning_reward` and composes it with suitability proxy for
  `base_plus_suitability_reward`.
- `src/paper11_geofm/planning_reward_readiness.py`: replaces brittle source
  inspection with explicit reward metadata.
- `tests/test_phase19_base_planning_reward.py`: unit tests for reward formula
  and environment reward behavior.
- Updates to Phase 18 tests to expect implemented base reward once Phase 19 is
  active.
- Documentation updates in `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`,
  and `reproducibility/FILE_MANIFEST.tsv`.

## Real Bishan Expected Result

For the current real Bishan largest B1 tile, Phase 14 should report a non-zero
base planning reward for the first selected block. Phase 18 should then report:

- `base_planning_reward_implemented` is `true`;
- `tiled_maskableppo_api_ready` remains `true`;
- `suitability_reward_allowed` remains `false`;
- `flat_full_scale_training_ready` remains `false`;
- `performance_experiment_ready` remains `false` until a bounded training
  protocol and evaluation gate are added.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: Phase 19 directly addresses the Phase 18 blocker without
  changing the Phase 10 suitability gate.
- Scope check: this phase implements the base reward only; it does not add full
  training or policy evaluation.
- Ambiguity check: formula, feature mapping, missing-feature behavior, and claim
  boundary are explicit.
