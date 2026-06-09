# Phase 4 DRL Smoke Environment Design

## Goal

Add a minimal Gymnasium-compatible environment that consumes Phase 3
`VariantInput` objects and verifies the later-DRL input contract:

- observation flattening;
- action-space sizing;
- action-mask behavior;
- reset/step API shape;
- reward-mode metadata wiring.

This phase does not train a policy, simulate real parcel/block transitions, or
report planning performance.

## Scope

Create a lightweight environment around a loaded Phase 2 variant table:

```text
experiment_variants.json + variant_B*_features.csv
  -> load_variant_input(...)
  -> Phase4InputContractEnv
```

The environment should expose:

- `observation_space`: `Box` with shape `(n_blocks * n_features + 3,)`;
- `action_space`: `Discrete(n_blocks)`;
- `reset()`: returns a flat `float32` observation and metadata;
- `step(action)`: marks one block as selected, returns a flat observation,
  contract reward, termination flags, and metadata;
- `action_masks()`: returns `True` for blocks not yet selected.

The three appended global smoke features are:

1. budget remaining fraction;
2. step fraction;
3. valid-action fraction.

## Contract Reward

The reward is only a smoke-test signal:

- for variants with `reward_mode == "base_plus_suitability_reward"` and a
  `suitability_proxy` column, return the selected row's suitability proxy;
- otherwise return `0.0`.

This verifies that B2/B3 suitability-aware reward metadata is connected without
claiming any real reward model or DRL result.

## Non-Goals

Do not:

- modify legacy Paper3/Paper4/Paper8 environments;
- load parcel geometry;
- simulate land-use transitions;
- train Stable-Baselines3 policies;
- compute planning metrics;
- report slope, contiguity, baimu-fang, transfer, or performance results.

## Public API

Create `src/paper11_geofm/drl_smoke_env.py` with:

```python
PHASE4_CLAIM_BOUNDARY = (
    "Phase 4 is a DRL input-contract smoke environment; it does not train "
    "or evaluate a policy and does not simulate planning outcomes."
)


class Phase4InputContractEnv(gym.Env):
    ...


def make_phase4_smoke_env(
    phase2_output_dir: Path | str,
    variant_id: str,
    max_steps: int | None = None,
) -> Phase4InputContractEnv:
    ...
```

The constructor takes a `VariantInput`, validates that it has at least one block
and one feature, and defaults `max_steps` to `n_blocks`.

## CLI

Create:

```text
experiments/phase4_drl_smoke_env/run_phase4_smoke.py
```

The command should:

- load a Phase 2 output directory and requested variant;
- build `Phase4InputContractEnv`;
- run `reset()`;
- choose the first valid action from `action_masks()`;
- run one `step()`;
- print observation shape, action-space size, valid-action count, step reward,
  selected block ID, reward mode, and claim boundary.

It must not train a policy.

## Documentation

Update README, reproduction guide, and file manifest to describe Phase 4 as a
contract smoke test after Phase 3. Keep the wording explicit that no DRL
performance evidence is produced.

## Test Strategy

Use TDD. Add tests that:

- instantiate the environment from ready B3 Phase 2 fixture outputs;
- verify observation/action spaces and reset observation shape;
- verify `action_masks()` changes after a valid step;
- verify B3 contract reward reads `suitability_proxy`;
- verify B1/base reward variants return zero contract reward;
- verify out-of-range actions fail clearly;
- verify the CLI prints the expected one-step smoke summary.

All tests remain offline and CPU-only.
