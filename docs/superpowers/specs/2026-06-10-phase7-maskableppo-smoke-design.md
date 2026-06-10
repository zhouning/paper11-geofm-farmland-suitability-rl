# Phase 7 MaskablePPO Smoke Design

## Goal

Add a minimal `sb3-contrib` `MaskablePPO` compatibility smoke check for the
Phase 4 Gymnasium input-contract environment. The smoke check verifies that a
ready Phase 2 variant can be wrapped by the Phase 4 env, expose action masks,
initialize a MaskablePPO model, run a tiny `learn()` call, and produce a masked
prediction.

This phase checks library integration only. It does not train, tune, evaluate,
or report a useful DRL policy.

## Rationale

Phases 4-6 verify the environment contract, deterministic rollout protocol, and
non-learning masked baselines. The next integration risk is whether the same env
can be consumed by the intended masked-policy training stack. Phase 7 reduces
that risk with a tiny, deterministic compatibility smoke instead of a real
training experiment.

## Scope

Create a focused MaskablePPO smoke module:

```text
Phase 2 ready variant table
  -> Phase4InputContractEnv
  -> sb3-contrib MaskablePPO
  -> tiny learn() smoke
  -> masked predict() smoke
  -> JSON summary artifact
```

Default variant is `B3`. Default total timesteps should be very small, such as
`8`, and model hyperparameters should be selected only to keep the smoke fast
and deterministic. The smoke should run on CPU.

## Artifact

Write one JSON file under the requested output directory:

```text
phase7_maskableppo_smoke.json
```

The summary records:

- `variant_id`;
- `n_blocks`;
- `n_features`;
- `observation_shape`;
- `action_space_n`;
- `masking_supported`;
- `initial_valid_actions`;
- `learn_timesteps`;
- `predicted_action`;
- `predicted_action_valid`;
- `selected_block_id`;
- `claim_boundary`;
- dependency metadata for `stable_baselines3` and `sb3_contrib` when available.

## Claim Boundary

Use an explicit Phase 7 claim boundary:

```text
Phase 7 is a MaskablePPO compatibility smoke check; it does not train, tune,
evaluate, or report a useful DRL policy.
```

The short `learn()` call is only a library-integration exercise. Do not report
episode reward, success rate, planning metrics, or policy performance.

## Non-Goals

Do not:

- run real DRL training;
- save model checkpoints;
- tune hyperparameters;
- compute planning, transfer, slope, contiguity, baimu-fang, compactness, or
  policy-quality metrics;
- compare B0/B1/B2/B3 policy results;
- modify legacy county or parcel environments;
- frame the predicted action as a recommended planning action.

## Dependency Handling

The repository already lists `stable-baselines3` and `sb3-contrib` in
`requirements.txt`. The module should still fail clearly if either dependency is
missing, because reviewers may run a minimal environment.

Tests may use `pytest.importorskip()` for SB3-dependent checks. The CLI should
catch missing optional dependencies and return exit code `1` with a concise
error.

## Public API

Create `src/paper11_geofm/maskableppo_smoke.py` with:

```python
PHASE7_CLAIM_BOUNDARY = (
    "Phase 7 is a MaskablePPO compatibility smoke check; it does not train, "
    "tune, evaluate, or report a useful DRL policy."
)


def run_phase7_maskableppo_smoke(
    phase2_output_dir: Path | str,
    variant_id: str = "B3",
    total_timesteps: int = 8,
    seed: int = 0,
) -> dict[str, object]:
    ...


def write_phase7_maskableppo_artifact(
    summary: Mapping[str, object],
    output_dir: Path | str,
) -> Path:
    ...
```

The function should build a Phase 4 env, verify mask support with
`sb3_contrib.common.maskable.utils.is_masking_supported`, create a `MaskablePPO`
model, run `learn(total_timesteps=...)`, reset the env, call `predict()` with
`action_masks`, and record whether the predicted action is valid.

## CLI

Create:

```text
experiments/phase7_maskableppo_smoke/run_phase7_maskableppo_smoke.py
```

The command accepts:

- `--phase2-output-dir`;
- `--output-dir`;
- `--variant`, defaulting to `B3`;
- `--total-timesteps`, defaulting to `8`;
- `--seed`, defaulting to `0`.

It prints the variant, observation shape, action-space size, mask support,
predicted action validity, artifact path, and claim boundary.

## Documentation

Update:

- `README.md`: add a Phase 7 command after Phase 6;
- `reproducibility/REPRODUCTION_GUIDE.md`: add expected Phase 7 output and
  dependency note;
- `reproducibility/FILE_MANIFEST.tsv`: add the design, plan, module, CLI, and
  test file.

Use `experiments/phase7_maskableppo_smoke/outputs/` for reviewer command
examples because generated experiment outputs are ignored by Git.

## Test Strategy

Use TDD. Add tests that:

- build ready Phase 2 fixture outputs for B3;
- skip clearly if `stable_baselines3` or `sb3_contrib` is unavailable;
- verify `is_masking_supported(env)` is true;
- run `run_phase7_maskableppo_smoke(..., total_timesteps=8, seed=0)`;
- verify predicted action is within the initial valid mask;
- verify the claim boundary is present;
- verify the JSON artifact is written;
- verify the CLI prints a concise smoke summary.

All tests remain offline, CPU-only, deterministic in structure, and make no
policy-performance claims.
