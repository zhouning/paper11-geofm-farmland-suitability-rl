# Phase 5 Rollout Protocol Smoke Design

## Goal

Add a deterministic masked-rollout smoke protocol that runs the Phase 4
Gymnasium input-contract environment across ready B0/B1/B2/B3 variants and
writes comparable protocol artifacts.

This phase verifies that all ready variants can enter the same episode protocol
without training or evaluating a policy.

## Rationale

Phase 2 creates ready variant tables, Phase 3 validates numeric DRL inputs, and
Phase 4 verifies Gymnasium reset/step/action-mask wiring for one variant. The
next gap is a reproducible multi-variant protocol layer. Without this layer,
later trained-policy results would be harder to compare because the episode
summary, action-mask accounting, reward metadata, and artifact format would not
be standardized.

## Scope

Create a lightweight rollout protocol around `Phase4InputContractEnv`:

```text
Phase 2 ready variant tables
  -> load_variant_input(...)
  -> Phase4InputContractEnv
  -> deterministic masked rollout
  -> CSV + JSON protocol artifacts
```

The rollout policy is intentionally simple:

- call `reset()`;
- at each step, read `action_masks()`;
- select the first currently valid action;
- call `step(action)`;
- stop when the environment terminates, truncates, reaches `max_steps`, or has
  no valid action.

The default variant set is `B0,B1,B2,B3`. The command should skip no ready
variant silently: if a requested variant is missing or not ready, it should fail
with the same validation clarity as Phase 3/Phase 4.

## Artifacts

Write two files under the requested output directory:

```text
phase5_rollout_summary.csv
phase5_rollout_steps.json
```

The summary CSV has one row per variant with:

- `variant_id`;
- `n_blocks`;
- `n_features`;
- `observation_shape`;
- `action_space_n`;
- `reward_mode`;
- `max_steps`;
- `episode_steps`;
- `terminated`;
- `truncated`;
- `valid_action_rate`;
- `total_contract_reward`;
- `selected_block_ids`;
- `claim_boundary`.

The step JSON records:

- protocol metadata;
- claim boundary;
- requested variants;
- one step list per variant, including `step`, `action`,
  `selected_block_id`, `reward`, `valid_actions_before`, and
  `valid_actions_after`.

## Claim Boundary

Use an explicit Phase 5 claim boundary:

```text
Phase 5 is a deterministic rollout-protocol smoke check; it does not train or
evaluate a policy and does not report planning performance.
```

The only reward reported is the Phase 4 contract reward. It is useful for
checking reward-mode wiring, not for claiming spatial optimization quality.

## Non-Goals

Do not:

- train Stable-Baselines3 or sb3-contrib policies;
- add learned policies, random policies, or greedy suitability policies;
- compute slope, contiguity, baimu-fang, compactness, transfer, or planning
  quality metrics;
- simulate land-use transitions;
- modify legacy county or parcel environments;
- frame `total_contract_reward` as model performance.

## Public API

Create `src/paper11_geofm/rollout_smoke.py` with:

```python
PHASE5_CLAIM_BOUNDARY = (
    "Phase 5 is a deterministic rollout-protocol smoke check; it does not "
    "train or evaluate a policy and does not report planning performance."
)


def run_phase5_rollout_protocol(
    phase2_output_dir: Path | str,
    variant_ids: Sequence[str] = ("B0", "B1", "B2", "B3"),
    max_steps: int | None = None,
) -> dict[str, object]:
    ...


def write_phase5_rollout_artifacts(
    protocol: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    ...
```

The protocol dictionary should be JSON-serializable and contain enough
information to write both artifacts without re-running environments.

## CLI

Create:

```text
experiments/phase5_rollout_protocol/run_phase5_rollout.py
```

The command accepts:

- `--phase2-output-dir`;
- `--output-dir`;
- `--variants`, defaulting to `B0,B1,B2,B3`;
- `--max-steps`, optional.

It prints one concise line per variant and the two artifact paths. Output should
make clear that this is a protocol smoke check, not training.

## Documentation

Update:

- `README.md`: add a Phase 5 command after Phase 4;
- `reproducibility/REPRODUCTION_GUIDE.md`: add expected Phase 5 artifact names
  and claim boundary;
- `reproducibility/FILE_MANIFEST.tsv`: add the design, plan, module, CLI, and
  test file.

## Test Strategy

Use TDD. Add tests that:

- build Phase 2 ready fixture outputs for B0/B1/B2/B3;
- run the protocol with all four variants;
- verify feature counts for the fixture variants: B0 = 17, B1 = 81,
  B2 = 18, B3 = 82;
- verify every variant has valid action rate `1.0`;
- verify B0/B1 total contract reward is `0.0`;
- verify B2/B3 total contract reward is positive for the fixture;
- verify CSV and JSON artifacts are written with the claim boundary;
- verify the CLI prints variant summaries and artifact paths.

All tests remain offline, CPU-only, deterministic, and independent of full DRL
training.
