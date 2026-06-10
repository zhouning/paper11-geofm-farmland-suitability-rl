# Phase 6 Masked Baseline Evaluator Design

## Goal

Add a deterministic non-learning masked baseline evaluator on top of the Phase 4
environment and Phase 5 rollout protocol shape. The evaluator compares simple
action selectors across ready B0/B1/B2/B3 variants and writes reproducible
CSV/JSON artifacts.

This phase checks baseline-evaluation plumbing only. It does not train,
optimize, or evaluate a learned DRL policy.

## Rationale

Phase 5 standardizes one deterministic rollout protocol across variants. Phase 6
adds the next missing protocol layer: multiple baseline action selectors under
the same action-mask and artifact contract. This gives later policy-training
work a stable baseline-output format without creating premature planning
performance claims.

## Scope

Create a lightweight baseline evaluator:

```text
Phase 2 ready variant tables
  -> Phase4InputContractEnv
  -> masked baseline action selector
  -> baseline summary CSV + trace JSON
```

Evaluate these policies by default:

- `first_valid`: always select the first currently valid action;
- `seeded_random`: select a random valid action using NumPy `default_rng(seed)`.

Default variants remain `B0,B1,B2,B3`. Default seed is `0`. The evaluator should
run every requested policy for every requested variant and fail clearly if a
variant is missing or not ready.

## Artifacts

Write two files under the requested output directory:

```text
phase6_baseline_summary.csv
phase6_baseline_traces.json
```

The summary CSV has one row per `(policy_id, variant_id)` pair:

- `policy_id`;
- `variant_id`;
- `seed`;
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

The trace JSON records protocol metadata plus the step list for each policy and
variant. Each step includes:

- `step`;
- `action`;
- `selected_block_id`;
- `reward`;
- `valid_actions_before`;
- `valid_actions_after`;
- `terminated`;
- `truncated`.

## Claim Boundary

Use an explicit Phase 6 claim boundary:

```text
Phase 6 is a non-learning masked baseline evaluator; it does not train or
evaluate a DRL policy and does not report planning performance.
```

The reported reward remains the Phase 4 contract reward. It checks reward-mode
wiring and policy/evaluator plumbing only.

## Non-Goals

Do not:

- train Stable-Baselines3 or sb3-contrib policies;
- add learned policies;
- add greedy suitability policies or any semantic policy that could be mistaken
  for a planning method;
- compute slope, contiguity, baimu-fang, compactness, transfer, or planning
  quality metrics;
- simulate land-use transitions;
- modify legacy county or parcel environments;
- frame policy differences as real planning performance.

## Public API

Create `src/paper11_geofm/baseline_eval.py` with:

```python
PHASE6_CLAIM_BOUNDARY = (
    "Phase 6 is a non-learning masked baseline evaluator; it does not train "
    "or evaluate a DRL policy and does not report planning performance."
)


def run_phase6_baseline_evaluator(
    phase2_output_dir: Path | str,
    variant_ids: Sequence[str] = ("B0", "B1", "B2", "B3"),
    policy_ids: Sequence[str] = ("first_valid", "seeded_random"),
    max_steps: int | None = None,
    seed: int = 0,
) -> dict[str, object]:
    ...


def write_phase6_baseline_artifacts(
    protocol: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    ...
```

The protocol dictionary must be JSON-serializable and contain all summary rows
and traces needed by the artifact writer.

## Policy Selection Rules

`first_valid`:

- reads `env.action_masks()`;
- chooses the lowest valid action index.

`seeded_random`:

- creates one deterministic RNG stream from the requested seed;
- for each `(policy_id, variant_id)` run, derives a stable child seed from the
  base seed, policy ID, and variant ID;
- chooses uniformly from currently valid action indices.

The random policy must be deterministic for the same seed and should change when
the seed changes.

## CLI

Create:

```text
experiments/phase6_masked_baselines/run_phase6_baselines.py
```

The command accepts:

- `--phase2-output-dir`;
- `--output-dir`;
- `--variants`, defaulting to `B0,B1,B2,B3`;
- `--policies`, defaulting to `first_valid,seeded_random`;
- `--max-steps`, optional;
- `--seed`, defaulting to `0`.

It prints one concise line per `(policy_id, variant_id)` summary row and the two
artifact paths. Output must state the claim boundary.

## Documentation

Update:

- `README.md`: add a Phase 6 command after Phase 5;
- `reproducibility/REPRODUCTION_GUIDE.md`: add expected Phase 6 artifact names,
  policy names, seed behavior, and claim boundary;
- `reproducibility/FILE_MANIFEST.tsv`: add the design, plan, module, CLI, and
  test file.

Use `experiments/phase6_masked_baselines/outputs/` for reviewer command examples
because generated experiment outputs are already ignored by Git.

## Test Strategy

Use TDD. Add tests that:

- build Phase 2 ready fixture outputs for B0/B1/B2/B3;
- run default policies for all four variants;
- verify summary row count is `8`;
- verify `first_valid` selects blocks in fixture order;
- verify `seeded_random` is deterministic for the same seed and changes for a
  different seed;
- verify B0/B1 contract reward is `0.0`;
- verify B2/B3 contract reward is positive for both policies;
- verify CSV and JSON artifacts are written with the claim boundary;
- verify the CLI prints policy/variant summaries and artifact paths;
- verify unknown policy IDs fail clearly.

All tests remain offline, CPU-only, deterministic, and independent of full DRL
training.
