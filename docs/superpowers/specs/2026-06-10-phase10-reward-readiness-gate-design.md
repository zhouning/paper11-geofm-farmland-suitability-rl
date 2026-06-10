# Phase 10 Reward-Readiness Gate Design

## Goal

Add a dedicated Phase 10 gate that converts a Phase 9
`phase9_proxy_validation_report.json` into a machine-readable decision about
whether `suitability_proxy` is ready to be used in reward experiments. The gate
should protect Paper11 from accidentally treating a weak or negatively aligned
proxy as valid suitability-reward evidence.

## Rationale

Phase 9 made suitability proxy validation explicit. On the included fixture,
both weak labels are available, but their diagnostics are
`negative_or_no_alignment`. This is useful reviewer-facing evidence, but it also
means the next engineering step should not be real suitability-reward training.

The experiment plan says suitability should be validated before it is used in a
reward. Phase 10 turns that requirement into an executable guardrail. It lets
future B2/B3 reward work depend on a small JSON decision instead of informal
interpretation.

## Approach Options

Recommended approach: build a standalone gate that reads the Phase 9 JSON
report and writes `phase10_reward_readiness_gate.json`. This keeps the
validation decision independent of Phase 2 feature assembly, Phase 8 ablation
tables, and later policy code.

Alternative 1: fold the gate into the Phase 9 report writer. This would reduce
one command, but it would mix measurement and decision logic and make it harder
to change gate thresholds without regenerating Phase 9 diagnostics.

Alternative 2: enforce the gate inside future reward-environment code only.
This would protect runtime calls, but reviewers would not get a standalone
artifact explaining why the reward is or is not ready.

## Scope

Create a focused reward-readiness gate:

```text
phase9_proxy_validation_report.json
  -> required-label check
  -> per-label eligibility check
  -> global readiness decision
  -> phase10_reward_readiness_gate.json
```

Default required labels are:

```text
stable_farmland_label
high_standard_farmland_label
```

Default thresholds are intentionally conservative:

```text
min_rank_auc = 0.5
min_mean_difference = 0.0
require_positive_interpretation = true
```

A label passes only when it is available, has both positive and negative
examples, has `interpretation == "positive_alignment"`, has rank AUC at least
`min_rank_auc`, and has mean difference greater than `min_mean_difference`.

## Gate Decisions

The top-level `status` should be one of:

- `ready_for_suitability_reward_smoke`: all required labels pass the gate;
- `not_ready_for_suitability_reward`: at least one required label is available
  but negatively aligned or below threshold;
- `insufficient_evidence`: no required label has enough binary variation for a
  meaningful check.

The top-level `recommendation` should be one of:

- `allow_bounded_suitability_reward_smoke`;
- `do_not_enable_suitability_reward`;
- `collect_or_rebuild_weak_labels_before_reward_use`.

The included fixture should produce `not_ready_for_suitability_reward` and
`do_not_enable_suitability_reward` because both available weak labels are
`negative_or_no_alignment`.

## Artifact Contents

Write `phase10_reward_readiness_gate.json` under the requested output
directory. The artifact should contain:

- `phase`: `phase10_reward_readiness_gate`;
- `phase9_report`;
- `required_labels`;
- `thresholds`;
- `status`;
- `recommendation`;
- `passing_label_count`;
- `failing_label_count`;
- `insufficient_label_count`;
- `labels`, keyed by required label;
- `reasons`;
- `claim_boundary`.

Each label entry should include:

- `available`;
- `validation_available`;
- `interpretation`;
- `rank_auc`;
- `mean_difference`;
- `positive_count`;
- `negative_count`;
- `passes_gate`;
- `reason`.

## Claim Boundary

Use an explicit Phase 10 claim boundary:

```text
Phase 10 is a reward-readiness gate for suitability_proxy; it does not train,
tune, evaluate, or report a DRL policy, and it does not prove agronomic
validity.
```

The gate only decides whether current weak-label diagnostics are strong enough
to permit later bounded suitability-reward smoke experiments. It must not claim
that AlphaEarth directly measures soil quality, fertility, irrigation, or legal
farmland quality.

## Public API

Create `src/paper11_geofm/reward_readiness.py` with:

```python
PHASE10_CLAIM_BOUNDARY = (
    "Phase 10 is a reward-readiness gate for suitability_proxy; it does not "
    "train, tune, evaluate, or report a DRL policy, and it does not prove "
    "agronomic validity."
)


def build_phase10_reward_readiness_gate(
    phase9_report_path: Path | str,
    required_labels: Sequence[str] = (
        "stable_farmland_label",
        "high_standard_farmland_label",
    ),
    min_rank_auc: float = 0.5,
    min_mean_difference: float = 0.0,
) -> dict[str, object]:
    """Return the in-memory Phase 10 gate dictionary."""


def write_phase10_reward_readiness_gate(
    gate: Mapping[str, object],
    output_dir: Path | str,
) -> Path:
    """Write phase10_reward_readiness_gate.json and return its path."""
```

The builder should validate that the Phase 9 report exists, has
`phase == "phase9_proxy_validation_report"`, and contains a `labels` object.
Invalid or incomplete Phase 9 reports should raise `ValueError` with a clear
message.

## CLI

Create:

```text
experiments/phase10_reward_readiness/run_phase10_reward_readiness.py
```

The command accepts:

- `--phase9-report`;
- `--output-dir`;
- `--required-labels`, defaulting to
  `stable_farmland_label,high_standard_farmland_label`;
- `--min-rank-auc`, defaulting to `0.5`;
- `--min-mean-difference`, defaulting to `0.0`.

It prints the gate path, status, recommendation, required-label counts, each
label reason, and the claim boundary. It returns exit code `1` for missing or
invalid Phase 9 reports.

## Documentation

Update:

- `README.md`: add a Phase 10 command after Phase 9;
- `reproducibility/REPRODUCTION_GUIDE.md`: describe the gate, expected fixture
  result, artifact file, and no-policy-performance boundary;
- `reproducibility/FILE_MANIFEST.tsv`: add the design, plan, module, CLI, and
  test file.

Use `experiments/phase10_reward_readiness/outputs/` for reviewer command
examples because generated experiment outputs are ignored by Git.

## Test Strategy

Use TDD. Add tests that:

- build Phase 2 fixture outputs, run Phase 9 in memory, and verify Phase 10
  marks the fixture `not_ready_for_suitability_reward`;
- verify a synthetic positive Phase 9 report can pass the gate;
- verify missing labels and one-class labels produce `insufficient_evidence`;
- verify invalid Phase 9 reports raise `ValueError`;
- verify JSON writing creates `phase10_reward_readiness_gate.json`;
- verify the CLI prints a concise gate summary.

All tests remain offline and CPU-only and make no policy-training,
planning-performance, or agronomic-validity claims.

