# Phase 26 Main Empirical Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Phase 26 analysis package that turns Phase 25 padded held-out policy outputs into Paper11's first multi-seed, multi-held-out-tile empirical result tables.

**Architecture:** Reuse the Phase 25 runner for all RL training and add a separate Phase 26 analyzer that reads Phase 25 CSV/JSON artifacts. The analyzer computes learned-policy B1-B0 deltas by held-out tile and seed, writes manuscript-facing summaries, and assigns a conservative claim status without introducing new rewards, B2/B3 variants, or long training inside unit tests.

**Tech Stack:** Python standard library, CSV/JSON/Markdown artifact writing, pytest synthetic fixtures, existing Phase 25 padded MaskablePPO runner for optional run-and-analyze execution.

---

## File Structure

- Create: `tests/test_phase26_main_experiment.py`
  - Owns synthetic fixture tests for Phase 26 aggregation, claim status rules, artifact writing, and CLI behavior.
- Create: `src/paper11_geofm/phase26_main_experiment.py`
  - Owns Phase 25 artifact readers, validation, summary aggregation, tile-seed delta calculation, claim status assignment, and artifact writer.
- Create: `experiments/phase26_main_experiment/run_phase26_main_experiment.py`
  - Owns `analyze-only` and `run-and-analyze` CLI modes.
- Modify: `README.md`
  - Adds Phase 26 entry point and bounded empirical-claim language.
- Modify: `reproducibility/REPRODUCTION_GUIDE.md`
  - Adds Windows timing probe, Colab Pro+ main run, and analysis-only reproduction commands.
- Modify: `reproducibility/FILE_MANIFEST.tsv`
  - Adds Phase 26 spec, plan, runtime module, runner, and tests.
- Modify: `paper/submission/01_ijaeog_submission_readiness.md`
  - Adds Phase 26 as the recommended main empirical result package while keeping submission readiness guarded until real main-run outputs are available.
- Modify: `paper/submission/02_draft_titles_highlights_declarations.md`
  - Adds guarded Phase 26 result-language scaffolding.

Do not modify `src/paper11_geofm/padded_heldout_policy.py` unless a Phase 26 test exposes a genuine Phase 25 artifact schema bug. Phase 26 should analyze Phase 25 outputs rather than fork training logic.

## Shared Constants and Output Names

Use these constants in `src/paper11_geofm/phase26_main_experiment.py`:

```python
PHASE26_CLAIM_BOUNDARY = (
    "Phase 26 is a main empirical analysis package for B0/B1 padded held-out "
    "Bishan tile learned-policy results under the deterministic base planning "
    "reward; it does not enable suitability reward, does not test B2/B3, and "
    "does not support cross-region transfer or final submission-level claims."
)

PHASE26_REMAINING_EVIDENCE_GAPS = [
    "longer_budget_replication_if_1024_steps_is_used",
    "suitability_reward_validation_before_B2_B3",
    "held_out_region_transfer_evaluation",
    "spatial_case_maps_and_uncertainty",
    "submission_level_ablation_and_robustness_package",
]

MAIN_SUMMARY_FIELDNAMES = [
    "row_type",
    "variant_id",
    "eval_tile_id",
    "seed_count",
    "mean_total_contract_reward",
    "std_total_contract_reward",
    "min_total_contract_reward",
    "max_total_contract_reward",
    "train_timesteps",
    "eval_max_steps",
    "claim_boundary",
]

DELTA_FIELDNAMES = [
    "eval_tile_id",
    "seed",
    "b0_reward",
    "b1_reward",
    "b1_minus_b0_reward",
    "b1_improves_b0",
    "train_timesteps",
    "eval_max_steps",
]
```

Artifact names:

```text
phase26_main_summary.csv
phase26_tile_seed_delta_table.csv
phase26_main_comparison.json
phase26_claim_readiness.md
```

---

### Task 1: Failing Phase 26 Analysis Tests

**Files:**
- Create: `tests/test_phase26_main_experiment.py`

- [ ] **Step 1: Write synthetic Phase 25 fixture helpers**

Create `tests/test_phase26_main_experiment.py`:

```python
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_phase25_fixture(output_dir: Path, learned_delta_pattern: str = "supported") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "phase25_padded_heldout_policy_summary.csv"
    fieldnames = [
        "row_type",
        "variant_id",
        "train_tile_id",
        "eval_tile_id",
        "eval_tile_rank",
        "seed",
        "phase25_seed_rank",
        "train_timesteps",
        "eval_max_steps",
        "max_blocks",
        "train_n_blocks",
        "eval_n_blocks",
        "n_features",
        "observation_shape",
        "action_space_n",
        "episode_steps",
        "terminated",
        "truncated",
        "all_actions_valid",
        "invalid_action_count",
        "total_contract_reward",
        "selected_block_ids",
        "claim_boundary",
    ]
    rows = _phase25_rows(learned_delta_pattern)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    comparison = {
        "phase": "phase25_padded_heldout_policy_comparison",
        "train_tile_id": "tile_train",
        "train_tile_ids": ["tile_train"],
        "eval_tile_ids": ["tile_eval_a", "tile_eval_b"],
        "variants": ["B0", "B1"],
        "seeds": [0, 1],
        "seed_count": 2,
        "total_timesteps": 1024,
        "eval_max_steps": 8,
        "max_blocks": 10,
        "learned_policy": {"B1_minus_B0_mean_reward": 0.5},
        "remaining_evidence_gaps": ["suitability_reward_validation_before_B2_B3"],
    }
    (output_dir / "phase25_padded_heldout_policy_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    return output_dir
```

Add `_phase25_rows(pattern)`:

```python
def _phase25_rows(pattern: str) -> list[dict[str, object]]:
    if pattern == "supported":
        learned = {
            ("tile_eval_a", 0): (1.0, 1.4),
            ("tile_eval_a", 1): (0.8, 1.1),
            ("tile_eval_b", 0): (0.4, 0.7),
            ("tile_eval_b", 1): (0.6, 0.6),
        }
    elif pattern == "mixed":
        learned = {
            ("tile_eval_a", 0): (1.0, 1.5),
            ("tile_eval_a", 1): (1.0, 1.4),
            ("tile_eval_b", 0): (1.0, 0.8),
            ("tile_eval_b", 1): (1.0, 0.9),
        }
    elif pattern == "not_supported":
        learned = {
            ("tile_eval_a", 0): (1.0, 0.8),
            ("tile_eval_a", 1): (1.0, 0.9),
            ("tile_eval_b", 0): (1.0, 1.0),
            ("tile_eval_b", 1): (1.0, 0.7),
        }
    else:
        learned = {}

    rows: list[dict[str, object]] = []
    for tile_rank, tile_id in enumerate(["tile_eval_a", "tile_eval_b"], start=1):
        for seed_rank, seed in enumerate([0, 1], start=1):
            b0, b1 = learned.get((tile_id, seed), (1.0, 1.0))
            rows.append(_phase25_row("trained_policy", "B0", tile_id, tile_rank, seed, seed_rank, b0))
            rows.append(_phase25_row("trained_policy", "B1", tile_id, tile_rank, seed, seed_rank, b1))
            rows.append(_phase25_row("first_valid", "B0", tile_id, tile_rank, seed, seed_rank, 0.2))
            rows.append(_phase25_row("first_valid", "B1", tile_id, tile_rank, seed, seed_rank, 0.2))
            rows.append(_phase25_row("seeded_random", "B0", tile_id, tile_rank, seed, seed_rank, 0.1))
            rows.append(_phase25_row("seeded_random", "B1", tile_id, tile_rank, seed, seed_rank, 0.15))
    return rows
```

Add `_phase25_row(...)`:

```python
def _phase25_row(row_type, variant_id, eval_tile_id, eval_tile_rank, seed, seed_rank, reward):
    return {
        "row_type": row_type,
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": eval_tile_id,
        "eval_tile_rank": eval_tile_rank,
        "seed": seed,
        "phase25_seed_rank": seed_rank,
        "train_timesteps": 1024,
        "eval_max_steps": 8,
        "max_blocks": 10,
        "train_n_blocks": 10,
        "eval_n_blocks": 5,
        "n_features": 17 if variant_id == "B0" else 81,
        "observation_shape": 190,
        "action_space_n": 10,
        "episode_steps": 4,
        "terminated": True,
        "truncated": False,
        "all_actions_valid": True,
        "invalid_action_count": 0,
        "total_contract_reward": reward,
        "selected_block_ids": "b1;b2",
        "claim_boundary": "phase25 fixture",
    }
```

- [ ] **Step 2: Write aggregation test**

```python
def test_phase26_builds_main_empirical_analysis_from_phase25_outputs(tmp_path):
    from paper11_geofm.phase26_main_experiment import (
        PHASE26_CLAIM_BOUNDARY,
        build_phase26_main_empirical_analysis,
    )

    phase25_dir = _write_phase25_fixture(tmp_path / "phase25", "supported")
    analysis = build_phase26_main_empirical_analysis(phase25_dir)

    assert analysis["phase"] == "phase26_main_empirical_experiment"
    assert analysis["source_phase25"]["summary_csv"].endswith("phase25_padded_heldout_policy_summary.csv")
    assert analysis["variants"] == ["B0", "B1"]
    assert analysis["seeds"] == [0, 1]
    assert analysis["eval_tile_ids"] == ["tile_eval_a", "tile_eval_b"]
    assert analysis["train_timesteps"] == 1024
    assert analysis["eval_max_steps"] == 8
    assert analysis["learned_policy"]["B1_minus_B0_mean_reward"] == 0.25
    assert analysis["learned_policy"]["positive_tile_seed_count"] == 3
    assert analysis["learned_policy"]["total_tile_seed_count"] == 4
    assert analysis["phase26_claim_status"] == "pilot_supported"
    assert analysis["claim_boundary"] == PHASE26_CLAIM_BOUNDARY
```

- [ ] **Step 3: Write claim status rule tests**

```python
@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("supported", "pilot_supported"),
        ("mixed", "mixed"),
        ("not_supported", "not_supported"),
    ],
)
def test_phase26_claim_status_rules(tmp_path, pattern, expected):
    from paper11_geofm.phase26_main_experiment import build_phase26_main_empirical_analysis

    phase25_dir = _write_phase25_fixture(tmp_path / pattern, pattern)
    analysis = build_phase26_main_empirical_analysis(phase25_dir)

    assert analysis["phase26_claim_status"] == expected
```

Add insufficient case:

```python
def test_phase26_reports_insufficient_when_b1_rows_are_missing(tmp_path):
    from paper11_geofm.phase26_main_experiment import build_phase26_main_empirical_analysis

    phase25_dir = _write_phase25_fixture(tmp_path / "phase25", "supported")
    summary_path = phase25_dir / "phase25_padded_heldout_policy_summary.csv"
    rows = [row for row in csv.DictReader(summary_path.open("r", encoding="utf-8")) if row["variant_id"] != "B1"]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    analysis = build_phase26_main_empirical_analysis(phase25_dir)

    assert analysis["phase26_claim_status"] == "insufficient"
```

- [ ] **Step 4: Write artifact writer test**

```python
def test_phase26_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase26_main_experiment import (
        build_phase26_main_empirical_analysis,
        write_phase26_main_empirical_artifacts,
    )

    phase25_dir = _write_phase25_fixture(tmp_path / "phase25", "supported")
    analysis = build_phase26_main_empirical_analysis(phase25_dir)
    paths = write_phase26_main_empirical_artifacts(analysis, tmp_path / "outputs")

    assert paths["main_summary_csv"].name == "phase26_main_summary.csv"
    assert paths["tile_seed_delta_csv"].name == "phase26_tile_seed_delta_table.csv"
    assert paths["comparison_json"].name == "phase26_main_comparison.json"
    assert paths["claim_readiness_md"].name == "phase26_claim_readiness.md"
    delta_rows = list(csv.DictReader(paths["tile_seed_delta_csv"].open("r", encoding="utf-8")))
    assert delta_rows[0]["eval_tile_id"] == "tile_eval_a"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase26_claim_status"] == "pilot_supported"
    markdown = paths["claim_readiness_md"].read_text(encoding="utf-8")
    assert "pilot_supported" in markdown
    assert "suitability reward" in markdown
```

- [ ] **Step 5: Write CLI tests**

```python
def test_phase26_cli_analyze_only_writes_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase26_main_experiment"
        / "run_phase26_main_experiment.py"
    )
    spec = importlib.util.spec_from_file_location("phase26_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    phase25_dir = _write_phase25_fixture(tmp_path / "phase25", "supported")
    exit_code = module.main(
        [
            "--mode",
            "analyze-only",
            "--phase25-output-dir",
            str(phase25_dir),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Phase 26 claim status: pilot_supported" in stdout
    assert "B1-B0 learned-policy mean reward delta: 0.25" in stdout
    assert "phase26_main_comparison.json" in stdout
```

Add run-and-analyze validation test:

```python
def test_phase26_cli_run_and_analyze_requires_phase25_run_inputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase26_main_experiment"
        / "run_phase26_main_experiment.py"
    )
    spec = importlib.util.spec_from_file_location("phase26_runner_validation", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--mode",
            "run-and-analyze",
            "--phase25-output-dir",
            str(tmp_path / "phase25"),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert "run-and-analyze requires" in stderr
```

- [ ] **Step 6: Verify RED**

Run:

```powershell
python -m pytest tests\test_phase26_main_experiment.py -q
```

Expected: FAIL because `paper11_geofm.phase26_main_experiment` and the runner do not exist.

- [ ] **Step 7: Commit RED tests**

Run:

```powershell
git add tests\test_phase26_main_experiment.py
git commit -m "test: add Phase 26 main empirical analysis tests"
```

### Task 2: Phase 26 Analysis Module

**Files:**
- Create: `src/paper11_geofm/phase26_main_experiment.py`
- Test: `tests/test_phase26_main_experiment.py`

- [ ] **Step 1: Implement constants, field names, and entry function**

Create `src/paper11_geofm/phase26_main_experiment.py` with the constants from this plan and:

```python
from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from statistics import pstdev


def build_phase26_main_empirical_analysis(phase25_output_dir: Path | str) -> dict[str, object]:
    output_dir = Path(phase25_output_dir)
    summary_path = output_dir / "phase25_padded_heldout_policy_summary.csv"
    comparison_path = output_dir / "phase25_padded_heldout_policy_comparison.json"
    rows = _read_summary_rows(summary_path)
    comparison = _read_json_object(comparison_path)
```

The returned dict must include:

```python
{
    "phase": "phase26_main_empirical_experiment",
    "source_phase25": {
        "summary_csv": str(summary_path),
        "comparison_json": str(comparison_path),
    },
    "train_tile_id": ...,
    "eval_tile_ids": ...,
    "variants": ...,
    "seeds": ...,
    "train_timesteps": ...,
    "eval_max_steps": ...,
    "main_summary_rows": ...,
    "tile_seed_delta_rows": ...,
    "learned_policy": ...,
    "baselines": ...,
    "phase26_claim_status": ...,
    "remaining_evidence_gaps": ...,
    "claim_boundary": PHASE26_CLAIM_BOUNDARY,
}
```

- [ ] **Step 2: Implement readers and numeric helpers**

Add:

```python
def _read_summary_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 25 summary CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 25 comparison JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Phase 25 comparison JSON must be an object")
    return value


def _float_value(row: Mapping[str, object], field: str) -> float:
    return float(str(row.get(field, "0")).strip())


def _int_value(row: Mapping[str, object], field: str) -> int:
    return int(str(row.get(field, "0")).strip())


def _round_float(value: float) -> float:
    return round(float(value), 10)
```

- [ ] **Step 3: Implement summary aggregation**

Create `_main_summary_rows(rows)` that groups by `row_type`, `variant_id`, and `eval_tile_id`.

For each group return:

```python
{
    "row_type": row_type,
    "variant_id": variant_id,
    "eval_tile_id": eval_tile_id,
    "seed_count": len(unique_seeds),
    "mean_total_contract_reward": ...,
    "std_total_contract_reward": pstdev(values) if len(values) > 1 else 0.0,
    "min_total_contract_reward": min(values),
    "max_total_contract_reward": max(values),
    "train_timesteps": train_timesteps,
    "eval_max_steps": eval_max_steps,
    "claim_boundary": PHASE26_CLAIM_BOUNDARY,
}
```

- [ ] **Step 4: Implement learned-policy tile-seed deltas**

Create `_tile_seed_delta_rows(rows)` that uses only `row_type == "trained_policy"`.

For each `(eval_tile_id, seed)`, find B0 and B1 rewards. If either is missing, skip that row and let claim status become `insufficient`.

Return rows:

```python
{
    "eval_tile_id": tile_id,
    "seed": seed,
    "b0_reward": b0,
    "b1_reward": b1,
    "b1_minus_b0_reward": delta,
    "b1_improves_b0": delta > 0,
    "train_timesteps": train_timesteps,
    "eval_max_steps": eval_max_steps,
}
```

- [ ] **Step 5: Implement learned and baseline summaries**

Create:

```python
def _learned_policy_summary(delta_rows: list[dict[str, object]]) -> dict[str, object]:
```

Return:

```python
{
    "B1_minus_B0_mean_reward": mean_delta_or_none,
    "B1_minus_B0_std_reward": std_delta_or_none,
    "positive_tile_seed_count": positive_count,
    "total_tile_seed_count": total_count,
    "positive_fraction": positive_count / total_count if total_count else None,
    "per_tile_mean_delta": {...},
    "per_seed_mean_delta": {...},
}
```

Create `_baseline_summaries(rows)` for `first_valid` and `seeded_random`, computing mean reward by variant and B1-B0 mean deltas where possible.

- [ ] **Step 6: Implement claim status**

Add:

```python
def _phase26_claim_status(learned: Mapping[str, object], expected_total: int) -> str:
    total = int(learned.get("total_tile_seed_count", 0))
    delta = learned.get("B1_minus_B0_mean_reward")
    positive_fraction = learned.get("positive_fraction")
    if total <= 0 or total < expected_total or delta is None or positive_fraction is None:
        return "insufficient"
    if float(delta) <= 0:
        return "not_supported"
    if float(positive_fraction) >= 0.6:
        return "pilot_supported"
    return "mixed"
```

Compute `expected_total = len(eval_tile_ids) * len(seeds)`.

- [ ] **Step 7: Verify module tests GREEN except CLI**

Run:

```powershell
python -m pytest tests\test_phase26_main_experiment.py -q -k "not cli"
```

Expected: PASS for module-level tests; CLI tests still fail because runner does not exist.

- [ ] **Step 8: Commit module**

Run:

```powershell
git add src\paper11_geofm\phase26_main_experiment.py tests\test_phase26_main_experiment.py
git commit -m "feat: add Phase 26 main empirical analysis module"
```

### Task 3: Phase 26 Artifact Writer

**Files:**
- Modify: `src/paper11_geofm/phase26_main_experiment.py`
- Test: `tests/test_phase26_main_experiment.py`

- [ ] **Step 1: Implement writer**

Add:

```python
def write_phase26_main_empirical_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    main_summary_path = output_path / "phase26_main_summary.csv"
    delta_path = output_path / "phase26_tile_seed_delta_table.csv"
    comparison_path = output_path / "phase26_main_comparison.json"
    markdown_path = output_path / "phase26_claim_readiness.md"
```

Write `analysis["main_summary_rows"]` with `MAIN_SUMMARY_FIELDNAMES`.

Write `analysis["tile_seed_delta_rows"]` with `DELTA_FIELDNAMES`.

Write full `analysis` to JSON.

Write markdown from `_claim_readiness_markdown(analysis)`.

Return:

```python
{
    "main_summary_csv": main_summary_path,
    "tile_seed_delta_csv": delta_path,
    "comparison_json": comparison_path,
    "claim_readiness_md": markdown_path,
}
```

- [ ] **Step 2: Implement markdown renderer**

Add:

```python
def _claim_readiness_markdown(analysis: Mapping[str, object]) -> str:
    learned = analysis.get("learned_policy", {})
    return "\n".join([...])
```

The markdown must include:

- title `# Phase 26 Claim Readiness`;
- `phase26_claim_status`;
- B1-B0 mean delta;
- positive tile-seed count and total count;
- evidence boundary text;
- remaining evidence gaps.

- [ ] **Step 3: Verify writer test**

Run:

```powershell
python -m pytest tests\test_phase26_main_experiment.py::test_phase26_writer_outputs_csv_json_and_markdown -q
```

Expected: PASS.

- [ ] **Step 4: Commit writer**

Run:

```powershell
git add src\paper11_geofm\phase26_main_experiment.py tests\test_phase26_main_experiment.py
git commit -m "feat: add Phase 26 empirical artifact writer"
```

### Task 4: Phase 26 CLI

**Files:**
- Create: `experiments/phase26_main_experiment/run_phase26_main_experiment.py`
- Test: `tests/test_phase26_main_experiment.py`

- [ ] **Step 1: Implement CLI imports and parser**

Create:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase26_main_experiment import (
    build_phase26_main_empirical_analysis,
    write_phase26_main_empirical_artifacts,
)
from paper11_geofm.padded_heldout_policy import (
    run_phase25_padded_heldout_policy,
    write_phase25_padded_heldout_policy_artifacts,
)
```

Parser args:

```python
parser.add_argument("--mode", choices=("analyze-only", "run-and-analyze"), default="analyze-only")
parser.add_argument("--phase25-output-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--phase2-output-dir", type=Path, default=None)
parser.add_argument("--tile-index-csv", type=Path, default=None)
parser.add_argument("--variants", default="B0,B1")
parser.add_argument("--train-tile-id", default=None)
parser.add_argument("--eval-tile-ids", default=None)
parser.add_argument("--max-eval-tiles", type=int, default=3)
parser.add_argument("--total-timesteps", type=int, default=1024)
parser.add_argument("--eval-max-steps", type=int, default=8)
parser.add_argument("--seeds", default="0,1,2")
```

- [ ] **Step 2: Implement `run-and-analyze` validation and execution**

Add:

```python
def _validate_run_and_analyze_args(args) -> None:
    missing = []
    if args.phase2_output_dir is None:
        missing.append("--phase2-output-dir")
    if args.tile_index_csv is None:
        missing.append("--tile-index-csv")
    if missing:
        raise ValueError("run-and-analyze requires " + ", ".join(missing))
```

When mode is `run-and-analyze`:

1. Validate args.
2. Call `run_phase25_padded_heldout_policy(...)`.
3. Write Phase 25 artifacts into `args.phase25_output_dir`.
4. Then analyze that directory.

When mode is `analyze-only`, just analyze `args.phase25_output_dir`.

- [ ] **Step 3: Print summary**

Print:

```python
print(f"Mode: {args.mode}")
print(f"Phase 26 claim status: {analysis['phase26_claim_status']}")
print(f"B1-B0 learned-policy mean reward delta: {analysis['learned_policy']['B1_minus_B0_mean_reward']}")
print(f"Positive tile-seed count: {analysis['learned_policy']['positive_tile_seed_count']} / {analysis['learned_policy']['total_tile_seed_count']}")
print(f"Main summary CSV: {paths['main_summary_csv']}")
print(f"Tile-seed delta CSV: {paths['tile_seed_delta_csv']}")
print(f"Comparison JSON: {paths['comparison_json']}")
print(f"Claim readiness Markdown: {paths['claim_readiness_md']}")
```

Catch `FileNotFoundError`, `RuntimeError`, and `ValueError`; print `Error: ...` to stderr and return `1`.

- [ ] **Step 4: Verify CLI tests**

Run:

```powershell
python -m pytest tests\test_phase26_main_experiment.py::test_phase26_cli_analyze_only_writes_outputs tests\test_phase26_main_experiment.py::test_phase26_cli_run_and_analyze_requires_phase25_run_inputs -q
```

Expected: PASS.

- [ ] **Step 5: Commit CLI**

Run:

```powershell
git add experiments\phase26_main_experiment\run_phase26_main_experiment.py tests\test_phase26_main_experiment.py
git commit -m "feat: add Phase 26 main empirical CLI"
```

### Task 5: Windows Timing Probe and Artifact Check

**Files:**
- Generated ignored artifacts under `experiments/phase26_main_experiment/outputs/windows_timing_probe/`

- [ ] **Step 1: Run Windows timing probe if real Phase 11/13 outputs exist**

Run:

```powershell
python experiments\phase26_main_experiment\run_phase26_main_experiment.py --mode run-and-analyze --phase25-output-dir experiments\phase26_main_experiment\outputs\windows_timing_probe\phase25_run --output-dir experiments\phase26_main_experiment\outputs\windows_timing_probe\phase26_analysis --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --total-timesteps 128 --eval-max-steps 4 --seeds 0 --max-eval-tiles 1
```

Expected:

- Phase 25 artifacts are written under `phase25_run`;
- Phase 26 artifacts are written under `phase26_analysis`;
- claim status is one of `pilot_supported`, `mixed`, `not_supported`, or
  `insufficient`;
- output does not claim Colab main-run completion.

If real Phase 11/13 outputs are not present in the local checkout, skip this
step and document that only synthetic tests were run.

- [ ] **Step 2: Inspect comparison JSON**

Run:

```powershell
Get-Content -Raw experiments\phase26_main_experiment\outputs\windows_timing_probe\phase26_analysis\phase26_main_comparison.json
```

Expected:

- `phase` is `phase26_main_empirical_experiment`;
- `learned_policy.total_tile_seed_count` is positive;
- `remaining_evidence_gaps` includes `suitability_reward_validation_before_B2_B3`;
- `claim_boundary` does not support B2/B3, suitability reward, or cross-region transfer.

### Task 6: Documentation Updates

**Files:**
- Modify: `README.md`
- Modify: `reproducibility/REPRODUCTION_GUIDE.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `paper/submission/01_ijaeog_submission_readiness.md`
- Modify: `paper/submission/02_draft_titles_highlights_declarations.md`

- [ ] **Step 1: Update README**

Add Phase 26 to repository layout and key entry points:

```text
experiments/phase26_main_experiment/: executable Phase 26 main empirical analysis runner.
src/paper11_geofm/phase26_main_experiment.py: Phase 26 analysis package for Phase 25 multi-seed, multi-held-out-tile outputs.
```

Add a short Phase 26 section:

```text
Phase 26 analyzes Phase 25 padded held-out B0/B1 outputs into manuscript-facing empirical tables. It reports B1-B0 learned-policy deltas by held-out tile and seed, assigns a conservative claim status, and keeps suitability reward, B2/B3, and cross-region transfer out of scope.
```

- [ ] **Step 2: Update reproduction guide**

Add a section after Phase 25:

```text
## 28. Run the Phase 26 Main Empirical Analysis Package
```

Include:

- Windows timing probe command from Task 5;
- Colab Pro+ Phase 25 main-run command from the Phase 26 spec;
- analysis-only command:

```powershell
python experiments\phase26_main_experiment\run_phase26_main_experiment.py --mode analyze-only --phase25-output-dir experiments\phase26_main_experiment\outputs\colab_main\phase25_run --output-dir experiments\phase26_main_experiment\outputs\colab_main\phase26_analysis
```

State that Phase 26 unit tests use synthetic fixtures and do not run long RL
training.

- [ ] **Step 3: Update file manifest**

Add rows:

```text
docs/superpowers/specs/2026-06-17-phase26-main-empirical-experiment-design.md	design	Phase 26 design for the B0/B1 padded held-out multi-seed, multi-tile main empirical analysis package.
docs/superpowers/plans/2026-06-17-phase26-main-empirical-experiment.md	plan	Implementation plan for the Phase 26 main empirical analysis package.
src/paper11_geofm/phase26_main_experiment.py	runtime	Phase 26 analyzer for Phase 25 outputs, tile-seed B1-B0 deltas, claim status, and manuscript-facing artifacts.
experiments/phase26_main_experiment/run_phase26_main_experiment.py	experiment	Executable Phase 26 analyze-only and run-and-analyze runner for main empirical artifacts.
tests/test_phase26_main_experiment.py	verification	Pytest checks for Phase 26 aggregation, claim status rules, artifact writing, and CLI behavior.
```

- [ ] **Step 4: Update submission docs**

In `paper/submission/01_ijaeog_submission_readiness.md`, add Phase 26 as the
recommended main empirical package. State that submission readiness remains
dependent on actual Colab/main-run outputs and figures.

In `paper/submission/02_draft_titles_highlights_declarations.md`, add guarded
language:

```text
- A main empirical analysis reports B1-B0 learned-policy deltas across held-out Bishan tiles and random seeds under a deterministic base planning reward.
```

Do not claim the deltas are positive unless real Phase 26 main-run artifacts
support that statement.

- [ ] **Step 5: Verify docs**

Run:

```powershell
git diff --check
python scripts\smoke_check.py
```

Expected: PASS.

- [ ] **Step 6: Commit docs**

Run:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv paper\submission\01_ijaeog_submission_readiness.md paper\submission\02_draft_titles_highlights_declarations.md docs\superpowers\specs\2026-06-17-phase26-main-empirical-experiment-design.md docs\superpowers\plans\2026-06-17-phase26-main-empirical-experiment.md
git commit -m "docs: add Phase 26 main empirical analysis guidance"
```

### Task 7: Final Verification, Merge, and Push

**Files:**
- Stage only Phase 26 source, runner, tests, docs, spec, and plan files.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests\test_phase26_main_experiment.py -q
```

Expected: PASS.

- [ ] **Step 2: Run repository verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests -q
git diff --check
git status --short --ignored=matching experiments\phase26_main_experiment
```

Expected:

- smoke check passes;
- full pytest passes;
- no whitespace errors;
- generated Phase 26 outputs are ignored.

- [ ] **Step 3: Commit any remaining tracked changes**

If there are remaining tracked changes:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv paper\submission\01_ijaeog_submission_readiness.md paper\submission\02_draft_titles_highlights_declarations.md docs\superpowers\specs\2026-06-17-phase26-main-empirical-experiment-design.md docs\superpowers\plans\2026-06-17-phase26-main-empirical-experiment.md src\paper11_geofm\phase26_main_experiment.py experiments\phase26_main_experiment\run_phase26_main_experiment.py tests\test_phase26_main_experiment.py
git commit -m "Add Phase 26 main empirical analysis package"
```

- [ ] **Step 4: Push**

Run:

```powershell
git push
```

Expected: `main -> main`.

## Plan Self-Review

- Spec coverage: The plan covers Phase 25 artifact ingestion, tile-seed B1-B0
  delta analysis, claim status rules, all four Phase 26 outputs, analyze-only
  and run-and-analyze CLI modes, Windows timing probe, Colab main-run recipe,
  docs, tests, and final verification.
- Placeholder scan: no unresolved placeholder instructions remain.
- Type consistency: Function names, artifact names, CLI flags, claim status
  strings, and CSV field names are consistent across tasks.
- Scope check: Phase 26 stays focused on B0/B1 `base_planning_reward`
  empirical analysis and does not add B2/B3, suitability reward, or cross-region
  transfer claims.
