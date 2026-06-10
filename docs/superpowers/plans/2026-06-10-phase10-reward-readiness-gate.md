# Phase 10 Reward-Readiness Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Phase 10 gate that reads a Phase 9 proxy-validation report and decides whether `suitability_proxy` is ready for later bounded suitability-reward smoke experiments.

**Architecture:** Add `paper11_geofm.reward_readiness` to load `phase9_proxy_validation_report.json`, evaluate required labels against explicit thresholds, and write `phase10_reward_readiness_gate.json`. Add a CLI under `experiments/phase10_reward_readiness/` and document the reviewer path after Phase 9.

**Tech Stack:** Python, JSON artifacts, argparse, pytest.

---

## File Structure

- Create `src/paper11_geofm/reward_readiness.py`: Phase 10 claim boundary, Phase 9 report loader, per-label gate evaluator, global status reducer, JSON writer.
- Create `experiments/phase10_reward_readiness/run_phase10_reward_readiness.py`: CLI runner.
- Create `tests/test_phase10_reward_readiness.py`: fixture not-ready, synthetic ready, insufficient-evidence, invalid-report, writer, and CLI tests.
- Modify `README.md`: add Phase 10 quick-start command and entry point.
- Modify `reproducibility/REPRODUCTION_GUIDE.md`: add Phase 10 reviewer step and executable file list.
- Modify `reproducibility/FILE_MANIFEST.tsv`: add Phase 10 design, plan, runtime module, CLI, and tests.

## Task 1: Fixture Not-Ready Gate Contract

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase10_reward_readiness.py`:

```python
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _run_phase2_fixture(output_dir: Path) -> Path:
    runner_path = (
        ROOT / "experiments" / "phase2_block_geofm_features" / "run_phase2.py"
    )
    spec = importlib.util.spec_from_file_location("phase2_runner_phase10", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    fixture_dir = ROOT / "data" / "bishan_phase2_csv_sample"
    exit_code = module.main(
        [
            "--mapping-csv",
            str(fixture_dir / "block_pixel_mapping.csv"),
            "--attributes-csv",
            str(fixture_dir / "block_attributes.csv"),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    return output_dir


def _write_phase9_fixture_report(tmp_path: Path) -> Path:
    from paper11_geofm.proxy_validation import (
        build_phase9_proxy_validation_report,
        write_phase9_proxy_validation_report,
    )

    phase2_dir = _run_phase2_fixture(tmp_path / "phase2")
    report = build_phase9_proxy_validation_report(phase2_dir)
    return write_phase9_proxy_validation_report(report, tmp_path / "phase9")


def test_phase10_marks_fixture_not_ready_for_suitability_reward(tmp_path):
    from paper11_geofm.reward_readiness import (
        PHASE10_CLAIM_BOUNDARY,
        build_phase10_reward_readiness_gate,
    )

    phase9_report_path = _write_phase9_fixture_report(tmp_path)

    gate = build_phase10_reward_readiness_gate(phase9_report_path)

    assert gate["phase"] == "phase10_reward_readiness_gate"
    assert gate["phase9_report"] == str(phase9_report_path)
    assert gate["required_labels"] == [
        "stable_farmland_label",
        "high_standard_farmland_label",
    ]
    assert gate["status"] == "not_ready_for_suitability_reward"
    assert gate["recommendation"] == "do_not_enable_suitability_reward"
    assert gate["passing_label_count"] == 0
    assert gate["failing_label_count"] == 2
    assert gate["insufficient_label_count"] == 0
    assert gate["labels"]["stable_farmland_label"]["passes_gate"] is False
    assert gate["labels"]["stable_farmland_label"]["interpretation"] == (
        "negative_or_no_alignment"
    )
    assert gate["labels"]["high_standard_farmland_label"]["passes_gate"] is False
    assert "failed suitability proxy alignment gate" in gate["reasons"][0]
    assert gate["claim_boundary"] == PHASE10_CLAIM_BOUNDARY
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```powershell
python -m pytest tests\test_phase10_reward_readiness.py::test_phase10_marks_fixture_not_ready_for_suitability_reward -q
```

Expected result: fail with `ModuleNotFoundError: No module named 'paper11_geofm.reward_readiness'`.

## Task 2: Gate Builder Implementation

- [ ] **Step 1: Create `src/paper11_geofm/reward_readiness.py`**

Implement:

```python
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path


PHASE10_CLAIM_BOUNDARY = (
    "Phase 10 is a reward-readiness gate for suitability_proxy; it does not "
    "train, tune, evaluate, or report a DRL policy, and it does not prove "
    "agronomic validity."
)
DEFAULT_REQUIRED_LABELS = (
    "stable_farmland_label",
    "high_standard_farmland_label",
)


def build_phase10_reward_readiness_gate(
    phase9_report_path: Path | str,
    required_labels: Sequence[str] = DEFAULT_REQUIRED_LABELS,
    min_rank_auc: float = 0.5,
    min_mean_difference: float = 0.0,
) -> dict[str, object]:
    report_path = Path(phase9_report_path)
    report = _load_phase9_report(report_path)
    labels = report["labels"]
    requested = [str(label) for label in required_labels]
    thresholds = {
        "min_rank_auc": float(min_rank_auc),
        "min_mean_difference": float(min_mean_difference),
        "require_positive_interpretation": True,
    }
    label_results = {
        label: _evaluate_label(
            label,
            labels.get(label),
            min_rank_auc=float(min_rank_auc),
            min_mean_difference=float(min_mean_difference),
        )
        for label in requested
    }
    status, recommendation, reasons = _reduce_gate(label_results)
    return {
        "phase": "phase10_reward_readiness_gate",
        "phase9_report": str(report_path),
        "required_labels": requested,
        "thresholds": thresholds,
        "status": status,
        "recommendation": recommendation,
        "passing_label_count": sum(
            1 for result in label_results.values() if result["passes_gate"]
        ),
        "failing_label_count": sum(
            1 for result in label_results.values() if result["category"] == "failing"
        ),
        "insufficient_label_count": sum(
            1
            for result in label_results.values()
            if result["category"] == "insufficient"
        ),
        "labels": label_results,
        "reasons": reasons,
        "claim_boundary": PHASE10_CLAIM_BOUNDARY,
    }


def write_phase10_reward_readiness_gate(
    gate: Mapping[str, object],
    output_dir: Path | str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    gate_path = output_path / "phase10_reward_readiness_gate.json"
    gate_path.write_text(
        json.dumps(dict(gate), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return gate_path
```

- [ ] **Step 2: Implement helpers**

Required helper behavior:

- `_load_phase9_report(path)` raises `FileNotFoundError(f"Missing Phase 9 proxy-validation report: {path}")` when absent.
- `_load_phase9_report(path)` raises `ValueError("Phase 10 requires a Phase 9 proxy-validation report")` unless `phase` equals `phase9_proxy_validation_report`.
- `_load_phase9_report(path)` raises `ValueError("Phase 9 report is missing labels")` unless `labels` is a mapping.
- `_evaluate_label(label, payload, min_rank_auc, min_mean_difference)` returns a label result with `category` equal to `passing`, `failing`, or `insufficient`.
- A label passes only when `validation_available is True`, `interpretation == "positive_alignment"`, rank AUC is at least `min_rank_auc`, and mean difference is greater than `min_mean_difference`.
- Missing labels, `label_unavailable`, and `insufficient_label_variation` produce `category == "insufficient"`.
- Negative alignment or below-threshold metrics produce `category == "failing"`.
- `_reduce_gate(label_results)` returns `ready_for_suitability_reward_smoke` only when every required label passes.
- `_reduce_gate(label_results)` returns `insufficient_evidence` when all required labels are insufficient.
- `_reduce_gate(label_results)` returns `not_ready_for_suitability_reward` when any required label fails.

- [ ] **Step 3: Run the focused test**

Run:

```powershell
python -m pytest tests\test_phase10_reward_readiness.py::test_phase10_marks_fixture_not_ready_for_suitability_reward -q
```

Expected result: pass.

## Task 3: Ready, Insufficient, Invalid, and Writer Tests

- [ ] **Step 1: Add synthetic ready and insufficient tests**

Append tests to `tests/test_phase10_reward_readiness.py`:

```python
def _write_report(path: Path, labels: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "phase9_proxy_validation_report",
        "labels": labels,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _positive_label() -> dict[str, object]:
    return {
        "validation_available": True,
        "interpretation": "positive_alignment",
        "rank_auc": 0.75,
        "mean_difference": 0.2,
        "positive_count": 3,
        "negative_count": 3,
    }


def test_phase10_marks_positive_synthetic_report_ready(tmp_path):
    from paper11_geofm.reward_readiness import build_phase10_reward_readiness_gate

    report_path = _write_report(
        tmp_path / "phase9" / "phase9_proxy_validation_report.json",
        {
            "stable_farmland_label": _positive_label(),
            "high_standard_farmland_label": _positive_label(),
        },
    )

    gate = build_phase10_reward_readiness_gate(report_path)

    assert gate["status"] == "ready_for_suitability_reward_smoke"
    assert gate["recommendation"] == "allow_bounded_suitability_reward_smoke"
    assert gate["passing_label_count"] == 2
    assert gate["failing_label_count"] == 0
    assert gate["insufficient_label_count"] == 0


def test_phase10_marks_missing_and_one_class_labels_insufficient(tmp_path):
    from paper11_geofm.reward_readiness import build_phase10_reward_readiness_gate

    report_path = _write_report(
        tmp_path / "phase9" / "phase9_proxy_validation_report.json",
        {
            "stable_farmland_label": {
                "validation_available": False,
                "interpretation": "insufficient_label_variation",
                "rank_auc": None,
                "mean_difference": None,
                "positive_count": 4,
                "negative_count": 0,
            }
        },
    )

    gate = build_phase10_reward_readiness_gate(report_path)

    assert gate["status"] == "insufficient_evidence"
    assert gate["recommendation"] == "collect_or_rebuild_weak_labels_before_reward_use"
    assert gate["insufficient_label_count"] == 2
    assert gate["labels"]["high_standard_farmland_label"]["available"] is False
```

- [ ] **Step 2: Add invalid-report and writer tests**

Append:

```python
def test_phase10_invalid_phase9_report_raises(tmp_path):
    from paper11_geofm.reward_readiness import build_phase10_reward_readiness_gate

    report_path = tmp_path / "bad_report.json"
    report_path.write_text(json.dumps({"phase": "wrong", "labels": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="Phase 9 proxy-validation report"):
        build_phase10_reward_readiness_gate(report_path)


def test_phase10_reward_readiness_gate_is_written(tmp_path):
    from paper11_geofm.reward_readiness import (
        build_phase10_reward_readiness_gate,
        write_phase10_reward_readiness_gate,
    )

    report_path = _write_report(
        tmp_path / "phase9" / "phase9_proxy_validation_report.json",
        {
            "stable_farmland_label": _positive_label(),
            "high_standard_farmland_label": _positive_label(),
        },
    )
    gate = build_phase10_reward_readiness_gate(report_path)

    gate_path = write_phase10_reward_readiness_gate(gate, tmp_path / "phase10")

    assert gate_path.name == "phase10_reward_readiness_gate.json"
    written = json.loads(gate_path.read_text(encoding="utf-8"))
    assert written["phase"] == "phase10_reward_readiness_gate"
    assert written["status"] == "ready_for_suitability_reward_smoke"
```

- [ ] **Step 3: Run Phase 10 tests**

Run:

```powershell
python -m pytest tests\test_phase10_reward_readiness.py -q
```

Expected result: all current Phase 10 tests pass.

## Task 4: CLI Runner

- [ ] **Step 1: Add CLI test**

Append:

```python
def test_phase10_cli_writes_gate_and_prints_summary(tmp_path, capsys):
    phase9_report_path = _write_phase9_fixture_report(tmp_path)
    output_dir = tmp_path / "phase10"
    runner_path = (
        ROOT
        / "experiments"
        / "phase10_reward_readiness"
        / "run_phase10_reward_readiness.py"
    )
    spec = importlib.util.spec_from_file_location("phase10_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--phase9-report",
            str(phase9_report_path),
            "--output-dir",
            str(output_dir),
            "--required-labels",
            "stable_farmland_label,high_standard_farmland_label",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "Gate:" in stdout
    assert "Status: not_ready_for_suitability_reward" in stdout
    assert "Recommendation: do_not_enable_suitability_reward" in stdout
    assert "stable_farmland_label:" in stdout
    assert "Claim boundary: Phase 10 is a reward-readiness gate" in stdout
    assert (output_dir / "phase10_reward_readiness_gate.json").exists()
```

- [ ] **Step 2: Run the CLI test and confirm it fails before the file exists**

Run:

```powershell
python -m pytest tests\test_phase10_reward_readiness.py::test_phase10_cli_writes_gate_and_prints_summary -q
```

Expected result: fail because `experiments/phase10_reward_readiness/run_phase10_reward_readiness.py` does not exist.

- [ ] **Step 3: Create the CLI**

Create `experiments/phase10_reward_readiness/run_phase10_reward_readiness.py` with argparse flags `--phase9-report`, `--output-dir`, `--required-labels`, `--min-rank-auc`, and `--min-mean-difference`. The CLI should call `build_phase10_reward_readiness_gate()`, write the gate JSON, print status, recommendation, count summary, per-label reasons, and `PHASE10_CLAIM_BOUNDARY`, and return `1` for `FileNotFoundError` or `ValueError`.

- [ ] **Step 4: Run the CLI test**

Run:

```powershell
python -m pytest tests\test_phase10_reward_readiness.py::test_phase10_cli_writes_gate_and_prints_summary -q
```

Expected result: pass.

## Task 5: Documentation and Manifest

- [ ] **Step 1: Update README**

Add `experiments/phase10_reward_readiness/` to the repository layout, add a
Phase 10 quick-start command after Phase 9, and add the Phase 10 runner to key
entry points.

Command block:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase10_reward_readiness\outputs\phase2_fixture
python experiments\phase9_proxy_validation\run_phase9_proxy_validation.py --phase2-output-dir experiments\phase10_reward_readiness\outputs\phase2_fixture --output-dir experiments\phase10_reward_readiness\outputs\phase9_report --label-columns stable_farmland_label,high_standard_farmland_label
python experiments\phase10_reward_readiness\run_phase10_reward_readiness.py --phase9-report experiments\phase10_reward_readiness\outputs\phase9_report\phase9_proxy_validation_report.json --output-dir experiments\phase10_reward_readiness\outputs\phase10_gate --required-labels stable_farmland_label,high_standard_farmland_label
```

- [ ] **Step 2: Update reproduction guide**

Insert a new Phase 10 section after Phase 9. Expected fixture result:

- `phase10_reward_readiness_gate.json` is written;
- status is `not_ready_for_suitability_reward`;
- recommendation is `do_not_enable_suitability_reward`;
- both included weak labels fail the gate because Phase 9 reported `negative_or_no_alignment`;
- the claim boundary states that Phase 10 does not train, tune, evaluate, or report a DRL policy and does not prove agronomic validity.

Add Phase 10 executable files to the runtime inspection section.

- [ ] **Step 3: Update file manifest**

Add rows for:

```text
docs/superpowers/specs/2026-06-10-phase10-reward-readiness-gate-design.md
docs/superpowers/plans/2026-06-10-phase10-reward-readiness-gate.md
src/paper11_geofm/reward_readiness.py
experiments/phase10_reward_readiness/run_phase10_reward_readiness.py
tests/test_phase10_reward_readiness.py
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests\test_phase10_reward_readiness.py -q
```

Expected result: all Phase 10 tests pass.

## Task 6: Verification, Commit, Merge

- [ ] **Step 1: Run reviewer CLI commands**

Run:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase10_reward_readiness\outputs\phase2_fixture
python experiments\phase9_proxy_validation\run_phase9_proxy_validation.py --phase2-output-dir experiments\phase10_reward_readiness\outputs\phase2_fixture --output-dir experiments\phase10_reward_readiness\outputs\phase9_report --label-columns stable_farmland_label,high_standard_farmland_label
python experiments\phase10_reward_readiness\run_phase10_reward_readiness.py --phase9-report experiments\phase10_reward_readiness\outputs\phase9_report\phase9_proxy_validation_report.json --output-dir experiments\phase10_reward_readiness\outputs\phase10_gate --required-labels stable_farmland_label,high_standard_farmland_label
```

Expected result: the Phase 10 CLI writes `phase10_reward_readiness_gate.json`, reports status `not_ready_for_suitability_reward`, and prints the claim boundary.

- [ ] **Step 2: Run full verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests
git diff --check
```

Expected result: smoke check passes, all tests pass, and diff check reports no whitespace errors.

- [ ] **Step 3: Commit implementation**

Run:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv src\paper11_geofm\reward_readiness.py experiments\phase10_reward_readiness\run_phase10_reward_readiness.py tests\test_phase10_reward_readiness.py docs\superpowers\plans\2026-06-10-phase10-reward-readiness-gate.md
git commit -m "Add Phase 10 reward readiness gate"
```

- [ ] **Step 4: Integrate**

Push the feature branch, fast-forward merge it to `main`, rerun reviewer CLI commands plus full verification on `main`, push `main`, and delete the local feature branch after `main` is synchronized with `origin/main`.

---

## Self-Review

- Spec coverage: covers Phase 9 report loading, required labels, thresholds, status/recommendation categories, fixture not-ready behavior, synthetic ready behavior, insufficient evidence, invalid reports, JSON writing, CLI, docs, manifest, and verification.
- Scope check: the plan does not train a policy, run rollout evaluation, compute planning metrics, or claim agronomic validity.
- Type consistency: function names, artifact filename, status values, CLI flags, and claim boundary match the Phase 10 design spec.

