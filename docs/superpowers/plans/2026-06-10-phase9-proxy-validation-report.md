# Phase 9 Proxy-Validation Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Phase 9 weak-label validation report for `suitability_proxy` from Phase 2 block feature outputs.

**Architecture:** Add `paper11_geofm.proxy_validation` to read `block_geofm_features.csv`, compute suitability distribution and per-label alignment diagnostics, and write one JSON report. Add a CLI under `experiments/phase9_proxy_validation/` and document the reviewer path after Phase 8.

**Tech Stack:** Python, NumPy, CSV/JSON artifacts, argparse, pytest.

---

## File Structure

- Create `src/paper11_geofm/proxy_validation.py`: Phase 9 claim boundary, CSV loader, suitability summary, per-label diagnostics, interpretation categories, JSON writer.
- Create `experiments/phase9_proxy_validation/run_phase9_proxy_validation.py`: CLI runner.
- Create `tests/test_phase9_proxy_validation.py`: builder, validation failure, writer, and CLI tests.
- Modify `README.md`: add Phase 9 quick-start command and entry point.
- Modify `reproducibility/REPRODUCTION_GUIDE.md`: add Phase 9 reviewer step and executable file list.
- Modify `reproducibility/FILE_MANIFEST.tsv`: add the Phase 9 design, plan, runtime module, CLI, and tests.

## Task 1: Report Builder Test Contract

- [ ] **Step 1: Write the failing tests**

Create `tests/test_phase9_proxy_validation.py` with helpers that run the existing Phase 2 fixture runner and then assert the Phase 9 report contract.

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
    spec = importlib.util.spec_from_file_location("phase2_runner_phase9", runner_path)
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


def test_phase9_builds_proxy_validation_report_from_phase2_fixture(tmp_path):
    from paper11_geofm.proxy_validation import (
        PHASE9_CLAIM_BOUNDARY,
        build_phase9_proxy_validation_report,
    )

    phase2_dir = _run_phase2_fixture(tmp_path / "phase2")

    report = build_phase9_proxy_validation_report(
        phase2_dir,
        label_columns=(
            "stable_farmland_label",
            "high_standard_farmland_label",
            "missing_label",
        ),
    )

    assert report["phase"] == "phase9_proxy_validation_report"
    assert report["block_table"] == "block_geofm_features.csv"
    assert report["n_blocks"] == 4
    assert report["label_columns_requested"] == [
        "stable_farmland_label",
        "high_standard_farmland_label",
        "missing_label",
    ]
    assert report["label_columns_available"] == [
        "stable_farmland_label",
        "high_standard_farmland_label",
    ]
    assert report["label_columns_missing"] == ["missing_label"]
    assert report["suitability_summary"]["count"] == 4
    assert set(report["suitability_summary"]) == {
        "count",
        "min",
        "max",
        "mean",
        "std",
        "q25",
        "median",
        "q75",
    }
    stable = report["labels"]["stable_farmland_label"]
    assert stable["validation_available"] is True
    assert stable["positive_count"] == 2
    assert stable["negative_count"] == 2
    assert stable["valid_label_count"] == 4
    assert stable["missing_label_count"] == 0
    assert stable["mean_difference"] is not None
    assert stable["rank_auc"] is not None
    assert stable["interpretation"] in {
        "positive_alignment",
        "negative_or_no_alignment",
    }
    assert report["labels"]["missing_label"]["interpretation"] == "label_unavailable"
    assert report["claim_boundary"] == PHASE9_CLAIM_BOUNDARY
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```powershell
python -m pytest tests\test_phase9_proxy_validation.py::test_phase9_builds_proxy_validation_report_from_phase2_fixture -q
```

Expected result: fail with `ModuleNotFoundError: No module named 'paper11_geofm.proxy_validation'`.

## Task 2: Report Builder Implementation

- [ ] **Step 1: Create the module**

Create `src/paper11_geofm/proxy_validation.py` with these public functions and helpers:

```python
from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


PHASE9_CLAIM_BOUNDARY = (
    "Phase 9 is a weak-label proxy-validation report for suitability_proxy; "
    "it does not prove agronomic validity, train a policy, evaluate a policy, "
    "or report planning performance."
)
DEFAULT_LABEL_COLUMNS = (
    "stable_farmland_label",
    "high_standard_farmland_label",
)


def build_phase9_proxy_validation_report(
    phase2_output_dir: Path | str,
    label_columns: Sequence[str] = DEFAULT_LABEL_COLUMNS,
) -> dict[str, object]:
    output_dir = Path(phase2_output_dir)
    block_table = output_dir / "block_geofm_features.csv"
    rows = _read_block_rows(block_table)
    suitability_values = _extract_suitability(rows, block_table)
    requested = [str(column) for column in label_columns]
    available = [
        column
        for column in requested
        if any(column in row and str(row.get(column, "")).strip() != "" for row in rows)
    ]
    missing = [column for column in requested if column not in available]
    labels = {
        column: _label_report(rows, column)
        for column in requested
    }
    return {
        "phase": "phase9_proxy_validation_report",
        "phase2_output_dir": str(output_dir),
        "block_table": block_table.name,
        "label_columns_requested": requested,
        "label_columns_available": available,
        "label_columns_missing": missing,
        "n_blocks": len(rows),
        "suitability_summary": _suitability_summary(suitability_values),
        "labels": labels,
        "claim_boundary": PHASE9_CLAIM_BOUNDARY,
    }


def write_phase9_proxy_validation_report(
    report: Mapping[str, object],
    output_dir: Path | str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "phase9_proxy_validation_report.json"
    report_path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path
```

- [ ] **Step 2: Implement helper behavior**

Implementation details:

- `_read_block_rows(path)` raises `FileNotFoundError(f"Missing Phase 2 block feature table: {path}")` when the table is absent and returns `list[dict[str, str]]`.
- `_extract_suitability(rows, path)` requires column `suitability_proxy`; it parses numeric values and raises `ValueError("Phase 9 requires at least one numeric suitability_proxy value")` if none are usable.
- `_suitability_summary(values)` returns rounded `count`, `min`, `max`, `mean`, `std`, `q25`, `median`, and `q75` values.
- `_parse_binary_label(value)` accepts `1`, `1.0`, `true`, `yes`, `0`, `0.0`, `false`, and `no`, matching Phase 2 behavior.
- `_rank_auc_or_none(positives, negatives)` uses pair counting with ties counted as `0.5`.
- `_label_report(rows, column)` returns `label_unavailable`, `insufficient_label_variation`, `positive_alignment`, or `negative_or_no_alignment` according to the design spec.
- `_quantiles(values)` returns `{"min", "q25", "median", "q75", "max"}` with `None` values for empty input.

- [ ] **Step 3: Run the focused test**

Run:

```powershell
python -m pytest tests\test_phase9_proxy_validation.py::test_phase9_builds_proxy_validation_report_from_phase2_fixture -q
```

Expected result: pass.

## Task 3: Error Handling and JSON Writer Tests

- [ ] **Step 1: Add missing-table and bad-suitability tests**

Append these tests to `tests/test_phase9_proxy_validation.py`:

```python
def test_phase9_missing_block_table_raises(tmp_path):
    from paper11_geofm.proxy_validation import build_phase9_proxy_validation_report

    with pytest.raises(FileNotFoundError, match="Missing Phase 2 block feature table"):
        build_phase9_proxy_validation_report(tmp_path)


def test_phase9_unusable_suitability_column_raises(tmp_path):
    from paper11_geofm.proxy_validation import build_phase9_proxy_validation_report

    phase2_dir = tmp_path / "phase2"
    phase2_dir.mkdir()
    (phase2_dir / "block_geofm_features.csv").write_text(
        "block_id,suitability_proxy,stable_farmland_label\n"
        "b0,not_numeric,1\n"
        "b1,,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="numeric suitability_proxy"):
        build_phase9_proxy_validation_report(phase2_dir)
```

- [ ] **Step 2: Add writer test**

Append:

```python
def test_phase9_proxy_validation_report_is_written(tmp_path):
    from paper11_geofm.proxy_validation import (
        build_phase9_proxy_validation_report,
        write_phase9_proxy_validation_report,
    )

    phase2_dir = _run_phase2_fixture(tmp_path / "phase2")
    report = build_phase9_proxy_validation_report(phase2_dir)

    report_path = write_phase9_proxy_validation_report(report, tmp_path / "phase9")

    assert report_path.name == "phase9_proxy_validation_report.json"
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["phase"] == "phase9_proxy_validation_report"
    assert written["n_blocks"] == 4
    assert written["claim_boundary"] == report["claim_boundary"]
```

- [ ] **Step 3: Run Phase 9 unit tests**

Run:

```powershell
python -m pytest tests\test_phase9_proxy_validation.py -q
```

Expected result: all Phase 9 tests pass.

## Task 4: CLI Runner

- [ ] **Step 1: Add CLI test**

Append:

```python
def test_phase9_cli_writes_report_and_prints_summary(tmp_path, capsys):
    phase2_dir = _run_phase2_fixture(tmp_path / "phase2")
    output_dir = tmp_path / "phase9"
    runner_path = (
        ROOT / "experiments" / "phase9_proxy_validation" / "run_phase9_proxy_validation.py"
    )
    spec = importlib.util.spec_from_file_location("phase9_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--phase2-output-dir",
            str(phase2_dir),
            "--output-dir",
            str(output_dir),
            "--label-columns",
            "stable_farmland_label,high_standard_farmland_label,missing_label",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "Report:" in stdout
    assert "Blocks: 4" in stdout
    assert "Available labels: stable_farmland_label,high_standard_farmland_label" in stdout
    assert "stable_farmland_label rank_auc:" in stdout
    assert "missing_label: label_unavailable" in stdout
    assert "Claim boundary: Phase 9 is a weak-label proxy-validation report" in stdout
    assert (output_dir / "phase9_proxy_validation_report.json").exists()
```

- [ ] **Step 2: Run the CLI test and confirm it fails before the file exists**

Run:

```powershell
python -m pytest tests\test_phase9_proxy_validation.py::test_phase9_cli_writes_report_and_prints_summary -q
```

Expected result: fail because `experiments/phase9_proxy_validation/run_phase9_proxy_validation.py` does not exist.

- [ ] **Step 3: Create the CLI**

Create `experiments/phase9_proxy_validation/run_phase9_proxy_validation.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.proxy_validation import (
    PHASE9_CLAIM_BOUNDARY,
    build_phase9_proxy_validation_report,
    write_phase9_proxy_validation_report,
)


def _parse_label_columns(text: str) -> tuple[str, ...]:
    columns = tuple(part.strip() for part in text.split(",") if part.strip())
    if not columns:
        raise ValueError("At least one label column must be provided")
    return columns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Paper11 Phase 9 weak-label proxy-validation report "
            "without training or evaluating a policy."
        )
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--label-columns",
        default="stable_farmland_label,high_standard_farmland_label",
    )
    args = parser.parse_args(argv)

    try:
        report = build_phase9_proxy_validation_report(
            args.phase2_output_dir,
            label_columns=_parse_label_columns(args.label_columns),
        )
        report_path = write_phase9_proxy_validation_report(report, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Report: {report_path}")
    print(f"Blocks: {report['n_blocks']}")
    available = ",".join(report["label_columns_available"])
    print(f"Available labels: {available if available else 'none'}")
    for label_column, label_report in report["labels"].items():
        if label_report["validation_available"]:
            print(
                f"{label_column} rank_auc: {label_report['rank_auc']} "
                f"mean_difference: {label_report['mean_difference']} "
                f"interpretation: {label_report['interpretation']}"
            )
        else:
            print(f"{label_column}: {label_report['interpretation']}")
    print(f"Claim boundary: {PHASE9_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the CLI test**

Run:

```powershell
python -m pytest tests\test_phase9_proxy_validation.py::test_phase9_cli_writes_report_and_prints_summary -q
```

Expected result: pass.

## Task 5: Documentation and Manifest

- [ ] **Step 1: Update README**

Add `experiments/phase9_proxy_validation/` to the repository layout, add a
Phase 9 quick-start command after Phase 8, and add the Phase 9 runner to key
entry points.

Command block:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase9_proxy_validation\outputs\phase2_fixture
python experiments\phase9_proxy_validation\run_phase9_proxy_validation.py --phase2-output-dir experiments\phase9_proxy_validation\outputs\phase2_fixture --output-dir experiments\phase9_proxy_validation\outputs\phase9_report --label-columns stable_farmland_label,high_standard_farmland_label
```

- [ ] **Step 2: Update reproduction guide**

Insert a new Phase 9 section after Phase 8 with expected outcomes:

- `phase9_proxy_validation_report.json` is written;
- block count is `4` for the fixture;
- both included weak labels are available;
- missing or single-class labels are reported without becoming policy evidence;
- the claim boundary states that Phase 9 does not prove agronomic validity, train a policy, evaluate a policy, or report planning performance.

Update later section numbers and add Phase 9 executable files:

```text
experiments/phase9_proxy_validation/run_phase9_proxy_validation.py
src/paper11_geofm/proxy_validation.py
```

- [ ] **Step 3: Update file manifest**

Add rows for:

```text
docs/superpowers/specs/2026-06-10-phase9-proxy-validation-report-design.md
docs/superpowers/plans/2026-06-10-phase9-proxy-validation-report.md
src/paper11_geofm/proxy_validation.py
experiments/phase9_proxy_validation/run_phase9_proxy_validation.py
tests/test_phase9_proxy_validation.py
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests\test_phase9_proxy_validation.py -q
```

Expected result: all Phase 9 tests pass.

## Task 6: Verification, Commit, Merge

- [ ] **Step 1: Run reviewer CLI commands**

Run:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase9_proxy_validation\outputs\phase2_fixture
python experiments\phase9_proxy_validation\run_phase9_proxy_validation.py --phase2-output-dir experiments\phase9_proxy_validation\outputs\phase2_fixture --output-dir experiments\phase9_proxy_validation\outputs\phase9_report --label-columns stable_farmland_label,high_standard_farmland_label
```

Expected result: Phase 2 fixture artifacts and `phase9_proxy_validation_report.json` are written under ignored `outputs/`; CLI prints block count, label diagnostics, report path, and claim boundary.

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
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv src\paper11_geofm\proxy_validation.py experiments\phase9_proxy_validation\run_phase9_proxy_validation.py tests\test_phase9_proxy_validation.py docs\superpowers\plans\2026-06-10-phase9-proxy-validation-report.md
git commit -m "Add Phase 9 proxy validation report"
```

- [ ] **Step 4: Integrate**

Push the feature branch, fast-forward merge it to `main`, rerun the reviewer CLI commands plus full verification on `main`, push `main`, and delete the local feature branch after `main` is synchronized with `origin/main`.

---

## Self-Review

- Spec coverage: the plan covers report construction, label availability, suitability summary, per-label diagnostics, missing-label handling, JSON artifact writing, CLI output, docs, manifest, and verification.
- Scope check: the plan does not train a policy, run rollout evaluation, compute planning metrics, or claim agronomic validity.
- Type consistency: function names, artifact filename, label-column defaults, CLI flags, and claim boundary match the Phase 9 design spec.

