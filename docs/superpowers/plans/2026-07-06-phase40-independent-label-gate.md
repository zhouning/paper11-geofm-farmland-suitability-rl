# Phase 40 Independent Label Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 40 as a hard go/no-go gate for independent suitability labels before any Phase 38 rerun, B2/B3 reward smoke, or positive suitability claim.

**Architecture:** Add one focused analysis module, one thin CLI runner, focused pytest coverage, one reviewer-facing result document, and documentation updates. Phase 40 deliberately does not train PPO or enable B2/B3; it converts the current uncertainty into a decision: either a defensible independent label is present, or the suitability-reward route remains blocked and the paper must be framed as a diagnostic platform.

**Tech Stack:** Python standard library (`csv`, `json`, `dataclasses`, `pathlib`, `math`), pytest, existing Paper11 experiment runner layout, ignored `experiments/**/outputs/` artifact convention.

---

## Decision Backbone

Do not treat Phase 40 as another experiment that can quietly fail and be followed by Phase 41. Phase 40 exists to stop ambiguous progress:

- If Phase 40 returns `independent_label_gate_passed`, the next justified step is a Phase 38 proxy-rebuild rerun using the accepted label.
- If Phase 40 returns `independent_label_gate_diagnostic_only`, the label can be discussed as diagnostic evidence only; B2/B3 remain blocked.
- If Phase 40 returns `independent_label_gate_blocked` or `independent_label_inputs_missing`, stop the suitability-reward route and update the manuscript as a negative/diagnostic paper until an external label source is supplied.

## File Structure

- Create: `tests/test_phase40_independent_label_gate.py`
  - Synthetic fixtures for Phase 2 feature tables, CSV/JSON registries, pass/blocked/diagnostic statuses, artifact writing, and CLI behavior.
- Create: `src/paper11_geofm/phase40_independent_label_gate.py`
  - Owns constants, thresholds, CSV/JSON registry parsing, label normalization, per-label gate evaluation, gate reduction, artifact writing, and Markdown output.
- Create: `experiments/phase40_independent_label_gate/run_phase40_independent_label_gate.py`
  - Thin argparse runner that calls the module and prints the decision.
- Create after real no-registry run: `paper/phase28_results/14_phase40_independent_label_gate.md`
  - Reviewer-facing interpretation of the current real Phase 40 status.
- Modify after real no-registry run: `README.md`
  - Add Phase 40 runner, current status, and go/no-go boundary.
- Modify after real no-registry run: `paper/phase28_results/README.md`
  - Add Phase 40 entry and reproduction command.
- Modify after real no-registry run: `paper/submission/01_ijaeog_submission_readiness.md`
  - Update readiness table and claim gate with Phase 40.
- Modify after real no-registry run: `paper/submission/02_draft_titles_highlights_declarations.md`
  - Update abstract scaffold and claim boundary so it does not imply B2/B3 readiness.
- Modify after real no-registry run: `reproducibility/FILE_MANIFEST.tsv`
  - Add Phase 40 source, runner, tests, spec, plan, and result doc entries.
- Modify after real no-registry run: `docs/superpowers/phase33_current_progress_handoff.md`
  - Record Phase 40 status, verification, and the decision branch.

## Task 1: Failing Tests For Registry And Gate Decisions

**Files:**
- Create: `tests/test_phase40_independent_label_gate.py`
- Target later: `src/paper11_geofm/phase40_independent_label_gate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_phase40_independent_label_gate.py` with this content:

```python
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _phase2_dir(tmp_path: Path, rows: list[dict[str, object]] | None = None) -> Path:
    if rows is None:
        rows = []
        for index in range(12):
            rows.append(
                {
                    "block_id": f"b{index:03d}",
                    "split": "train" if index < 8 else "test",
                    "independent_irrigation_label": 1 if index in {0, 1, 8} else 0,
                    "diagnostic_internal_label": 1 if index % 2 == 0 else 0,
                    "single_class_label": 1,
                }
            )
    return _write_csv(
        tmp_path / "phase2" / "block_geofm_features.csv",
        rows,
        [
            "block_id",
            "split",
            "independent_irrigation_label",
            "diagnostic_internal_label",
            "single_class_label",
        ],
    ).parent


def _registry_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    return _write_csv(
        path,
        rows,
        [
            "label_column",
            "label_source",
            "source_type",
            "independence_level",
            "allowed_eval_roles",
            "provenance_note",
            "license_or_access",
            "expected_positive_definition",
        ],
    )


def _independent_registry_row(label_column: str = "independent_irrigation_label") -> dict[str, object]:
    return {
        "label_column": label_column,
        "label_source": "synthetic independent irrigation fixture",
        "source_type": "external_irrigation",
        "independence_level": "independent",
        "allowed_eval_roles": "test,validation,eval",
        "provenance_note": "not derived from DLTB, slope, source metadata, or GeoFM features",
        "license_or_access": "test fixture",
        "expected_positive_definition": "1",
    }


def test_phase40_no_registry_returns_missing_inputs(tmp_path):
    from paper11_geofm.phase40_independent_label_gate import (
        run_phase40_independent_label_gate,
    )

    result = run_phase40_independent_label_gate(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=None,
    )

    assert result["phase"] == "phase40_independent_label_gate"
    assert result["phase40_independent_label_gate_status"] == "independent_label_inputs_missing"
    assert result["row_counts"]["registry_rows"] == 0
    assert "go/no-go" in result["claim_boundary"]


def test_phase40_empty_registry_returns_missing_inputs(tmp_path):
    from paper11_geofm.phase40_independent_label_gate import (
        run_phase40_independent_label_gate,
    )

    registry = _registry_csv(tmp_path / "registry.csv", [])
    result = run_phase40_independent_label_gate(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=registry,
    )

    assert result["phase40_independent_label_gate_status"] == "independent_label_inputs_missing"
    assert result["row_counts"]["registry_rows"] == 0


def test_phase40_independent_csv_registry_can_pass_gate(tmp_path):
    from paper11_geofm.phase40_independent_label_gate import (
        run_phase40_independent_label_gate,
    )

    registry = _registry_csv(tmp_path / "registry.csv", [_independent_registry_row()])
    result = run_phase40_independent_label_gate(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=registry,
        min_valid_count=10,
        min_split_valid_count=2,
    )

    assert result["phase40_independent_label_gate_status"] == "independent_label_gate_passed"
    row = result["label_gate_rows"][0]
    assert row["label_column"] == "independent_irrigation_label"
    assert row["label_gate_status"] == "label_gate_passed"
    assert row["valid_label_count"] == 12
    assert row["train_valid_count"] == 8
    assert row["eval_valid_count"] == 4
    assert row["positive_rate"] == 0.25


def test_phase40_json_registry_has_csv_semantics(tmp_path):
    from paper11_geofm.phase40_independent_label_gate import (
        run_phase40_independent_label_gate,
    )

    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps([_independent_registry_row()]), encoding="utf-8")

    result = run_phase40_independent_label_gate(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=registry,
        min_valid_count=10,
        min_split_valid_count=2,
    )

    assert result["phase40_independent_label_gate_status"] == "independent_label_gate_passed"
    assert result["label_gate_rows"][0]["label_gate_status"] == "label_gate_passed"


def test_phase40_internal_label_is_diagnostic_only(tmp_path):
    from paper11_geofm.phase40_independent_label_gate import (
        run_phase40_independent_label_gate,
    )

    row = _independent_registry_row("diagnostic_internal_label")
    row["source_type"] = "diagnostic_internal"
    row["independence_level"] = "diagnostic_only"
    registry = _registry_csv(tmp_path / "registry.csv", [row])

    result = run_phase40_independent_label_gate(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=registry,
        min_valid_count=10,
        min_split_valid_count=2,
    )

    assert result["phase40_independent_label_gate_status"] == "independent_label_gate_diagnostic_only"
    assert result["label_gate_rows"][0]["label_gate_status"] == "label_gate_diagnostic_only"
    assert "not independent enough" in result["label_gate_rows"][0]["decision_reason"]


def test_phase40_missing_label_column_blocks_gate(tmp_path):
    from paper11_geofm.phase40_independent_label_gate import (
        run_phase40_independent_label_gate,
    )

    registry = _registry_csv(
        tmp_path / "registry.csv",
        [_independent_registry_row("missing_external_label")],
    )

    result = run_phase40_independent_label_gate(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=registry,
    )

    assert result["phase40_independent_label_gate_status"] == "independent_label_gate_blocked"
    assert result["label_gate_rows"][0]["label_gate_status"] == "label_missing"


def test_phase40_single_class_label_blocks_gate(tmp_path):
    from paper11_geofm.phase40_independent_label_gate import (
        run_phase40_independent_label_gate,
    )

    registry = _registry_csv(
        tmp_path / "registry.csv",
        [_independent_registry_row("single_class_label")],
    )

    result = run_phase40_independent_label_gate(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=registry,
        min_valid_count=10,
        min_split_valid_count=2,
    )

    assert result["phase40_independent_label_gate_status"] == "independent_label_gate_blocked"
    assert result["label_gate_rows"][0]["label_gate_status"] == "label_gate_blocked"
    assert "positive_rate" in result["label_gate_rows"][0]["decision_reason"]


def test_phase40_artifact_writer_creates_csv_json_markdown(tmp_path):
    from paper11_geofm.phase40_independent_label_gate import (
        run_phase40_independent_label_gate,
        write_phase40_independent_label_gate_artifacts,
    )

    result = run_phase40_independent_label_gate(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=_registry_csv(tmp_path / "registry.csv", [_independent_registry_row()]),
        min_valid_count=10,
        min_split_valid_count=2,
    )
    artifacts = write_phase40_independent_label_gate_artifacts(result, tmp_path / "outputs")

    assert {path.name for path in artifacts.values()} == {
        "phase40_label_gate_summary.csv",
        "phase40_independent_label_gate.json",
        "phase40_independent_label_gate.md",
    }
    saved = json.loads(artifacts["diagnosis_json"].read_text(encoding="utf-8"))
    assert saved["phase40_independent_label_gate_status"] == "independent_label_gate_passed"
    markdown = artifacts["diagnosis_md"].read_text(encoding="utf-8")
    assert "Phase 40 Independent Label Gate" in markdown
    assert "independent_label_gate_passed" in markdown


def test_phase40_cli_writes_outputs(tmp_path):
    phase2_dir = _phase2_dir(tmp_path)
    registry = _registry_csv(tmp_path / "registry.csv", [_independent_registry_row()])
    output_dir = tmp_path / "outputs"
    runner = ROOT / "experiments" / "phase40_independent_label_gate" / (
        "run_phase40_independent_label_gate.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--phase2-output-dir",
            str(phase2_dir),
            "--label-registry",
            str(registry),
            "--output-dir",
            str(output_dir),
            "--min-valid-count",
            "10",
            "--min-split-valid-count",
            "2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Phase 40 independent-label gate status:" in result.stdout
    assert "independent_label_gate_passed" in result.stdout
    assert (output_dir / "phase40_independent_label_gate.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests\test_phase40_independent_label_gate.py -q --basetemp=.pytest_tmp_phase40_t1 -p no:cacheprovider
```

Expected: FAIL because `paper11_geofm.phase40_independent_label_gate` does not exist.

## Task 2: Core Module And Gate Logic

**Files:**
- Create: `src/paper11_geofm/phase40_independent_label_gate.py`
- Test: `tests/test_phase40_independent_label_gate.py`

- [ ] **Step 1: Implement the Phase 40 module**

Create `src/paper11_geofm/phase40_independent_label_gate.py` with this structure:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path


PHASE40_INDEPENDENT_LABEL_GATE_CLAIM_BOUNDARY = (
    "Phase 40 is a go/no-go independent-label gate. It validates whether a "
    "registered non-leakage label can justify a later Phase 38 proxy-rebuild "
    "rerun. It does not train PPO, does not alter rewards, does not enable "
    "B2/B3, and does not prove suitability or planning-performance improvement."
)

PHASE40_TRAIN_SPLIT_VALUES = {"train", "training"}
PHASE40_EVAL_SPLIT_VALUES = {"test", "eval", "evaluation", "validation", "val"}

PHASE40_REGISTRY_FIELDNAMES = (
    "label_column",
    "label_source",
    "source_type",
    "independence_level",
    "allowed_eval_roles",
    "provenance_note",
    "license_or_access",
    "expected_positive_definition",
)

PHASE40_LABEL_GATE_FIELDNAMES = (
    "label_column",
    "label_gate_status",
    "label_source",
    "source_type",
    "independence_level",
    "allowed_eval_roles",
    "valid_label_count",
    "missing_count",
    "missing_rate",
    "positive_count",
    "negative_count",
    "positive_rate",
    "train_valid_count",
    "eval_valid_count",
    "train_positive_count",
    "train_negative_count",
    "eval_positive_count",
    "eval_negative_count",
    "decision_reason",
    "claim_boundary",
)

INDEPENDENT_SOURCE_TYPES = {
    "external_field_survey",
    "external_agronomic",
    "external_soil",
    "external_irrigation",
    "external_yield",
    "external_high_standard_farmland",
    "external_retention_or_policy",
    "remote_sensing_independent_product",
}

DIAGNOSTIC_SOURCE_TYPES = {
    "diagnostic_internal",
    "dltb_derived",
    "slope_derived",
    "source_metadata",
    "geofm_derived",
    "unknown",
}

VALID_SOURCE_TYPES = INDEPENDENT_SOURCE_TYPES | DIAGNOSTIC_SOURCE_TYPES
PASSING_INDEPENDENCE_LEVELS = {"independent", "partially_independent"}
DIAGNOSTIC_INDEPENDENCE_LEVELS = {"diagnostic_only", "leakage_risk", "unknown"}
VALID_INDEPENDENCE_LEVELS = PASSING_INDEPENDENCE_LEVELS | DIAGNOSTIC_INDEPENDENCE_LEVELS


@dataclass(frozen=True)
class Phase40Thresholds:
    min_valid_count: int = 100
    max_missing_rate: float = 0.20
    min_positive_rate: float = 0.02
    max_positive_rate: float = 0.98
    min_split_valid_count: int = 20


def run_phase40_independent_label_gate(
    phase2_output_dir: Path | str,
    label_registry: Path | str | None = None,
    min_valid_count: int = 100,
    max_missing_rate: float = 0.20,
    min_positive_rate: float = 0.02,
    max_positive_rate: float = 0.98,
    min_split_valid_count: int = 20,
) -> dict[str, object]:
    phase2_dir = Path(phase2_output_dir)
    feature_csv = phase2_dir / "block_geofm_features.csv"
    fieldnames, feature_rows = _read_csv_table(feature_csv, "Phase 2 block feature CSV")
    if "block_id" not in fieldnames:
        raise ValueError(f"Phase 40 Phase 2 CSV is missing block_id: {feature_csv}")
    if "split" not in fieldnames:
        raise ValueError(f"Phase 40 Phase 2 CSV is missing split: {feature_csv}")

    thresholds = Phase40Thresholds(
        min_valid_count=min_valid_count,
        max_missing_rate=max_missing_rate,
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
        min_split_valid_count=min_split_valid_count,
    )
    registry_rows = load_label_registry(label_registry)
    label_gate_rows = [
        evaluate_label_candidate(feature_rows, row, thresholds)
        for row in registry_rows
    ]
    status = summarize_phase40_gate(label_gate_rows)
    return {
        "phase": "phase40_independent_label_gate",
        "phase40_independent_label_gate_status": status,
        "source_paths": {
            "phase2_output_dir": str(phase2_dir),
            "phase2_block_geofm_features_csv": str(feature_csv),
            "label_registry": str(Path(label_registry)) if label_registry is not None else None,
        },
        "thresholds": thresholds.__dict__,
        "row_counts": {
            "feature_rows": len(feature_rows),
            "registry_rows": len(registry_rows),
            "label_gate_rows": len(label_gate_rows),
        },
        "label_gate_rows": label_gate_rows,
        "interpretation": _phase40_interpretation(status),
        "claim_boundary": PHASE40_INDEPENDENT_LABEL_GATE_CLAIM_BOUNDARY,
        "recommended_next_step": _phase40_next_step(status),
    }


def load_label_registry(label_registry: Path | str | None) -> list[dict[str, str]]:
    if label_registry is None:
        return []
    registry_path = Path(label_registry)
    if not registry_path.exists():
        raise ValueError(f"Missing Phase 40 label registry: {registry_path}")
    suffix = registry_path.suffix.lower()
    if suffix == ".csv":
        _, rows = _read_csv_table(registry_path, "Phase 40 label registry CSV")
    elif suffix == ".json":
        rows = _read_json_registry_rows(registry_path)
    else:
        raise ValueError(f"Unsupported Phase 40 label registry extension: {registry_path}")
    return [_normalize_registry_row(row, registry_path, index) for index, row in enumerate(rows, start=2)]


def evaluate_label_candidate(
    feature_rows: Sequence[Mapping[str, str]],
    registry_row: Mapping[str, str],
    thresholds: Phase40Thresholds,
) -> dict[str, object]:
    label_column = str(registry_row.get("label_column", "")).strip()
    if not label_column:
        return _blocked_row(registry_row, "label_gate_blocked", "registry row has blank label_column")
    if not feature_rows or label_column not in feature_rows[0]:
        return _blocked_row(registry_row, "label_missing", f"label column {label_column!r} is missing from the feature table")

    labels: list[int] = []
    train_labels: list[int] = []
    eval_labels: list[int] = []
    missing_count = 0
    positive_definition = str(registry_row.get("expected_positive_definition", "1")).strip() or "1"
    for row in feature_rows:
        parsed = _parse_label(row.get(label_column), positive_definition)
        if parsed is None:
            missing_count += 1
            continue
        labels.append(parsed)
        split_role = _split_role(row.get("split"))
        if split_role == "train":
            train_labels.append(parsed)
        elif split_role == "eval":
            eval_labels.append(parsed)

    valid_count = len(labels)
    positive_count = sum(1 for label in labels if label == 1)
    negative_count = sum(1 for label in labels if label == 0)
    train_positive = sum(1 for label in train_labels if label == 1)
    train_negative = sum(1 for label in train_labels if label == 0)
    eval_positive = sum(1 for label in eval_labels if label == 1)
    eval_negative = sum(1 for label in eval_labels if label == 0)
    missing_rate = missing_count / len(feature_rows) if feature_rows else 1.0
    positive_rate = positive_count / valid_count if valid_count else 0.0
    source_type = str(registry_row.get("source_type", "")).strip()
    independence_level = str(registry_row.get("independence_level", "")).strip()
    allowed_roles = _normalize_csvish_values(registry_row.get("allowed_eval_roles", ""))
    available_eval_roles = {
        str(row.get("split", "")).strip().lower()
        for row in feature_rows
        if _split_role(row.get("split")) == "eval"
    }

    failure_reasons = _label_failure_reasons(
        valid_count,
        missing_rate,
        positive_rate,
        len(train_labels),
        len(eval_labels),
        source_type,
        independence_level,
        allowed_roles,
        available_eval_roles,
        thresholds,
    )
    if not failure_reasons:
        status = "label_gate_passed"
        reason = "label passed independent source, balance, missingness, and split coverage gates"
    elif _is_diagnostic_source(source_type, independence_level):
        status = "label_gate_diagnostic_only"
        reason = "label is computable but not independent enough for B2/B3: " + "; ".join(failure_reasons)
    else:
        status = "label_gate_blocked"
        reason = "; ".join(failure_reasons)

    return {
        "label_column": label_column,
        "label_gate_status": status,
        "label_source": registry_row.get("label_source", ""),
        "source_type": source_type,
        "independence_level": independence_level,
        "allowed_eval_roles": ",".join(allowed_roles),
        "valid_label_count": valid_count,
        "missing_count": missing_count,
        "missing_rate": _round_float(missing_rate),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": _round_float(positive_rate),
        "train_valid_count": len(train_labels),
        "eval_valid_count": len(eval_labels),
        "train_positive_count": train_positive,
        "train_negative_count": train_negative,
        "eval_positive_count": eval_positive,
        "eval_negative_count": eval_negative,
        "decision_reason": reason,
        "claim_boundary": PHASE40_INDEPENDENT_LABEL_GATE_CLAIM_BOUNDARY,
    }
```

Continue the same file with helper functions:

```python
def summarize_phase40_gate(label_gate_rows: Sequence[Mapping[str, object]]) -> str:
    if not label_gate_rows:
        return "independent_label_inputs_missing"
    statuses = {str(row.get("label_gate_status", "")) for row in label_gate_rows}
    if "label_gate_passed" in statuses:
        return "independent_label_gate_passed"
    if "label_gate_diagnostic_only" in statuses:
        return "independent_label_gate_diagnostic_only"
    return "independent_label_gate_blocked"


def write_phase40_independent_label_gate_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "label_gate_summary_csv": output_path / "phase40_label_gate_summary.csv",
        "diagnosis_json": output_path / "phase40_independent_label_gate.json",
        "diagnosis_md": output_path / "phase40_independent_label_gate.md",
    }
    _write_csv_mapping_rows(
        artifacts["label_gate_summary_csv"],
        PHASE40_LABEL_GATE_FIELDNAMES,
        analysis.get("label_gate_rows", []),
        "Phase 40 label gate rows",
    )
    artifacts["diagnosis_json"].write_text(
        json.dumps(_json_ready(analysis), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    artifacts["diagnosis_md"].write_text(_phase40_markdown(analysis), encoding="utf-8")
    return artifacts


def _normalize_registry_row(
    row: Mapping[str, object],
    registry_path: Path,
    row_index: int,
) -> dict[str, str]:
    normalized = {field: str(row.get(field, "")).strip() for field in PHASE40_REGISTRY_FIELDNAMES}
    if not normalized["label_column"]:
        raise ValueError(f"Phase 40 label registry row {row_index} has blank label_column: {registry_path}")
    source_type = normalized["source_type"]
    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError(f"Phase 40 unsupported source_type {source_type!r} for label {normalized['label_column']}")
    independence_level = normalized["independence_level"]
    if independence_level not in VALID_INDEPENDENCE_LEVELS:
        raise ValueError(f"Phase 40 unsupported independence_level {independence_level!r} for label {normalized['label_column']}")
    return normalized


def _label_failure_reasons(
    valid_count: int,
    missing_rate: float,
    positive_rate: float,
    train_count: int,
    eval_count: int,
    source_type: str,
    independence_level: str,
    allowed_roles: Sequence[str],
    available_eval_roles: set[str],
    thresholds: Phase40Thresholds,
) -> list[str]:
    reasons: list[str] = []
    if valid_count < thresholds.min_valid_count:
        reasons.append(f"valid_label_count {valid_count} is below min_valid_count {thresholds.min_valid_count}")
    if missing_rate > thresholds.max_missing_rate:
        reasons.append(f"missing_rate {missing_rate:.10f} exceeds max_missing_rate {thresholds.max_missing_rate:.10f}")
    if positive_rate < thresholds.min_positive_rate or positive_rate > thresholds.max_positive_rate:
        reasons.append(
            f"positive_rate {positive_rate:.10f} is outside "
            f"[{thresholds.min_positive_rate:.10f}, {thresholds.max_positive_rate:.10f}]"
        )
    if train_count < thresholds.min_split_valid_count:
        reasons.append(f"train_valid_count {train_count} is below min_split_valid_count {thresholds.min_split_valid_count}")
    if eval_count < thresholds.min_split_valid_count:
        reasons.append(f"eval_valid_count {eval_count} is below min_split_valid_count {thresholds.min_split_valid_count}")
    if source_type not in INDEPENDENT_SOURCE_TYPES:
        reasons.append(f"source_type {source_type!r} is not independent enough")
    if independence_level not in PASSING_INDEPENDENCE_LEVELS:
        reasons.append(f"independence_level {independence_level!r} is not independent enough")
    if allowed_roles and not set(allowed_roles).intersection(available_eval_roles):
        reasons.append("allowed_eval_roles do not include an available evaluation split role")
    if not allowed_roles:
        reasons.append("allowed_eval_roles is empty")
    return reasons


def _blocked_row(
    registry_row: Mapping[str, str],
    status: str,
    reason: str,
) -> dict[str, object]:
    return {
        "label_column": str(registry_row.get("label_column", "")).strip(),
        "label_gate_status": status,
        "label_source": registry_row.get("label_source", ""),
        "source_type": registry_row.get("source_type", ""),
        "independence_level": registry_row.get("independence_level", ""),
        "allowed_eval_roles": registry_row.get("allowed_eval_roles", ""),
        "valid_label_count": 0,
        "missing_count": 0,
        "missing_rate": "",
        "positive_count": 0,
        "negative_count": 0,
        "positive_rate": "",
        "train_valid_count": 0,
        "eval_valid_count": 0,
        "train_positive_count": 0,
        "train_negative_count": 0,
        "eval_positive_count": 0,
        "eval_negative_count": 0,
        "decision_reason": reason,
        "claim_boundary": PHASE40_INDEPENDENT_LABEL_GATE_CLAIM_BOUNDARY,
    }


def _read_csv_table(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise ValueError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, [dict(row) for row in reader]


def _read_json_registry_rows(registry_path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Phase 40 label registry JSON is invalid: {registry_path}") from exc
    if isinstance(payload, list):
        if not all(isinstance(row, Mapping) for row in payload):
            raise ValueError(f"Phase 40 label registry JSON rows must be objects: {registry_path}")
        return [dict(row) for row in payload]
    if isinstance(payload, Mapping):
        rows = []
        for label_column, row in payload.items():
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 40 label registry JSON entry {label_column!r} is not an object: {registry_path}")
            normalized = dict(row)
            normalized.setdefault("label_column", str(label_column))
            rows.append(normalized)
        return rows
    raise ValueError(f"Phase 40 label registry JSON must be a list or object: {registry_path}")


def _parse_label(value: object, positive_definition: str) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    positive = str(positive_definition).strip().lower()
    if text == "":
        return None
    if text == positive or text in {"1", "1.0", "true", "yes", "y"}:
        return 1
    if text in {"0", "0.0", "false", "no", "n"}:
        return 0
    return None


def _split_role(value: object) -> str | None:
    split_text = str(value or "").strip().lower()
    if split_text in PHASE40_TRAIN_SPLIT_VALUES:
        return "train"
    if split_text in PHASE40_EVAL_SPLIT_VALUES:
        return "eval"
    return None


def _normalize_csvish_values(value: object) -> list[str]:
    return [item.strip().lower() for item in str(value or "").split(",") if item.strip()]


def _is_diagnostic_source(source_type: str, independence_level: str) -> bool:
    return source_type in DIAGNOSTIC_SOURCE_TYPES or independence_level in DIAGNOSTIC_INDEPENDENCE_LEVELS


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    label: str,
) -> None:
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{label} must be a list of mappings")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _csv_value(value: object) -> object:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True, allow_nan=False)
    return _json_ready(value)


def _json_ready(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _phase40_markdown(analysis: Mapping[str, object]) -> str:
    rows = analysis.get("label_gate_rows", [])
    table_rows = rows if isinstance(rows, list) else []
    lines = [
        "# Phase 40 Independent Label Gate",
        "",
        f"Status: {analysis.get('phase40_independent_label_gate_status', '')}",
        "",
        "## Label Gate Summary",
        "",
        *_markdown_table(
            (
                "label_column",
                "label_gate_status",
                "source_type",
                "independence_level",
                "valid_label_count",
                "positive_rate",
                "decision_reason",
            ),
            table_rows,
        ),
        "",
        "## Interpretation",
        "",
        str(analysis.get("interpretation", "")),
        "",
        "## Recommended Next Step",
        "",
        str(analysis.get("recommended_next_step", "")),
        "",
        "## Boundary",
        "",
        str(analysis.get("claim_boundary", PHASE40_INDEPENDENT_LABEL_GATE_CLAIM_BOUNDARY)),
        "",
    ]
    return "\n".join(lines)


def _markdown_table(fieldnames: Sequence[str], rows: Sequence[object]) -> list[str]:
    header = [str(field) for field in fieldnames]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        lines.append("| " + " | ".join(_markdown_cell(row.get(field, "")) for field in fieldnames) + " |")
    return lines


def _markdown_cell(value: object) -> str:
    return str(_csv_value(value)).replace("|", "\\|").replace("\n", " ")


def _phase40_interpretation(status: str) -> str:
    if status == "independent_label_gate_passed":
        return "At least one registered independent label passed the Phase 40 admission gate for a later Phase 38 rerun."
    if status == "independent_label_gate_diagnostic_only":
        return "At least one label is computable, but no label is independent enough to unlock suitability-reward work."
    if status == "independent_label_gate_blocked":
        return "A registry was supplied, but no label passed the independent-label gate."
    return "No usable independent-label registry was supplied."


def _phase40_next_step(status: str) -> str:
    if status == "independent_label_gate_passed":
        return "Rerun Phase 38 proxy rebuild with the accepted independent label before any B2/B3 smoke."
    return "Do not run B2/B3 or claim suitability reward readiness; obtain an independent label or frame the manuscript as diagnostic evidence."


def _round_float(value: float | int) -> float:
    return round(float(value), 10)
```

- [ ] **Step 2: Run tests**

Run:

```powershell
python -m pytest tests\test_phase40_independent_label_gate.py -q --basetemp=.pytest_tmp_phase40_t2 -p no:cacheprovider
```

Expected: all tests except CLI test pass. The CLI test fails because the runner does not exist.

- [ ] **Step 3: Commit the core module and core tests**

Run:

```powershell
git add tests\test_phase40_independent_label_gate.py src\paper11_geofm\phase40_independent_label_gate.py
git commit -m "feat: add Phase 40 independent label gate"
```

Expected: commit succeeds.

## Task 3: Thin CLI Runner

**Files:**
- Create: `experiments/phase40_independent_label_gate/run_phase40_independent_label_gate.py`
- Test: `tests/test_phase40_independent_label_gate.py`

- [ ] **Step 1: Add the runner**

Create `experiments/phase40_independent_label_gate/run_phase40_independent_label_gate.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase40_independent_label_gate import (
    run_phase40_independent_label_gate,
    write_phase40_independent_label_gate_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 40 independent-label gate."
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--label-registry", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-valid-count", type=int, default=100)
    parser.add_argument("--max-missing-rate", type=float, default=0.20)
    parser.add_argument("--min-positive-rate", type=float, default=0.02)
    parser.add_argument("--max-positive-rate", type=float, default=0.98)
    parser.add_argument("--min-split-valid-count", type=int, default=20)
    args = parser.parse_args(argv)

    try:
        analysis = run_phase40_independent_label_gate(
            phase2_output_dir=args.phase2_output_dir,
            label_registry=args.label_registry,
            min_valid_count=args.min_valid_count,
            max_missing_rate=args.max_missing_rate,
            min_positive_rate=args.min_positive_rate,
            max_positive_rate=args.max_positive_rate,
            min_split_valid_count=args.min_split_valid_count,
        )
        artifacts = write_phase40_independent_label_gate_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Phase 40 independent-label gate status: "
        f"{analysis['phase40_independent_label_gate_status']}"
    )
    print(f"Label gate CSV: {artifacts['label_gate_summary_csv']}")
    print(f"Diagnosis JSON: {artifacts['diagnosis_json']}")
    print(f"Diagnosis Markdown: {artifacts['diagnosis_md']}")
    print(f"Recommended next step: {analysis['recommended_next_step']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run CLI test**

Run:

```powershell
python -m pytest tests\test_phase40_independent_label_gate.py::test_phase40_cli_writes_outputs -q --basetemp=.pytest_tmp_phase40_cli -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 3: Run all Phase 40 tests**

Run:

```powershell
python -m pytest tests\test_phase40_independent_label_gate.py -q --basetemp=.pytest_tmp_phase40_all -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 4: Commit the runner**

Run:

```powershell
git add experiments\phase40_independent_label_gate\run_phase40_independent_label_gate.py tests\test_phase40_independent_label_gate.py
git commit -m "feat: add Phase 40 gate runner"
```

Expected: commit succeeds.

## Task 4: Real No-Registry Run And Reviewer-Facing Result

**Files:**
- Create generated local outputs under ignored `experiments/phase40_independent_label_gate/outputs/real_bishan_no_registry`
- Create: `paper/phase28_results/14_phase40_independent_label_gate.md`
- Modify: `paper/phase28_results/README.md`

- [ ] **Step 1: Run Phase 40 on real Bishan without a registry**

Run:

```powershell
python experiments\phase40_independent_label_gate\run_phase40_independent_label_gate.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase40_independent_label_gate\outputs\real_bishan_no_registry
```

Expected stdout includes:

```text
Phase 40 independent-label gate status: independent_label_inputs_missing
```

This result is acceptable and expected if no external registry has been supplied.

- [ ] **Step 2: Inspect generated JSON**

Run:

```powershell
Get-Content -Raw experiments\phase40_independent_label_gate\outputs\real_bishan_no_registry\phase40_independent_label_gate.json
```

Expected JSON fields:

```json
{
  "phase": "phase40_independent_label_gate",
  "phase40_independent_label_gate_status": "independent_label_inputs_missing"
}
```

- [ ] **Step 3: Create reviewer-facing Phase 40 result document**

Create `paper/phase28_results/14_phase40_independent_label_gate.md`:

````markdown
# Phase 40 Independent Label Gate

Phase 40 is the go/no-go gate introduced after Phase 39 to prevent Paper11
from moving into Phase 38 proxy rebuild, B2/B3 reward integration, or positive
suitability claims without a defensible independent label.

## Current Real Bishan Run

The current real run used:

```text
experiments/phase11_bishan_dltb_real/outputs/phase2_real
```

No independent label registry was supplied. The current status is:

```text
independent_label_inputs_missing
```

## Interpretation

This is not a failed policy experiment. It is a hard gate result. Paper11 still
does not have a registered independent, non-leakage suitability label that can
justify a Phase 38 proxy-rebuild rerun or any B2/B3 suitability-reward smoke.

The correct decision is therefore to stop the suitability-reward branch until
an external label source is supplied. Continuing to B2/B3 with DLTB, slope, or
source-code-derived labels would reproduce the leakage problem already
identified in Phases 36-39.

## Claim Boundary

Phase 40 does not run PPO, alter rewards, enable B2/B3, prove suitability, or
support planning-performance claims. It only records whether Paper11 has the
independent label evidence needed to continue the suitability-reward route.

## Next Step

If the authors can supply an external label registry, rerun Phase 40 first. If
Phase 40 passes, rerun Phase 38 proxy rebuild with the accepted label before
any B2/B3 reward smoke. If Phase 40 remains blocked, frame the manuscript as a
reproducible GeoFM-planning diagnostic platform with explicit negative
evidence rather than a positive suitability-reward paper.
````

- [ ] **Step 4: Update Phase 28 results README**

Modify `paper/phase28_results/README.md`:

- Add a bullet for `14_phase40_independent_label_gate.md`.
- Add the reproduction command:

```powershell
python experiments\phase40_independent_label_gate\run_phase40_independent_label_gate.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase40_independent_label_gate\outputs\real_bishan_no_registry
```

- Add the boundary:

```text
The current Phase 40 status is independent_label_inputs_missing. This is a
hard stop for suitability reward, Phase 38 rerun, and B2/B3 until an external
independent label registry is supplied and passes the gate.
```

- [ ] **Step 5: Commit result document**

Run:

```powershell
git add paper\phase28_results\README.md paper\phase28_results\14_phase40_independent_label_gate.md
git commit -m "docs: record Phase 40 independent label gate result"
```

Expected: commit succeeds.

## Task 5: Submission And Repository Documentation

**Files:**
- Modify: `README.md`
- Modify: `paper/submission/01_ijaeog_submission_readiness.md`
- Modify: `paper/submission/02_draft_titles_highlights_declarations.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [ ] **Step 1: Update README**

Add Phase 40 to the repository layout list and entry points. Add this status paragraph near Phase 39:

```text
The Phase 40 independent-label gate is a hard go/no-go check for the
suitability-reward branch. The current real Bishan no-registry run reports
`independent_label_inputs_missing`, so Phase 38 cannot be rerun with a stronger
label and B2/B3 remains blocked. This is a decision point, not a policy
performance experiment.
```

Add runner command:

```powershell
python experiments\phase40_independent_label_gate\run_phase40_independent_label_gate.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase40_independent_label_gate\outputs\real_bishan_no_registry
```

- [ ] **Step 2: Update submission readiness audit**

In `paper/submission/01_ijaeog_submission_readiness.md`, add Phase 40 to the readiness table and replace any wording that implies the next action is simply "obtain labels" with this stronger branch:

```text
Phase 40 now makes the suitability branch conditional: no Phase 38 rerun, B2/B3
reward smoke, or positive suitability-reward claim should proceed until an
external independent label registry passes the gate. The current no-registry
run remains `independent_label_inputs_missing`.
```

- [ ] **Step 3: Update draft titles, highlights, and abstract scaffold**

In `paper/submission/02_draft_titles_highlights_declarations.md`, add a guarded highlight:

```text
- An independent-label gate now blocks suitability-reward experiments unless a non-leakage external label source is registered and passes readiness checks.
```

Replace any language implying B2/B3 is only waiting for future experiments with this boundary:

```text
The suitability-reward route is not merely incomplete; it is conditionally
stopped until Phase 40 passes with an independent label registry.
```

- [ ] **Step 4: Update file manifest**

Append entries to `reproducibility/FILE_MANIFEST.tsv`:

```text
docs/superpowers/specs/2026-07-06-phase40-independent-label-gate-design.md	design	Phase 40 design for a hard independent-label go/no-go gate before Phase 38 or B2/B3.
docs/superpowers/plans/2026-07-06-phase40-independent-label-gate.md	implementation_plan	Phase 40 implementation plan.
src/paper11_geofm/phase40_independent_label_gate.py	source	Phase 40 independent-label gate module.
experiments/phase40_independent_label_gate/run_phase40_independent_label_gate.py	experiment_runner	Phase 40 independent-label gate CLI runner.
tests/test_phase40_independent_label_gate.py	test	Tests for Phase 40 registry parsing, gate status, artifact writing, and CLI behavior.
paper/phase28_results/14_phase40_independent_label_gate.md	results	Reviewer-facing interpretation of the Phase 40 real no-registry gate result.
```

- [ ] **Step 5: Update handoff**

Append a Phase 40 section to `docs/superpowers/phase33_current_progress_handoff.md`:

````markdown
## Phase 40 Independent-Label Gate

Phase 40 adds the hard go/no-go gate requested by the reviewer critique. It
does not try to rescue B2/B3 by adding another diagnostic stage. Instead, it
requires a registered independent, non-leakage label before any Phase 38 rerun
or suitability-reward smoke.

Current real no-registry status:

```text
independent_label_inputs_missing
```

Decision: B2/B3 remains blocked. The next scientifically valid action is to
provide an external independent label registry and rerun Phase 40. Without
that, Paper11 should be framed as a reproducible diagnostic platform with
negative suitability-reward readiness evidence.
````

- [ ] **Step 6: Commit documentation**

Run:

```powershell
git add README.md paper\submission\01_ijaeog_submission_readiness.md paper\submission\02_draft_titles_highlights_declarations.md reproducibility\FILE_MANIFEST.tsv docs\superpowers\phase33_current_progress_handoff.md
git commit -m "docs: update Paper11 submission gate after Phase 40"
```

Expected: commit succeeds.

## Task 6: Final Verification And Handoff

**Files:**
- Verify all edited files.

- [ ] **Step 1: Run focused Phase 40 tests**

Run:

```powershell
python -m pytest tests\test_phase40_independent_label_gate.py -q --basetemp=.pytest_tmp_phase40_final -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 2: Run regression tests for adjacent label gates**

Run:

```powershell
python -m pytest tests\test_phase39_independent_label_audit.py tests\test_phase38_proxy_rebuild.py tests\test_phase36_suitability_proxy_validation.py -q --basetemp=.pytest_tmp_phase40_adjacent -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 3: Run smoke check**

Run:

```powershell
python scripts\smoke_check.py
```

Expected stdout includes:

```text
Paper11 smoke check passed.
```

- [ ] **Step 4: Check docs for unresolved placeholders**

Run:

```powershell
$pattern = 'T' + 'BD|TO' + 'DO|REPLACE_' + 'WITH|PLACE' + 'HOLDER'
rg -n $pattern README.md paper\phase28_results\README.md paper\phase28_results\14_phase40_independent_label_gate.md paper\submission\01_ijaeog_submission_readiness.md paper\submission\02_draft_titles_highlights_declarations.md docs\superpowers\phase33_current_progress_handoff.md docs\superpowers\plans\2026-07-06-phase40-independent-label-gate.md
```

Expected: no matches.

- [ ] **Step 5: Check working tree**

Run:

```powershell
git status --short --branch
```

Expected: clean except ignored generated Phase 40 outputs.

- [ ] **Step 6: Final response**

Report:

- Phase 40 status from the real no-registry run.
- Whether Phase 40 changed the decision on B2/B3.
- Test commands and results.
- Commit hashes created during implementation.
- The remaining blocker: a real external independent label registry.
