# Phase 68 External Independent Label Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only Phase 68 package generator and preflight gate for external independent labels before any Phase 40, Phase 41, or reward-redesign rerun.

**Architecture:** Add one focused Phase 68 module that loads the real Phase 2 block universe, generates external-label templates, validates supplied block-level label CSVs and registries, classifies labels against Phase 40-compatible independence rules, and writes JSON/CSV/Markdown artifacts. A thin CLI runner exposes template-only and validation modes; the real run records a conservative result note without touching formal manuscript files.

**Tech Stack:** Python 3 standard library (`csv`, `json`, `dataclasses`, `pathlib`, `argparse`), existing Phase 40 constants and threshold semantics, pytest, PowerShell commands using `D:\adk\.venv\Scripts\python.exe`.

---

## File Structure

- Create `src/paper11_geofm/phase68_external_independent_label_package.py`
  - Owns the Phase 68 claim boundary, registry schema, template generation, external CSV loading, preflight metrics, status classification, Markdown rendering, JSON/CSV writing, and full analysis entry point.
- Create `experiments/phase68_external_independent_label_package/run_phase68_external_independent_label_package.py`
  - Thin CLI wrapper. It accepts the Phase 2 output directory, optional external CSVs, optional registry, validation-mode flag, Phase 40-compatible thresholds, and output directory.
- Create `tests/test_phase68_external_independent_label_package.py`
  - Covers template-only mode, validation-mode missing inputs, duplicate/blank `block_id` errors, diagnostic-only source blocking, ready independent labels, artifact writing, and CLI success.
- Create `paper/phase28_results/34_phase68_external_independent_label_package.md`
  - Filled after the real template-only run. It should report status, artifacts, reproduction command, and claim boundary.
- Modify `paper/phase28_results/README.md`
  - Add a one-line entry for the Phase 68 result note if the README file still has a file list section in the current worktree.
- Do not modify `paper/submission/final/*`.

---

### Task 1: Template-Only Package Skeleton

**Files:**
- Create: `tests/test_phase68_external_independent_label_package.py`
- Create: `src/paper11_geofm/phase68_external_independent_label_package.py`

- [ ] **Step 1: Write failing tests for template-only mode**

Create `tests/test_phase68_external_independent_label_package.py` with these helpers and tests:

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


def _phase2_rows(count: int = 12) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        rows.append(
            {
                "block_id": f"b{index:03d}",
                "split": "train" if index < 8 else "test",
                "explicit_feature_00": index,
            }
        )
    return rows


def _phase2_dir(tmp_path: Path, rows: list[dict[str, object]] | None = None) -> Path:
    if rows is None:
        rows = _phase2_rows()
    return _write_csv(
        tmp_path / "phase2" / "block_geofm_features.csv",
        rows,
        ["block_id", "split", "explicit_feature_00"],
    ).parent


def _external_labels(
    path: Path,
    values: list[object],
    label_column: str = "external_irrigation_label",
    block_ids: list[str] | None = None,
) -> Path:
    if block_ids is None:
        block_ids = [f"b{index:03d}" for index in range(len(values))]
    rows = [
        {"block_id": block_id, label_column: value}
        for block_id, value in zip(block_ids, values)
    ]
    return _write_csv(path, rows, ["block_id", label_column])


def _registry(
    path: Path,
    label_column: str = "external_irrigation_label",
    source_type: str = "external_irrigation",
    independence_level: str = "independent",
) -> Path:
    rows = [
        {
            "label_column": label_column,
            "label_source": "synthetic external fixture",
            "source_type": source_type,
            "independence_level": independence_level,
            "allowed_eval_roles": "test,validation,eval",
            "provenance_note": "not derived from DLTB, slope, source metadata, or GeoFM",
            "license_or_access": "test fixture",
            "expected_positive_definition": "1",
            "source_owner": "fixture owner",
            "collection_date_or_period": "2026 fixture",
            "spatial_join_method": "block_id fixture join",
            "original_unit": "block",
            "label_scale": "binary",
            "missing_value_policy": "blank means missing",
            "known_overlap_with_dltb_slope_or_source_metadata": "none",
            "contact_or_access_note": "fixture only",
        }
    ]
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
            "source_owner",
            "collection_date_or_period",
            "spatial_join_method",
            "original_unit",
            "label_scale",
            "missing_value_policy",
            "known_overlap_with_dltb_slope_or_source_metadata",
            "contact_or_access_note",
        ],
    )


def test_phase68_template_only_generates_package_ready_status_and_templates(tmp_path):
    from paper11_geofm.phase68_external_independent_label_package import (
        build_phase68_external_independent_label_package,
        write_phase68_external_independent_label_package_artifacts,
    )

    analysis = build_phase68_external_independent_label_package(
        phase2_output_dir=_phase2_dir(tmp_path),
    )

    assert analysis["phase"] == "phase68_external_independent_label_package"
    assert analysis["phase68_status"] == "external_label_package_ready"
    assert analysis["row_counts"]["phase2_block_rows"] == 12
    assert analysis["row_counts"]["template_rows"] == 12
    assert analysis["label_preflight_rows"] == []
    assert "does not train" in analysis["claim_boundary"]

    template_rows = analysis["external_label_template_rows"]
    assert template_rows[0]["block_id"] == "b000"
    assert "external_independent_label" in template_rows[0]
    registry_rows = analysis["registry_template_rows"]
    assert registry_rows[0]["source_type"] == "external_soil"

    artifacts = write_phase68_external_independent_label_package_artifacts(
        analysis,
        tmp_path / "outputs",
    )
    expected_names = {
        "phase68_external_label_template.csv",
        "phase68_label_registry_template.csv",
        "phase68_external_label_package_readme.md",
        "phase68_label_preflight.csv",
        "phase68_package_summary.csv",
        "phase68_external_independent_label_package.json",
        "phase68_external_independent_label_package.md",
    }
    assert {path.name for path in artifacts.values()} == expected_names
    readme = (tmp_path / "outputs" / "phase68_external_label_package_readme.md").read_text(
        encoding="utf-8"
    )
    assert "block_id" in readme
    assert "Phase 40" in readme
```

- [ ] **Step 2: Run the template-only test and verify it fails**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase68_external_independent_label_package.py::test_phase68_template_only_generates_package_ready_status_and_templates -q
```

Expected: FAIL with `ModuleNotFoundError` for `paper11_geofm.phase68_external_independent_label_package`.

- [ ] **Step 3: Add the minimal Phase 68 module skeleton and template writer**

Create `src/paper11_geofm/phase68_external_independent_label_package.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

from paper11_geofm.phase40_independent_label_gate import (
    DIAGNOSTIC_SOURCE_TYPES,
    INDEPENDENT_SOURCE_TYPES,
    PASSING_INDEPENDENCE_LEVELS,
    Phase40Thresholds,
)


PHASE68_EXTERNAL_LABEL_PACKAGE_CLAIM_BOUNDARY = (
    "Phase 68 builds and audits an external independent-label package before "
    "Phase 40/41 or reward-redesign work. It does not train PPO, does not alter "
    "rewards, does not enable B2/B3, does not prove suitability, and does not "
    "justify formal submission-level claims."
)

PHASE68_REGISTRY_FIELDNAMES = (
    "label_column",
    "label_source",
    "source_type",
    "independence_level",
    "allowed_eval_roles",
    "provenance_note",
    "license_or_access",
    "expected_positive_definition",
    "source_owner",
    "collection_date_or_period",
    "spatial_join_method",
    "original_unit",
    "label_scale",
    "missing_value_policy",
    "known_overlap_with_dltb_slope_or_source_metadata",
    "contact_or_access_note",
)

PHASE68_EXTERNAL_LABEL_TEMPLATE_FIELDNAMES = (
    "block_id",
    "external_independent_label",
    "label_missing_reason",
    "source_record_id",
)

PHASE68_LABEL_PREFLIGHT_FIELDNAMES = (
    "label_column",
    "label_preflight_status",
    "label_source",
    "source_type",
    "independence_level",
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
    "unjoined_external_count",
    "decision_reason",
    "claim_boundary",
)

PHASE68_PACKAGE_SUMMARY_FIELDNAMES = (
    "phase68_status",
    "phase2_block_rows",
    "external_label_csv_count",
    "registry_rows",
    "ready_label_count",
    "invalid_label_count",
    "diagnostic_label_count",
    "recommended_next_step",
    "claim_boundary",
)


def build_phase68_external_independent_label_package(
    phase2_output_dir: Path | str,
    external_label_csvs: Sequence[Path | str] | Path | str | None = None,
    label_registry: Path | str | None = None,
    validation_mode: bool = False,
    min_valid_count: int = 100,
    max_missing_rate: float = 0.20,
    min_positive_rate: float = 0.02,
    max_positive_rate: float = 0.98,
    min_split_valid_count: int = 20,
) -> dict[str, object]:
    phase2_dir = Path(phase2_output_dir)
    phase2_csv = phase2_dir / "block_geofm_features.csv"
    phase2_fieldnames, phase2_rows = _read_csv_table(phase2_csv, "Phase 2 block feature CSV")
    if "block_id" not in phase2_fieldnames:
        raise ValueError(f"Phase 68 Phase 2 CSV is missing block_id: {phase2_csv}")
    if "split" not in phase2_fieldnames:
        raise ValueError(f"Phase 68 Phase 2 CSV is missing split: {phase2_csv}")

    external_paths = _normalize_paths(external_label_csvs)
    thresholds = Phase40Thresholds(
        min_valid_count=min_valid_count,
        max_missing_rate=max_missing_rate,
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
        min_split_valid_count=min_split_valid_count,
    )
    template_rows = _external_label_template_rows(phase2_rows)
    registry_template_rows = [_default_registry_template_row()]

    if not external_paths and label_registry is None:
        status = "external_label_inputs_missing" if validation_mode else "external_label_package_ready"
        label_preflight_rows: list[dict[str, object]] = []
        registry_rows: list[dict[str, str]] = []
    else:
        label_preflight_rows = []
        registry_rows = []
        status = "external_label_inputs_invalid"

    summary_rows = [
        _package_summary_row(
            status=status,
            phase2_block_rows=len(phase2_rows),
            external_label_csv_count=len(external_paths),
            registry_rows=len(registry_rows),
            label_preflight_rows=label_preflight_rows,
        )
    ]
    return {
        "phase": "phase68_external_independent_label_package",
        "phase68_status": status,
        "source_paths": {
            "phase2_output_dir": str(phase2_dir),
            "phase2_block_geofm_features_csv": str(phase2_csv),
            "external_label_csvs": [str(path) for path in external_paths],
            "label_registry": str(Path(label_registry)) if label_registry is not None else None,
        },
        "thresholds": thresholds.__dict__,
        "row_counts": {
            "phase2_block_rows": len(phase2_rows),
            "template_rows": len(template_rows),
            "external_label_csvs": len(external_paths),
            "registry_rows": len(registry_rows),
            "label_preflight_rows": len(label_preflight_rows),
        },
        "external_label_template_rows": template_rows,
        "registry_template_rows": registry_template_rows,
        "label_preflight_rows": label_preflight_rows,
        "package_summary_rows": summary_rows,
        "interpretation": _phase68_interpretation(status),
        "recommended_next_step": _phase68_next_step(status),
        "claim_boundary": PHASE68_EXTERNAL_LABEL_PACKAGE_CLAIM_BOUNDARY,
    }


def write_phase68_external_independent_label_package_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "external_label_template_csv": output_path / "phase68_external_label_template.csv",
        "label_registry_template_csv": output_path / "phase68_label_registry_template.csv",
        "package_readme_md": output_path / "phase68_external_label_package_readme.md",
        "label_preflight_csv": output_path / "phase68_label_preflight.csv",
        "package_summary_csv": output_path / "phase68_package_summary.csv",
        "diagnosis_json": output_path / "phase68_external_independent_label_package.json",
        "diagnosis_md": output_path / "phase68_external_independent_label_package.md",
    }
    _write_csv_mapping_rows(
        artifacts["external_label_template_csv"],
        PHASE68_EXTERNAL_LABEL_TEMPLATE_FIELDNAMES,
        analysis.get("external_label_template_rows", []),
        "Phase 68 external label template rows",
    )
    _write_csv_mapping_rows(
        artifacts["label_registry_template_csv"],
        PHASE68_REGISTRY_FIELDNAMES,
        analysis.get("registry_template_rows", []),
        "Phase 68 registry template rows",
    )
    _write_csv_mapping_rows(
        artifacts["label_preflight_csv"],
        PHASE68_LABEL_PREFLIGHT_FIELDNAMES,
        analysis.get("label_preflight_rows", []),
        "Phase 68 label preflight rows",
    )
    _write_csv_mapping_rows(
        artifacts["package_summary_csv"],
        PHASE68_PACKAGE_SUMMARY_FIELDNAMES,
        analysis.get("package_summary_rows", []),
        "Phase 68 package summary rows",
    )
    artifacts["package_readme_md"].write_text(_phase68_package_readme(), encoding="utf-8")
    artifacts["diagnosis_json"].write_text(
        json.dumps(_json_ready(analysis), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    artifacts["diagnosis_md"].write_text(_phase68_markdown(analysis), encoding="utf-8")
    return artifacts


def _read_csv_table(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise ValueError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _normalize_paths(paths: Sequence[Path | str] | Path | str | None) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _external_label_template_rows(phase2_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "block_id": str(row.get("block_id", "")).strip(),
            "external_independent_label": "",
            "label_missing_reason": "",
            "source_record_id": "",
        }
        for row in phase2_rows
    ]


def _default_registry_template_row() -> dict[str, str]:
    return {
        "label_column": "external_independent_label",
        "label_source": "name_of_external_dataset_or_survey",
        "source_type": "external_soil",
        "independence_level": "independent",
        "allowed_eval_roles": "test,validation,eval",
        "provenance_note": "Describe why this label is not derived from DLTB, slope, source metadata, explicit planning features, GeoFM, or model predictions.",
        "license_or_access": "Describe license, access permission, or restricted-access handling.",
        "expected_positive_definition": "1",
        "source_owner": "organization_or_data_owner",
        "collection_date_or_period": "YYYY or YYYY-YYYY",
        "spatial_join_method": "block_id_join",
        "original_unit": "block",
        "label_scale": "binary",
        "missing_value_policy": "blank means missing",
        "known_overlap_with_dltb_slope_or_source_metadata": "none",
        "contact_or_access_note": "contact or access note",
    }


def _package_summary_row(
    status: str,
    phase2_block_rows: int,
    external_label_csv_count: int,
    registry_rows: int,
    label_preflight_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "phase68_status": status,
        "phase2_block_rows": phase2_block_rows,
        "external_label_csv_count": external_label_csv_count,
        "registry_rows": registry_rows,
        "ready_label_count": sum(
            1 for row in label_preflight_rows if row.get("label_preflight_status") == "label_ready_for_phase40"
        ),
        "invalid_label_count": sum(
            1 for row in label_preflight_rows if row.get("label_preflight_status") == "label_inputs_invalid"
        ),
        "diagnostic_label_count": sum(
            1 for row in label_preflight_rows if row.get("label_preflight_status") == "label_diagnostic_only"
        ),
        "recommended_next_step": _phase68_next_step(status),
        "claim_boundary": PHASE68_EXTERNAL_LABEL_PACKAGE_CLAIM_BOUNDARY,
    }


def _write_csv_mapping_rows(path: Path, fieldnames: Sequence[str], rows: object, label: str) -> None:
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


def _phase68_package_readme() -> str:
    return "\n".join(
        [
            "# Phase 68 External Label Package",
            "",
            "Provide a block-level CSV with `block_id` and one or more external independent label columns.",
            "The `block_id` values must come from the Paper11 Phase 2 `block_geofm_features.csv` table.",
            "Complete `phase68_label_registry_template.csv` so Phase 68 can check Phase 40-compatible source type, independence level, positive definition, access, and provenance.",
            "Acceptable sources include field survey, soil, irrigation, yield, high-standard-farmland, retention, policy outcome, or independent remote-sensing products.",
            "DLTB-derived, slope-derived, source-metadata-derived, GeoFM-derived, or model-generated labels remain diagnostic-only.",
            "",
        ]
    )


def _phase68_markdown(analysis: Mapping[str, object]) -> str:
    rows = analysis.get("label_preflight_rows", [])
    table_rows = rows if isinstance(rows, list) else []
    lines = [
        "# Phase 68 External Independent Label Package",
        "",
        f"Status: {analysis.get('phase68_status', '')}",
        "",
        "## Label Preflight",
        "",
        *_markdown_table(
            (
                "label_column",
                "label_preflight_status",
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
        str(analysis.get("claim_boundary", PHASE68_EXTERNAL_LABEL_PACKAGE_CLAIM_BOUNDARY)),
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


def _phase68_interpretation(status: str) -> str:
    if status == "external_label_package_ready":
        return "Templates and documentation are ready for an external independent label provider."
    if status == "external_label_inputs_missing":
        return "Validation mode was requested, but an external label CSV or registry is missing."
    if status == "phase40_ready_to_rerun_with_external_label":
        return "At least one supplied external label appears ready for a Phase 40 rerun."
    if status == "independent_label_route_blocked":
        return "Supplied labels are diagnostic-only or leakage-risk and cannot unlock Phase 40/41."
    return "Supplied external label inputs failed Phase 68 preflight checks."


def _phase68_next_step(status: str) -> str:
    if status == "external_label_package_ready":
        return "Provide a completed external label CSV and registry, then rerun Phase 68 in validation mode."
    if status == "external_label_inputs_missing":
        return "Supply both an external label CSV and a Phase 40-compatible registry, then rerun validation mode."
    if status == "phase40_ready_to_rerun_with_external_label":
        return "Rerun Phase 40 with the accepted external label registry before Phase 41 or reward redesign."
    return "Do not run B2/B3 or reward redesign; fix the external label package first."
```

- [ ] **Step 4: Run the template-only test and verify it passes**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase68_external_independent_label_package.py::test_phase68_template_only_generates_package_ready_status_and_templates -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src\paper11_geofm\phase68_external_independent_label_package.py tests\test_phase68_external_independent_label_package.py
git commit -m "feat: add Phase 68 external label package templates"
```

Expected: commit succeeds.

---

### Task 2: External CSV And Registry Validation

**Files:**
- Modify: `tests/test_phase68_external_independent_label_package.py`
- Modify: `src/paper11_geofm/phase68_external_independent_label_package.py`

- [ ] **Step 1: Add failing validation tests**

Append these tests to `tests/test_phase68_external_independent_label_package.py`:

```python
def test_phase68_validation_mode_without_inputs_reports_missing(tmp_path):
    from paper11_geofm.phase68_external_independent_label_package import (
        build_phase68_external_independent_label_package,
    )

    analysis = build_phase68_external_independent_label_package(
        phase2_output_dir=_phase2_dir(tmp_path),
        validation_mode=True,
    )

    assert analysis["phase68_status"] == "external_label_inputs_missing"
    assert analysis["row_counts"]["label_preflight_rows"] == 0
    assert "missing" in analysis["recommended_next_step"].lower()


def test_phase68_external_csv_rejects_blank_block_id(tmp_path):
    from paper11_geofm.phase68_external_independent_label_package import (
        build_phase68_external_independent_label_package,
    )

    labels = _external_labels(
        tmp_path / "labels.csv",
        [1, 0],
        block_ids=["b000", ""],
    )
    registry = _registry(tmp_path / "registry.csv")

    try:
        build_phase68_external_independent_label_package(
            phase2_output_dir=_phase2_dir(tmp_path),
            external_label_csvs=labels,
            label_registry=registry,
            validation_mode=True,
        )
    except ValueError as exc:
        assert "blank block_id" in str(exc)
    else:
        raise AssertionError("Expected blank block_id to raise ValueError")


def test_phase68_external_csv_rejects_duplicate_block_id(tmp_path):
    from paper11_geofm.phase68_external_independent_label_package import (
        build_phase68_external_independent_label_package,
    )

    labels = _external_labels(
        tmp_path / "labels.csv",
        [1, 0],
        block_ids=["b000", "b000"],
    )
    registry = _registry(tmp_path / "registry.csv")

    try:
        build_phase68_external_independent_label_package(
            phase2_output_dir=_phase2_dir(tmp_path),
            external_label_csvs=labels,
            label_registry=registry,
            validation_mode=True,
        )
    except ValueError as exc:
        assert "duplicate block_id b000" in str(exc)
    else:
        raise AssertionError("Expected duplicate block_id to raise ValueError")
```

- [ ] **Step 2: Run validation tests and verify they fail**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase68_external_independent_label_package.py -q
```

Expected: new blank/duplicate `block_id` tests fail because supplied external CSVs are not loaded yet.

- [ ] **Step 3: Add registry and external CSV loaders**

In `src/paper11_geofm/phase68_external_independent_label_package.py`, add these functions below `_normalize_paths`:

```python
def _load_external_label_csvs(
    phase2_rows: Sequence[Mapping[str, str]],
    external_paths: Sequence[Path],
) -> tuple[dict[str, dict[str, str]], dict[str, list[str]], dict[str, int]]:
    phase2_block_ids = {
        str(row.get("block_id", "")).strip()
        for row in phase2_rows
        if str(row.get("block_id", "")).strip()
    }
    values_by_label: dict[str, dict[str, str]] = {}
    sources_by_label: dict[str, list[str]] = {}
    unjoined_by_label: dict[str, int] = {}
    for external_path in external_paths:
        fieldnames, external_rows = _read_csv_table(external_path, "Phase 68 external label CSV")
        if "block_id" not in fieldnames:
            raise ValueError(f"Phase 68 external label CSV is missing block_id: {external_path}")
        label_columns = [field for field in fieldnames if field != "block_id"]
        seen_block_ids: set[str] = set()
        for external_row in external_rows:
            block_id = str(external_row.get("block_id", "")).strip()
            if not block_id:
                raise ValueError(f"Phase 68 external label CSV contains a blank block_id: {external_path}")
            if block_id in seen_block_ids:
                raise ValueError(
                    f"Phase 68 external label CSV has duplicate block_id {block_id}: {external_path}"
                )
            seen_block_ids.add(block_id)
            joined = block_id in phase2_block_ids
            for label_column in label_columns:
                sources_by_label.setdefault(label_column, []).append(str(external_path))
                if not joined:
                    unjoined_by_label[label_column] = unjoined_by_label.get(label_column, 0) + 1
                    continue
                values_by_label.setdefault(label_column, {})[block_id] = str(external_row.get(label_column, ""))
    return values_by_label, sources_by_label, unjoined_by_label


def _load_phase68_registry(label_registry: Path | str | None) -> list[dict[str, str]]:
    if label_registry is None:
        return []
    registry_path = Path(label_registry)
    if not registry_path.exists():
        raise ValueError(f"Missing Phase 68 label registry: {registry_path}")
    if registry_path.suffix.lower() == ".csv":
        _, rows = _read_csv_table(registry_path, "Phase 68 label registry CSV")
    elif registry_path.suffix.lower() == ".json":
        rows = _read_json_registry_rows(registry_path)
    else:
        raise ValueError(f"Unsupported Phase 68 label registry extension: {registry_path}")
    normalized_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=2):
        normalized = {field: str(row.get(field, "")).strip() for field in PHASE68_REGISTRY_FIELDNAMES}
        if not normalized["label_column"]:
            raise ValueError(f"Phase 68 label registry row {index} has blank label_column: {registry_path}")
        normalized_rows.append(normalized)
    return normalized_rows


def _read_json_registry_rows(registry_path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Phase 68 label registry JSON is invalid: {registry_path}") from exc
    if isinstance(payload, list):
        if not all(isinstance(row, Mapping) for row in payload):
            raise ValueError(f"Phase 68 label registry JSON rows must be objects: {registry_path}")
        return [dict(row) for row in payload]
    if isinstance(payload, Mapping):
        rows: list[dict[str, object]] = []
        for label_column, row in payload.items():
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"Phase 68 label registry JSON entry {label_column!r} is not an object: {registry_path}"
                )
            normalized = dict(row)
            normalized.setdefault("label_column", str(label_column))
            rows.append(normalized)
        return rows
    raise ValueError(f"Phase 68 label registry JSON must be a list or object: {registry_path}")
```

Then replace the `else:` branch in `build_phase68_external_independent_label_package` with:

```python
    elif validation_mode and (not external_paths or label_registry is None):
        status = "external_label_inputs_missing"
        label_preflight_rows = []
        registry_rows = []
    else:
        external_values_by_label, sources_by_label, unjoined_by_label = _load_external_label_csvs(
            phase2_rows,
            external_paths,
        )
        registry_rows = _load_phase68_registry(label_registry)
        label_preflight_rows = []
        status = "external_label_inputs_invalid"
```

- [ ] **Step 4: Run validation tests and verify they pass**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase68_external_independent_label_package.py -q
```

Expected: all current Phase 68 tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src\paper11_geofm\phase68_external_independent_label_package.py tests\test_phase68_external_independent_label_package.py
git commit -m "feat: add Phase 68 external label input validation"
```

Expected: commit succeeds.

---

### Task 3: Preflight Metrics And Conservative Status Gate

**Files:**
- Modify: `tests/test_phase68_external_independent_label_package.py`
- Modify: `src/paper11_geofm/phase68_external_independent_label_package.py`

- [ ] **Step 1: Add failing preflight status tests**

Append these tests:

```python
def test_phase68_diagnostic_label_is_blocked_from_phase40_ready_status(tmp_path):
    from paper11_geofm.phase68_external_independent_label_package import (
        build_phase68_external_independent_label_package,
    )

    labels = _external_labels(tmp_path / "labels.csv", [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    registry = _registry(
        tmp_path / "registry.csv",
        source_type="dltb_derived",
        independence_level="leakage_risk",
    )

    analysis = build_phase68_external_independent_label_package(
        phase2_output_dir=_phase2_dir(tmp_path),
        external_label_csvs=labels,
        label_registry=registry,
        validation_mode=True,
        min_valid_count=10,
        min_split_valid_count=2,
    )

    assert analysis["phase68_status"] == "independent_label_route_blocked"
    row = analysis["label_preflight_rows"][0]
    assert row["label_preflight_status"] == "label_diagnostic_only"
    assert "not independent enough" in row["decision_reason"]


def test_phase68_valid_independent_label_is_ready_for_phase40(tmp_path):
    from paper11_geofm.phase68_external_independent_label_package import (
        build_phase68_external_independent_label_package,
    )

    labels = _external_labels(tmp_path / "labels.csv", [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    registry = _registry(tmp_path / "registry.csv")

    analysis = build_phase68_external_independent_label_package(
        phase2_output_dir=_phase2_dir(tmp_path),
        external_label_csvs=labels,
        label_registry=registry,
        validation_mode=True,
        min_valid_count=10,
        min_split_valid_count=2,
    )

    assert analysis["phase68_status"] == "phase40_ready_to_rerun_with_external_label"
    row = analysis["label_preflight_rows"][0]
    assert row["label_preflight_status"] == "label_ready_for_phase40"
    assert row["valid_label_count"] == 12
    assert row["missing_count"] == 0
    assert row["positive_count"] == 6
    assert row["negative_count"] == 6
    assert row["train_positive_count"] == 4
    assert row["eval_positive_count"] == 2
```

- [ ] **Step 2: Run preflight tests and verify they fail**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase68_external_independent_label_package.py -q
```

Expected: new preflight tests fail because `label_preflight_rows` are not computed yet.

- [ ] **Step 3: Add preflight row computation and status summarization**

In the module, add these functions after `_load_phase68_registry`:

```python
PHASE68_TRAIN_SPLIT_VALUES = {"train", "training"}
PHASE68_EVAL_SPLIT_VALUES = {"test", "eval", "evaluation", "validation", "val"}


def _build_label_preflight_rows(
    phase2_rows: Sequence[Mapping[str, str]],
    registry_rows: Sequence[Mapping[str, str]],
    external_values_by_label: Mapping[str, Mapping[str, str]],
    sources_by_label: Mapping[str, Sequence[str]],
    unjoined_by_label: Mapping[str, int],
    thresholds: Phase40Thresholds,
) -> list[dict[str, object]]:
    return [
        _label_preflight_row(
            phase2_rows=phase2_rows,
            registry_row=registry_row,
            external_values_by_label=external_values_by_label,
            sources_by_label=sources_by_label,
            unjoined_by_label=unjoined_by_label,
            thresholds=thresholds,
        )
        for registry_row in registry_rows
    ]


def _label_preflight_row(
    phase2_rows: Sequence[Mapping[str, str]],
    registry_row: Mapping[str, str],
    external_values_by_label: Mapping[str, Mapping[str, str]],
    sources_by_label: Mapping[str, Sequence[str]],
    unjoined_by_label: Mapping[str, int],
    thresholds: Phase40Thresholds,
) -> dict[str, object]:
    label_column = str(registry_row.get("label_column", "")).strip()
    values_by_block = external_values_by_label.get(label_column, {})
    positive_definition = str(registry_row.get("expected_positive_definition", "1")).strip() or "1"
    labels: list[int] = []
    train_labels: list[int] = []
    eval_labels: list[int] = []
    missing_count = 0
    for phase2_row in phase2_rows:
        block_id = str(phase2_row.get("block_id", "")).strip()
        parsed = _parse_label(values_by_block.get(block_id), positive_definition)
        if parsed is None:
            missing_count += 1
            continue
        labels.append(parsed)
        split_role = _split_role(phase2_row.get("split"))
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
    missing_rate = missing_count / len(phase2_rows) if phase2_rows else 1.0
    positive_rate = positive_count / valid_count if valid_count else 0.0
    source_type = str(registry_row.get("source_type", "")).strip()
    independence_level = str(registry_row.get("independence_level", "")).strip()
    failure_reasons = _preflight_failure_reasons(
        label_column=label_column,
        valid_count=valid_count,
        missing_rate=missing_rate,
        positive_rate=positive_rate,
        train_count=len(train_labels),
        eval_count=len(eval_labels),
        train_positive=train_positive,
        train_negative=train_negative,
        eval_positive=eval_positive,
        eval_negative=eval_negative,
        source_type=source_type,
        independence_level=independence_level,
        thresholds=thresholds,
        sources_by_label=sources_by_label,
        unjoined_external_count=int(unjoined_by_label.get(label_column, 0)),
    )
    if not failure_reasons:
        status = "label_ready_for_phase40"
        reason = "label passed Phase 68 preflight and is ready for Phase 40 rerun"
    elif source_type in DIAGNOSTIC_SOURCE_TYPES or independence_level not in PASSING_INDEPENDENCE_LEVELS:
        status = "label_diagnostic_only"
        reason = "label is not independent enough for Phase 40: " + "; ".join(failure_reasons)
    else:
        status = "label_inputs_invalid"
        reason = "; ".join(failure_reasons)
    return {
        "label_column": label_column,
        "label_preflight_status": status,
        "label_source": registry_row.get("label_source", ""),
        "source_type": source_type,
        "independence_level": independence_level,
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
        "unjoined_external_count": int(unjoined_by_label.get(label_column, 0)),
        "decision_reason": reason,
        "claim_boundary": PHASE68_EXTERNAL_LABEL_PACKAGE_CLAIM_BOUNDARY,
    }


def _preflight_failure_reasons(
    label_column: str,
    valid_count: int,
    missing_rate: float,
    positive_rate: float,
    train_count: int,
    eval_count: int,
    train_positive: int,
    train_negative: int,
    eval_positive: int,
    eval_negative: int,
    source_type: str,
    independence_level: str,
    thresholds: Phase40Thresholds,
    sources_by_label: Mapping[str, Sequence[str]],
    unjoined_external_count: int,
) -> list[str]:
    reasons: list[str] = []
    if label_column not in sources_by_label:
        reasons.append(f"label column {label_column!r} is missing from external label CSVs")
    if unjoined_external_count > 0:
        reasons.append(f"{unjoined_external_count} external rows did not join to Phase 2 block_id values")
    if valid_count < thresholds.min_valid_count:
        reasons.append(f"valid_label_count {valid_count} is below min_valid_count {thresholds.min_valid_count}")
    if missing_rate > thresholds.max_missing_rate:
        reasons.append(
            f"missing_rate {missing_rate:.10f} exceeds max_missing_rate {thresholds.max_missing_rate:.10f}"
        )
    if positive_rate < thresholds.min_positive_rate or positive_rate > thresholds.max_positive_rate:
        reasons.append(
            f"positive_rate {positive_rate:.10f} is outside [{thresholds.min_positive_rate:.10f}, {thresholds.max_positive_rate:.10f}]"
        )
    if train_count < thresholds.min_split_valid_count:
        reasons.append(f"train_valid_count {train_count} is below min_split_valid_count {thresholds.min_split_valid_count}")
    if eval_count < thresholds.min_split_valid_count:
        reasons.append(f"eval_valid_count {eval_count} is below min_split_valid_count {thresholds.min_split_valid_count}")
    if train_positive == 0 or train_negative == 0:
        reasons.append("train split does not contain both positive and negative labels")
    if eval_positive == 0 or eval_negative == 0:
        reasons.append("evaluation split does not contain both positive and negative labels")
    if source_type not in INDEPENDENT_SOURCE_TYPES:
        reasons.append(f"source_type {source_type!r} is not independent enough")
    if independence_level not in PASSING_INDEPENDENCE_LEVELS:
        reasons.append(f"independence_level {independence_level!r} is not independent enough")
    return reasons


def _phase68_status(label_preflight_rows: Sequence[Mapping[str, object]]) -> str:
    if any(row.get("label_preflight_status") == "label_ready_for_phase40" for row in label_preflight_rows):
        return "phase40_ready_to_rerun_with_external_label"
    if label_preflight_rows and all(row.get("label_preflight_status") == "label_diagnostic_only" for row in label_preflight_rows):
        return "independent_label_route_blocked"
    if label_preflight_rows:
        return "external_label_inputs_invalid"
    return "external_label_inputs_invalid"


def _split_role(value: object) -> str | None:
    split_text = str(value or "").strip().lower()
    if split_text in PHASE68_TRAIN_SPLIT_VALUES:
        return "train"
    if split_text in PHASE68_EVAL_SPLIT_VALUES:
        return "eval"
    return None


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


def _round_float(value: float | int) -> float:
    return round(float(value), 10)
```

Then replace the validation branch in `build_phase68_external_independent_label_package` with:

```python
        label_preflight_rows = _build_label_preflight_rows(
            phase2_rows=phase2_rows,
            registry_rows=registry_rows,
            external_values_by_label=external_values_by_label,
            sources_by_label=sources_by_label,
            unjoined_by_label=unjoined_by_label,
            thresholds=thresholds,
        )
        status = _phase68_status(label_preflight_rows)
```

- [ ] **Step 4: Run all Phase 68 tests and verify they pass**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase68_external_independent_label_package.py -q
```

Expected: all current Phase 68 tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add src\paper11_geofm\phase68_external_independent_label_package.py tests\test_phase68_external_independent_label_package.py
git commit -m "feat: add Phase 68 external label preflight gate"
```

Expected: commit succeeds.

---

### Task 4: CLI Runner And Artifact Regression

**Files:**
- Create: `experiments/phase68_external_independent_label_package/run_phase68_external_independent_label_package.py`
- Modify: `tests/test_phase68_external_independent_label_package.py`

- [ ] **Step 1: Add failing CLI test**

Append:

```python
def test_phase68_runner_template_only_cli(tmp_path):
    script = (
        ROOT
        / "experiments"
        / "phase68_external_independent_label_package"
        / "run_phase68_external_independent_label_package.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--phase2-output-dir",
            str(_phase2_dir(tmp_path)),
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Phase 68 external-label package status: external_label_package_ready" in result.stdout
    assert (tmp_path / "outputs" / "phase68_external_independent_label_package.json").exists()
```

- [ ] **Step 2: Run CLI test and verify it fails**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase68_external_independent_label_package.py::test_phase68_runner_template_only_cli -q
```

Expected: FAIL because the runner does not exist yet.

- [ ] **Step 3: Add the runner**

Create `experiments/phase68_external_independent_label_package/run_phase68_external_independent_label_package.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase68_external_independent_label_package import (
    build_phase68_external_independent_label_package,
    write_phase68_external_independent_label_package_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 68 external independent-label package preflight."
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--external-label-csvs", default="")
    parser.add_argument("--label-registry", type=Path)
    parser.add_argument("--validation-mode", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-valid-count", type=int, default=100)
    parser.add_argument("--max-missing-rate", type=float, default=0.20)
    parser.add_argument("--min-positive-rate", type=float, default=0.02)
    parser.add_argument("--max-positive-rate", type=float, default=0.98)
    parser.add_argument("--min-split-valid-count", type=int, default=20)
    args = parser.parse_args(argv)

    try:
        analysis = build_phase68_external_independent_label_package(
            phase2_output_dir=args.phase2_output_dir,
            external_label_csvs=_parse_optional_paths(args.external_label_csvs),
            label_registry=args.label_registry,
            validation_mode=args.validation_mode,
            min_valid_count=args.min_valid_count,
            max_missing_rate=args.max_missing_rate,
            min_positive_rate=args.min_positive_rate,
            max_positive_rate=args.max_positive_rate,
            min_split_valid_count=args.min_split_valid_count,
        )
        artifacts = write_phase68_external_independent_label_package_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 68 external-label package status: {analysis['phase68_status']}")
    print(f"External label template CSV: {artifacts['external_label_template_csv']}")
    print(f"Label registry template CSV: {artifacts['label_registry_template_csv']}")
    print(f"Package README: {artifacts['package_readme_md']}")
    print(f"Label preflight CSV: {artifacts['label_preflight_csv']}")
    print(f"Package summary CSV: {artifacts['package_summary_csv']}")
    print(f"Diagnosis JSON: {artifacts['diagnosis_json']}")
    print(f"Diagnosis Markdown: {artifacts['diagnosis_md']}")
    print(f"Recommended next step: {analysis['recommended_next_step']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _parse_optional_paths(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run Phase 68 tests and adjacent independent-label regressions**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase68_external_independent_label_package.py tests\test_phase40_independent_label_gate.py tests\test_phase39_independent_label_audit.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add experiments\phase68_external_independent_label_package\run_phase68_external_independent_label_package.py tests\test_phase68_external_independent_label_package.py
git commit -m "feat: add Phase 68 external label package runner"
```

Expected: commit succeeds.

---

### Task 5: Real Template-Only Run And Result Note

**Files:**
- Create: `paper/phase28_results/34_phase68_external_independent_label_package.md`
- Modify: `paper/phase28_results/README.md`

- [ ] **Step 1: Run the real Phase 68 template-only package**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase68_external_independent_label_package\run_phase68_external_independent_label_package.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase68_external_independent_label_package\outputs\real_bishan_template_only
```

Expected: exit code `0`, console prints `Phase 68 external-label package status: external_label_package_ready`, artifact paths, recommended next step, and claim boundary.

- [ ] **Step 2: Inspect the real JSON status**

Run:

```powershell
Get-Content -Raw experiments\phase68_external_independent_label_package\outputs\real_bishan_template_only\phase68_external_independent_label_package.json
```

Expected: JSON contains:

```json
{
  "phase": "phase68_external_independent_label_package",
  "phase68_status": "external_label_package_ready"
}
```

The exact JSON includes additional fields; the status and phase must match.

- [ ] **Step 3: Add the Phase 68 result note**

Create `paper/phase28_results/34_phase68_external_independent_label_package.md`:

```markdown
# Phase 68 External Independent Label Package

Status: external_label_package_ready

## Key Evidence

- Phase 68 generated an external-label package template for the real Bishan Phase 2 block universe.
- The run is template-only because no external independent label CSV or registry has been supplied yet.
- The package includes a block-level label CSV template, Phase 40-compatible registry template, external data README, preflight CSV, summary CSV, JSON diagnosis, and Markdown diagnosis.
- The next valid algorithm route is to provide a completed external label CSV and registry, rerun Phase 68 in validation mode, and only then rerun Phase 40 if at least one label passes preflight.

## Reproduction

Run from the repository root:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase68_external_independent_label_package\run_phase68_external_independent_label_package.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase68_external_independent_label_package\outputs\real_bishan_template_only
```

## Boundary

Phase 68 builds and audits an external independent-label package before Phase 40/41 or reward-redesign work. It does not train PPO, alter rewards, enable B2/B3, prove suitability, or justify formal submission-level claims.
```

- [ ] **Step 4: Add README entry if appropriate**

If `paper/phase28_results/README.md` still contains the numbered file list, add this bullet after the Phase 67 entry or at the end of the list:

```markdown
- `34_phase68_external_independent_label_package.md`: external independent-label package and preflight contract showing that templates are ready, but no external independent label has been supplied yet.
```

If the README does not have a current Phase 67 entry, add the Phase 68 bullet near the end of the existing file list without rewriting unrelated entries.

- [ ] **Step 5: Commit Task 5**

Run:

```powershell
git add paper\phase28_results\34_phase68_external_independent_label_package.md paper\phase28_results\README.md
git commit -m "docs: record Phase 68 external label package result"
```

Expected: commit succeeds.

---

### Task 6: Final Verification And Push

**Files:**
- No new files expected beyond prior tasks.

- [ ] **Step 1: Run targeted regression suite**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase68_external_independent_label_package.py tests\test_phase40_independent_label_gate.py tests\test_phase39_independent_label_audit.py tests\test_phase67_candidate_reward_label_target_audit.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run smoke check**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe scripts\paper11_smoke_check.py
```

Expected: prints `Paper11 smoke check passed`.

- [ ] **Step 3: Check whitespace and formal manuscript untouched**

Run:

```powershell
git diff --check
git diff --name-only HEAD -- paper\submission\final
```

Expected: `git diff --check` prints nothing. Formal manuscript diff command prints nothing.

- [ ] **Step 4: Check repository status**

Run:

```powershell
git status --short --branch
git log -1 --oneline
```

Expected: worktree is clean except any intentionally unpushed commits on `main`.

- [ ] **Step 5: Push completed Phase 68 work**

Run:

```powershell
git push
```

Expected: push succeeds and `main` is synchronized with `origin/main`.

---

## Self-Review Checklist

- Spec coverage:
  - Template CSV, registry template, README, JSON, CSV, and Markdown artifacts are covered in Tasks 1 and 4.
  - Template-only and validation modes are covered in Tasks 1, 2, and 4.
  - Schema, join, label parsing, missingness, balance, split coverage, and source independence checks are covered in Tasks 2 and 3.
  - Phase 40-compatible thresholds are carried through the module and CLI.
  - The real run result note and formal manuscript boundary are covered in Tasks 5 and 6.
- Deferred-marker scan:
  - The plan contains no deferred implementation markers and no unspecified test commands.
- Type consistency:
  - Public functions are `build_phase68_external_independent_label_package` and `write_phase68_external_independent_label_package_artifacts`.
  - Public status key is `phase68_status`.
  - Per-label status key is `label_preflight_status`.
  - Runner imports match the module names and artifact keys listed in Task 1.
