# Phase 39 Independent Label Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible independent-label audit that decides whether Paper11 has any non-leakage label ready for a Phase 38 rerun.

**Architecture:** Follow the Phase 36/38 pattern: one pure analysis module, one artifact writer, one thin CLI runner, focused pytest coverage, and documentation after the real Bishan run. The module loads Phase 2 block features, optionally joins external labels by `block_id`, applies a registry-based provenance gate, checks train/evaluation usability, writes CSV/JSON/Markdown artifacts, and emits one conservative status.

**Tech Stack:** Python standard library (`csv`, `json`, `pathlib`), pytest, existing Paper11 experiment runner layout, existing ignored `experiments/**/outputs/` convention.

---

## File Structure

- Create: `tests/test_phase39_independent_label_audit.py`
  - Synthetic fixtures for Phase 2 block features, registry files, external labels, builder statuses, writer artifacts, and CLI behavior.
- Create: `src/paper11_geofm/phase39_independent_label_audit.py`
  - Owns constants, CSV loading, registry parsing, external label joins, provenance classification, readiness reduction, artifact writing, and Markdown output.
- Create: `experiments/phase39_independent_label_audit/run_phase39_independent_label_audit.py`
  - Thin argparse runner that calls the builder and writer.
- Modify after real run: `README.md`
  - Add Phase 39 runner and current status boundary.
- Modify after real run: `paper/phase28_results/README.md`
  - Add Phase 39 result entry and reproduction command.
- Create after real run: `paper/phase28_results/13_phase39_independent_label_audit.md`
  - Reviewer-facing interpretation of the real Phase 39 output.
- Modify after real run: `reproducibility/FILE_MANIFEST.tsv`
  - Add Phase 39 source, runner, test, spec, plan, and result doc entries.
- Modify after real run: `docs/superpowers/phase33_current_progress_handoff.md`
  - Record Phase 39 output status, row counts, verification, and the next experimental gate.

## Task 1: Failing Tests For Label Readiness Gates

**Files:**
- Create: `tests/test_phase39_independent_label_audit.py`
- Target later: `src/paper11_geofm/phase39_independent_label_audit.py`

- [ ] **Step 1: Write the failing builder tests**

Create `tests/test_phase39_independent_label_audit.py` with this content:

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
    phase2_dir = tmp_path / "phase2"
    if rows is None:
        rows = [
            {
                "block_id": f"b{index:03d}",
                "current_farmland_label": 1 if index % 2 == 0 else 0,
                "farmland_or_orchard_label": 1 if index % 3 == 0 else 0,
                "low_slope_farmland_label": 1 if index % 4 == 0 else 0,
                "source_bsm": f"s{index:03d}",
                "source_category": "farmland" if index % 2 == 0 else "other",
                "source_dlbm": "0101" if index % 2 == 0 else "0301",
                "source_dlmc": "paddy" if index % 2 == 0 else "forest",
                "split": "train" if index < 8 else "test",
            }
            for index in range(12)
        ]
    return _write_csv(
        phase2_dir / "block_geofm_features.csv",
        rows,
        [
            "block_id",
            "current_farmland_label",
            "farmland_or_orchard_label",
            "low_slope_farmland_label",
            "source_bsm",
            "source_category",
            "source_dlbm",
            "source_dlmc",
            "split",
        ],
    ).parent


def _external_labels(path: Path, values: list[int]) -> Path:
    rows = [
        {"block_id": f"b{index:03d}", "irrigation_proxy_label": value}
        for index, value in enumerate(values)
    ]
    return _write_csv(path, rows, ["block_id", "irrigation_proxy_label"])


def _registry(path: Path, provenance_class: str) -> Path:
    rows = [
        {
            "label_column": "irrigation_proxy_label",
            "source_path": "external_irrigation.csv",
            "provenance_class": provenance_class,
            "description": "Synthetic non-DLTB irrigation proxy label",
            "external_source_name": "synthetic_irrigation_fixture",
            "independence_rationale": "not derived from DLTB, slope, or explicit planning features",
            "allowed_for_phase38_rerun": "true",
        }
    ]
    return _write_csv(
        path,
        rows,
        [
            "label_column",
            "source_path",
            "provenance_class",
            "description",
            "external_source_name",
            "independence_rationale",
            "allowed_for_phase38_rerun",
        ],
    )


def test_phase39_current_labels_remain_missing_independent_inputs(tmp_path):
    from paper11_geofm.phase39_independent_label_audit import (
        build_phase39_independent_label_audit,
    )

    analysis = build_phase39_independent_label_audit(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_columns="current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label",
    )

    assert analysis["phase"] == "phase39_independent_label_audit"
    assert analysis["phase39_independent_label_audit_status"] == "independent_label_inputs_missing"
    assert analysis["label_readiness"]["current_farmland_label"]["provenance_class"] == "explicit_label_leakage_risk"
    assert analysis["label_readiness"]["current_farmland_label"]["allowed_for_phase38_rerun"] is False
    assert "does not train PPO" in analysis["claim_boundary"]


def test_phase39_source_fields_are_leakage_risks(tmp_path):
    from paper11_geofm.phase39_independent_label_audit import (
        build_phase39_independent_label_audit,
    )

    analysis = build_phase39_independent_label_audit(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_columns=["source_category", "source_dlbm", "source_dlmc"],
    )

    assert analysis["phase39_independent_label_audit_status"] == "independent_label_inputs_missing"
    assert analysis["label_readiness"]["source_category"]["provenance_class"] == "source_field_leakage_risk"
    assert analysis["label_readiness"]["source_dlbm"]["allowed_for_phase38_rerun"] is False


def test_phase39_external_candidate_label_clears_phase38_rerun_gate(tmp_path):
    from paper11_geofm.phase39_independent_label_audit import (
        build_phase39_independent_label_audit,
    )

    phase2_dir = _phase2_dir(tmp_path)
    external = _external_labels(
        tmp_path / "external_irrigation.csv",
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    )
    registry = _registry(tmp_path / "registry.csv", "candidate_independent_proxy")

    analysis = build_phase39_independent_label_audit(
        phase2_output_dir=phase2_dir,
        external_label_csvs=[external],
        label_registry=registry,
        label_columns=["irrigation_proxy_label"],
    )

    assert analysis["phase39_independent_label_audit_status"] == "independent_labels_ready_for_phase38_rerun"
    row = analysis["label_readiness"]["irrigation_proxy_label"]
    assert row["provenance_class"] == "candidate_independent_proxy"
    assert row["registry_entry_present"] is True
    assert row["allowed_for_phase38_rerun"] is True
    assert row["train_positive_count"] == 4
    assert row["eval_positive_count"] == 2


def test_phase39_unclassified_external_label_needs_review(tmp_path):
    from paper11_geofm.phase39_independent_label_audit import (
        build_phase39_independent_label_audit,
    )

    analysis = build_phase39_independent_label_audit(
        phase2_output_dir=_phase2_dir(tmp_path),
        external_label_csvs=[
            _external_labels(
                tmp_path / "external_irrigation.csv",
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            )
        ],
        label_columns=["irrigation_proxy_label"],
    )

    assert analysis["phase39_independent_label_audit_status"] == "candidate_proxy_labels_need_review"
    row = analysis["label_readiness"]["irrigation_proxy_label"]
    assert row["provenance_class"] == "unclassified"
    assert row["allowed_for_phase38_rerun"] is False


def test_phase39_single_class_candidate_is_insufficient(tmp_path):
    from paper11_geofm.phase39_independent_label_audit import (
        build_phase39_independent_label_audit,
    )

    analysis = build_phase39_independent_label_audit(
        phase2_output_dir=_phase2_dir(tmp_path),
        external_label_csvs=[_external_labels(tmp_path / "external_irrigation.csv", [1] * 12)],
        label_registry=_registry(tmp_path / "registry.csv", "candidate_independent_proxy"),
        label_columns=["irrigation_proxy_label"],
    )

    assert analysis["phase39_independent_label_audit_status"] == "independent_label_inputs_insufficient"
    row = analysis["label_readiness"]["irrigation_proxy_label"]
    assert row["usable"] is False
    assert row["allowed_for_phase38_rerun"] is False
    assert "both positive and negative labels" in row["decision_reason"]
```

- [ ] **Step 2: Run the new tests to verify they fail before implementation**

Run:

```powershell
python -m pytest tests\test_phase39_independent_label_audit.py -q --basetemp=.pytest_tmp_phase39_t1_red -p no:cacheprovider
```

Expected: FAIL with `ModuleNotFoundError` for `paper11_geofm.phase39_independent_label_audit`.

- [ ] **Step 3: Commit the failing tests**

```powershell
git add tests\test_phase39_independent_label_audit.py
git commit -m "test: add Phase 39 label audit fixtures"
```

## Task 2: Core Builder And Status Reduction

**Files:**
- Create: `src/paper11_geofm/phase39_independent_label_audit.py`
- Test: `tests/test_phase39_independent_label_audit.py`

- [ ] **Step 1: Create the Phase 39 analysis module**

Create `src/paper11_geofm/phase39_independent_label_audit.py` with this content:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path


PHASE39_CLAIM_BOUNDARY = (
    "Phase 39 audits independent-label readiness. It does not train PPO, does "
    "not alter rewards, does not rebuild suitability proxies, does not enable "
    "B2/B3, and does not prove agronomic validity."
)

DEFAULT_PHASE39_LABEL_COLUMNS = (
    "current_farmland_label",
    "farmland_or_orchard_label",
    "low_slope_farmland_label",
)

EXPLICIT_LEAKAGE_LABELS = {
    "current_farmland_label",
    "farmland_or_orchard_label",
    "low_slope_farmland_label",
}

SOURCE_FIELD_LEAKAGE_LABELS = {
    "source_bsm",
    "source_category",
    "source_dlbm",
    "source_dlmc",
}

VALID_PROVENANCE_CLASSES = {
    "explicit_label_leakage_risk",
    "source_field_leakage_risk",
    "candidate_independent_proxy",
    "independent_validation_label",
    "unclassified",
}

PHASE39_INVENTORY_FIELDNAMES = (
    "label_column",
    "column_source",
    "available",
    "registry_entry_present",
    "provenance_class",
    "description",
    "external_source_name",
    "independence_rationale",
)

PHASE39_READINESS_FIELDNAMES = (
    "label_column",
    "column_source",
    "available",
    "usable",
    "valid_label_count",
    "positive_count",
    "negative_count",
    "positive_rate",
    "train_count",
    "eval_count",
    "train_positive_count",
    "train_negative_count",
    "eval_positive_count",
    "eval_negative_count",
    "split_source",
    "provenance_class",
    "registry_entry_present",
    "join_missing_count",
    "allowed_for_phase38_rerun",
    "decision_reason",
    "claim_boundary",
)

PHASE39_REGISTRY_TEMPLATE_FIELDNAMES = (
    "label_column",
    "source_path",
    "provenance_class",
    "description",
    "external_source_name",
    "independence_rationale",
    "allowed_for_phase38_rerun",
)


def build_phase39_independent_label_audit(
    phase2_output_dir: Path | str,
    external_label_csvs: Sequence[Path | str] | Path | str | None = None,
    label_registry: Path | str | None = None,
    label_columns: Sequence[str] | str = DEFAULT_PHASE39_LABEL_COLUMNS,
) -> dict[str, object]:
    phase2_dir = Path(phase2_output_dir)
    block_rows = _read_csv_rows(
        phase2_dir / "block_geofm_features.csv",
        "Phase 2 block feature CSV",
    )
    registry = _read_registry(label_registry)
    external_sources = _load_external_sources(external_label_csvs)
    joined_rows, external_join_stats = _join_external_labels(block_rows, external_sources)
    requested_labels = _normalize_csvish_values(label_columns)
    if not requested_labels:
        requested_labels = _default_candidate_labels(joined_rows, registry)
    missing = [
        label for label in requested_labels
        if not _column_available(joined_rows, label)
    ]
    if missing:
        raise ValueError(
            "Phase 39 requested label columns are missing: "
            + ",".join(missing)
        )

    inventory_rows: list[dict[str, object]] = []
    readiness_rows: list[dict[str, object]] = []
    readiness_by_label: dict[str, dict[str, object]] = {}
    for label in requested_labels:
        registry_entry = registry.get(label, {})
        provenance = _provenance_for_label(label, registry_entry)
        column_source = _column_source(label, external_sources)
        join_missing_count = int(external_join_stats.get(label, 0))
        inventory = {
            "label_column": label,
            "column_source": column_source,
            "available": True,
            "registry_entry_present": bool(registry_entry),
            "provenance_class": provenance,
            "description": registry_entry.get("description", ""),
            "external_source_name": registry_entry.get("external_source_name", ""),
            "independence_rationale": registry_entry.get("independence_rationale", ""),
        }
        readiness = _readiness_for_label(
            joined_rows,
            label,
            column_source,
            provenance,
            bool(registry_entry),
            join_missing_count,
        )
        inventory_rows.append(inventory)
        readiness_rows.append(readiness)
        readiness_by_label[label] = readiness

    status = _phase39_status(readiness_rows)
    row_counts = {
        "block_rows": len(block_rows),
        "joined_block_rows": len(joined_rows),
        "external_label_files": len(external_sources),
        "inventory_rows": len(inventory_rows),
        "readiness_rows": len(readiness_rows),
    }
    return {
        "phase": "phase39_independent_label_audit",
        "phase39_independent_label_audit_status": status,
        "source_paths": {
            "phase2_output_dir": str(phase2_dir),
            "external_label_csvs": [str(source["path"]) for source in external_sources],
            "label_registry": str(Path(label_registry)) if label_registry is not None else None,
        },
        "label_columns_requested": requested_labels,
        "row_counts": row_counts,
        "label_inventory_rows": inventory_rows,
        "label_readiness_rows": readiness_rows,
        "label_readiness": readiness_by_label,
        "registry_template_rows": _registry_template_rows(readiness_rows),
        "interpretation": _phase39_interpretation(status),
        "claim_boundary": PHASE39_CLAIM_BOUNDARY,
    }


def _read_csv_rows(path: Path, label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_csvish_values(values: Sequence[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        parts = values.split(",")
    else:
        parts = []
        for value in values:
            parts.extend(str(value).split(","))
    return [part.strip() for part in parts if part.strip()]


def _read_registry(path: Path | str | None) -> dict[str, dict[str, str]]:
    if path is None or str(path).strip() == "":
        return {}
    rows = _read_csv_rows(Path(path), "Phase 39 label registry")
    registry: dict[str, dict[str, str]] = {}
    for row in rows:
        label = str(row.get("label_column", "")).strip()
        if not label:
            raise ValueError("Phase 39 registry row is missing label_column")
        provenance = str(row.get("provenance_class", "")).strip() or "unclassified"
        if provenance not in VALID_PROVENANCE_CLASSES:
            raise ValueError(f"Phase 39 unsupported provenance_class: {provenance}")
        registry[label] = {str(key): str(value) for key, value in row.items()}
        registry[label]["provenance_class"] = provenance
    return registry


def _load_external_sources(
    paths: Sequence[Path | str] | Path | str | None,
) -> list[dict[str, object]]:
    normalized = _normalize_path_values(paths)
    sources: list[dict[str, object]] = []
    for path in normalized:
        rows = _read_csv_rows(path, "Phase 39 external label CSV")
        if not rows:
            sources.append({"path": path, "rows": rows, "label_columns": []})
            continue
        if "block_id" not in rows[0]:
            raise ValueError(f"Phase 39 external label CSV lacks block_id: {path}")
        seen: set[str] = set()
        for row in rows:
            block_id = str(row.get("block_id", "")).strip()
            if block_id in seen:
                raise ValueError(f"Phase 39 duplicate external block_id in {path}: {block_id}")
            seen.add(block_id)
        label_columns = [
            column for column in rows[0].keys()
            if column != "block_id"
        ]
        sources.append({"path": path, "rows": rows, "label_columns": label_columns})
    return sources


def _normalize_path_values(
    paths: Sequence[Path | str] | Path | str | None,
) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        raw = str(paths)
        parts = raw.split(",")
    else:
        parts = []
        for path in paths:
            parts.extend(str(path).split(","))
    return [Path(part.strip()) for part in parts if part.strip()]


def _join_external_labels(
    block_rows: list[dict[str, str]],
    external_sources: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    joined = [dict(row) for row in block_rows]
    missing_by_label: dict[str, int] = {}
    by_block = {str(row.get("block_id", "")).strip(): row for row in joined}
    for source in external_sources:
        rows = source["rows"]
        label_columns = list(source["label_columns"])
        external_by_block = {
            str(row.get("block_id", "")).strip(): row
            for row in rows
            if isinstance(row, Mapping)
        }
        for label in label_columns:
            missing = 0
            for block_id, block_row in by_block.items():
                external_row = external_by_block.get(block_id)
                if external_row is None or str(external_row.get(label, "")).strip() == "":
                    missing += 1
                    block_row[label] = ""
                else:
                    block_row[label] = str(external_row.get(label, "")).strip()
            missing_by_label[label] = missing
    return joined, missing_by_label


def _default_candidate_labels(
    rows: Sequence[Mapping[str, str]],
    registry: Mapping[str, Mapping[str, str]],
) -> list[str]:
    labels = list(DEFAULT_PHASE39_LABEL_COLUMNS)
    labels.extend(label for label in registry.keys() if label not in labels)
    if rows:
        for label in rows[0].keys():
            if label.endswith("_label") and label not in labels:
                labels.append(label)
    return labels


def _column_available(rows: Sequence[Mapping[str, str]], label: str) -> bool:
    return bool(rows) and any(label in row for row in rows)


def _column_source(label: str, external_sources: Sequence[Mapping[str, object]]) -> str:
    for source in external_sources:
        if label in source["label_columns"]:
            return str(source["path"])
    return "phase2_block_features"


def _provenance_for_label(
    label: str,
    registry_entry: Mapping[str, str],
) -> str:
    if registry_entry:
        provenance = str(registry_entry.get("provenance_class", "unclassified")).strip()
        if provenance not in VALID_PROVENANCE_CLASSES:
            raise ValueError(f"Phase 39 unsupported provenance_class: {provenance}")
        return provenance
    if label in EXPLICIT_LEAKAGE_LABELS:
        return "explicit_label_leakage_risk"
    if label in SOURCE_FIELD_LEAKAGE_LABELS:
        return "source_field_leakage_risk"
    return "unclassified"


def _readiness_for_label(
    rows: Sequence[Mapping[str, str]],
    label: str,
    column_source: str,
    provenance: str,
    registry_entry_present: bool,
    join_missing_count: int,
) -> dict[str, object]:
    split_source = "split_column" if any("split" in row for row in rows) else "all_rows_eval"
    valid: list[tuple[int, str]] = []
    for row in rows:
        parsed = _binary_label(row.get(label, ""))
        if parsed is None:
            continue
        split = str(row.get("split", "eval")).strip().lower()
        valid.append((parsed, split))
    positives = sum(1 for value, _split in valid if value == 1)
    negatives = sum(1 for value, _split in valid if value == 0)
    train_values = [
        value for value, split in valid
        if split in {"train", "training"}
    ]
    eval_values = [
        value for value, split in valid
        if split not in {"train", "training"}
    ]
    if split_source == "all_rows_eval":
        train_values = []
        eval_values = [value for value, _split in valid]
    train_pos = sum(1 for value in train_values if value == 1)
    train_neg = sum(1 for value in train_values if value == 0)
    eval_pos = sum(1 for value in eval_values if value == 1)
    eval_neg = sum(1 for value in eval_values if value == 0)
    usable = (
        positives > 0
        and negatives > 0
        and train_pos > 0
        and train_neg > 0
        and eval_pos > 0
        and eval_neg > 0
    )
    allowed = (
        usable
        and provenance in {"candidate_independent_proxy", "independent_validation_label"}
        and registry_entry_present
    )
    return {
        "label_column": label,
        "column_source": column_source,
        "available": True,
        "usable": usable,
        "valid_label_count": len(valid),
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": _round_float(positives / len(valid)) if valid else "",
        "train_count": len(train_values),
        "eval_count": len(eval_values),
        "train_positive_count": train_pos,
        "train_negative_count": train_neg,
        "eval_positive_count": eval_pos,
        "eval_negative_count": eval_neg,
        "split_source": split_source,
        "provenance_class": provenance,
        "registry_entry_present": registry_entry_present,
        "join_missing_count": join_missing_count,
        "allowed_for_phase38_rerun": allowed,
        "decision_reason": _decision_reason(
            usable,
            provenance,
            registry_entry_present,
            positives,
            negatives,
            train_pos,
            train_neg,
            eval_pos,
            eval_neg,
        ),
        "claim_boundary": PHASE39_CLAIM_BOUNDARY,
    }


def _binary_label(value: object) -> int | None:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes"}:
        return 1
    if text in {"0", "0.0", "false", "no"}:
        return 0
    return None


def _decision_reason(
    usable: bool,
    provenance: str,
    registry_entry_present: bool,
    positives: int,
    negatives: int,
    train_pos: int,
    train_neg: int,
    eval_pos: int,
    eval_neg: int,
) -> str:
    if positives == 0 or negatives == 0:
        return "label does not contain both positive and negative labels"
    if train_pos == 0 or train_neg == 0:
        return "train split lacks both positive and negative labels"
    if eval_pos == 0 or eval_neg == 0:
        return "evaluation split lacks both positive and negative labels"
    if provenance in {"explicit_label_leakage_risk", "source_field_leakage_risk"}:
        return "label provenance is a leakage risk"
    if provenance == "unclassified" or not registry_entry_present:
        return "label needs registry provenance review"
    if usable:
        return "label is ready for Phase 38 rerun"
    return "label is not ready for Phase 38 rerun"


def _phase39_status(rows: Sequence[Mapping[str, object]]) -> str:
    if any(row.get("allowed_for_phase38_rerun") is True for row in rows):
        return "independent_labels_ready_for_phase38_rerun"
    if any(
        row.get("provenance_class") == "unclassified"
        and row.get("usable") is True
        for row in rows
    ):
        return "candidate_proxy_labels_need_review"
    if any(
        row.get("provenance_class") in {"candidate_independent_proxy", "independent_validation_label"}
        and row.get("usable") is False
        for row in rows
    ):
        return "independent_label_inputs_insufficient"
    return "independent_label_inputs_missing"


def _phase39_interpretation(status: str) -> str:
    if status == "independent_labels_ready_for_phase38_rerun":
        return "At least one non-leakage label is usable for a Phase 38 rerun. This does not enable B2/B3 by itself."
    if status == "candidate_proxy_labels_need_review":
        return "A usable candidate label exists, but registry provenance is not strong enough for a Phase 38 rerun."
    if status == "independent_label_inputs_insufficient":
        return "Candidate non-leakage labels exist but fail label variation, split coverage, or join-readiness checks."
    return "No usable independent non-leakage label is available, so Phase 38 cannot be rerun with a stronger label and B2/B3 remains blocked."


def _registry_template_rows(
    readiness_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in readiness_rows:
        rows.append(
            {
                "label_column": row.get("label_column", ""),
                "source_path": row.get("column_source", ""),
                "provenance_class": row.get("provenance_class", "unclassified"),
                "description": "",
                "external_source_name": "",
                "independence_rationale": "",
                "allowed_for_phase38_rerun": str(row.get("allowed_for_phase38_rerun", False)).lower(),
            }
        )
    return rows


def _round_float(value: float) -> float:
    return round(float(value), 10)
```

- [ ] **Step 2: Run the builder tests**

Run:

```powershell
python -m pytest tests\test_phase39_independent_label_audit.py -q --basetemp=.pytest_tmp_phase39_t2_green -p no:cacheprovider
```

Expected: five tests pass.

- [ ] **Step 3: Commit the core builder**

```powershell
git add src\paper11_geofm\phase39_independent_label_audit.py tests\test_phase39_independent_label_audit.py
git commit -m "feat: add Phase 39 independent label audit builder"
```

## Task 3: Artifact Writer And CLI Runner

**Files:**
- Modify: `src/paper11_geofm/phase39_independent_label_audit.py`
- Modify: `tests/test_phase39_independent_label_audit.py`
- Create: `experiments/phase39_independent_label_audit/run_phase39_independent_label_audit.py`

- [ ] **Step 1: Add writer and CLI tests**

Append this content to `tests/test_phase39_independent_label_audit.py`:

```python

def test_phase39_writer_outputs_csv_json_markdown_and_template(tmp_path):
    from paper11_geofm.phase39_independent_label_audit import (
        build_phase39_independent_label_audit,
        write_phase39_independent_label_audit_artifacts,
    )

    analysis = build_phase39_independent_label_audit(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_columns=["current_farmland_label"],
    )
    paths = write_phase39_independent_label_audit_artifacts(
        analysis,
        tmp_path / "outputs",
    )

    assert paths["label_inventory_csv"].name == "phase39_label_inventory.csv"
    assert paths["label_readiness_csv"].name == "phase39_label_readiness.csv"
    assert paths["diagnosis_json"].name == "phase39_independent_label_audit.json"
    assert paths["diagnosis_md"].name == "phase39_independent_label_audit.md"
    assert paths["registry_template_csv"].name == "phase39_label_registry_template.csv"
    assert all(path.exists() for path in paths.values())
    saved = json.loads(paths["diagnosis_json"].read_text(encoding="utf-8"))
    assert saved["phase"] == "phase39_independent_label_audit"
    markdown = paths["diagnosis_md"].read_text(encoding="utf-8")
    assert "Phase 39 Independent Label Audit" in markdown
    assert "independent_label_inputs_missing" in markdown


def test_phase39_cli_writes_outputs(tmp_path):
    phase2_dir = _phase2_dir(tmp_path)
    script = ROOT / "experiments" / "phase39_independent_label_audit" / "run_phase39_independent_label_audit.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--phase2-output-dir",
            str(phase2_dir),
            "--output-dir",
            str(tmp_path / "cli_outputs"),
            "--label-columns",
            "current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Phase 39 independent-label audit status:" in result.stdout
    assert "Claim boundary:" in result.stdout
    assert (tmp_path / "cli_outputs" / "phase39_independent_label_audit.json").exists()
    assert (tmp_path / "cli_outputs" / "phase39_label_registry_template.csv").exists()
```

- [ ] **Step 2: Run writer and CLI tests to verify they fail**

Run:

```powershell
python -m pytest tests\test_phase39_independent_label_audit.py::test_phase39_writer_outputs_csv_json_markdown_and_template tests\test_phase39_independent_label_audit.py::test_phase39_cli_writes_outputs -q --basetemp=.pytest_tmp_phase39_t3_red -p no:cacheprovider
```

Expected: FAIL because `write_phase39_independent_label_audit_artifacts` and the CLI runner do not exist.

- [ ] **Step 3: Add writer functions**

Append this content to `src/paper11_geofm/phase39_independent_label_audit.py`:

```python

def write_phase39_independent_label_audit_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "label_inventory_csv": output_path / "phase39_label_inventory.csv",
        "label_readiness_csv": output_path / "phase39_label_readiness.csv",
        "diagnosis_json": output_path / "phase39_independent_label_audit.json",
        "diagnosis_md": output_path / "phase39_independent_label_audit.md",
        "registry_template_csv": output_path / "phase39_label_registry_template.csv",
    }
    _write_csv_mapping_rows(
        paths["label_inventory_csv"],
        PHASE39_INVENTORY_FIELDNAMES,
        analysis.get("label_inventory_rows"),
        "label_inventory_rows",
    )
    _write_csv_mapping_rows(
        paths["label_readiness_csv"],
        PHASE39_READINESS_FIELDNAMES,
        analysis.get("label_readiness_rows"),
        "label_readiness_rows",
    )
    _write_csv_mapping_rows(
        paths["registry_template_csv"],
        PHASE39_REGISTRY_TEMPLATE_FIELDNAMES,
        analysis.get("registry_template_rows"),
        "registry_template_rows",
    )
    paths["diagnosis_json"].write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["diagnosis_md"].write_text(
        _phase39_markdown(analysis),
        encoding="utf-8",
    )
    return paths


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    label: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 39 analysis is missing {label}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 39 {label} contains a non-mapping row")
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_ready(value), sort_keys=True)
    return value


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _phase39_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Phase 39 Independent Label Audit",
        "",
        f"Status: {analysis.get('phase39_independent_label_audit_status', '')}",
        "",
        "## Label Readiness",
        "",
        "| Label | Provenance | Usable | Allowed for Phase 38 | Train / Eval | Reason |",
        "|---|---|---:|---:|---:|---|",
    ]
    rows = analysis.get("label_readiness_rows")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| {label} | {provenance} | {usable} | {allowed} | {train} / {eval} | {reason} |".format(
                    label=row.get("label_column", ""),
                    provenance=row.get("provenance_class", ""),
                    usable=row.get("usable", ""),
                    allowed=row.get("allowed_for_phase38_rerun", ""),
                    train=row.get("train_count", ""),
                    eval=row.get("eval_count", ""),
                    reason=row.get("decision_reason", ""),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(analysis.get("interpretation", "")),
            "",
            "## Boundary",
            "",
            str(analysis.get("claim_boundary", "")),
            "",
            "A Phase 39 ready status only authorizes a Phase 38 rerun with the accepted labels.",
            "",
        ]
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Add CLI runner**

Create `experiments/phase39_independent_label_audit/run_phase39_independent_label_audit.py` with this content:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase39_independent_label_audit import (
    build_phase39_independent_label_audit,
    write_phase39_independent_label_audit_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Paper11 Phase 39 independent-label readiness audit over "
            "Phase 2 block features and optional external label CSVs."
        )
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--external-label-csvs", default="")
    parser.add_argument("--label-registry", type=Path)
    parser.add_argument(
        "--label-columns",
        default="current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        analysis = build_phase39_independent_label_audit(
            phase2_output_dir=args.phase2_output_dir,
            external_label_csvs=args.external_label_csvs,
            label_registry=args.label_registry,
            label_columns=args.label_columns,
        )
        paths = write_phase39_independent_label_audit_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Phase 39 independent-label audit status: "
        f"{analysis['phase39_independent_label_audit_status']}"
    )
    print(f"Label inventory CSV: {paths['label_inventory_csv']}")
    print(f"Label readiness CSV: {paths['label_readiness_csv']}")
    print(f"Registry template CSV: {paths['registry_template_csv']}")
    print(f"Diagnosis JSON: {paths['diagnosis_json']}")
    print(f"Diagnosis Markdown: {paths['diagnosis_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run all Phase 39 tests**

Run:

```powershell
python -m pytest tests\test_phase39_independent_label_audit.py -q --basetemp=.pytest_tmp_phase39_t3_green -p no:cacheprovider
```

Expected: seven tests pass.

- [ ] **Step 6: Commit writer and CLI**

```powershell
git add src\paper11_geofm\phase39_independent_label_audit.py tests\test_phase39_independent_label_audit.py experiments\phase39_independent_label_audit\run_phase39_independent_label_audit.py
git commit -m "feat: add Phase 39 label audit artifacts and runner"
```

## Task 4: Real Bishan Phase 39 Run

**Files:**
- Generated ignored outputs under `experiments/phase39_independent_label_audit/outputs/real_bishan`
- No tracked source edits in this task

- [ ] **Step 1: Run focused tests before real data**

Run:

```powershell
python -m pytest tests\test_phase39_independent_label_audit.py tests\test_phase38_proxy_rebuild.py -q --basetemp=.pytest_tmp_phase39_pre_real -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the real Phase 39 audit**

Run:

```powershell
python experiments\phase39_independent_label_audit\run_phase39_independent_label_audit.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase39_independent_label_audit\outputs\real_bishan --label-columns current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label,source_category,source_dlbm,source_dlmc
```

Expected: command exits 0 and prints `Phase 39 independent-label audit status: independent_label_inputs_missing`.

- [ ] **Step 3: Inspect real status and row counts**

Run:

```powershell
Get-Content -Raw experiments\phase39_independent_label_audit\outputs\real_bishan\phase39_independent_label_audit.json
```

Expected: JSON includes `phase39_independent_label_audit_status`, `row_counts`, `label_readiness_rows`, and `registry_template_rows`.

- [ ] **Step 4: Confirm generated outputs are not staged**

Run:

```powershell
git status --short
```

Expected: generated files under `experiments/**/outputs/` are ignored. Do not add generated CSV/JSON/Markdown artifacts to Git.

## Task 5: Documentation And Handoff

**Files:**
- Modify: `README.md`
- Modify: `paper/phase28_results/README.md`
- Create: `paper/phase28_results/13_phase39_independent_label_audit.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [ ] **Step 1: Read exact Phase 39 real-run values**

Run:

```powershell
Get-Content -Raw experiments\phase39_independent_label_audit\outputs\real_bishan\phase39_independent_label_audit.json
```

Expected: copy the exact `phase39_independent_label_audit_status`, `row_counts.block_rows`, `row_counts.readiness_rows`, and `interpretation`.

- [ ] **Step 2: Add reviewer-facing Phase 39 result doc**

Create `paper/phase28_results/13_phase39_independent_label_audit.md` with:

```markdown
# Phase 39 Independent Label Audit

Phase 39 audits whether the current Paper11 real Bishan labels contain any defensible non-DLTB, non-slope, non-explicit-feature validation target before Phase 38 is rerun or B2/B3 reward work is considered.

## Experiment Snapshot

Input:

- Phase 2 real Bishan block features: `experiments/phase11_bishan_dltb_real/outputs/phase2_real`

Output directory:

```text
experiments/phase39_independent_label_audit/outputs/real_bishan
```

The real run evaluated the current DLTB/slope labels plus source land-use descriptor fields:

- `current_farmland_label`
- `farmland_or_orchard_label`
- `low_slope_farmland_label`
- `source_category`
- `source_dlbm`
- `source_dlmc`

## Main Result

Status: `independent_label_inputs_missing`

Row counts from `phase39_independent_label_audit.json`:

- block rows: `64984`
- readiness rows: `6`

Interpretation: no usable independent non-leakage label is available in the current real Bishan table. Phase 38 therefore cannot be rerun with a stronger validation label, and B2/B3 suitability reward remains blocked.

## Reproduction

Run from the repository root after the Phase 2 real Bishan outputs exist:

```powershell
python experiments\phase39_independent_label_audit\run_phase39_independent_label_audit.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase39_independent_label_audit\outputs\real_bishan --label-columns current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label,source_category,source_dlbm,source_dlmc
```

Expected local artifacts:

- `phase39_label_inventory.csv`
- `phase39_label_readiness.csv`
- `phase39_label_registry_template.csv`
- `phase39_independent_label_audit.json`
- `phase39_independent_label_audit.md`

## Claim Boundary

Phase 39 is a label-readiness gate. It does not train PPO, alter rewards, rebuild suitability proxies, enable B2/B3, prove GeoFM agronomic validity, or support final planning-performance claims.
```

If the real JSON has different counts or status, replace only the numeric values and status with the exact real-run values while preserving the conservative interpretation.

- [ ] **Step 3: Update README and result index**

In `README.md`:

- add `experiments/phase39_independent_label_audit/` to the repository layout after Phase 38;
- add a Phase 39 reproduction section after the Phase 38 section;
- state that the expected current status is `independent_label_inputs_missing` unless external independent labels are supplied;
- state that B2/B3 remains blocked.

In `paper/phase28_results/README.md`:

- add `13_phase39_independent_label_audit.md` to the result file list;
- add the same Phase 39 reproduction command and artifact list.

- [ ] **Step 4: Update manifest and handoff**

Add these rows to `reproducibility/FILE_MANIFEST.tsv` using the existing tab-separated style:

```text
docs/superpowers/specs/2026-06-27-phase39-independent-label-audit-design.md	documentation	Phase 39 independent-label audit design specification.
docs/superpowers/plans/2026-06-27-phase39-independent-label-audit.md	documentation	Phase 39 independent-label audit implementation plan.
src/paper11_geofm/phase39_independent_label_audit.py	source	Phase 39 independent-label readiness audit module.
experiments/phase39_independent_label_audit/run_phase39_independent_label_audit.py	experiment	Executable Phase 39 independent-label readiness audit runner.
tests/test_phase39_independent_label_audit.py	verification	Pytest checks for Phase 39 provenance gates, external label joins, artifact writing, and CLI behavior.
paper/phase28_results/13_phase39_independent_label_audit.md	documentation	Reviewer-facing interpretation of the Phase 39 independent-label audit.
```

In `docs/superpowers/phase33_current_progress_handoff.md`, add a Phase 39 section with:

- latest Phase 39 implementation commits;
- real output directory;
- generated artifact names;
- exact real status and row counts;
- verification commands and results;
- explicit statement that B2/B3 remains blocked unless a future external label clears the Phase 39 and Phase 38 gates.

- [ ] **Step 5: Run documentation placeholder scan**

Run:

```powershell
$pattern = 'T' + 'BD|TO' + 'DO|REPLACE_' + 'WITH|PLACE' + 'HOLDER'
rg -n $pattern README.md paper\phase28_results\README.md paper\phase28_results\13_phase39_independent_label_audit.md docs\superpowers\phase33_current_progress_handoff.md
```

Expected: no matches.

- [ ] **Step 6: Run final verification**

Run:

```powershell
python -m pytest tests\test_phase39_independent_label_audit.py tests\test_phase38_proxy_rebuild.py -q --basetemp=.pytest_tmp_phase39_final -p no:cacheprovider
python scripts\smoke_check.py
```

Expected: all selected tests pass and `Paper11 smoke check passed.`

- [ ] **Step 7: Commit documentation**

```powershell
git add README.md paper\phase28_results\README.md paper\phase28_results\13_phase39_independent_label_audit.md reproducibility\FILE_MANIFEST.tsv docs\superpowers\phase33_current_progress_handoff.md
git commit -m "docs: record Phase 39 independent label audit result"
```

## Task 6: Closeout

**Files:**
- No new files unless verification reveals a tracked documentation omission

- [ ] **Step 1: Check final repository state**

Run:

```powershell
git status --short --branch
git log -8 --oneline
```

Expected: only ignored generated outputs are absent from status; latest commits include Phase 39 tests, builder, runner, and docs.

- [ ] **Step 2: Summarize exact outcome**

Report:

- final Phase 39 status;
- whether Phase 38 can be rerun with any non-leakage label;
- whether B2/B3 remains blocked;
- tests run and pass counts;
- latest commit hash;
- any limitation, including if only DLTB/slope/source-field labels were available.

## Plan Self-Review Notes

- Spec coverage: tasks cover required inputs, optional external labels, registry provenance classes, readiness checks, aggregate statuses, artifacts, error handling, tests, real run, documentation, and claim boundaries.
- Placeholder scan target: this plan avoids unfinished-work marker strings and constructs documentation scan patterns dynamically.
- Type consistency: the plan uses `phase39_independent_label_audit_status`, `label_readiness_rows`, `registry_template_rows`, and `allowed_for_phase38_rerun` consistently across tests, builder, writer, CLI, and documentation.
