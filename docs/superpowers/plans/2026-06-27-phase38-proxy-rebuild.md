# Phase 38 Proxy-Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-aware Phase 38 suitability-proxy rebuild pipeline that trains lightweight proxy models, writes rebuilt proxy scores, and keeps B2/B3 reward blocked unless non-leakage labels clear conservative controls.

**Architecture:** Follow the Phase 36/37 pattern: one pure analysis module, one CLI runner, one focused pytest file, and documentation updates after the real run. The module loads existing feature tables through `load_variant_input`, classifies labels, evaluates model and feature-family combinations under spatial held-out validation, writes artifacts, and emits one conservative status.

**Tech Stack:** Python, NumPy, scikit-learn, csv/json pathlib utilities, pytest, existing Paper11 feature manifests and runner conventions.

---

## File Structure

- Create: `src/paper11_geofm/phase38_proxy_rebuild.py`
  - Owns constants, label classification, feature-family loading, model fitting, status reduction, rebuilt score generation, artifact writing, and Markdown output.
- Create: `experiments/phase38_proxy_rebuild/run_phase38_proxy_rebuild.py`
  - Thin argparse runner matching Phase 36 CLI style.
- Create: `tests/test_phase38_proxy_rebuild.py`
  - Synthetic fixtures and unit/CLI tests for Phase 38.
- Modify after real run: `README.md`
  - Add Phase 38 runner command and current result interpretation.
- Modify after real run: `paper/phase28_results/README.md`
  - Add Phase 38 result entry and reproduction command.
- Create after real run: `paper/phase28_results/12_phase38_proxy_rebuild.md`
  - Reviewer-facing interpretation of the real Phase 38 output.
- Modify after real run: `reproducibility/FILE_MANIFEST.tsv`
  - Add Phase 38 source, runner, test, spec, plan, and result doc entries.
- Modify after real run: `docs/superpowers/phase33_current_progress_handoff.md`
  - Record Phase 38 output status, row counts, verification, and next step.

## Task 1: Failing Tests For Label Boundaries And Input Gates

**Files:**
- Create: `tests/test_phase38_proxy_rebuild.py`
- Target later: `src/paper11_geofm/phase38_proxy_rebuild.py`

- [ ] **Step 1: Write the failing tests and shared synthetic fixture**

Create `tests/test_phase38_proxy_rebuild.py` with this content:

```python
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


EXPLICIT_COLUMNS = [f"explicit_feature_{index:02d}" for index in range(17)]
EMBEDDING_COLUMNS = [f"embedding_mean_{index:02d}" for index in range(64)]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _write_variant_manifest(
    output_dir: Path,
    variants: dict[str, tuple[str, list[str]]],
) -> None:
    payload = {
        "claim_boundary": "phase38 test manifest",
        "variants": {
            variant_id: {
                "description": f"{variant_id} test variant",
                "state_groups": [],
                "reward": "base_planning_reward",
                "required_columns": columns,
                "ready": True,
                "missing": [],
                "feature_table": table_name,
                "row_count": 24,
            }
            for variant_id, (table_name, columns) in variants.items()
        },
    }
    (output_dir / "experiment_variants.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _base_row(index: int, split: str) -> dict[str, object]:
    independent_label = 1 if index % 4 in {0, 1} else 0
    leakage_label = 1 if index % 2 == 0 else 0
    row: dict[str, object] = {
        "block_id": f"b{index:03d}",
        "suitability_proxy": 0.7 if leakage_label else 0.3,
        "current_farmland_label": leakage_label,
        "farmland_or_orchard_label": leakage_label,
        "low_slope_farmland_label": leakage_label,
        "independent_proxy_label": independent_label,
        "split": split,
    }
    for column_index, column in enumerate(EXPLICIT_COLUMNS):
        row[column] = float((index + column_index) % 5) / 10.0
    row["explicit_feature_04"] = float(leakage_label)
    row["explicit_feature_13"] = float(leakage_label)
    for column_index, column in enumerate(EMBEDDING_COLUMNS):
        if column_index == 0:
            value = 3.0 if independent_label else -3.0
        elif column_index == 1:
            value = 1.5 if independent_label else -1.5
        else:
            value = float(((index + column_index) % 7) - 3) / 20.0
        row[column] = value
    return row


def _fixture_inputs(tmp_path: Path) -> dict[str, Path]:
    phase2_dir = tmp_path / "phase2"
    phase8_dir = tmp_path / "phase8"
    normalized_dir = tmp_path / "phase30_controls"
    rows = [
        _base_row(index, split="train" if index < 16 else "test")
        for index in range(24)
    ]
    block_columns = [
        "block_id",
        *EXPLICIT_COLUMNS,
        *EMBEDDING_COLUMNS,
        "suitability_proxy",
        "current_farmland_label",
        "farmland_or_orchard_label",
        "low_slope_farmland_label",
        "independent_proxy_label",
        "split",
    ]
    _write_csv(phase2_dir / "block_geofm_features.csv", rows, block_columns)
    _write_csv(
        phase2_dir / "variant_B0_features.csv",
        rows,
        ["block_id", *EXPLICIT_COLUMNS],
    )
    _write_csv(
        phase2_dir / "variant_B1_features.csv",
        rows,
        ["block_id", *EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS],
    )
    _write_csv(
        phase2_dir / "variant_B2_features.csv",
        rows,
        ["block_id", *EXPLICIT_COLUMNS, "suitability_proxy"],
    )
    _write_csv(
        phase2_dir / "variant_B3_features.csv",
        rows,
        ["block_id", *EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS, "suitability_proxy"],
    )
    _write_variant_manifest(
        phase2_dir,
        {
            "B0": ("variant_B0_features.csv", EXPLICIT_COLUMNS),
            "B1": ("variant_B1_features.csv", [*EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS]),
            "B2": ("variant_B2_features.csv", [*EXPLICIT_COLUMNS, "suitability_proxy"]),
            "B3": (
                "variant_B3_features.csv",
                [*EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS, "suitability_proxy"],
            ),
        },
    )

    d2_rows: list[dict[str, object]] = []
    d3_rows: list[dict[str, object]] = []
    d4p8_rows: list[dict[str, object]] = []
    d4p16_rows: list[dict[str, object]] = []
    n1z_rows: list[dict[str, object]] = []
    n1zr_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        d2 = {"block_id": row["block_id"]}
        d3 = {"block_id": row["block_id"]}
        d4p8 = {"block_id": row["block_id"]}
        d4p16 = {"block_id": row["block_id"]}
        n1z = {"block_id": row["block_id"]}
        n1zr = {"block_id": row["block_id"]}
        for column in EXPLICIT_COLUMNS:
            for target in (d2, d3, d4p8, d4p16, n1z, n1zr):
                target[column] = row[column]
        for column_index, column in enumerate(EMBEDDING_COLUMNS):
            d2[column] = float(((index + column_index) % 11) - 5) / 10.0
            d3[column] = rows[(index + 3) % len(rows)][column]
            n1z[column] = row[column]
            n1zr[column] = row[column]
        for component in range(8):
            d4p8[f"embedding_pca_{component:02d}"] = row[EMBEDDING_COLUMNS[component]]
            d4p16[f"embedding_pca_{component:02d}"] = row[EMBEDDING_COLUMNS[component]]
        for component in range(8, 16):
            d4p16[f"embedding_pca_{component:02d}"] = row[EMBEDDING_COLUMNS[component]]
        d2_rows.append(d2)
        d3_rows.append(d3)
        d4p8_rows.append(d4p8)
        d4p16_rows.append(d4p16)
        n1z_rows.append(n1z)
        n1zr_rows.append(n1zr)

    _write_csv(phase8_dir / "variant_D2_features.csv", d2_rows, ["block_id", *EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS])
    _write_csv(phase8_dir / "variant_D3_features.csv", d3_rows, ["block_id", *EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS])
    _write_csv(
        phase8_dir / "variant_D4P8_features.csv",
        d4p8_rows,
        ["block_id", *EXPLICIT_COLUMNS, *[f"embedding_pca_{index:02d}" for index in range(8)]],
    )
    _write_csv(
        phase8_dir / "variant_D4P16_features.csv",
        d4p16_rows,
        ["block_id", *EXPLICIT_COLUMNS, *[f"embedding_pca_{index:02d}" for index in range(16)]],
    )
    _write_variant_manifest(
        phase8_dir,
        {
            "D2": ("variant_D2_features.csv", [*EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS]),
            "D3": ("variant_D3_features.csv", [*EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS]),
            "D4P8": (
                "variant_D4P8_features.csv",
                [*EXPLICIT_COLUMNS, *[f"embedding_pca_{index:02d}" for index in range(8)]],
            ),
            "D4P16": (
                "variant_D4P16_features.csv",
                [*EXPLICIT_COLUMNS, *[f"embedding_pca_{index:02d}" for index in range(16)]],
            ),
        },
    )

    _write_csv(normalized_dir / "variant_N1Z_features.csv", n1z_rows, ["block_id", *EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS])
    _write_csv(normalized_dir / "variant_N1ZR_features.csv", n1zr_rows, ["block_id", *EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS])
    _write_variant_manifest(
        normalized_dir,
        {
            "N1Z": ("variant_N1Z_features.csv", [*EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS]),
            "N1ZR": ("variant_N1ZR_features.csv", [*EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS]),
        },
    )
    return {
        "phase2_dir": phase2_dir,
        "phase8_dir": phase8_dir,
        "normalized_dir": normalized_dir,
    }


def test_phase38_classifies_label_boundaries_and_blocks_reward_unlock(tmp_path):
    from paper11_geofm.phase38_proxy_rebuild import build_phase38_proxy_rebuild

    paths = _fixture_inputs(tmp_path)
    analysis = build_phase38_proxy_rebuild(
        phase2_output_dir=paths["phase2_dir"],
        phase8_output_dir=paths["phase8_dir"],
        normalized_controls_dir=paths["normalized_dir"],
        label_columns=["current_farmland_label", "independent_proxy_label"],
        label_classifications="independent_proxy_label:candidate_independent_proxy",
        model_families=["logistic_elastic_net"],
        min_auc_delta=0.01,
        min_ap_delta=0.01,
    )

    assert analysis["phase"] == "phase38_proxy_rebuild"
    assert analysis["phase38_proxy_rebuild_status"] == "proxy_rebuild_supported_for_bounded_b2_b3_smoke"
    assert analysis["label_summaries"]["current_farmland_label"]["label_classification"] == "explicit_label_leakage_risk"
    assert analysis["label_summaries"]["independent_proxy_label"]["label_classification"] == "candidate_independent_proxy"
    assert analysis["row_counts"]["rebuilt_proxy_score_rows"] > 0
    assert "does not run PPO" in analysis["claim_boundary"]


def test_phase38_leakage_only_result_stays_diagnostic(tmp_path):
    from paper11_geofm.phase38_proxy_rebuild import build_phase38_proxy_rebuild

    paths = _fixture_inputs(tmp_path)
    analysis = build_phase38_proxy_rebuild(
        phase2_output_dir=paths["phase2_dir"],
        phase8_output_dir=paths["phase8_dir"],
        label_columns=["current_farmland_label"],
        model_families=["logistic_elastic_net"],
        min_auc_delta=0.01,
        min_ap_delta=0.01,
    )

    assert analysis["phase38_proxy_rebuild_status"] == "proxy_rebuild_diagnostic_only"
    assert analysis["label_summaries"]["current_farmland_label"]["label_classification"] == "explicit_label_leakage_risk"


def test_phase38_missing_label_raises(tmp_path):
    from paper11_geofm.phase38_proxy_rebuild import build_phase38_proxy_rebuild

    paths = _fixture_inputs(tmp_path)
    try:
        build_phase38_proxy_rebuild(
            phase2_output_dir=paths["phase2_dir"],
            label_columns=["missing_label"],
        )
    except ValueError as exc:
        assert "no requested label columns are available" in str(exc)
    else:
        raise AssertionError("missing label should raise")
```

- [ ] **Step 2: Run the new tests to verify they fail before implementation**

Run:

```powershell
python -m pytest tests\test_phase38_proxy_rebuild.py -q --basetemp=.pytest_tmp_phase38_t1 -p no:cacheprovider
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `paper11_geofm.phase38_proxy_rebuild`.

- [ ] **Step 3: Commit the failing test scaffold**

```powershell
git add tests\test_phase38_proxy_rebuild.py
git commit -m "test: add Phase 38 proxy rebuild fixtures"
```

## Task 2: Core Builder, Label Classification, And Model Evaluation

**Files:**
- Create: `src/paper11_geofm/phase38_proxy_rebuild.py`
- Test: `tests/test_phase38_proxy_rebuild.py`

- [ ] **Step 1: Implement the Phase 38 analysis module**

Create `src/paper11_geofm/phase38_proxy_rebuild.py` with this content:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .block_schema import EMBEDDING_COLUMNS
from .drl_inputs import load_variant_input


PHASE38_PROXY_REBUILD_CLAIM_BOUNDARY = (
    "Phase 38 is a leakage-aware proxy-rebuild diagnostic. It may train "
    "lightweight proxy models and write rebuilt proxy scores, but it does "
    "not run PPO, does not alter rewards, does not enable B2/B3 by default, "
    "and does not support final planning-performance claims."
)

DEFAULT_PHASE38_LABEL_COLUMNS = (
    "current_farmland_label",
    "farmland_or_orchard_label",
    "low_slope_farmland_label",
)

LEAKAGE_RISK_LABELS = {
    "current_farmland_label",
    "farmland_or_orchard_label",
    "low_slope_farmland_label",
}

VALID_LABEL_CLASSIFICATIONS = {
    "explicit_label_leakage_risk",
    "candidate_independent_proxy",
    "independent_validation_label",
}

DEFAULT_MODEL_FAMILIES = (
    "logistic_elastic_net",
    "random_forest",
    "hist_gradient_boosting",
)

PHASE38_LABEL_FIELDNAMES = [
    "label_column",
    "available",
    "usable",
    "label_classification",
    "valid_label_count",
    "positive_count",
    "negative_count",
    "positive_rate",
    "train_count",
    "eval_count",
    "split_source",
    "claim_boundary",
]

PHASE38_MODEL_FIELDNAMES = [
    "label_column",
    "label_classification",
    "model_family",
    "feature_family",
    "validation_status",
    "feature_count",
    "train_count",
    "eval_count",
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "accuracy",
    "positive_rate_eval",
    "calibration_bins",
    "top_diagnostics",
    "claim_boundary",
]

PHASE38_SCORE_FIELDNAMES = [
    "label_column",
    "label_classification",
    "model_family",
    "feature_family",
    "block_id",
    "split_role",
    "label_value",
    "rebuilt_proxy_score",
]


def build_phase38_proxy_rebuild(
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str | None = None,
    normalized_controls_dir: Path | str | None = None,
    label_columns: Sequence[str] | str = DEFAULT_PHASE38_LABEL_COLUMNS,
    label_classifications: Mapping[str, str] | str | None = None,
    model_families: Sequence[str] | str = DEFAULT_MODEL_FAMILIES,
    min_auc_delta: float = 0.02,
    min_ap_delta: float = 0.02,
) -> dict[str, object]:
    phase2_dir = Path(phase2_output_dir)
    block_rows = _read_csv_rows(phase2_dir / "block_geofm_features.csv", "Phase 2 block feature CSV")
    requested_labels = _normalize_csvish_values(label_columns)
    available_labels = [label for label in requested_labels if _column_available(block_rows, label)]
    if not available_labels:
        raise ValueError("Phase 38 no requested label columns are available")
    classification_overrides = _parse_label_classification_overrides(label_classifications)
    requested_models = _normalize_csvish_values(model_families)
    _validate_model_families(requested_models)
    feature_families = _build_feature_families(
        phase2_dir,
        Path(phase8_output_dir) if phase8_output_dir is not None else None,
        Path(normalized_controls_dir) if normalized_controls_dir is not None else None,
    )
    if not feature_families:
        raise ValueError("Phase 38 found no usable feature families")

    label_summaries: dict[str, dict[str, object]] = {}
    model_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for label in requested_labels:
        classification = _label_classification(label, classification_overrides)
        summary = _label_summary(block_rows, label, classification)
        label_summaries[label] = summary
        if not summary["usable"]:
            continue
        labels_by_block = _labels_by_block(block_rows, label)
        split = _split_for_blocks(block_rows, labels_by_block)
        for family in feature_families:
            for model_family in requested_models:
                evaluation = _evaluate_family(
                    label,
                    classification,
                    labels_by_block,
                    split,
                    family,
                    model_family,
                )
                model_rows.append(evaluation["model_row"])
                score_rows.extend(evaluation["score_rows"])

    status = _phase38_status(
        model_rows,
        min_auc_delta=float(min_auc_delta),
        min_ap_delta=float(min_ap_delta),
    )
    return {
        "phase": "phase38_proxy_rebuild",
        "phase38_proxy_rebuild_status": status,
        "source_paths": {
            "phase2_output_dir": str(phase2_dir),
            "phase8_output_dir": str(Path(phase8_output_dir)) if phase8_output_dir is not None else None,
            "normalized_controls_dir": str(Path(normalized_controls_dir)) if normalized_controls_dir is not None else None,
        },
        "label_columns_requested": requested_labels,
        "label_columns_available": available_labels,
        "label_classifications": {
            label: _label_classification(label, classification_overrides)
            for label in requested_labels
        },
        "model_families": requested_models,
        "feature_families": [str(family["feature_family"]) for family in feature_families],
        "min_auc_delta": float(min_auc_delta),
        "min_ap_delta": float(min_ap_delta),
        "row_counts": {
            "block_rows": len(block_rows),
            "feature_families": len(feature_families),
            "label_summaries": len(label_summaries),
            "model_rows": len(model_rows),
            "rebuilt_proxy_score_rows": len(score_rows),
        },
        "label_summaries": label_summaries,
        "label_summary_rows": list(label_summaries.values()),
        "model_rows": model_rows,
        "rebuilt_proxy_score_rows": score_rows,
        "interpretation": _phase38_interpretation(status),
        "claim_boundary": PHASE38_PROXY_REBUILD_CLAIM_BOUNDARY,
    }


def _build_feature_families(
    phase2_dir: Path,
    phase8_dir: Path | None,
    normalized_dir: Path | None,
) -> list[dict[str, object]]:
    families: list[dict[str, object]] = []
    b0 = _load_variant_or_none(phase2_dir, "B0")
    b1 = _load_variant_or_none(phase2_dir, "B1")
    b2 = _load_variant_or_none(phase2_dir, "B2")
    if b0 is not None:
        families.append(_family_from_variant("explicit_only", b0))
    if b1 is not None:
        families.append(_family_from_variant("raw_geofm_only", b1, EMBEDDING_COLUMNS))
        families.append(_family_from_variant("explicit_plus_raw_geofm", b1))
    if b2 is not None:
        families.append(_family_from_variant("suitability_proxy_only", b2, ["suitability_proxy"]))
        families.append(_family_from_variant("explicit_plus_suitability_proxy", b2))
    if phase8_dir is not None:
        for variant_id, family_id in (
            ("D2", "explicit_plus_random_geofm"),
            ("D3", "explicit_plus_shuffled_geofm"),
            ("D4P8", "explicit_plus_pca8_geofm"),
            ("D4P16", "explicit_plus_pca16_geofm"),
        ):
            variant = _load_variant_or_none(phase8_dir, variant_id)
            if variant is not None:
                families.append(_family_from_variant(family_id, variant))
    if normalized_dir is not None:
        for variant_id, family_id in (
            ("N1Z", "explicit_plus_normalized_geofm_zscore"),
            ("N1ZR", "explicit_plus_normalized_geofm_zscore_row_l2"),
        ):
            variant = _load_variant_or_none(normalized_dir, variant_id)
            if variant is not None:
                families.append(_family_from_variant(family_id, variant))
    return families


def _load_variant_or_none(output_dir: Path, variant_id: str):
    try:
        return load_variant_input(output_dir, variant_id)
    except (FileNotFoundError, ValueError):
        return None


def _family_from_variant( family_id: str, variant_input, selected_columns: Sequence[str] | None = None) -> dict[str, object]:
    if selected_columns is None:
        matrix = np.asarray(variant_input.state_matrix, dtype=float)
        columns = list(variant_input.feature_columns)
    else:
        indexes = [variant_input.feature_columns.index(column) for column in selected_columns]
        matrix = np.asarray(variant_input.state_matrix[:, indexes], dtype=float)
        columns = [str(column) for column in selected_columns]
    return {
        "feature_family": family_id,
        "block_ids": list(variant_input.block_ids),
        "feature_columns": columns,
        "matrix": matrix,
    }


def _label_summary(block_rows: Sequence[Mapping[str, object]], label_column: str, classification: str) -> dict[str, object]:
    if not _column_available(block_rows, label_column):
        return {
            "label_column": label_column,
            "available": False,
            "usable": False,
            "label_classification": classification,
            "valid_label_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_rate": "",
            "train_count": 0,
            "eval_count": 0,
            "split_source": "unavailable",
            "claim_boundary": PHASE38_PROXY_REBUILD_CLAIM_BOUNDARY,
        }
    labels = _labels_by_block(block_rows, label_column)
    split = _split_for_blocks(block_rows, labels)
    train_labels = [labels[block_id] for block_id in split["train_block_ids"]]
    eval_labels = [labels[block_id] for block_id in split["eval_block_ids"]]
    positives = sum(1 for value in labels.values() if value == 1)
    negatives = sum(1 for value in labels.values() if value == 0)
    return {
        "label_column": label_column,
        "available": True,
        "usable": _has_binary_variation(train_labels) and _has_binary_variation(eval_labels),
        "label_classification": classification,
        "valid_label_count": len(labels),
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": _round_float(positives / len(labels)) if labels else "",
        "train_count": len(train_labels),
        "eval_count": len(eval_labels),
        "split_source": split["split_source"],
        "claim_boundary": PHASE38_PROXY_REBUILD_CLAIM_BOUNDARY,
    }


def _evaluate_family(
    label_column: str,
    label_classification: str,
    labels_by_block: Mapping[str, int],
    split: Mapping[str, object],
    family: Mapping[str, object],
    model_family: str,
) -> dict[str, object]:
    block_ids = [str(block_id) for block_id in family["block_ids"]]
    matrix = np.asarray(family["matrix"], dtype=float)
    block_index = {block_id: index for index, block_id in enumerate(block_ids)}
    train_ids = [block_id for block_id in split["train_block_ids"] if block_id in block_index and block_id in labels_by_block]
    eval_ids = [block_id for block_id in split["eval_block_ids"] if block_id in block_index and block_id in labels_by_block]
    train_y = np.asarray([labels_by_block[block_id] for block_id in train_ids], dtype=int)
    eval_y = np.asarray([labels_by_block[block_id] for block_id in eval_ids], dtype=int)
    base_row = {
        "label_column": label_column,
        "label_classification": label_classification,
        "model_family": model_family,
        "feature_family": family["feature_family"],
        "feature_count": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "train_count": len(train_ids),
        "eval_count": len(eval_ids),
        "positive_rate_eval": _round_float(float(np.mean(eval_y))) if eval_y.size else "",
        "claim_boundary": PHASE38_PROXY_REBUILD_CLAIM_BOUNDARY,
    }
    if not _has_binary_variation(train_y) or not _has_binary_variation(eval_y):
        return {
            "model_row": {
                **base_row,
                "validation_status": "insufficient_label_variation",
                "roc_auc": "",
                "average_precision": "",
                "balanced_accuracy": "",
                "accuracy": "",
                "calibration_bins": [],
                "top_diagnostics": [],
            },
            "score_rows": [],
        }
    train_x = matrix[[block_index[block_id] for block_id in train_ids], :]
    eval_x = matrix[[block_index[block_id] for block_id in eval_ids], :]
    model = _make_model(model_family)
    model.fit(train_x, train_y)
    eval_probabilities = _predict_probabilities(model, eval_x)
    predictions = (eval_probabilities >= 0.5).astype(int)
    all_probabilities = _predict_probabilities(model, matrix)
    score_rows = [
        {
            "label_column": label_column,
            "label_classification": label_classification,
            "model_family": model_family,
            "feature_family": family["feature_family"],
            "block_id": block_id,
            "split_role": _split_role(block_id, split),
            "label_value": labels_by_block.get(block_id, ""),
            "rebuilt_proxy_score": _round_float(score),
        }
        for block_id, score in zip(block_ids, all_probabilities, strict=True)
    ]
    return {
        "model_row": {
            **base_row,
            "validation_status": "evaluated",
            "roc_auc": _round_float(roc_auc_score(eval_y, eval_probabilities)),
            "average_precision": _round_float(average_precision_score(eval_y, eval_probabilities)),
            "balanced_accuracy": _round_float(balanced_accuracy_score(eval_y, predictions)),
            "accuracy": _round_float(accuracy_score(eval_y, predictions)),
            "calibration_bins": _calibration_bins(eval_probabilities, eval_y),
            "top_diagnostics": _top_diagnostics(model, family),
        },
        "score_rows": score_rows,
    }


def _make_model(model_family: str):
    if model_family == "logistic_elastic_net":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                class_weight="balanced",
                max_iter=5000,
                random_state=0,
            ),
        )
    if model_family == "random_forest":
        return RandomForestClassifier(
            n_estimators=80,
            max_depth=6,
            class_weight="balanced",
            random_state=0,
            n_jobs=1,
        )
    if model_family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.05,
            random_state=0,
        )
    raise ValueError(f"Unknown Phase 38 model family: {model_family}")


def _predict_probabilities(model, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
    raise ValueError("Phase 38 model does not provide predict_proba")


def _top_diagnostics(model, family: Mapping[str, object]) -> list[dict[str, object]]:
    columns = [str(column) for column in family["feature_columns"]]
    if hasattr(model, "named_steps") and "logisticregression" in model.named_steps:
        values = np.asarray(model.named_steps["logisticregression"].coef_[0], dtype=float)
        label = "coefficient"
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        label = "importance"
    else:
        return []
    indexes = np.argsort(np.abs(values))[::-1][:5]
    return [
        {
            "feature": columns[int(index)] if int(index) < len(columns) else str(index),
            label: _round_float(values[int(index)]),
        }
        for index in indexes
    ]


def _calibration_bins(probabilities: np.ndarray, labels: np.ndarray, bin_count: int = 5) -> list[dict[str, object]]:
    bins: list[dict[str, object]] = []
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    for index in range(bin_count):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == bin_count - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            bins.append({"bin": index, "count": 0, "mean_probability": "", "positive_rate": ""})
            continue
        bins.append(
            {
                "bin": index,
                "count": int(np.sum(mask)),
                "mean_probability": _round_float(float(np.mean(probabilities[mask]))),
                "positive_rate": _round_float(float(np.mean(labels[mask]))),
            }
        )
    return bins


def _phase38_status(model_rows: Sequence[Mapping[str, object]], min_auc_delta: float, min_ap_delta: float) -> str:
    evaluated = [row for row in model_rows if str(row.get("validation_status")) == "evaluated"]
    if not evaluated:
        return "proxy_rebuild_inputs_insufficient"
    non_leakage = [
        row for row in evaluated
        if str(row.get("label_classification")) in {"candidate_independent_proxy", "independent_validation_label"}
    ]
    if not non_leakage:
        return "proxy_rebuild_diagnostic_only"
    candidate_families = {
        "raw_geofm_only",
        "explicit_plus_raw_geofm",
        "explicit_plus_normalized_geofm_zscore",
        "explicit_plus_normalized_geofm_zscore_row_l2",
    }
    control_families = {
        "explicit_only",
        "explicit_plus_random_geofm",
        "explicit_plus_shuffled_geofm",
    }
    for row in non_leakage:
        if str(row.get("feature_family")) not in candidate_families:
            continue
        peers = [
            peer for peer in evaluated
            if peer.get("label_column") == row.get("label_column")
            and peer.get("model_family") == row.get("model_family")
            and str(peer.get("feature_family")) in control_families
        ]
        if not peers:
            continue
        control_auc = max(_metric(peer, "roc_auc") for peer in peers)
        control_ap = max(_metric(peer, "average_precision") for peer in peers)
        if _metric(row, "roc_auc") >= control_auc + min_auc_delta and _metric(row, "average_precision") >= control_ap + min_ap_delta:
            return "proxy_rebuild_supported_for_bounded_b2_b3_smoke"
    return "proxy_rebuild_diagnostic_only"


def _phase38_interpretation(status: str) -> str:
    if status == "proxy_rebuild_supported_for_bounded_b2_b3_smoke":
        return "A non-leakage label cleared the GeoFM-derived model margin against explicit and diagnostic controls; this only permits a bounded B2/B3 smoke."
    if status == "proxy_rebuild_diagnostic_only":
        return "Models ran, but support is absent, control-limited, or based only on leakage-risk labels."
    return "Phase 38 inputs did not provide enough usable labels, features, or split coverage."


def _read_csv_rows(path: Path, label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _column_available(rows: Sequence[Mapping[str, object]], column: str) -> bool:
    return any(column in row and str(row.get(column, "")).strip() != "" for row in rows)


def _labels_by_block(rows: Sequence[Mapping[str, object]], label_column: str) -> dict[str, int]:
    labels: dict[str, int] = {}
    for row in rows:
        block_id = str(row.get("block_id", "")).strip()
        label = _parse_binary_label(row.get(label_column))
        if block_id and label is not None:
            labels[block_id] = label
    return labels


def _split_for_blocks(rows: Sequence[Mapping[str, object]], labels_by_block: Mapping[str, int]) -> dict[str, object]:
    split_by_block: dict[str, str] = {}
    for row in rows:
        block_id = str(row.get("block_id", "")).strip()
        if block_id not in labels_by_block:
            continue
        split_text = str(row.get("split", "")).strip().lower()
        if split_text:
            split_by_block[block_id] = split_text
    train_ids = [block_id for block_id, split in split_by_block.items() if split in {"train", "training"}]
    eval_ids = [block_id for block_id, split in split_by_block.items() if split in {"test", "val", "valid", "validation", "eval", "evaluation"}]
    if train_ids and eval_ids:
        return {"split_source": "split_column", "train_block_ids": sorted(train_ids), "eval_block_ids": sorted(eval_ids)}
    ordered = sorted(labels_by_block)
    eval_ids = ordered[::5]
    train_ids = [block_id for block_id in ordered if block_id not in set(eval_ids)]
    if not eval_ids and ordered:
        eval_ids = ordered[-1:]
        train_ids = ordered[:-1]
    return {"split_source": "deterministic_modulo_split", "train_block_ids": train_ids, "eval_block_ids": eval_ids}


def _split_role(block_id: str, split: Mapping[str, object]) -> str:
    if block_id in set(split["train_block_ids"]):
        return "train"
    if block_id in set(split["eval_block_ids"]):
        return "eval"
    return "unused"


def _parse_binary_label(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes"}:
        return 1
    if text in {"0", "0.0", "false", "no"}:
        return 0
    return None


def _has_binary_variation(values: Sequence[int] | np.ndarray) -> bool:
    clean = [int(value) for value in values]
    return 0 in clean and 1 in clean


def _label_classification(label_column: str, overrides: Mapping[str, str]) -> str:
    if label_column in overrides:
        return overrides[label_column]
    if label_column in LEAKAGE_RISK_LABELS:
        return "explicit_label_leakage_risk"
    return "candidate_independent_proxy"


def _parse_label_classification_overrides(values: Mapping[str, str] | str | None) -> dict[str, str]:
    if values is None:
        return {}
    if isinstance(values, Mapping):
        raw_items = values.items()
    else:
        raw_items = []
        for part in str(values).split(","):
            text = part.strip()
            if not text:
                continue
            if ":" not in text:
                raise ValueError(f"Invalid label classification override: {text}")
            label, classification = text.split(":", 1)
            raw_items.append((label.strip(), classification.strip()))
    overrides: dict[str, str] = {}
    for label, classification in raw_items:
        if classification not in VALID_LABEL_CLASSIFICATIONS:
            raise ValueError(f"Invalid label classification for {label}: {classification}")
        overrides[str(label)] = str(classification)
    return overrides


def _validate_model_families(model_families: Sequence[str]) -> None:
    valid = set(DEFAULT_MODEL_FAMILIES)
    for model_family in model_families:
        if model_family not in valid:
            raise ValueError(f"Unknown Phase 38 model family: {model_family}")


def _normalize_csvish_values(values: Sequence[str] | str) -> list[str]:
    if isinstance(values, str):
        raw = values.split(",")
    else:
        raw = []
        for value in values:
            raw.extend(str(value).split(","))
    return [str(value).strip() for value in raw if str(value).strip()]


def _metric(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return float("-inf")
    return float(value)


def _round_float(value: float | int) -> float:
    return round(float(value), 10)
```

- [ ] **Step 2: Run the focused tests**

Run:

```powershell
python -m pytest tests\test_phase38_proxy_rebuild.py -q --basetemp=.pytest_tmp_phase38_t2 -p no:cacheprovider
```

Expected: PASS for the first three tests.

- [ ] **Step 3: Commit the core builder**

```powershell
git add src\paper11_geofm\phase38_proxy_rebuild.py tests\test_phase38_proxy_rebuild.py
git commit -m "feat: add Phase 38 proxy rebuild core"
```

## Task 3: Artifact Writer And Markdown Report

**Files:**
- Modify: `src/paper11_geofm/phase38_proxy_rebuild.py`
- Modify: `tests/test_phase38_proxy_rebuild.py`

- [ ] **Step 1: Add failing writer test**

Append this test to `tests/test_phase38_proxy_rebuild.py`:

```python
def test_phase38_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase38_proxy_rebuild import (
        build_phase38_proxy_rebuild,
        write_phase38_proxy_rebuild_artifacts,
    )

    paths = _fixture_inputs(tmp_path)
    analysis = build_phase38_proxy_rebuild(
        phase2_output_dir=paths["phase2_dir"],
        phase8_output_dir=paths["phase8_dir"],
        normalized_controls_dir=paths["normalized_dir"],
        label_columns=["current_farmland_label", "independent_proxy_label"],
        label_classifications="independent_proxy_label:candidate_independent_proxy",
        model_families=["logistic_elastic_net"],
        min_auc_delta=0.01,
        min_ap_delta=0.01,
    )

    artifacts = write_phase38_proxy_rebuild_artifacts(analysis, tmp_path / "outputs")

    assert artifacts["label_summary_csv"].name == "phase38_label_summary.csv"
    assert artifacts["model_summary_csv"].name == "phase38_model_summary.csv"
    assert artifacts["rebuilt_proxy_scores_csv"].name == "phase38_rebuilt_proxy_scores.csv"
    assert artifacts["diagnosis_json"].name == "phase38_proxy_rebuild.json"
    assert artifacts["diagnosis_md"].name == "phase38_proxy_rebuild.md"
    assert all(path.exists() for path in artifacts.values())
    saved = json.loads(artifacts["diagnosis_json"].read_text(encoding="utf-8"))
    assert saved["phase"] == "phase38_proxy_rebuild"
    markdown = artifacts["diagnosis_md"].read_text(encoding="utf-8")
    assert "Phase 38 Proxy-Rebuild" in markdown
    assert "proxy_rebuild_supported_for_bounded_b2_b3_smoke" in markdown
```

- [ ] **Step 2: Run the writer test to verify it fails**

Run:

```powershell
python -m pytest tests\test_phase38_proxy_rebuild.py::test_phase38_writer_outputs_csv_json_and_markdown -q --basetemp=.pytest_tmp_phase38_t3_fail -p no:cacheprovider
```

Expected: FAIL with `ImportError` or `AttributeError` for `write_phase38_proxy_rebuild_artifacts`.

- [ ] **Step 3: Add writer and serialization functions**

Append these functions to `src/paper11_geofm/phase38_proxy_rebuild.py`:

```python
def write_phase38_proxy_rebuild_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    label_summary_path = output_path / "phase38_label_summary.csv"
    model_summary_path = output_path / "phase38_model_summary.csv"
    scores_path = output_path / "phase38_rebuilt_proxy_scores.csv"
    diagnosis_json_path = output_path / "phase38_proxy_rebuild.json"
    diagnosis_md_path = output_path / "phase38_proxy_rebuild.md"

    _write_csv_mapping_rows(
        label_summary_path,
        PHASE38_LABEL_FIELDNAMES,
        analysis.get("label_summary_rows"),
        "label_summary_rows",
    )
    _write_csv_mapping_rows(
        model_summary_path,
        PHASE38_MODEL_FIELDNAMES,
        analysis.get("model_rows"),
        "model_rows",
    )
    _write_csv_mapping_rows(
        scores_path,
        PHASE38_SCORE_FIELDNAMES,
        analysis.get("rebuilt_proxy_score_rows"),
        "rebuilt_proxy_score_rows",
    )
    diagnosis_json_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    diagnosis_md_path.write_text(_phase38_markdown(analysis), encoding="utf-8")
    return {
        "label_summary_csv": label_summary_path,
        "model_summary_csv": model_summary_path,
        "rebuilt_proxy_scores_csv": scores_path,
        "diagnosis_json": diagnosis_json_path,
        "diagnosis_md": diagnosis_md_path,
    }


def _phase38_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Phase 38 Proxy-Rebuild",
        "",
        f"Status: {analysis.get('phase38_proxy_rebuild_status', '')}",
        "",
        "## Label Summary",
        "",
        "| Label | Classification | Usable | Train / Eval | Positives / Negatives |",
        "|---|---|---:|---:|---:|",
    ]
    label_rows = analysis.get("label_summary_rows")
    if isinstance(label_rows, list):
        for row in label_rows:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| {label} | {classification} | {usable} | {train} / {eval} | {pos} / {neg} |".format(
                    label=row.get("label_column", ""),
                    classification=row.get("label_classification", ""),
                    usable=row.get("usable", ""),
                    train=row.get("train_count", ""),
                    eval=row.get("eval_count", ""),
                    pos=row.get("positive_count", ""),
                    neg=row.get("negative_count", ""),
                )
            )
    lines.extend(["", "## Model Summary", ""])
    model_rows = analysis.get("model_rows")
    if isinstance(model_rows, list):
        grouped: dict[str, list[Mapping[str, object]]] = {}
        for row in model_rows:
            if isinstance(row, Mapping):
                grouped.setdefault(str(row.get("label_column", "")), []).append(row)
        for label in sorted(grouped):
            lines.extend(
                [
                    f"### {label}",
                    "",
                    "| Model | Feature family | ROC AUC | AP | Balanced accuracy | Status |",
                    "|---|---|---:|---:|---:|---|",
                ]
            )
            for row in sorted(
                grouped[label],
                key=lambda item: (str(item.get("model_family", "")), str(item.get("feature_family", ""))),
            ):
                lines.append(
                    "| {model} | {family} | {auc} | {ap} | {bal} | {status} |".format(
                        model=row.get("model_family", ""),
                        family=row.get("feature_family", ""),
                        auc=row.get("roc_auc", ""),
                        ap=row.get("average_precision", ""),
                        bal=row.get("balanced_accuracy", ""),
                        status=row.get("validation_status", ""),
                    )
                )
            lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            str(analysis.get("interpretation", "")),
            "",
            "## Boundary",
            "",
            str(analysis.get("claim_boundary", "")),
            "",
            "Leakage-risk labels may validate the pipeline but must not unlock B2/B3 reward.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv_mapping_rows(path: Path, fieldnames: Sequence[str], rows: object, label: str) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 38 analysis is missing {label}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 38 {label} contains a non-mapping row")
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, list):
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
    if isinstance(value, np.generic):
        return value.item()
    return value
```

- [ ] **Step 4: Run focused writer test**

Run:

```powershell
python -m pytest tests\test_phase38_proxy_rebuild.py::test_phase38_writer_outputs_csv_json_and_markdown -q --basetemp=.pytest_tmp_phase38_t3_pass -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Run all Phase 38 tests**

Run:

```powershell
python -m pytest tests\test_phase38_proxy_rebuild.py -q --basetemp=.pytest_tmp_phase38_t3_all -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 6: Commit artifact writer**

```powershell
git add src\paper11_geofm\phase38_proxy_rebuild.py tests\test_phase38_proxy_rebuild.py
git commit -m "feat: write Phase 38 proxy rebuild artifacts"
```

## Task 4: CLI Runner

**Files:**
- Create: `experiments/phase38_proxy_rebuild/run_phase38_proxy_rebuild.py`
- Modify: `tests/test_phase38_proxy_rebuild.py`

- [ ] **Step 1: Add failing CLI test**

Append this test to `tests/test_phase38_proxy_rebuild.py`:

```python
def test_phase38_cli_writes_outputs(tmp_path):
    paths = _fixture_inputs(tmp_path)
    script = ROOT / "experiments" / "phase38_proxy_rebuild" / "run_phase38_proxy_rebuild.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--phase2-output-dir",
            str(paths["phase2_dir"]),
            "--phase8-output-dir",
            str(paths["phase8_dir"]),
            "--normalized-controls-dir",
            str(paths["normalized_dir"]),
            "--output-dir",
            str(tmp_path / "cli_outputs"),
            "--label-columns",
            "current_farmland_label,independent_proxy_label",
            "--label-classifications",
            "independent_proxy_label:candidate_independent_proxy",
            "--model-families",
            "logistic_elastic_net",
            "--min-auc-delta",
            "0.01",
            "--min-ap-delta",
            "0.01",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Phase 38 proxy-rebuild status:" in result.stdout
    assert "Claim boundary:" in result.stdout
    assert (tmp_path / "cli_outputs" / "phase38_proxy_rebuild.json").exists()
    assert (tmp_path / "cli_outputs" / "phase38_rebuilt_proxy_scores.csv").exists()
```

- [ ] **Step 2: Run CLI test to verify it fails**

Run:

```powershell
python -m pytest tests\test_phase38_proxy_rebuild.py::test_phase38_cli_writes_outputs -q --basetemp=.pytest_tmp_phase38_t4_fail -p no:cacheprovider
```

Expected: FAIL because `experiments/phase38_proxy_rebuild/run_phase38_proxy_rebuild.py` does not exist.

- [ ] **Step 3: Implement CLI runner**

Create `experiments/phase38_proxy_rebuild/run_phase38_proxy_rebuild.py` with this content:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase38_proxy_rebuild import (
    build_phase38_proxy_rebuild,
    write_phase38_proxy_rebuild_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Paper11 Phase 38 leakage-aware suitability proxy rebuild over "
            "existing feature tables."
        )
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--phase8-output-dir", type=Path)
    parser.add_argument("--normalized-controls-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--label-columns",
        default="current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label",
    )
    parser.add_argument("--label-classifications", default="")
    parser.add_argument(
        "--model-families",
        default="logistic_elastic_net,random_forest,hist_gradient_boosting",
    )
    parser.add_argument("--min-auc-delta", type=float, default=0.02)
    parser.add_argument("--min-ap-delta", type=float, default=0.02)
    args = parser.parse_args(argv)

    try:
        analysis = build_phase38_proxy_rebuild(
            phase2_output_dir=args.phase2_output_dir,
            phase8_output_dir=args.phase8_output_dir,
            normalized_controls_dir=args.normalized_controls_dir,
            label_columns=args.label_columns,
            label_classifications=args.label_classifications,
            model_families=args.model_families,
            min_auc_delta=args.min_auc_delta,
            min_ap_delta=args.min_ap_delta,
        )
        paths = write_phase38_proxy_rebuild_artifacts(analysis, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 38 proxy-rebuild status: {analysis['phase38_proxy_rebuild_status']}")
    print(
        "Available labels: "
        f"{','.join(str(label) for label in analysis['label_columns_available'])}"
    )
    print(f"Label summary CSV: {paths['label_summary_csv']}")
    print(f"Model summary CSV: {paths['model_summary_csv']}")
    print(f"Rebuilt proxy scores CSV: {paths['rebuilt_proxy_scores_csv']}")
    print(f"Diagnosis JSON: {paths['diagnosis_json']}")
    print(f"Diagnosis Markdown: {paths['diagnosis_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI test**

Run:

```powershell
python -m pytest tests\test_phase38_proxy_rebuild.py::test_phase38_cli_writes_outputs -q --basetemp=.pytest_tmp_phase38_t4_pass -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Run Phase 38 and Phase 36 focused tests**

Run:

```powershell
python -m pytest tests\test_phase38_proxy_rebuild.py tests\test_phase36_suitability_proxy_validation.py -q --basetemp=.pytest_tmp_phase38_t4_all -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 6: Commit CLI runner**

```powershell
git add experiments\phase38_proxy_rebuild\run_phase38_proxy_rebuild.py tests\test_phase38_proxy_rebuild.py
git commit -m "feat: add Phase 38 proxy rebuild runner"
```

## Task 5: Real Bishan Phase 38 Run

**Files:**
- Generated ignored outputs under `experiments/phase38_proxy_rebuild/outputs/real_bishan`
- No tracked source changes in this task

- [ ] **Step 1: Run focused tests before real data**

Run:

```powershell
python -m pytest tests\test_phase38_proxy_rebuild.py tests\test_phase36_suitability_proxy_validation.py -q --basetemp=.pytest_tmp_phase38_pre_real -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the real Phase 38 diagnostic**

Run:

```powershell
python experiments\phase38_proxy_rebuild\run_phase38_proxy_rebuild.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --normalized-controls-dir experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental\derived_normalized_controls --output-dir experiments\phase38_proxy_rebuild\outputs\real_bishan --label-columns current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label --model-families logistic_elastic_net,random_forest,hist_gradient_boosting
```

Expected: command exits 0 and prints `Phase 38 proxy-rebuild status:` plus artifact paths.

- [ ] **Step 3: Inspect the real status and row counts**

Run:

```powershell
Get-Content -Raw experiments\phase38_proxy_rebuild\outputs\real_bishan\phase38_proxy_rebuild.json
```

Expected: JSON includes `phase38_proxy_rebuild_status`, `row_counts`, `model_rows`, and `rebuilt_proxy_score_rows`.

- [ ] **Step 4: Commit no generated artifacts**

Run:

```powershell
git status --short
```

Expected: generated files under `experiments/**/outputs/` are ignored or left untracked only if `.gitignore` does not cover a new path. Do not add real generated CSV/JSON/Markdown artifacts to Git.

## Task 6: Documentation And Handoff

**Files:**
- Modify: `README.md`
- Modify: `paper/phase28_results/README.md`
- Create: `paper/phase28_results/12_phase38_proxy_rebuild.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [ ] **Step 1: Read exact Phase 38 values for the result doc**

Run:

```powershell
Get-Content -Raw experiments\phase38_proxy_rebuild\outputs\real_bishan\phase38_proxy_rebuild.json
```

Expected: the JSON contains these fields to copy into the result document:

- `phase38_proxy_rebuild_status`
- `row_counts.block_rows`
- `row_counts.model_rows`
- `row_counts.rebuilt_proxy_score_rows`
- `interpretation`

- [ ] **Step 2: Add reviewer-facing Phase 38 result doc**

Create `paper/phase28_results/12_phase38_proxy_rebuild.md` with:

- H1 `# Phase 38 Proxy-Rebuild Diagnostic`
- one-sentence argument stating that Phase 38 rebuilds the suitability proxy under leakage-aware spatial held-out validation before any B2/B3 reward integration;
- current experiment snapshot listing the Phase 2, Phase 8, Phase 30 normalized-control inputs and `experiments/phase38_proxy_rebuild/outputs/real_bishan`;
- main result with the exact status copied from `phase38_proxy_rebuild_status`;
- row counts copied from the real JSON;
- interpretation copied from the real JSON;
- claim boundary stating that Phase 38 is diagnostic, does not run PPO, does not alter rewards, does not enable B2/B3 by default, does not prove GeoFM agronomic validity, and does not support final planning-performance claims.
- [ ] **Step 3: Update README and result index**

In `README.md`, add one Phase 38 bullet after the Phase 37 entry in the repository layout and add the real Phase 38 command after the Phase 37 section. State the exact status and preserve the B2/B3 boundary.

In `paper/phase28_results/README.md`, add `12_phase38_proxy_rebuild.md` to the file list and add the exact Phase 38 reproduction command.

- [ ] **Step 4: Update manifest and handoff**

Add these rows to `reproducibility/FILE_MANIFEST.tsv` using the same tab-separated style as existing rows:

```text
src/paper11_geofm/phase38_proxy_rebuild.py	source	Phase 38 leakage-aware suitability-proxy rebuild module.
experiments/phase38_proxy_rebuild/run_phase38_proxy_rebuild.py	experiment	Executable Phase 38 proxy-rebuild runner over existing feature tables.
tests/test_phase38_proxy_rebuild.py	verification	Pytest checks for Phase 38 label boundaries, model evaluation, artifact writing, and CLI behavior.
paper/phase28_results/12_phase38_proxy_rebuild.md	documentation	Reviewer-facing interpretation of the Phase 38 proxy-rebuild diagnostic.
docs/superpowers/specs/2026-06-27-phase38-proxy-rebuild-design.md	documentation	Phase 38 proxy-rebuild design specification.
docs/superpowers/plans/2026-06-27-phase38-proxy-rebuild.md	documentation	Phase 38 proxy-rebuild implementation plan.
```

In `docs/superpowers/phase33_current_progress_handoff.md`, add a Phase 38 section with:

- latest implementation commits;
- real output directory;
- generated artifact names;
- real status and row counts;
- verification commands and results;
- explicit statement whether B2/B3 remains blocked or only a bounded smoke is allowed.

- [ ] **Step 5: Run documentation checks**

Run:

```powershell
$pattern = 'REPLACE_WITH_' + 'REAL|TB' + 'D|TO' + 'DO'
rg -n $pattern README.md paper\phase28_results\README.md paper\phase28_results\12_phase38_proxy_rebuild.md docs\superpowers\phase33_current_progress_handoff.md
```

Expected: no matches.

- [ ] **Step 6: Run final verification**

Run:

```powershell
python -m pytest tests\test_phase38_proxy_rebuild.py tests\test_phase36_suitability_proxy_validation.py -q --basetemp=.pytest_tmp_phase38_final -p no:cacheprovider
python scripts\smoke_check.py
```

Expected: all selected tests pass and `Paper11 smoke check passed.`

- [ ] **Step 7: Commit documentation**

```powershell
git add README.md paper\phase28_results\README.md paper\phase28_results\12_phase38_proxy_rebuild.md reproducibility\FILE_MANIFEST.tsv docs\superpowers\phase33_current_progress_handoff.md
git commit -m "docs: record Phase 38 proxy rebuild result"
```

## Task 7: Closeout

**Files:**
- No new files unless verification reveals a tracked documentation omission

- [ ] **Step 1: Check final repository state**

Run:

```powershell
git status --short --branch
git log -5 --oneline
```

Expected: only ignored generated outputs are absent from status; latest commits include Phase 38 test/core/runner/docs commits.

- [ ] **Step 2: Summarize exact outcome**

Report:

- final Phase 38 status;
- whether B2/B3 remains blocked or bounded smoke is allowed;
- tests run and pass counts;
- latest commit hash;
- any real-run limitation, including if only leakage-risk labels were available.