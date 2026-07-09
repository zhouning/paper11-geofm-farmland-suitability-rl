# Phase 67 Candidate Reward/Label Target Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only Phase 67 audit that inventories candidate diagnostic reward/label targets and decides whether any target can support later diagnostic training without violating existing suitability-reward and independent-label gates.

**Architecture:** Add one focused Phase 67 module that loads existing feature tables and gate artifacts, constructs candidate per-block targets, audits leakage and gate constraints, computes explicit-versus-GeoFM information gain, and writes JSON/CSV/Markdown artifacts. A thin CLI runner will call the module; the real run result will be recorded in `paper/phase28_results` only after generated artifacts exist.

**Tech Stack:** Python 3, NumPy, existing `paper11_geofm.drl_inputs.load_variant_input`, raw Phase 2 `block_geofm_features.csv` label/embedding rows, existing `planning_reward` helpers, Phase 66 rank metric patterns, pytest, PowerShell commands using `D:\adk\.venv\Scripts\python.exe`.

---

## File Structure

- Create `src/paper11_geofm/phase67_candidate_reward_label_target_audit.py`
  - Owns the Phase 67 claim boundary, candidate target construction, target inventory, leakage and gate audit, information-gain diagnostics, conservative target gate, artifact writing, and full run orchestration.
- Create `experiments/phase67_candidate_reward_label_target_audit/run_phase67_candidate_reward_label_target_audit.py`
  - Thin CLI wrapper. It accepts Phase 2, Phase 8, Phase 61, Phase 10, Phase 18, Phase 66, optional Phase 39/40 paths, tile metadata, variants, label columns, top-k values, and output directory.
- Create `tests/test_phase67_candidate_reward_label_target_audit.py`
  - Covers inventory rows, gate classification, feature-group separation, residual target construction, information gain, all four gate statuses, writer output, and CLI parsing.
- Create `paper/phase28_results/33_phase67_candidate_reward_label_target_audit.md`
  - Filled after the real Phase 67 run. It should report status, key numeric evidence, reproduction command, and claim boundary.
- Do not modify `paper/submission/final/*`.

---

### Task 1: Candidate Target Inventory

**Files:**
- Create: `src/paper11_geofm/phase67_candidate_reward_label_target_audit.py`
- Create: `tests/test_phase67_candidate_reward_label_target_audit.py`

- [ ] **Step 1: Write failing tests for candidate target inventory**

Create `tests/test_phase67_candidate_reward_label_target_audit.py` with these tests and helpers:

```python
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _feature_rows() -> list[dict[str, object]]:
    return [
        {
            "block_id": "b1",
            "explicit_feature_00": 5.0,
            "explicit_feature_01": 0.0,
            "explicit_feature_02": 0.0,
            "explicit_feature_04": 0.4,
            "explicit_feature_07": 0.1,
            "explicit_feature_09": 0.0,
            "explicit_feature_10": 0.0,
            "explicit_feature_13": 0.8,
            "explicit_feature_16": 0.9,
            "current_farmland_label": 1,
            "farmland_or_orchard_label": 1,
            "low_slope_farmland_label": 1,
            "embedding_pca_00": 0.9,
            "embedding_pca_01": 0.0,
        },
        {
            "block_id": "b2",
            "explicit_feature_00": 4.0,
            "explicit_feature_01": 5.0,
            "explicit_feature_02": 7.0,
            "explicit_feature_04": 0.2,
            "explicit_feature_07": 0.6,
            "explicit_feature_09": 0.0,
            "explicit_feature_10": 0.0,
            "explicit_feature_13": 0.7,
            "explicit_feature_16": 0.8,
            "current_farmland_label": 1,
            "farmland_or_orchard_label": 1,
            "low_slope_farmland_label": 0,
            "embedding_pca_00": 0.8,
            "embedding_pca_01": 0.0,
        },
        {
            "block_id": "b3",
            "explicit_feature_00": 3.0,
            "explicit_feature_01": 15.0,
            "explicit_feature_02": 20.0,
            "explicit_feature_04": 0.0,
            "explicit_feature_07": 0.1,
            "explicit_feature_09": 0.2,
            "explicit_feature_10": 0.0,
            "explicit_feature_13": 0.2,
            "explicit_feature_16": 0.3,
            "current_farmland_label": 0,
            "farmland_or_orchard_label": 1,
            "low_slope_farmland_label": 0,
            "embedding_pca_00": 0.3,
            "embedding_pca_01": 0.0,
        },
        {
            "block_id": "b4",
            "explicit_feature_00": 1.0,
            "explicit_feature_01": 25.0,
            "explicit_feature_02": 35.0,
            "explicit_feature_04": 0.0,
            "explicit_feature_07": 0.0,
            "explicit_feature_09": 0.5,
            "explicit_feature_10": 0.4,
            "explicit_feature_13": 0.1,
            "explicit_feature_16": 0.1,
            "current_farmland_label": 0,
            "farmland_or_orchard_label": 0,
            "low_slope_farmland_label": 0,
            "embedding_pca_00": 0.1,
            "embedding_pca_01": 0.0,
        },
    ]


def test_phase67_builds_base_weak_geofm_and_residual_candidate_targets():
    from paper11_geofm.phase67_candidate_reward_label_target_audit import (
        build_phase67_candidate_targets,
    )

    targets = build_phase67_candidate_targets(
        rows=_feature_rows(),
        label_columns=[
            "current_farmland_label",
            "farmland_or_orchard_label",
            "low_slope_farmland_label",
        ],
        representation_prefixes=["embedding_pca_"],
    )
    by_id = {target["target_id"]: target for target in targets}

    assert "base_planning_reward" in by_id
    assert "weak_label_current_farmland_label" in by_id
    assert "weak_label_farmland_or_orchard_label" in by_id
    assert "weak_label_low_slope_farmland_label" in by_id
    assert "geofm_norm_embedding_pca" in by_id
    assert "residual_base_after_explicit" in by_id
    assert "residual_weak_label_current_farmland_label_after_explicit" in by_id
    assert "residual_geofm_norm_embedding_pca_after_explicit" in by_id
    assert by_id["base_planning_reward"]["target_family"] == "base_reward"
    assert by_id["weak_label_current_farmland_label"]["target_kind"] == "binary"
    assert by_id["geofm_norm_embedding_pca"]["depends_on_geofm"] is True
    assert by_id["residual_base_after_explicit"]["target_family"] == "explicit_residual"


def test_phase67_inventory_marks_zero_variance_and_missing_targets_unusable():
    from paper11_geofm.phase67_candidate_reward_label_target_audit import (
        build_phase67_candidate_target_inventory,
    )

    targets = [
        {
            "target_id": "constant_score",
            "target_family": "fixture",
            "values_by_block": {"b1": 1.0, "b2": 1.0, "b3": 1.0},
            "higher_is_better": True,
            "directly_uses_explicit": False,
            "depends_on_geofm": False,
            "source_detail": "fixture",
            "target_kind": "continuous",
        },
        {
            "target_id": "partial_score",
            "target_family": "fixture",
            "values_by_block": {"b1": 0.1, "b2": None, "b3": 0.9},
            "higher_is_better": True,
            "directly_uses_explicit": False,
            "depends_on_geofm": False,
            "source_detail": "fixture",
            "target_kind": "continuous",
        },
    ]

    rows = build_phase67_candidate_target_inventory(targets, expected_block_ids=["b1", "b2", "b3"])
    by_id = {row["target_id"]: row for row in rows}

    assert by_id["constant_score"]["usable"] is False
    assert by_id["constant_score"]["unusable_reason"] == "zero_variance"
    assert by_id["partial_score"]["non_missing_count"] == 2
    assert by_id["partial_score"]["usable"] is True
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase67_candidate_reward_label_target_audit.py -q --basetemp=.pytest_tmp_phase67_task1_red -p no:cacheprovider
```

Expected: `ModuleNotFoundError` for `paper11_geofm.phase67_candidate_reward_label_target_audit`.

- [ ] **Step 3: Add Phase 67 module with candidate target construction**

Create `src/paper11_geofm/phase67_candidate_reward_label_target_audit.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
import statistics

import numpy as np

from .drl_inputs import load_variant_input
from .planning_reward import (
    BASE_PLANNING_REWARD_REQUIRED_COLUMNS,
    compute_base_planning_reward,
)


PHASE67_CLAIM_BOUNDARY = (
    "Phase 67 is a read-only candidate reward/label target audit. It inventories "
    "diagnostic targets and checks leakage, gate status, and explicit-versus-GeoFM "
    "information gain. It does not train a policy, modify rewards, enable "
    "suitability reward, create B2/B3 variants, prove GeoFM advantage, or justify "
    "formal submission-level claims."
)

PHASE67_STATUS_CANDIDATE_FOUND = "candidate_target_found_for_diagnostic_training"
PHASE67_STATUS_ONLY_LEAKAGE_OR_EXPLICIT = "only_leakage_or_explicit_targets_found"
PHASE67_STATUS_INDEPENDENT_LABEL_REQUIRED = "independent_label_required_before_reward_redesign"
PHASE67_STATUS_INSUFFICIENT = "insufficient"

DEFAULT_PHASE67_LABEL_COLUMNS = (
    "current_farmland_label",
    "farmland_or_orchard_label",
    "low_slope_farmland_label",
)
DEFAULT_PHASE67_REPRESENTATION_PREFIXES = ("embedding_pca_", "embedding_mean_", "projection_")
DEFAULT_PHASE67_TOP_K_VALUES = (8, 16, 32)

PHASE67_INVENTORY_FIELDNAMES = [
    "target_id",
    "target_family",
    "target_kind",
    "source_detail",
    "row_count",
    "non_missing_count",
    "unique_count",
    "min_value",
    "max_value",
    "mean_value",
    "variance",
    "higher_is_better",
    "directly_uses_explicit",
    "depends_on_geofm",
    "self_referential",
    "usable",
    "unusable_reason",
    "claim_boundary",
]


def _round_float(value: object, digits: int = 10) -> float:
    rounded = round(float(value), digits)
    compact = round(rounded, 6)
    if abs(rounded - compact) < 5.0e-8:
        return compact
    return rounded


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    if np.isnan(parsed):
        return None
    return parsed


def _numeric_values_by_block(
    rows: Sequence[Mapping[str, object]],
    column: str,
) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for row in rows:
        block_id = str(row.get("block_id", "")).strip()
        if not block_id:
            raise ValueError("Phase 67 rows require block_id")
        values[block_id] = _safe_float(row.get(column))
    return values


def _feature_matrix(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    block_ids: list[str] = []
    matrix_rows: list[list[float]] = []
    for row in rows:
        block_id = str(row.get("block_id", "")).strip()
        if not block_id:
            raise ValueError("Phase 67 rows require block_id")
        values = []
        for column in columns:
            value = _safe_float(row.get(column))
            if value is None:
                raise ValueError(f"Phase 67 missing numeric column {column} for {block_id}")
            values.append(value)
        block_ids.append(block_id)
        matrix_rows.append(values)
    return block_ids, np.asarray(matrix_rows, dtype=np.float64)


def _ols_residual_values(
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("Phase 67 residual target inputs are not aligned")
    keep = np.std(x, axis=0) > 1.0e-12
    if not bool(np.any(keep)):
        return y - np.mean(y)
    z = x[:, keep]
    z = (z - np.mean(z, axis=0)) / np.std(z, axis=0)
    design = np.column_stack([np.ones(z.shape[0]), z])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - (design @ coeffs)


def _infer_kind(values: Sequence[float]) -> str:
    unique = sorted({float(value) for value in values})
    if unique and all(value in {0.0, 1.0} for value in unique):
        return "binary"
    if unique and all(float(value).is_integer() for value in unique) and len(unique) <= 10:
        return "ordinal"
    return "continuous"


def _representation_target_id(columns: Sequence[str]) -> str:
    if any(str(column).startswith("embedding_pca_") for column in columns):
        return "geofm_norm_embedding_pca"
    if any(str(column).startswith("embedding_mean_") for column in columns):
        return "geofm_norm_embedding_mean"
    if any(str(column).startswith("projection_") for column in columns):
        return "geofm_norm_projection"
    return "geofm_norm_representation"


def _residual_target_id(source_target_id: str) -> str:
    if source_target_id == "base_planning_reward":
        return "residual_base_after_explicit"
    return f"residual_{source_target_id}_after_explicit"


def build_phase67_candidate_targets(
    rows: Sequence[Mapping[str, object]],
    label_columns: Sequence[str] = DEFAULT_PHASE67_LABEL_COLUMNS,
    representation_prefixes: Sequence[str] = DEFAULT_PHASE67_REPRESENTATION_PREFIXES,
) -> list[dict[str, object]]:
    if not rows:
        raise ValueError("Phase 67 candidate target construction requires rows")
    fieldnames = set(rows[0].keys())
    missing_reward = [column for column in BASE_PLANNING_REWARD_REQUIRED_COLUMNS if column not in fieldnames]
    if missing_reward:
        raise ValueError(f"Phase 67 rows are missing base reward columns: {missing_reward}")
    block_ids = [str(row["block_id"]) for row in rows]
    base_values = {
        str(row["block_id"]): compute_base_planning_reward(row)
        for row in rows
    }
    targets: list[dict[str, object]] = [
        {
            "target_id": "base_planning_reward",
            "target_family": "base_reward",
            "target_kind": "continuous",
            "source_detail": "planning_reward.compute_base_planning_reward",
            "values_by_block": base_values,
            "higher_is_better": True,
            "directly_uses_explicit": True,
            "depends_on_geofm": False,
            "self_referential": False,
        }
    ]
    for label_column in label_columns:
        if label_column not in fieldnames:
            continue
        values = _numeric_values_by_block(rows, label_column)
        present = [value for value in values.values() if value is not None]
        targets.append(
            {
                "target_id": f"weak_label_{label_column}",
                "target_family": "weak_label",
                "target_kind": _infer_kind(present),
                "source_detail": label_column,
                "values_by_block": values,
                "higher_is_better": True,
                "directly_uses_explicit": True,
                "depends_on_geofm": False,
                "self_referential": False,
            }
        )
    representation_columns = [
        column for column in sorted(fieldnames)
        if any(str(column).startswith(prefix) for prefix in representation_prefixes)
    ]
    if representation_columns:
        rep_block_ids, rep_matrix = _feature_matrix(rows, representation_columns)
        norm_values = np.linalg.norm(rep_matrix, axis=1)
        targets.append(
            {
                "target_id": _representation_target_id(representation_columns),
                "target_family": "geofm_self_reference",
                "target_kind": "continuous",
                "source_detail": ";".join(representation_columns),
                "values_by_block": {
                    block_id: _round_float(value)
                    for block_id, value in zip(rep_block_ids, norm_values, strict=True)
                },
                "higher_is_better": True,
                "directly_uses_explicit": False,
                "depends_on_geofm": True,
                "self_referential": True,
            }
        )
    explicit_columns = [column for column in sorted(fieldnames) if str(column).startswith("explicit_feature_")]
    source_targets_for_residuals = list(targets)
    for source_target in source_targets_for_residuals:
        values_by_block = dict(source_target.get("values_by_block", {}))
        aligned_rows = []
        aligned_blocks = []
        y_values = []
        for row in rows:
            block_id = str(row["block_id"])
            value = _safe_float(values_by_block.get(block_id))
            if value is None:
                continue
            aligned_rows.append(row)
            aligned_blocks.append(block_id)
            y_values.append(value)
        if len(y_values) < 2:
            continue
        _, explicit_matrix = _feature_matrix(aligned_rows, explicit_columns)
        residual = _ols_residual_values(np.asarray(y_values, dtype=np.float64), explicit_matrix)
        source_target_id = str(source_target.get("target_id", ""))
        targets.append(
            {
                "target_id": _residual_target_id(source_target_id),
                "target_family": "explicit_residual",
                "target_kind": "continuous",
                "source_detail": f"{source_target_id} residual after explicit columns",
                "values_by_block": {
                    block_id: _round_float(value)
                    for block_id, value in zip(aligned_blocks, residual, strict=True)
                },
                "higher_is_better": True,
                "directly_uses_explicit": True,
                "depends_on_geofm": bool(source_target.get("depends_on_geofm", False)),
                "self_referential": bool(source_target.get("self_referential", False)),
            }
        )
    return targets


def build_phase67_candidate_target_inventory(
    targets: Sequence[Mapping[str, object]],
    expected_block_ids: Sequence[str],
) -> list[dict[str, object]]:
    expected = [str(block_id) for block_id in expected_block_ids]
    rows: list[dict[str, object]] = []
    for target in targets:
        values_by_block = dict(target.get("values_by_block", {}))
        values = [_safe_float(values_by_block.get(block_id)) for block_id in expected]
        present = [float(value) for value in values if value is not None]
        unique = sorted(set(present))
        variance = float(np.var(present)) if present else 0.0
        usable = bool(present) and variance > 1.0e-12
        unusable_reason = ""
        if not present:
            unusable_reason = "no_non_missing_values"
        elif variance <= 1.0e-12:
            unusable_reason = "zero_variance"
        rows.append(
            {
                "target_id": str(target.get("target_id", "")),
                "target_family": str(target.get("target_family", "")),
                "target_kind": str(target.get("target_kind", "")),
                "source_detail": str(target.get("source_detail", "")),
                "row_count": len(expected),
                "non_missing_count": len(present),
                "unique_count": len(unique),
                "min_value": "" if not present else _round_float(min(present)),
                "max_value": "" if not present else _round_float(max(present)),
                "mean_value": "" if not present else _round_float(statistics.mean(present)),
                "variance": _round_float(variance),
                "higher_is_better": bool(target.get("higher_is_better", True)),
                "directly_uses_explicit": bool(target.get("directly_uses_explicit", False)),
                "depends_on_geofm": bool(target.get("depends_on_geofm", False)),
                "self_referential": bool(target.get("self_referential", False)),
                "usable": usable,
                "unusable_reason": unusable_reason,
                "claim_boundary": PHASE67_CLAIM_BOUNDARY,
            }
        )
    return rows
```

- [ ] **Step 4: Run tests and verify Task 1 passes**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase67_candidate_reward_label_target_audit.py -q --basetemp=.pytest_tmp_phase67_task1_green -p no:cacheprovider
```

Expected: the two Phase 67 tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src\paper11_geofm\phase67_candidate_reward_label_target_audit.py tests\test_phase67_candidate_reward_label_target_audit.py
git commit -m "feat: add Phase 67 candidate target inventory"
```

Expected: commit succeeds.

---

### Task 2: Leakage And Gate Audit

**Files:**
- Modify: `src/paper11_geofm/phase67_candidate_reward_label_target_audit.py`
- Modify: `tests/test_phase67_candidate_reward_label_target_audit.py`

- [ ] **Step 1: Add failing tests for gate risk classification**

Append these tests:

```python
def test_phase67_gate_audit_blocks_base_weak_and_geofm_self_reference_targets():
    from paper11_geofm.phase67_candidate_reward_label_target_audit import (
        build_phase67_candidate_target_gate_audit,
    )

    inventory_rows = [
        {
            "target_id": "base_planning_reward",
            "target_family": "base_reward",
            "usable": True,
            "directly_uses_explicit": True,
            "depends_on_geofm": False,
            "self_referential": False,
        },
        {
            "target_id": "weak_label_current_farmland_label",
            "target_family": "weak_label",
            "usable": True,
            "directly_uses_explicit": True,
            "depends_on_geofm": False,
            "self_referential": False,
        },
        {
            "target_id": "geofm_norm_embedding_pca",
            "target_family": "geofm_self_reference",
            "usable": True,
            "directly_uses_explicit": False,
            "depends_on_geofm": True,
            "self_referential": True,
        },
    ]
    gate_context = {
        "phase10_status": "not_ready_for_suitability_reward",
        "phase10_recommendation": "do_not_enable_suitability_reward",
        "phase18_suitability_reward_allowed": False,
        "phase39_status": "independent_label_inputs_missing",
        "phase40_status": "independent_label_inputs_missing",
    }

    rows = build_phase67_candidate_target_gate_audit(inventory_rows, gate_context)
    by_id = {row["target_id"]: row for row in rows}

    assert by_id["base_planning_reward"]["gate_risk"] == "explicit_reward_defined"
    assert by_id["base_planning_reward"]["reward_training_allowed"] is False
    assert by_id["weak_label_current_farmland_label"]["gate_risk"] == "explicit_label_leakage_risk"
    assert by_id["geofm_norm_embedding_pca"]["gate_risk"] == "geofm_self_reference"
    assert by_id["geofm_norm_embedding_pca"]["diagnostic_only_allowed"] is True


def test_phase67_gate_context_accepts_real_phase10_and_phase18_keys():
    from paper11_geofm.phase67_candidate_reward_label_target_audit import (
        build_phase67_gate_context,
    )

    context = build_phase67_gate_context(
        phase10={"status": "not_ready_for_suitability_reward", "recommendation": "do_not_enable_suitability_reward"},
        phase18={"suitability_reward_allowed": False, "phase10_status": "not_ready_for_suitability_reward"},
        phase39={},
        phase40={},
    )

    assert context["phase10_status"] == "not_ready_for_suitability_reward"
    assert context["phase10_recommendation"] == "do_not_enable_suitability_reward"
    assert context["phase18_suitability_reward_allowed"] is False
    assert context["phase39_status"] == "missing"
    assert context["phase40_status"] == "missing"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase67_candidate_reward_label_target_audit.py::test_phase67_gate_audit_blocks_base_weak_and_geofm_self_reference_targets tests\test_phase67_candidate_reward_label_target_audit.py::test_phase67_gate_context_accepts_real_phase10_and_phase18_keys -q --basetemp=.pytest_tmp_phase67_task2_red -p no:cacheprovider
```

Expected: imports fail for gate audit helpers.

- [ ] **Step 3: Implement gate context and target gate audit**

Append to `src/paper11_geofm/phase67_candidate_reward_label_target_audit.py`:

```python
PHASE67_GATE_AUDIT_FIELDNAMES = [
    "target_id",
    "target_family",
    "usable",
    "gate_risk",
    "diagnostic_only_allowed",
    "reward_training_allowed",
    "independent_label_required",
    "phase10_status",
    "phase10_recommendation",
    "phase18_suitability_reward_allowed",
    "phase39_status",
    "phase40_status",
    "reason",
    "claim_boundary",
]


def build_phase67_gate_context(
    phase10: Mapping[str, object],
    phase18: Mapping[str, object],
    phase39: Mapping[str, object] | None = None,
    phase40: Mapping[str, object] | None = None,
) -> dict[str, object]:
    phase39 = {} if phase39 is None else dict(phase39)
    phase40 = {} if phase40 is None else dict(phase40)
    return {
        "phase10_status": str(phase10.get("phase10_status", phase10.get("status", ""))),
        "phase10_recommendation": str(phase10.get("phase10_recommendation", phase10.get("recommendation", ""))),
        "phase18_suitability_reward_allowed": bool(phase18.get("suitability_reward_allowed", False)),
        "phase39_status": str(phase39.get("status", phase39.get("phase39_status", "missing"))),
        "phase40_status": str(phase40.get("status", phase40.get("phase40_status", "missing"))),
    }


def _target_gate_risk(inventory_row: Mapping[str, object]) -> tuple[str, str]:
    target_family = str(inventory_row.get("target_family", ""))
    target_id = str(inventory_row.get("target_id", ""))
    if target_family == "base_reward" or target_id == "base_planning_reward":
        return "explicit_reward_defined", "Target is the current explicit-feature-defined base reward."
    if target_family == "weak_label":
        return "explicit_label_leakage_risk", "Target is an existing weak DLTB/slope-derived label."
    if bool(inventory_row.get("self_referential")) or target_family == "geofm_self_reference":
        return "geofm_self_reference", "Target is constructed from GeoFM representation values."
    if target_family == "explicit_residual":
        return "diagnostic_only_allowed", "Residual target is allowed only for diagnostic analysis."
    return "independent_label_missing", "Target is not backed by a registered independent label."


def build_phase67_candidate_target_gate_audit(
    inventory_rows: Sequence[Mapping[str, object]],
    gate_context: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for inventory in inventory_rows:
        gate_risk, reason = _target_gate_risk(inventory)
        usable = bool(inventory.get("usable", False))
        diagnostic_only_allowed = usable and gate_risk in {
            "geofm_self_reference",
            "diagnostic_only_allowed",
            "explicit_label_leakage_risk",
            "explicit_reward_defined",
        }
        reward_training_allowed = False
        independent_required = gate_risk in {
            "explicit_reward_defined",
            "explicit_label_leakage_risk",
            "geofm_self_reference",
            "independent_label_missing",
        }
        rows.append(
            {
                "target_id": str(inventory.get("target_id", "")),
                "target_family": str(inventory.get("target_family", "")),
                "usable": usable,
                "gate_risk": gate_risk,
                "diagnostic_only_allowed": diagnostic_only_allowed,
                "reward_training_allowed": reward_training_allowed,
                "independent_label_required": independent_required,
                "phase10_status": gate_context.get("phase10_status", ""),
                "phase10_recommendation": gate_context.get("phase10_recommendation", ""),
                "phase18_suitability_reward_allowed": bool(gate_context.get("phase18_suitability_reward_allowed", False)),
                "phase39_status": gate_context.get("phase39_status", ""),
                "phase40_status": gate_context.get("phase40_status", ""),
                "reason": reason,
                "claim_boundary": PHASE67_CLAIM_BOUNDARY,
            }
        )
    return rows
```

- [ ] **Step 4: Run tests and verify Task 2 passes**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase67_candidate_reward_label_target_audit.py -q --basetemp=.pytest_tmp_phase67_task2_green -p no:cacheprovider
```

Expected: all current Phase 67 tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src\paper11_geofm\phase67_candidate_reward_label_target_audit.py tests\test_phase67_candidate_reward_label_target_audit.py
git commit -m "feat: add Phase 67 candidate target gate audit"
```

Expected: commit succeeds.

---

### Task 3: Information Gain And Candidate Gate

**Files:**
- Modify: `src/paper11_geofm/phase67_candidate_reward_label_target_audit.py`
- Modify: `tests/test_phase67_candidate_reward_label_target_audit.py`

- [ ] **Step 1: Add failing tests for information gain and status gate**

Append:

```python
def test_phase67_information_gain_detects_geofm_residual_signal():
    from paper11_geofm.phase67_candidate_reward_label_target_audit import (
        build_phase67_candidate_target_information_gain,
    )

    rows = _feature_rows()
    targets = [
        {
            "target_id": "geofm_explained_residual",
            "target_family": "explicit_residual",
            "values_by_block": {"b1": 0.9, "b2": 0.8, "b3": 0.3, "b4": 0.1},
            "higher_is_better": True,
            "directly_uses_explicit": False,
            "depends_on_geofm": False,
            "self_referential": False,
            "target_kind": "continuous",
            "source_detail": "fixture",
        }
    ]

    info_rows = build_phase67_candidate_target_information_gain(
        feature_rows_by_variant={"D4P8": rows},
        targets=targets,
        top_k_values=[2],
    )
    row = info_rows[0]

    assert row["target_id"] == "geofm_explained_residual"
    assert row["variant_id"] == "D4P8"
    assert row["explicit_proxy_r2"] >= 0.0
    assert row["geofm_proxy_r2"] > 0.9
    assert row["geofm_spearman"] > 0.9
    assert row["geofm_minus_explicit_r2"] > 0.0
    assert row["geofm_topk_enrichment"] == 1.0


def test_phase67_candidate_gate_covers_all_statuses():
    from paper11_geofm.phase67_candidate_reward_label_target_audit import (
        build_phase67_candidate_target_gate,
    )

    candidate_info = [
        {
            "target_id": "residual_a",
            "target_family": "explicit_residual",
            "variant_id": "D4P8",
            "geofm_minus_explicit_r2": 0.2,
            "geofm_minus_d6_r2": 0.1,
            "residual_after_explicit_r2": 0.2,
        }
    ]
    diagnostic_gate = [
        {"target_id": "residual_a", "usable": True, "gate_risk": "diagnostic_only_allowed", "diagnostic_only_allowed": True},
    ]
    explicit_info = [
        {
            "target_id": "base_planning_reward",
            "target_family": "base_reward",
            "variant_id": "B0",
            "geofm_minus_explicit_r2": -0.9,
            "geofm_minus_d6_r2": 0.0,
            "residual_after_explicit_r2": 0.0,
        }
    ]
    explicit_gate = [
        {"target_id": "base_planning_reward", "usable": True, "gate_risk": "explicit_reward_defined", "diagnostic_only_allowed": True},
    ]

    assert build_phase67_candidate_target_gate([], candidate_info, diagnostic_gate)["phase67_status"] == "candidate_target_found_for_diagnostic_training"
    assert build_phase67_candidate_target_gate([], explicit_info, explicit_gate)["phase67_status"] == "only_leakage_or_explicit_targets_found"
    assert build_phase67_candidate_target_gate([], [], [])["phase67_status"] == "independent_label_required_before_reward_redesign"
    assert build_phase67_candidate_target_gate(["missing artifact"], explicit_info, explicit_gate)["phase67_status"] == "insufficient"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase67_candidate_reward_label_target_audit.py::test_phase67_information_gain_detects_geofm_residual_signal tests\test_phase67_candidate_reward_label_target_audit.py::test_phase67_candidate_gate_covers_all_statuses -q --basetemp=.pytest_tmp_phase67_task3_red -p no:cacheprovider
```

Expected: imports fail for information-gain and gate functions.

- [ ] **Step 3: Implement information gain metrics and gate**

Append:

```python
PHASE67_INFORMATION_GAIN_FIELDNAMES = [
    "target_id",
    "target_family",
    "variant_id",
    "n_blocks",
    "explicit_proxy_r2",
    "all_explicit_proxy_r2",
    "geofm_proxy_r2",
    "explicit_spearman",
    "geofm_spearman",
    "combined_proxy_r2",
    "residual_after_explicit_r2",
    "geofm_minus_explicit_r2",
    "geofm_minus_d6_r2",
    "geofm_topk_enrichment",
    "explicit_topk_enrichment",
    "claim_boundary",
]


def _rank_average(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def phase67_topk_enrichment(feature_values: Sequence[float], target_values: Sequence[float], top_k: int) -> float:
    x = np.asarray(feature_values, dtype=np.float64)
    y = np.asarray(target_values, dtype=np.float64)
    if x.size != y.size:
        raise ValueError("Phase 67 top-k enrichment inputs must have equal length")
    k = min(int(top_k), int(x.size))
    if k <= 0:
        return 0.0
    target_top = set(np.argsort(-y, kind="mergesort")[:k].tolist())
    high_top = set(np.argsort(-x, kind="mergesort")[:k].tolist())
    low_top = set(np.argsort(x, kind="mergesort")[:k].tolist())
    return _round_float(max(len(target_top & high_top), len(target_top & low_top)) / k)


def phase67_spearman(feature_values: Sequence[float], target_values: Sequence[float]) -> float:
    x = np.asarray(feature_values, dtype=np.float64)
    y = np.asarray(target_values, dtype=np.float64)
    if x.size != y.size or x.size == 0 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    rx = _rank_average(x)
    ry = _rank_average(y)
    corr = np.corrcoef(rx, ry)[0, 1]
    if np.isnan(corr):
        return 0.0
    return _round_float(corr)


def _proxy_r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] == 0:
        return 0.0
    keep = np.std(x, axis=0) > 1.0e-12
    if not bool(np.any(keep)) or float(np.std(y)) == 0.0:
        return 0.0
    z = x[:, keep]
    z = (z - np.mean(z, axis=0)) / np.std(z, axis=0)
    design = np.column_stack([np.ones(z.shape[0]), z])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coeffs
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 1.0e-12:
        return 0.0
    residual = float(np.sum((y - predicted) ** 2))
    return _round_float(max(0.0, min(1.0, 1.0 - residual / total)))


def _feature_group_indexes(fieldnames: Sequence[str]) -> dict[str, list[str]]:
    reward_explicit = [column for column in fieldnames if column in BASE_PLANNING_REWARD_REQUIRED_COLUMNS]
    all_explicit = [column for column in fieldnames if str(column).startswith("explicit_feature_")]
    geofm = [column for column in fieldnames if str(column).startswith(("embedding_pca_", "embedding_mean_", "projection_"))]
    return {"reward_explicit": reward_explicit, "all_explicit": all_explicit, "geofm": geofm}


def _aligned_target_vector(
    rows: Sequence[Mapping[str, object]],
    target: Mapping[str, object],
) -> tuple[list[Mapping[str, object]], np.ndarray]:
    values_by_block = dict(target.get("values_by_block", {}))
    aligned_rows = []
    values = []
    for row in rows:
        block_id = str(row.get("block_id", ""))
        value = _safe_float(values_by_block.get(block_id))
        if value is None:
            continue
        aligned_rows.append(row)
        values.append(value)
    if not values:
        raise ValueError(f"Phase 67 target has no aligned values: {target.get('target_id')}")
    return aligned_rows, np.asarray(values, dtype=np.float64)


def _matrix_from_rows(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.zeros((len(rows), 0), dtype=np.float64)
    matrix = []
    for row in rows:
        values = []
        for column in columns:
            value = _safe_float(row.get(column))
            if value is None:
                value = 0.0
            values.append(value)
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float64)


def build_phase67_candidate_target_information_gain(
    feature_rows_by_variant: Mapping[str, Sequence[Mapping[str, object]]],
    targets: Sequence[Mapping[str, object]],
    top_k_values: Sequence[int] = DEFAULT_PHASE67_TOP_K_VALUES,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    d6_r2_by_target: dict[str, float] = {}
    for variant_id, feature_rows in feature_rows_by_variant.items():
        if not feature_rows:
            continue
        groups = _feature_group_indexes(feature_rows[0].keys())
        for target in targets:
            aligned_rows, y = _aligned_target_vector(feature_rows, target)
            reward_x = _matrix_from_rows(aligned_rows, groups["reward_explicit"])
            explicit_x = _matrix_from_rows(aligned_rows, groups["all_explicit"])
            geofm_x = _matrix_from_rows(aligned_rows, groups["geofm"])
            combined_x = np.column_stack([explicit_x, geofm_x]) if geofm_x.shape[1] else explicit_x
            explicit_r2 = _proxy_r2(reward_x, y)
            all_explicit_r2 = _proxy_r2(explicit_x, y)
            geofm_r2 = _proxy_r2(geofm_x, y)
            combined_r2 = _proxy_r2(combined_x, y)
            residual_after_explicit = max(0.0, combined_r2 - all_explicit_r2)
            geofm_scores = np.linalg.norm(geofm_x, axis=1) if geofm_x.shape[1] else np.zeros(len(y))
            explicit_scores = np.mean(explicit_x, axis=1) if explicit_x.shape[1] else np.zeros(len(y))
            geofm_topk = 0.0
            if geofm_x.shape[1]:
                geofm_topk = max(phase67_topk_enrichment(geofm_scores, y, k) for k in top_k_values)
            explicit_topk = max(phase67_topk_enrichment(explicit_scores, y, k) for k in top_k_values)
            target_id = str(target.get("target_id", ""))
            if str(variant_id).startswith("D6"):
                d6_r2_by_target[target_id] = max(d6_r2_by_target.get(target_id, 0.0), geofm_r2)
            rows.append(
                {
                    "target_id": target_id,
                    "target_family": str(target.get("target_family", "")),
                    "variant_id": str(variant_id),
                    "n_blocks": len(y),
                    "explicit_proxy_r2": explicit_r2,
                    "all_explicit_proxy_r2": all_explicit_r2,
                    "geofm_proxy_r2": geofm_r2,
                    "explicit_spearman": phase67_spearman(explicit_scores, y),
                    "geofm_spearman": phase67_spearman(geofm_scores, y),
                    "combined_proxy_r2": combined_r2,
                    "residual_after_explicit_r2": _round_float(residual_after_explicit),
                    "geofm_minus_explicit_r2": _round_float(geofm_r2 - all_explicit_r2),
                    "geofm_minus_d6_r2": 0.0,
                    "geofm_topk_enrichment": _round_float(geofm_topk),
                    "explicit_topk_enrichment": _round_float(explicit_topk),
                    "claim_boundary": PHASE67_CLAIM_BOUNDARY,
                }
            )
    for row in rows:
        row["geofm_minus_d6_r2"] = _round_float(
            float(row["geofm_proxy_r2"]) - d6_r2_by_target.get(str(row["target_id"]), 0.0)
        )
    return rows


def build_phase67_candidate_target_gate(
    coverage_issues: Sequence[object],
    information_gain_rows: Sequence[Mapping[str, object]],
    gate_audit_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if coverage_issues:
        return {
            "phase67_status": PHASE67_STATUS_INSUFFICIENT,
            "coverage_issues": list(coverage_issues),
            "claim_boundary": PHASE67_CLAIM_BOUNDARY,
        }
    if not information_gain_rows or not gate_audit_rows:
        return {
            "phase67_status": PHASE67_STATUS_INDEPENDENT_LABEL_REQUIRED,
            "coverage_issues": [],
            "claim_boundary": PHASE67_CLAIM_BOUNDARY,
        }
    gate_by_target = {str(row.get("target_id")): row for row in gate_audit_rows}
    candidate_rows = [
        row for row in information_gain_rows
        if bool(gate_by_target.get(str(row.get("target_id")), {}).get("diagnostic_only_allowed", False))
        and str(gate_by_target.get(str(row.get("target_id")), {}).get("gate_risk")) == "diagnostic_only_allowed"
        and float(row.get("residual_after_explicit_r2", 0.0)) >= 0.05
        and float(row.get("geofm_minus_explicit_r2", 0.0)) >= 0.05
        and float(row.get("geofm_minus_d6_r2", 0.0)) >= 0.0
    ]
    if candidate_rows:
        status = PHASE67_STATUS_CANDIDATE_FOUND
    elif all(
        str(row.get("gate_risk")) in {"explicit_reward_defined", "explicit_label_leakage_risk"}
        for row in gate_audit_rows
        if bool(row.get("usable", False))
    ):
        status = PHASE67_STATUS_ONLY_LEAKAGE_OR_EXPLICIT
    else:
        status = PHASE67_STATUS_INDEPENDENT_LABEL_REQUIRED
    return {
        "phase67_status": status,
        "coverage_issues": [],
        "candidate_count": len(candidate_rows),
        "best_candidate_target_ids": sorted({str(row.get("target_id")) for row in candidate_rows}),
        "claim_boundary": PHASE67_CLAIM_BOUNDARY,
    }
```

- [ ] **Step 4: Run all current Phase 67 tests**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase67_candidate_reward_label_target_audit.py -q --basetemp=.pytest_tmp_phase67_task3_green -p no:cacheprovider
```

Expected: all current Phase 67 tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add src\paper11_geofm\phase67_candidate_reward_label_target_audit.py tests\test_phase67_candidate_reward_label_target_audit.py
git commit -m "feat: add Phase 67 target information gain gate"
```

Expected: commit succeeds.

---

### Task 4: Writer, CLI, And Run Orchestration

**Files:**
- Modify: `src/paper11_geofm/phase67_candidate_reward_label_target_audit.py`
- Create: `experiments/phase67_candidate_reward_label_target_audit/run_phase67_candidate_reward_label_target_audit.py`
- Modify: `tests/test_phase67_candidate_reward_label_target_audit.py`

- [ ] **Step 1: Add failing tests for writer, parser, and tiny run wrapper**

Append:

```python
def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_variant_fixture(output_dir: Path, variant_id: str, rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    explicit_columns = sorted(column for column in rows[0] if str(column).startswith("explicit_feature_"))
    representation_columns = sorted(
        column
        for column in rows[0]
        if str(column).startswith(("embedding_pca_", "projection_"))
    )
    required_columns = explicit_columns if variant_id == "B0" else explicit_columns + representation_columns
    table = output_dir / f"variant_{variant_id}_features.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["block_id", *required_columns])
        writer.writeheader()
        for row in rows:
            writer.writerow({"block_id": row["block_id"], **{column: row[column] for column in required_columns}})
    manifest = {
        "variants": {
            variant_id: {
                "ready": True,
                "feature_table": table.name,
                "required_columns": required_columns,
                "reward": "base_planning_reward",
                "state_groups": ["synthetic"],
            }
        }
    }
    (output_dir / "experiment_variants.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_phase67_writer_outputs_json_csv_and_markdown(tmp_path):
    from paper11_geofm.phase67_candidate_reward_label_target_audit import (
        write_phase67_artifacts,
    )

    analysis = {
        "phase": "phase67_candidate_reward_label_target_audit",
        "candidate_target_inventory_rows": [{"target_id": "base_planning_reward", "claim_boundary": "phase67"}],
        "candidate_target_gate_audit_rows": [{"target_id": "base_planning_reward", "gate_risk": "explicit_reward_defined", "claim_boundary": "phase67"}],
        "candidate_target_information_gain_rows": [{"target_id": "base_planning_reward", "variant_id": "B0", "claim_boundary": "phase67"}],
        "candidate_target_summary_rows": [{"target_id": "base_planning_reward", "summary": "control", "claim_boundary": "phase67"}],
        "candidate_target_gate": {"phase67_status": "only_leakage_or_explicit_targets_found"},
        "claim_boundary": "phase67",
    }

    paths = write_phase67_artifacts(analysis, tmp_path / "outputs")

    assert paths["inventory_csv"].name == "phase67_candidate_target_inventory.csv"
    assert paths["gate_audit_csv"].name == "phase67_candidate_target_gate_audit.csv"
    assert paths["information_gain_csv"].name == "phase67_candidate_target_information_gain.csv"
    assert paths["summary_csv"].name == "phase67_candidate_target_summary.csv"
    assert paths["audit_json"].name == "phase67_candidate_reward_label_target_audit.json"
    assert paths["audit_md"].name == "phase67_candidate_reward_label_target_audit.md"
    saved = json.loads(paths["audit_json"].read_text(encoding="utf-8"))
    assert saved["phase67_status"] == "only_leakage_or_explicit_targets_found"
    assert "Phase 67 Candidate Reward/Label Target Audit" in paths["audit_md"].read_text(encoding="utf-8")


def test_phase67_cli_parser_accepts_required_and_optional_inputs():
    runner_path = ROOT / "experiments" / "phase67_candidate_reward_label_target_audit" / "run_phase67_candidate_reward_label_target_audit.py"
    spec = importlib.util.spec_from_file_location("phase67_runner_args", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--phase2-output-dir", "phase2",
            "--phase8-output-dir", "phase8",
            "--phase61-output-dir", "phase61",
            "--tile-index-csv", "tiles.csv",
            "--phase10-json", "phase10.json",
            "--phase18-json", "phase18.json",
            "--phase66-json", "phase66.json",
            "--phase39-json", "phase39.json",
            "--phase40-json", "phase40.json",
            "--variants", "B0,D4P8,D6R8",
            "--label-columns", "current_farmland_label,farmland_or_orchard_label",
            "--top-k-values", "8,16",
            "--output-dir", "outputs",
        ]
    )

    assert args.phase2_output_dir == Path("phase2")
    assert args.phase39_json == Path("phase39.json")
    assert args.output_dir == Path("outputs")


def test_phase67_run_wrapper_loads_fixture_and_returns_gate(tmp_path):
    from paper11_geofm.phase67_candidate_reward_label_target_audit import (
        run_phase67_candidate_reward_label_target_audit,
    )

    phase2 = tmp_path / "phase2"
    _write_variant_fixture(phase2, "B0", _feature_rows())
    _write_csv(phase2 / "block_geofm_features.csv", _feature_rows())
    phase10 = tmp_path / "phase10.json"
    phase10.write_text(json.dumps({"status": "not_ready_for_suitability_reward", "recommendation": "do_not_enable_suitability_reward"}), encoding="utf-8")
    phase18 = tmp_path / "phase18.json"
    phase18.write_text(json.dumps({"suitability_reward_allowed": False, "phase10_status": "not_ready_for_suitability_reward"}), encoding="utf-8")
    phase66 = tmp_path / "phase66.json"
    phase66.write_text(json.dumps({"phase66_status": "base_reward_target_masks_geofm_signal"}), encoding="utf-8")

    analysis = run_phase67_candidate_reward_label_target_audit(
        phase2_output_dir=phase2,
        phase8_output_dir=None,
        phase61_output_dir=None,
        tile_index_csv=None,
        phase10_json=phase10,
        phase18_json=phase18,
        phase66_json=phase66,
        phase39_json=None,
        phase40_json=None,
        variants=["B0"],
        label_columns=["current_farmland_label", "farmland_or_orchard_label", "low_slope_farmland_label"],
        top_k_values=[2],
    )

    assert analysis["phase"] == "phase67_candidate_reward_label_target_audit"
    target_ids = {row["target_id"] for row in analysis["candidate_target_inventory_rows"]}
    assert "weak_label_current_farmland_label" in target_ids
    assert "geofm_norm_embedding_pca" in target_ids
    assert len(analysis["candidate_target_inventory_rows"]) >= 4
    assert analysis["candidate_target_gate"]["phase67_status"] in {
        "candidate_target_found_for_diagnostic_training",
        "only_leakage_or_explicit_targets_found",
        "independent_label_required_before_reward_redesign",
        "insufficient",
    }
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase67_candidate_reward_label_target_audit.py::test_phase67_writer_outputs_json_csv_and_markdown tests\test_phase67_candidate_reward_label_target_audit.py::test_phase67_cli_parser_accepts_required_and_optional_inputs tests\test_phase67_candidate_reward_label_target_audit.py::test_phase67_run_wrapper_loads_fixture_and_returns_gate -q --basetemp=.pytest_tmp_phase67_task4_red -p no:cacheprovider
```

Expected: imports or runner file fail because writer, CLI, and run wrapper are not implemented.

- [ ] **Step 3: Implement artifact writer and run wrapper**

Append:

```python
PHASE67_SUMMARY_FIELDNAMES = [
    "target_id",
    "summary",
    "claim_boundary",
]


def _load_json_object(path: Path | str | None, label: str) -> dict[str, object]:
    if path is None:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {label}: {json_path}")
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return loaded


def _read_csv_dict_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 67 CSV input: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _variant_input_rows(source_dir: Path, variant_id: str) -> list[dict[str, object]]:
    loaded = load_variant_input(source_dir, variant_id)
    rows: list[dict[str, object]] = []
    for row_index, block_id in enumerate(loaded.block_ids):
        row: dict[str, object] = {"block_id": str(block_id)}
        for column_index, column in enumerate(loaded.feature_columns):
            row[str(column)] = float(loaded.state_matrix[row_index, column_index])
        rows.append(row)
    return rows


def _load_phase2_candidate_rows(
    phase2_output_dir: Path | str,
    base_rows: Sequence[Mapping[str, object]],
    label_columns: Sequence[str],
) -> list[dict[str, object]]:
    phase2_dir = Path(phase2_output_dir)
    block_rows = _read_csv_dict_rows(phase2_dir / "block_geofm_features.csv")
    by_block = {str(row.get("block_id", "")): row for row in block_rows}
    rows: list[dict[str, object]] = []
    for base_row in base_rows:
        block_id = str(base_row.get("block_id", ""))
        merged = dict(base_row)
        extra = by_block.get(block_id, {})
        for column, value in extra.items():
            column_name = str(column)
            if column_name in merged:
                continue
            if column_name in label_columns or column_name.startswith(("embedding_mean_", "embedding_pca_", "projection_")):
                merged[column_name] = value
        rows.append(merged)
    return rows


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


def _phase67_markdown(analysis: Mapping[str, object]) -> str:
    gate = dict(analysis.get("candidate_target_gate", {}))
    lines = [
        "# Phase 67 Candidate Reward/Label Target Audit",
        "",
        f"Status: {gate.get('phase67_status', '')}",
        "",
        f"Candidate count: {gate.get('candidate_count', 0)}",
        f"Best candidate target IDs: {gate.get('best_candidate_target_ids', [])}",
        "",
        "Claim boundary:",
        str(analysis.get("claim_boundary", PHASE67_CLAIM_BOUNDARY)),
        "",
    ]
    return "\n".join(lines)


def write_phase67_artifacts(analysis: Mapping[str, object], output_dir: Path | str) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "inventory_csv": output_path / "phase67_candidate_target_inventory.csv",
        "gate_audit_csv": output_path / "phase67_candidate_target_gate_audit.csv",
        "information_gain_csv": output_path / "phase67_candidate_target_information_gain.csv",
        "summary_csv": output_path / "phase67_candidate_target_summary.csv",
        "audit_json": output_path / "phase67_candidate_reward_label_target_audit.json",
        "audit_md": output_path / "phase67_candidate_reward_label_target_audit.md",
    }
    _write_csv_rows(paths["inventory_csv"], PHASE67_INVENTORY_FIELDNAMES, analysis.get("candidate_target_inventory_rows", []))
    _write_csv_rows(paths["gate_audit_csv"], PHASE67_GATE_AUDIT_FIELDNAMES, analysis.get("candidate_target_gate_audit_rows", []))
    _write_csv_rows(paths["information_gain_csv"], PHASE67_INFORMATION_GAIN_FIELDNAMES, analysis.get("candidate_target_information_gain_rows", []))
    _write_csv_rows(paths["summary_csv"], PHASE67_SUMMARY_FIELDNAMES, analysis.get("candidate_target_summary_rows", []))
    saved = dict(analysis)
    saved["phase67_status"] = dict(analysis.get("candidate_target_gate", {})).get("phase67_status", PHASE67_STATUS_INSUFFICIENT)
    paths["audit_json"].write_text(json.dumps(_json_ready(saved), indent=2, sort_keys=True), encoding="utf-8")
    paths["audit_md"].write_text(_phase67_markdown(analysis), encoding="utf-8")
    return paths


def _csvish(value: Sequence[str] | str) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(item) for item in value if str(item).strip()]


def run_phase67_candidate_reward_label_target_audit(
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str | None,
    phase61_output_dir: Path | str | None,
    tile_index_csv: Path | str | None,
    phase10_json: Path | str,
    phase18_json: Path | str,
    phase66_json: Path | str,
    phase39_json: Path | str | None = None,
    phase40_json: Path | str | None = None,
    variants: Sequence[str] | str = ("B0", "D4P8", "D4P16", "D6R8", "D6R16"),
    label_columns: Sequence[str] | str = DEFAULT_PHASE67_LABEL_COLUMNS,
    top_k_values: Sequence[int] | str = DEFAULT_PHASE67_TOP_K_VALUES,
) -> dict[str, object]:
    variant_ids = _csvish(variants)
    labels = _csvish(label_columns)
    top_k = [int(value) for value in _csvish(top_k_values)]
    phase10 = _load_json_object(phase10_json, "Phase 10 JSON")
    phase18 = _load_json_object(phase18_json, "Phase 18 JSON")
    phase66 = _load_json_object(phase66_json, "Phase 66 JSON")
    phase39 = _load_json_object(phase39_json, "Phase 39 JSON")
    phase40 = _load_json_object(phase40_json, "Phase 40 JSON")
    source_dirs = {
        "B0": Path(phase2_output_dir),
        "B1": Path(phase2_output_dir),
        "D4P8": None if phase8_output_dir is None else Path(phase8_output_dir),
        "D4P16": None if phase8_output_dir is None else Path(phase8_output_dir),
        "D6R8": None if phase61_output_dir is None else Path(phase61_output_dir),
        "D6R16": None if phase61_output_dir is None else Path(phase61_output_dir),
    }
    feature_rows_by_variant: dict[str, list[dict[str, object]]] = {}
    coverage_issues: list[object] = []
    for variant_id in variant_ids:
        source_dir = source_dirs.get(str(variant_id))
        if source_dir is None:
            coverage_issues.append(f"missing source dir for {variant_id}")
            continue
        feature_rows_by_variant[str(variant_id)] = _variant_input_rows(source_dir, str(variant_id))
    if "B0" not in feature_rows_by_variant:
        raise ValueError("Phase 67 requires B0 feature rows")
    candidate_rows = _load_phase2_candidate_rows(phase2_output_dir, feature_rows_by_variant["B0"], labels)
    targets = build_phase67_candidate_targets(candidate_rows, label_columns=labels)
    inventory_rows = build_phase67_candidate_target_inventory(
        targets,
        expected_block_ids=[str(row["block_id"]) for row in candidate_rows],
    )
    gate_context = build_phase67_gate_context(phase10, phase18, phase39, phase40)
    gate_audit_rows = build_phase67_candidate_target_gate_audit(inventory_rows, gate_context)
    info_rows = build_phase67_candidate_target_information_gain(feature_rows_by_variant, targets, top_k_values=top_k)
    candidate_gate = build_phase67_candidate_target_gate(coverage_issues, info_rows, gate_audit_rows)
    summary_rows = [
        {
            "target_id": row["target_id"],
            "summary": f"{row['gate_risk']} usable={row['usable']}",
            "claim_boundary": PHASE67_CLAIM_BOUNDARY,
        }
        for row in gate_audit_rows
    ]
    return {
        "phase": "phase67_candidate_reward_label_target_audit",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "phase8_output_dir": "" if phase8_output_dir is None else str(Path(phase8_output_dir)),
        "phase61_output_dir": "" if phase61_output_dir is None else str(Path(phase61_output_dir)),
        "tile_index_csv": "" if tile_index_csv is None else str(Path(tile_index_csv)),
        "phase10_json": str(Path(phase10_json)),
        "phase18_json": str(Path(phase18_json)),
        "phase66_json": str(Path(phase66_json)),
        "phase39_json": "" if phase39_json is None else str(Path(phase39_json)),
        "phase40_json": "" if phase40_json is None else str(Path(phase40_json)),
        "phase66_status": phase66.get("phase66_status", phase66.get("diagnostic_gate", {}).get("phase66_status", "")),
        "variants": variant_ids,
        "label_columns": labels,
        "top_k_values": top_k,
        "gate_context": gate_context,
        "candidate_target_inventory_rows": inventory_rows,
        "candidate_target_gate_audit_rows": gate_audit_rows,
        "candidate_target_information_gain_rows": info_rows,
        "candidate_target_summary_rows": summary_rows,
        "candidate_target_gate": candidate_gate,
        "claim_boundary": PHASE67_CLAIM_BOUNDARY,
    }
```

- [ ] **Step 4: Add CLI runner**

Create `experiments/phase67_candidate_reward_label_target_audit/run_phase67_candidate_reward_label_target_audit.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase67_candidate_reward_label_target_audit import (
    run_phase67_candidate_reward_label_target_audit,
    write_phase67_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = run_phase67_candidate_reward_label_target_audit(
            phase2_output_dir=args.phase2_output_dir,
            phase8_output_dir=args.phase8_output_dir,
            phase61_output_dir=args.phase61_output_dir,
            tile_index_csv=args.tile_index_csv,
            phase10_json=args.phase10_json,
            phase18_json=args.phase18_json,
            phase66_json=args.phase66_json,
            phase39_json=args.phase39_json,
            phase40_json=args.phase40_json,
            variants=args.variants,
            label_columns=args.label_columns,
            top_k_values=args.top_k_values,
        )
        paths = write_phase67_artifacts(analysis, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    gate = analysis["candidate_target_gate"]
    print(f"Phase 67 status: {gate['phase67_status']}")
    print(f"Audit JSON: {paths['audit_json']}")
    print(f"Inventory CSV: {paths['inventory_csv']}")
    print(f"Gate Audit CSV: {paths['gate_audit_csv']}")
    print(f"Information Gain CSV: {paths['information_gain_csv']}")
    print(f"Audit Markdown: {paths['audit_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 67 candidate reward/label target audit."
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--phase8-output-dir", type=Path, default=None)
    parser.add_argument("--phase61-output-dir", type=Path, default=None)
    parser.add_argument("--tile-index-csv", type=Path, default=None)
    parser.add_argument("--phase10-json", type=Path, required=True)
    parser.add_argument("--phase18-json", type=Path, required=True)
    parser.add_argument("--phase66-json", type=Path, required=True)
    parser.add_argument("--phase39-json", type=Path, default=None)
    parser.add_argument("--phase40-json", type=Path, default=None)
    parser.add_argument("--variants", default="B0,D4P8,D4P16,D6R8,D6R16")
    parser.add_argument("--label-columns", default="current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label")
    parser.add_argument("--top-k-values", default="8,16,32")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run all Phase 67 tests**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase67_candidate_reward_label_target_audit.py -q --basetemp=.pytest_tmp_phase67_task4_green -p no:cacheprovider
```

Expected: all Phase 67 tests pass.

- [ ] **Step 6: Commit Task 4**

Run:

```powershell
git add src\paper11_geofm\phase67_candidate_reward_label_target_audit.py experiments\phase67_candidate_reward_label_target_audit\run_phase67_candidate_reward_label_target_audit.py tests\test_phase67_candidate_reward_label_target_audit.py
git commit -m "feat: add Phase 67 audit runner and artifacts"
```

Expected: commit succeeds.

---

### Task 5: Real Phase 67 Run And Result Note

**Files:**
- Create: `paper/phase28_results/33_phase67_candidate_reward_label_target_audit.md`
- Generated ignored outputs under: `experiments/phase67_candidate_reward_label_target_audit/outputs/phase52_full5_seed3/`

- [ ] **Step 1: Run the real Phase 67 audit**

Run from repository root:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase67_candidate_reward_label_target_audit\run_phase67_candidate_reward_label_target_audit.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --phase61-output-dir experiments\phase61_d6_geofm_projection_controls\outputs\phase52_full5_seed3 --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --phase10-json experiments\phase11_bishan_dltb_real\outputs\phase10_real\phase10_reward_readiness_gate.json --phase18-json experiments\phase18_planning_reward_readiness\outputs\real_bishan\phase18_planning_reward_readiness.json --phase66-json experiments\phase66_reward_label_representation_audit\outputs\phase52_full5_seed3\phase66_reward_label_representation_audit.json --variants B0,D4P8,D4P16,D6R8,D6R16 --label-columns current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label --top-k-values 8,16,32 --output-dir experiments\phase67_candidate_reward_label_target_audit\outputs\phase52_full5_seed3
```

Expected: exit code `0`, console prints `Phase 67 status:` plus artifact paths and claim boundary. Phase 39/40 real JSON paths are omitted because Phase 40 real output is not present in this workspace; missing optional independent-label evidence should not create a positive reward-ready claim.

- [ ] **Step 2: Inspect generated audit JSON**

Run:

```powershell
Get-Content -Raw experiments\phase67_candidate_reward_label_target_audit\outputs\phase52_full5_seed3\phase67_candidate_reward_label_target_audit.json
```

Expected: JSON contains `phase67_status`, `candidate_target_gate`, `candidate_target_inventory_rows`, `candidate_target_gate_audit_rows`, `candidate_target_information_gain_rows`, and `claim_boundary`.

- [ ] **Step 3: Create the tracked result note**

Run:

````powershell
Copy-Item -LiteralPath experiments\phase67_candidate_reward_label_target_audit\outputs\phase52_full5_seed3\phase67_candidate_reward_label_target_audit.md -Destination paper\phase28_results\33_phase67_candidate_reward_label_target_audit.md
Add-Content -LiteralPath paper\phase28_results\33_phase67_candidate_reward_label_target_audit.md -Value @'

## Reproduction

Run Phase 67 from the repository root after Phase 66 artifacts exist:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase67_candidate_reward_label_target_audit\run_phase67_candidate_reward_label_target_audit.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --phase61-output-dir experiments\phase61_d6_geofm_projection_controls\outputs\phase52_full5_seed3 --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --phase10-json experiments\phase11_bishan_dltb_real\outputs\phase10_real\phase10_reward_readiness_gate.json --phase18-json experiments\phase18_planning_reward_readiness\outputs\real_bishan\phase18_planning_reward_readiness.json --phase66-json experiments\phase66_reward_label_representation_audit\outputs\phase52_full5_seed3\phase66_reward_label_representation_audit.json --variants B0,D4P8,D4P16,D6R8,D6R16 --label-columns current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label --top-k-values 8,16,32 --output-dir experiments\phase67_candidate_reward_label_target_audit\outputs\phase52_full5_seed3
```

## Boundary

No formal manuscript files were changed in this phase.
'@
````

- [ ] **Step 4: Verify result note has no unresolved markers**

Run:

```powershell
rg -n "<[^>]+>|copy status|temporary marker|replace before run" paper\phase28_results\33_phase67_candidate_reward_label_target_audit.md
```

Expected: no output.

- [ ] **Step 5: Commit Task 5**

Run:

```powershell
git add paper\phase28_results\33_phase67_candidate_reward_label_target_audit.md
git commit -m "docs: record Phase 67 candidate target audit results"
```

Expected: commit succeeds. Generated `experiments/**/outputs/**` files remain ignored unless repository policy changes.

---

### Task 6: Regression Verification And Final Boundary Checks

**Files:**
- No new files unless a failing verification requires a targeted fix.

- [ ] **Step 1: Run targeted Phase 67/66/65/64/63 tests**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase67_candidate_reward_label_target_audit.py tests\test_phase66_reward_label_representation_audit.py tests\test_phase65_standardized_set_policy_bc_rerun.py tests\test_phase64_set_policy_error_diagnosis.py tests\test_phase63_set_policy_oracle_pretraining.py -q --basetemp=.pytest_tmp_phase67_final -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 2: Run smoke check**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe scripts\smoke_check.py
```

Expected: `Paper11 smoke check passed`.

- [ ] **Step 3: Check whitespace**

Run:

```powershell
git diff --check
```

Expected: no output.

- [ ] **Step 4: Confirm formal manuscript files are untouched**

Run:

```powershell
git diff --name-only HEAD -- paper\submission\final
```

Expected: no output.

- [ ] **Step 5: Confirm final git status**

Run:

```powershell
git status --short --branch
```

Expected: clean working tree on `main`. If local commits are ahead of `origin/main`, push after reviewing the commit list.

- [ ] **Step 6: Push completed Phase 67 work**

Run:

```powershell
git push
```

Expected: `main -> main` push succeeds.

---

## Self-Review Checklist

- Spec coverage:
  - Candidate target inventory is covered by Task 1.
  - Leakage and gate audit is covered by Task 2.
  - Explicit-versus-GeoFM information gain and candidate target gate are covered by Task 3.
  - Artifact writing, CLI, full orchestration, real run, and result note are covered by Tasks 4-5.
  - Regression and formal manuscript boundary checks are covered by Task 6.
- Claim boundary:
  - The plan does not train policies, modify `planning_reward.py`, enable suitability reward, create B2/B3 variants, or edit `paper/submission/final/*`.
- Verification boundary:
  - The final command set includes Phase 67 targeted tests, Phase 66/65/64/63 regressions, smoke check, `git diff --check`, formal manuscript diff check, and push.
