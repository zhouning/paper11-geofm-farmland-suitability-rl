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
DEFAULT_PHASE67_REPRESENTATION_PREFIXES = (
    "embedding_pca_",
    "embedding_mean_",
    "projection_",
)
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
    missing_reward = [
        column for column in BASE_PLANNING_REWARD_REQUIRED_COLUMNS if column not in fieldnames
    ]
    if missing_reward:
        raise ValueError(f"Phase 67 rows are missing base reward columns: {missing_reward}")
    block_ids = [str(row["block_id"]) for row in rows]
    base_values = {str(row["block_id"]): compute_base_planning_reward(row) for row in rows}
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
        column
        for column in sorted(fieldnames)
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
    explicit_columns = [
        column for column in sorted(fieldnames) if str(column).startswith("explicit_feature_")
    ]
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
        "phase10_recommendation": str(
            phase10.get("phase10_recommendation", phase10.get("recommendation", ""))
        ),
        "phase18_suitability_reward_allowed": bool(
            phase18.get("suitability_reward_allowed", False)
        ),
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
    for inventory_row in inventory_rows:
        gate_risk, reason = _target_gate_risk(inventory_row)
        usable = bool(inventory_row.get("usable", False))
        diagnostic_only_allowed = usable and gate_risk in {
            "explicit_label_leakage_risk",
            "geofm_self_reference",
            "diagnostic_only_allowed",
        }
        rows.append(
            {
                "target_id": str(inventory_row.get("target_id", "")),
                "target_family": str(inventory_row.get("target_family", "")),
                "usable": usable,
                "gate_risk": gate_risk,
                "diagnostic_only_allowed": diagnostic_only_allowed,
                "reward_training_allowed": False,
                "independent_label_required": gate_risk
                in {"independent_label_missing", "diagnostic_only_allowed"},
                "phase10_status": str(gate_context.get("phase10_status", "")),
                "phase10_recommendation": str(gate_context.get("phase10_recommendation", "")),
                "phase18_suitability_reward_allowed": bool(
                    gate_context.get("phase18_suitability_reward_allowed", False)
                ),
                "phase39_status": str(gate_context.get("phase39_status", "missing")),
                "phase40_status": str(gate_context.get("phase40_status", "missing")),
                "reason": reason,
                "claim_boundary": PHASE67_CLAIM_BOUNDARY,
            }
        )
    return rows
