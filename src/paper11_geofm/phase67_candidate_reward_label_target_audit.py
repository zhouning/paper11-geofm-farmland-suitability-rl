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


def _linear_predictions(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(z.shape[0]), z])
    xtx = design.T @ design
    ridge = np.eye(xtx.shape[0], dtype=np.float64) * 1.0e-9
    ridge[0, 0] = 0.0
    xty = design.T @ y
    try:
        coeffs = np.linalg.solve(xtx + ridge, xty)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.pinv(xtx + ridge) @ xty
    return design @ coeffs


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
    return y - _linear_predictions(z, y)


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
    all_block_ids, explicit_matrix = _feature_matrix(rows, explicit_columns)
    source_targets_for_residuals = list(targets)
    for source_target in source_targets_for_residuals:
        values_by_block = dict(source_target.get("values_by_block", {}))
        aligned_indexes = []
        aligned_blocks = []
        y_values = []
        for row_index, block_id in enumerate(all_block_ids):
            value = _safe_float(values_by_block.get(block_id))
            if value is None:
                continue
            aligned_indexes.append(row_index)
            aligned_blocks.append(block_id)
            y_values.append(value)
        if len(y_values) < 2:
            continue
        aligned_explicit = explicit_matrix[np.asarray(aligned_indexes, dtype=np.int64), :]
        residual = _ols_residual_values(np.asarray(y_values, dtype=np.float64), aligned_explicit)
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


def phase67_topk_enrichment(
    feature_values: Sequence[float],
    target_values: Sequence[float],
    top_k: int,
) -> float:
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


def _univariate_proxy_r2(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] == 0:
        return 0.0
    if float(np.std(y)) == 0.0:
        return 0.0
    scores: list[float] = []
    for column_index in range(x.shape[1]):
        column = x[:, column_index]
        if float(np.std(column)) == 0.0:
            continue
        corr = float(np.corrcoef(column, y)[0, 1])
        if not np.isnan(corr):
            scores.append(max(0.0, min(1.0, corr * corr)))
    return _round_float(max(scores) if scores else 0.0)


def _proxy_r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] == 0:
        return 0.0
    keep = np.std(x, axis=0) > 1.0e-12
    if not bool(np.any(keep)) or float(np.std(y)) == 0.0:
        return 0.0
    if x.shape[0] <= int(np.sum(keep)) + 1:
        return _univariate_proxy_r2(x[:, keep], y)
    z = x[:, keep]
    z = (z - np.mean(z, axis=0)) / np.std(z, axis=0)
    predicted = _linear_predictions(z, y)
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 1.0e-12:
        return 0.0
    residual = float(np.sum((y - predicted) ** 2))
    return _round_float(max(0.0, min(1.0, 1.0 - residual / total)))


def _feature_group_indexes(fieldnames: Sequence[str]) -> dict[str, list[str]]:
    reward_explicit = [column for column in fieldnames if column in BASE_PLANNING_REWARD_REQUIRED_COLUMNS]
    all_explicit = [column for column in fieldnames if str(column).startswith("explicit_feature_")]
    geofm = [
        column
        for column in fieldnames
        if str(column).startswith(("embedding_pca_", "embedding_mean_", "projection_"))
    ]
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
        fieldnames = list(feature_rows[0].keys())
        groups = _feature_group_indexes(fieldnames)
        block_ids = [str(row.get("block_id", "")) for row in feature_rows]
        reward_matrix = _matrix_from_rows(feature_rows, groups["reward_explicit"])
        explicit_matrix = _matrix_from_rows(feature_rows, groups["all_explicit"])
        geofm_matrix = _matrix_from_rows(feature_rows, groups["geofm"])
        combined_matrix = (
            np.column_stack([explicit_matrix, geofm_matrix])
            if geofm_matrix.shape[1]
            else explicit_matrix
        )
        for target in targets:
            values_by_block = dict(target.get("values_by_block", {}))
            aligned_indexes = []
            values = []
            for row_index, block_id in enumerate(block_ids):
                value = _safe_float(values_by_block.get(block_id))
                if value is None:
                    continue
                aligned_indexes.append(row_index)
                values.append(value)
            if not values:
                raise ValueError(f"Phase 67 target has no aligned values: {target.get('target_id')}")
            indexes = np.asarray(aligned_indexes, dtype=np.int64)
            y = np.asarray(values, dtype=np.float64)
            reward_x = reward_matrix[indexes, :]
            explicit_x = explicit_matrix[indexes, :]
            geofm_x = geofm_matrix[indexes, :]
            combined_x = combined_matrix[indexes, :]
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
        row
        for row in information_gain_rows
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
            if column_name in label_columns or column_name.startswith(
                ("embedding_mean_", "embedding_pca_", "projection_")
            ):
                merged[column_name] = value
        rows.append(merged)
    return rows


def _write_csv_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
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


def write_phase67_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
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
    _write_csv_rows(
        paths["inventory_csv"],
        PHASE67_INVENTORY_FIELDNAMES,
        analysis.get("candidate_target_inventory_rows", []),
    )
    _write_csv_rows(
        paths["gate_audit_csv"],
        PHASE67_GATE_AUDIT_FIELDNAMES,
        analysis.get("candidate_target_gate_audit_rows", []),
    )
    _write_csv_rows(
        paths["information_gain_csv"],
        PHASE67_INFORMATION_GAIN_FIELDNAMES,
        analysis.get("candidate_target_information_gain_rows", []),
    )
    _write_csv_rows(
        paths["summary_csv"],
        PHASE67_SUMMARY_FIELDNAMES,
        analysis.get("candidate_target_summary_rows", []),
    )
    saved = dict(analysis)
    saved["phase67_status"] = dict(analysis.get("candidate_target_gate", {})).get(
        "phase67_status",
        PHASE67_STATUS_INSUFFICIENT,
    )
    paths["audit_json"].write_text(
        json.dumps(_json_ready(saved), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["audit_md"].write_text(_phase67_markdown(analysis), encoding="utf-8")
    return paths


def _csvish(value: Sequence[object] | str) -> list[str]:
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
    candidate_rows = _load_phase2_candidate_rows(
        phase2_output_dir,
        feature_rows_by_variant["B0"],
        labels,
    )
    targets = build_phase67_candidate_targets(candidate_rows, label_columns=labels)
    inventory_rows = build_phase67_candidate_target_inventory(
        targets,
        expected_block_ids=[str(row["block_id"]) for row in candidate_rows],
    )
    gate_context = build_phase67_gate_context(phase10, phase18, phase39, phase40)
    gate_audit_rows = build_phase67_candidate_target_gate_audit(inventory_rows, gate_context)
    info_rows = build_phase67_candidate_target_information_gain(
        feature_rows_by_variant,
        targets,
        top_k_values=top_k,
    )
    candidate_gate = build_phase67_candidate_target_gate(
        coverage_issues,
        info_rows,
        gate_audit_rows,
    )
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
        "phase66_status": phase66.get(
            "phase66_status",
            phase66.get("diagnostic_gate", {}).get("phase66_status", "")
            if isinstance(phase66.get("diagnostic_gate", {}), Mapping)
            else "",
        ),
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
