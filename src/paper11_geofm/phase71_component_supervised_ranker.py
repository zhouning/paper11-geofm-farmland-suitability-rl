from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np

from paper11_geofm.phase63_set_policy_oracle_pretraining import (
    PHASE63_DEFAULT_VARIANTS,
    _phase63_oracle_summary_row,
    _round_float,
    build_phase63_oracle_trajectory,
    build_phase63_set_policy_contract,
)
from paper11_geofm.planning_reward import compute_base_planning_reward_from_matrix_row
from paper11_geofm.tiled_inputs import TiledVariantInput, load_tiled_variant_input


PHASE71_CLAIM_BOUNDARY = (
    "Phase 71 is a component-supervised listwise-ranker experiment under the "
    "existing Bishan base-reward protocol. It trains ranking models from "
    "deterministic base-reward totals and reward-component contributions while "
    "preserving original features for scoring. It does not alter rewards, "
    "enable B2/B3, validate suitability, prove GeoFM superiority, prove PCA "
    "optimality, test transfer, or justify formal submission-level claims."
)

PHASE71_STATUS_DECISION = "ranker_improves_decision_route"
PHASE71_STATUS_TARGET_MASKS_GEOFM = "ranker_improves_but_target_masks_geofm"
PHASE71_STATUS_GEOFM = "ranker_supports_geofm_followup"
PHASE71_STATUS_NOT_SUFFICIENT = "ranker_not_sufficient"
PHASE71_STATUS_INCOMPLETE = "ranker_incomplete"

PHASE71_COMPONENT_NAMES = (
    "low_slope_farmland_or_orchard",
    "current_farmland_or_orchard",
    "low_slope",
    "area_score",
    "mean_slope_penalty",
    "max_slope_penalty",
    "built_up_penalty",
    "water_penalty",
)


@dataclass(frozen=True)
class Phase71PreparedTile:
    tiled_input: TiledVariantInput
    model_matrix: np.ndarray
    reward_matrix: np.ndarray
    standardization: Mapping[str, object]


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _column_value(
    feature_columns: Sequence[str],
    values: Sequence[float],
    column: str,
) -> float:
    index = {str(name): offset for offset, name in enumerate(feature_columns)}
    if column not in index:
        raise ValueError(f"Phase 71 requires reward column {column}")
    return float(values[index[column]])


def decompose_phase71_reward_components(
    feature_columns: Sequence[str],
    values: Sequence[float],
) -> dict[str, float]:
    current_farmland_or_orchard = max(
        _clip01(_column_value(feature_columns, values, "explicit_feature_04")),
        _clip01(_column_value(feature_columns, values, "explicit_feature_07")),
    )
    components = {
        "low_slope_farmland_or_orchard": 0.35
        * _clip01(_column_value(feature_columns, values, "explicit_feature_16")),
        "current_farmland_or_orchard": 0.20 * current_farmland_or_orchard,
        "low_slope": 0.10
        * _clip01(_column_value(feature_columns, values, "explicit_feature_13")),
        "area_score": 0.10
        * _clip01(_column_value(feature_columns, values, "explicit_feature_00") / 5.0),
        "mean_slope_penalty": -0.15
        * _clip01(_column_value(feature_columns, values, "explicit_feature_01") / 25.0),
        "max_slope_penalty": -0.05
        * _clip01(_column_value(feature_columns, values, "explicit_feature_02") / 35.0),
        "built_up_penalty": -0.10
        * _clip01(_column_value(feature_columns, values, "explicit_feature_09")),
        "water_penalty": -0.10
        * _clip01(_column_value(feature_columns, values, "explicit_feature_10")),
    }
    return {key: _round_float(value) for key, value in components.items()}


def build_phase71_component_targets(
    tiled_input: TiledVariantInput,
) -> list[dict[str, object]]:
    rows = []
    for index, block_id in enumerate(tiled_input.block_ids):
        values = tiled_input.state_matrix[index]
        components = decompose_phase71_reward_components(
            tiled_input.feature_columns,
            values,
        )
        reward_total = compute_base_planning_reward_from_matrix_row(
            tiled_input.feature_columns,
            values,
        )
        components["water_penalty"] = float(components["water_penalty"]) + (
            float(reward_total) - round(sum(components.values()), 10)
        )
        rows.append(
            {
                "variant_id": str(tiled_input.variant_id),
                "tile_id": str(tiled_input.tile_id),
                "block_id": str(block_id),
                "action_index": int(index),
                "reward_total": float(reward_total),
                "components": components,
                "component_sum": _round_float(sum(components.values())),
                "claim_boundary": PHASE71_CLAIM_BOUNDARY,
            }
        )
    return rows


def fit_phase71_fold_standardization(
    training_tiles: Sequence[TiledVariantInput],
    variant_id: str,
    fold_id: str,
) -> dict[str, object]:
    if not training_tiles:
        raise ValueError(
            "Phase 71 fold standardization requires at least one training tile"
        )
    feature_columns = tuple(training_tiles[0].feature_columns)
    matrices = []
    tile_ids = []
    for tile in training_tiles:
        if tuple(tile.feature_columns) != feature_columns:
            raise ValueError(
                "Phase 71 fold standardization feature columns do not match"
            )
        matrices.append(np.asarray(tile.state_matrix, dtype=np.float32))
        tile_ids.append(str(tile.tile_id))
    matrix = np.vstack(matrices)
    means = np.nanmean(matrix, axis=0)
    scales = np.nanstd(matrix, axis=0)
    safe_scales = np.where(
        np.isfinite(scales) & (np.abs(scales) >= 1.0e-8),
        scales,
        1.0,
    )
    return {
        "variant_id": str(variant_id),
        "fold_id": str(fold_id),
        "training_tile_ids": tile_ids,
        "feature_columns": list(feature_columns),
        "means": [round(float(value), 10) for value in means.tolist()],
        "scales": [round(float(value), 10) for value in safe_scales.tolist()],
        "claim_boundary": PHASE71_CLAIM_BOUNDARY,
    }


def apply_phase71_fold_standardization(
    tiled_input: TiledVariantInput,
    params: Mapping[str, object],
) -> Phase71PreparedTile:
    feature_columns = tuple(str(value) for value in params.get("feature_columns", []))
    if feature_columns != tuple(tiled_input.feature_columns):
        raise ValueError("Phase 71 standardization feature columns do not match input")
    matrix = np.asarray(tiled_input.state_matrix, dtype=np.float32)
    means = np.asarray(params.get("means", []), dtype=np.float32)
    scales = np.asarray(params.get("scales", []), dtype=np.float32)
    if means.shape[0] != matrix.shape[1] or scales.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Phase 71 standardization parameter length does not match input"
        )
    safe_scales = np.where(
        np.isfinite(scales) & (np.abs(scales) >= 1.0e-8),
        scales,
        1.0,
    )
    model_matrix = (matrix - means) / safe_scales
    model_matrix = np.nan_to_num(
        model_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)
    return Phase71PreparedTile(
        tiled_input=tiled_input,
        model_matrix=model_matrix,
        reward_matrix=matrix.astype(np.float32, copy=True),
        standardization=dict(params),
    )
