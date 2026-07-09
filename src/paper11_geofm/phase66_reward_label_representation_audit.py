from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
import statistics

import numpy as np

from .planning_reward import (
    BASE_PLANNING_REWARD_REQUIRED_COLUMNS,
    compute_base_planning_reward_from_matrix_row,
)
from .tiled_inputs import load_tiled_variant_input


PHASE66_CLAIM_BOUNDARY = (
    "Phase 66 is a read-only reward-label and representation attribution audit "
    "over existing Phase 63, Phase 64, and Phase 65 artifacts plus raw tiled "
    "feature matrices. It does not train or fine-tune a policy, does not change "
    "the base reward, does not enable suitability reward, does not test B2/B3 "
    "or transfer, does not prove GeoFM advantage or PCA optimality, and does "
    "not justify formal submission-level claims."
)

PHASE66_STATUS_REPRESENTATION_ADDS_SIGNAL = "representation_adds_reward_ranking_signal"
PHASE66_STATUS_REPRESENTATION_REDUNDANT = "representation_signal_redundant_with_explicit_reward"
PHASE66_STATUS_BASE_REWARD_MASKS = "base_reward_target_masks_geofm_signal"
PHASE66_STATUS_INSUFFICIENT = "insufficient"

PHASE66_REWARD_EQUIVALENT_TOLERANCE = 0.02
PHASE66_ALIGNMENT_ADVANTAGE_THRESHOLD = 0.05

PHASE66_COMPONENT_FIELDNAMES = [
    "variant_id",
    "tile_id",
    "block_id",
    "reward_rank",
    "source",
    "seed",
    "action_group",
    "total_reward",
    "low_slope_farmland_or_orchard_component",
    "current_farmland_or_orchard_component",
    "low_slope_component",
    "area_component",
    "mean_slope_penalty_component",
    "max_slope_penalty_component",
    "built_up_penalty_component",
    "water_penalty_component",
    "claim_boundary",
]


def _round_float(value: object, digits: int = 10) -> float:
    rounded = round(float(value), digits)
    compact = round(rounded, 6)
    if abs(rounded - compact) < 5.0e-8:
        return compact
    return rounded


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _column_value(
    column_to_index: Mapping[str, int],
    values: Sequence[float],
    column: str,
) -> float:
    return float(values[int(column_to_index[column])])


def decompose_phase66_base_reward_components(
    feature_columns: Sequence[str],
    values: Sequence[float],
) -> dict[str, float]:
    column_to_index = {str(column): index for index, column in enumerate(feature_columns)}
    missing = [
        column
        for column in BASE_PLANNING_REWARD_REQUIRED_COLUMNS
        if column not in column_to_index
    ]
    if missing:
        raise ValueError(
            "Phase 66 reward decomposition requires explicit feature columns: "
            f"{', '.join(missing)}"
        )
    low_slope_farmland_or_orchard = _clip01(
        _column_value(column_to_index, values, "explicit_feature_16")
    )
    current_farmland_or_orchard = max(
        _clip01(_column_value(column_to_index, values, "explicit_feature_04")),
        _clip01(_column_value(column_to_index, values, "explicit_feature_07")),
    )
    low_slope = _clip01(_column_value(column_to_index, values, "explicit_feature_13"))
    area_score = _clip01(
        _column_value(column_to_index, values, "explicit_feature_00") / 5.0
    )
    mean_slope_score = _clip01(
        _column_value(column_to_index, values, "explicit_feature_01") / 25.0
    )
    max_slope_score = _clip01(
        _column_value(column_to_index, values, "explicit_feature_02") / 35.0
    )
    built_up = _clip01(_column_value(column_to_index, values, "explicit_feature_09"))
    water = _clip01(_column_value(column_to_index, values, "explicit_feature_10"))
    components = {
        "low_slope_farmland_or_orchard_component": 0.35
        * low_slope_farmland_or_orchard,
        "current_farmland_or_orchard_component": 0.20
        * current_farmland_or_orchard,
        "low_slope_component": 0.10 * low_slope,
        "area_component": 0.10 * area_score,
        "mean_slope_penalty_component": -0.15 * mean_slope_score,
        "max_slope_penalty_component": -0.05 * max_slope_score,
        "built_up_penalty_component": -0.10 * built_up,
        "water_penalty_component": -0.10 * water,
    }
    rounded_components = {key: _round_float(value) for key, value in components.items()}
    rounded_components["total_reward"] = compute_base_planning_reward_from_matrix_row(
        feature_columns,
        values,
    )
    return rounded_components


def build_phase66_block_reward_table(tiled_input) -> list[dict[str, object]]:
    block_rows: list[dict[str, object]] = []
    for row_index, block_id in enumerate(tiled_input.block_ids):
        components = decompose_phase66_base_reward_components(
            tiled_input.feature_columns,
            tiled_input.state_matrix[row_index],
        )
        block_rows.append(
            {
                "variant_id": str(tiled_input.variant_id),
                "tile_id": str(tiled_input.tile_id),
                "block_id": str(block_id),
                **components,
            }
        )
    ranked_ids = {
        str(row["block_id"]): rank
        for rank, row in enumerate(
            sorted(
                block_rows,
                key=lambda item: (-float(item["total_reward"]), str(item["block_id"])),
            ),
            start=1,
        )
    }
    for row in block_rows:
        row["reward_rank"] = int(ranked_ids[str(row["block_id"])])
        row["claim_boundary"] = PHASE66_CLAIM_BOUNDARY
    return block_rows
