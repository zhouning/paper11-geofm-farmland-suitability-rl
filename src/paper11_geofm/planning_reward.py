from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


BASE_PLANNING_REWARD_REQUIRED_COLUMNS = (
    "explicit_feature_00",
    "explicit_feature_01",
    "explicit_feature_02",
    "explicit_feature_04",
    "explicit_feature_07",
    "explicit_feature_09",
    "explicit_feature_10",
    "explicit_feature_13",
    "explicit_feature_16",
)

BASE_PLANNING_REWARD_IMPLEMENTED = True
BASE_PLANNING_REWARD_EVIDENCE = (
    "base_planning_reward is implemented as a bounded weighted score over "
    "explicit planning features exported by Phase 11."
)
BASE_PLANNING_REWARD_CLAIM_BOUNDARY = (
    "The base_planning_reward is a first deterministic planning-score "
    "implementation; it is not calibrated policy performance evidence."
)


def compute_base_planning_reward(row: Mapping[str, Any]) -> float:
    """Compute the Phase 19 base planning reward from explicit features."""
    missing = [
        column
        for column in BASE_PLANNING_REWARD_REQUIRED_COLUMNS
        if column not in row or str(row[column]).strip() == ""
    ]
    if missing:
        raise ValueError(
            "base_planning_reward requires explicit feature columns: "
            f"{', '.join(missing)}"
        )

    low_slope_farmland_or_orchard = _clip01(_as_float(row, "explicit_feature_16"))
    current_farmland_or_orchard = max(
        _clip01(_as_float(row, "explicit_feature_04")),
        _clip01(_as_float(row, "explicit_feature_07")),
    )
    low_slope = _clip01(_as_float(row, "explicit_feature_13"))
    area_score = _clip01(_as_float(row, "explicit_feature_00") / 5.0)
    mean_slope_score = _clip01(_as_float(row, "explicit_feature_01") / 25.0)
    max_slope_score = _clip01(_as_float(row, "explicit_feature_02") / 35.0)
    built_up = _clip01(_as_float(row, "explicit_feature_09"))
    water = _clip01(_as_float(row, "explicit_feature_10"))

    reward = (
        0.35 * low_slope_farmland_or_orchard
        + 0.20 * current_farmland_or_orchard
        + 0.10 * low_slope
        + 0.10 * area_score
        - 0.15 * mean_slope_score
        - 0.05 * max_slope_score
        - 0.10 * built_up
        - 0.10 * water
    )
    return round(float(reward), 10)


def compute_base_planning_reward_from_matrix_row(
    feature_columns: Sequence[str],
    values: Sequence[float],
) -> float:
    """Compute base reward from a loaded Phase 3/4 state-matrix row."""
    column_to_index = {str(column): index for index, column in enumerate(feature_columns)}
    missing = [
        column
        for column in BASE_PLANNING_REWARD_REQUIRED_COLUMNS
        if column not in column_to_index
    ]
    if missing:
        raise ValueError(
            "base_planning_reward requires explicit feature columns: "
            f"{', '.join(missing)}"
        )

    row = {
        column: values[column_to_index[column]]
        for column in BASE_PLANNING_REWARD_REQUIRED_COLUMNS
    }
    return compute_base_planning_reward(row)


def has_base_planning_reward_columns(feature_columns: Sequence[str]) -> bool:
    available = {str(column) for column in feature_columns}
    return all(column in available for column in BASE_PLANNING_REWARD_REQUIRED_COLUMNS)


def _as_float(row: Mapping[str, Any], column: str) -> float:
    value = row[column]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"base_planning_reward requires numeric {column}: {value!r}"
        ) from exc


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
