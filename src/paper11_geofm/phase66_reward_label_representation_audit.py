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


PHASE66_ATLAS_FIELDNAMES = [
    "variant_id",
    "eval_tile_id",
    "seed",
    "oracle_block_ids",
    "phase63_selected_block_ids",
    "phase65_selected_block_ids",
    "phase63_oracle_overlap_count",
    "phase65_oracle_overlap_count",
    "phase63_oracle_jaccard",
    "phase65_oracle_jaccard",
    "phase63_phase65_jaccard",
    "phase63_selected_rank_values",
    "phase65_selected_rank_values",
    "phase63_missed_oracle_block_ids",
    "phase63_extra_selected_block_ids",
    "phase65_missed_oracle_block_ids",
    "phase65_extra_selected_block_ids",
    "phase63_reward_equivalent_substitution",
    "phase65_reward_equivalent_substitution",
    "claim_boundary",
]


def _split_semicolon_values(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _safe_int(value: object, default: int = 0) -> int:
    if value is None or str(value).strip() == "":
        return int(default)
    return int(float(value))


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def _row_key(row: Mapping[str, object]) -> tuple[str, str, int]:
    return (
        str(row.get("variant_id", "")),
        str(row.get("eval_tile_id", row.get("tile_id", ""))),
        _safe_int(row.get("seed")),
    )


def _index_unique_rows(
    rows: Sequence[Mapping[str, object]],
    label: str,
) -> dict[tuple[str, str, int], Mapping[str, object]]:
    index: dict[tuple[str, str, int], Mapping[str, object]] = {}
    for row in rows:
        key = _row_key(row)
        if key in index:
            raise ValueError(f"Phase 66 found duplicate {label} row for {key}")
        index[key] = row
    return index


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return _round_float(len(left_set & right_set) / len(union))


def _block_rank_and_reward(tiled_input) -> dict[str, dict[str, object]]:
    return {str(row["block_id"]): row for row in build_phase66_block_reward_table(tiled_input)}


def _rank_values(
    block_ids: Sequence[str],
    reward_index: Mapping[str, Mapping[str, object]],
) -> str:
    return ";".join(
        str(int(reward_index[str(block_id)]["reward_rank"])) for block_id in block_ids
    )


def _reward_equivalent_substitution(
    missed_block_ids: Sequence[str],
    extra_block_ids: Sequence[str],
    reward_index: Mapping[str, Mapping[str, object]],
    tolerance: float,
) -> bool:
    if not missed_block_ids and not extra_block_ids:
        return True
    if not missed_block_ids or not extra_block_ids:
        return False
    missed_rewards = [
        float(reward_index[str(block_id)]["total_reward"]) for block_id in missed_block_ids
    ]
    extra_rewards = [
        float(reward_index[str(block_id)]["total_reward"]) for block_id in extra_block_ids
    ]
    return bool(
        abs(statistics.mean(missed_rewards) - statistics.mean(extra_rewards))
        <= float(tolerance)
    )


def _missing_block_ids(
    block_ids: Sequence[str],
    reward_index: Mapping[str, Mapping[str, object]],
) -> list[str]:
    return [str(block_id) for block_id in block_ids if str(block_id) not in reward_index]


def build_phase66_selected_block_atlas(
    phase63_rollout_rows: Sequence[Mapping[str, object]],
    phase65_rollout_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
    tiled_inputs: Mapping[tuple[str, str], object],
    reward_tolerance: float = PHASE66_REWARD_EQUIVALENT_TOLERANCE,
) -> list[dict[str, object]]:
    phase63_index = _index_unique_rows(
        [
            row
            for row in phase63_rollout_rows
            if str(row.get("row_type", "")) == "bc_greedy_policy"
        ],
        "Phase 63 rollout",
    )
    phase65_index = _index_unique_rows(
        [
            row
            for row in phase65_rollout_rows
            if str(row.get("row_type", "")) == "bc_greedy_policy"
        ],
        "Phase 65 rollout",
    )
    oracle_index = _index_unique_rows(oracle_rows, "Phase 63 oracle")
    rows: list[dict[str, object]] = []
    for key in sorted(oracle_index):
        variant_id, tile_id, seed = key
        if key not in phase63_index:
            raise ValueError(f"Phase 66 missing Phase 63 rollout row for {key}")
        if key not in phase65_index:
            raise ValueError(f"Phase 66 missing Phase 65 rollout row for {key}")
        tiled = tiled_inputs.get((variant_id, tile_id))
        if tiled is None:
            raise ValueError(f"Phase 66 missing tiled input for {(variant_id, tile_id)}")
        oracle_ids = _split_semicolon_values(oracle_index[key].get("selected_block_ids"))
        phase63_ids = _split_semicolon_values(
            phase63_index[key].get("selected_block_ids")
        )
        phase65_ids = _split_semicolon_values(
            phase65_index[key].get("selected_block_ids")
        )
        reward_index = _block_rank_and_reward(tiled)
        missing = (
            _missing_block_ids(oracle_ids, reward_index)
            + _missing_block_ids(phase63_ids, reward_index)
            + _missing_block_ids(phase65_ids, reward_index)
        )
        if missing:
            raise ValueError(
                f"Phase 66 selected block IDs missing from tiled input: {missing[:5]}"
            )
        phase63_missed = [
            block_id for block_id in oracle_ids if block_id not in set(phase63_ids)
        ]
        phase63_extra = [
            block_id for block_id in phase63_ids if block_id not in set(oracle_ids)
        ]
        phase65_missed = [
            block_id for block_id in oracle_ids if block_id not in set(phase65_ids)
        ]
        phase65_extra = [
            block_id for block_id in phase65_ids if block_id not in set(oracle_ids)
        ]
        rows.append(
            {
                "variant_id": variant_id,
                "eval_tile_id": tile_id,
                "seed": int(seed),
                "oracle_block_ids": ";".join(oracle_ids),
                "phase63_selected_block_ids": ";".join(phase63_ids),
                "phase65_selected_block_ids": ";".join(phase65_ids),
                "phase63_oracle_overlap_count": len(set(phase63_ids) & set(oracle_ids)),
                "phase65_oracle_overlap_count": len(set(phase65_ids) & set(oracle_ids)),
                "phase63_oracle_jaccard": _jaccard(phase63_ids, oracle_ids),
                "phase65_oracle_jaccard": _jaccard(phase65_ids, oracle_ids),
                "phase63_phase65_jaccard": _jaccard(phase63_ids, phase65_ids),
                "phase63_selected_rank_values": _rank_values(phase63_ids, reward_index),
                "phase65_selected_rank_values": _rank_values(phase65_ids, reward_index),
                "phase63_missed_oracle_block_ids": ";".join(phase63_missed),
                "phase63_extra_selected_block_ids": ";".join(phase63_extra),
                "phase65_missed_oracle_block_ids": ";".join(phase65_missed),
                "phase65_extra_selected_block_ids": ";".join(phase65_extra),
                "phase63_reward_equivalent_substitution": _reward_equivalent_substitution(
                    phase63_missed,
                    phase63_extra,
                    reward_index,
                    reward_tolerance,
                ),
                "phase65_reward_equivalent_substitution": _reward_equivalent_substitution(
                    phase65_missed,
                    phase65_extra,
                    reward_index,
                    reward_tolerance,
                ),
                "claim_boundary": PHASE66_CLAIM_BOUNDARY,
            }
        )
    return rows


PHASE66_ALIGNMENT_FIELDNAMES = [
    "variant_id",
    "tile_id",
    "feature_group",
    "n_columns",
    "mean_abs_spearman",
    "max_abs_spearman",
    "best_topk_enrichment",
    "proxy_r2",
    "best_feature_name",
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
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def phase66_spearman_abs(
    feature_values: Sequence[float],
    reward_values: Sequence[float],
) -> float:
    x = np.asarray(feature_values, dtype=np.float64)
    y = np.asarray(reward_values, dtype=np.float64)
    if x.size != y.size:
        raise ValueError("Phase 66 Spearman inputs must have equal length")
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    rx = _rank_average(x)
    ry = _rank_average(y)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return 0.0
    corr = float(np.corrcoef(rx, ry)[0, 1])
    if np.isnan(corr):
        return 0.0
    return _round_float(abs(corr))


def phase66_topk_enrichment(
    feature_values: Sequence[float],
    reward_values: Sequence[float],
    top_k: int,
) -> float:
    x = np.asarray(feature_values, dtype=np.float64)
    y = np.asarray(reward_values, dtype=np.float64)
    if x.size != y.size:
        raise ValueError("Phase 66 top-k enrichment inputs must have equal length")
    k = min(int(top_k), int(x.size))
    if k <= 0:
        return 0.0
    reward_top = set(np.argsort(-y, kind="mergesort")[:k].tolist())
    high_top = set(np.argsort(-x, kind="mergesort")[:k].tolist())
    low_top = set(np.argsort(x, kind="mergesort")[:k].tolist())
    return _round_float(max(len(reward_top & high_top), len(reward_top & low_top)) / k)


def _proxy_r2(matrix: np.ndarray, reward_values: np.ndarray) -> float:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(reward_values, dtype=np.float64)
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


def _phase66_feature_groups(feature_columns: Sequence[str]) -> dict[str, list[int]]:
    reward_required = set(BASE_PLANNING_REWARD_REQUIRED_COLUMNS)
    reward_explicit = [
        index for index, column in enumerate(feature_columns) if str(column) in reward_required
    ]
    nonreward_explicit = [
        index
        for index, column in enumerate(feature_columns)
        if str(column).startswith("explicit_feature_") and str(column) not in reward_required
    ]
    representation_extra = [
        index
        for index, column in enumerate(feature_columns)
        if not str(column).startswith("explicit_feature_")
    ]
    return {
        "reward_explicit": reward_explicit,
        "nonreward_explicit": nonreward_explicit,
        "representation_extra": representation_extra,
    }


def _alignment_row(
    tiled_input,
    group_name: str,
    indexes: Sequence[int],
    reward_values: np.ndarray,
    eval_max_steps: int,
) -> dict[str, object]:
    if not indexes:
        return {
            "variant_id": str(tiled_input.variant_id),
            "tile_id": str(tiled_input.tile_id),
            "feature_group": group_name,
            "n_columns": 0,
            "mean_abs_spearman": 0.0,
            "max_abs_spearman": 0.0,
            "best_topk_enrichment": 0.0,
            "proxy_r2": 0.0,
            "best_feature_name": "",
            "claim_boundary": PHASE66_CLAIM_BOUNDARY,
        }
    matrix = np.asarray(tiled_input.state_matrix[:, list(indexes)], dtype=np.float64)
    spearman_values = [
        phase66_spearman_abs(matrix[:, col], reward_values)
        for col in range(matrix.shape[1])
    ]
    enrichment_values = [
        phase66_topk_enrichment(matrix[:, col], reward_values, top_k=eval_max_steps)
        for col in range(matrix.shape[1])
    ]
    best_index = int(np.argmax(spearman_values)) if spearman_values else 0
    return {
        "variant_id": str(tiled_input.variant_id),
        "tile_id": str(tiled_input.tile_id),
        "feature_group": group_name,
        "n_columns": int(len(indexes)),
        "mean_abs_spearman": _round_float(statistics.mean(spearman_values)),
        "max_abs_spearman": _round_float(max(spearman_values)),
        "best_topk_enrichment": _round_float(max(enrichment_values)),
        "proxy_r2": _proxy_r2(matrix, reward_values),
        "best_feature_name": str(tiled_input.feature_columns[int(indexes[best_index])]),
        "claim_boundary": PHASE66_CLAIM_BOUNDARY,
    }


def build_phase66_representation_rank_alignment(
    tiled_inputs: Mapping[tuple[str, str], object],
    eval_max_steps: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(tiled_inputs):
        tiled = tiled_inputs[key]
        reward_values = np.asarray(
            [
                compute_base_planning_reward_from_matrix_row(
                    tiled.feature_columns,
                    tiled.state_matrix[row_index],
                )
                for row_index in range(len(tiled.block_ids))
            ],
            dtype=np.float64,
        )
        groups = _phase66_feature_groups(tiled.feature_columns)
        if str(tiled.variant_id) != "B0" and not groups["representation_extra"]:
            raise ValueError(
                f"Phase 66 cannot separate representation columns for {tiled.variant_id}"
            )
        for group_name, indexes in groups.items():
            rows.append(
                _alignment_row(
                    tiled,
                    group_name,
                    indexes,
                    reward_values,
                    eval_max_steps=eval_max_steps,
                )
            )
    return rows


PHASE66_FAILURE_FIELDNAMES = [
    "failure_mode",
    "case_count",
    "representative_cases",
    "claim_boundary",
]


def _alignment_advantage_summary(
    alignment_rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    b0_explicit = [
        row
        for row in alignment_rows
        if str(row.get("variant_id")) == "B0"
        and str(row.get("feature_group")) == "reward_explicit"
    ]
    geofm_rep = [
        row
        for row in alignment_rows
        if str(row.get("variant_id", "")).startswith(("D4", "D6"))
        and str(row.get("feature_group")) == "representation_extra"
    ]
    geofm_explicit = [
        row
        for row in alignment_rows
        if str(row.get("variant_id", "")).startswith(("D4", "D6"))
        and str(row.get("feature_group")) == "reward_explicit"
    ]
    b0_r2 = (
        statistics.mean([_safe_float(row.get("proxy_r2")) for row in b0_explicit])
        if b0_explicit
        else 0.0
    )
    rep_r2 = (
        statistics.mean([_safe_float(row.get("proxy_r2")) for row in geofm_rep])
        if geofm_rep
        else 0.0
    )
    explicit_r2 = (
        statistics.mean([_safe_float(row.get("proxy_r2")) for row in geofm_explicit])
        if geofm_explicit
        else 0.0
    )
    rep_topk = (
        statistics.mean([_safe_float(row.get("best_topk_enrichment")) for row in geofm_rep])
        if geofm_rep
        else 0.0
    )
    explicit_topk = (
        statistics.mean([_safe_float(row.get("best_topk_enrichment")) for row in geofm_explicit])
        if geofm_explicit
        else 0.0
    )
    return {
        "b0_explicit_proxy_r2_mean": _round_float(b0_r2),
        "geofm_explicit_proxy_r2_mean": _round_float(explicit_r2),
        "geofm_representation_proxy_r2_mean": _round_float(rep_r2),
        "representation_minus_b0_proxy_r2": _round_float(rep_r2 - b0_r2),
        "representation_minus_explicit_proxy_r2": _round_float(rep_r2 - explicit_r2),
        "representation_minus_explicit_topk": _round_float(rep_topk - explicit_topk),
    }


def _case_id(row: Mapping[str, object]) -> str:
    return f"{row.get('variant_id')}:{row.get('eval_tile_id')}:{row.get('seed')}"


def build_phase66_failure_mode_summary(
    atlas_rows: Sequence[Mapping[str, object]],
    alignment_rows: Sequence[Mapping[str, object]],
    phase65_pairwise_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    modes: dict[str, set[str]] = {
        "near_oracle_reward_equivalent": set(),
        "misses_explicit_reward_components": set(),
        "representation_not_aligned_with_base_reward": set(),
        "standardization_hurts_rank_geometry": set(),
        "tile_specific_instability": set(),
        "seed_instability": set(),
    }
    for row in atlas_rows:
        case = _case_id(row)
        if _safe_float(row.get("phase63_oracle_jaccard")) < 0.5 and bool(
            row.get("phase63_reward_equivalent_substitution")
        ):
            modes["near_oracle_reward_equivalent"].add(case)
        if not bool(row.get("phase63_reward_equivalent_substitution")) and str(
            row.get("phase63_missed_oracle_block_ids", "")
        ):
            modes["misses_explicit_reward_components"].add(case)
    by_variant: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in alignment_rows:
        by_variant.setdefault(str(row.get("variant_id")), {})[
            str(row.get("feature_group"))
        ] = row
    for variant_id, groups in by_variant.items():
        if not variant_id.startswith(("D4", "D6")):
            continue
        rep = groups.get("representation_extra")
        explicit = groups.get("reward_explicit")
        if rep is None or explicit is None:
            continue
        if _safe_float(rep.get("proxy_r2")) + PHASE66_ALIGNMENT_ADVANTAGE_THRESHOLD < _safe_float(
            explicit.get("proxy_r2")
        ):
            for row in atlas_rows:
                if str(row.get("variant_id")) == variant_id:
                    modes["representation_not_aligned_with_base_reward"].add(_case_id(row))
    for row in phase65_pairwise_rows:
        if _safe_float(row.get("standardized_minus_unstandardized_reward")) < 0.0:
            modes["standardization_hurts_rank_geometry"].add(_case_id(row))
    tile_counts: dict[str, int] = {}
    seed_counts: dict[str, int] = {}
    for row in atlas_rows:
        if _safe_float(row.get("phase63_oracle_jaccard")) < 0.5:
            tile_id = str(row.get("eval_tile_id"))
            seed = str(row.get("seed"))
            tile_counts[tile_id] = tile_counts.get(tile_id, 0) + 1
            seed_counts[seed] = seed_counts.get(seed, 0) + 1
    for tile_id, count in tile_counts.items():
        if count >= 2:
            modes["tile_specific_instability"].add(tile_id)
    for seed, count in seed_counts.items():
        if count >= 2:
            modes["seed_instability"].add(seed)
    return [
        {
            "failure_mode": mode,
            "case_count": len(cases),
            "representative_cases": ";".join(sorted(cases)[:5]),
            "claim_boundary": PHASE66_CLAIM_BOUNDARY,
        }
        for mode, cases in modes.items()
    ]


def build_phase66_diagnostic_gate(
    coverage_issues: Sequence[object],
    alignment_rows: Sequence[Mapping[str, object]],
    failure_summary_rows: Sequence[Mapping[str, object]],
    suitability_context: Mapping[str, object],
) -> dict[str, object]:
    if coverage_issues:
        return {
            "phase66_status": PHASE66_STATUS_INSUFFICIENT,
            "coverage_issues": list(coverage_issues),
            "alignment_advantage": {},
            "claim_boundary": PHASE66_CLAIM_BOUNDARY,
        }
    advantage = _alignment_advantage_summary(alignment_rows)
    rep_minus_b0 = float(advantage["representation_minus_b0_proxy_r2"])
    rep_minus_explicit = float(advantage["representation_minus_explicit_proxy_r2"])
    failure_counts = {
        str(row.get("failure_mode")): _safe_int(row.get("case_count"))
        for row in failure_summary_rows
    }
    if (
        rep_minus_b0 >= PHASE66_ALIGNMENT_ADVANTAGE_THRESHOLD
        and rep_minus_explicit >= PHASE66_ALIGNMENT_ADVANTAGE_THRESHOLD
    ):
        status = PHASE66_STATUS_REPRESENTATION_ADDS_SIGNAL
    elif (
        failure_counts.get("misses_explicit_reward_components", 0) > 0
        and failure_counts.get("representation_not_aligned_with_base_reward", 0) > 0
        and _suitability_context_not_ready(suitability_context)
    ):
        status = PHASE66_STATUS_BASE_REWARD_MASKS
    else:
        status = PHASE66_STATUS_REPRESENTATION_REDUNDANT
    return {
        "phase66_status": status,
        "coverage_issues": [],
        "alignment_advantage": advantage,
        "failure_mode_counts": failure_counts,
        "suitability_context": dict(suitability_context),
        "claim_boundary": PHASE66_CLAIM_BOUNDARY,
    }


def _component_rows_for_blocks(
    tiled_input,
    reward_index: Mapping[str, Mapping[str, object]],
    block_ids: Sequence[str],
    source: str,
    seed: int,
    action_group: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for block_id in block_ids:
        if str(block_id) not in reward_index:
            raise ValueError(f"Phase 66 block ID missing from reward index: {block_id}")
        base_row = dict(reward_index[str(block_id)])
        rows.append(
            {
                **base_row,
                "source": str(source),
                "seed": int(seed),
                "action_group": str(action_group),
                "claim_boundary": PHASE66_CLAIM_BOUNDARY,
            }
        )
    return rows


def build_phase66_reward_component_attribution(
    phase63_rollout_rows: Sequence[Mapping[str, object]],
    phase65_rollout_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
    tiled_inputs: Mapping[tuple[str, str], object],
) -> list[dict[str, object]]:
    phase63_index = _index_unique_rows(
        [
            row
            for row in phase63_rollout_rows
            if str(row.get("row_type", "")) == "bc_greedy_policy"
        ],
        "Phase 63 rollout",
    )
    phase65_index = _index_unique_rows(
        [
            row
            for row in phase65_rollout_rows
            if str(row.get("row_type", "")) == "bc_greedy_policy"
        ],
        "Phase 65 rollout",
    )
    oracle_index = _index_unique_rows(oracle_rows, "Phase 63 oracle")
    rows: list[dict[str, object]] = []
    for key in sorted(oracle_index):
        variant_id, tile_id, seed = key
        if key not in phase63_index:
            raise ValueError(f"Phase 66 missing Phase 63 rollout row for {key}")
        if key not in phase65_index:
            raise ValueError(f"Phase 66 missing Phase 65 rollout row for {key}")
        tiled = tiled_inputs.get((variant_id, tile_id))
        if tiled is None:
            raise ValueError(f"Phase 66 missing tiled input for {(variant_id, tile_id)}")
        reward_index = _block_rank_and_reward(tiled)
        oracle_ids = _split_semicolon_values(oracle_index[key].get("selected_block_ids"))
        phase63_ids = _split_semicolon_values(phase63_index[key].get("selected_block_ids"))
        phase65_ids = _split_semicolon_values(phase65_index[key].get("selected_block_ids"))
        phase63_missed = [block_id for block_id in oracle_ids if block_id not in set(phase63_ids)]
        phase63_extra = [block_id for block_id in phase63_ids if block_id not in set(oracle_ids)]
        phase65_missed = [block_id for block_id in oracle_ids if block_id not in set(phase65_ids)]
        phase65_extra = [block_id for block_id in phase65_ids if block_id not in set(oracle_ids)]
        rows.extend(
            _component_rows_for_blocks(
                tiled,
                reward_index,
                oracle_ids,
                source="oracle",
                seed=seed,
                action_group="oracle",
            )
        )
        rows.extend(
            _component_rows_for_blocks(
                tiled,
                reward_index,
                phase63_ids,
                source="phase63",
                seed=seed,
                action_group="selected",
            )
        )
        rows.extend(
            _component_rows_for_blocks(
                tiled,
                reward_index,
                phase63_missed,
                source="phase63",
                seed=seed,
                action_group="missed_oracle",
            )
        )
        rows.extend(
            _component_rows_for_blocks(
                tiled,
                reward_index,
                phase63_extra,
                source="phase63",
                seed=seed,
                action_group="extra_selected",
            )
        )
        rows.extend(
            _component_rows_for_blocks(
                tiled,
                reward_index,
                phase65_ids,
                source="phase65",
                seed=seed,
                action_group="selected",
            )
        )
        rows.extend(
            _component_rows_for_blocks(
                tiled,
                reward_index,
                phase65_missed,
                source="phase65",
                seed=seed,
                action_group="missed_oracle",
            )
        )
        rows.extend(
            _component_rows_for_blocks(
                tiled,
                reward_index,
                phase65_extra,
                source="phase65",
                seed=seed,
                action_group="extra_selected",
            )
        )
    return rows


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


def _load_csv_rows(path: Path | str | None, label: str) -> list[dict[str, object]]:
    if path is None:
        return []
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {label}: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _contract_string_list(contract: Mapping[str, object], key: str) -> list[str]:
    value = contract.get(key)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item).strip()]
    return []


def _contract_int_list(contract: Mapping[str, object], key: str) -> list[int]:
    value = contract.get(key)
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, Sequence):
        return [int(item) for item in value if str(item).strip()]
    return []


def _load_phase66_tiled_inputs(contract: Mapping[str, object]) -> dict[tuple[str, str], object]:
    variant_source_dirs = contract.get("variant_source_dirs")
    if not isinstance(variant_source_dirs, Mapping):
        raise ValueError("Phase 66 contract is missing variant_source_dirs")
    tile_index_csv = contract.get("tile_index_csv")
    if not tile_index_csv:
        raise ValueError("Phase 66 contract is missing tile_index_csv")
    variants = _contract_string_list(contract, "variants")
    eval_tile_ids = _contract_string_list(contract, "eval_tile_ids")
    train_tile_id = str(contract.get("train_tile_id", ""))
    tile_ids = [train_tile_id, *eval_tile_ids] if train_tile_id else eval_tile_ids
    if not variants:
        raise ValueError("Phase 66 contract has no variants")
    if not tile_ids:
        raise ValueError("Phase 66 contract has no train/eval tile IDs")
    tiled_inputs: dict[tuple[str, str], object] = {}
    for variant_id in variants:
        source_dir = variant_source_dirs.get(variant_id)
        if source_dir is None:
            raise ValueError(f"Phase 66 contract has no source for variant {variant_id}")
        for tile_id in tile_ids:
            tiled_inputs[(str(variant_id), str(tile_id))] = load_tiled_variant_input(
                source_dir,
                str(tile_index_csv),
                str(tile_id),
                variant_id=str(variant_id),
            )
    return tiled_inputs


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


def _phase66_markdown(analysis: Mapping[str, object]) -> str:
    gate = dict(analysis.get("diagnostic_gate", {}))
    lines = [
        "# Phase 66 Reward-Label Representation Audit",
        "",
        f"Status: {gate.get('phase66_status', '')}",
        "",
        f"Alignment advantage: {gate.get('alignment_advantage', {})}",
        f"Failure mode counts: {gate.get('failure_mode_counts', {})}",
        "",
        "Claim boundary:",
        str(analysis.get("claim_boundary", PHASE66_CLAIM_BOUNDARY)),
        "",
    ]
    return "\n".join(lines)


def write_phase66_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "component_csv": output_path / "phase66_reward_component_attribution.csv",
        "atlas_csv": output_path / "phase66_selected_block_atlas.csv",
        "alignment_csv": output_path / "phase66_representation_rank_alignment.csv",
        "failure_csv": output_path / "phase66_failure_mode_summary.csv",
        "audit_json": output_path / "phase66_reward_label_representation_audit.json",
        "audit_md": output_path / "phase66_reward_label_representation_audit.md",
    }
    _write_csv_rows(
        paths["component_csv"],
        PHASE66_COMPONENT_FIELDNAMES,
        analysis.get("reward_component_rows", []),
    )
    _write_csv_rows(
        paths["atlas_csv"],
        PHASE66_ATLAS_FIELDNAMES,
        analysis.get("selected_block_atlas_rows", []),
    )
    _write_csv_rows(
        paths["alignment_csv"],
        PHASE66_ALIGNMENT_FIELDNAMES,
        analysis.get("representation_rank_alignment_rows", []),
    )
    _write_csv_rows(
        paths["failure_csv"],
        PHASE66_FAILURE_FIELDNAMES,
        analysis.get("failure_mode_summary_rows", []),
    )
    saved = dict(analysis)
    saved["phase66_status"] = dict(analysis.get("diagnostic_gate", {})).get(
        "phase66_status",
        PHASE66_STATUS_INSUFFICIENT,
    )
    paths["audit_json"].write_text(
        json.dumps(_json_ready(saved), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["audit_md"].write_text(_phase66_markdown(analysis), encoding="utf-8")
    return paths


def _filter_rows_for_eval_contract(
    rows: Sequence[Mapping[str, object]],
    variants: Sequence[str],
    eval_tile_ids: Sequence[str],
    seeds: Sequence[int],
) -> list[Mapping[str, object]]:
    allowed = {
        (str(variant_id), str(tile_id), int(seed))
        for variant_id in variants
        for tile_id in eval_tile_ids
        for seed in seeds
    }
    return [row for row in rows if _row_key(row) in allowed]


def run_phase66_reward_label_representation_audit(
    phase63_comparison_json: Path | str,
    phase63_rollout_csv: Path | str,
    phase63_oracle_summary_csv: Path | str,
    phase64_failure_cases_csv: Path | str | None,
    phase64_feature_effective_rank_csv: Path | str | None,
    phase65_comparison_json: Path | str,
    phase65_rollout_csv: Path | str,
    phase65_pairwise_delta_csv: Path | str,
    phase10_reward_readiness_json: Path | str | None = None,
) -> dict[str, object]:
    phase63_comparison = _load_json_object(
        phase63_comparison_json,
        "Phase 63 comparison JSON",
    )
    phase65_comparison = _load_json_object(
        phase65_comparison_json,
        "Phase 65 comparison JSON",
    )
    suitability_context = _load_json_object(
        phase10_reward_readiness_json,
        "Phase 10 reward readiness JSON",
    )
    contract = phase63_comparison.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Phase 63 comparison JSON is missing contract metadata")
    variants = _contract_string_list(contract, "variants")
    eval_tile_ids = _contract_string_list(contract, "eval_tile_ids")
    seeds = _contract_int_list(contract, "seeds")
    if not variants:
        raise ValueError("Phase 66 contract has no variants")
    if not eval_tile_ids:
        raise ValueError("Phase 66 contract has no eval_tile_ids")
    if not seeds:
        raise ValueError("Phase 66 contract has no seeds")
    phase63_rows = _filter_rows_for_eval_contract(
        _load_csv_rows(phase63_rollout_csv, "Phase 63 rollout CSV"),
        variants,
        eval_tile_ids,
        seeds,
    )
    oracle_rows = _filter_rows_for_eval_contract(
        _load_csv_rows(phase63_oracle_summary_csv, "Phase 63 oracle summary CSV"),
        variants,
        eval_tile_ids,
        seeds,
    )
    phase64_failure_rows = _load_csv_rows(
        phase64_failure_cases_csv,
        "Phase 64 failure cases CSV",
    )
    phase64_rank_rows = _load_csv_rows(
        phase64_feature_effective_rank_csv,
        "Phase 64 feature effective rank CSV",
    )
    phase65_rows = _filter_rows_for_eval_contract(
        _load_csv_rows(phase65_rollout_csv, "Phase 65 rollout CSV"),
        variants,
        eval_tile_ids,
        seeds,
    )
    phase65_pairwise_rows = _filter_rows_for_eval_contract(
        _load_csv_rows(phase65_pairwise_delta_csv, "Phase 65 pairwise delta CSV"),
        variants,
        eval_tile_ids,
        seeds,
    )
    tiled_inputs = _load_phase66_tiled_inputs(contract)
    eval_tiled_inputs = {
        key: tiled
        for key, tiled in tiled_inputs.items()
        if key[1] in set(eval_tile_ids)
    }
    eval_max_steps = int(contract.get("eval_max_steps", 8))
    atlas_rows = build_phase66_selected_block_atlas(
        phase63_rollout_rows=phase63_rows,
        phase65_rollout_rows=phase65_rows,
        oracle_rows=oracle_rows,
        tiled_inputs=eval_tiled_inputs,
    )
    component_rows = build_phase66_reward_component_attribution(
        phase63_rollout_rows=phase63_rows,
        phase65_rollout_rows=phase65_rows,
        oracle_rows=oracle_rows,
        tiled_inputs=eval_tiled_inputs,
    )
    alignment_rows = build_phase66_representation_rank_alignment(
        tiled_inputs=eval_tiled_inputs,
        eval_max_steps=eval_max_steps,
    )
    failure_rows = build_phase66_failure_mode_summary(
        atlas_rows,
        alignment_rows,
        phase65_pairwise_rows,
    )
    coverage_issues: list[object] = []
    if dict(phase65_comparison).get("phase65_status") == PHASE66_STATUS_INSUFFICIENT:
        coverage_issues.append("Phase 65 status is insufficient")
    gate = build_phase66_diagnostic_gate(
        coverage_issues,
        alignment_rows,
        failure_rows,
        suitability_context,
    )
    return {
        "phase": "phase66_reward_label_representation_audit",
        "phase63_comparison_json": str(Path(phase63_comparison_json)),
        "phase63_rollout_csv": str(Path(phase63_rollout_csv)),
        "phase63_oracle_summary_csv": str(Path(phase63_oracle_summary_csv)),
        "phase64_failure_cases_csv": ""
        if phase64_failure_cases_csv is None
        else str(Path(phase64_failure_cases_csv)),
        "phase64_feature_effective_rank_csv": ""
        if phase64_feature_effective_rank_csv is None
        else str(Path(phase64_feature_effective_rank_csv)),
        "phase65_comparison_json": str(Path(phase65_comparison_json)),
        "phase65_rollout_csv": str(Path(phase65_rollout_csv)),
        "phase65_pairwise_delta_csv": str(Path(phase65_pairwise_delta_csv)),
        "contract": dict(contract),
        "phase64_failure_case_rows_loaded": len(phase64_failure_rows),
        "phase64_feature_effective_rank_rows_loaded": len(phase64_rank_rows),
        "reward_component_rows": component_rows,
        "selected_block_atlas_rows": atlas_rows,
        "representation_rank_alignment_rows": alignment_rows,
        "failure_mode_summary_rows": failure_rows,
        "diagnostic_gate": gate,
        "claim_boundary": PHASE66_CLAIM_BOUNDARY,
    }


def _suitability_context_not_ready(suitability_context: Mapping[str, object]) -> bool:
    status = str(
        suitability_context.get(
            "phase10_status",
            suitability_context.get("status", ""),
        )
    )
    recommendation = str(suitability_context.get("recommendation", ""))
    return status.startswith("not_ready") or recommendation == "do_not_enable_suitability_reward"