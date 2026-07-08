from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

from .planning_reward import compute_base_planning_reward_from_matrix_row
from .tiled_inputs import load_tiled_variant_input


PHASE64_CLAIM_BOUNDARY = (
    "Phase 64 is a read-only set-policy error-diagnosis and standardization-gate "
    "phase. It uses Phase 63 base-reward artifacts to diagnose behavior-cloned "
    "set-policy errors and decide whether a train-tile-fitted standardization "
    "rerun is justified. It does not enable suitability reward, does not test "
    "B2/B3, does not test transfer, does not prove GeoFM advantage or PCA "
    "optimality, and does not justify formal submission-level claims."
)

PHASE64_STATUS_STANDARDIZATION = "standardization_route_supported"
PHASE64_STATUS_CAPACITY = "bc_training_capacity_limited"
PHASE64_STATUS_NOT_HELPFUL = "geofm_features_not_helpful_under_set_policy"
PHASE64_STATUS_INCONCLUSIVE = "diagnostic_inconclusive"

PHASE64_WEAK_TOP1_THRESHOLD = 0.25
PHASE64_WEAK_TOPK_THRESHOLD = 0.50
PHASE64_STD_RATIO_THRESHOLD = 100.0
PHASE64_MEAN_SCALE_RATIO_THRESHOLD = 10.0
PHASE64_Z_SHIFT_THRESHOLD = 3.0
PHASE64_EFFECTIVE_RANK_FRACTION_THRESHOLD = 0.30
PHASE64_PC1_SHARE_THRESHOLD = 0.80

PHASE64_CONVERGENCE_FIELDNAMES = [
    "variant_id",
    "train_tile_id",
    "seed",
    "first_epoch",
    "final_epoch",
    "best_epoch",
    "first_loss",
    "final_loss",
    "best_loss",
    "final_top1_accuracy",
    "best_top1_accuracy",
    "final_topk_hit_rate",
    "best_topk_hit_rate",
    "loss_delta",
    "claim_boundary",
]


def _split_semicolon_values(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _round_float(value: object, digits: int = 10) -> float:
    rounded = round(float(value), digits)
    compact = round(rounded, 6)
    if abs(rounded - compact) < 5.0e-8:
        return compact
    return rounded


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def _safe_int(value: object, default: int = 0) -> int:
    if value is None or str(value).strip() == "":
        return int(default)
    return int(float(value))


def _load_csv_rows(path: Path | str, label: str) -> list[dict[str, object]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {label}: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_json_object(path: Path | str, label: str) -> dict[str, object]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {label}: {json_path}")
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return loaded


def _group_key(row: Mapping[str, object], fields: Sequence[str]) -> tuple[object, ...]:
    return tuple(row.get(field, "") for field in fields)


def build_phase64_convergence_summary(
    history_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(history_rows_or_csv, (str, Path)):
        history_rows = _load_csv_rows(history_rows_or_csv, "Phase 63 BC history CSV")
    else:
        history_rows = [dict(row) for row in history_rows_or_csv]

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in history_rows:
        key = _group_key(row, ("variant_id", "train_tile_id", "seed"))
        grouped.setdefault(key, []).append(dict(row))

    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: _safe_int(row.get("epoch")))
        first = rows[0]
        final = rows[-1]
        best = min(rows, key=lambda row: _safe_float(row.get("loss"), math.inf))
        first_loss = _safe_float(first.get("loss"))
        final_loss = _safe_float(final.get("loss"))
        output.append(
            {
                "variant_id": str(key[0]),
                "train_tile_id": str(key[1]),
                "seed": _safe_int(key[2]),
                "first_epoch": _safe_int(first.get("epoch")),
                "final_epoch": _safe_int(final.get("epoch")),
                "best_epoch": _safe_int(best.get("epoch")),
                "first_loss": _round_float(first_loss),
                "final_loss": _round_float(final_loss),
                "best_loss": _round_float(best.get("loss")),
                "final_top1_accuracy": _round_float(final.get("top1_accuracy")),
                "best_top1_accuracy": _round_float(
                    max(_safe_float(row.get("top1_accuracy")) for row in rows)
                ),
                "final_topk_hit_rate": _round_float(final.get("topk_hit_rate")),
                "best_topk_hit_rate": _round_float(
                    max(_safe_float(row.get("topk_hit_rate")) for row in rows)
                ),
                "loss_delta": _round_float(final_loss - first_loss),
                "claim_boundary": PHASE64_CLAIM_BOUNDARY,
            }
        )
    return output

PHASE64_OVERLAP_FIELDNAMES = [
    "variant_id",
    "train_tile_id",
    "eval_tile_id",
    "seed",
    "eval_max_steps",
    "selected_overlap_count",
    "selected_overlap_fraction",
    "prefix_overlap_count",
    "jaccard_similarity",
    "duplicate_selection_count",
    "invalid_action_count",
    "bc_reward",
    "oracle_total_reward",
    "oracle_gap",
    "oracle_gap_fraction",
    "selected_block_ids",
    "oracle_block_ids",
    "missed_oracle_block_ids",
    "extra_selected_block_ids",
    "claim_boundary",
]

PHASE64_ORACLE_RANK_FIELDNAMES = [
    "variant_id",
    "eval_tile_id",
    "seed",
    "eval_max_steps",
    "selected_rank_values",
    "selected_reward_values",
    "missed_oracle_block_ids",
    "missed_oracle_rewards",
    "reward_loss_from_missed_oracle",
    "worst_selected_rank",
    "selected_outside_top_eval_max_steps",
    "selected_outside_top16",
    "selected_outside_top32",
    "claim_boundary",
]


def _phase64_row_key(row: Mapping[str, object]) -> tuple[str, str, int]:
    return (
        str(row.get("variant_id", "")),
        str(row.get("eval_tile_id", row.get("tile_id", ""))),
        _safe_int(row.get("seed")),
    )


def build_phase64_rollout_overlap(
    rollout_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    oracle_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rollout_rows = (
        _load_csv_rows(rollout_rows_or_csv, "Phase 63 rollout CSV")
        if isinstance(rollout_rows_or_csv, (str, Path))
        else [dict(row) for row in rollout_rows_or_csv]
    )
    oracle_rows = (
        _load_csv_rows(oracle_rows_or_csv, "Phase 63 oracle summary CSV")
        if isinstance(oracle_rows_or_csv, (str, Path))
        else [dict(row) for row in oracle_rows_or_csv]
    )
    oracle_index = {
        (
            str(row.get("variant_id", "")),
            str(row.get("tile_id", "")),
            _safe_int(row.get("seed")),
        ): row
        for row in oracle_rows
    }

    output: list[dict[str, object]] = []
    for rollout in rollout_rows:
        key = _phase64_row_key(rollout)
        oracle = oracle_index.get(key)
        if oracle is None:
            oracle_blocks: list[str] = []
        else:
            oracle_blocks = _split_semicolon_values(oracle.get("selected_block_ids"))
        selected_blocks = _split_semicolon_values(rollout.get("selected_block_ids"))
        selected_unique = set(selected_blocks)
        oracle_set = set(oracle_blocks)
        overlap = selected_unique.intersection(oracle_set)
        union = selected_unique.union(oracle_set)
        prefix_overlap = sum(
            1
            for left, right in zip(selected_blocks, oracle_blocks)
            if left == right
        )
        duplicate_count = len(selected_blocks) - len(selected_unique)
        missed = [block_id for block_id in oracle_blocks if block_id not in selected_unique]
        extra = list(dict.fromkeys(block_id for block_id in selected_blocks if block_id not in oracle_set))
        denom = max(len(oracle_blocks), 1)
        output.append(
            {
                "variant_id": key[0],
                "train_tile_id": str(rollout.get("train_tile_id", "")),
                "eval_tile_id": key[1],
                "seed": key[2],
                "eval_max_steps": _safe_int(rollout.get("eval_max_steps")),
                "selected_overlap_count": len(overlap),
                "selected_overlap_fraction": _round_float(len(overlap) / denom),
                "prefix_overlap_count": int(prefix_overlap),
                "jaccard_similarity": _round_float(len(overlap) / max(len(union), 1)),
                "duplicate_selection_count": int(duplicate_count),
                "invalid_action_count": _safe_int(rollout.get("invalid_action_count")),
                "bc_reward": _round_float(rollout.get("total_contract_reward")),
                "oracle_total_reward": _round_float(rollout.get("oracle_total_reward")),
                "oracle_gap": _round_float(rollout.get("oracle_gap")),
                "oracle_gap_fraction": _round_float(rollout.get("oracle_gap_fraction")),
                "selected_block_ids": ";".join(selected_blocks),
                "oracle_block_ids": ";".join(oracle_blocks),
                "missed_oracle_block_ids": ";".join(missed),
                "extra_selected_block_ids": ";".join(extra),
                "claim_boundary": PHASE64_CLAIM_BOUNDARY,
            }
        )
    return output


def _block_reward_ranking(tiled_input) -> list[dict[str, object]]:
    rewards = [
        compute_base_planning_reward_from_matrix_row(
            tiled_input.feature_columns,
            tiled_input.state_matrix[index],
        )
        for index in range(len(tiled_input.block_ids))
    ]
    ranked_indices = sorted(
        range(len(tiled_input.block_ids)),
        key=lambda index: (-rewards[index], str(tiled_input.block_ids[index]), index),
    )
    output: list[dict[str, object]] = []
    for rank, index in enumerate(ranked_indices, start=1):
        output.append(
            {
                "rank": rank,
                "action_index": int(index),
                "block_id": str(tiled_input.block_ids[index]),
                "reward": _round_float(rewards[index]),
            }
        )
    return output


def build_phase64_oracle_rank_gap(
    rollout_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    tiled_inputs: Mapping[tuple[str, str], object],
) -> list[dict[str, object]]:
    rollout_rows = (
        _load_csv_rows(rollout_rows_or_csv, "Phase 63 rollout CSV")
        if isinstance(rollout_rows_or_csv, (str, Path))
        else [dict(row) for row in rollout_rows_or_csv]
    )
    output: list[dict[str, object]] = []
    ranking_cache: dict[tuple[str, str], list[dict[str, object]]] = {}
    for rollout in rollout_rows:
        variant_id, eval_tile_id, seed = _phase64_row_key(rollout)
        input_key = (variant_id, eval_tile_id)
        if input_key not in ranking_cache:
            ranking_cache[input_key] = _block_reward_ranking(tiled_inputs[input_key])
        ranking = ranking_cache[input_key]
        by_block = {str(row["block_id"]): row for row in ranking}
        eval_max_steps = _safe_int(rollout.get("eval_max_steps"))
        oracle_top = [str(row["block_id"]) for row in ranking[:eval_max_steps]]
        selected = _split_semicolon_values(rollout.get("selected_block_ids"))
        selected_set = set(selected)
        selected_rows = [by_block[block_id] for block_id in selected if block_id in by_block]
        selected_ranks = [_safe_int(row["rank"]) for row in selected_rows]
        selected_rewards = [_safe_float(row["reward"]) for row in selected_rows]
        missed = [block_id for block_id in oracle_top if block_id not in selected_set]
        missed_rewards = [_safe_float(by_block[block_id]["reward"]) for block_id in missed]
        selected_non_oracle = [
            row
            for row in selected_rows
            if _safe_int(row["rank"]) > eval_max_steps
        ]
        selected_non_oracle_rewards = [_safe_float(row["reward"]) for row in selected_non_oracle]
        reward_loss = sum(missed_rewards) - sum(selected_non_oracle_rewards[: len(missed_rewards)])
        output.append(
            {
                "variant_id": variant_id,
                "eval_tile_id": eval_tile_id,
                "seed": seed,
                "eval_max_steps": eval_max_steps,
                "selected_rank_values": ";".join(str(rank) for rank in selected_ranks),
                "selected_reward_values": ";".join(str(_round_float(value)) for value in selected_rewards),
                "missed_oracle_block_ids": ";".join(missed),
                "missed_oracle_rewards": ";".join(str(_round_float(value)) for value in missed_rewards),
                "reward_loss_from_missed_oracle": _round_float(max(reward_loss, 0.0)),
                "worst_selected_rank": max(selected_ranks) if selected_ranks else 0,
                "selected_outside_top_eval_max_steps": sum(1 for rank in selected_ranks if rank > eval_max_steps),
                "selected_outside_top16": sum(1 for rank in selected_ranks if rank > 16),
                "selected_outside_top32": sum(1 for rank in selected_ranks if rank > 32),
                "claim_boundary": PHASE64_CLAIM_BOUNDARY,
            }
        )
    return output

PHASE64_FEATURE_SCALE_FIELDNAMES = [
    "variant_id",
    "tile_role",
    "tile_id",
    "feature_index",
    "feature_name",
    "mean",
    "std",
    "min",
    "max",
    "median",
    "p1",
    "p99",
    "train_mean",
    "train_std",
    "eval_mean_z_shift",
    "zero_variance",
    "claim_boundary",
]

PHASE64_EFFECTIVE_RANK_FIELDNAMES = [
    "variant_id",
    "tile_role",
    "tile_id",
    "n_blocks",
    "n_features",
    "zero_variance_feature_count",
    "std_ratio",
    "mean_scale_ratio",
    "max_train_eval_abs_z_shift",
    "effective_rank",
    "effective_rank_fraction",
    "pc1_variance_share",
    "pc3_variance_share",
    "scale_flag",
    "shift_flag",
    "rank_flag",
    "claim_boundary",
]


def _feature_distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": _round_float(np.mean(array)),
        "std": _round_float(np.std(array, ddof=0)),
        "min": _round_float(np.min(array)),
        "max": _round_float(np.max(array)),
        "median": _round_float(np.median(array)),
        "p1": _round_float(np.percentile(array, 1)),
        "p99": _round_float(np.percentile(array, 99)),
    }


def _effective_rank_stats(matrix: np.ndarray) -> dict[str, float]:
    values = np.asarray(matrix, dtype=float)
    if values.size == 0:
        return {
            "effective_rank": 0.0,
            "effective_rank_fraction": 0.0,
            "pc1_variance_share": 0.0,
            "pc3_variance_share": 0.0,
        }
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    positive = singular_values[singular_values > 1.0e-12]
    if positive.size == 0:
        effective_rank = 0.0
    else:
        probabilities = positive / np.sum(positive)
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        effective_rank = float(np.exp(entropy))
    squared = singular_values ** 2
    variance_total = float(np.sum(squared))
    pc1_share = float(squared[0] / variance_total) if variance_total > 0.0 else 0.0
    pc3_share = float(np.sum(squared[:3]) / variance_total) if variance_total > 0.0 else 0.0
    max_rank = max(1, min(values.shape))
    return {
        "effective_rank": _round_float(effective_rank),
        "effective_rank_fraction": _round_float(effective_rank / max_rank),
        "pc1_variance_share": _round_float(pc1_share),
        "pc3_variance_share": _round_float(pc3_share),
    }


def _train_feature_reference(
    tiled_inputs: Sequence[tuple[str, object]],
    train_tile_ids: Mapping[str, str],
) -> dict[str, dict[str, np.ndarray]]:
    reference: dict[str, dict[str, np.ndarray]] = {}
    for role, tiled_input in tiled_inputs:
        variant_id = str(tiled_input.variant_id)
        if str(role) != "train":
            continue
        if str(tiled_input.tile_id) != str(train_tile_ids.get(variant_id, tiled_input.tile_id)):
            continue
        matrix = np.asarray(tiled_input.state_matrix, dtype=float)
        reference[variant_id] = {
            "mean": np.mean(matrix, axis=0),
            "std": np.std(matrix, axis=0, ddof=0),
        }
    return reference


def build_phase64_feature_diagnostics(
    tiled_inputs: Sequence[tuple[str, object]],
    train_tile_ids: Mapping[str, str],
) -> dict[str, list[dict[str, object]]]:
    train_reference = _train_feature_reference(tiled_inputs, train_tile_ids)
    feature_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    for tile_role, tiled_input in tiled_inputs:
        variant_id = str(tiled_input.variant_id)
        matrix = np.asarray(tiled_input.state_matrix, dtype=float)
        train_stats = train_reference.get(variant_id)
        if train_stats is None:
            train_mean = np.mean(matrix, axis=0)
            train_std = np.std(matrix, axis=0, ddof=0)
        else:
            train_mean = train_stats["mean"]
            train_std = train_stats["std"]
        safe_train_std = np.where(train_std > 1.0e-12, train_std, np.nan)
        z_shift_values: list[float] = []
        std_values: list[float] = []
        mean_scale_values: list[float] = []
        zero_variance_count = 0
        for feature_index, feature_name in enumerate(tiled_input.feature_columns):
            values = matrix[:, feature_index]
            dist = _feature_distribution(values)
            feature_std = float(dist["std"])
            if feature_std <= 1.0e-12:
                zero_variance_count += 1
            else:
                std_values.append(feature_std)
            train_std_value = float(train_std[feature_index])
            z_shift = 0.0
            if train_std_value > 1.0e-12:
                z_shift = (float(dist["mean"]) - float(train_mean[feature_index])) / train_std_value
                z_shift_values.append(abs(z_shift))
            median_non_zero_std = max(float(np.nanmedian(safe_train_std)), 1.0e-12)
            mean_scale_values.append(abs(float(dist["mean"])) / median_non_zero_std)
            feature_rows.append(
                {
                    "variant_id": variant_id,
                    "tile_role": str(tile_role),
                    "tile_id": str(tiled_input.tile_id),
                    "feature_index": int(feature_index),
                    "feature_name": str(feature_name),
                    **dist,
                    "train_mean": _round_float(train_mean[feature_index]),
                    "train_std": _round_float(train_std_value),
                    "eval_mean_z_shift": _round_float(z_shift),
                    "zero_variance": bool(feature_std <= 1.0e-12),
                    "claim_boundary": PHASE64_CLAIM_BOUNDARY,
                }
            )
        non_zero_std = [value for value in std_values if value > 1.0e-12]
        std_ratio = max(non_zero_std) / min(non_zero_std) if non_zero_std else 0.0
        mean_scale_ratio = max(mean_scale_values) if mean_scale_values else 0.0
        max_z_shift = max(z_shift_values) if z_shift_values else 0.0
        rank_stats = _effective_rank_stats(matrix)
        scale_flag = (
            std_ratio >= PHASE64_STD_RATIO_THRESHOLD
            or mean_scale_ratio >= PHASE64_MEAN_SCALE_RATIO_THRESHOLD
        )
        shift_flag = max_z_shift >= PHASE64_Z_SHIFT_THRESHOLD
        rank_flag = (
            rank_stats["effective_rank_fraction"] <= PHASE64_EFFECTIVE_RANK_FRACTION_THRESHOLD
            or rank_stats["pc1_variance_share"] >= PHASE64_PC1_SHARE_THRESHOLD
        )
        rank_rows.append(
            {
                "variant_id": variant_id,
                "tile_role": str(tile_role),
                "tile_id": str(tiled_input.tile_id),
                "n_blocks": int(matrix.shape[0]),
                "n_features": int(matrix.shape[1]),
                "zero_variance_feature_count": int(zero_variance_count),
                "std_ratio": _round_float(std_ratio),
                "mean_scale_ratio": _round_float(mean_scale_ratio),
                "max_train_eval_abs_z_shift": _round_float(max_z_shift),
                **rank_stats,
                "scale_flag": bool(scale_flag),
                "shift_flag": bool(shift_flag),
                "rank_flag": bool(rank_flag),
                "claim_boundary": PHASE64_CLAIM_BOUNDARY,
            }
        )
    return {
        "feature_scale_rows": feature_rows,
        "feature_effective_rank_rows": rank_rows,
    }


PHASE64_FAILURE_CASE_FIELDNAMES = [
    "case_type",
    "variant_id",
    "eval_tile_id",
    "seed",
    "bc_reward",
    "oracle_total_reward",
    "oracle_gap",
    "oracle_gap_fraction",
    "selected_overlap_fraction",
    "worst_selected_rank",
    "reward_loss_from_missed_oracle",
    "selected_block_ids",
    "missed_oracle_block_ids",
    "training_best_top1_accuracy",
    "training_best_topk_hit_rate",
    "feature_flags",
    "claim_boundary",
]


def _mean_numeric(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [
        _safe_float(row.get(field))
        for row in rows
        if str(row.get(field, "")).strip() != ""
    ]
    return statistics.mean(values) if values else 0.0


def _coverage_incomplete(comparison: Mapping[str, object]) -> bool:
    coverage = comparison.get("coverage_issues", {})
    if not isinstance(coverage, Mapping):
        return True
    return bool(
        coverage.get("missing_rollout_rows")
        or coverage.get("duplicate_rollout_rows")
        or coverage.get("unexpected_rollout_rows")
    )


def _d4_underperforms(comparison: Mapping[str, object]) -> bool:
    d4_b0 = comparison.get("d4_b0_delta_summary", {})
    d4_d6 = comparison.get("d4_d6_delta_summary", {})
    return (
        _safe_float(d4_b0.get("mean_delta"), 0.0) <= 0.0
        or _safe_float(d4_d6.get("mean_delta"), 0.0) <= 0.0
    )


def _geofm_feature_flags(feature_effective_rank_rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    flags = {"scale_flag_count": 0, "shift_flag_count": 0, "rank_flag_count": 0}
    for row in feature_effective_rank_rows:
        variant_id = str(row.get("variant_id", ""))
        if not (variant_id.startswith("D4") or variant_id.startswith("D6")):
            continue
        if str(row.get("tile_role", "")) not in {"train", "eval"}:
            continue
        if bool(row.get("scale_flag")):
            flags["scale_flag_count"] += 1
        if bool(row.get("shift_flag")):
            flags["shift_flag_count"] += 1
        if bool(row.get("rank_flag")):
            flags["rank_flag_count"] += 1
    return flags


def build_phase64_standardization_gate(
    phase63_comparison: Mapping[str, object],
    convergence_rows: Sequence[Mapping[str, object]],
    feature_effective_rank_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    flags = _geofm_feature_flags(feature_effective_rank_rows)
    mean_best_top1 = _mean_numeric(convergence_rows, "best_top1_accuracy")
    mean_best_topk = _mean_numeric(convergence_rows, "best_topk_hit_rate")
    capacity_limited = (
        mean_best_top1 < PHASE64_WEAK_TOP1_THRESHOLD
        and mean_best_topk < PHASE64_WEAK_TOPK_THRESHOLD
    )
    d4_underperformance = _d4_underperforms(phase63_comparison)
    feature_flagged = any(value > 0 for value in flags.values())
    if _coverage_incomplete(phase63_comparison):
        status = PHASE64_STATUS_INCONCLUSIVE
        recommendation = False
        reason = "Phase 63 coverage is incomplete."
    elif capacity_limited:
        status = PHASE64_STATUS_CAPACITY
        recommendation = False
        reason = "Behavior cloning convergence is weak across variants."
    elif d4_underperformance and feature_flagged:
        status = PHASE64_STATUS_STANDARDIZATION
        recommendation = True
        reason = "D4/D6 underperformance coincides with feature scale, shift, or rank flags."
    elif d4_underperformance and not feature_flagged and mean_best_topk >= PHASE64_WEAK_TOPK_THRESHOLD:
        status = PHASE64_STATUS_NOT_HELPFUL
        recommendation = False
        reason = "D4 remains behind without scale, shift, or rank flags under adequate convergence."
    else:
        status = PHASE64_STATUS_INCONCLUSIVE
        recommendation = False
        reason = "Diagnostics do not isolate one next experiment."
    return {
        "phase": "phase64_standardization_gate",
        "phase64_status": status,
        "recommend_standardized_rerun": bool(recommendation),
        "reason": reason,
        "mean_best_top1_accuracy": _round_float(mean_best_top1),
        "mean_best_topk_hit_rate": _round_float(mean_best_topk),
        "d4_underperformance": bool(d4_underperformance),
        **flags,
        "claim_boundary": PHASE64_CLAIM_BOUNDARY,
    }


def _index_by_variant_tile_seed(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str, int], Mapping[str, object]]:
    return {
        (
            str(row.get("variant_id", "")),
            str(row.get("eval_tile_id", row.get("tile_id", ""))),
            _safe_int(row.get("seed")),
        ): row
        for row in rows
    }


def _training_index(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, int], Mapping[str, object]]:
    return {
        (str(row.get("variant_id", "")), _safe_int(row.get("seed"))): row
        for row in rows
    }


def _feature_flag_index(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], str]:
    output: dict[tuple[str, str], str] = {}
    for row in rows:
        flags = []
        if bool(row.get("scale_flag")):
            flags.append("scale")
        if bool(row.get("shift_flag")):
            flags.append("shift")
        if bool(row.get("rank_flag")):
            flags.append("rank")
        output[(str(row.get("variant_id", "")), str(row.get("tile_id", "")))] = ";".join(flags)
    return output


def build_phase64_failure_cases(
    phase63_comparison: Mapping[str, object],
    overlap_rows: Sequence[Mapping[str, object]],
    oracle_rank_rows: Sequence[Mapping[str, object]],
    convergence_rows: Sequence[Mapping[str, object]],
    feature_effective_rank_rows: Sequence[Mapping[str, object]],
    limit: int = 12,
) -> list[dict[str, object]]:
    rank_index = _index_by_variant_tile_seed(oracle_rank_rows)
    train_index = _training_index(convergence_rows)
    flag_index = _feature_flag_index(feature_effective_rank_rows)
    candidates: list[tuple[float, str, Mapping[str, object]]] = []
    for row in overlap_rows:
        candidates.append((_safe_float(row.get("oracle_gap_fraction")), "highest_oracle_gap", row))
        if str(row.get("variant_id", "")).startswith("D4"):
            candidates.append((_safe_float(row.get("oracle_gap_fraction")), "d4_high_oracle_gap", row))
        if _safe_float(row.get("selected_overlap_fraction"), 1.0) < 0.5:
            candidates.append((1.0 - _safe_float(row.get("selected_overlap_fraction")), "weak_selected_overlap", row))
    for delta_row in phase63_comparison.get("d4_b0_delta_rows", []):
        if _safe_float(delta_row.get("left_minus_right_reward")) < 0.0:
            candidates.append(
                (
                    abs(_safe_float(delta_row.get("left_minus_right_reward"))),
                    "d4_loses_to_b0",
                    {
                        "variant_id": delta_row.get("left_variant_id", ""),
                        "eval_tile_id": delta_row.get("eval_tile_id", ""),
                        "seed": delta_row.get("seed", 0),
                    },
                )
            )
    for delta_row in phase63_comparison.get("d4_d6_delta_rows", []):
        if _safe_float(delta_row.get("left_minus_right_reward")) < 0.0:
            candidates.append(
                (
                    abs(_safe_float(delta_row.get("left_minus_right_reward"))),
                    "d4_loses_to_d6",
                    {
                        "variant_id": delta_row.get("left_variant_id", ""),
                        "eval_tile_id": delta_row.get("eval_tile_id", ""),
                        "seed": delta_row.get("seed", 0),
                    },
                )
            )
    output: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, int]] = set()
    overlap_index = _index_by_variant_tile_seed(overlap_rows)
    for _score, case_type, base_row in sorted(candidates, key=lambda item: (-item[0], item[1])):
        variant_id = str(base_row.get("variant_id", ""))
        eval_tile_id = str(base_row.get("eval_tile_id", ""))
        seed = _safe_int(base_row.get("seed"))
        seen_key = (case_type, variant_id, eval_tile_id, seed)
        if seen_key in seen:
            continue
        seen.add(seen_key)
        key = (variant_id, eval_tile_id, seed)
        overlap = overlap_index.get(key, base_row)
        rank = rank_index.get(key, {})
        training = train_index.get((variant_id, seed), {})
        feature_flags = flag_index.get((variant_id, eval_tile_id), "")
        output.append(
            {
                "case_type": case_type,
                "variant_id": variant_id,
                "eval_tile_id": eval_tile_id,
                "seed": seed,
                "bc_reward": _round_float(overlap.get("bc_reward", 0.0)),
                "oracle_total_reward": _round_float(overlap.get("oracle_total_reward", 0.0)),
                "oracle_gap": _round_float(overlap.get("oracle_gap", 0.0)),
                "oracle_gap_fraction": _round_float(overlap.get("oracle_gap_fraction", 0.0)),
                "selected_overlap_fraction": _round_float(overlap.get("selected_overlap_fraction", 0.0)),
                "worst_selected_rank": _safe_int(rank.get("worst_selected_rank")),
                "reward_loss_from_missed_oracle": _round_float(rank.get("reward_loss_from_missed_oracle", 0.0)),
                "selected_block_ids": str(overlap.get("selected_block_ids", "")),
                "missed_oracle_block_ids": str(overlap.get("missed_oracle_block_ids", "")),
                "training_best_top1_accuracy": _round_float(training.get("best_top1_accuracy", 0.0)),
                "training_best_topk_hit_rate": _round_float(training.get("best_topk_hit_rate", 0.0)),
                "feature_flags": feature_flags,
                "claim_boundary": PHASE64_CLAIM_BOUNDARY,
            }
        )
        if len(output) >= int(limit):
            break
    return output


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
    if isinstance(value, Path):
        return str(value)
    return value


def _phase64_markdown(analysis: Mapping[str, object]) -> str:
    gate = dict(analysis.get("standardization_gate", {}))
    lines = [
        "# Phase 64 Set-Policy Error Diagnosis",
        "",
        f"Status: {gate.get('phase64_status', '')}",
        "",
        f"Recommendation: standardized rerun = {gate.get('recommend_standardized_rerun', False)}",
        "",
        f"Reason: {gate.get('reason', '')}",
        "",
        "Gate evidence:",
        f"- mean best top-1 accuracy: {gate.get('mean_best_top1_accuracy', '')}",
        f"- mean best top-k hit rate: {gate.get('mean_best_topk_hit_rate', '')}",
        f"- D4 underperformance: {gate.get('d4_underperformance', '')}",
        f"- scale flag count: {gate.get('scale_flag_count', '')}",
        f"- shift flag count: {gate.get('shift_flag_count', '')}",
        f"- rank flag count: {gate.get('rank_flag_count', '')}",
        "",
        "Failure case rows:",
        f"- {len(analysis.get('failure_case_rows', []))}",
        "",
        "Claim boundary:",
        str(analysis.get("claim_boundary", PHASE64_CLAIM_BOUNDARY)),
        "",
    ]
    return "\n".join(lines)


def write_phase64_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "convergence_csv": output_path / "phase64_convergence_summary.csv",
        "overlap_csv": output_path / "phase64_rollout_overlap.csv",
        "oracle_rank_csv": output_path / "phase64_oracle_rank_gap.csv",
        "feature_scale_csv": output_path / "phase64_feature_scale_summary.csv",
        "effective_rank_csv": output_path / "phase64_feature_effective_rank.csv",
        "failure_cases_csv": output_path / "phase64_failure_cases.csv",
        "gate_json": output_path / "phase64_standardization_gate.json",
        "diagnosis_md": output_path / "phase64_set_policy_error_diagnosis.md",
    }
    _write_csv_rows(paths["convergence_csv"], PHASE64_CONVERGENCE_FIELDNAMES, analysis.get("convergence_rows", []))
    _write_csv_rows(paths["overlap_csv"], PHASE64_OVERLAP_FIELDNAMES, analysis.get("overlap_rows", []))
    _write_csv_rows(paths["oracle_rank_csv"], PHASE64_ORACLE_RANK_FIELDNAMES, analysis.get("oracle_rank_gap_rows", []))
    _write_csv_rows(paths["feature_scale_csv"], PHASE64_FEATURE_SCALE_FIELDNAMES, analysis.get("feature_scale_rows", []))
    _write_csv_rows(paths["effective_rank_csv"], PHASE64_EFFECTIVE_RANK_FIELDNAMES, analysis.get("feature_effective_rank_rows", []))
    _write_csv_rows(paths["failure_cases_csv"], PHASE64_FAILURE_CASE_FIELDNAMES, analysis.get("failure_case_rows", []))
    paths["gate_json"].write_text(
        json.dumps(_json_ready(analysis.get("standardization_gate", {})), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["diagnosis_md"].write_text(_phase64_markdown(analysis), encoding="utf-8")
    return paths


def _contract_tiled_inputs(contract: Mapping[str, object]) -> list[tuple[str, object]]:
    variant_source_dirs = contract.get("variant_source_dirs", {})
    if not isinstance(variant_source_dirs, Mapping):
        raise ValueError("Phase 63 contract is missing variant_source_dirs")
    variants = [str(value) for value in contract.get("variants", [])]
    train_tile_id = str(contract.get("train_tile_id", ""))
    eval_tile_ids = [str(value) for value in contract.get("eval_tile_ids", [])]
    tile_index_csv = str(contract.get("tile_index_csv", ""))
    tiled_inputs: list[tuple[str, object]] = []
    for variant_id in variants:
        source_dir = variant_source_dirs.get(variant_id)
        if source_dir is None:
            raise ValueError(f"Phase 63 contract has no source for variant {variant_id}")
        tiled_inputs.append(
            (
                "train",
                load_tiled_variant_input(
                    source_dir,
                    tile_index_csv,
                    train_tile_id,
                    variant_id=variant_id,
                ),
            )
        )
        for eval_tile_id in eval_tile_ids:
            tiled_inputs.append(
                (
                    "eval",
                    load_tiled_variant_input(
                        source_dir,
                        tile_index_csv,
                        eval_tile_id,
                        variant_id=variant_id,
                    ),
                )
            )
    return tiled_inputs


def _tiled_input_index(tiled_inputs: Sequence[tuple[str, object]]) -> dict[tuple[str, str], object]:
    return {
        (str(tiled_input.variant_id), str(tiled_input.tile_id)): tiled_input
        for _role, tiled_input in tiled_inputs
    }


def run_phase64_set_policy_error_diagnosis(
    phase63_comparison_json: Path | str,
    phase63_rollout_csv: Path | str,
    phase63_history_csv: Path | str,
    phase63_oracle_summary_csv: Path | str,
) -> dict[str, object]:
    comparison = _load_json_object(phase63_comparison_json, "Phase 63 comparison JSON")
    contract = comparison.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Phase 63 comparison JSON is missing contract metadata")
    train_tile_id = str(contract.get("train_tile_id", ""))
    variants = [str(value) for value in contract.get("variants", [])]
    train_tile_ids = {variant_id: train_tile_id for variant_id in variants}
    tiled_inputs = _contract_tiled_inputs(contract)
    tiled_index = _tiled_input_index(tiled_inputs)
    convergence_rows = build_phase64_convergence_summary(phase63_history_csv)
    overlap_rows = build_phase64_rollout_overlap(phase63_rollout_csv, phase63_oracle_summary_csv)
    oracle_rank_gap_rows = build_phase64_oracle_rank_gap(phase63_rollout_csv, tiled_index)
    feature_diagnostics = build_phase64_feature_diagnostics(tiled_inputs, train_tile_ids)
    standardization_gate = build_phase64_standardization_gate(
        comparison,
        convergence_rows,
        feature_diagnostics["feature_effective_rank_rows"],
    )
    failure_case_rows = build_phase64_failure_cases(
        comparison,
        overlap_rows,
        oracle_rank_gap_rows,
        convergence_rows,
        feature_diagnostics["feature_effective_rank_rows"],
    )
    return {
        "phase": "phase64_set_policy_error_diagnosis",
        "phase63_comparison_json": str(Path(phase63_comparison_json)),
        "phase63_rollout_csv": str(Path(phase63_rollout_csv)),
        "phase63_history_csv": str(Path(phase63_history_csv)),
        "phase63_oracle_summary_csv": str(Path(phase63_oracle_summary_csv)),
        "contract": dict(contract),
        "phase63_comparison": comparison,
        "convergence_rows": convergence_rows,
        "overlap_rows": overlap_rows,
        "oracle_rank_gap_rows": oracle_rank_gap_rows,
        "feature_scale_rows": feature_diagnostics["feature_scale_rows"],
        "feature_effective_rank_rows": feature_diagnostics["feature_effective_rank_rows"],
        "failure_case_rows": failure_case_rows,
        "standardization_gate": standardization_gate,
        "claim_boundary": PHASE64_CLAIM_BOUNDARY,
    }
