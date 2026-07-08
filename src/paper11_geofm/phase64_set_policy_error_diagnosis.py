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
