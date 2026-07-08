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
