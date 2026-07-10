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


def _reward_view(prepared: Phase71PreparedTile) -> TiledVariantInput:
    return TiledVariantInput(
        tile_id=prepared.tiled_input.tile_id,
        variant_id=prepared.tiled_input.variant_id,
        block_ids=prepared.tiled_input.block_ids,
        feature_columns=prepared.tiled_input.feature_columns,
        state_matrix=prepared.reward_matrix.astype(np.float32, copy=True),
        reward_mode=prepared.tiled_input.reward_mode,
        state_groups=prepared.tiled_input.state_groups,
        source_table=prepared.tiled_input.source_table,
        tile_index_csv=prepared.tiled_input.tile_index_csv,
        claim_boundary=prepared.tiled_input.claim_boundary,
    )


def build_phase71_oracle_trajectory(
    prepared: Phase71PreparedTile,
    eval_max_steps: int,
) -> dict[str, object]:
    oracle = build_phase63_oracle_trajectory(_reward_view(prepared), eval_max_steps)
    oracle["claim_boundary"] = PHASE71_CLAIM_BOUNDARY
    oracle["phase71_component_supervised"] = True
    return oracle


def build_phase71_listwise_training_tile(
    prepared: Phase71PreparedTile,
) -> dict[str, object]:
    target_rows = build_phase71_component_targets(_reward_view(prepared))
    reward_targets = np.asarray(
        [float(row["reward_total"]) for row in target_rows],
        dtype=np.float32,
    )
    component_targets = np.asarray(
        [
            [float(row["components"][name]) for name in PHASE71_COMPONENT_NAMES]
            for row in target_rows
        ],
        dtype=np.float32,
    )
    return {
        "variant_id": str(prepared.tiled_input.variant_id),
        "tile_id": str(prepared.tiled_input.tile_id),
        "block_ids": tuple(prepared.tiled_input.block_ids),
        "model_matrix": prepared.model_matrix.astype(np.float32, copy=True),
        "reward_targets": reward_targets,
        "component_targets": component_targets,
    }


class Phase71ComponentRanker(__import__("torch").nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        n_components: int = 8,
    ) -> None:
        from torch import nn

        super().__init__()
        if int(n_features) <= 0:
            raise ValueError("n_features must be positive")
        self.encoder = nn.Sequential(
            nn.Linear(int(n_features), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.score_head = nn.Linear(int(hidden_dim), 1)
        self.component_head = nn.Linear(int(hidden_dim), int(n_components))

    def forward(self, block_features):
        encoded = self.encoder(block_features)
        return self.score_head(encoded).squeeze(-1), self.component_head(encoded)


def train_phase71_component_ranker(
    prepared_training_tiles: Sequence[Phase71PreparedTile],
    seed: int,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    component_weight: float = 0.05,
    top_k: int = 3,
    device: str = "cpu",
):
    import torch
    import torch.nn.functional as F

    if not prepared_training_tiles:
        raise ValueError("Phase 71 training requires at least one prepared training tile")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    examples = [
        build_phase71_listwise_training_tile(tile) for tile in prepared_training_tiles
    ]
    model = Phase71ComponentRanker(
        len(prepared_training_tiles[0].tiled_input.feature_columns),
        int(hidden_dim),
        len(PHASE71_COMPONENT_NAMES),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    history: list[dict[str, object]] = []
    for epoch in range(1, int(epochs) + 1):
        losses = []
        top1_hits = 0
        topk_hits = 0
        for example in examples:
            features = torch.tensor(
                example["model_matrix"],
                dtype=torch.float32,
                device=device,
            )
            reward_targets = torch.tensor(
                example["reward_targets"],
                dtype=torch.float32,
                device=device,
            )
            component_targets = torch.tensor(
                example["component_targets"],
                dtype=torch.float32,
                device=device,
            )
            target_probs = torch.softmax(reward_targets, dim=0)
            scores, component_predictions = model(features)
            listwise_loss = -(target_probs * torch.log_softmax(scores, dim=0)).sum()
            component_loss = F.mse_loss(component_predictions, component_targets)
            loss = listwise_loss + float(component_weight) * component_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            best_target = int(torch.argmax(reward_targets).item())
            predicted_order = torch.argsort(
                scores.detach(),
                descending=True,
            ).cpu().tolist()
            top1_hits += int(predicted_order[0] == best_target)
            topk_hits += int(
                best_target in predicted_order[: min(int(top_k), len(predicted_order))]
            )
        history.append(
            {
                "variant_id": str(prepared_training_tiles[0].tiled_input.variant_id),
                "train_tile_ids": ";".join(
                    str(tile.tiled_input.tile_id) for tile in prepared_training_tiles
                ),
                "seed": int(seed),
                "epoch": int(epoch),
                "loss": _round_float(statistics.mean(losses)),
                "top1_accuracy": _round_float(top1_hits / len(examples)),
                "topk_hit_rate": _round_float(topk_hits / len(examples)),
                "learning_rate": float(learning_rate),
                "hidden_dim": int(hidden_dim),
                "component_weight": float(component_weight),
                "phase71_component_supervised": True,
                "claim_boundary": PHASE71_CLAIM_BOUNDARY,
            }
        )
    model.eval()
    return model, history


def rollout_phase71_ranker(
    model,
    prepared_eval_tile: Phase71PreparedTile,
    train_tile_ids: Sequence[str],
    eval_tile_rank: int,
    seed: int,
    phase71_seed_rank: int,
    eval_max_steps: int,
    device: str = "cpu",
) -> dict[str, object]:
    import torch

    with torch.no_grad():
        features = torch.tensor(
            prepared_eval_tile.model_matrix,
            dtype=torch.float32,
            device=device,
        )
        scores, _components = model(features)
        score_values = [float(value) for value in scores.detach().cpu().tolist()]
    ranked_indices = sorted(
        range(len(score_values)),
        key=lambda index: (
            -score_values[index],
            str(prepared_eval_tile.tiled_input.block_ids[index]),
            index,
        ),
    )
    selected = ranked_indices[: min(int(eval_max_steps), len(ranked_indices))]
    rewards = [
        compute_base_planning_reward_from_matrix_row(
            prepared_eval_tile.tiled_input.feature_columns,
            prepared_eval_tile.reward_matrix[index],
        )
        for index in selected
    ]
    selected_block_ids = [
        str(prepared_eval_tile.tiled_input.block_ids[index]) for index in selected
    ]
    oracle = build_phase71_oracle_trajectory(prepared_eval_tile, int(eval_max_steps))
    oracle_total = float(oracle["total_oracle_reward"])
    total_reward = _round_float(sum(rewards))
    oracle_gap = _round_float(oracle_total - total_reward)
    oracle_blocks = [str(value) for value in oracle.get("selected_block_ids", [])]
    overlap = set(selected_block_ids).intersection(oracle_blocks)
    oracle_rank = {block_id: rank for rank, block_id in enumerate(oracle_blocks, start=1)}
    worst_rank = max(
        (oracle_rank.get(block_id, 999999) for block_id in selected_block_ids),
        default=0,
    )
    terminated = len(selected) == len(prepared_eval_tile.tiled_input.block_ids)
    return {
        "row_type": "component_ranker_policy",
        "variant_id": str(prepared_eval_tile.tiled_input.variant_id),
        "train_tile_ids": ";".join(str(value) for value in train_tile_ids),
        "eval_tile_id": str(prepared_eval_tile.tiled_input.tile_id),
        "eval_tile_rank": int(eval_tile_rank),
        "seed": int(seed),
        "phase71_seed_rank": int(phase71_seed_rank),
        "eval_max_steps": int(eval_max_steps),
        "n_blocks": len(prepared_eval_tile.tiled_input.block_ids),
        "n_features": len(prepared_eval_tile.tiled_input.feature_columns),
        "episode_steps": len(selected),
        "terminated": bool(terminated),
        "truncated": not terminated,
        "all_actions_valid": True,
        "invalid_action_count": 0,
        "total_contract_reward": total_reward,
        "oracle_total_reward": _round_float(oracle_total),
        "oracle_gap": oracle_gap,
        "oracle_gap_fraction": _round_float(
            oracle_gap / max(abs(oracle_total), 1.0e-9)
        ),
        "topk_oracle_overlap_count": len(overlap),
        "topk_oracle_overlap_fraction": _round_float(
            len(overlap) / max(len(oracle_blocks), 1)
        ),
        "worst_selected_oracle_rank": int(worst_rank),
        "selected_block_ids": ";".join(selected_block_ids),
        "selected_action_indices": ";".join(str(index) for index in selected),
        "selected_model_scores": ";".join(
            str(_round_float(score_values[index])) for index in selected
        ),
        "phase71_component_supervised": True,
        "claim_boundary": PHASE71_CLAIM_BOUNDARY,
    }


PHASE71_HISTORY_FIELDNAMES = (
    "variant_id",
    "train_tile_ids",
    "seed",
    "epoch",
    "loss",
    "top1_accuracy",
    "topk_hit_rate",
    "learning_rate",
    "hidden_dim",
    "component_weight",
    "phase71_component_supervised",
    "claim_boundary",
)
PHASE71_ROLLOUT_FIELDNAMES = (
    "row_type",
    "variant_id",
    "train_tile_ids",
    "eval_tile_id",
    "eval_tile_rank",
    "seed",
    "phase71_seed_rank",
    "eval_max_steps",
    "n_blocks",
    "n_features",
    "episode_steps",
    "terminated",
    "truncated",
    "all_actions_valid",
    "invalid_action_count",
    "total_contract_reward",
    "oracle_total_reward",
    "oracle_gap",
    "oracle_gap_fraction",
    "topk_oracle_overlap_count",
    "topk_oracle_overlap_fraction",
    "worst_selected_oracle_rank",
    "selected_block_ids",
    "selected_action_indices",
    "selected_model_scores",
    "phase71_component_supervised",
    "claim_boundary",
)
PHASE71_ORACLE_FIELDNAMES = (
    "variant_id",
    "tile_role",
    "tile_id",
    "seed",
    "eval_max_steps",
    "n_blocks",
    "n_features",
    "episode_steps",
    "terminated",
    "total_oracle_reward",
    "top_k_reward_ceiling",
    "selected_block_ids",
    "action_indices",
    "claim_boundary",
)
PHASE71_COMPONENT_FIELDNAMES = (
    "variant_id",
    "tile_id",
    "block_id",
    "action_index",
    "reward_total",
    "component_name",
    "component_value",
    "claim_boundary",
)
PHASE71_DELTA_FIELDNAMES = (
    "reference_phase",
    "variant_id",
    "eval_tile_id",
    "seed",
    "phase71_reward",
    "reference_reward",
    "phase71_minus_reference_reward",
    "phase71_oracle_gap_fraction",
    "reference_oracle_gap_fraction",
    "claim_boundary",
)


def _rollout_key(row: Mapping[str, object]) -> tuple[str, str, int]:
    return (
        str(row.get("variant_id", "")),
        str(row.get("eval_tile_id", row.get("tile_id", ""))),
        int(float(row.get("seed", 0))),
    )


def _coverage_issues(
    rows: Sequence[Mapping[str, object]],
    variants: Sequence[str],
    eval_tile_ids: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, list[dict[str, object]]]:
    counts: dict[tuple[str, str, int], int] = {}
    for row in rows:
        if str(row.get("row_type", "")) != "component_ranker_policy":
            continue
        key = _rollout_key(row)
        counts[key] = counts.get(key, 0) + 1
    expected = {
        (str(variant), str(tile_id), int(seed))
        for variant in variants
        for tile_id in eval_tile_ids
        for seed in seeds
    }
    observed = set(counts)
    return {
        "missing_rollout_rows": [
            {"variant_id": key[0], "eval_tile_id": key[1], "seed": key[2]}
            for key in sorted(expected - observed)
        ],
        "unexpected_rollout_rows": [
            {"variant_id": key[0], "eval_tile_id": key[1], "seed": key[2]}
            for key in sorted(observed - expected)
        ],
        "duplicate_rollout_rows": [
            {
                "variant_id": key[0],
                "eval_tile_id": key[1],
                "seed": key[2],
                "count": count,
            }
            for key, count in sorted(counts.items())
            if count > 1
        ],
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, object]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "mean_delta": 0.0,
            "positive_count": 0,
            "total_count": 0,
            "min_delta": 0.0,
            "max_delta": 0.0,
        }
    return {
        "mean_delta": _round_float(statistics.mean(numbers)),
        "positive_count": sum(1 for value in numbers if value > 0.0),
        "total_count": len(numbers),
        "min_delta": _round_float(min(numbers)),
        "max_delta": _round_float(max(numbers)),
    }


def _mean_by_variant(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if str(row.get(field, "")).strip() == "":
            continue
        grouped.setdefault(str(row.get("variant_id", "")), []).append(float(row[field]))
    return {
        key: _round_float(statistics.mean(values))
        for key, values in sorted(grouped.items())
    }


def _delta_rows(
    reference_phase: str,
    phase71_rows: Sequence[Mapping[str, object]],
    reference_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    baseline = {
        _rollout_key(row): row
        for row in reference_rows
        if str(row.get("total_contract_reward", "")).strip() != ""
    }
    rows = []
    for row in phase71_rows:
        if str(row.get("row_type", "")) != "component_ranker_policy":
            continue
        key = _rollout_key(row)
        old = baseline.get(key)
        if old is None:
            continue
        phase71_reward = float(row.get("total_contract_reward", 0.0))
        reference_reward = float(old.get("total_contract_reward", 0.0))
        rows.append(
            {
                "reference_phase": reference_phase,
                "variant_id": key[0],
                "eval_tile_id": key[1],
                "seed": key[2],
                "phase71_reward": _round_float(phase71_reward),
                "reference_reward": _round_float(reference_reward),
                "phase71_minus_reference_reward": _round_float(
                    phase71_reward - reference_reward
                ),
                "phase71_oracle_gap_fraction": _round_float(
                    row.get("oracle_gap_fraction", 0.0)
                ),
                "reference_oracle_gap_fraction": _round_float(
                    old.get("oracle_gap_fraction", 0.0)
                ),
                "claim_boundary": PHASE71_CLAIM_BOUNDARY,
            }
        )
    return rows


def _paired_delta_summary(
    rows: Sequence[Mapping[str, object]],
    pairs: Sequence[tuple[str, str]],
) -> dict[str, object]:
    index = {
        _rollout_key(row): float(row.get("total_contract_reward", 0.0))
        for row in rows
        if str(row.get("row_type", "")) == "component_ranker_policy"
    }
    values = []
    tile_seed_keys = sorted({(key[1], key[2]) for key in index})
    for left, right in pairs:
        for tile_id, seed in tile_seed_keys:
            left_key = (left, tile_id, seed)
            right_key = (right, tile_id, seed)
            if left_key in index and right_key in index:
                values.append(index[left_key] - index[right_key])
    return _numeric_summary(values)


def _positive_majority(summary: Mapping[str, object]) -> bool:
    total = int(summary.get("total_count", 0))
    return total > 0 and int(summary.get("positive_count", 0)) * 2 >= total


def _phase71_status(
    coverage: Mapping[str, object],
    phase63_summary: Mapping[str, object],
    phase70_summary: Mapping[str, object],
    d4_b0: Mapping[str, object],
    d4_d6: Mapping[str, object],
) -> str:
    if (
        coverage["missing_rollout_rows"]
        or coverage["duplicate_rollout_rows"]
        or coverage["unexpected_rollout_rows"]
    ):
        return PHASE71_STATUS_INCOMPLETE
    decision_improved = (
        float(phase63_summary["mean_delta"]) > 0.0
        and _positive_majority(phase63_summary)
        and float(phase70_summary["mean_delta"]) > 0.0
        and _positive_majority(phase70_summary)
    )
    geofm_improved = (
        float(d4_b0["mean_delta"]) > 0.0
        and _positive_majority(d4_b0)
        and float(d4_d6["mean_delta"]) > 0.0
        and _positive_majority(d4_d6)
    )
    if decision_improved and geofm_improved:
        return PHASE71_STATUS_GEOFM
    if decision_improved and d4_b0["total_count"] and d4_d6["total_count"]:
        return PHASE71_STATUS_TARGET_MASKS_GEOFM
    if decision_improved:
        return PHASE71_STATUS_DECISION
    return PHASE71_STATUS_NOT_SUFFICIENT


def build_phase71_component_ranker_comparison(
    phase71_rollout_rows: Sequence[Mapping[str, object]],
    phase63_rollout_rows: Sequence[Mapping[str, object]],
    phase70_rollout_rows: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    metadata = dict(metadata or {})
    variants = [str(value) for value in metadata.get("variants", [])] or sorted(
        _mean_by_variant(phase71_rollout_rows, "total_contract_reward")
    )
    eval_tile_ids = [str(value) for value in metadata.get("eval_tile_ids", [])] or sorted(
        {str(row.get("eval_tile_id", "")) for row in phase71_rollout_rows}
    )
    seeds = [int(value) for value in metadata.get("seeds", [])] or sorted(
        {int(float(row.get("seed", 0))) for row in phase71_rollout_rows}
    )
    coverage = _coverage_issues(phase71_rollout_rows, variants, eval_tile_ids, seeds)
    phase63_delta_rows = _delta_rows("phase63", phase71_rollout_rows, phase63_rollout_rows)
    phase70_delta_rows = _delta_rows("phase70", phase71_rollout_rows, phase70_rollout_rows)
    phase63_summary = _numeric_summary(
        [float(row["phase71_minus_reference_reward"]) for row in phase63_delta_rows]
    )
    phase70_summary = _numeric_summary(
        [float(row["phase71_minus_reference_reward"]) for row in phase70_delta_rows]
    )
    d4_b0 = _paired_delta_summary(
        phase71_rollout_rows,
        (("D4P8", "B0"), ("D4P16", "B0")),
    )
    d4_d6 = _paired_delta_summary(
        phase71_rollout_rows,
        (("D4P8", "D6R8"), ("D4P16", "D6R16")),
    )
    status = _phase71_status(coverage, phase63_summary, phase70_summary, d4_b0, d4_d6)
    return {
        "phase": "phase71_component_supervised_ranker",
        "phase71_status": status,
        "variants": variants,
        "eval_tile_ids": eval_tile_ids,
        "seeds": seeds,
        "coverage_issues": coverage,
        "mean_phase71_reward_by_variant": _mean_by_variant(
            phase71_rollout_rows,
            "total_contract_reward",
        ),
        "mean_phase71_minus_phase63_by_variant": _mean_by_variant(
            phase63_delta_rows,
            "phase71_minus_reference_reward",
        ),
        "mean_phase71_minus_phase70_by_variant": _mean_by_variant(
            phase70_delta_rows,
            "phase71_minus_reference_reward",
        ),
        "phase71_minus_phase63_summary": phase63_summary,
        "phase71_minus_phase70_summary": phase70_summary,
        "d4_b0_delta_summary": d4_b0,
        "d4_d6_delta_summary": d4_d6,
        "delta_rows": [*phase63_delta_rows, *phase70_delta_rows],
        "recommended_next_step": _phase71_next_step(status),
        "claim_boundary": PHASE71_CLAIM_BOUNDARY,
    }


def _phase71_next_step(status: str) -> str:
    if status == PHASE71_STATUS_GEOFM:
        return "Use Phase 71 as the next bounded algorithm evidence gate and design a GeoFM follow-up without broad suitability claims."
    if status == PHASE71_STATUS_TARGET_MASKS_GEOFM:
        return "Keep Phase 71 as a stronger decision-learning baseline, but treat the explicit base target as masking GeoFM-specific value."
    if status == PHASE71_STATUS_DECISION:
        return "Use Phase 71 as an algorithm baseline and run targeted attribution before changing manuscript claims."
    if status == PHASE71_STATUS_INCOMPLETE:
        return "Repair missing or duplicated Phase 71 coverage before interpreting results."
    return "Treat the component-supervised ranker as insufficient and design a different algorithm route."


def write_phase71_component_ranker_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "history_csv": output_path / "phase71_ranker_training_history.csv",
        "rollout_csv": output_path / "phase71_ranker_rollout_summary.csv",
        "oracle_summary_csv": output_path / "phase71_ranker_oracle_summary.csv",
        "component_diagnostics_csv": output_path
        / "phase71_ranker_component_diagnostics.csv",
        "delta_csv": output_path / "phase71_ranker_delta_table.csv",
        "comparison_json": output_path / "phase71_component_supervised_ranker.json",
        "readiness_md": output_path / "phase71_component_supervised_ranker.md",
    }
    _write_csv_rows(
        paths["history_csv"],
        PHASE71_HISTORY_FIELDNAMES,
        analysis.get("history_rows", []),
    )
    _write_csv_rows(
        paths["rollout_csv"],
        PHASE71_ROLLOUT_FIELDNAMES,
        analysis.get("rollout_rows", []),
    )
    _write_csv_rows(
        paths["oracle_summary_csv"],
        PHASE71_ORACLE_FIELDNAMES,
        analysis.get("oracle_summary_rows", []),
    )
    _write_csv_rows(
        paths["component_diagnostics_csv"],
        PHASE71_COMPONENT_FIELDNAMES,
        analysis.get("component_diagnostic_rows", []),
    )
    _write_csv_rows(
        paths["delta_csv"],
        PHASE71_DELTA_FIELDNAMES,
        analysis.get("delta_rows", []),
    )
    comparison = {
        key: value
        for key, value in analysis.items()
        if key
        not in {
            "history_rows",
            "rollout_rows",
            "oracle_summary_rows",
            "component_diagnostic_rows",
        }
    }
    paths["comparison_json"].write_text(
        json.dumps(_json_ready(comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["readiness_md"].write_text(_phase71_markdown(analysis), encoding="utf-8")
    return paths


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: object) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 71 rows must be a list for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 71 CSV rows must be objects for {path.name}")
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _phase71_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Phase 71 Component-Supervised Listwise Ranker",
        "",
        f"Status: {analysis.get('phase71_status', '')}",
        "",
        "Mean Phase 71 reward by variant:",
    ]
    for variant_id, value in dict(
        analysis.get("mean_phase71_reward_by_variant", {})
    ).items():
        lines.append(f"- {variant_id}: {value}")
    lines.extend(
        [
            "",
            f"Phase 71 minus Phase 63 summary: {analysis.get('phase71_minus_phase63_summary', {})}",
            f"Phase 71 minus Phase 70 summary: {analysis.get('phase71_minus_phase70_summary', {})}",
            f"D4/B0 delta summary: {analysis.get('d4_b0_delta_summary', {})}",
            f"D4/D6 delta summary: {analysis.get('d4_d6_delta_summary', {})}",
            "",
            "Recommended next step:",
            str(analysis.get("recommended_next_step", "")),
            "",
            "Claim boundary:",
            str(analysis.get("claim_boundary", PHASE71_CLAIM_BOUNDARY)),
            "",
        ]
    )
    return "\n".join(lines)


def _load_csv_rows(path: Path | str, label: str) -> list[dict[str, object]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {label}: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_phase71_tiled_variant_input(
    contract: Mapping[str, object],
    tile_id: str,
    variant_id: str,
) -> TiledVariantInput:
    variant_source_dirs = contract.get("variant_source_dirs")
    if not isinstance(variant_source_dirs, Mapping):
        raise ValueError("Phase 71 contract is missing variant source routing")
    source_dir = variant_source_dirs.get(variant_id)
    if source_dir is None:
        raise ValueError(f"Phase 71 contract has no source for variant {variant_id}")
    return load_tiled_variant_input(
        source_dir,
        str(contract["tile_index_csv"]),
        tile_id,
        variant_id=variant_id,
    )


def _phase71_fold_train_tile_ids(
    contract: Mapping[str, object],
    held_out_eval_tile_id: str,
) -> list[str]:
    ids = [str(contract["train_tile_id"])]
    ids.extend(
        str(tile_id)
        for tile_id in contract.get("eval_tile_ids", [])
        if str(tile_id) != str(held_out_eval_tile_id)
    )
    return list(dict.fromkeys(ids))


def _component_diagnostic_rows(tiled_input: TiledVariantInput) -> list[dict[str, object]]:
    rows = []
    for target in build_phase71_component_targets(tiled_input):
        for component_name, component_value in dict(target["components"]).items():
            rows.append(
                {
                    "variant_id": target["variant_id"],
                    "tile_id": target["tile_id"],
                    "block_id": target["block_id"],
                    "action_index": target["action_index"],
                    "reward_total": target["reward_total"],
                    "component_name": component_name,
                    "component_value": component_value,
                    "claim_boundary": PHASE71_CLAIM_BOUNDARY,
                }
            )
    return rows


def run_phase71_component_supervised_ranker(
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str,
    phase61_output_dir: Path | str,
    tile_index_csv: Path | str,
    phase63_rollout_csv: Path | str,
    phase70_rollout_csv: Path | str,
    variants: Sequence[str] | str = PHASE63_DEFAULT_VARIANTS,
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 5,
    eval_max_steps: int = 8,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
    ranker_epochs: int = 80,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    component_weight: float = 0.05,
    top_k: int = 3,
) -> dict[str, object]:
    contract = build_phase63_set_policy_contract(
        phase2_output_dir=phase2_output_dir,
        phase8_output_dir=phase8_output_dir,
        phase61_output_dir=phase61_output_dir,
        tile_index_csv=tile_index_csv,
        variants=variants,
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
        eval_max_steps=eval_max_steps,
        seeds=seeds,
        bc_epochs=ranker_epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        top_k=top_k,
    )
    contract["phase"] = "phase71_component_supervised_ranker"
    contract["ranker_epochs"] = int(ranker_epochs)
    contract["component_weight"] = float(component_weight)
    history_rows: list[dict[str, object]] = []
    rollout_rows: list[dict[str, object]] = []
    oracle_summary_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    for variant_id in contract["variants"]:
        variant = str(variant_id)
        for eval_tile_id in contract["eval_tile_ids"]:
            eval_id = str(eval_tile_id)
            train_tile_ids = _phase71_fold_train_tile_ids(contract, eval_id)
            raw_training_tiles = [
                _load_phase71_tiled_variant_input(contract, tile_id, variant)
                for tile_id in train_tile_ids
            ]
            params = fit_phase71_fold_standardization(
                raw_training_tiles,
                variant_id=variant,
                fold_id=eval_id,
            )
            prepared_training_tiles = [
                apply_phase71_fold_standardization(tile, params)
                for tile in raw_training_tiles
            ]
            raw_eval_tile = _load_phase71_tiled_variant_input(contract, eval_id, variant)
            prepared_eval = apply_phase71_fold_standardization(raw_eval_tile, params)
            component_rows.extend(_component_diagnostic_rows(raw_eval_tile))
            for seed in contract["seeds"]:
                model, history = train_phase71_component_ranker(
                    prepared_training_tiles,
                    seed=int(seed),
                    epochs=int(ranker_epochs),
                    learning_rate=float(learning_rate),
                    hidden_dim=int(hidden_dim),
                    component_weight=float(component_weight),
                    top_k=int(top_k),
                )
                history_rows.extend(history)
                oracle = build_phase71_oracle_trajectory(prepared_eval, int(eval_max_steps))
                oracle_row = _phase63_oracle_summary_row(
                    oracle,
                    seed=int(seed),
                    tile_role="eval",
                )
                oracle_row["claim_boundary"] = PHASE71_CLAIM_BOUNDARY
                oracle_summary_rows.append(oracle_row)
                rollout_rows.append(
                    rollout_phase71_ranker(
                        model,
                        prepared_eval,
                        train_tile_ids=train_tile_ids,
                        eval_tile_rank=int(contract["eval_tile_ranks"][eval_id]),
                        seed=int(seed),
                        phase71_seed_rank=int(contract["seed_ranks"][str(int(seed))]),
                        eval_max_steps=int(eval_max_steps),
                    )
                )
    phase63_rows = _load_csv_rows(phase63_rollout_csv, "Phase 63 rollout CSV")
    phase70_rows = _load_csv_rows(phase70_rollout_csv, "Phase 70 rollout CSV")
    analysis = build_phase71_component_ranker_comparison(
        rollout_rows,
        phase63_rows,
        phase70_rows,
        metadata={
            "variants": contract["variants"],
            "eval_tile_ids": contract["eval_tile_ids"],
            "seeds": contract["seeds"],
        },
    )
    analysis["contract"] = contract
    analysis["history_rows"] = history_rows
    analysis["rollout_rows"] = rollout_rows
    analysis["oracle_summary_rows"] = oracle_summary_rows
    analysis["component_diagnostic_rows"] = component_rows
    return analysis
