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
