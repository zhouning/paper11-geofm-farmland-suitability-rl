from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import statistics

import numpy as np

from paper11_geofm.phase63_set_policy_oracle_pretraining import (
    Phase63SetPolicyScorer,
    _round_float,
    build_phase63_model_inputs,
    build_phase63_oracle_trajectory,
)
from paper11_geofm.planning_reward import compute_base_planning_reward_from_matrix_row
from paper11_geofm.tiled_inputs import TiledVariantInput


PHASE70_CLAIM_BOUNDARY = (
    "Phase 70 is a standardized set-policy rerun under the existing Bishan "
    "base-reward protocol. It standardizes model inputs with train-tile-fitted "
    "parameters while preserving original features for reward and oracle "
    "scoring. It does not alter rewards, enable B2/B3, validate suitability, "
    "prove PCA optimality, or justify formal submission-level claims."
)

PHASE70_STATUS_GEOFM = "standardization_improves_geofm_set_policy_route"
PHASE70_STATUS_ARCHITECTURE = "standardization_improves_architecture_not_geofm"
PHASE70_STATUS_NOT_SUFFICIENT = "standardization_not_sufficient"
PHASE70_STATUS_INCOMPLETE = "standardized_rerun_incomplete"


@dataclass(frozen=True)
class Phase70StandardizedTiledInput:
    tiled_input: TiledVariantInput
    model_matrix: np.ndarray
    reward_matrix: np.ndarray
    standardization: Mapping[str, object]


def fit_phase70_standardization(tiled_input: TiledVariantInput) -> dict[str, object]:
    matrix = np.asarray(tiled_input.state_matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Phase 70 standardization requires a non-empty 2D matrix")
    means = np.nanmean(matrix, axis=0)
    scales = np.nanstd(matrix, axis=0)
    safe_scales = np.where(np.isfinite(scales) & (np.abs(scales) >= 1.0e-8), scales, 1.0)
    return {
        "variant_id": str(tiled_input.variant_id),
        "tile_id": str(tiled_input.tile_id),
        "feature_columns": list(tiled_input.feature_columns),
        "means": [_round_float(value) for value in means.tolist()],
        "scales": [_round_float(value) for value in safe_scales.tolist()],
        "claim_boundary": PHASE70_CLAIM_BOUNDARY,
    }


def apply_phase70_standardization(
    tiled_input: TiledVariantInput,
    params: Mapping[str, object],
) -> Phase70StandardizedTiledInput:
    feature_columns = tuple(str(value) for value in params.get("feature_columns", []))
    if feature_columns != tuple(tiled_input.feature_columns):
        raise ValueError("Phase 70 standardization feature columns do not match input")
    matrix = np.asarray(tiled_input.state_matrix, dtype=np.float32)
    means = np.asarray(params.get("means", []), dtype=np.float32)
    scales = np.asarray(params.get("scales", []), dtype=np.float32)
    if means.shape[0] != matrix.shape[1] or scales.shape[0] != matrix.shape[1]:
        raise ValueError("Phase 70 standardization parameter length does not match input")
    safe_scales = np.where(np.isfinite(scales) & (np.abs(scales) >= 1.0e-8), scales, 1.0)
    model_matrix = (matrix - means) / safe_scales
    model_matrix = np.nan_to_num(model_matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32,
        copy=False,
    )
    return Phase70StandardizedTiledInput(
        tiled_input=tiled_input,
        model_matrix=model_matrix,
        reward_matrix=matrix.astype(np.float32, copy=True),
        standardization=dict(params),
    )


def _model_view(tiled: Phase70StandardizedTiledInput) -> TiledVariantInput:
    return TiledVariantInput(
        tile_id=tiled.tiled_input.tile_id,
        variant_id=tiled.tiled_input.variant_id,
        block_ids=tiled.tiled_input.block_ids,
        feature_columns=tiled.tiled_input.feature_columns,
        state_matrix=tiled.model_matrix.astype(np.float32, copy=True),
        reward_mode=tiled.tiled_input.reward_mode,
        state_groups=tiled.tiled_input.state_groups,
        source_table=tiled.tiled_input.source_table,
        tile_index_csv=tiled.tiled_input.tile_index_csv,
        claim_boundary=tiled.tiled_input.claim_boundary,
    )


def _reward_view(tiled: Phase70StandardizedTiledInput) -> TiledVariantInput:
    return TiledVariantInput(
        tile_id=tiled.tiled_input.tile_id,
        variant_id=tiled.tiled_input.variant_id,
        block_ids=tiled.tiled_input.block_ids,
        feature_columns=tiled.tiled_input.feature_columns,
        state_matrix=tiled.reward_matrix.astype(np.float32, copy=True),
        reward_mode=tiled.tiled_input.reward_mode,
        state_groups=tiled.tiled_input.state_groups,
        source_table=tiled.tiled_input.source_table,
        tile_index_csv=tiled.tiled_input.tile_index_csv,
        claim_boundary=tiled.tiled_input.claim_boundary,
    )


def build_phase70_oracle_trajectory(
    standardized_input: Phase70StandardizedTiledInput,
    eval_max_steps: int,
) -> dict[str, object]:
    oracle = build_phase63_oracle_trajectory(_reward_view(standardized_input), eval_max_steps)
    oracle["claim_boundary"] = PHASE70_CLAIM_BOUNDARY
    oracle["phase70_standardized_input"] = True
    return oracle


def _phase70_model_inputs(
    standardized_input: Phase70StandardizedTiledInput,
    selected_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    return build_phase63_model_inputs(_model_view(standardized_input), selected_indices)


def build_phase70_bc_examples(
    standardized_input: Phase70StandardizedTiledInput,
    eval_max_steps: int,
) -> list[dict[str, object]]:
    trajectory = build_phase70_oracle_trajectory(standardized_input, eval_max_steps)
    examples: list[dict[str, object]] = []
    selected: list[int] = []
    for step in trajectory["steps"]:
        action_index = int(step["action_index"])
        inputs = _phase70_model_inputs(standardized_input, selected)
        examples.append(
            {
                "block_features": inputs["block_features"],
                "valid_mask": inputs["valid_mask"],
                "selected_mask": inputs["selected_mask"],
                "target_action": action_index,
            }
        )
        selected.append(action_index)
    return examples


def train_phase70_standardized_behavior_cloner(
    standardized_input: Phase70StandardizedTiledInput,
    seed: int,
    eval_max_steps: int,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    top_k: int = 3,
    device: str = "cpu",
):
    import random
    import torch
    import torch.nn.functional as F

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    examples = build_phase70_bc_examples(standardized_input, eval_max_steps)
    if not examples:
        raise ValueError("Phase 70 behavior cloning requires at least one example")
    model = Phase63SetPolicyScorer(
        n_features=len(standardized_input.tiled_input.feature_columns),
        hidden_dim=int(hidden_dim),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    history: list[dict[str, object]] = []
    for epoch in range(1, int(epochs) + 1):
        losses = []
        correct = 0
        topk_hits = 0
        for example in examples:
            block_features = torch.tensor(
                example["block_features"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            valid_mask = torch.tensor(
                example["valid_mask"], dtype=torch.bool, device=device
            ).unsqueeze(0)
            selected_mask = torch.tensor(
                example["selected_mask"], dtype=torch.bool, device=device
            ).unsqueeze(0)
            target = torch.tensor(
                [int(example["target_action"])], dtype=torch.long, device=device
            )
            logits = model(block_features, valid_mask, selected_mask)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            predicted = int(torch.argmax(logits.detach(), dim=1).item())
            correct += int(predicted == int(example["target_action"]))
            k = min(int(top_k), logits.shape[1])
            topk = torch.topk(logits.detach(), k=k, dim=1).indices[0].cpu().tolist()
            topk_hits += int(int(example["target_action"]) in topk)
        history.append(
            {
                "variant_id": str(standardized_input.tiled_input.variant_id),
                "train_tile_id": str(standardized_input.tiled_input.tile_id),
                "seed": int(seed),
                "epoch": int(epoch),
                "loss": _round_float(statistics.mean(losses)),
                "top1_accuracy": _round_float(correct / len(examples)),
                "topk_hit_rate": _round_float(topk_hits / len(examples)),
                "learning_rate": float(learning_rate),
                "hidden_dim": int(hidden_dim),
                "phase70_standardized_input": True,
                "claim_boundary": PHASE70_CLAIM_BOUNDARY,
            }
        )
    model.eval()
    return model, history


def rollout_phase70_standardized_greedy_policy(
    model,
    standardized_input: Phase70StandardizedTiledInput,
    train_tile_id: str,
    eval_tile_rank: int,
    seed: int,
    phase70_seed_rank: int,
    eval_max_steps: int,
    device: str = "cpu",
) -> dict[str, object]:
    import torch

    selected: list[int] = []
    selected_block_ids: list[str] = []
    rewards: list[float] = []
    invalid_action_count = 0
    for _step_index in range(min(int(eval_max_steps), len(standardized_input.tiled_input.block_ids))):
        inputs = _phase70_model_inputs(standardized_input, selected)
        available = inputs["available_mask"]
        if not bool(available.any()):
            break
        with torch.no_grad():
            logits = model(
                torch.tensor(
                    inputs["block_features"], dtype=torch.float32, device=device
                ).unsqueeze(0),
                torch.tensor(
                    inputs["valid_mask"], dtype=torch.bool, device=device
                ).unsqueeze(0),
                torch.tensor(
                    inputs["selected_mask"], dtype=torch.bool, device=device
                ).unsqueeze(0),
            )
        action = int(torch.argmax(logits, dim=1).item())
        if action in selected or not bool(available[action]):
            invalid_action_count += 1
            valid_indices = [
                int(index) for index, flag in enumerate(available.tolist()) if flag
            ]
            action = valid_indices[0]
        selected.append(action)
        selected_block_ids.append(str(standardized_input.tiled_input.block_ids[action]))
        rewards.append(
            compute_base_planning_reward_from_matrix_row(
                standardized_input.tiled_input.feature_columns,
                standardized_input.reward_matrix[action],
            )
        )
    oracle = build_phase70_oracle_trajectory(standardized_input, eval_max_steps)
    total_reward = _round_float(sum(rewards))
    oracle_total = float(oracle["total_oracle_reward"])
    oracle_gap = _round_float(oracle_total - total_reward)
    oracle_gap_fraction = _round_float(oracle_gap / max(abs(oracle_total), 1.0e-9))
    terminated = len(selected) == len(standardized_input.tiled_input.block_ids)
    return {
        "row_type": "bc_greedy_policy",
        "variant_id": str(standardized_input.tiled_input.variant_id),
        "train_tile_id": str(train_tile_id),
        "eval_tile_id": str(standardized_input.tiled_input.tile_id),
        "eval_tile_rank": int(eval_tile_rank),
        "seed": int(seed),
        "phase70_seed_rank": int(phase70_seed_rank),
        "eval_max_steps": int(eval_max_steps),
        "n_blocks": len(standardized_input.tiled_input.block_ids),
        "n_features": len(standardized_input.tiled_input.feature_columns),
        "episode_steps": len(selected),
        "terminated": bool(terminated),
        "truncated": not terminated,
        "all_actions_valid": invalid_action_count == 0,
        "invalid_action_count": int(invalid_action_count),
        "total_contract_reward": total_reward,
        "oracle_total_reward": _round_float(oracle_total),
        "oracle_gap": oracle_gap,
        "oracle_gap_fraction": oracle_gap_fraction,
        "selected_block_ids": ";".join(selected_block_ids),
        "selected_action_indices": ";".join(str(index) for index in selected),
        "phase70_standardized_input": True,
        "claim_boundary": PHASE70_CLAIM_BOUNDARY,
    }