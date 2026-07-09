from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
import statistics

import numpy as np
import torch
from torch.nn import functional as F

from .phase63_set_policy_oracle_pretraining import (
    PHASE63_D4_B0_COMPARISONS,
    PHASE63_D4_D6_COMPARISONS,
    PHASE63_DELTA_FIELDNAMES,
    PHASE63_HISTORY_FIELDNAMES,
    PHASE63_ROLLOUT_FIELDNAMES,
    Phase63SetPolicyScorer,
    build_phase63_model_inputs,
    build_phase63_oracle_trajectory,
    build_phase63_set_policy_analysis,
)
from .planning_reward import compute_base_planning_reward_from_matrix_row
from .tiled_inputs import load_tiled_variant_input


PHASE65_CLAIM_BOUNDARY = (
    "Phase 65 is a base-reward train-tile-fitted standardized set-policy "
    "behavior-cloning rerun. Standardization is applied only to policy model "
    "inputs; oracle targets and rollout rewards remain computed from raw "
    "unstandardized feature matrices. It does not enable suitability reward, "
    "does not test B2/B3, does not test transfer, does not prove GeoFM "
    "advantage or PCA optimality, and does not justify formal submission-level "
    "claims."
)

PHASE65_STATUS_GEOFM = "standardization_improves_geofm_set_policy"
PHASE65_STATUS_ALL_VARIANTS = "standardization_improves_all_variants_no_geofm_advantage"
PHASE65_STATUS_NOT_HELPFUL = "standardization_not_helpful"
PHASE65_STATUS_INCONCLUSIVE = "standardization_hurts_or_inconclusive"
PHASE65_STATUS_INSUFFICIENT = "insufficient"

PHASE65_EPSILON = 1.0e-12


@dataclass(frozen=True)
class Phase65Standardizer:
    variant_id: str
    train_tile_id: str
    feature_columns: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    safe_std: np.ndarray
    zero_variance_feature_count: int
    epsilon: float = PHASE65_EPSILON

    def transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("Phase 65 standardizer expects a 2-D state matrix")
        if values.shape[1] != len(self.feature_columns):
            raise ValueError("Phase 65 state matrix feature count does not match transform")
        return ((values - self.mean) / self.safe_std).astype(np.float32, copy=True)

    def to_json_row(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "train_tile_id": self.train_tile_id,
            "n_features": len(self.feature_columns),
            "zero_variance_feature_count": int(self.zero_variance_feature_count),
            "epsilon": float(self.epsilon),
            "feature_columns": list(self.feature_columns),
            "mean": [round(float(value), 10) for value in self.mean.tolist()],
            "std": [round(float(value), 10) for value in self.std.tolist()],
            "safe_std": [round(float(value), 10) for value in self.safe_std.tolist()],
            "claim_boundary": PHASE65_CLAIM_BOUNDARY,
        }


def fit_phase65_train_tile_standardizer(
    tiled_input,
    epsilon: float = PHASE65_EPSILON,
) -> Phase65Standardizer:
    matrix = np.asarray(tiled_input.state_matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Phase 65 train tile state matrix must be 2-D")
    if matrix.shape[0] <= 0:
        raise ValueError("Phase 65 train tile has no blocks")
    if matrix.shape[1] <= 0:
        raise ValueError("Phase 65 train tile has no feature columns")
    mean = np.mean(matrix, axis=0).astype(np.float32)
    std = np.std(matrix, axis=0, ddof=0).astype(np.float32)
    safe_std = np.where(std > float(epsilon), std, 1.0).astype(np.float32)
    return Phase65Standardizer(
        variant_id=str(tiled_input.variant_id),
        train_tile_id=str(tiled_input.tile_id),
        feature_columns=tuple(str(column) for column in tiled_input.feature_columns),
        mean=mean,
        std=std,
        safe_std=safe_std,
        zero_variance_feature_count=int(np.sum(std <= float(epsilon))),
        epsilon=float(epsilon),
    )


def apply_phase65_standardizer(tiled_input, standardizer: Phase65Standardizer):
    if str(tiled_input.variant_id) != standardizer.variant_id:
        raise ValueError(
            "Phase 65 standardizer variant mismatch: "
            f"{tiled_input.variant_id} != {standardizer.variant_id}"
        )
    if tuple(tiled_input.feature_columns) != standardizer.feature_columns:
        raise ValueError("Phase 65 standardizer feature columns do not match tiled input")
    standardized = standardizer.transform_matrix(tiled_input.state_matrix)
    return replace(
        tiled_input,
        state_matrix=standardized,
        claim_boundary=PHASE65_CLAIM_BOUNDARY,
    )


def _validate_aligned_tiled_inputs(raw_tiled, standardized_tiled) -> None:
    if tuple(raw_tiled.block_ids) != tuple(standardized_tiled.block_ids):
        raise ValueError("Phase 65 raw and standardized block IDs are not aligned")
    if tuple(raw_tiled.feature_columns) != tuple(standardized_tiled.feature_columns):
        raise ValueError("Phase 65 raw and standardized feature columns are not aligned")
    if str(raw_tiled.tile_id) != str(standardized_tiled.tile_id):
        raise ValueError("Phase 65 raw and standardized tile IDs are not aligned")
    if str(raw_tiled.variant_id) != str(standardized_tiled.variant_id):
        raise ValueError("Phase 65 raw and standardized variant IDs are not aligned")


def build_phase65_bc_examples(
    raw_tiled_input,
    standardizer: Phase65Standardizer,
    eval_max_steps: int,
) -> list[dict[str, object]]:
    standardized_tiled = apply_phase65_standardizer(raw_tiled_input, standardizer)
    _validate_aligned_tiled_inputs(raw_tiled_input, standardized_tiled)
    trajectory = build_phase63_oracle_trajectory(raw_tiled_input, eval_max_steps)
    examples: list[dict[str, object]] = []
    selected: list[int] = []
    for step in trajectory["steps"]:
        action_index = int(step["action_index"])
        inputs = build_phase63_model_inputs(standardized_tiled, selected)
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


def _round_float(value: object, digits: int = 10) -> float:
    rounded = round(float(value), digits)
    compact = round(rounded, 6)
    if abs(rounded - compact) < 5.0e-8:
        return compact
    return rounded


def train_phase65_behavior_cloner(
    raw_tiled_input,
    standardizer: Phase65Standardizer,
    seed: int,
    eval_max_steps: int,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    top_k: int = 3,
    device: str = "cpu",
) -> tuple[Phase63SetPolicyScorer, list[dict[str, object]]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    examples = build_phase65_bc_examples(raw_tiled_input, standardizer, eval_max_steps)
    if not examples:
        raise ValueError("Phase 65 behavior cloning requires at least one example")
    model = Phase63SetPolicyScorer(
        n_features=len(raw_tiled_input.feature_columns),
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
                example["block_features"],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            valid_mask = torch.tensor(
                example["valid_mask"],
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0)
            selected_mask = torch.tensor(
                example["selected_mask"],
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0)
            target = torch.tensor(
                [int(example["target_action"])],
                dtype=torch.long,
                device=device,
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
                "variant_id": str(raw_tiled_input.variant_id),
                "train_tile_id": str(raw_tiled_input.tile_id),
                "seed": int(seed),
                "epoch": int(epoch),
                "loss": _round_float(statistics.mean(losses)),
                "top1_accuracy": _round_float(correct / len(examples)),
                "topk_hit_rate": _round_float(topk_hits / len(examples)),
                "learning_rate": float(learning_rate),
                "hidden_dim": int(hidden_dim),
                "claim_boundary": PHASE65_CLAIM_BOUNDARY,
            }
        )
    model.eval()
    return model, history


def rollout_phase65_greedy_policy(
    model: Phase63SetPolicyScorer,
    raw_tiled_input,
    standardizer: Phase65Standardizer,
    train_tile_id: str,
    eval_tile_rank: int,
    seed: int,
    phase65_seed_rank: int,
    eval_max_steps: int,
    device: str = "cpu",
) -> dict[str, object]:
    standardized_tiled = apply_phase65_standardizer(raw_tiled_input, standardizer)
    _validate_aligned_tiled_inputs(raw_tiled_input, standardized_tiled)
    selected: list[int] = []
    selected_block_ids: list[str] = []
    rewards: list[float] = []
    invalid_action_count = 0
    for _step_index in range(min(int(eval_max_steps), len(raw_tiled_input.block_ids))):
        inputs = build_phase63_model_inputs(standardized_tiled, selected)
        available = inputs["available_mask"]
        if not bool(available.any()):
            break
        with torch.no_grad():
            logits = model(
                torch.tensor(
                    inputs["block_features"],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0),
                torch.tensor(
                    inputs["valid_mask"],
                    dtype=torch.bool,
                    device=device,
                ).unsqueeze(0),
                torch.tensor(
                    inputs["selected_mask"],
                    dtype=torch.bool,
                    device=device,
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
        selected_block_ids.append(str(raw_tiled_input.block_ids[action]))
        rewards.append(
            compute_base_planning_reward_from_matrix_row(
                raw_tiled_input.feature_columns,
                raw_tiled_input.state_matrix[action],
            )
        )
    oracle = build_phase63_oracle_trajectory(raw_tiled_input, eval_max_steps)
    total_reward = _round_float(sum(rewards))
    oracle_total = float(oracle["total_oracle_reward"])
    oracle_gap = _round_float(oracle_total - total_reward)
    oracle_gap_fraction = _round_float(oracle_gap / max(abs(oracle_total), 1.0e-9))
    terminated = len(selected) == len(raw_tiled_input.block_ids)
    return {
        "row_type": "bc_greedy_policy",
        "variant_id": str(raw_tiled_input.variant_id),
        "train_tile_id": str(train_tile_id),
        "eval_tile_id": str(raw_tiled_input.tile_id),
        "eval_tile_rank": int(eval_tile_rank),
        "seed": int(seed),
        "phase63_seed_rank": int(phase65_seed_rank),
        "eval_max_steps": int(eval_max_steps),
        "n_blocks": len(raw_tiled_input.block_ids),
        "n_features": len(raw_tiled_input.feature_columns),
        "episode_steps": len(selected),
        "terminated": bool(terminated),
        "truncated": bool(not terminated and len(selected) >= int(eval_max_steps)),
        "all_actions_valid": bool(invalid_action_count == 0),
        "invalid_action_count": int(invalid_action_count),
        "total_contract_reward": total_reward,
        "oracle_total_reward": _round_float(oracle_total),
        "oracle_gap": oracle_gap,
        "oracle_gap_fraction": oracle_gap_fraction,
        "selected_block_ids": ";".join(selected_block_ids),
        "selected_action_indices": ";".join(str(index) for index in selected),
        "claim_boundary": PHASE65_CLAIM_BOUNDARY,
    }
