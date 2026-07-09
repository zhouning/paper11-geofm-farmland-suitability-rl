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
