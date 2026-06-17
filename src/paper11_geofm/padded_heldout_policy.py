from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from importlib.metadata import PackageNotFoundError, version

from .planning_reward import compute_base_planning_reward_from_matrix_row
from .tiled_inputs import load_tiled_variant_input


PHASE25_CLAIM_BOUNDARY = (
    "Phase 25 is a bounded padded variable-size held-out-tile B0/B1 MaskablePPO "
    "learned-policy pilot under the deterministic base planning reward; it tests "
    "distinct Bishan tiles, does not enable suitability reward, does not test B2/B3, "
    "and does not support cross-region transfer or submission-level planning-performance claims."
)

PHASE25_REMAINING_EVIDENCE_GAPS = [
    "longer_training_budget_and_hyperparameter_sensitivity",
    "suitability_reward_validation_before_B2_B3",
    "held_out_region_transfer_evaluation",
    "spatial_case_maps_and_uncertainty",
    "submission_level_ablation_and_robustness_package",
]

PHASE25_GLOBAL_FEATURE_COUNT = 5


class Phase25PaddedTileEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        tiled_input,
        max_blocks: int,
        max_steps: int | None = None,
    ) -> None:
        super().__init__()
        if not tiled_input.block_ids:
            raise ValueError("Phase 25 padded env requires at least one block")
        if not tiled_input.feature_columns:
            raise ValueError("Phase 25 padded env requires at least one feature column")
        if tiled_input.reward_mode != "base_planning_reward":
            raise ValueError("Phase 25 only supports base_planning_reward")

        self.tiled_input = tiled_input
        self.tile_id = tiled_input.tile_id
        self.variant_id = tiled_input.variant_id
        self.block_ids = tuple(tiled_input.block_ids)
        self.feature_columns = tuple(tiled_input.feature_columns)
        self.reward_mode = tiled_input.reward_mode
        self.state_groups = tuple(tiled_input.state_groups)
        self.state_matrix = tiled_input.state_matrix.astype(np.float32, copy=True)
        self.n_blocks, self.n_features = self.state_matrix.shape
        self.max_blocks = int(max_blocks)
        if self.max_blocks < self.n_blocks:
            raise ValueError("max_blocks must cover tile block count")
        self.max_steps = int(max_steps) if max_steps is not None else self.n_blocks
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

        obs_dim = (
            self.max_blocks * self.n_features
            + self.max_blocks
            + self.max_blocks
            + PHASE25_GLOBAL_FEATURE_COUNT
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.max_blocks)

        self._selected = np.zeros(self.max_blocks, dtype=bool)
        self._valid_block_mask = np.zeros(self.max_blocks, dtype=bool)
        self._valid_block_mask[: self.n_blocks] = True
        self._step = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._selected = np.zeros(self.max_blocks, dtype=bool)
        self._step = 0
        return self._get_obs(), self._info()

    def step(self, action: int):
        action = int(action)
        if action < 0 or action >= self.max_blocks:
            raise ValueError(f"Action out of range: {action}")
        if not self._valid_block_mask[action]:
            raise ValueError(f"Action is a padded action: {action}")
        if self._selected[action]:
            raise ValueError(f"Action already selected: {action}")

        self._selected[action] = True
        self._step += 1
        reward = self._contract_reward(action)
        terminated = self._step >= self.max_steps or not self.action_masks().any()
        info = self._info()
        info.update(
            {
                "action": action,
                "action_valid": True,
                "selected_block_id": self.block_ids[action],
                "step": self._step,
                "valid_actions": int(self.action_masks().sum()),
                "terminated": bool(terminated),
            }
        )
        return self._get_obs(), reward, bool(terminated), False, info

    def action_masks(self) -> np.ndarray:
        return np.logical_and(self._valid_block_mask, ~self._selected).copy()

    def _get_obs(self) -> np.ndarray:
        padded = np.zeros((self.max_blocks, self.n_features), dtype=np.float32)
        padded[: self.n_blocks] = self.state_matrix
        return np.concatenate(
            [
                padded.reshape(-1),
                self._selected.astype(np.float32),
                self._valid_block_mask.astype(np.float32),
                self._global_features(),
            ]
        ).astype(np.float32)

    def _global_features(self) -> np.ndarray:
        budget_remaining = max(1.0 - (self._step / self.max_steps), 0.0)
        step_fraction = min(self._step / self.max_steps, 1.0)
        valid_action_fraction = float(self.action_masks().sum()) / float(self.max_blocks)
        real_block_fraction = float(self.n_blocks) / float(self.max_blocks)
        real_block_count_fraction = float(self.n_blocks) / float(self.max_blocks)
        return np.array(
            [
                budget_remaining,
                step_fraction,
                valid_action_fraction,
                real_block_fraction,
                real_block_count_fraction,
            ],
            dtype=np.float32,
        )

    def _contract_reward(self, action: int) -> float:
        return compute_base_planning_reward_from_matrix_row(
            self.feature_columns,
            self.state_matrix[action],
        )

    def _info(self) -> dict[str, object]:
        return {
            "phase": "phase25_padded_heldout_policy",
            "variant_id": self.variant_id,
            "tile_id": self.tile_id,
            "n_blocks": self.n_blocks,
            "n_features": self.n_features,
            "max_blocks": self.max_blocks,
            "reward_mode": self.reward_mode,
            "state_groups": self.state_groups,
            "claim_boundary": PHASE25_CLAIM_BOUNDARY,
        }
