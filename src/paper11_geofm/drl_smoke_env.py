from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .drl_inputs import VariantInput, load_variant_input
from .planning_reward import (
    compute_base_planning_reward_from_matrix_row,
    has_base_planning_reward_columns,
)


PHASE4_CLAIM_BOUNDARY = (
    "Phase 4 is a DRL input-contract smoke environment; it does not train "
    "or evaluate a policy and does not simulate planning outcomes."
)


class Phase4InputContractEnv(gym.Env):
    """Gymnasium smoke environment for Paper11 variant input contracts."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        variant_input: VariantInput,
        max_steps: int | None = None,
    ) -> None:
        super().__init__()
        if not variant_input.block_ids:
            raise ValueError("Phase 4 smoke env requires at least one block")
        if not variant_input.feature_columns:
            raise ValueError("Phase 4 smoke env requires at least one feature column")

        self.variant_input = variant_input
        self.variant_id = variant_input.variant_id
        self.block_ids = variant_input.block_ids
        self.feature_columns = variant_input.feature_columns
        self.reward_mode = variant_input.reward_mode
        self.state_groups = variant_input.state_groups
        self.state_matrix = variant_input.state_matrix.astype(np.float32, copy=True)
        self.n_blocks, self.n_features = self.state_matrix.shape
        self.max_steps = int(max_steps) if max_steps is not None else self.n_blocks
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

        obs_dim = self.n_blocks * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.n_blocks)

        self._selected = np.zeros(self.n_blocks, dtype=bool)
        self._step = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._selected = np.zeros(self.n_blocks, dtype=bool)
        self._step = 0
        return self._get_obs(), self._info()

    def step(self, action: int):
        action = int(action)
        if action < 0 or action >= self.n_blocks:
            raise ValueError(f"Action out of range: {action}")
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
                "selected_block_id": self.block_ids[action],
                "step": self._step,
                "valid_actions": int(self.action_masks().sum()),
            }
        )
        return self._get_obs(), reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        return ~self._selected.copy()

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.state_matrix.reshape(-1),
                self._global_smoke_features(),
            ]
        ).astype(np.float32)

    def _global_smoke_features(self) -> np.ndarray:
        step_fraction = min(self._step / self.max_steps, 1.0)
        budget_remaining = max(1.0 - step_fraction, 0.0)
        valid_action_fraction = float(self.action_masks().mean())
        return np.array(
            [budget_remaining, step_fraction, valid_action_fraction],
            dtype=np.float32,
        )

    def _contract_reward(self, action: int) -> float:
        if self.reward_mode == "base_planning_reward":
            return compute_base_planning_reward_from_matrix_row(
                self.feature_columns,
                self.state_matrix[action],
            )
        if self.reward_mode == "base_plus_suitability_reward":
            reward = 0.0
            if has_base_planning_reward_columns(self.feature_columns):
                reward += compute_base_planning_reward_from_matrix_row(
                    self.feature_columns,
                    self.state_matrix[action],
                )
            if "suitability_proxy" in self.feature_columns:
                column_index = self.feature_columns.index("suitability_proxy")
                reward += float(self.state_matrix[action, column_index])
            return round(float(reward), 10)
        return 0.0

    def _info(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "n_blocks": self.n_blocks,
            "n_features": self.n_features,
            "reward_mode": self.reward_mode,
            "state_groups": self.state_groups,
            "claim_boundary": PHASE4_CLAIM_BOUNDARY,
        }


def make_phase4_smoke_env(
    phase2_output_dir: Path | str,
    variant_id: str,
    max_steps: int | None = None,
) -> Phase4InputContractEnv:
    return Phase4InputContractEnv(
        load_variant_input(phase2_output_dir, variant_id),
        max_steps=max_steps,
    )
