from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .padded_heldout_policy import _normalize_seeds, _select_train_eval_tiles
from .planning_reward import compute_base_planning_reward_from_matrix_row


PHASE63_CLAIM_BOUNDARY = (
    "Phase 63 is a base-reward set-policy oracle-pretraining experiment. "
    "It tests whether task-aware block scoring and deterministic oracle behavior "
    "cloning improve candidate-block selection under existing Bishan tile inputs. "
    "It does not enable suitability reward, does not test B2/B3, does not test "
    "cross-region transfer, does not prove independent agronomic suitability, "
    "does not prove PCA optimality, and does not justify final submission-level "
    "planning-performance claims."
)

PHASE63_DEFAULT_VARIANTS = ("B0", "D4P8", "D4P16", "D6R8", "D6R16")
PHASE63_ALLOWED_VARIANTS = PHASE63_DEFAULT_VARIANTS


def build_phase63_set_policy_contract(
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str,
    phase61_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] | str = PHASE63_DEFAULT_VARIANTS,
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 5,
    eval_max_steps: int = 8,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
    bc_epochs: int = 80,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    top_k: int = 3,
) -> dict[str, object]:
    if int(eval_max_steps) <= 0:
        raise ValueError("eval_max_steps must be positive")
    if int(bc_epochs) <= 0:
        raise ValueError("bc_epochs must be positive")
    if float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if int(hidden_dim) <= 0:
        raise ValueError("hidden_dim must be positive")
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive")

    normalized_variants = _normalize_phase63_variants(variants)
    normalized_seeds = _normalize_seeds(seeds)
    selected = _select_train_eval_tiles(
        Path(tile_index_csv),
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
    )
    eval_ids = list(selected["eval_tile_ids"])
    train_id = str(selected["train_tile_id"])
    selected_counts = dict(selected["selected_tile_block_counts"])
    return {
        "phase": "phase63_set_policy_oracle_pretraining",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "phase8_output_dir": str(Path(phase8_output_dir)),
        "phase61_output_dir": str(Path(phase61_output_dir)),
        "tile_index_csv": str(Path(tile_index_csv)),
        "variants": normalized_variants,
        "variant_source_dirs": _phase63_variant_source_dirs(
            normalized_variants,
            phase2_output_dir=phase2_output_dir,
            phase8_output_dir=phase8_output_dir,
            phase61_output_dir=phase61_output_dir,
        ),
        "train_tile_id": train_id,
        "train_tile_ids": [train_id],
        "eval_tile_ids": eval_ids,
        "eval_tile_count": len(eval_ids),
        "eval_tile_ranks": {
            str(tile_id): rank for rank, tile_id in enumerate(eval_ids, start=1)
        },
        "selected_tile_block_counts": selected_counts,
        "train_tile_selection": selected["train_tile_selection"],
        "eval_tile_selection": selected["eval_tile_selection"],
        "max_blocks": max(int(count) for count in selected_counts.values()),
        "eval_max_steps": int(eval_max_steps),
        "seeds": normalized_seeds,
        "seed_count": len(normalized_seeds),
        "seed_ranks": {
            str(seed): rank for rank, seed in enumerate(normalized_seeds, start=1)
        },
        "bc_epochs": int(bc_epochs),
        "learning_rate": float(learning_rate),
        "hidden_dim": int(hidden_dim),
        "top_k": int(top_k),
        "claim_boundary": PHASE63_CLAIM_BOUNDARY,
    }


def build_phase63_model_inputs(
    tiled_input,
    selected_indices: Sequence[int] = (),
) -> dict[str, np.ndarray]:
    n_blocks = len(tiled_input.block_ids)
    selected = np.zeros(n_blocks, dtype=bool)
    for index in selected_indices:
        action_index = int(index)
        if action_index < 0 or action_index >= n_blocks:
            raise ValueError(f"Selected action out of range: {action_index}")
        if selected[action_index]:
            raise ValueError(f"Selected action repeated: {action_index}")
        selected[action_index] = True
    valid = np.ones(n_blocks, dtype=bool)
    available = np.logical_and(valid, ~selected)
    return {
        "block_features": tiled_input.state_matrix.astype(np.float32, copy=True),
        "valid_mask": valid,
        "selected_mask": selected,
        "available_mask": available,
    }


class Phase63SetPolicyScorer(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if int(n_features) <= 0:
            raise ValueError("n_features must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        self.n_features = int(n_features)
        self.hidden_dim = int(hidden_dim)
        self.block_encoder = nn.Sequential(
            nn.Linear(self.n_features + 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear((3 * self.n_features) + 3, self.hidden_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        block_features: torch.Tensor,
        valid_mask: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        if block_features.ndim != 3:
            raise ValueError("block_features must have shape [batch, blocks, features]")
        if block_features.shape[-1] != self.n_features:
            raise ValueError("block_features feature count does not match model")
        valid = valid_mask.to(dtype=torch.bool, device=block_features.device)
        selected = selected_mask.to(dtype=torch.bool, device=block_features.device)
        available = torch.logical_and(valid, torch.logical_not(selected))
        valid_f = valid.to(dtype=block_features.dtype).unsqueeze(-1)
        selected_f = selected.to(dtype=block_features.dtype).unsqueeze(-1)
        block_input = torch.cat([block_features, valid_f, selected_f], dim=-1)
        block_encoded = self.block_encoder(block_input)
        context = self._context_features(block_features, valid, selected, available)
        context_encoded = self.context_encoder(context).unsqueeze(1)
        context_encoded = context_encoded.expand(-1, block_features.shape[1], -1)
        logits = self.scorer(
            torch.cat([block_encoded, context_encoded], dim=-1)
        ).squeeze(-1)
        return logits.masked_fill(torch.logical_not(available), -1.0e9)

    def _context_features(
        self,
        block_features: torch.Tensor,
        valid_mask: torch.Tensor,
        selected_mask: torch.Tensor,
        available_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_mean = _masked_mean_tensor(block_features, valid_mask)
        selected_mean = _masked_mean_tensor(block_features, selected_mask)
        available_mean = _masked_mean_tensor(block_features, available_mask)
        denom = torch.clamp(
            valid_mask.sum(dim=1, keepdim=True).to(block_features.dtype),
            min=1.0,
        )
        valid_fraction = valid_mask.to(block_features.dtype).mean(dim=1, keepdim=True)
        selected_fraction = (
            selected_mask.sum(dim=1, keepdim=True).to(block_features.dtype) / denom
        )
        available_fraction = (
            available_mask.sum(dim=1, keepdim=True).to(block_features.dtype) / denom
        )
        return torch.cat(
            [
                valid_mean,
                selected_mean,
                available_mean,
                valid_fraction,
                selected_fraction,
                available_fraction,
            ],
            dim=1,
        )


def _masked_mean_tensor(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    denom = torch.clamp(weights.sum(dim=1), min=1.0)
    return (values * weights).sum(dim=1) / denom


def build_phase63_oracle_trajectory(tiled_input, eval_max_steps: int) -> dict[str, object]:
    if int(eval_max_steps) <= 0:
        raise ValueError("eval_max_steps must be positive")
    rewards = _phase63_block_rewards(tiled_input)
    ranked = sorted(
        range(len(tiled_input.block_ids)),
        key=lambda index: (-rewards[index], str(tiled_input.block_ids[index]), index),
    )
    selected_indices = ranked[: min(int(eval_max_steps), len(ranked))]
    selected_block_ids = [str(tiled_input.block_ids[index]) for index in selected_indices]
    step_rewards = [_round_float(rewards[index]) for index in selected_indices]
    total = _round_float(sum(step_rewards))
    terminated = len(selected_indices) == len(tiled_input.block_ids)
    steps = []
    cumulative = 0.0
    for step_index, action_index in enumerate(selected_indices):
        cumulative = _round_float(cumulative + rewards[action_index])
        steps.append(
            {
                "step_index": int(step_index),
                "action_index": int(action_index),
                "block_id": str(tiled_input.block_ids[action_index]),
                "reward": _round_float(rewards[action_index]),
                "cumulative_reward": cumulative,
                "valid_action_count": int(len(tiled_input.block_ids) - step_index),
            }
        )
    return {
        "variant_id": str(tiled_input.variant_id),
        "tile_id": str(tiled_input.tile_id),
        "eval_max_steps": int(eval_max_steps),
        "n_blocks": len(tiled_input.block_ids),
        "n_features": len(tiled_input.feature_columns),
        "episode_steps": len(selected_indices),
        "terminated": bool(terminated),
        "action_indices": [int(index) for index in selected_indices],
        "selected_block_ids": selected_block_ids,
        "step_rewards": step_rewards,
        "steps": steps,
        "total_oracle_reward": total,
        "top_k_reward_ceiling": total,
        "claim_boundary": PHASE63_CLAIM_BOUNDARY,
    }


def _normalize_phase63_variants(variants: Sequence[str] | str) -> list[str]:
    if isinstance(variants, str):
        values = [part.strip().upper() for part in variants.split(",")]
    else:
        values = [str(item).strip().upper() for item in variants]
    normalized = [value for value in values if value]
    if not normalized:
        raise ValueError("At least one Phase 63 variant must be requested")
    unsupported = [value for value in normalized if value not in PHASE63_ALLOWED_VARIANTS]
    if unsupported:
        raise ValueError(f"Phase 63 unsupported variant: {unsupported[0]}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Phase 63 variants must be unique")
    return normalized


def _phase63_variant_source_dirs(
    variants: Sequence[str],
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str,
    phase61_output_dir: Path | str,
) -> dict[str, str]:
    source_dirs: dict[str, str] = {}
    for variant_id in variants:
        if variant_id == "B0":
            source_dirs[variant_id] = str(Path(phase2_output_dir))
        elif variant_id.startswith("D4"):
            source_dirs[variant_id] = str(Path(phase8_output_dir))
        elif variant_id.startswith("D6"):
            source_dirs[variant_id] = str(Path(phase61_output_dir))
        else:
            raise ValueError(f"Phase 63 has no source routing for {variant_id}")
    return source_dirs


def _phase63_block_rewards(tiled_input) -> list[float]:
    return [
        compute_base_planning_reward_from_matrix_row(
            tiled_input.feature_columns,
            tiled_input.state_matrix[index],
        )
        for index in range(len(tiled_input.block_ids))
    ]


def _round_float(value: object, digits: int = 10) -> float:
    rounded = round(float(value), digits)
    compact = round(rounded, 6)
    if abs(rounded - compact) < 5.0e-8:
        return compact
    return rounded
