from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

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
