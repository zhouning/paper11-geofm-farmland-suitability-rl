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

SUMMARY_FIELDNAMES = [
    "row_type",
    "variant_id",
    "train_tile_id",
    "eval_tile_id",
    "eval_tile_rank",
    "seed",
    "phase25_seed_rank",
    "train_timesteps",
    "eval_max_steps",
    "max_blocks",
    "train_n_blocks",
    "eval_n_blocks",
    "n_features",
    "observation_shape",
    "action_space_n",
    "episode_steps",
    "terminated",
    "truncated",
    "all_actions_valid",
    "invalid_action_count",
    "total_contract_reward",
    "selected_block_ids",
    "claim_boundary",
]


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


def build_phase25_padded_heldout_policy_contract(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] | str = ("B0", "B1"),
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 3,
    total_timesteps: int = 32,
    eval_max_steps: int = 4,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
) -> dict[str, object]:
    if int(total_timesteps) <= 0:
        raise ValueError("total_timesteps must be positive")
    if int(eval_max_steps) <= 0:
        raise ValueError("eval_max_steps must be positive")

    normalized_variants = _normalize_variants(variants)
    normalized_seeds = _normalize_seeds(seeds)
    selected = _select_train_eval_tiles(
        Path(tile_index_csv),
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
    )
    eval_ids = list(selected["eval_tile_ids"])
    selected_counts = dict(selected["selected_tile_block_counts"])
    max_blocks = max(int(selected_counts[tile_id]) for tile_id in selected_counts)
    eval_tile_ranks = {
        str(tile_id): rank for rank, tile_id in enumerate(eval_ids, start=1)
    }
    seed_ranks = {
        str(seed): rank for rank, seed in enumerate(normalized_seeds, start=1)
    }

    train_id = str(selected["train_tile_id"])
    return {
        "phase": "phase25_padded_heldout_policy",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "tile_index_csv": str(Path(tile_index_csv)),
        "variants": normalized_variants,
        "train_tile_id": train_id,
        "train_tile_ids": [train_id],
        "eval_tile_ids": eval_ids,
        "eval_tile_count": len(eval_ids),
        "eval_tile_ranks": eval_tile_ranks,
        "selected_tile_block_counts": selected_counts,
        "train_tile_selection": selected["train_tile_selection"],
        "eval_tile_selection": selected["eval_tile_selection"],
        "padded_policy_status": "enabled_distinct_heldout_tiles",
        "learned_policy_evaluation_scope": (
            "padded_variable_size_heldout_tile_b0_b1_training_pilot"
        ),
        "max_blocks": int(max_blocks),
        "total_timesteps": int(total_timesteps),
        "eval_max_steps": int(eval_max_steps),
        "seeds": normalized_seeds,
        "seed_count": len(normalized_seeds),
        "seed_ranks": seed_ranks,
        "claim_boundary": PHASE25_CLAIM_BOUNDARY,
        "remaining_evidence_gaps": list(PHASE25_REMAINING_EVIDENCE_GAPS),
    }


def run_phase25_padded_heldout_policy(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] | str = ("B0", "B1"),
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 3,
    total_timesteps: int = 32,
    eval_max_steps: int = 4,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
) -> dict[str, object]:
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import is_masking_supported
    except ImportError as exc:
        raise RuntimeError(
            "Phase 25 padded held-out policy requires stable-baselines3 "
            "and sb3-contrib"
        ) from exc

    contract = build_phase25_padded_heldout_policy_contract(
        phase2_output_dir,
        tile_index_csv,
        variants=variants,
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
        total_timesteps=total_timesteps,
        eval_max_steps=eval_max_steps,
        seeds=seeds,
    )

    summaries: list[dict[str, object]] = []
    traces: dict[str, dict[str, dict[str, dict[str, list[dict[str, object]]]]]] = {
        "trained_policy": {},
        "first_valid": {},
        "seeded_random": {},
    }

    for variant_id in contract["variants"]:
        for seed in contract["seeds"]:
            train_tiled = load_tiled_variant_input(
                phase2_output_dir,
                tile_index_csv,
                str(contract["train_tile_id"]),
                variant_id=str(variant_id),
            )
            train_env = Phase25PaddedTileEnv(
                train_tiled,
                max_blocks=int(contract["max_blocks"]),
                max_steps=int(contract["total_timesteps"]),
            )
            train_env.reset(seed=int(seed))
            if not is_masking_supported(train_env):
                raise ValueError("Phase 25 train env does not expose action_masks")

            model = MaskablePPO(
                "MlpPolicy",
                train_env,
                seed=int(seed),
                device="cpu",
                verbose=0,
                n_steps=4,
                batch_size=4,
                n_epochs=1,
                gamma=0.99,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="XPU device count is zero!.*",
                    category=UserWarning,
                )
                model.learn(total_timesteps=int(contract["total_timesteps"]))

            for eval_tile_id in contract["eval_tile_ids"]:
                eval_tiled = load_tiled_variant_input(
                    phase2_output_dir,
                    tile_index_csv,
                    str(eval_tile_id),
                    variant_id=str(variant_id),
                )
                eval_tile_rank = int(contract["eval_tile_ranks"][str(eval_tile_id)])
                seed_rank = int(contract["seed_ranks"][str(int(seed))])

                trained_summary, trained_steps = _evaluate_trained_policy(
                    model,
                    eval_tiled,
                    train_tile_id=str(contract["train_tile_id"]),
                    train_n_blocks=int(
                        contract["selected_tile_block_counts"][
                            str(contract["train_tile_id"])
                        ]
                    ),
                    max_blocks=int(contract["max_blocks"]),
                    eval_tile_rank=eval_tile_rank,
                    phase25_seed_rank=seed_rank,
                    eval_max_steps=int(contract["eval_max_steps"]),
                    train_timesteps=int(contract["total_timesteps"]),
                    seed=int(seed),
                )
                summaries.append(trained_summary)
                _store_trace(
                    traces,
                    "trained_policy",
                    str(variant_id),
                    str(eval_tile_id),
                    int(seed),
                    trained_steps,
                )

                for policy_id in ("first_valid", "seeded_random"):
                    baseline_summary, baseline_steps = _evaluate_baseline_policy(
                        eval_tiled,
                        policy_id=policy_id,
                        train_tile_id=str(contract["train_tile_id"]),
                        train_n_blocks=int(
                            contract["selected_tile_block_counts"][
                                str(contract["train_tile_id"])
                            ]
                        ),
                        max_blocks=int(contract["max_blocks"]),
                        eval_tile_rank=eval_tile_rank,
                        phase25_seed_rank=seed_rank,
                        eval_max_steps=int(contract["eval_max_steps"]),
                        train_timesteps=int(contract["total_timesteps"]),
                        seed=int(seed),
                    )
                    summaries.append(baseline_summary)
                    _store_trace(
                        traces,
                        policy_id,
                        str(variant_id),
                        str(eval_tile_id),
                        int(seed),
                        baseline_steps,
                    )

    comparison = _build_comparison(summaries, contract)
    return {
        **contract,
        "training_completed": True,
        "all_evaluations_completed": all(
            bool(row["terminated"]) or bool(row["truncated"]) for row in summaries
        ),
        "summary_count": len(summaries),
        "summaries": summaries,
        "traces": traces,
        "comparison": comparison,
        "dependencies": _dependency_metadata(),
    }


def write_phase25_padded_heldout_policy_artifacts(
    protocol: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "phase25_padded_heldout_policy_summary.csv"
    traces_path = output_path / "phase25_padded_heldout_policy_traces.json"
    comparison_path = output_path / "phase25_padded_heldout_policy_comparison.json"

    summaries = protocol.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("Phase 25 protocol is missing a summaries list")

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for summary in summaries:
            if not isinstance(summary, Mapping):
                raise ValueError("Phase 25 summary rows must be objects")
            row = {field: summary.get(field, "") for field in SUMMARY_FIELDNAMES}
            selected = row.get("selected_block_ids")
            if isinstance(selected, list):
                row["selected_block_ids"] = ";".join(str(item) for item in selected)
            writer.writerow(row)

    traces_path.write_text(
        json.dumps(dict(protocol), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    comparison = protocol.get("comparison")
    if not isinstance(comparison, Mapping):
        raise ValueError("Phase 25 protocol is missing a comparison object")
    comparison_path.write_text(
        json.dumps(dict(comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "summary_csv": summary_path,
        "traces_json": traces_path,
        "comparison_json": comparison_path,
    }


def _evaluate_trained_policy(
    model: Any,
    tiled,
    train_tile_id: str,
    train_n_blocks: int,
    max_blocks: int,
    eval_tile_rank: int,
    phase25_seed_rank: int,
    eval_max_steps: int,
    train_timesteps: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    env = Phase25PaddedTileEnv(tiled, max_blocks=max_blocks, max_steps=eval_max_steps)
    obs, info = env.reset(seed=seed)
    steps: list[dict[str, object]] = []
    selected_block_ids: list[str] = []
    total_reward = 0.0
    terminated = False
    truncated = False
    invalid_action_count = 0

    while True:
        masks = env.action_masks()
        valid_actions = _valid_actions(masks)
        if not valid_actions:
            break
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        action_index = int(action)
        if action_index not in valid_actions:
            invalid_action_count += 1
            raise ValueError("Phase 25 trained policy selected an invalid action")
        obs, reward, terminated, truncated, step_info = env.step(action_index)
        reward_value = float(reward)
        total_reward += reward_value
        selected_block_id = str(step_info["selected_block_id"])
        selected_block_ids.append(selected_block_id)
        steps.append(_step_record(step_info, action_index, reward_value, env))
        if terminated or truncated:
            break

    return (
        _summary_row(
            row_type="trained_policy",
            variant_id=str(info["variant_id"]),
            train_tile_id=train_tile_id,
            eval_tile_id=tiled.tile_id,
            eval_tile_rank=eval_tile_rank,
            seed=seed,
            phase25_seed_rank=phase25_seed_rank,
            train_timesteps=train_timesteps,
            eval_max_steps=eval_max_steps,
            max_blocks=max_blocks,
            train_n_blocks=train_n_blocks,
            info=info,
            obs_shape=int(obs.shape[0]),
            action_space_n=int(env.action_space.n),
            episode_steps=len(steps),
            terminated=terminated,
            truncated=truncated,
            invalid_action_count=invalid_action_count,
            total_reward=total_reward,
            selected_block_ids=selected_block_ids,
        ),
        steps,
    )


def _evaluate_baseline_policy(
    tiled,
    policy_id: str,
    train_tile_id: str,
    train_n_blocks: int,
    max_blocks: int,
    eval_tile_rank: int,
    phase25_seed_rank: int,
    eval_max_steps: int,
    train_timesteps: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    env = Phase25PaddedTileEnv(tiled, max_blocks=max_blocks, max_steps=eval_max_steps)
    obs, info = env.reset(seed=seed)
    rng = _rng_for(seed, policy_id, tiled.variant_id, tiled.tile_id)
    steps: list[dict[str, object]] = []
    selected_block_ids: list[str] = []
    total_reward = 0.0
    terminated = False
    truncated = False
    invalid_action_count = 0

    while True:
        valid_actions = _valid_actions(env.action_masks())
        if not valid_actions:
            break
        action = _select_baseline_action(policy_id, valid_actions, rng)
        if action not in valid_actions:
            invalid_action_count += 1
            raise ValueError(f"Phase 25 baseline policy selected an invalid action: {policy_id}")
        obs, reward, terminated, truncated, step_info = env.step(action)
        reward_value = float(reward)
        total_reward += reward_value
        selected_block_id = str(step_info["selected_block_id"])
        selected_block_ids.append(selected_block_id)
        steps.append(_step_record(step_info, action, reward_value, env))
        if terminated or truncated:
            break

    return (
        _summary_row(
            row_type=policy_id,
            variant_id=str(info["variant_id"]),
            train_tile_id=train_tile_id,
            eval_tile_id=tiled.tile_id,
            eval_tile_rank=eval_tile_rank,
            seed=seed,
            phase25_seed_rank=phase25_seed_rank,
            train_timesteps=train_timesteps,
            eval_max_steps=eval_max_steps,
            max_blocks=max_blocks,
            train_n_blocks=train_n_blocks,
            info=info,
            obs_shape=int(obs.shape[0]),
            action_space_n=int(env.action_space.n),
            episode_steps=len(steps),
            terminated=terminated,
            truncated=truncated,
            invalid_action_count=invalid_action_count,
            total_reward=total_reward,
            selected_block_ids=selected_block_ids,
        ),
        steps,
    )


def _summary_row(
    row_type: str,
    variant_id: str,
    train_tile_id: str,
    eval_tile_id: str,
    eval_tile_rank: int,
    seed: int,
    phase25_seed_rank: int,
    train_timesteps: int,
    eval_max_steps: int,
    max_blocks: int,
    train_n_blocks: int,
    info: Mapping[str, object],
    obs_shape: int,
    action_space_n: int,
    episode_steps: int,
    terminated: bool,
    truncated: bool,
    invalid_action_count: int,
    total_reward: float,
    selected_block_ids: list[str],
) -> dict[str, object]:
    return {
        "row_type": row_type,
        "variant_id": variant_id,
        "train_tile_id": train_tile_id,
        "eval_tile_id": eval_tile_id,
        "eval_tile_rank": int(eval_tile_rank),
        "seed": int(seed),
        "phase25_seed_rank": int(phase25_seed_rank),
        "train_timesteps": int(train_timesteps),
        "eval_max_steps": int(eval_max_steps),
        "max_blocks": int(max_blocks),
        "train_n_blocks": int(train_n_blocks),
        "eval_n_blocks": int(info["n_blocks"]),
        "n_features": int(info["n_features"]),
        "observation_shape": int(obs_shape),
        "action_space_n": int(action_space_n),
        "episode_steps": int(episode_steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "all_actions_valid": int(invalid_action_count) == 0,
        "invalid_action_count": int(invalid_action_count),
        "total_contract_reward": _round_float(total_reward),
        "selected_block_ids": selected_block_ids,
        "claim_boundary": PHASE25_CLAIM_BOUNDARY,
    }


def _step_record(
    step_info: Mapping[str, object],
    action: int,
    reward: float,
    env: Phase25PaddedTileEnv,
) -> dict[str, object]:
    return {
        "step": int(step_info["step"]),
        "action": int(action),
        "selected_block_id": str(step_info["selected_block_id"]),
        "reward": _round_float(reward),
        "valid_actions_after": int(env.action_masks().sum()),
        "terminated": bool(step_info.get("terminated", False)),
    }


def _build_comparison(
    summaries: list[dict[str, object]],
    contract: Mapping[str, object],
) -> dict[str, object]:
    nested = _nested_mean_rewards(summaries)
    learned_summary = _policy_summary(
        summaries,
        "trained_policy",
        list(contract["variants"]),
        list(contract["eval_tile_ids"]),
    )
    baselines = {
        policy_id: _policy_summary(
            summaries,
            policy_id,
            list(contract["variants"]),
            list(contract["eval_tile_ids"]),
        )
        for policy_id in ("first_valid", "seeded_random")
    }
    delta = learned_summary["B1_minus_B0_mean_reward"]
    return {
        "phase": "phase25_padded_heldout_policy_comparison",
        "train_tile_id": contract["train_tile_id"],
        "train_tile_ids": list(contract["train_tile_ids"]),
        "eval_tile_ids": list(contract["eval_tile_ids"]),
        "variants": list(contract["variants"]),
        "seeds": list(contract["seeds"]),
        "seed_count": int(contract["seed_count"]),
        "policies": ["trained_policy", "first_valid", "seeded_random"],
        "total_timesteps": int(contract["total_timesteps"]),
        "eval_max_steps": int(contract["eval_max_steps"]),
        "max_blocks": int(contract["max_blocks"]),
        "summary_count": len(summaries),
        "mean_reward_by_row_type_variant_eval_tile": nested,
        "learned_policy": learned_summary,
        "baselines": baselines,
        "pilot_result_status": _pilot_result_status(delta),
        "claim_boundary": PHASE25_CLAIM_BOUNDARY,
        "remaining_evidence_gaps": list(PHASE25_REMAINING_EVIDENCE_GAPS),
    }


def _nested_mean_rewards(
    summaries: list[dict[str, object]],
) -> dict[str, dict[str, dict[str, float]]]:
    buckets: dict[str, dict[str, dict[str, list[float]]]] = {}
    for row in summaries:
        row_type = str(row["row_type"])
        variant_id = str(row["variant_id"])
        eval_tile_id = str(row["eval_tile_id"])
        buckets.setdefault(row_type, {}).setdefault(variant_id, {}).setdefault(
            eval_tile_id,
            [],
        ).append(float(row["total_contract_reward"]))

    result: dict[str, dict[str, dict[str, float]]] = {}
    for row_type, variants in buckets.items():
        result[row_type] = {}
        for variant_id, tiles in variants.items():
            result[row_type][variant_id] = {}
            for eval_tile_id, values in tiles.items():
                result[row_type][variant_id][eval_tile_id] = _round_float(
                    sum(values) / len(values)
                )
    return result


def _policy_summary(
    summaries: list[dict[str, object]],
    row_type: str,
    variants: list[object],
    eval_tile_ids: list[object],
) -> dict[str, object]:
    variant_means: dict[str, float] = {}
    for variant_id in variants:
        values = [
            float(row["total_contract_reward"])
            for row in summaries
            if row["row_type"] == row_type and row["variant_id"] == variant_id
        ]
        if values:
            variant_means[str(variant_id)] = _round_float(sum(values) / len(values))

    b1_minus_b0 = None
    if "B0" in variant_means and "B1" in variant_means:
        b1_minus_b0 = _round_float(variant_means["B1"] - variant_means["B0"])

    tile_deltas: dict[str, float | None] = {}
    for eval_tile_id in eval_tile_ids:
        tile_id = str(eval_tile_id)
        b0_values = [
            float(row["total_contract_reward"])
            for row in summaries
            if row["row_type"] == row_type
            and row["variant_id"] == "B0"
            and row["eval_tile_id"] == eval_tile_id
        ]
        b1_values = [
            float(row["total_contract_reward"])
            for row in summaries
            if row["row_type"] == row_type
            and row["variant_id"] == "B1"
            and row["eval_tile_id"] == eval_tile_id
        ]
        if b0_values and b1_values:
            tile_deltas[tile_id] = _round_float(
                (sum(b1_values) / len(b1_values))
                - (sum(b0_values) / len(b0_values))
            )
        else:
            tile_deltas[tile_id] = None

    return {
        "mean_reward_by_variant": variant_means,
        "B1_minus_B0_mean_reward": b1_minus_b0,
        "heldout_tile_B1_minus_B0_mean_reward": tile_deltas,
    }


def _pilot_result_status(delta: object) -> str:
    if delta is None:
        return "insufficient_B0_B1_learned_policy_rows"
    delta_value = float(delta)
    if delta_value > 1e-9:
        return "B1_improves_B0"
    if delta_value < -1e-9:
        return "B1_underperforms_B0"
    return "B1_matches_B0"


def _select_train_eval_tiles(
    tile_index_csv: Path,
    train_tile_id: str | None,
    eval_tile_ids: Sequence[str] | str | None,
    max_eval_tiles: int,
) -> dict[str, object]:
    rows = sorted(_read_tile_rows(tile_index_csv), key=lambda row: -int(row["n_blocks"]))
    train_selection = "explicit" if train_tile_id else "largest"
    train_id = str(train_tile_id).strip() if train_tile_id else str(rows[0]["tile_id"])
    known_counts = {str(row["tile_id"]): int(row["n_blocks"]) for row in rows}
    if train_id not in known_counts:
        raise ValueError(f"Train tile ID not found: {train_id}")

    explicit_eval_ids = _normalize_eval_tile_ids(eval_tile_ids)
    if explicit_eval_ids:
        eval_selection = "explicit"
        selected_eval_ids = explicit_eval_ids
    else:
        if int(max_eval_tiles) <= 0:
            raise ValueError("max_eval_tiles must be positive")
        eval_selection = "largest_distinct"
        selected_eval_ids = [
            str(row["tile_id"]) for row in rows if str(row["tile_id"]) != train_id
        ][: int(max_eval_tiles)]

    if not selected_eval_ids:
        raise ValueError("Phase 25 requires at least one distinct evaluation tile")

    seen: set[str] = set()
    for eval_id in selected_eval_ids:
        if eval_id in seen:
            raise ValueError(
                f"Phase 25 evaluation tiles must be distinct; duplicate evaluation tile ID: {eval_id}"
            )
        seen.add(eval_id)
        if eval_id not in known_counts:
            raise ValueError(f"Evaluation tile ID not found: {eval_id}")
        if eval_id == train_id:
            raise ValueError("Phase 25 train and evaluation tiles must be distinct")

    selected_counts = {train_id: known_counts[train_id]}
    for eval_id in selected_eval_ids:
        selected_counts[eval_id] = known_counts[eval_id]

    return {
        "train_tile_id": train_id,
        "eval_tile_ids": selected_eval_ids,
        "selected_tile_block_counts": selected_counts,
        "train_tile_selection": train_selection,
        "eval_tile_selection": eval_selection,
    }


def _normalize_variants(variants: Sequence[str] | str) -> list[str]:
    if isinstance(variants, str):
        values = [part.strip() for part in variants.split(",")]
    else:
        values = [str(item).strip() for item in variants]
    normalized = [item.upper() for item in values if item]
    if not normalized:
        raise ValueError("At least one Phase 25 variant must be requested")
    unsupported = [variant for variant in normalized if variant not in {"B0", "B1"}]
    if unsupported:
        raise ValueError(
            "Phase 25 is restricted to B0/B1 base-reward variants; "
            f"unsupported variant: {unsupported[0]}"
        )
    return normalized


def _normalize_eval_tile_ids(
    eval_tile_ids: Sequence[str] | str | None,
) -> list[str]:
    if eval_tile_ids is None:
        return []
    if isinstance(eval_tile_ids, str):
        values = [part.strip() for part in eval_tile_ids.split(",")]
    else:
        values = [str(item).strip() for item in eval_tile_ids]
    return [item for item in values if item]


def _normalize_seeds(
    seeds: Sequence[int | str] | str | int | None,
) -> list[int]:
    if seeds is None:
        values: list[int | str] = [0, 1, 2]
    elif isinstance(seeds, str):
        values = [part.strip() for part in seeds.split(",")]
    elif isinstance(seeds, int):
        values = [seeds]
    else:
        values = list(seeds)

    normalized: list[int] = []
    for value in values:
        if str(value).strip() == "":
            continue
        normalized.append(int(value))
    if not normalized:
        raise ValueError("At least one Phase 25 seed must be requested")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Phase 25 seeds must be unique")
    return normalized


def _read_tile_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 25 tile index CSV: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [field for field in ("tile_id", "block_ids") if field not in fieldnames]
        if missing:
            raise ValueError(f"Phase 25 tile index is missing columns: {missing}")
        for row in reader:
            tile_id = str(row.get("tile_id", "")).strip()
            block_ids = [
                part.strip()
                for part in str(row.get("block_ids", "")).split(";")
                if part.strip()
            ]
            if not tile_id:
                raise ValueError("Phase 25 tile index contains a row without tile_id")
            if not block_ids:
                raise ValueError(f"Tile {tile_id} contains no block IDs")
            rows.append(
                {
                    "tile_id": tile_id,
                    "block_ids": block_ids,
                    "n_blocks": len(block_ids),
                }
            )
    if len(rows) < 2:
        raise ValueError("Phase 25 requires at least two non-empty tile rows")
    return rows


def _valid_actions(mask) -> list[int]:
    return [int(index) for index, valid in enumerate(mask.tolist()) if bool(valid)]


def _select_baseline_action(
    policy_id: str,
    valid_actions: list[int],
    rng: np.random.Generator,
) -> int:
    if policy_id == "first_valid":
        return valid_actions[0]
    if policy_id == "seeded_random":
        return int(rng.choice(valid_actions))
    raise ValueError(f"Unknown Phase 25 baseline policy: {policy_id}")


def _rng_for(
    seed: int,
    policy_id: str,
    variant_id: str,
    tile_id: str,
) -> np.random.Generator:
    key = f"{int(seed)}:{policy_id}:{variant_id}:{tile_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    child_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(child_seed)


def _store_trace(
    traces: dict[str, dict[str, dict[str, dict[str, list[dict[str, object]]]]]],
    row_type: str,
    variant_id: str,
    eval_tile_id: str,
    seed: int,
    steps: list[dict[str, object]],
) -> None:
    variant_traces = traces[row_type].setdefault(variant_id, {})
    tile_traces = variant_traces.setdefault(eval_tile_id, {})
    tile_traces[str(int(seed))] = steps


def _dependency_metadata() -> dict[str, dict[str, object]]:
    return {
        "stable_baselines3": _package_metadata("stable-baselines3"),
        "sb3_contrib": _package_metadata("sb3-contrib"),
    }


def _package_metadata(distribution_name: str) -> dict[str, object]:
    try:
        package_version = version(distribution_name)
    except PackageNotFoundError:
        return {"available": False, "version": None}
    return {"available": True, "version": package_version}


def _round_float(value: float) -> float:
    return round(float(value), 10)
