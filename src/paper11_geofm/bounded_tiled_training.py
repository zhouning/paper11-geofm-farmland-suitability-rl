from __future__ import annotations

import csv
import hashlib
import json
import warnings
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .drl_smoke_env import Phase4InputContractEnv
from .tiled_inputs import load_tiled_variant_input


PHASE20_CLAIM_BOUNDARY = (
    "Phase 20 is a bounded same-tile B0/B1 training pilot; it verifies a short "
    "controlled training/evaluation protocol under the deterministic base "
    "planning reward, does not enable suitability reward, and does not support "
    "cross-tile transfer, final policy-performance, or GeoFM-superiority claims."
)

PHASE20_CROSS_TILE_BLOCKER = (
    "Cross-tile learned-policy evaluation is blocked under the current flat "
    "observation design because observation and action spaces vary with tile "
    "block count; a variable-size, padded, or per-block policy design is "
    "required before transfer claims."
)

SUMMARY_FIELDNAMES = [
    "row_type",
    "variant_id",
    "train_tile_id",
    "eval_tile_id",
    "seed",
    "train_timesteps",
    "eval_max_steps",
    "n_blocks",
    "n_features",
    "observation_shape",
    "action_space_n",
    "episode_steps",
    "terminated",
    "truncated",
    "total_contract_reward",
    "selected_block_ids",
    "claim_boundary",
]


def build_phase20_bounded_training_contract(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] = ("B0", "B1"),
    train_tile_id: str | None = None,
    eval_tile_id: str | None = None,
    total_timesteps: int = 8,
    eval_max_steps: int = 4,
    seed: int = 0,
) -> dict[str, object]:
    if int(total_timesteps) <= 0:
        raise ValueError("total_timesteps must be positive")
    if int(eval_max_steps) <= 0:
        raise ValueError("eval_max_steps must be positive")

    normalized_variants = _normalize_variants(variants)
    selected = _select_train_eval_tiles(
        Path(tile_index_csv),
        train_tile_id=train_tile_id,
        eval_tile_id=eval_tile_id,
    )
    return {
        "phase": "phase20_bounded_tiled_training",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "tile_index_csv": str(Path(tile_index_csv)),
        "variants": normalized_variants,
        "train_tile_id": selected["train_tile_id"],
        "eval_tile_id": selected["eval_tile_id"],
        "train_tile_selection": selected["train_tile_selection"],
        "eval_tile_selection": selected["eval_tile_selection"],
        "learned_policy_evaluation_scope": "same_tile_bounded_pilot",
        "cross_tile_evaluation_status": "blocked_variable_observation_shape",
        "cross_tile_blocker": PHASE20_CROSS_TILE_BLOCKER,
        "total_timesteps": int(total_timesteps),
        "eval_max_steps": int(eval_max_steps),
        "seed": int(seed),
        "claim_boundary": PHASE20_CLAIM_BOUNDARY,
    }


def run_phase20_bounded_tiled_training(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] = ("B0", "B1"),
    train_tile_id: str | None = None,
    eval_tile_id: str | None = None,
    total_timesteps: int = 8,
    eval_max_steps: int = 4,
    seed: int = 0,
) -> dict[str, object]:
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import is_masking_supported
    except ImportError as exc:
        raise RuntimeError(
            "Phase 20 bounded tiled training requires stable-baselines3 "
            "and sb3-contrib"
        ) from exc

    contract = build_phase20_bounded_training_contract(
        phase2_output_dir,
        tile_index_csv,
        variants=variants,
        train_tile_id=train_tile_id,
        eval_tile_id=eval_tile_id,
        total_timesteps=total_timesteps,
        eval_max_steps=eval_max_steps,
        seed=seed,
    )

    summaries: list[dict[str, object]] = []
    traces: dict[str, dict[str, list[dict[str, object]]]] = {
        "trained_policy": {},
        "first_valid": {},
        "seeded_random": {},
    }

    for variant_id in contract["variants"]:
        train_tiled = load_tiled_variant_input(
            phase2_output_dir,
            tile_index_csv,
            str(contract["train_tile_id"]),
            variant_id=str(variant_id),
        )
        eval_tiled = load_tiled_variant_input(
            phase2_output_dir,
            tile_index_csv,
            str(contract["eval_tile_id"]),
            variant_id=str(variant_id),
        )
        train_env = Phase4InputContractEnv(
            train_tiled,
            max_steps=int(contract["total_timesteps"]),
        )
        train_env.reset(seed=int(seed))
        if not is_masking_supported(train_env):
            raise ValueError("Phase 20 train env does not expose action_masks")

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

        trained_summary, trained_steps = _evaluate_trained_policy(
            model,
            eval_tiled,
            train_tile_id=str(contract["train_tile_id"]),
            eval_max_steps=int(contract["eval_max_steps"]),
            train_timesteps=int(contract["total_timesteps"]),
            seed=int(seed),
        )
        summaries.append(trained_summary)
        traces["trained_policy"][str(variant_id)] = trained_steps

        for policy_id in ("first_valid", "seeded_random"):
            baseline_summary, baseline_steps = _evaluate_baseline_policy(
                eval_tiled,
                policy_id=policy_id,
                train_tile_id=str(contract["train_tile_id"]),
                eval_max_steps=int(contract["eval_max_steps"]),
                train_timesteps=int(contract["total_timesteps"]),
                seed=int(seed),
            )
            summaries.append(baseline_summary)
            traces[policy_id][str(variant_id)] = baseline_steps

    return {
        **contract,
        "training_completed": True,
        "all_evaluations_completed": all(
            bool(row["terminated"]) or bool(row["truncated"]) for row in summaries
        ),
        "summary_count": len(summaries),
        "summaries": summaries,
        "traces": traces,
        "dependencies": _dependency_metadata(),
    }


def write_phase20_bounded_tiled_training_artifacts(
    protocol: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "phase20_bounded_tiled_training_summary.csv"
    traces_path = output_path / "phase20_bounded_tiled_training_traces.json"

    summaries = protocol.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("Phase 20 protocol is missing a summaries list")

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for summary in summaries:
            if not isinstance(summary, Mapping):
                raise ValueError("Phase 20 summary rows must be objects")
            row = {field: summary.get(field, "") for field in SUMMARY_FIELDNAMES}
            selected = row.get("selected_block_ids")
            if isinstance(selected, list):
                row["selected_block_ids"] = ";".join(str(item) for item in selected)
            writer.writerow(row)

    traces_path.write_text(
        json.dumps(dict(protocol), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"summary_csv": summary_path, "traces_json": traces_path}


def _evaluate_trained_policy(
    model: Any,
    tiled,
    train_tile_id: str,
    eval_max_steps: int,
    train_timesteps: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    env = Phase4InputContractEnv(tiled, max_steps=eval_max_steps)
    obs, info = env.reset(seed=seed)
    steps: list[dict[str, object]] = []
    selected_block_ids: list[str] = []
    total_reward = 0.0
    terminated = False
    truncated = False

    while True:
        masks = env.action_masks()
        valid_actions = _valid_actions(masks)
        if not valid_actions:
            break
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        action_index = int(action)
        if action_index not in valid_actions:
            raise ValueError("Phase 20 trained policy selected an invalid action")
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
            seed=seed,
            train_timesteps=train_timesteps,
            eval_max_steps=eval_max_steps,
            info=info,
            obs_shape=int(obs.shape[0]),
            action_space_n=int(env.action_space.n),
            episode_steps=len(steps),
            terminated=terminated,
            truncated=truncated,
            total_reward=total_reward,
            selected_block_ids=selected_block_ids,
        ),
        steps,
    )


def _evaluate_baseline_policy(
    tiled,
    policy_id: str,
    train_tile_id: str,
    eval_max_steps: int,
    train_timesteps: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    env = Phase4InputContractEnv(tiled, max_steps=eval_max_steps)
    obs, info = env.reset(seed=seed)
    rng = _rng_for(seed, policy_id, tiled.variant_id, tiled.tile_id)
    steps: list[dict[str, object]] = []
    selected_block_ids: list[str] = []
    total_reward = 0.0
    terminated = False
    truncated = False

    while True:
        valid_actions = _valid_actions(env.action_masks())
        if not valid_actions:
            break
        action = _select_baseline_action(policy_id, valid_actions, rng)
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
            seed=seed,
            train_timesteps=train_timesteps,
            eval_max_steps=eval_max_steps,
            info=info,
            obs_shape=int(obs.shape[0]),
            action_space_n=int(env.action_space.n),
            episode_steps=len(steps),
            terminated=terminated,
            truncated=truncated,
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
    seed: int,
    train_timesteps: int,
    eval_max_steps: int,
    info: Mapping[str, object],
    obs_shape: int,
    action_space_n: int,
    episode_steps: int,
    terminated: bool,
    truncated: bool,
    total_reward: float,
    selected_block_ids: list[str],
) -> dict[str, object]:
    return {
        "row_type": row_type,
        "variant_id": variant_id,
        "train_tile_id": train_tile_id,
        "eval_tile_id": eval_tile_id,
        "seed": int(seed),
        "train_timesteps": int(train_timesteps),
        "eval_max_steps": int(eval_max_steps),
        "n_blocks": int(info["n_blocks"]),
        "n_features": int(info["n_features"]),
        "observation_shape": int(obs_shape),
        "action_space_n": int(action_space_n),
        "episode_steps": int(episode_steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "total_contract_reward": _round_float(total_reward),
        "selected_block_ids": selected_block_ids,
        "claim_boundary": PHASE20_CLAIM_BOUNDARY,
    }


def _step_record(
    step_info: Mapping[str, object],
    action: int,
    reward: float,
    env: Phase4InputContractEnv,
) -> dict[str, object]:
    return {
        "step": int(step_info["step"]),
        "action": int(action),
        "selected_block_id": str(step_info["selected_block_id"]),
        "reward": _round_float(reward),
        "valid_actions_after": int(env.action_masks().sum()),
        "terminated": bool(step_info.get("terminated", False)),
    }


def _select_train_eval_tiles(
    tile_index_csv: Path,
    train_tile_id: str | None,
    eval_tile_id: str | None,
) -> dict[str, str]:
    rows = sorted(_read_tile_rows(tile_index_csv), key=lambda row: -int(row["n_blocks"]))
    train_selection = "explicit" if train_tile_id else "largest"

    train_id = str(train_tile_id).strip() if train_tile_id else str(rows[0]["tile_id"])
    known = {str(row["tile_id"]) for row in rows}
    if train_id not in known:
        raise ValueError(f"Train tile ID not found: {train_id}")

    if eval_tile_id:
        eval_id = str(eval_tile_id).strip()
        if eval_id not in known:
            raise ValueError(f"Evaluation tile ID not found: {eval_id}")
        if eval_id != train_id:
            raise ValueError(
                "Phase 20 learned-policy evaluation currently requires the "
                "same train/evaluation tile because flat observations are "
                "tile-size specific; cross-tile evaluation is blocked until a "
                "variable-size or padded policy is implemented."
            )
        eval_selection = "same_as_train_explicit"
    else:
        eval_id = train_id
        eval_selection = "same_as_train_default"

    return {
        "train_tile_id": train_id,
        "eval_tile_id": eval_id,
        "train_tile_selection": train_selection,
        "eval_tile_selection": eval_selection,
    }


def _normalize_variants(variants: Sequence[str]) -> list[str]:
    normalized = [str(item).strip().upper() for item in variants]
    normalized = [item for item in normalized if item]
    if not normalized:
        raise ValueError("At least one Phase 20 variant must be requested")
    unsupported = [variant for variant in normalized if variant not in {"B0", "B1"}]
    if unsupported:
        raise ValueError(
            "Phase 20 is restricted to B0/B1 base-reward variants; "
            f"unsupported variant: {unsupported[0]}"
        )
    return normalized


def _read_tile_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 20 tile index CSV: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [field for field in ("tile_id", "block_ids") if field not in fieldnames]
        if missing:
            raise ValueError(f"Phase 20 tile index is missing columns: {missing}")
        for row in reader:
            tile_id = str(row.get("tile_id", "")).strip()
            block_ids = [
                part.strip()
                for part in str(row.get("block_ids", "")).split(";")
                if part.strip()
            ]
            if not tile_id:
                raise ValueError("Phase 20 tile index contains a row without tile_id")
            if not block_ids:
                raise ValueError(f"Tile {tile_id} contains no block IDs")
            rows.append({"tile_id": tile_id, "block_ids": block_ids, "n_blocks": len(block_ids)})
    if not rows:
        raise ValueError("Phase 20 requires at least one non-empty tile row")
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
    raise ValueError(f"Unknown Phase 20 baseline policy: {policy_id}")


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
