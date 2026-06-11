from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .drl_smoke_env import Phase4InputContractEnv
from .planning_reward import compute_base_planning_reward_from_matrix_row
from .tiled_inputs import load_tiled_variant_input


PHASE21_CLAIM_BOUNDARY = (
    "Phase 21 is a bounded cross-tile per-block scorer pilot; it verifies that "
    "a learned block scorer can train on one tile and evaluate a distinct tile "
    "without relying on flat tile-specific observation shape, does not enable "
    "suitability reward, and does not support final policy-performance, "
    "cross-region transfer, or GeoFM-superiority claims."
)

SUMMARY_FIELDNAMES = [
    "row_type",
    "variant_id",
    "train_tile_id",
    "eval_tile_id",
    "seed",
    "ridge_alpha",
    "eval_max_steps",
    "train_n_blocks",
    "eval_n_blocks",
    "n_features",
    "eval_observation_shape",
    "action_space_n",
    "episode_steps",
    "terminated",
    "truncated",
    "total_contract_reward",
    "selected_block_ids",
    "claim_boundary",
]


@dataclass(frozen=True)
class RidgeBlockScorer:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray

    def score_matrix(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Phase 21 scorer requires a 2D feature matrix")
        if values.shape[1] != self.feature_mean.shape[0]:
            raise ValueError(
                "Phase 21 train/evaluation per-block feature dimensions differ: "
                f"expected {self.feature_mean.shape[0]}, got {values.shape[1]}"
            )
        normalized = (values - self.feature_mean) / self.feature_scale
        design = np.column_stack([np.ones(values.shape[0]), normalized])
        return design @ self.coefficients


def build_phase21_cross_tile_scorer_contract(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] = ("B0", "B1"),
    train_tile_id: str | None = None,
    eval_tile_id: str | None = None,
    ridge_alpha: float = 1e-6,
    eval_max_steps: int = 4,
    seed: int = 0,
) -> dict[str, object]:
    if float(ridge_alpha) < 0.0:
        raise ValueError("ridge_alpha must be non-negative")
    if int(eval_max_steps) <= 0:
        raise ValueError("eval_max_steps must be positive")

    normalized_variants = _normalize_variants(variants)
    selected = _select_train_eval_tiles(
        Path(tile_index_csv),
        train_tile_id=train_tile_id,
        eval_tile_id=eval_tile_id,
    )
    return {
        "phase": "phase21_cross_tile_block_scorer",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "tile_index_csv": str(Path(tile_index_csv)),
        "variants": normalized_variants,
        "train_tile_id": selected["train_tile_id"],
        "eval_tile_id": selected["eval_tile_id"],
        "train_tile_selection": selected["train_tile_selection"],
        "eval_tile_selection": selected["eval_tile_selection"],
        "learned_policy_evaluation_scope": "cross_tile_per_block_scorer_pilot",
        "cross_tile_evaluation_status": "executed_distinct_tile",
        "ridge_alpha": float(ridge_alpha),
        "eval_max_steps": int(eval_max_steps),
        "seed": int(seed),
        "claim_boundary": PHASE21_CLAIM_BOUNDARY,
    }


def run_phase21_cross_tile_block_scorer(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] = ("B0", "B1"),
    train_tile_id: str | None = None,
    eval_tile_id: str | None = None,
    ridge_alpha: float = 1e-6,
    eval_max_steps: int = 4,
    seed: int = 0,
) -> dict[str, object]:
    contract = build_phase21_cross_tile_scorer_contract(
        phase2_output_dir,
        tile_index_csv,
        variants=variants,
        train_tile_id=train_tile_id,
        eval_tile_id=eval_tile_id,
        ridge_alpha=ridge_alpha,
        eval_max_steps=eval_max_steps,
        seed=seed,
    )

    summaries: list[dict[str, object]] = []
    traces: dict[str, dict[str, list[dict[str, object]]]] = {
        "learned_block_scorer": {},
        "first_valid": {},
        "seeded_random": {},
    }
    model_metadata: dict[str, dict[str, object]] = {}

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
        if train_tiled.state_matrix.shape[1] != eval_tiled.state_matrix.shape[1]:
            raise ValueError(
                "Phase 21 train/evaluation per-block feature dimensions differ: "
                f"{train_tiled.state_matrix.shape[1]} vs {eval_tiled.state_matrix.shape[1]}"
            )

        scorer, metadata = _fit_ridge_block_scorer(
            train_tiled,
            ridge_alpha=float(contract["ridge_alpha"]),
        )
        model_metadata[str(variant_id)] = metadata

        learned_summary, learned_steps = _evaluate_learned_scorer(
            scorer,
            eval_tiled,
            train_tile_id=str(contract["train_tile_id"]),
            train_n_blocks=len(train_tiled.block_ids),
            ridge_alpha=float(contract["ridge_alpha"]),
            eval_max_steps=int(contract["eval_max_steps"]),
            seed=int(seed),
        )
        summaries.append(learned_summary)
        traces["learned_block_scorer"][str(variant_id)] = learned_steps

        for policy_id in ("first_valid", "seeded_random"):
            baseline_summary, baseline_steps = _evaluate_baseline_policy(
                eval_tiled,
                policy_id=policy_id,
                train_tile_id=str(contract["train_tile_id"]),
                train_n_blocks=len(train_tiled.block_ids),
                ridge_alpha=float(contract["ridge_alpha"]),
                eval_max_steps=int(contract["eval_max_steps"]),
                seed=int(seed),
            )
            summaries.append(baseline_summary)
            traces[policy_id][str(variant_id)] = baseline_steps

    return {
        **contract,
        "all_evaluations_completed": all(
            bool(row["terminated"]) or bool(row["truncated"]) for row in summaries
        ),
        "summary_count": len(summaries),
        "summaries": summaries,
        "traces": traces,
        "model_metadata": model_metadata,
    }


def write_phase21_cross_tile_block_scorer_artifacts(
    protocol: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "phase21_cross_tile_block_scorer_summary.csv"
    traces_path = output_path / "phase21_cross_tile_block_scorer_traces.json"

    summaries = protocol.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("Phase 21 protocol is missing a summaries list")

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for summary in summaries:
            if not isinstance(summary, Mapping):
                raise ValueError("Phase 21 summary rows must be objects")
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


def _fit_ridge_block_scorer(tiled, ridge_alpha: float) -> tuple[RidgeBlockScorer, dict[str, object]]:
    matrix = np.asarray(tiled.state_matrix, dtype=np.float64)
    targets = _base_reward_targets(tiled)
    feature_mean = matrix.mean(axis=0)
    feature_scale = matrix.std(axis=0)
    feature_scale = np.where(feature_scale < 1e-12, 1.0, feature_scale)
    normalized = (matrix - feature_mean) / feature_scale
    design = np.column_stack([np.ones(matrix.shape[0]), normalized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge_alpha)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ targets
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(lhs) @ rhs

    scorer = RidgeBlockScorer(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
    )
    metadata = {
        "model_type": "standardized_ridge_linear",
        "ridge_alpha": float(ridge_alpha),
        "train_n_blocks": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "target_min": _round_float(float(targets.min())),
        "target_max": _round_float(float(targets.max())),
        "target_mean": _round_float(float(targets.mean())),
        "coefficient_l2": _round_float(float(np.linalg.norm(coefficients[1:]))),
        "intercept": _round_float(float(coefficients[0])),
    }
    return scorer, metadata


def _base_reward_targets(tiled) -> np.ndarray:
    return np.asarray(
        [
            compute_base_planning_reward_from_matrix_row(
                tiled.feature_columns,
                tiled.state_matrix[index],
            )
            for index in range(tiled.state_matrix.shape[0])
        ],
        dtype=np.float64,
    )


def _evaluate_learned_scorer(
    scorer: RidgeBlockScorer,
    tiled,
    train_tile_id: str,
    train_n_blocks: int,
    ridge_alpha: float,
    eval_max_steps: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    env = Phase4InputContractEnv(tiled, max_steps=eval_max_steps)
    obs, info = env.reset(seed=seed)
    scores = scorer.score_matrix(tiled.state_matrix)
    steps: list[dict[str, object]] = []
    selected_block_ids: list[str] = []
    total_reward = 0.0
    terminated = False
    truncated = False

    while True:
        valid_actions = _valid_actions(env.action_masks())
        if not valid_actions:
            break
        action = max(valid_actions, key=lambda index: (float(scores[index]), -index))
        obs, reward, terminated, truncated, step_info = env.step(action)
        reward_value = float(reward)
        total_reward += reward_value
        selected_block_id = str(step_info["selected_block_id"])
        selected_block_ids.append(selected_block_id)
        steps.append(
            _step_record(
                step_info,
                action,
                reward_value,
                env,
                predicted_score=float(scores[action]),
            )
        )
        if terminated or truncated:
            break

    return (
        _summary_row(
            row_type="learned_block_scorer",
            variant_id=str(info["variant_id"]),
            train_tile_id=train_tile_id,
            eval_tile_id=tiled.tile_id,
            seed=seed,
            ridge_alpha=ridge_alpha,
            eval_max_steps=eval_max_steps,
            train_n_blocks=train_n_blocks,
            info=info,
            eval_obs_shape=int(obs.shape[0]),
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
    train_n_blocks: int,
    ridge_alpha: float,
    eval_max_steps: int,
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
            ridge_alpha=ridge_alpha,
            eval_max_steps=eval_max_steps,
            train_n_blocks=train_n_blocks,
            info=info,
            eval_obs_shape=int(obs.shape[0]),
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
    ridge_alpha: float,
    eval_max_steps: int,
    train_n_blocks: int,
    info: Mapping[str, object],
    eval_obs_shape: int,
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
        "ridge_alpha": float(ridge_alpha),
        "eval_max_steps": int(eval_max_steps),
        "train_n_blocks": int(train_n_blocks),
        "eval_n_blocks": int(info["n_blocks"]),
        "n_features": int(info["n_features"]),
        "eval_observation_shape": int(eval_obs_shape),
        "action_space_n": int(action_space_n),
        "episode_steps": int(episode_steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "total_contract_reward": _round_float(total_reward),
        "selected_block_ids": selected_block_ids,
        "claim_boundary": PHASE21_CLAIM_BOUNDARY,
    }


def _step_record(
    step_info: Mapping[str, object],
    action: int,
    reward: float,
    env: Phase4InputContractEnv,
    predicted_score: float | None = None,
) -> dict[str, object]:
    record = {
        "step": int(step_info["step"]),
        "action": int(action),
        "selected_block_id": str(step_info["selected_block_id"]),
        "reward": _round_float(reward),
        "valid_actions_after": int(env.action_masks().sum()),
        "terminated": bool(step_info.get("terminated", False)),
    }
    if predicted_score is not None:
        record["predicted_score"] = _round_float(predicted_score)
    return record


def _select_train_eval_tiles(
    tile_index_csv: Path,
    train_tile_id: str | None,
    eval_tile_id: str | None,
) -> dict[str, str]:
    rows = sorted(_read_tile_rows(tile_index_csv), key=lambda row: -int(row["n_blocks"]))
    train_selection = "explicit" if train_tile_id else "largest"
    eval_selection = "explicit" if eval_tile_id else "next_largest_distinct"

    train_id = str(train_tile_id).strip() if train_tile_id else str(rows[0]["tile_id"])
    known = {str(row["tile_id"]) for row in rows}
    if train_id not in known:
        raise ValueError(f"Train tile ID not found: {train_id}")

    if eval_tile_id:
        eval_id = str(eval_tile_id).strip()
        if eval_id not in known:
            raise ValueError(f"Evaluation tile ID not found: {eval_id}")
    else:
        eval_candidates = [str(row["tile_id"]) for row in rows if row["tile_id"] != train_id]
        if not eval_candidates:
            raise ValueError("Phase 21 requires a distinct evaluation tile")
        eval_id = eval_candidates[0]

    if eval_id == train_id:
        raise ValueError("Phase 21 train and evaluation tiles must be distinct")

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
        raise ValueError("At least one Phase 21 variant must be requested")
    unsupported = [variant for variant in normalized if variant not in {"B0", "B1"}]
    if unsupported:
        raise ValueError(
            "Phase 21 is restricted to B0/B1 base-reward variants; "
            f"unsupported variant: {unsupported[0]}"
        )
    return normalized


def _read_tile_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 21 tile index CSV: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [field for field in ("tile_id", "block_ids") if field not in fieldnames]
        if missing:
            raise ValueError(f"Phase 21 tile index is missing columns: {missing}")
        for row in reader:
            tile_id = str(row.get("tile_id", "")).strip()
            block_ids = [
                part.strip()
                for part in str(row.get("block_ids", "")).split(";")
                if part.strip()
            ]
            if not tile_id:
                raise ValueError("Phase 21 tile index contains a row without tile_id")
            if not block_ids:
                raise ValueError(f"Tile {tile_id} contains no block IDs")
            rows.append({"tile_id": tile_id, "block_ids": block_ids, "n_blocks": len(block_ids)})
    if len(rows) < 2:
        raise ValueError("Phase 21 requires at least two non-empty tile rows")
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
    raise ValueError(f"Unknown Phase 21 baseline policy: {policy_id}")


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


def _round_float(value: float) -> float:
    return round(float(value), 10)
