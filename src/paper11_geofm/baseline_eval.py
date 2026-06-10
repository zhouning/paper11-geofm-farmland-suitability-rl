from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .drl_smoke_env import make_phase4_smoke_env


PHASE6_CLAIM_BOUNDARY = (
    "Phase 6 is a non-learning masked baseline evaluator; it does not train "
    "or evaluate a DRL policy and does not report planning performance."
)

SUMMARY_FIELDNAMES = [
    "policy_id",
    "variant_id",
    "seed",
    "n_blocks",
    "n_features",
    "observation_shape",
    "action_space_n",
    "reward_mode",
    "max_steps",
    "episode_steps",
    "terminated",
    "truncated",
    "valid_action_rate",
    "total_contract_reward",
    "selected_block_ids",
    "claim_boundary",
]


def run_phase6_baseline_evaluator(
    phase2_output_dir: Path | str,
    variant_ids: Sequence[str] = ("B0", "B1", "B2", "B3"),
    policy_ids: Sequence[str] = ("first_valid", "seeded_random"),
    max_steps: int | None = None,
    seed: int = 0,
) -> dict[str, object]:
    normalized_variant_ids = _normalize_ids(variant_ids, "variant")
    normalized_policy_ids = _normalize_ids(policy_ids, "policy")
    summaries: list[dict[str, object]] = []
    traces: dict[str, dict[str, list[dict[str, object]]]] = {}

    for policy_id in normalized_policy_ids:
        if policy_id not in {"first_valid", "seeded_random"}:
            raise ValueError(f"Unknown Phase 6 policy: {policy_id}")
        traces[policy_id] = {}
        for variant_id in normalized_variant_ids:
            summary, steps = _run_one_baseline(
                phase2_output_dir,
                variant_id,
                policy_id,
                max_steps=max_steps,
                seed=seed,
            )
            summaries.append(summary)
            traces[policy_id][variant_id] = steps

    return {
        "phase": "phase6_masked_baseline_evaluator",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "variant_ids": normalized_variant_ids,
        "policy_ids": normalized_policy_ids,
        "seed": int(seed),
        "max_steps_requested": max_steps,
        "claim_boundary": PHASE6_CLAIM_BOUNDARY,
        "summaries": summaries,
        "traces": traces,
    }


def write_phase6_baseline_artifacts(
    protocol: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "phase6_baseline_summary.csv"
    traces_path = output_path / "phase6_baseline_traces.json"

    summaries = protocol.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("Phase 6 protocol is missing a summaries list")

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for summary in summaries:
            if not isinstance(summary, Mapping):
                raise ValueError("Phase 6 summary rows must be objects")
            row = {field: summary.get(field, "") for field in SUMMARY_FIELDNAMES}
            selected = row.get("selected_block_ids")
            if isinstance(selected, list):
                row["selected_block_ids"] = ";".join(str(item) for item in selected)
            writer.writerow(row)

    traces_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"summary_csv": summary_path, "traces_json": traces_path}


def _run_one_baseline(
    phase2_output_dir: Path | str,
    variant_id: str,
    policy_id: str,
    max_steps: int | None,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    env = make_phase4_smoke_env(
        phase2_output_dir,
        variant_id,
        max_steps=max_steps,
    )
    obs, info = env.reset()
    rng = _rng_for(seed, policy_id, variant_id)
    steps: list[dict[str, object]] = []
    selected_block_ids: list[str] = []
    total_contract_reward = 0.0
    valid_attempts = 0
    terminated = False
    truncated = False

    while True:
        valid_actions = _valid_actions(env.action_masks())
        if not valid_actions:
            break

        action = _select_action(policy_id, valid_actions, rng)
        valid_actions_before = len(valid_actions)
        _, reward, terminated, truncated, step_info = env.step(action)
        reward_value = float(reward)
        selected_block_id = str(step_info["selected_block_id"])
        selected_block_ids.append(selected_block_id)
        total_contract_reward += reward_value
        valid_attempts += 1
        steps.append(
            {
                "step": int(step_info["step"]),
                "action": int(action),
                "selected_block_id": selected_block_id,
                "reward": _round_float(reward_value),
                "valid_actions_before": valid_actions_before,
                "valid_actions_after": int(env.action_masks().sum()),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        if terminated or truncated:
            break

    episode_steps = len(steps)
    valid_action_rate = (
        float(valid_attempts / episode_steps) if episode_steps else 0.0
    )
    summary = {
        "policy_id": policy_id,
        "variant_id": variant_id,
        "seed": int(seed),
        "n_blocks": int(info["n_blocks"]),
        "n_features": int(info["n_features"]),
        "observation_shape": int(obs.shape[0]),
        "action_space_n": int(env.action_space.n),
        "reward_mode": str(info["reward_mode"]),
        "max_steps": int(env.max_steps),
        "episode_steps": episode_steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "valid_action_rate": _round_float(valid_action_rate),
        "total_contract_reward": _round_float(total_contract_reward),
        "selected_block_ids": selected_block_ids,
        "claim_boundary": PHASE6_CLAIM_BOUNDARY,
    }
    return summary, steps


def _normalize_ids(ids: Sequence[str], label: str) -> list[str]:
    normalized = [str(item).strip() for item in ids]
    normalized = [item for item in normalized if item]
    if not normalized:
        raise ValueError(f"At least one Phase 6 {label} must be requested")
    if label == "variant":
        return [item.upper() for item in normalized]
    return normalized


def _valid_actions(mask) -> list[int]:
    return [int(index) for index, valid in enumerate(mask.tolist()) if bool(valid)]


def _select_action(
    policy_id: str,
    valid_actions: list[int],
    rng: np.random.Generator,
) -> int:
    if policy_id == "first_valid":
        return valid_actions[0]
    if policy_id == "seeded_random":
        return int(rng.choice(valid_actions))
    raise ValueError(f"Unknown Phase 6 policy: {policy_id}")


def _rng_for(seed: int, policy_id: str, variant_id: str) -> np.random.Generator:
    key = f"{int(seed)}:{policy_id}:{variant_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    child_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(child_seed)


def _round_float(value: float) -> float:
    return round(float(value), 10)
