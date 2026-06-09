from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .drl_smoke_env import make_phase4_smoke_env


PHASE5_CLAIM_BOUNDARY = (
    "Phase 5 is a deterministic rollout-protocol smoke check; it does not "
    "train or evaluate a policy and does not report planning performance."
)

SUMMARY_FIELDNAMES = [
    "variant_id",
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


def run_phase5_rollout_protocol(
    phase2_output_dir: Path | str,
    variant_ids: Sequence[str] = ("B0", "B1", "B2", "B3"),
    max_steps: int | None = None,
) -> dict[str, object]:
    normalized_variant_ids = _normalize_variant_ids(variant_ids)
    summaries: list[dict[str, object]] = []
    steps_by_variant: dict[str, list[dict[str, object]]] = {}

    for variant_id in normalized_variant_ids:
        env = make_phase4_smoke_env(
            phase2_output_dir,
            variant_id,
            max_steps=max_steps,
        )
        obs, info = env.reset()
        steps: list[dict[str, object]] = []
        selected_block_ids: list[str] = []
        total_contract_reward = 0.0
        valid_attempts = 0
        terminated = False
        truncated = False

        while True:
            mask = env.action_masks()
            valid_actions = [
                int(index)
                for index, valid in enumerate(mask.tolist())
                if bool(valid)
            ]
            if not valid_actions:
                break

            action = valid_actions[0]
            valid_actions_before = len(valid_actions)
            _, reward, terminated, truncated, step_info = env.step(action)
            reward_value = float(reward)
            selected_block_id = str(step_info["selected_block_id"])
            selected_block_ids.append(selected_block_id)
            total_contract_reward += reward_value
            valid_attempts += 1
            valid_actions_after = int(env.action_masks().sum())
            steps.append(
                {
                    "step": int(step_info["step"]),
                    "action": action,
                    "selected_block_id": selected_block_id,
                    "reward": _round_float(reward_value),
                    "valid_actions_before": valid_actions_before,
                    "valid_actions_after": valid_actions_after,
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
        summaries.append(
            {
                "variant_id": variant_id,
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
                "claim_boundary": PHASE5_CLAIM_BOUNDARY,
            }
        )
        steps_by_variant[variant_id] = steps

    return {
        "phase": "phase5_rollout_protocol_smoke",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "variant_ids": normalized_variant_ids,
        "max_steps_requested": max_steps,
        "claim_boundary": PHASE5_CLAIM_BOUNDARY,
        "summaries": summaries,
        "steps": steps_by_variant,
    }


def write_phase5_rollout_artifacts(
    protocol: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "phase5_rollout_summary.csv"
    steps_path = output_path / "phase5_rollout_steps.json"

    summaries = protocol.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("Phase 5 protocol is missing a summaries list")

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for summary in summaries:
            if not isinstance(summary, Mapping):
                raise ValueError("Phase 5 summary rows must be objects")
            row = {field: summary.get(field, "") for field in SUMMARY_FIELDNAMES}
            selected = row.get("selected_block_ids")
            if isinstance(selected, list):
                row["selected_block_ids"] = ";".join(str(item) for item in selected)
            writer.writerow(row)

    steps_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"summary_csv": summary_path, "steps_json": steps_path}


def _normalize_variant_ids(variant_ids: Sequence[str]) -> list[str]:
    normalized = [str(variant_id).strip().upper() for variant_id in variant_ids]
    normalized = [variant_id for variant_id in normalized if variant_id]
    if not normalized:
        raise ValueError("At least one Phase 5 variant must be requested")
    return normalized


def _round_float(value: float) -> float:
    return round(float(value), 10)
