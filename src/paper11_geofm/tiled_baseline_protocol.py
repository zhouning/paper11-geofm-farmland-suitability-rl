from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .drl_inputs import load_variant_input
from .drl_smoke_env import Phase4InputContractEnv
from .tiled_inputs import TiledVariantInput


PHASE16_CLAIM_BOUNDARY = (
    "Phase 16 is a tiled non-learning baseline protocol; it does not train, "
    "tune, evaluate, or compare a DRL policy, does not enable suitability "
    "reward, and does not report planning performance."
)

SUMMARY_FIELDNAMES = [
    "policy_id",
    "variant_id",
    "tile_id",
    "seed",
    "n_blocks",
    "n_features",
    "observation_shape",
    "action_space_n",
    "max_steps",
    "episode_steps",
    "terminated",
    "truncated",
    "valid_action_rate",
    "total_contract_reward",
    "selected_block_ids",
    "claim_boundary",
]


def run_phase16_tiled_baseline_protocol(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variant_id: str = "B1",
    policy_ids: Sequence[str] = ("first_valid", "seeded_random"),
    max_steps: int = 4,
    seed: int = 0,
    max_tiles: int | None = None,
) -> dict[str, object]:
    if int(max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    if max_tiles is not None and int(max_tiles) <= 0:
        raise ValueError("max_tiles must be positive when provided")

    normalized_policy_ids = _normalize_policy_ids(policy_ids)
    loaded = load_variant_input(phase2_output_dir, variant_id)
    if loaded.reward_mode == "base_plus_suitability_reward":
        raise ValueError(
            "Phase 16 suitability reward variants are disabled by default; "
            "use a representation-only variant such as B0 or B1"
        )

    tile_path = Path(tile_index_csv)
    tile_rows = _read_tile_rows(tile_path)
    if max_tiles is not None:
        tile_rows = tile_rows[: int(max_tiles)]

    block_positions = {
        block_id: index for index, block_id in enumerate(loaded.block_ids)
    }
    summaries: list[dict[str, object]] = []
    traces: dict[str, dict[str, list[dict[str, object]]]] = {
        policy_id: {} for policy_id in normalized_policy_ids
    }
    tile_block_counts = [len([str(item) for item in tile["block_ids"]]) for tile in tile_rows]

    for policy_id in normalized_policy_ids:
        for tile in tile_rows:
            tiled = _tiled_input_for_row(loaded, tile_path, tile, block_positions)
            summary, steps = _run_one_tile_baseline(
                tiled,
                policy_id,
                max_steps=int(max_steps),
                seed=int(seed),
            )
            summaries.append(summary)
            traces[policy_id][tiled.tile_id] = steps

    max_observation_shape = max(
        (int(summary["observation_shape"]) for summary in summaries),
        default=0,
    )
    return {
        "phase": "phase16_tiled_baseline_protocol",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "tile_index_csv": str(tile_path),
        "variant_id": loaded.variant_id,
        "policy_ids": normalized_policy_ids,
        "seed": int(seed),
        "max_steps_requested": int(max_steps),
        "tile_count": len(tile_rows),
        "summary_count": len(summaries),
        "total_blocks": sum(tile_block_counts),
        "max_observation_shape": max_observation_shape,
        "all_rollouts_completed": all(
            bool(summary["terminated"]) or bool(summary["truncated"])
            for summary in summaries
        ),
        "summaries": summaries,
        "traces": traces,
        "claim_boundary": PHASE16_CLAIM_BOUNDARY,
    }


def write_phase16_tiled_baseline_artifacts(
    protocol: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "phase16_tiled_baseline_summary.csv"
    traces_path = output_path / "phase16_tiled_baseline_traces.json"

    summaries = protocol.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("Phase 16 protocol is missing a summaries list")

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for summary in summaries:
            if not isinstance(summary, Mapping):
                raise ValueError("Phase 16 summary rows must be objects")
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


def _normalize_policy_ids(policy_ids: Sequence[str]) -> list[str]:
    normalized = [str(item).strip() for item in policy_ids]
    normalized = [item for item in normalized if item]
    if not normalized:
        raise ValueError("At least one Phase 16 policy must be requested")
    unsupported = [
        policy_id
        for policy_id in normalized
        if policy_id not in {"first_valid", "seeded_random"}
    ]
    if unsupported:
        raise ValueError(f"Unknown Phase 16 policy: {unsupported[0]}")
    return normalized


def _read_tile_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 16 tile index CSV: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [field for field in ("tile_id", "block_ids") if field not in fieldnames]
        if missing:
            raise ValueError(f"Phase 16 tile index is missing columns: {missing}")
        for row in reader:
            tile_id = str(row.get("tile_id", "")).strip()
            block_ids = [
                part.strip()
                for part in str(row.get("block_ids", "")).split(";")
                if part.strip()
            ]
            if not tile_id:
                raise ValueError("Phase 16 tile index contains a row without tile_id")
            if not block_ids:
                raise ValueError(f"Tile {tile_id} contains no block IDs")
            rows.append({"tile_id": tile_id, "block_ids": block_ids})
    if not rows:
        raise ValueError("Phase 16 tile index contains no tile rows")
    return rows


def _tiled_input_for_row(
    loaded,
    tile_index_csv: Path,
    tile: Mapping[str, object],
    block_positions: Mapping[str, int],
) -> TiledVariantInput:
    tile_id = str(tile["tile_id"])
    block_ids = [str(block_id) for block_id in tile["block_ids"]]
    missing = [block_id for block_id in block_ids if block_id not in block_positions]
    if missing:
        raise ValueError(
            f"Tile {tile_id} contains block IDs missing from variant "
            f"{loaded.variant_id}: {missing[:5]}"
        )
    indexes = [int(block_positions[block_id]) for block_id in block_ids]
    return TiledVariantInput(
        tile_id=tile_id,
        variant_id=loaded.variant_id,
        block_ids=tuple(block_ids),
        feature_columns=loaded.feature_columns,
        state_matrix=loaded.state_matrix[indexes, :].astype(np.float32, copy=True),
        reward_mode=loaded.reward_mode,
        state_groups=loaded.state_groups,
        source_table=loaded.source_table,
        tile_index_csv=tile_index_csv,
        claim_boundary=PHASE16_CLAIM_BOUNDARY,
    )


def _run_one_tile_baseline(
    tiled: TiledVariantInput,
    policy_id: str,
    max_steps: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    env = Phase4InputContractEnv(tiled, max_steps=max_steps)
    obs, info = env.reset()
    rng = _rng_for(seed, policy_id, tiled.variant_id, tiled.tile_id)
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
    return (
        {
            "policy_id": policy_id,
            "variant_id": tiled.variant_id,
            "tile_id": tiled.tile_id,
            "seed": int(seed),
            "n_blocks": int(info["n_blocks"]),
            "n_features": int(info["n_features"]),
            "observation_shape": int(obs.shape[0]),
            "action_space_n": int(env.action_space.n),
            "max_steps": int(env.max_steps),
            "episode_steps": episode_steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "valid_action_rate": _round_float(valid_action_rate),
            "total_contract_reward": _round_float(total_contract_reward),
            "selected_block_ids": selected_block_ids,
            "claim_boundary": PHASE16_CLAIM_BOUNDARY,
        },
        steps,
    )


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
    raise ValueError(f"Unknown Phase 16 policy: {policy_id}")


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
