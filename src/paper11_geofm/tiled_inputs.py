from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .drl_inputs import load_variant_input
from .drl_smoke_env import Phase4InputContractEnv


PHASE14_CLAIM_BOUNDARY = (
    "Phase 14 is a tile-level input-contract smoke check; it does not train, "
    "tune, evaluate, or compare a DRL policy and does not enable suitability "
    "reward."
)


@dataclass(frozen=True)
class TiledVariantInput:
    tile_id: str
    variant_id: str
    block_ids: tuple[str, ...]
    feature_columns: tuple[str, ...]
    state_matrix: np.ndarray
    reward_mode: str
    state_groups: tuple[str, ...]
    source_table: Path
    tile_index_csv: Path
    claim_boundary: str = PHASE14_CLAIM_BOUNDARY


def load_tiled_variant_input(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    tile_id: str,
    variant_id: str = "B1",
    allow_suitability_reward_contract: bool = False,
) -> TiledVariantInput:
    tile_path = Path(tile_index_csv)
    tile_block_ids = _read_tile_block_ids(tile_path, tile_id)
    loaded = load_variant_input(phase2_output_dir, variant_id)

    if (
        loaded.reward_mode == "base_plus_suitability_reward"
        and not allow_suitability_reward_contract
    ):
        raise ValueError(
            "Phase 14 suitability reward variants are disabled by default; "
            "use a representation-only variant such as B0 or B1"
        )

    block_positions = {
        block_id: index for index, block_id in enumerate(loaded.block_ids)
    }
    missing = [block_id for block_id in tile_block_ids if block_id not in block_positions]
    if missing:
        raise ValueError(
            f"Tile {tile_id} contains block IDs missing from variant "
            f"{loaded.variant_id}: {missing[:5]}"
        )
    indexes = [block_positions[block_id] for block_id in tile_block_ids]
    return TiledVariantInput(
        tile_id=str(tile_id),
        variant_id=loaded.variant_id,
        block_ids=tuple(tile_block_ids),
        feature_columns=loaded.feature_columns,
        state_matrix=loaded.state_matrix[indexes, :].astype(np.float32, copy=True),
        reward_mode=loaded.reward_mode,
        state_groups=loaded.state_groups,
        source_table=loaded.source_table,
        tile_index_csv=tile_path,
    )


def run_phase14_tiled_smoke(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    tile_id: str,
    variant_id: str = "B1",
    max_steps: int | None = None,
) -> dict[str, object]:
    tiled = load_tiled_variant_input(
        phase2_output_dir,
        tile_index_csv,
        tile_id,
        variant_id=variant_id,
    )
    env = Phase4InputContractEnv(tiled, max_steps=max_steps)
    obs, info = env.reset()
    mask = env.action_masks()
    valid_actions = [idx for idx, valid in enumerate(mask.tolist()) if valid]
    if not valid_actions:
        raise ValueError(f"Tile {tile_id} has no valid actions")
    action = valid_actions[0]
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    return {
        "phase": "phase14_tiled_smoke_env",
        "tile_id": tiled.tile_id,
        "variant_id": tiled.variant_id,
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "tile_index_csv": str(Path(tile_index_csv)),
        "source_table": str(tiled.source_table),
        "n_blocks": len(tiled.block_ids),
        "n_features": len(tiled.feature_columns),
        "observation_shape": int(obs.shape[0]),
        "next_observation_shape": int(next_obs.shape[0]),
        "action_space_n": int(env.action_space.n),
        "initial_valid_actions": len(valid_actions),
        "selected_action": int(action),
        "selected_block_id": str(step_info["selected_block_id"]),
        "step_reward": round(float(reward), 10),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reward_mode": str(info["reward_mode"]),
        "max_steps": int(env.max_steps),
        "claim_boundary": PHASE14_CLAIM_BOUNDARY,
    }


def write_phase14_tiled_smoke_summary(
    summary: dict[str, object],
    output_dir: Path | str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "phase14_tiled_smoke_summary.json"
    summary_path.write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_path


def _read_tile_block_ids(tile_index_csv: Path, tile_id: str) -> list[str]:
    if not tile_index_csv.exists():
        raise FileNotFoundError(f"Missing Phase 14 tile index CSV: {tile_index_csv}")
    with tile_index_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [field for field in ("tile_id", "block_ids") if field not in fieldnames]
        if missing:
            raise ValueError(f"Phase 14 tile index is missing columns: {missing}")
        for row in reader:
            if str(row.get("tile_id", "")).strip() != str(tile_id):
                continue
            block_ids = [
                part.strip()
                for part in str(row.get("block_ids", "")).split(";")
                if part.strip()
            ]
            if not block_ids:
                raise ValueError(f"Tile {tile_id} contains no block IDs")
            return block_ids
    raise ValueError(f"Tile ID not found in tile index: {tile_id}")
