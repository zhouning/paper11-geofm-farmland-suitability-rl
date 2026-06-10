from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .drl_inputs import load_variant_input
from .drl_smoke_env import Phase4InputContractEnv
from .tiled_inputs import TiledVariantInput


PHASE15_CLAIM_BOUNDARY = (
    "Phase 15 is a batch tile-level input-contract smoke check; it does not "
    "train, tune, evaluate, or compare a DRL policy and does not enable "
    "suitability reward."
)

SUMMARY_FIELDNAMES = [
    "tile_id",
    "variant_id",
    "n_blocks",
    "n_features",
    "observation_shape",
    "action_space_n",
    "selected_block_id",
    "step_reward",
    "reward_mode",
    "status",
]


def run_phase15_tiled_batch_smoke(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variant_id: str = "B1",
    max_tiles: int | None = None,
) -> dict[str, object]:
    if max_tiles is not None and int(max_tiles) <= 0:
        raise ValueError("max_tiles must be positive when provided")

    loaded = load_variant_input(phase2_output_dir, variant_id)
    if loaded.reward_mode == "base_plus_suitability_reward":
        raise ValueError(
            "Phase 15 suitability reward variants are disabled by default; "
            "use a representation-only variant such as B0 or B1"
        )

    tile_rows = _read_tile_rows(Path(tile_index_csv))
    if max_tiles is not None:
        tile_rows = tile_rows[: int(max_tiles)]

    block_positions = {
        block_id: index for index, block_id in enumerate(loaded.block_ids)
    }
    rows = []
    for tile in tile_rows:
        tiled = _tiled_input_for_row(loaded, Path(tile_index_csv), tile, block_positions)
        rows.append(_run_one_tile(tiled))

    block_counts = [int(row["n_blocks"]) for row in rows]
    max_observation_shape = max(
        (int(row["observation_shape"]) for row in rows),
        default=0,
    )
    all_passed = all(row["status"] == "passed" for row in rows)
    return {
        "phase": "phase15_tiled_batch_smoke",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "tile_index_csv": str(Path(tile_index_csv)),
        "variant_id": loaded.variant_id,
        "tile_count": len(rows),
        "total_blocks": sum(block_counts),
        "block_count_summary": _block_count_summary(block_counts),
        "max_observation_shape": max_observation_shape,
        "all_tile_smokes_passed": all_passed,
        "recommendation": (
            "all_tiles_passed_representation_only_input_contract"
            if all_passed
            else "inspect_failed_tile_contracts"
        ),
        "rows": rows,
        "claim_boundary": PHASE15_CLAIM_BOUNDARY,
    }


def write_phase15_tiled_batch_smoke(
    report: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_csv = output_path / "phase15_tiled_batch_smoke_summary.csv"
    report_json = output_path / "phase15_tiled_batch_smoke_report.json"

    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Phase 15 report is missing rows")

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Phase 15 summary rows must be objects")
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDNAMES})

    report_payload = {key: value for key, value in report.items() if key != "rows"}
    report_payload["artifacts"] = {
        "summary_csv": summary_csv.name,
        "report_json": report_json.name,
    }
    report_json.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"summary_csv": summary_csv, "report_json": report_json}


def _read_tile_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 15 tile index CSV: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [field for field in ("tile_id", "block_ids") if field not in fieldnames]
        if missing:
            raise ValueError(f"Phase 15 tile index is missing columns: {missing}")
        for row in reader:
            tile_id = str(row.get("tile_id", "")).strip()
            block_ids = [
                part.strip()
                for part in str(row.get("block_ids", "")).split(";")
                if part.strip()
            ]
            if not tile_id:
                raise ValueError("Phase 15 tile index contains a row without tile_id")
            if not block_ids:
                raise ValueError(f"Tile {tile_id} contains no block IDs")
            rows.append({"tile_id": tile_id, "block_ids": block_ids})
    if not rows:
        raise ValueError("Phase 15 tile index contains no tile rows")
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
        claim_boundary=PHASE15_CLAIM_BOUNDARY,
    )


def _run_one_tile(tiled: TiledVariantInput) -> dict[str, object]:
    env = Phase4InputContractEnv(tiled)
    obs, info = env.reset()
    valid_actions = [
        index for index, valid in enumerate(env.action_masks().tolist()) if valid
    ]
    if not valid_actions:
        raise ValueError(f"Tile {tiled.tile_id} has no valid actions")
    action = valid_actions[0]
    _, reward, _, _, step_info = env.step(action)
    return {
        "tile_id": tiled.tile_id,
        "variant_id": tiled.variant_id,
        "n_blocks": len(tiled.block_ids),
        "n_features": len(tiled.feature_columns),
        "observation_shape": int(obs.shape[0]),
        "action_space_n": int(env.action_space.n),
        "selected_block_id": str(step_info["selected_block_id"]),
        "step_reward": round(float(reward), 10),
        "reward_mode": str(info["reward_mode"]),
        "status": "passed",
    }


def _block_count_summary(counts: list[int]) -> dict[str, object]:
    if not counts:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(counts),
        "max": max(counts),
        "mean": round(sum(counts) / len(counts), 6),
    }
