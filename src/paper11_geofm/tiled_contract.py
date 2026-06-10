from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PHASE13_CLAIM_BOUNDARY = (
    "Phase 13 builds tiled real-data contract metadata only; it does not "
    "train, tune, evaluate, or compare a DRL policy and does not enable "
    "suitability reward."
)
DEFAULT_TILE_ROWS = 8
DEFAULT_TILE_COLS = 8
DEFAULT_OBSERVATION_THRESHOLD = 1_000_000
REQUIRED_VARIANTS = ("B0", "B1", "B2", "B3")

TILE_INDEX_FIELDNAMES = [
    "tile_id",
    "tile_row",
    "tile_col",
    "n_blocks",
    "min_grid_row",
    "max_grid_row",
    "min_grid_col",
    "max_grid_col",
    "block_ids",
]


def build_phase13_tiled_contract(
    mapping_csv: Path | str,
    variant_manifest_path: Path | str,
    tile_rows: int = DEFAULT_TILE_ROWS,
    tile_cols: int = DEFAULT_TILE_COLS,
    observation_threshold: int = DEFAULT_OBSERVATION_THRESHOLD,
) -> dict[str, object]:
    tile_rows_value = _positive_int(tile_rows, "tile_rows")
    tile_cols_value = _positive_int(tile_cols, "tile_cols")
    threshold = _positive_int(observation_threshold, "observation_threshold")
    mapping_path = Path(mapping_csv)
    manifest_path = Path(variant_manifest_path)

    mapping_rows = _read_mapping_rows(mapping_path)
    manifest = _read_json_object(manifest_path)
    tiles = _build_tiles(mapping_rows, tile_rows_value, tile_cols_value)
    variants = _build_variant_summary(manifest, tiles, threshold)

    total_blocks = len(mapping_rows)
    block_count_summary = _block_count_summary(tiles)
    all_variants_ready = all(
        bool(variants[variant_id]["ready"]) for variant_id in REQUIRED_VARIANTS
    )
    all_tiles_within_threshold = all(
        bool(variant["all_tiles_within_observation_threshold"])
        for variant in variants.values()
    )
    tiled_contract_ready = bool(
        total_blocks > 0 and all_variants_ready and all_tiles_within_threshold
    )

    return {
        "phase": "phase13_tiled_real_contract",
        "mapping_csv": str(mapping_path),
        "variant_manifest": str(manifest_path),
        "tile_rows": tile_rows_value,
        "tile_cols": tile_cols_value,
        "observation_threshold": threshold,
        "total_blocks": total_blocks,
        "tile_count": len(tiles),
        "block_count_summary": block_count_summary,
        "variants": variants,
        "all_required_variants_ready": all_variants_ready,
        "all_tiles_within_observation_threshold": all_tiles_within_threshold,
        "tiled_contract_ready": tiled_contract_ready,
        "recommendation": _recommendation(
            total_blocks,
            all_variants_ready,
            all_tiles_within_threshold,
        ),
        "tiles": tiles,
        "claim_boundary": PHASE13_CLAIM_BOUNDARY,
    }


def write_phase13_tiled_contract(
    report: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tile_index_path = output_path / "phase13_tile_index.csv"
    summary_path = output_path / "phase13_tiled_real_contract.json"

    tiles = report.get("tiles")
    if not isinstance(tiles, list):
        raise ValueError("Phase 13 report is missing a tiles list")

    with tile_index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TILE_INDEX_FIELDNAMES)
        writer.writeheader()
        for tile in tiles:
            if not isinstance(tile, Mapping):
                raise ValueError("Phase 13 tile rows must be objects")
            row = {field: tile.get(field, "") for field in TILE_INDEX_FIELDNAMES}
            if isinstance(row.get("block_ids"), list):
                row["block_ids"] = ";".join(str(item) for item in row["block_ids"])
            writer.writerow(row)

    summary_payload = {
        key: value
        for key, value in report.items()
        if key != "tiles"
    }
    summary_payload["artifacts"] = {
        "tile_index": tile_index_path.name,
        "summary": summary_path.name,
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"tile_index": tile_index_path, "summary": summary_path}


def _read_mapping_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 13 mapping CSV: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [field for field in ("block_id", "row", "col") if field not in fieldnames]
        if missing:
            raise ValueError(f"Phase 13 mapping CSV is missing columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            block_id = str(row.get("block_id", "")).strip()
            if not block_id:
                raise ValueError(f"Missing block_id at {path}:{row_number}")
            grid_row = _non_negative_int(row.get("row"), path, row_number, "row")
            grid_col = _non_negative_int(row.get("col"), path, row_number, "col")
            rows.append(
                {
                    "block_id": block_id,
                    "row": grid_row,
                    "col": grid_col,
                    "input_order": len(rows),
                }
            )
    if not rows:
        raise ValueError("Phase 13 mapping CSV contains no block rows")
    return rows


def _build_tiles(
    mapping_rows: list[dict[str, object]],
    tile_rows: int,
    tile_cols: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in mapping_rows:
        tile_row = math.floor(int(row["row"]) / tile_rows)
        tile_col = math.floor(int(row["col"]) / tile_cols)
        grouped.setdefault((tile_row, tile_col), []).append(row)

    tiles: list[dict[str, object]] = []
    for tile_row, tile_col in sorted(grouped):
        rows = sorted(grouped[(tile_row, tile_col)], key=lambda item: int(item["input_order"]))
        grid_rows = [int(row["row"]) for row in rows]
        grid_cols = [int(row["col"]) for row in rows]
        tiles.append(
            {
                "tile_id": f"tile_r{tile_row:03d}_c{tile_col:03d}",
                "tile_row": tile_row,
                "tile_col": tile_col,
                "n_blocks": len(rows),
                "min_grid_row": min(grid_rows),
                "max_grid_row": max(grid_rows),
                "min_grid_col": min(grid_cols),
                "max_grid_col": max(grid_cols),
                "block_ids": [str(row["block_id"]) for row in rows],
            }
        )
    return tiles


def _build_variant_summary(
    manifest: Mapping[str, object],
    tiles: list[dict[str, object]],
    threshold: int,
) -> dict[str, dict[str, object]]:
    variants = manifest.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError("Phase 2 experiment_variants.json is missing variants")

    summary: dict[str, dict[str, object]] = {}
    for variant_id in REQUIRED_VARIANTS:
        variant = variants.get(variant_id)
        if not isinstance(variant, Mapping):
            summary[variant_id] = _missing_variant_summary(threshold)
            continue
        required_columns = variant.get("required_columns", [])
        if not isinstance(required_columns, list):
            raise ValueError(f"Variant {variant_id} required_columns must be a list")
        n_features = len(required_columns)
        dimensions = [
            int(tile["n_blocks"]) * n_features + 3
            for tile in tiles
        ]
        max_dimension = max(dimensions, default=0)
        summary[variant_id] = {
            "ready": bool(variant.get("ready")),
            "missing": list(variant.get("missing", [])),
            "n_features": n_features,
            "reward_mode": str(variant.get("reward", "")),
            "feature_table": variant.get("feature_table"),
            "state_groups": list(variant.get("state_groups", [])),
            "max_tile_observation_dimension": max_dimension,
            "min_tile_observation_dimension": min(dimensions, default=0),
            "all_tiles_within_observation_threshold": max_dimension <= threshold,
        }
    return summary


def _missing_variant_summary(threshold: int) -> dict[str, object]:
    return {
        "ready": False,
        "missing": ["variant_metadata"],
        "n_features": 0,
        "reward_mode": "",
        "feature_table": None,
        "state_groups": [],
        "max_tile_observation_dimension": 0,
        "min_tile_observation_dimension": 0,
        "all_tiles_within_observation_threshold": 0 <= threshold,
    }


def _block_count_summary(tiles: list[dict[str, object]]) -> dict[str, object]:
    counts = [int(tile["n_blocks"]) for tile in tiles]
    if not counts:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(counts),
        "max": max(counts),
        "mean": round(sum(counts) / len(counts), 6),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 13 variant manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Phase 13 variant manifest must be a JSON object: {path}")
    return payload


def _recommendation(
    total_blocks: int,
    all_variants_ready: bool,
    all_tiles_within_threshold: bool,
) -> str:
    if total_blocks <= 0:
        return "repair_mapping_before_tiled_contract"
    if not all_variants_ready:
        return "repair_variant_manifest_before_tiled_contract"
    if not all_tiles_within_threshold:
        return "increase_tile_partitioning_or_raise_observation_threshold"
    return "use_tiled_contract_for_representation_only_environment_design"


def _positive_int(value: object, label: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _non_negative_int(value: object, path: Path, row_number: int, column: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid integer at {path}:{row_number} column {column}: {value!r}"
        ) from exc
    if parsed < 0:
        raise ValueError(
            f"Negative grid index at {path}:{row_number} column {column}: {parsed}"
        )
    return parsed
