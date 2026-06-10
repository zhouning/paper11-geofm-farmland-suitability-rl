from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.tiled_contract import (
    DEFAULT_OBSERVATION_THRESHOLD,
    DEFAULT_TILE_COLS,
    DEFAULT_TILE_ROWS,
    PHASE13_CLAIM_BOUNDARY,
    build_phase13_tiled_contract,
    write_phase13_tiled_contract,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a tiled real-data contract from Phase 11 block mappings "
            "and Phase 2 variant metadata without training a policy."
        )
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        required=True,
        help="Path to Phase 11 block_pixel_mapping.csv.",
    )
    parser.add_argument(
        "--variant-manifest",
        type=Path,
        required=True,
        help="Path to Phase 2 experiment_variants.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where Phase 13 tile artifacts will be written.",
    )
    parser.add_argument(
        "--tile-rows",
        type=int,
        default=DEFAULT_TILE_ROWS,
        help=f"AlphaEarth grid rows per tile. Default: {DEFAULT_TILE_ROWS}.",
    )
    parser.add_argument(
        "--tile-cols",
        type=int,
        default=DEFAULT_TILE_COLS,
        help=f"AlphaEarth grid columns per tile. Default: {DEFAULT_TILE_COLS}.",
    )
    parser.add_argument(
        "--observation-threshold",
        type=int,
        default=DEFAULT_OBSERVATION_THRESHOLD,
        help=(
            "Maximum tiled observation dimension allowed for tiled contract "
            f"readiness. Default: {DEFAULT_OBSERVATION_THRESHOLD}."
        ),
    )
    args = parser.parse_args(argv)

    try:
        report = build_phase13_tiled_contract(
            args.mapping_csv,
            args.variant_manifest,
            tile_rows=args.tile_rows,
            tile_cols=args.tile_cols,
            observation_threshold=args.observation_threshold,
        )
        paths = write_phase13_tiled_contract(report, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    b3 = report["variants"]["B3"]
    print(f"Blocks: {report['total_blocks']}")
    print(f"Tiles: {report['tile_count']}")
    print(f"Max blocks per tile: {report['block_count_summary']['max']}")
    print(
        "B3 max tile observation dimension: "
        f"{b3['max_tile_observation_dimension']}"
    )
    print(
        "All tiles within observation threshold: "
        f"{report['all_tiles_within_observation_threshold']}"
    )
    print(f"Tiled contract ready: {report['tiled_contract_ready']}")
    print(f"Recommendation: {report['recommendation']}")
    print(f"Tile index CSV: {paths['tile_index']}")
    print(f"Summary JSON: {paths['summary']}")
    print(f"Claim boundary: {PHASE13_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
