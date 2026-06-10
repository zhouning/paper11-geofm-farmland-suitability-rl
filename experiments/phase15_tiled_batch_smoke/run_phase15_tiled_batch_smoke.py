from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.tiled_batch_smoke import (
    PHASE15_CLAIM_BOUNDARY,
    run_phase15_tiled_batch_smoke,
    write_phase15_tiled_batch_smoke,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-step tile-level input-contract smoke checks across a "
            "Phase 13 tile index without training a policy."
        )
    )
    parser.add_argument(
        "--phase2-output-dir",
        type=Path,
        required=True,
        help="Directory containing Phase 2 experiment_variants.json and variant CSVs.",
    )
    parser.add_argument(
        "--tile-index-csv",
        type=Path,
        required=True,
        help="Path to Phase 13 phase13_tile_index.csv.",
    )
    parser.add_argument(
        "--variant",
        default="B1",
        help="Representation-only variant to batch smoke-check. Default: B1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where Phase 15 CSV/JSON artifacts will be written.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Optional cap on number of tile rows for quick checks.",
    )
    args = parser.parse_args(argv)

    try:
        report = run_phase15_tiled_batch_smoke(
            args.phase2_output_dir,
            args.tile_index_csv,
            variant_id=args.variant,
            max_tiles=args.max_tiles,
        )
        paths = write_phase15_tiled_batch_smoke(report, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Variant: {report['variant_id']}")
    print(f"Tiles processed: {report['tile_count']}")
    print(f"Total blocks: {report['total_blocks']}")
    print(f"Max observation shape: {report['max_observation_shape']}")
    print(f"All passed: {report['all_tile_smokes_passed']}")
    print(f"Summary CSV: {paths['summary_csv']}")
    print(f"Report JSON: {paths['report_json']}")
    print(f"Claim boundary: {PHASE15_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
