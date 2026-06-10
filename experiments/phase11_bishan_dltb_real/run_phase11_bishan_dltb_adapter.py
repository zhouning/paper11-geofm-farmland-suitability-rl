from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.dltb_adapter import (
    PHASE11_CLAIM_BOUNDARY,
    build_bishan_dltb_phase2_inputs,
    write_bishan_dltb_phase2_inputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build Paper11 Phase 2 CSV inputs from real Bishan DLTB polygons "
            "with centroid-to-AlphaEarth-grid assignment."
        )
    )
    parser.add_argument(
        "--dltb-path",
        type=Path,
        required=True,
        help="Path to Bishan DLTB_with_slope.gpkg or equivalent DLTB source.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=ROOT / "data" / "bishan_alphaearth_sample" / "metadata.json",
        help="Path to Bishan AlphaEarth metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where Phase 11 adapter artifacts will be written.",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Optional deterministic cap for local smoke checks.",
    )
    args = parser.parse_args(argv)

    try:
        payload = build_bishan_dltb_phase2_inputs(
            args.dltb_path,
            args.metadata_path,
            max_blocks=args.max_blocks,
        )
        paths = write_bishan_dltb_phase2_inputs(payload, args.output_dir)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    summary = payload["summary"]
    print(f"Rows exported: {summary['rows_exported']}")
    print(f"Rows read in bbox: {summary['rows_read_in_bbox']}")
    print(f"Category counts: {summary['category_counts']}")
    print(f"Label positive counts: {summary['label_positive_counts']}")
    print(f"Mapping CSV: {paths['mapping_csv']}")
    print(f"Attributes CSV: {paths['attributes_csv']}")
    print(f"Summary: {paths['summary']}")
    print(f"Claim boundary: {PHASE11_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

