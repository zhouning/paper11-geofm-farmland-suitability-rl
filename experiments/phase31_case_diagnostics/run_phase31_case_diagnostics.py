from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase31_case_diagnostics import (
    build_phase31_case_diagnostics,
    write_phase31_case_diagnostics_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the read-only Paper11 Phase 31 case diagnostics over "
            "existing Phase 30 outputs."
        )
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--traces-json", type=Path, required=True)
    parser.add_argument("--phase2-features-csv", type=Path, required=True)
    parser.add_argument("--tile-index-csv", type=Path, required=True)
    parser.add_argument("--block-mapping-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focal-variant", default="N1ZR")
    parser.add_argument("--comparator-variant", default="B1")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args(argv)

    try:
        analysis = build_phase31_case_diagnostics(
            summary_csv=args.summary_csv,
            traces_json=args.traces_json,
            phase2_features_csv=args.phase2_features_csv,
            tile_index_csv=args.tile_index_csv,
            block_mapping_csv=args.block_mapping_csv,
            focal_variant=args.focal_variant,
            comparator_variant=args.comparator_variant,
            top_k=args.top_k,
        )
        paths = write_phase31_case_diagnostics_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Phase 31 case diagnostic status: "
        f"{analysis['phase31_case_diagnostic_status']}"
    )
    print(f"Ranked cases CSV: {paths['ranked_cases_csv']}")
    print(f"Selected blocks CSV: {paths['selected_blocks_csv']}")
    print(f"Tile geometry CSV: {paths['tile_geometry_csv']}")
    print(f"Diagnosis JSON: {paths['diagnosis_json']}")
    print(f"Diagnosis Markdown: {paths['diagnosis_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
