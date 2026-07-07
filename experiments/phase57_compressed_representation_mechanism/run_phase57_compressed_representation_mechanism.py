from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase57_compressed_representation_mechanism import (
    build_phase57_compressed_representation_mechanism,
    write_phase57_compressed_representation_mechanism_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = build_phase57_compressed_representation_mechanism(
            args.b1_features_csv,
            args.d4p8_features_csv,
            args.d4p16_features_csv,
            delta_rows_or_csv=args.delta_csv,
            tile_rows_or_csv=args.tile_index_csv,
        )
        paths = write_phase57_compressed_representation_mechanism_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 57 mechanism status: {analysis['phase57_mechanism_status']}")
    print(f"Conclusion: {analysis['conclusion']}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Geometry CSV: {paths['geometry_csv']}")
    print(f"Reward gain CSV: {paths['reward_gain_csv']}")
    print(f"Tile geometry-gain CSV: {paths['tile_geometry_gain_csv']}")
    print(f"Readiness Markdown: {paths['readiness_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the read-only compressed-representation mechanism audit over "
            "B1, D4P8, D4P16 feature tables and expanded compressed-route "
            "delta rows."
        )
    )
    parser.add_argument("--b1-features-csv", type=Path, required=True)
    parser.add_argument("--d4p8-features-csv", type=Path, required=True)
    parser.add_argument("--d4p16-features-csv", type=Path, required=True)
    parser.add_argument("--delta-csv", type=Path, required=True)
    parser.add_argument("--tile-index-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
