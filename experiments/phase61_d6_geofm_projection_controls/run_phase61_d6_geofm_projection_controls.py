from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase61_d6_geofm_projection_controls import (
    build_phase61_d6_projection_controls,
    write_phase61_d6_projection_control_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        protocol = build_phase61_d6_projection_controls(
            b0_rows_or_csv=args.b0_features_csv,
            b1_rows_or_csv=args.b1_features_csv,
            d4p8_rows_or_csv=args.d4p8_features_csv,
            d4p16_rows_or_csv=args.d4p16_features_csv,
            dimensions=args.dimensions,
            seed=args.seed,
        )
        paths = write_phase61_d6_projection_control_artifacts(
            protocol,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Phase 61 D6 projection status: "
        f"{protocol['phase61_d6_projection_status']}"
    )
    print(f"Manifest: {paths['manifest']}")
    print(f"Feature summary: {paths['feature_summary']}")
    print(f"Geometry JSON: {paths['geometry_json']}")
    print(f"Geometry CSV: {paths['geometry_csv']}")
    print(f"Similarity CSV: {paths['similarity_csv']}")
    print(f"Readiness Markdown: {paths['readiness_md']}")
    print(f"Claim boundary: {protocol['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Paper11 Phase 61 D6 GeoFM projection controls."
    )
    parser.add_argument("--b0-features-csv", type=Path, required=True)
    parser.add_argument("--b1-features-csv", type=Path, required=True)
    parser.add_argument("--d4p8-features-csv", type=Path, required=True)
    parser.add_argument("--d4p16-features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimensions", default="8,16")
    parser.add_argument("--seed", type=int, default=61)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())