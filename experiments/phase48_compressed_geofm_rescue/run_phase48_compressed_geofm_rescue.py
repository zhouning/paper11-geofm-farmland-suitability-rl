from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase48_compressed_geofm_rescue import (
    build_phase48_compressed_geofm_rescue_analysis,
    write_phase48_compressed_geofm_rescue_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = build_phase48_compressed_geofm_rescue_analysis(
            args.existing_summary_csv,
            pooled_positive_threshold=args.pooled_positive_threshold,
        )
        paths = write_phase48_compressed_geofm_rescue_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Mode: analyze-only")
    print(
        "Phase 48 compressed GeoFM status: "
        f"{analysis['phase48_compressed_geofm_status']}"
    )
    learned = analysis.get("learned_policy", {})
    if isinstance(learned, dict):
        deltas = learned.get("compressed_deltas", {})
        if isinstance(deltas, dict):
            for key in sorted(deltas):
                value = deltas[key]
                if isinstance(value, dict):
                    print(f"{key} mean reward delta: {value.get('mean_reward_delta')}")
    print(f"Summary CSV: {paths['summary_csv']}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Delta CSV: {paths['delta_csv']}")
    print(f"Readiness Markdown: {paths['readiness_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the Paper11 Phase 48 compressed GeoFM rescue audit from "
            "existing held-out summary rows."
        )
    )
    parser.add_argument("--existing-summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pooled-positive-threshold", type=float, default=0.5)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
