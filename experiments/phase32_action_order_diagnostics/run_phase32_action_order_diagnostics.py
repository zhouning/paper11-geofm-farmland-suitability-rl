from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase32_action_order_diagnostics import (
    build_phase32_action_order_diagnostics,
    write_phase32_action_order_diagnostics_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the read-only Paper11 Phase 32 action-order diagnostics over "
            "existing Phase 28, Phase 30, and Phase 31 outputs."
        )
    )
    parser.add_argument("--ranked-cases-csv", type=Path, required=True)
    parser.add_argument("--focal-traces-json", type=Path, required=True)
    parser.add_argument("--comparator-traces-json", type=Path, required=True)
    parser.add_argument("--phase2-features-csv", type=Path, required=True)
    parser.add_argument("--tile-index-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args(argv)

    try:
        analysis = build_phase32_action_order_diagnostics(
            ranked_cases_csv=args.ranked_cases_csv,
            focal_traces_json=args.focal_traces_json,
            comparator_traces_json=args.comparator_traces_json,
            phase2_features_csv=args.phase2_features_csv,
            tile_index_csv=args.tile_index_csv,
            top_k=args.top_k,
        )
        paths = write_phase32_action_order_diagnostics_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Phase 32 action-order status: "
        f"{analysis['phase32_action_order_status']}"
    )
    print(f"Step alignment CSV: {paths['step_alignment_csv']}")
    print(f"Case summary CSV: {paths['case_summary_csv']}")
    print(f"Tile pool CSV: {paths['tile_pool_csv']}")
    print(f"Diagnosis JSON: {paths['diagnosis_json']}")
    print(f"Diagnosis Markdown: {paths['diagnosis_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
