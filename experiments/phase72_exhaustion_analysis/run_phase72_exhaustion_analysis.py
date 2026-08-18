from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase72_exhaustion_analysis import (
    build_phase72_exhaustion_analysis,
    write_phase72_exhaustion_analysis_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the read-only Paper11 Phase 72 exhaustion analysis."
    )
    parser.add_argument("--phase72a-json", type=Path, required=True)
    parser.add_argument("--phase72a-summary-csv", type=Path, required=True)
    parser.add_argument("--phase72a-review-csv", type=Path, required=True)
    parser.add_argument("--phase72b-json", type=Path, required=True)
    parser.add_argument("--phase72b-protocol-json", type=Path, required=True)
    parser.add_argument("--phase72b-metrics-csv", type=Path, required=True)
    parser.add_argument("--phase72b-control-csv", type=Path, required=True)
    parser.add_argument("--phase72b-transfer-csv", type=Path, required=True)
    parser.add_argument("--phase72b-receipt-json", type=Path, required=True)
    parser.add_argument("--phase72b-receipt-sha256", type=Path, required=True)
    parser.add_argument("--phase72b-confirmation-dir", type=Path, required=True)
    parser.add_argument("--phase72-two-year-json", type=Path)
    parser.add_argument("--phase72-two-year-receipt-json", type=Path)
    parser.add_argument("--phase72-two-year-receipt-sha256", type=Path)
    parser.add_argument("--phase72-two-year-confirmation-dir", type=Path)
    parser.add_argument("--phase72-residual-json", type=Path)
    parser.add_argument("--phase72-residual-receipt-json", type=Path)
    parser.add_argument("--phase72-residual-receipt-sha256", type=Path)
    parser.add_argument("--phase72-residual-confirmation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        analysis = build_phase72_exhaustion_analysis(
            phase72a_json=args.phase72a_json,
            phase72a_summary_csv=args.phase72a_summary_csv,
            phase72a_review_csv=args.phase72a_review_csv,
            phase72b_json=args.phase72b_json,
            phase72b_protocol_json=args.phase72b_protocol_json,
            phase72b_metrics_csv=args.phase72b_metrics_csv,
            phase72b_control_csv=args.phase72b_control_csv,
            phase72b_transfer_csv=args.phase72b_transfer_csv,
            phase72b_receipt_json=args.phase72b_receipt_json,
            phase72b_receipt_sha256=args.phase72b_receipt_sha256,
            phase72b_confirmation_dir=args.phase72b_confirmation_dir,
            phase72_two_year_json=args.phase72_two_year_json,
            phase72_two_year_receipt_json=args.phase72_two_year_receipt_json,
            phase72_two_year_receipt_sha256=(
                args.phase72_two_year_receipt_sha256
            ),
            phase72_two_year_confirmation_dir=(
                args.phase72_two_year_confirmation_dir
            ),
            phase72_residual_json=args.phase72_residual_json,
            phase72_residual_receipt_json=args.phase72_residual_receipt_json,
            phase72_residual_receipt_sha256=(
                args.phase72_residual_receipt_sha256
            ),
            phase72_residual_confirmation_dir=(
                args.phase72_residual_confirmation_dir
            ),
        )
        artifacts = write_phase72_exhaustion_analysis_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 72 exhaustion status: {analysis['phase72_exhaustion_status']}")
    print(f"Route decision: {analysis['route_decision']}")
    print(f"Criteria CSV: {artifacts['criteria_csv']}")
    print(f"Claim boundary CSV: {artifacts['claim_boundary_csv']}")
    print(f"Analysis JSON: {artifacts['analysis_json']}")
    print(f"Analysis Markdown: {artifacts['analysis_md']}")
    print(f"Recommended next step: {analysis['recommended_next_step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
