from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
    run_phase65_standardized_set_policy_bc_rerun,
    write_phase65_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = run_phase65_standardized_set_policy_bc_rerun(
            phase63_comparison_json=args.phase63_comparison_json,
            phase63_rollout_csv=args.phase63_rollout_csv,
            existing_flattened_summary_csvs=args.existing_flattened_summary_csvs,
        )
        paths = write_phase65_artifacts(analysis, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    comparison = analysis["standardization_comparison"]
    print(f"Phase 65 status: {comparison['phase65_status']}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Rollout CSV: {paths['rollout_csv']}")
    print(f"Pairwise Delta CSV: {paths['pairwise_delta_csv']}")
    print(f"Readiness Markdown: {paths['readiness_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 65 standardized set-policy BC rerun."
    )
    parser.add_argument("--phase63-comparison-json", type=Path, required=True)
    parser.add_argument("--phase63-rollout-csv", type=Path, required=True)
    parser.add_argument("--existing-flattened-summary-csvs", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
