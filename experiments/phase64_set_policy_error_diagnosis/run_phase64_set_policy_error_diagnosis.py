from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase64_set_policy_error_diagnosis import (
    run_phase64_set_policy_error_diagnosis,
    write_phase64_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = run_phase64_set_policy_error_diagnosis(
            phase63_comparison_json=args.phase63_comparison_json,
            phase63_rollout_csv=args.phase63_rollout_csv,
            phase63_history_csv=args.phase63_history_csv,
            phase63_oracle_summary_csv=args.phase63_oracle_summary_csv,
        )
        paths = write_phase64_artifacts(analysis, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    gate = analysis["standardization_gate"]
    print(f"Phase 64 status: {gate['phase64_status']}")
    print(f"Standardized rerun recommended: {gate['recommend_standardized_rerun']}")
    print(f"Gate JSON: {paths['gate_json']}")
    print(f"Diagnosis Markdown: {paths['diagnosis_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 64 set-policy error diagnosis."
    )
    parser.add_argument("--phase63-comparison-json", type=Path, required=True)
    parser.add_argument("--phase63-rollout-csv", type=Path, required=True)
    parser.add_argument("--phase63-history-csv", type=Path, required=True)
    parser.add_argument("--phase63-oracle-summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())