from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase66_reward_label_representation_audit import (
    run_phase66_reward_label_representation_audit,
    write_phase66_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = run_phase66_reward_label_representation_audit(
            phase63_comparison_json=args.phase63_comparison_json,
            phase63_rollout_csv=args.phase63_rollout_csv,
            phase63_oracle_summary_csv=args.phase63_oracle_summary_csv,
            phase64_failure_cases_csv=args.phase64_failure_cases_csv,
            phase64_feature_effective_rank_csv=args.phase64_feature_effective_rank_csv,
            phase65_comparison_json=args.phase65_comparison_json,
            phase65_rollout_csv=args.phase65_rollout_csv,
            phase65_pairwise_delta_csv=args.phase65_pairwise_delta_csv,
            phase10_reward_readiness_json=args.phase10_reward_readiness_json,
        )
        paths = write_phase66_artifacts(analysis, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    gate = analysis["diagnostic_gate"]
    print(f"Phase 66 status: {gate['phase66_status']}")
    print(f"Audit JSON: {paths['audit_json']}")
    print(f"Atlas CSV: {paths['atlas_csv']}")
    print(f"Alignment CSV: {paths['alignment_csv']}")
    print(f"Failure CSV: {paths['failure_csv']}")
    print(f"Audit Markdown: {paths['audit_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 66 reward-label representation audit."
    )
    parser.add_argument("--phase63-comparison-json", type=Path, required=True)
    parser.add_argument("--phase63-rollout-csv", type=Path, required=True)
    parser.add_argument("--phase63-oracle-summary-csv", type=Path, required=True)
    parser.add_argument("--phase64-failure-cases-csv", type=Path, default=None)
    parser.add_argument("--phase64-feature-effective-rank-csv", type=Path, default=None)
    parser.add_argument("--phase65-comparison-json", type=Path, required=True)
    parser.add_argument("--phase65-rollout-csv", type=Path, required=True)
    parser.add_argument("--phase65-pairwise-delta-csv", type=Path, required=True)
    parser.add_argument("--phase10-reward-readiness-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
