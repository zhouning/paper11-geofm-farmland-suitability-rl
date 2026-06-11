from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.planning_reward_readiness import (
    PHASE18_CLAIM_BOUNDARY,
    build_phase18_planning_reward_readiness,
    write_phase18_planning_reward_readiness,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether Paper11 real Bishan artifacts are ready for true "
            "planning-performance DRL experiments."
        )
    )
    parser.add_argument(
        "--phase2-output-dir",
        type=Path,
        required=True,
        help="Directory containing Phase 2 experiment_variants.json.",
    )
    parser.add_argument(
        "--phase10-gate",
        type=Path,
        required=True,
        help="Path to phase10_reward_readiness_gate.json.",
    )
    parser.add_argument(
        "--phase12-audit",
        type=Path,
        required=True,
        help="Path to phase12_real_dltb_scale_audit.json.",
    )
    parser.add_argument(
        "--phase17-readiness",
        type=Path,
        default=None,
        help="Optional path to phase17_tiled_maskableppo_readiness.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the Phase 18 readiness JSON will be written.",
    )
    args = parser.parse_args(argv)

    try:
        report = build_phase18_planning_reward_readiness(
            args.phase2_output_dir,
            args.phase10_gate,
            args.phase12_audit,
            phase17_readiness_path=args.phase17_readiness,
        )
        output_path = write_phase18_planning_reward_readiness(
            report,
            args.output_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Base planning reward implemented: "
        f"{report['base_planning_reward_implemented']}"
    )
    print(f"Suitability reward allowed: {report['suitability_reward_allowed']}")
    print(
        "Flat full-scale training ready: "
        f"{report['flat_full_scale_training_ready']}"
    )
    print(f"Tiled MaskablePPO API ready: {report['tiled_maskableppo_api_ready']}")
    print(f"Performance experiment ready: {report['performance_experiment_ready']}")
    print(f"Blocked reasons: {', '.join(report['blocked_reasons'])}")
    print(f"Recommendation: {report['recommended_next_step']}")
    print(f"Readiness report: {output_path}")
    print(f"Claim boundary: {PHASE18_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
