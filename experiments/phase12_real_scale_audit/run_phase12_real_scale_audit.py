from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.real_scale_audit import (
    DEFAULT_FLAT_OBSERVATION_THRESHOLD,
    PHASE12_CLAIM_BOUNDARY,
    build_phase12_real_scale_audit,
    write_phase12_real_scale_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit real Bishan DLTB Phase 11/2/9/10 artifacts for scale, "
            "reward readiness, and defensible next experiment actions."
        )
    )
    parser.add_argument(
        "--phase11-summary",
        type=Path,
        required=True,
        help="Path to phase11_bishan_dltb_adapter_summary.json.",
    )
    parser.add_argument(
        "--phase2-output-dir",
        type=Path,
        required=True,
        help="Directory containing Phase 2 summary.json and experiment_variants.json.",
    )
    parser.add_argument(
        "--phase9-report",
        type=Path,
        required=True,
        help="Path to phase9_proxy_validation_report.json.",
    )
    parser.add_argument(
        "--phase10-gate",
        type=Path,
        required=True,
        help="Path to phase10_reward_readiness_gate.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the Phase 12 audit JSON will be written.",
    )
    parser.add_argument(
        "--flat-observation-threshold",
        type=int,
        default=DEFAULT_FLAT_OBSERVATION_THRESHOLD,
        help=(
            "Maximum flat observation dimension allowed for full-scale flat "
            f"training readiness. Default: {DEFAULT_FLAT_OBSERVATION_THRESHOLD}."
        ),
    )
    args = parser.parse_args(argv)

    try:
        report = build_phase12_real_scale_audit(
            args.phase11_summary,
            args.phase2_output_dir,
            args.phase9_report,
            args.phase10_gate,
            flat_observation_threshold=args.flat_observation_threshold,
        )
        output_path = write_phase12_real_scale_audit(report, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Blocks: {report['n_blocks']}")
    print(f"Max observation dimension: {report['max_observation_dimension']}")
    print(f"Real feature tables ready: {report['real_feature_tables_ready']}")
    print(
        "Representation-only smoke allowed: "
        f"{report['representation_only_smoke_allowed']}"
    )
    print(f"Suitability reward allowed: {report['suitability_reward_allowed']}")
    print(
        "Flat full-scale training ready: "
        f"{report['flat_full_scale_training_ready']}"
    )
    print(
        "Tiled or hierarchical env required: "
        f"{report['requires_tiled_or_hierarchical_env']}"
    )
    print(f"Recommendation: {report['recommendation']}")
    print(f"Audit report: {output_path}")
    print(f"Claim boundary: {PHASE12_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
