from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.reward_readiness import (
    PHASE10_CLAIM_BOUNDARY,
    build_phase10_reward_readiness_gate,
    write_phase10_reward_readiness_gate,
)


def _parse_label_list(text: str) -> tuple[str, ...]:
    labels = tuple(part.strip() for part in text.split(",") if part.strip())
    if not labels:
        raise ValueError("At least one required label must be provided")
    return labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Paper11 Phase 10 suitability reward-readiness gate "
            "without training or evaluating a policy."
        )
    )
    parser.add_argument(
        "--phase9-report",
        type=Path,
        required=True,
        help="Path to phase9_proxy_validation_report.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the Phase 10 gate JSON will be written.",
    )
    parser.add_argument(
        "--required-labels",
        default="stable_farmland_label,high_standard_farmland_label",
        help="Comma-separated weak-label columns required by the gate.",
    )
    parser.add_argument(
        "--min-rank-auc",
        type=float,
        default=0.5,
        help="Minimum rank AUC for a required label to pass. Default: 0.5.",
    )
    parser.add_argument(
        "--min-mean-difference",
        type=float,
        default=0.0,
        help=(
            "Minimum positive-minus-negative suitability mean difference for "
            "a required label to pass. Default: 0.0."
        ),
    )
    args = parser.parse_args(argv)

    try:
        gate = build_phase10_reward_readiness_gate(
            args.phase9_report,
            required_labels=_parse_label_list(args.required_labels),
            min_rank_auc=args.min_rank_auc,
            min_mean_difference=args.min_mean_difference,
        )
        gate_path = write_phase10_reward_readiness_gate(gate, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Gate: {gate_path}")
    print(f"Status: {gate['status']}")
    print(f"Recommendation: {gate['recommendation']}")
    print(f"Passing labels: {gate['passing_label_count']}")
    print(f"Failing labels: {gate['failing_label_count']}")
    print(f"Insufficient labels: {gate['insufficient_label_count']}")
    for label, label_result in gate["labels"].items():
        print(f"{label}: {label_result['reason']}")
    print(f"Claim boundary: {PHASE10_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

