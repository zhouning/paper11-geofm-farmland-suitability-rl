from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.proxy_validation import (
    PHASE9_CLAIM_BOUNDARY,
    build_phase9_proxy_validation_report,
    write_phase9_proxy_validation_report,
)


def _parse_label_columns(text: str) -> tuple[str, ...]:
    columns = tuple(part.strip() for part in text.split(",") if part.strip())
    if not columns:
        raise ValueError("At least one label column must be provided")
    return columns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Paper11 Phase 9 weak-label proxy-validation report "
            "without training or evaluating a policy."
        )
    )
    parser.add_argument(
        "--phase2-output-dir",
        type=Path,
        required=True,
        help="Directory containing Phase 2 block_geofm_features.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the Phase 9 JSON report will be written.",
    )
    parser.add_argument(
        "--label-columns",
        default="stable_farmland_label,high_standard_farmland_label",
        help=(
            "Comma-separated weak-label columns to validate against "
            "suitability_proxy."
        ),
    )
    args = parser.parse_args(argv)

    try:
        report = build_phase9_proxy_validation_report(
            args.phase2_output_dir,
            label_columns=_parse_label_columns(args.label_columns),
        )
        report_path = write_phase9_proxy_validation_report(report, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Report: {report_path}")
    print(f"Blocks: {report['n_blocks']}")
    available = ",".join(report["label_columns_available"])
    print(f"Available labels: {available if available else 'none'}")
    for label_column, label_report in report["labels"].items():
        if label_report["validation_available"]:
            print(
                f"{label_column} rank_auc: {label_report['rank_auc']} "
                f"mean_difference: {label_report['mean_difference']} "
                f"interpretation: {label_report['interpretation']}"
            )
        else:
            print(f"{label_column}: {label_report['interpretation']}")
    print(f"Claim boundary: {PHASE9_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

