from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase37_decision_alignment import (
    build_phase37_decision_alignment,
    write_phase37_decision_alignment_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Paper11 Phase 37 read-only decision-alignment diagnostics "
            "over existing Phase 34, Phase 35, and optional Phase 36 artifacts."
        )
    )
    parser.add_argument("--phase34-cases-csv", type=Path, required=True)
    parser.add_argument("--phase34-blocks-csv", type=Path, required=True)
    parser.add_argument("--phase35-cases-csv", type=Path, required=True)
    parser.add_argument("--phase36-diagnosis-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        analysis = build_phase37_decision_alignment(
            args.phase34_cases_csv,
            args.phase34_blocks_csv,
            args.phase35_cases_csv,
            phase36_diagnosis_json=args.phase36_diagnosis_json,
        )
        paths = write_phase37_decision_alignment_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Phase 37 decision-alignment status: "
        f"{analysis['phase37_decision_alignment_status']}"
    )
    print(
        "Phase 36 proxy-validation status: "
        f"{analysis['phase36_proxy_validation_status']}"
    )
    print(f"Case alignment CSV: {paths['case_alignment_csv']}")
    print(f"Summary CSV: {paths['summary_csv']}")
    print(f"Diagnosis JSON: {paths['diagnosis_json']}")
    print(f"Diagnosis Markdown: {paths['diagnosis_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
