from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase35_phase33_action_overlap_diagnostics import (
    build_phase35_phase33_action_overlap_diagnostics,
    write_phase35_phase33_action_overlap_diagnostics_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the read-only Paper11 Phase 35 action-overlap diagnostics "
            "over existing Phase 33 matched pilot outputs."
        )
    )
    parser.add_argument(
        "--phase33-output-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more Phase 33 matched output directories.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", default="N1Z,N1ZR")
    parser.add_argument("--comparators", default="B1,D4P8,D4P16")
    args = parser.parse_args(argv)

    try:
        analysis = build_phase35_phase33_action_overlap_diagnostics(
            phase33_output_dirs=args.phase33_output_dirs,
            variants=args.variants,
            comparators=args.comparators,
        )
        paths = write_phase35_phase33_action_overlap_diagnostics_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Phase 35 action-overlap status: "
        f"{analysis['phase35_action_overlap_status']}"
    )
    print(f"Case summary CSV: {paths['case_summary_csv']}")
    print(f"Step alignment CSV: {paths['step_alignment_csv']}")
    print(f"Diagnosis JSON: {paths['diagnosis_json']}")
    print(f"Diagnosis Markdown: {paths['diagnosis_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
