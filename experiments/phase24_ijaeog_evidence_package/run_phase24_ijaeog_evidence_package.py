from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.ijaeog_evidence_package import (
    PHASE24_CLAIM_BOUNDARY,
    build_phase24_ijaeog_evidence_package,
    write_phase24_ijaeog_evidence_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an IJAEOG claim-readiness evidence package from Phase 22 "
            "and Phase 23 pilot outputs."
        )
    )
    parser.add_argument("--phase22-summary-csv", type=Path, required=True)
    parser.add_argument("--phase23-summary-csv", type=Path, required=True)
    parser.add_argument("--phase23-comparison-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        package = build_phase24_ijaeog_evidence_package(
            args.phase22_summary_csv,
            args.phase23_summary_csv,
            args.phase23_comparison_json,
        )
        paths = write_phase24_ijaeog_evidence_artifacts(package, args.output_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    submission_status = package["claim_readiness"]["submission_ready"]["status"]
    print(f"Phase 22 summary rows: {package['phase22']['summary_rows']}")
    print(f"Phase 23 summary rows: {package['phase23']['summary_rows']}")
    print(
        "B1-B0 learned-policy mean reward delta: "
        f"{package['phase23']['B1_minus_B0_mean_reward']}"
    )
    print(f"Submission readiness: {submission_status}")
    print(f"Evidence CSV: {paths['evidence_csv']}")
    print(f"Summary JSON: {paths['summary_json']}")
    print(f"Claim readiness Markdown: {paths['claim_readiness_md']}")
    print(f"Claim boundary: {PHASE24_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
