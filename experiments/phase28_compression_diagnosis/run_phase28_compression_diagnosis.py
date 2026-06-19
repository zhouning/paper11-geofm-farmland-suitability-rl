from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase28_compression_diagnosis import (
    build_phase28_compression_diagnosis,
    write_phase28_compression_diagnosis_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the read-only Paper11 Phase 28 compression diagnosis over "
            "existing representation-control outputs."
        )
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--phase2-b1-features-csv", type=Path, required=True)
    parser.add_argument("--d4p8-features-csv", type=Path, required=True)
    parser.add_argument("--d4p16-features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank-threshold", type=float, default=1e-12)
    args = parser.parse_args(argv)

    try:
        analysis = build_phase28_compression_diagnosis(
            summary_csv=args.summary_csv,
            phase2_b1_features_csv=args.phase2_b1_features_csv,
            d4p8_features_csv=args.d4p8_features_csv,
            d4p16_features_csv=args.d4p16_features_csv,
            rank_threshold=args.rank_threshold,
        )
        paths = write_phase28_compression_diagnosis_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Phase 28 compression diagnostic status: "
        f"{analysis['phase28_compression_diagnostic_status']}"
    )
    print(f"Overlap CSV: {paths['overlap_csv']}")
    print(f"Reward components CSV: {paths['reward_components_csv']}")
    print(f"Diagnosis JSON: {paths['diagnosis_json']}")
    print(f"Diagnosis Markdown: {paths['diagnosis_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
