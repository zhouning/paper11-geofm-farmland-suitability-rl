from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase69_label_free_evidence_synthesis_gate import (
    build_phase69_label_free_evidence_synthesis_gate,
    write_phase69_label_free_evidence_synthesis_gate_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 69 label-free evidence synthesis gate."
    )
    parser.add_argument("--phase60-json", type=Path, required=True)
    parser.add_argument("--phase57-json", type=Path, required=True)
    parser.add_argument("--phase59-json", type=Path, required=True)
    parser.add_argument("--phase62-json", type=Path, required=True)
    parser.add_argument("--phase66-json", type=Path, required=True)
    parser.add_argument("--phase67-json", type=Path, required=True)
    parser.add_argument("--phase68-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        analysis = build_phase69_label_free_evidence_synthesis_gate(
            phase60_json=args.phase60_json,
            phase57_json=args.phase57_json,
            phase59_json=args.phase59_json,
            phase62_json=args.phase62_json,
            phase66_json=args.phase66_json,
            phase67_json=args.phase67_json,
            phase68_json=args.phase68_json,
        )
        artifacts = write_phase69_label_free_evidence_synthesis_gate_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 69 label-free synthesis status: {analysis['phase69_status']}")
    print(f"Evidence axes CSV: {artifacts['evidence_axes_csv']}")
    print(f"Claim boundary matrix CSV: {artifacts['claim_boundary_matrix_csv']}")
    print(f"Diagnosis JSON: {artifacts['diagnosis_json']}")
    print(f"Diagnosis Markdown: {artifacts['diagnosis_md']}")
    print(f"Allowed claim: {analysis['allowed_claim']}")
    print(f"Recommended next step: {analysis['recommended_next_step']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())