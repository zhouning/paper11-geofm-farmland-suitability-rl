from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase72_claim_drift_audit import (
    build_phase72_claim_drift_audit,
    write_phase72_claim_drift_audit_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the read-only Paper11 Phase 72 claim-drift audit."
    )
    parser.add_argument("--manuscript-md", type=Path, required=True)
    parser.add_argument("--phase60-json", type=Path, required=True)
    parser.add_argument("--phase62-json", type=Path, required=True)
    parser.add_argument("--phase69-json", type=Path, required=True)
    parser.add_argument("--phase71-json", type=Path, required=True)
    parser.add_argument("--phase72b-json", type=Path, required=True)
    parser.add_argument("--phase72-exhaustion-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        analysis = build_phase72_claim_drift_audit(
            manuscript_md=args.manuscript_md,
            phase60_json=args.phase60_json,
            phase62_json=args.phase62_json,
            phase69_json=args.phase69_json,
            phase71_json=args.phase71_json,
            phase72b_json=args.phase72b_json,
            phase72_exhaustion_json=args.phase72_exhaustion_json,
        )
        artifacts = write_phase72_claim_drift_audit_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 72 claim-drift status: {analysis['phase72_claim_drift_status']}")
    print(f"Claims CSV: {artifacts['claims_csv']}")
    print(f"Audit JSON: {artifacts['audit_json']}")
    print(f"Audit Markdown: {artifacts['audit_md']}")
    print(f"Recommended action: {analysis['recommended_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
