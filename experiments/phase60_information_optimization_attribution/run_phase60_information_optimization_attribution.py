from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase60_information_optimization_attribution import (
    build_phase60_information_optimization_attribution_from_paths,
    write_phase60_information_optimization_attribution_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = build_phase60_information_optimization_attribution_from_paths(
            phase48_json=args.phase48_json,
            phase53_json=args.phase53_json,
            phase57_json=args.phase57_json,
            phase59_json=args.phase59_json,
        )
        paths = write_phase60_information_optimization_attribution_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 60 attribution status: {analysis['phase60_attribution_status']}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Axes CSV: {paths['axes_csv']}")
    print(f"Readiness Markdown: {paths['readiness_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Paper11 Phase 60 information-vs-optimization attribution audit."
    )
    parser.add_argument("--phase48-json", type=Path, required=True)
    parser.add_argument("--phase53-json", type=Path, required=True)
    parser.add_argument("--phase57-json", type=Path, required=True)
    parser.add_argument("--phase59-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
