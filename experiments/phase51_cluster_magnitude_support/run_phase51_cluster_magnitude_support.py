from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase51_cluster_magnitude_support import (
    build_phase51_cluster_magnitude_support,
    write_phase51_cluster_magnitude_support_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = build_phase51_cluster_magnitude_support(
            args.phase50_cluster_csv,
            alpha=args.alpha,
        )
        paths = write_phase51_cluster_magnitude_support_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 51 magnitude status: {analysis['phase51_magnitude_status']}")
    print(f"Positive rank sum: {analysis['positive_rank_sum']}")
    print(f"Total rank sum: {analysis['total_rank_sum']}")
    print(f"One-sided signed-rank p: {analysis['one_sided_signed_rank_p']}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Rank CSV: {paths['rank_csv']}")
    print(f"Readiness Markdown: {paths['readiness_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact signed-rank magnitude support over Phase 50 cluster "
            "delta rows."
        )
    )
    parser.add_argument("--phase50-cluster-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
