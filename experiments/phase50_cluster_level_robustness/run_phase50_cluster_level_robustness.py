from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase50_cluster_level_robustness import (
    build_phase50_cluster_level_robustness,
    write_phase50_cluster_level_robustness_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = build_phase50_cluster_level_robustness(
            args.phase48_delta_csv,
            alpha=args.alpha,
        )
        paths = write_phase50_cluster_level_robustness_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 50 cluster status: {analysis['phase50_cluster_status']}")
    summary = analysis.get("cluster_summary", {})
    if isinstance(summary, dict):
        print(f"Cluster mean delta: {summary.get('mean_cluster_delta')}")
        print(
            "Positive clusters: "
            f"{summary.get('positive_cluster_count')} / {summary.get('cluster_count')}"
        )
        print(f"Cluster one-sided sign-test p: {summary.get('one_sided_sign_test_p')}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Cluster CSV: {paths['cluster_csv']}")
    print(f"Readiness Markdown: {paths['readiness_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze tile-seed cluster-level robustness for the Paper11 "
            "compressed GeoFM route."
        )
    )
    parser.add_argument("--phase48-delta-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
