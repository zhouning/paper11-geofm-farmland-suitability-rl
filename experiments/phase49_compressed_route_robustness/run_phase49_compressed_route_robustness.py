from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase49_compressed_route_robustness import (
    build_phase49_compressed_route_robustness,
    write_phase49_compressed_route_robustness_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = build_phase49_compressed_route_robustness(
            args.phase48_delta_csv,
            bootstrap_iterations=args.bootstrap_iterations,
            random_seed=args.random_seed,
            alpha=args.alpha,
        )
        paths = write_phase49_compressed_route_robustness_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 49 robustness status: {analysis['phase49_robustness_status']}")
    pooled = analysis.get("pooled_delta", {})
    if isinstance(pooled, dict):
        print(f"Pooled mean delta: {pooled.get('mean_delta')}")
        print(
            "Pooled positive comparisons: "
            f"{pooled.get('positive_count')} / {pooled.get('total_count')}"
        )
        print(f"Pooled one-sided sign-test p: {pooled.get('one_sided_sign_test_p')}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Per-comparison CSV: {paths['per_comparison_csv']}")
    print(f"Leave-one CSV: {paths['leave_one_csv']}")
    print(f"Readiness Markdown: {paths['readiness_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the Paper11 Phase 49 compressed GeoFM route robustness "
            "from Phase 48 delta rows."
        )
    )
    parser.add_argument("--phase48-delta-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=49)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
