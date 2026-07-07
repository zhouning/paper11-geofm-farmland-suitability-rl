from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase53_cluster_mean_support import (
    build_phase53_cluster_mean_support,
    write_phase53_cluster_mean_support_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = build_phase53_cluster_mean_support(
            args.phase50_cluster_csv,
            bootstrap_iterations=args.bootstrap_iterations,
            random_seed=args.random_seed,
            alpha=args.alpha,
        )
        paths = write_phase53_cluster_mean_support_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 53 cluster mean status: {analysis['phase53_cluster_mean_status']}")
    print(f"Mean cluster delta: {analysis['mean_cluster_delta']}")
    print(f"Exact sign-flip p: {analysis['exact_sign_flip_mean_p']}")
    print(
        "Bootstrap CI95: "
        f"[{analysis['bootstrap_ci95_low']}, {analysis['bootstrap_ci95_high']}]"
    )
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Leave-one CSV: {paths['leave_one_csv']}")
    print(f"Readiness Markdown: {paths['readiness_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact sign-flip, bootstrap, and leave-one influence checks "
            "over Phase 50 cluster mean deltas."
        )
    )
    parser.add_argument("--phase50-cluster-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=53)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
