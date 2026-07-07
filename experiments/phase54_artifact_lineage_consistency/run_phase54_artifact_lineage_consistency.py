from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase54_artifact_lineage_consistency import (
    build_phase54_artifact_lineage_consistency,
    write_phase54_artifact_lineage_consistency_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = build_phase54_artifact_lineage_consistency(
            delta_csv=args.phase48_delta_csv,
            cluster_csv=args.phase50_cluster_csv,
            phase51_json=args.phase51_json,
            phase53_json=args.phase53_json,
            tolerance=args.tolerance,
        )
        paths = write_phase54_artifact_lineage_consistency_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 54 artifact lineage status: {analysis['phase54_lineage_status']}")
    print(f"All checks passed: {analysis['all_checks_passed']}")
    print(f"Recomputed cluster count: {analysis['recomputed_cluster_count']}")
    print(f"Recomputed mean cluster delta: {analysis['recomputed_mean_cluster_delta']}")
    print(f"Recomputed Phase 51 p: {analysis['recomputed_phase51_signed_rank_p']}")
    print(f"Recomputed Phase 53 p: {analysis['recomputed_phase53_sign_flip_p']}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Checks CSV: {paths['checks_csv']}")
    print(f"Readiness Markdown: {paths['readiness_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit that the formal Phase 52/53 compressed GeoFM evidence values "
            "come from one internally consistent artifact lineage."
        )
    )
    parser.add_argument("--phase48-delta-csv", type=Path, required=True)
    parser.add_argument("--phase50-cluster-csv", type=Path, required=True)
    parser.add_argument("--phase51-json", type=Path, required=True)
    parser.add_argument("--phase53-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
