from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase41_geofm_suitability_prior import (
    run_phase41_geofm_suitability_prior,
    write_phase41_geofm_suitability_prior_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 41 GeoFM suitability prior gate."
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--label-registry", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-valid-count", type=int, default=100)
    parser.add_argument("--max-missing-rate", type=float, default=0.20)
    parser.add_argument("--min-positive-rate", type=float, default=0.02)
    parser.add_argument("--max-positive-rate", type=float, default=0.98)
    parser.add_argument("--min-split-valid-count", type=int, default=20)
    parser.add_argument("--min-auc-delta", type=float, default=0.03)
    parser.add_argument("--min-ap-delta", type=float, default=0.03)
    parser.add_argument("--min-positive-fold-fraction", type=float, default=0.67)
    parser.add_argument("--max-brier-regression", type=float, default=0.02)
    parser.add_argument("--n-pca-components", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        analysis = run_phase41_geofm_suitability_prior(
            phase2_output_dir=args.phase2_output_dir,
            label_registry=args.label_registry,
            min_valid_count=args.min_valid_count,
            max_missing_rate=args.max_missing_rate,
            min_positive_rate=args.min_positive_rate,
            max_positive_rate=args.max_positive_rate,
            min_split_valid_count=args.min_split_valid_count,
            min_auc_delta=args.min_auc_delta,
            min_ap_delta=args.min_ap_delta,
            min_positive_fold_fraction=args.min_positive_fold_fraction,
            max_brier_regression=args.max_brier_regression,
            n_pca_components=args.n_pca_components,
        )
        artifacts = write_phase41_geofm_suitability_prior_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 41 GeoFM prior status: {analysis['phase41_geofm_prior_status']}")
    print(f"Summary CSV: {artifacts['summary_csv']}")
    print(f"Metrics CSV: {artifacts['metrics_csv']}")
    print(f"Diagnosis JSON: {artifacts['diagnosis_json']}")
    print(f"Diagnosis Markdown: {artifacts['diagnosis_md']}")
    if artifacts.get("prior_csv") is not None:
        print(f"Suitability prior CSV: {artifacts['prior_csv']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
