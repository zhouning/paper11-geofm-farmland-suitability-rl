from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase38_proxy_rebuild import (
    build_phase38_proxy_rebuild,
    write_phase38_proxy_rebuild_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Paper11 Phase 38 proxy-rebuild diagnostics over existing "
            "feature tables."
        )
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--phase8-output-dir", type=Path)
    parser.add_argument("--normalized-controls-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--label-columns",
        default="current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label",
    )
    parser.add_argument("--label-classifications", default="")
    parser.add_argument(
        "--model-families",
        default="logistic_elastic_net,random_forest,hist_gradient_boosting",
    )
    parser.add_argument("--min-auc-delta", type=float, default=0.02)
    parser.add_argument("--min-ap-delta", type=float, default=0.02)
    args = parser.parse_args(argv)

    try:
        analysis = build_phase38_proxy_rebuild(
            phase2_output_dir=args.phase2_output_dir,
            phase8_output_dir=args.phase8_output_dir,
            normalized_controls_dir=args.normalized_controls_dir,
            label_columns=args.label_columns,
            label_classifications=args.label_classifications,
            model_families=args.model_families,
            min_auc_delta=args.min_auc_delta,
            min_ap_delta=args.min_ap_delta,
        )
        paths = write_phase38_proxy_rebuild_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Phase 38 proxy-rebuild status: "
        f"{analysis['phase38_proxy_rebuild_status']}"
    )
    print(
        "Available labels: "
        f"{','.join(str(label) for label in analysis['label_columns_available'])}"
    )
    print(f"Label summary CSV: {paths['label_summary_csv']}")
    print(f"Model summary CSV: {paths['model_summary_csv']}")
    print(f"Rebuilt proxy scores CSV: {paths['rebuilt_proxy_scores_csv']}")
    print(f"Diagnosis JSON: {paths['diagnosis_json']}")
    print(f"Diagnosis Markdown: {paths['diagnosis_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
