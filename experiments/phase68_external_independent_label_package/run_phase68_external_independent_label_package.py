from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase68_external_independent_label_package import (
    build_phase68_external_independent_label_package,
    write_phase68_external_independent_label_package_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Paper11 Phase 68 external independent-label package preflight."
        )
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--external-label-csvs", default="")
    parser.add_argument("--label-registry", type=Path)
    parser.add_argument("--validation-mode", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-valid-count", type=int, default=100)
    parser.add_argument("--max-missing-rate", type=float, default=0.20)
    parser.add_argument("--min-positive-rate", type=float, default=0.02)
    parser.add_argument("--max-positive-rate", type=float, default=0.98)
    parser.add_argument("--min-split-valid-count", type=int, default=20)
    args = parser.parse_args(argv)

    try:
        analysis = build_phase68_external_independent_label_package(
            phase2_output_dir=args.phase2_output_dir,
            external_label_csvs=_parse_optional_paths(args.external_label_csvs),
            label_registry=args.label_registry,
            validation_mode=args.validation_mode,
            min_valid_count=args.min_valid_count,
            max_missing_rate=args.max_missing_rate,
            min_positive_rate=args.min_positive_rate,
            max_positive_rate=args.max_positive_rate,
            min_split_valid_count=args.min_split_valid_count,
        )
        artifacts = write_phase68_external_independent_label_package_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 68 external-label package status: {analysis['phase68_status']}")
    print(f"External label template CSV: {artifacts['external_label_template_csv']}")
    print(f"Label registry template CSV: {artifacts['label_registry_template_csv']}")
    print(f"Package README: {artifacts['package_readme_md']}")
    print(f"Label preflight CSV: {artifacts['label_preflight_csv']}")
    print(f"Package summary CSV: {artifacts['package_summary_csv']}")
    print(f"Diagnosis JSON: {artifacts['diagnosis_json']}")
    print(f"Diagnosis Markdown: {artifacts['diagnosis_md']}")
    print(f"Recommended next step: {analysis['recommended_next_step']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _parse_optional_paths(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
