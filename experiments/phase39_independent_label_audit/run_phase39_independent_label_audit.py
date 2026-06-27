from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase39_independent_label_audit import (
    DEFAULT_PHASE39_LABEL_COLUMNS,
    build_phase39_independent_label_audit,
    write_phase39_independent_label_audit_artifacts,
)


DEFAULT_CLI_LABEL_COLUMNS = ",".join(DEFAULT_PHASE39_LABEL_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Paper11 Phase 39 independent-label audit over existing "
            "Phase 2 block tables and optional external label CSVs."
        )
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--external-label-csvs", default="")
    parser.add_argument("--label-registry", type=Path)
    parser.add_argument("--label-columns", default=DEFAULT_CLI_LABEL_COLUMNS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        analysis = build_phase39_independent_label_audit(
            phase2_output_dir=args.phase2_output_dir,
            external_label_csvs=_parse_optional_paths(args.external_label_csvs),
            label_registry=args.label_registry,
            label_columns=args.label_columns,
        )
        paths = write_phase39_independent_label_audit_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Phase 39 independent-label audit status: "
        f"{analysis['phase39_independent_label_audit_status']}"
    )
    print(
        "Requested labels: "
        f"{','.join(str(label) for label in analysis['label_columns_requested'])}"
    )
    print(f"Label inventory CSV: {paths['label_inventory_csv']}")
    print(f"Label readiness CSV: {paths['label_readiness_csv']}")
    print(f"Registry template CSV: {paths['registry_template_csv']}")
    print(f"Diagnosis JSON: {paths['diagnosis_json']}")
    print(f"Diagnosis Markdown: {paths['diagnosis_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _parse_optional_paths(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())