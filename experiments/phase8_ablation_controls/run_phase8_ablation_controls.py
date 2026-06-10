from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.ablation_controls import (
    PHASE8_CLAIM_BOUNDARY,
    build_phase8_ablation_controls,
    write_phase8_ablation_artifacts,
)


def _parse_pca_dimensions(text: str) -> tuple[int, ...]:
    dimensions = tuple(
        int(part.strip()) for part in text.split(",") if part.strip()
    )
    if not dimensions:
        raise ValueError("At least one PCA dimension must be provided")
    invalid = [dimension for dimension in dimensions if dimension <= 0]
    if invalid:
        raise ValueError(f"PCA dimensions must be positive: {invalid}")
    return dimensions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Paper11 Phase 8 diagnostic ablation-control feature "
            "tables without training or evaluating a DRL policy."
        )
    )
    parser.add_argument(
        "--phase2-output-dir",
        type=Path,
        required=True,
        help="Directory containing ready Phase 2 B0/B1 variant CSV exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where Phase 8 ablation-control artifacts will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic random and shuffled controls. Default: 0.",
    )
    parser.add_argument(
        "--pca-dimensions",
        default="8,16",
        help="Comma-separated PCA component counts. Default: 8,16.",
    )
    args = parser.parse_args(argv)

    try:
        protocol = build_phase8_ablation_controls(
            args.phase2_output_dir,
            seed=args.seed,
            pca_dimensions=_parse_pca_dimensions(args.pca_dimensions),
        )
        paths = write_phase8_ablation_artifacts(protocol, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Generated variants: {','.join(protocol['variant_ids'])}")
    for variant_id in protocol["variant_ids"]:
        print(f"{variant_id} features: {protocol['summary'][variant_id]['n_features']}")
    print(f"Manifest: {paths['manifest']}")
    print(f"Summary: {paths['summary']}")
    print(f"Claim boundary: {PHASE8_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
