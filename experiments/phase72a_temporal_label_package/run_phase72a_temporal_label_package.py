from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase72a_temporal_label_package import (  # noqa: E402
    build_phase72a_temporal_label_package,
    write_phase72a_temporal_label_package_artifacts,
)


def _parse_region_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected region=path, got {value}")
        region, raw_path = value.split("=", 1)
        region = region.strip().lower()
        if not region or region in result:
            raise ValueError(
                f"region mapping must be nonblank and unique: {region}"
            )
        result[region] = Path(raw_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build Paper11 Phase 72A label package"
    )
    parser.add_argument("--region-config", type=Path, required=True)
    parser.add_argument(
        "--embedding-dir", action="append", default=[], required=True
    )
    parser.add_argument(
        "--label-dir", action="append", default=[], required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manual-review-per-stratum", type=int, default=20
    )
    parser.add_argument("--spatial-block-size", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        package = build_phase72a_temporal_label_package(
            region_config=args.region_config,
            embedding_dirs=_parse_region_paths(args.embedding_dir),
            label_dirs=_parse_region_paths(args.label_dir),
            manual_review_per_stratum=args.manual_review_per_stratum,
            spatial_block_size=args.spatial_block_size,
        )
        paths = write_phase72a_temporal_label_package_artifacts(
            package, args.output_dir
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Phase 72A temporal label status: "
        f"{package['phase72a_status']}"
    )
    print(f"Row counts: {package['row_counts']}")
    for key, path in paths.items():
        print(f"{key}: {path}")
    print(f"Recommended next step: {package['recommended_next_step']}")
    print(f"Claim boundary: {package['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
