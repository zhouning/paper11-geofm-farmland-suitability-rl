from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.artifacts import write_phase2_artifacts
from paper11_geofm.block_features import compute_block_geofm_features
from paper11_geofm.block_mapping import validate_block_pixel_mapping
from paper11_geofm.regions import make_grid_region_labels
from paper11_geofm.sample_data import (
    load_annual_embeddings,
    load_embedding,
    load_metadata,
)
from paper11_geofm.suitability import add_suitability_proxy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Paper11 Phase 2 block GeoFM feature assembly baseline."
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=ROOT / "data" / "bishan_alphaearth_sample",
        help="Directory containing Bishan AlphaEarth sample arrays.",
    )
    parser.add_argument(
        "--base-year",
        type=int,
        default=2020,
        help="Requested annual embedding year used for block-level aggregation.",
    )
    parser.add_argument(
        "--row-bins",
        type=int,
        default=5,
        help="Number of generated grid-block bins along rows.",
    )
    parser.add_argument(
        "--col-bins",
        type=int,
        default=5,
        help="Number of generated grid-block bins along columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for block_geofm_features.csv and summary.json.",
    )
    return parser


def build_generated_grid_mapping(
    grid_shape: tuple[int, int],
    row_bins: int,
    col_bins: int,
) -> list[dict[str, object]]:
    labels = make_grid_region_labels(grid_shape, row_bins, col_bins)
    rows = []
    for pixel_row in range(grid_shape[0]):
        for pixel_col in range(grid_shape[1]):
            rows.append(
                {
                    "block_id": f"grid_block_{int(labels[pixel_row, pixel_col]):02d}",
                    "row": pixel_row,
                    "col": pixel_col,
                    "weight": 1.0,
                }
            )
    return rows


def run_phase2(args: argparse.Namespace) -> dict[str, Path]:
    metadata = load_metadata(args.sample_dir)
    years = list(metadata["years"])
    base_year_used = _nearest_year(args.base_year, years)
    base_embedding = load_embedding(args.sample_dir, base_year_used)
    annual_embeddings = load_annual_embeddings(args.sample_dir, years)
    grid_shape = tuple(base_embedding.shape[:2])
    mapping_rows = build_generated_grid_mapping(
        grid_shape,
        row_bins=args.row_bins,
        col_bins=args.col_bins,
    )
    mapping = validate_block_pixel_mapping(mapping_rows, grid_shape)
    rows = compute_block_geofm_features(base_embedding, mapping, annual_embeddings)
    scored_rows = add_suitability_proxy(rows)

    return write_phase2_artifacts(
        scored_rows,
        args.output_dir,
        {
            "metadata_source": metadata["source"],
            "base_year_requested": args.base_year,
            "base_year_used": base_year_used,
            "years": years,
            "grid_shape": list(grid_shape),
            "embedding_dim": metadata["embedding_dim"],
            "mapping_mode": "generated_grid",
            "row_bins": args.row_bins,
            "col_bins": args.col_bins,
            "feature_groups_present": ["geofm_embedding", "suitability_proxy"],
            "missing_feature_groups": [
                "explicit_planning_features",
                "weak_labels",
            ],
        },
    )


def _nearest_year(requested_year: int, years: list[int]) -> int:
    return min(years, key=lambda year: (abs(year - requested_year), year))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = run_phase2(args)
    print(f"Wrote block table: {paths['block_table']}")
    print(f"Wrote summary: {paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
