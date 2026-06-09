from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.artifacts import write_phase2_artifacts
from paper11_geofm.block_features import (
    attach_optional_block_attributes,
    compute_block_geofm_features,
)
from paper11_geofm.block_schema import EXPLICIT_FEATURE_COLUMNS
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
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with block_id,row,col[,weight] mapping rows. "
            "When omitted, a generated grid-derived mapping is used."
        ),
    )
    parser.add_argument(
        "--attributes-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV keyed by block_id with explicit_feature_00..16, "
            "weak labels, splits, or other block attributes."
        ),
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
    if args.mapping_csv is None:
        mapping_rows = build_generated_grid_mapping(
            grid_shape,
            row_bins=args.row_bins,
            col_bins=args.col_bins,
        )
        mapping_mode = "generated_grid"
    else:
        mapping_rows = _read_csv_rows(args.mapping_csv)
        mapping_mode = "mapping_csv"

    mapping = validate_block_pixel_mapping(mapping_rows, grid_shape)
    rows = compute_block_geofm_features(base_embedding, mapping, annual_embeddings)
    scored_rows = add_suitability_proxy(rows)
    attributes = _read_csv_rows(args.attributes_csv) if args.attributes_csv else None
    output_rows = attach_optional_block_attributes(scored_rows, attributes)
    feature_groups_present, missing_feature_groups = _feature_group_status(attributes)
    summary = {
        "metadata_source": metadata["source"],
        "base_year_requested": args.base_year,
        "base_year_used": base_year_used,
        "years": years,
        "grid_shape": list(grid_shape),
        "embedding_dim": metadata["embedding_dim"],
        "mapping_mode": mapping_mode,
        "row_bins": args.row_bins,
        "col_bins": args.col_bins,
        "feature_groups_present": feature_groups_present,
        "missing_feature_groups": missing_feature_groups,
    }
    if args.mapping_csv is not None:
        summary["mapping_csv"] = str(args.mapping_csv)
    if args.attributes_csv is not None:
        summary["attributes_csv"] = str(args.attributes_csv)

    return write_phase2_artifacts(
        output_rows,
        args.output_dir,
        summary,
    )


def _nearest_year(requested_year: int, years: list[int]) -> int:
    return min(years, key=lambda year: (abs(year - requested_year), year))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _feature_group_status(
    attributes: list[dict[str, str]] | None,
) -> tuple[list[str], list[str]]:
    present = ["geofm_embedding", "suitability_proxy"]
    missing: list[str] = []

    if attributes and _rows_have_columns(attributes, EXPLICIT_FEATURE_COLUMNS):
        present.append("explicit_planning_features")
    else:
        missing.append("explicit_planning_features")

    weak_label_columns = ["stable_farmland_label", "high_standard_farmland_label"]
    if attributes and any(_rows_have_columns(attributes, [column]) for column in weak_label_columns):
        present.append("weak_labels")
    else:
        missing.append("weak_labels")

    return present, missing


def _rows_have_columns(rows: list[dict[str, str]], columns: list[str]) -> bool:
    return bool(rows) and all(all(column in row for column in columns) for row in rows)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = run_phase2(args)
    print(f"Wrote block table: {paths['block_table']}")
    print(f"Wrote summary: {paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
