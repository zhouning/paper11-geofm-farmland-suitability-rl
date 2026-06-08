from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.artifacts import write_phase1_artifacts
from paper11_geofm.features import compute_region_features
from paper11_geofm.regions import make_grid_region_labels
from paper11_geofm.sample_data import (
    load_annual_embeddings,
    load_embedding,
    load_metadata,
)
from paper11_geofm.suitability import add_suitability_proxy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Paper11 Phase 1 Bishan GeoFM baseline."
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
        help="Annual embedding year used for region-level feature aggregation.",
    )
    parser.add_argument(
        "--row-bins",
        type=int,
        default=5,
        help="Number of deterministic grid bins along rows.",
    )
    parser.add_argument(
        "--col-bins",
        type=int,
        default=5,
        help="Number of deterministic grid bins along columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for region_features.csv and summary.json.",
    )
    return parser


def run_phase1(args: argparse.Namespace) -> dict[str, Path]:
    metadata = load_metadata(args.sample_dir)
    base_embedding = load_embedding(args.sample_dir, args.base_year)
    annual_embeddings = load_annual_embeddings(args.sample_dir, metadata["years"])
    labels = make_grid_region_labels(
        base_embedding.shape[:2],
        n_row_bins=args.row_bins,
        n_col_bins=args.col_bins,
    )
    rows = compute_region_features(base_embedding, labels, annual_embeddings)
    scored_rows = add_suitability_proxy(rows)

    return write_phase1_artifacts(
        scored_rows,
        args.output_dir,
        {
            "metadata_source": metadata["source"],
            "base_year": args.base_year,
            "years": metadata["years"],
            "grid_shape": metadata["grid_shape"],
            "embedding_dim": metadata["embedding_dim"],
            "row_bins": args.row_bins,
            "col_bins": args.col_bins,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = run_phase1(args)
    print(f"Wrote region table: {paths['region_table']}")
    print(f"Wrote summary: {paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
