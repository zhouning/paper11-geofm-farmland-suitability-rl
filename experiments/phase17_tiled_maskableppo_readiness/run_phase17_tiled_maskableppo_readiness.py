from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.tiled_maskableppo_readiness import (
    PHASE17_CLAIM_BOUNDARY,
    run_phase17_tiled_maskableppo_readiness,
    write_phase17_tiled_maskableppo_readiness_artifact,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a tiny tiled MaskablePPO readiness smoke check without "
            "training or evaluating a useful policy."
        )
    )
    parser.add_argument(
        "--phase2-output-dir",
        type=Path,
        required=True,
        help="Directory containing Phase 2 experiment_variants.json and variant CSVs.",
    )
    parser.add_argument(
        "--tile-index-csv",
        type=Path,
        required=True,
        help="Path to Phase 13 phase13_tile_index.csv.",
    )
    parser.add_argument(
        "--variant",
        default="B1",
        help="Representation-only variant to smoke-check. Default: B1.",
    )
    parser.add_argument(
        "--tile-id",
        default=None,
        help="Optional explicit tile ID. If omitted, --tile-selection is used.",
    )
    parser.add_argument(
        "--tile-selection",
        default="largest",
        help="Tile selection mode when --tile-id is omitted. Default: largest.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=8,
        help="Tiny MaskablePPO learn() timestep count. Default: 8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for env reset and MaskablePPO initialization. Default: 0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the Phase 17 JSON artifact will be written.",
    )
    args = parser.parse_args(argv)

    try:
        summary = run_phase17_tiled_maskableppo_readiness(
            args.phase2_output_dir,
            args.tile_index_csv,
            variant_id=args.variant,
            tile_id=args.tile_id,
            tile_selection=args.tile_selection,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
        )
        artifact_path = write_phase17_tiled_maskableppo_readiness_artifact(
            summary,
            args.output_dir,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Tile: {summary['tile_id']}")
    print(f"Tile selection: {summary['tile_selection']}")
    print(f"Variant: {summary['variant_id']}")
    print(f"Observation shape: {summary['observation_shape']}")
    print(f"Action space: Discrete({summary['action_space_n']})")
    print(f"Masking supported: {summary['masking_supported']}")
    print(f"Initial valid actions: {summary['initial_valid_actions']}")
    print(f"Learn timesteps: {summary['learn_timesteps']}")
    print(f"Predicted action: {summary['predicted_action']}")
    print(f"Predicted action valid: {summary['predicted_action_valid']}")
    print(f"Selected block: {summary['selected_block_id']}")
    print(f"Readiness status: {summary['readiness_status']}")
    print(f"Artifact: {artifact_path}")
    print(f"Claim boundary: {PHASE17_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
