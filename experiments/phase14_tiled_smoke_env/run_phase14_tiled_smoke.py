from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.tiled_inputs import (
    PHASE14_CLAIM_BOUNDARY,
    run_phase14_tiled_smoke,
    write_phase14_tiled_smoke_summary,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a one-step tile-level Paper11 input-contract smoke check "
            "without training a policy or enabling suitability reward."
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
        "--tile-id",
        required=True,
        help="Tile ID to smoke-check, for example tile_r003_c003.",
    )
    parser.add_argument(
        "--variant",
        default="B1",
        help="Representation-only variant to load. Default: B1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where Phase 14 summary JSON will be written.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max step count for the tile smoke environment.",
    )
    args = parser.parse_args(argv)

    try:
        summary = run_phase14_tiled_smoke(
            args.phase2_output_dir,
            args.tile_index_csv,
            args.tile_id,
            variant_id=args.variant,
            max_steps=args.max_steps,
        )
        output_path = write_phase14_tiled_smoke_summary(summary, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Tile: {summary['tile_id']}")
    print(f"Variant: {summary['variant_id']}")
    print(f"Rows: {summary['n_blocks']}")
    print(f"Features: {summary['n_features']}")
    print(f"Observation shape: {summary['observation_shape']}")
    print(f"Action space: Discrete({summary['action_space_n']})")
    print(f"Reward mode: {summary['reward_mode']}")
    print(f"Selected block: {summary['selected_block_id']}")
    print(f"Step reward: {float(summary['step_reward']):.6f}")
    print(f"Summary JSON: {output_path}")
    print(f"Claim boundary: {PHASE14_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
