from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.tiled_baseline_protocol import (
    PHASE16_CLAIM_BOUNDARY,
    run_phase16_tiled_baseline_protocol,
    write_phase16_tiled_baseline_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run short non-learning masked rollouts across Phase 13 tiles "
            "without training a DRL policy or enabling suitability reward."
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
        help="Representation-only variant to run. Default: B1.",
    )
    parser.add_argument(
        "--policies",
        default="first_valid,seeded_random",
        help="Comma-separated non-learning policies. Default: first_valid,seeded_random.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4,
        help="Maximum rollout steps per tile and policy. Default: 4.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for deterministic seeded_random policy. Default: 0.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Optional cap on tile rows for quick checks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where Phase 16 CSV/JSON artifacts will be written.",
    )
    args = parser.parse_args(argv)

    try:
        policy_ids = [
            policy_id.strip()
            for policy_id in str(args.policies).split(",")
            if policy_id.strip()
        ]
        protocol = run_phase16_tiled_baseline_protocol(
            args.phase2_output_dir,
            args.tile_index_csv,
            variant_id=args.variant,
            policy_ids=policy_ids,
            max_steps=args.max_steps,
            seed=args.seed,
            max_tiles=args.max_tiles,
        )
        paths = write_phase16_tiled_baseline_artifacts(protocol, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Variant: {protocol['variant_id']}")
    print(f"Tiles processed: {protocol['tile_count']}")
    print(f"Policies: {len(protocol['policy_ids'])}")
    print(f"Summary rows: {protocol['summary_count']}")
    print(f"Total blocks: {protocol['total_blocks']}")
    print(f"Max observation shape: {protocol['max_observation_shape']}")
    print(f"All completed: {protocol['all_rollouts_completed']}")
    print(f"Summary CSV: {paths['summary_csv']}")
    print(f"Trace JSON: {paths['traces_json']}")
    print(f"Claim boundary: {PHASE16_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
