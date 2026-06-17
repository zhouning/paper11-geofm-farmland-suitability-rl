from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.padded_heldout_policy import (
    PHASE25_CLAIM_BOUNDARY,
    run_phase25_padded_heldout_policy,
    write_phase25_padded_heldout_policy_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded padded variable-size held-out-tile B0/B1 "
            "MaskablePPO learned-policy pilot under the deterministic base "
            "planning reward."
        )
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--tile-index-csv", type=Path, required=True)
    parser.add_argument("--variants", default="B0,B1")
    parser.add_argument("--train-tile-id", default=None)
    parser.add_argument("--eval-tile-ids", default=None)
    parser.add_argument("--max-eval-tiles", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=32)
    parser.add_argument("--eval-max-steps", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        protocol = run_phase25_padded_heldout_policy(
            args.phase2_output_dir,
            args.tile_index_csv,
            variants=tuple(
                part.strip() for part in args.variants.split(",") if part.strip()
            ),
            train_tile_id=args.train_tile_id,
            eval_tile_ids=args.eval_tile_ids,
            max_eval_tiles=args.max_eval_tiles,
            total_timesteps=args.total_timesteps,
            eval_max_steps=args.eval_max_steps,
            seeds=args.seeds,
        )
        paths = write_phase25_padded_heldout_policy_artifacts(
            protocol,
            args.output_dir,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    delta = protocol["comparison"]["learned_policy"]["B1_minus_B0_mean_reward"]
    print(f"Train tile: {protocol['train_tile_id']}")
    print(
        "Held-out evaluation tiles: "
        f"{', '.join(str(tile_id) for tile_id in protocol['eval_tile_ids'])}"
    )
    print(f"Padded max blocks: {protocol['max_blocks']}")
    print(f"Seeds: {', '.join(str(seed) for seed in protocol['seeds'])}")
    print(f"Variants: {', '.join(str(variant) for variant in protocol['variants'])}")
    print(f"Total timesteps: {protocol['total_timesteps']}")
    print(f"Evaluation max steps: {protocol['eval_max_steps']}")
    print(f"Padded held-out policy status: {protocol['padded_policy_status']}")
    print(f"Summary rows: {protocol['summary_count']}")
    print(f"All evaluations completed: {protocol['all_evaluations_completed']}")
    print(f"B1-B0 held-out learned-policy mean reward delta: {delta}")
    print(f"Pilot result status: {protocol['comparison']['pilot_result_status']}")
    print(f"Summary CSV path: {paths['summary_csv']}")
    print(f"Trace JSON path: {paths['traces_json']}")
    print(f"Comparison JSON path: {paths['comparison_json']}")
    print(f"Claim boundary: {PHASE25_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
