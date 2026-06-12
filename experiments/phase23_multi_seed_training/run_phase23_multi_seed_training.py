from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.multi_seed_training import (
    PHASE23_CLAIM_BOUNDARY,
    run_phase23_multi_seed_training,
    write_phase23_multi_seed_training_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded multi-seed same-tile B0/B1 MaskablePPO training "
            "pilot without suitability reward, cross-tile transfer, or final "
            "performance claims."
        )
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--tile-index-csv", type=Path, required=True)
    parser.add_argument("--variants", default="B0,B1")
    parser.add_argument("--train-tile-id", default=None)
    parser.add_argument("--eval-tile-id", default=None)
    parser.add_argument("--total-timesteps", type=int, default=8)
    parser.add_argument("--eval-max-steps", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        protocol = run_phase23_multi_seed_training(
            args.phase2_output_dir,
            args.tile_index_csv,
            variants=tuple(
                part.strip() for part in args.variants.split(",") if part.strip()
            ),
            train_tile_id=args.train_tile_id,
            eval_tile_id=args.eval_tile_id,
            total_timesteps=args.total_timesteps,
            eval_max_steps=args.eval_max_steps,
            seeds=args.seeds,
        )
        paths = write_phase23_multi_seed_training_artifacts(
            protocol,
            args.output_dir,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    delta = protocol["comparison"]["learned_policy"]["B1_minus_B0_mean_reward"]
    print(f"Train tile: {protocol['train_tile_id']}")
    print(f"Evaluation tile: {protocol['eval_tile_id']}")
    print(f"Seeds: {', '.join(str(seed) for seed in protocol['seeds'])}")
    print(f"Variants: {', '.join(protocol['variants'])}")
    print(f"Total timesteps: {protocol['total_timesteps']}")
    print(f"Evaluation max steps: {protocol['eval_max_steps']}")
    print(f"Cross-tile learned-policy status: {protocol['cross_tile_evaluation_status']}")
    print(f"Summary rows: {protocol['summary_count']}")
    print(f"All evaluations completed: {protocol['all_evaluations_completed']}")
    print(f"B1-B0 learned-policy mean reward delta: {delta}")
    print(f"Summary CSV: {paths['summary_csv']}")
    print(f"Trace JSON: {paths['traces_json']}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Claim boundary: {PHASE23_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
