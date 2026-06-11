from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.bounded_tiled_training import (
    PHASE20_CLAIM_BOUNDARY,
    run_phase20_bounded_tiled_training,
    write_phase20_bounded_tiled_training_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded B0/B1 tiled MaskablePPO training/evaluation pilot "
            "without suitability reward or final performance claims."
        )
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--tile-index-csv", type=Path, required=True)
    parser.add_argument("--variants", default="B0,B1")
    parser.add_argument("--train-tile-id", default=None)
    parser.add_argument("--eval-tile-id", default=None)
    parser.add_argument("--total-timesteps", type=int, default=8)
    parser.add_argument("--eval-max-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        protocol = run_phase20_bounded_tiled_training(
            args.phase2_output_dir,
            args.tile_index_csv,
            variants=tuple(
                part.strip() for part in args.variants.split(",") if part.strip()
            ),
            train_tile_id=args.train_tile_id,
            eval_tile_id=args.eval_tile_id,
            total_timesteps=args.total_timesteps,
            eval_max_steps=args.eval_max_steps,
            seed=args.seed,
        )
        paths = write_phase20_bounded_tiled_training_artifacts(
            protocol,
            args.output_dir,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Train tile: {protocol['train_tile_id']}")
    print(f"Evaluation tile: {protocol['eval_tile_id']}")
    print(
        "Cross-tile learned-policy status: "
        f"{protocol['cross_tile_evaluation_status']}"
    )
    print(f"Variants: {', '.join(protocol['variants'])}")
    print(f"Train timesteps: {protocol['total_timesteps']}")
    print(f"Evaluation max steps: {protocol['eval_max_steps']}")
    print(f"Summary rows: {protocol['summary_count']}")
    print(f"Training completed: {protocol['training_completed']}")
    print(f"All evaluations completed: {protocol['all_evaluations_completed']}")
    print(f"Summary CSV: {paths['summary_csv']}")
    print(f"Trace JSON: {paths['traces_json']}")
    print(f"Claim boundary: {PHASE20_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
