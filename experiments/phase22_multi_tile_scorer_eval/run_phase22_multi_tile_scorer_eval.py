from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.multi_tile_scorer_eval import (
    PHASE22_CLAIM_BOUNDARY,
    run_phase22_multi_tile_scorer_eval,
    write_phase22_multi_tile_scorer_eval_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded multi-tile, multi-seed B0/B1 per-block scorer "
            "evaluation pilot without suitability reward, PPO training, or "
            "final performance claims."
        )
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--tile-index-csv", type=Path, required=True)
    parser.add_argument("--variants", default="B0,B1")
    parser.add_argument("--train-tile-id", default=None)
    parser.add_argument("--eval-tile-ids", default=None)
    parser.add_argument("--max-eval-tiles", type=int, default=2)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--eval-max-steps", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        protocol = run_phase22_multi_tile_scorer_eval(
            args.phase2_output_dir,
            args.tile_index_csv,
            variants=tuple(
                part.strip() for part in args.variants.split(",") if part.strip()
            ),
            train_tile_id=args.train_tile_id,
            eval_tile_ids=args.eval_tile_ids,
            max_eval_tiles=args.max_eval_tiles,
            ridge_alpha=args.ridge_alpha,
            eval_max_steps=args.eval_max_steps,
            seeds=args.seeds,
        )
        paths = write_phase22_multi_tile_scorer_eval_artifacts(
            protocol,
            args.output_dir,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Train tile: {protocol['train_tile_id']}")
    print(f"Evaluation tiles: {', '.join(protocol['eval_tile_ids'])}")
    print(f"Seeds: {', '.join(str(seed) for seed in protocol['seeds'])}")
    print(
        "Multi-tile learned-policy status: "
        f"{protocol['multi_tile_evaluation_status']}"
    )
    print(f"Variants: {', '.join(protocol['variants'])}")
    print(f"Ridge alpha: {protocol['ridge_alpha']}")
    print(f"Evaluation max steps: {protocol['eval_max_steps']}")
    print(f"Summary rows: {protocol['summary_count']}")
    print(f"All evaluations completed: {protocol['all_evaluations_completed']}")
    print(f"Summary CSV: {paths['summary_csv']}")
    print(f"Trace JSON: {paths['traces_json']}")
    print(f"Claim boundary: {PHASE22_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
