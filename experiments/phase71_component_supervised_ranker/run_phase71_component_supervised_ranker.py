from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase71_component_supervised_ranker import (
    run_phase71_component_supervised_ranker,
    write_phase71_component_ranker_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = run_phase71_component_supervised_ranker(
            phase2_output_dir=args.phase2_output_dir,
            phase8_output_dir=args.phase8_output_dir,
            phase61_output_dir=args.phase61_output_dir,
            tile_index_csv=args.tile_index_csv,
            phase63_rollout_csv=args.phase63_rollout_csv,
            phase70_rollout_csv=args.phase70_rollout_csv,
            variants=args.variants,
            train_tile_id=args.train_tile_id,
            eval_tile_ids=args.eval_tile_ids,
            max_eval_tiles=args.max_eval_tiles,
            eval_max_steps=args.eval_max_steps,
            seeds=args.seeds,
            ranker_epochs=args.ranker_epochs,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            component_weight=args.component_weight,
            top_k=args.top_k,
        )
        paths = write_phase71_component_ranker_artifacts(analysis, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 71 component-supervised ranker status: {analysis['phase71_status']}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Rollout CSV: {paths['rollout_csv']}")
    print(f"Delta CSV: {paths['delta_csv']}")
    print(f"Readiness Markdown: {paths['readiness_md']}")
    print(f"Recommended next step: {analysis['recommended_next_step']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 71 component-supervised listwise ranker."
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--phase8-output-dir", type=Path, required=True)
    parser.add_argument("--phase61-output-dir", type=Path, required=True)
    parser.add_argument("--tile-index-csv", type=Path, required=True)
    parser.add_argument("--phase63-rollout-csv", type=Path, required=True)
    parser.add_argument("--phase70-rollout-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-tile-id", default=None)
    parser.add_argument("--variants", default="B0,D4P8,D4P16,D6R8,D6R16")
    parser.add_argument("--eval-tile-ids", default=None)
    parser.add_argument("--max-eval-tiles", type=int, default=5)
    parser.add_argument("--eval-max-steps", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--ranker-epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--component-weight", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=3)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())