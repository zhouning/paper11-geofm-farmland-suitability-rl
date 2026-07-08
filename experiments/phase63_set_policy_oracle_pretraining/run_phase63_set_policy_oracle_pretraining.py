from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase63_set_policy_oracle_pretraining import (
    build_phase63_set_policy_analysis,
    run_phase63_set_policy_oracle_pretraining,
    write_phase63_set_policy_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        if args.mode in {"rollout-only", "run-and-analyze"}:
            protocol = run_phase63_set_policy_oracle_pretraining(
                phase2_output_dir=args.phase2_output_dir,
                phase8_output_dir=args.phase8_output_dir,
                phase61_output_dir=args.phase61_output_dir,
                tile_index_csv=args.tile_index_csv,
                variants=args.variants,
                existing_flattened_summary_csvs=(
                    args.existing_flattened_summary_csvs
                    if args.mode == "run-and-analyze"
                    else None
                ),
                train_tile_id=args.train_tile_id,
                eval_tile_ids=args.eval_tile_ids,
                max_eval_tiles=args.max_eval_tiles,
                eval_max_steps=args.eval_max_steps,
                seeds=args.seeds,
                bc_epochs=args.bc_epochs,
                learning_rate=args.learning_rate,
                hidden_dim=args.hidden_dim,
                top_k=args.top_k,
            )
        else:
            protocol = build_phase63_set_policy_analysis(
                args.existing_rollout_csv,
                existing_flattened_summary_csvs=args.existing_flattened_summary_csvs,
                metadata={
                    "variants": args.variants,
                    "eval_tile_ids": args.eval_tile_ids,
                    "seeds": args.seeds,
                },
            )
        paths = write_phase63_set_policy_artifacts(protocol, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 63 set-policy status: {protocol['phase63_set_policy_status']}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Rollout CSV: {paths['rollout_csv']}")
    print(f"Delta CSV: {paths['delta_csv']}")
    print(f"Readiness Markdown: {paths['readiness_md']}")
    print(f"Claim boundary: {protocol['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 63 set-policy oracle-pretraining."
    )
    parser.add_argument(
        "--mode",
        choices=("rollout-only", "run-and-analyze", "analyze-only"),
        default="analyze-only",
    )
    parser.add_argument("--phase2-output-dir", type=Path, default=None)
    parser.add_argument("--phase8-output-dir", type=Path, default=None)
    parser.add_argument("--phase61-output-dir", type=Path, default=None)
    parser.add_argument("--tile-index-csv", type=Path, default=None)
    parser.add_argument("--existing-rollout-csv", type=Path, default=None)
    parser.add_argument("--existing-flattened-summary-csvs", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-tile-id", default=None)
    parser.add_argument("--variants", default="B0,D4P8,D4P16,D6R8,D6R16")
    parser.add_argument("--eval-tile-ids", default=None)
    parser.add_argument("--max-eval-tiles", type=int, default=5)
    parser.add_argument("--eval-max-steps", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--bc-epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    missing = []
    if args.mode in {"rollout-only", "run-and-analyze"}:
        for attr, flag in (
            ("phase2_output_dir", "--phase2-output-dir"),
            ("phase8_output_dir", "--phase8-output-dir"),
            ("phase61_output_dir", "--phase61-output-dir"),
            ("tile_index_csv", "--tile-index-csv"),
        ):
            if getattr(args, attr) is None:
                missing.append(flag)
    if args.mode == "analyze-only" and args.existing_rollout_csv is None:
        missing.append("--existing-rollout-csv")
    if missing:
        raise ValueError(f"{args.mode} requires " + ", ".join(missing))


if __name__ == "__main__":
    raise SystemExit(main())
