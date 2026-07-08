from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase59_matched_dimension_controls import (
    build_phase59_matched_dimension_control_analysis,
    build_phase59_matched_dimension_control_tables,
    run_phase59_matched_dimension_control_evaluation,
    write_phase59_matched_dimension_control_artifacts,
    write_phase59_matched_dimension_control_tables,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        if args.mode == "build-controls":
            protocol = build_phase59_matched_dimension_control_tables(
                args.b0_features_csv,
                args.d4p8_features_csv,
                args.d4p16_features_csv,
                seed=args.seed,
            )
            paths = write_phase59_matched_dimension_control_tables(
                protocol,
                args.output_dir,
            )
            print(f"Phase 59 control tables: {paths['manifest']}")
            return 0

        if args.mode == "run-and-analyze":
            protocol = run_phase59_matched_dimension_control_evaluation(
                phase8_output_dir=args.phase8_output_dir,
                phase59_control_dir=args.phase59_control_dir,
                tile_index_csv=args.tile_index_csv,
                variants=args.variants,
                existing_summary_csv=args.existing_summary_csv,
                train_tile_id=args.train_tile_id,
                eval_tile_ids=args.eval_tile_ids,
                max_eval_tiles=args.max_eval_tiles,
                total_timesteps=args.total_timesteps,
                eval_max_steps=args.eval_max_steps,
                seeds=args.seeds,
                bootstrap_iterations=args.bootstrap_iterations,
                random_seed=args.seed,
            )
        else:
            protocol = build_phase59_matched_dimension_control_analysis(
                args.existing_summary_csv,
                metadata={
                    "eval_tile_ids": args.eval_tile_ids,
                    "seeds": args.seeds,
                },
                bootstrap_iterations=args.bootstrap_iterations,
                random_seed=args.seed,
            )
        paths = write_phase59_matched_dimension_control_artifacts(
            protocol,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Phase 59 matched-dimension status: "
        f"{protocol['phase59_matched_dimension_status']}"
    )
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Delta CSV: {paths['delta_csv']}")
    print(f"Readiness Markdown: {paths['readiness_md']}")
    print(f"Claim boundary: {protocol['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Paper11 Phase 59 matched-dimension control audit."
    )
    parser.add_argument(
        "--mode",
        choices=("build-controls", "run-and-analyze", "analyze-only"),
        default="analyze-only",
    )
    parser.add_argument("--b0-features-csv", type=Path, default=None)
    parser.add_argument("--d4p8-features-csv", type=Path, default=None)
    parser.add_argument("--d4p16-features-csv", type=Path, default=None)
    parser.add_argument("--phase8-output-dir", type=Path, default=None)
    parser.add_argument("--phase59-control-dir", type=Path, default=None)
    parser.add_argument("--tile-index-csv", type=Path, default=None)
    parser.add_argument("--existing-summary-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-tile-id", default=None)
    parser.add_argument("--variants", default="D4P8,D4P16,D5R8,D5S8,D5R16,D5S16")
    parser.add_argument("--eval-tile-ids", default=None)
    parser.add_argument("--max-eval-tiles", type=int, default=5)
    parser.add_argument("--total-timesteps", type=int, default=4096)
    parser.add_argument("--eval-max-steps", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=59)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    missing = []
    if args.mode == "build-controls":
        for attr, flag in (
            ("b0_features_csv", "--b0-features-csv"),
            ("d4p8_features_csv", "--d4p8-features-csv"),
            ("d4p16_features_csv", "--d4p16-features-csv"),
        ):
            if getattr(args, attr) is None:
                missing.append(flag)
    elif args.mode == "run-and-analyze":
        for attr, flag in (
            ("phase8_output_dir", "--phase8-output-dir"),
            ("phase59_control_dir", "--phase59-control-dir"),
            ("tile_index_csv", "--tile-index-csv"),
        ):
            if getattr(args, attr) is None:
                missing.append(flag)
    elif args.existing_summary_csv is None:
        missing.append("--existing-summary-csv")

    if missing:
        raise ValueError(f"{args.mode} requires " + ", ".join(missing))


if __name__ == "__main__":
    raise SystemExit(main())
