from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase30_normalized_b1_ablation import (
    run_phase30_normalized_b1_ablation,
    write_phase30_normalized_b1_artifacts,
)
from paper11_geofm.phase33_budget_robustness import (
    build_phase33_budget_robustness,
    write_phase33_matched_baseline_comparison,
    write_phase33_budget_robustness_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    provided_args = set(argv or sys.argv[1:])
    try:
        if args.mode == "run-and-analyze":
            _validate_run_and_analyze_args(args, provided_args)
            phase30_protocol = run_phase30_normalized_b1_ablation(
                phase2_output_dir=args.phase2_output_dir,
                phase8_output_dir=args.phase8_output_dir,
                tile_index_csv=args.tile_index_csv,
                output_dir=args.output_dir / "phase30_high_budget",
                normalized_controls_dir=args.normalized_controls_dir,
                existing_control_summary_csv=(
                    args.baseline_control_summary_csv
                    if args.baseline_control_summary_csv is not None
                    else args.existing_control_summary_csv
                ),
                variants=tuple(part.strip() for part in args.variants.split(",") if part.strip()),
                train_tile_id=args.train_tile_id,
                eval_tile_ids=args.eval_tile_ids,
                max_eval_tiles=args.max_eval_tiles,
                total_timesteps=args.total_timesteps,
                eval_max_steps=args.eval_max_steps,
                seeds=args.seeds,
            )
            phase30_paths = write_phase30_normalized_b1_artifacts(
                phase30_protocol,
                args.output_dir / "phase30_high_budget",
            )
            matched_baseline = write_phase33_matched_baseline_comparison(
                args.baseline_phase30_comparison_json,
                phase30_paths["comparison_json"],
                args.output_dir / "phase33_matched_baseline_comparison.json",
            )
            comparison_inputs = [
                matched_baseline,
                phase30_paths["comparison_json"],
            ]
        else:
            if len(args.phase30_comparison_json) < 2:
                raise ValueError("analyze-only requires at least two --phase30-comparison-json inputs")
            comparison_inputs = list(args.phase30_comparison_json)
        analysis = build_phase33_budget_robustness(comparison_inputs)
        paths = write_phase33_budget_robustness_artifacts(analysis, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Mode: {args.mode}")
    print(f"Phase 33 budget status: {analysis['phase33_budget_status']}")
    print(f"Budget transition CSV: {paths['budget_transition_csv']}")
    print(f"Focal gap transition CSV: {paths['focal_gap_transition_csv']}")
    print(f"Tile-seed stability CSV: {paths['tile_seed_stability_csv']}")
    print(f"Summary JSON: {paths['summary_json']}")
    print(f"Summary Markdown: {paths['summary_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or analyze the Paper11 Phase 33 budget-robustness follow-up."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("run-and-analyze", "analyze-only"),
        default="analyze-only",
    )
    parser.add_argument(
        "--phase30-comparison-json",
        type=Path,
        action="append",
        default=[],
        help="Repeat for each existing Phase 30 comparison JSON in analyze-only mode.",
    )
    parser.add_argument("--baseline-phase30-comparison-json", type=Path, default=None)
    parser.add_argument("--baseline-control-summary-csv", type=Path, default=None)
    parser.add_argument("--phase2-output-dir", type=Path, default=None)
    parser.add_argument("--phase8-output-dir", type=Path, default=None)
    parser.add_argument("--tile-index-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalized-controls-dir", type=Path, default=None)
    parser.add_argument("--existing-control-summary-csv", type=Path, default=None)
    parser.add_argument("--variants", default="B1,N1Z,N1ZR,D4P8,D4P16")
    parser.add_argument("--train-tile-id", default=None)
    parser.add_argument("--eval-tile-ids", default=None)
    parser.add_argument("--max-eval-tiles", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=8192)
    parser.add_argument("--eval-max-steps", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    return parser


def _validate_run_and_analyze_args(
    args: argparse.Namespace,
    provided_args: set[str],
) -> None:
    missing = []
    if args.baseline_phase30_comparison_json is None:
        missing.append("--baseline-phase30-comparison-json")
    if args.baseline_control_summary_csv is None and args.existing_control_summary_csv is None:
        missing.append("--baseline-control-summary-csv or --existing-control-summary-csv")
    if args.phase2_output_dir is None:
        missing.append("--phase2-output-dir")
    if args.phase8_output_dir is None:
        missing.append("--phase8-output-dir")
    if args.tile_index_csv is None:
        missing.append("--tile-index-csv")
    for flag in (
        "--variants",
        "--total-timesteps",
        "--eval-max-steps",
        "--seeds",
    ):
        if not _was_provided(provided_args, flag):
            missing.append(flag)
    if not _was_provided(provided_args, "--max-eval-tiles") and not _was_provided(
        provided_args,
        "--eval-tile-ids",
    ):
        missing.append("--max-eval-tiles or --eval-tile-ids")
    if missing:
        raise ValueError("run-and-analyze requires " + ", ".join(missing))


def _was_provided(provided_args: set[str], flag: str) -> bool:
    return flag in provided_args or any(
        item.startswith(f"{flag}=") for item in provided_args
    )


if __name__ == "__main__":
    raise SystemExit(main())
