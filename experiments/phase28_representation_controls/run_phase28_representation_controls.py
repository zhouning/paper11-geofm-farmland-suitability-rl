from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase28_representation_controls import (
    build_phase28_representation_control_analysis,
    run_phase28_representation_control_evaluation,
    write_phase28_representation_control_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    provided_args = set(argv or sys.argv[1:])
    try:
        if args.mode == "run-and-analyze":
            _validate_run_and_analyze_args(args, provided_args)
            protocol = run_phase28_representation_control_evaluation(
                phase2_output_dir=args.phase2_output_dir,
                phase8_output_dir=args.phase8_output_dir,
                tile_index_csv=args.tile_index_csv,
                variants=tuple(
                    part.strip() for part in args.variants.split(",") if part.strip()
                ),
                train_tile_id=args.train_tile_id,
                eval_tile_ids=args.eval_tile_ids,
                max_eval_tiles=args.max_eval_tiles,
                total_timesteps=args.total_timesteps,
                eval_max_steps=args.eval_max_steps,
                seeds=args.seeds,
                compression_match_tolerance=args.compression_match_tolerance,
            )
        else:
            if args.existing_summary_csv is None:
                raise ValueError("analyze-only requires --existing-summary-csv")
            protocol = build_phase28_representation_control_analysis(
                args.existing_summary_csv,
                compression_match_tolerance=args.compression_match_tolerance,
            )
        paths = write_phase28_representation_control_artifacts(
            protocol,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    learned = protocol.get("learned_policy", {})
    print(f"Mode: {args.mode}")
    print(f"Phase 28 diagnostic status: {protocol['phase28_diagnostic_status']}")
    if isinstance(learned, dict):
        comparator_deltas = learned.get("comparator_deltas", {})
        if isinstance(comparator_deltas, dict):
            for key in sorted(comparator_deltas):
                value = comparator_deltas[key]
                if isinstance(value, dict):
                    print(f"{key} mean reward delta: {value.get('mean_reward_delta')}")
    print(f"Summary CSV: {paths['summary_csv']}")
    print(f"Trace JSON: {paths['traces_json']}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Tile-seed delta CSV: {paths['tile_seed_delta_csv']}")
    print(f"Control readiness Markdown: {paths['control_readiness_md']}")
    print(f"Claim boundary: {protocol['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or analyze the Paper11 Phase 28 B1 representation-control "
            "evaluation package."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("run-and-analyze", "analyze-only"),
        default="analyze-only",
    )
    parser.add_argument("--phase2-output-dir", type=Path, default=None)
    parser.add_argument("--phase8-output-dir", type=Path, default=None)
    parser.add_argument("--tile-index-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing-summary-csv", type=Path, default=None)
    parser.add_argument("--variants", default="B0,B1,D2,D3,D4P8,D4P16")
    parser.add_argument("--train-tile-id", default=None)
    parser.add_argument("--eval-tile-ids", default=None)
    parser.add_argument("--max-eval-tiles", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=1024)
    parser.add_argument("--eval-max-steps", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--compression-match-tolerance", type=float, default=1e-9)
    return parser


def _validate_run_and_analyze_args(
    args: argparse.Namespace,
    provided_args: set[str],
) -> None:
    missing = []
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
