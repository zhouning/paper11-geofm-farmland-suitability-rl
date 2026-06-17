from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.padded_heldout_policy import (
    run_phase25_padded_heldout_policy,
    write_phase25_padded_heldout_policy_artifacts,
)
from paper11_geofm.phase26_main_experiment import (
    build_phase26_main_empirical_analysis,
    write_phase26_main_empirical_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    provided_args = set(argv or sys.argv[1:])

    try:
        if args.mode == "run-and-analyze":
            _validate_run_and_analyze_args(args, provided_args)
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
            write_phase25_padded_heldout_policy_artifacts(
                protocol,
                args.phase25_output_dir,
            )

        analysis = build_phase26_main_empirical_analysis(args.phase25_output_dir)
        paths = write_phase26_main_empirical_artifacts(analysis, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    learned = analysis["learned_policy"]
    print(f"Mode: {args.mode}")
    print(f"Phase 26 claim status: {analysis['phase26_claim_status']}")
    print(
        "B1-B0 learned-policy mean reward delta: "
        f"{learned['B1_minus_B0_mean_reward']}"
    )
    print(
        "Positive tile-seed count: "
        f"{learned['positive_tile_seed_count']} / "
        f"{learned['total_tile_seed_count']}"
    )
    print(f"Main summary CSV: {paths['main_summary_csv']}")
    print(f"Tile-seed delta CSV: {paths['tile_seed_delta_csv']}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Claim readiness Markdown: {paths['claim_readiness_md']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or analyze the Phase 26 B0/B1 padded held-out main "
            "empirical experiment package."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("analyze-only", "run-and-analyze"),
        default="analyze-only",
    )
    parser.add_argument("--phase25-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase2-output-dir", type=Path, default=None)
    parser.add_argument("--tile-index-csv", type=Path, default=None)
    parser.add_argument("--variants", default="B0,B1")
    parser.add_argument("--train-tile-id", default=None)
    parser.add_argument("--eval-tile-ids", default=None)
    parser.add_argument("--max-eval-tiles", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=1024)
    parser.add_argument("--eval-max-steps", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    return parser


def _validate_run_and_analyze_args(
    args: argparse.Namespace,
    provided_args: set[str],
) -> None:
    missing = []
    if args.phase2_output_dir is None:
        missing.append("--phase2-output-dir")
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
