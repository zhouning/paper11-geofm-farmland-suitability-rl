from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.baseline_eval import (
    PHASE6_CLAIM_BOUNDARY,
    run_phase6_baseline_evaluator,
    write_phase6_baseline_artifacts,
)


def _parse_csv_ids(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic Paper11 Phase 6 non-learning masked baseline "
            "evaluations without training a policy."
        )
    )
    parser.add_argument(
        "--phase2-output-dir",
        type=Path,
        required=True,
        help="Directory containing experiment_variants.json and variant CSV exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where Phase 6 baseline artifacts will be written.",
    )
    parser.add_argument(
        "--variants",
        default="B0,B1,B2,B3",
        help="Comma-separated variant IDs. Default: B0,B1,B2,B3.",
    )
    parser.add_argument(
        "--policies",
        default="first_valid,seeded_random",
        help="Comma-separated policy IDs. Default: first_valid,seeded_random.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional maximum steps per baseline rollout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for deterministic random baseline selection. Default: 0.",
    )
    args = parser.parse_args(argv)

    try:
        protocol = run_phase6_baseline_evaluator(
            args.phase2_output_dir,
            variant_ids=_parse_csv_ids(args.variants),
            policy_ids=_parse_csv_ids(args.policies),
            max_steps=args.max_steps,
            seed=args.seed,
        )
        paths = write_phase6_baseline_artifacts(protocol, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for summary in protocol["summaries"]:
        print(
            "Policy "
            f"{summary['policy_id']} / Variant {summary['variant_id']}: "
            f"steps={summary['episode_steps']} "
            f"features={summary['n_features']} "
            f"total_contract_reward={float(summary['total_contract_reward']):.6f}"
        )
    print(f"Summary CSV: {paths['summary_csv']}")
    print(f"Trace JSON: {paths['traces_json']}")
    print(f"Claim boundary: {PHASE6_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
