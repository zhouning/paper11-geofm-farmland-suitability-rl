from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.rollout_smoke import (
    PHASE5_CLAIM_BOUNDARY,
    run_phase5_rollout_protocol,
    write_phase5_rollout_artifacts,
)


def _parse_variants(text: str) -> list[str]:
    return [part.strip().upper() for part in text.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic Paper11 Phase 5 masked-rollout protocol "
            "smoke check without training a policy."
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
        help="Directory where Phase 5 summary artifacts will be written.",
    )
    parser.add_argument(
        "--variants",
        default="B0,B1,B2,B3",
        help="Comma-separated variant IDs. Default: B0,B1,B2,B3.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional maximum steps per variant rollout.",
    )
    args = parser.parse_args(argv)

    try:
        protocol = run_phase5_rollout_protocol(
            args.phase2_output_dir,
            variant_ids=_parse_variants(args.variants),
            max_steps=args.max_steps,
        )
        paths = write_phase5_rollout_artifacts(protocol, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for summary in protocol["summaries"]:
        print(
            "Variant "
            f"{summary['variant_id']}: "
            f"steps={summary['episode_steps']} "
            f"features={summary['n_features']} "
            f"total_contract_reward={float(summary['total_contract_reward']):.6f}"
        )
    print(f"Summary CSV: {paths['summary_csv']}")
    print(f"Steps JSON: {paths['steps_json']}")
    print(f"Claim boundary: {PHASE5_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
