from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.maskableppo_smoke import (
    PHASE7_CLAIM_BOUNDARY,
    run_phase7_maskableppo_smoke,
    write_phase7_maskableppo_artifact,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Paper11 Phase 7 MaskablePPO compatibility smoke check "
            "without training or evaluating a useful policy."
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
        help="Directory where the Phase 7 smoke artifact will be written.",
    )
    parser.add_argument(
        "--variant",
        default="B3",
        help="Variant ID to smoke-test. Default: B3.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=8,
        help="Tiny MaskablePPO learn() timestep count. Default: 8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for env reset and MaskablePPO initialization. Default: 0.",
    )
    args = parser.parse_args(argv)

    try:
        summary = run_phase7_maskableppo_smoke(
            args.phase2_output_dir,
            variant_id=args.variant,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
        )
        artifact_path = write_phase7_maskableppo_artifact(summary, args.output_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Variant: {summary['variant_id']}")
    print(f"Observation shape: {summary['observation_shape']}")
    print(f"Action space: Discrete({summary['action_space_n']})")
    print(f"Masking supported: {summary['masking_supported']}")
    print(f"Initial valid actions: {summary['initial_valid_actions']}")
    print(f"Learn timesteps: {summary['learn_timesteps']}")
    print(f"Predicted action: {summary['predicted_action']}")
    print(f"Predicted action valid: {summary['predicted_action_valid']}")
    print(f"Selected block: {summary['selected_block_id']}")
    print(f"Artifact: {artifact_path}")
    print(f"Claim boundary: {PHASE7_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
