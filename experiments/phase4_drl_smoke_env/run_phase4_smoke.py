from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.drl_smoke_env import (
    PHASE4_CLAIM_BOUNDARY,
    make_phase4_smoke_env,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a one-step Paper11 Phase 4 DRL input-contract smoke check "
            "without training a policy."
        )
    )
    parser.add_argument(
        "--phase2-output-dir",
        type=Path,
        required=True,
        help="Directory containing experiment_variants.json and variant CSV exports.",
    )
    parser.add_argument(
        "--variant",
        default="B3",
        help="Variant ID to smoke-check: B0, B1, B2, or B3. Default: B3.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max step count for the smoke environment.",
    )
    args = parser.parse_args(argv)

    try:
        env = make_phase4_smoke_env(
            args.phase2_output_dir,
            args.variant,
            max_steps=args.max_steps,
        )
        obs, info = env.reset()
        mask = env.action_masks()
        valid_actions = [idx for idx, valid in enumerate(mask.tolist()) if valid]
        if not valid_actions:
            raise ValueError("Phase 4 smoke env has no valid actions")
        action = valid_actions[0]
        next_obs, reward, terminated, truncated, step_info = env.step(action)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Variant: {info['variant_id']}")
    print(f"Observation shape: {obs.shape[0]}")
    print(f"Action space: {env.action_space}")
    print(f"Initial valid actions: {len(valid_actions)}")
    print(f"Selected block: {step_info['selected_block_id']}")
    print(f"Step reward: {reward:.6f}")
    print(f"Next observation shape: {next_obs.shape[0]}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Reward mode: {info['reward_mode']}")
    print(f"Claim boundary: {PHASE4_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
