from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.drl_inputs import load_variant_input


CLAIM_BOUNDARY = "input contract only; no DRL policy is trained or evaluated."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a ready Paper11 Phase 2 variant feature table as a DRL "
            "input matrix without training a policy."
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
        help="Variant ID to inspect: B0, B1, B2, or B3. Default: B3.",
    )
    args = parser.parse_args(argv)

    try:
        loaded = load_variant_input(args.phase2_output_dir, args.variant)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Variant: {loaded.variant_id}")
    print(f"Source table: {loaded.source_table}")
    print(f"Rows: {len(loaded.block_ids)}")
    print(f"Features: {len(loaded.feature_columns)}")
    print(
        f"Matrix shape: {loaded.state_matrix.shape[0]} x "
        f"{loaded.state_matrix.shape[1]}"
    )
    print(f"Reward mode: {loaded.reward_mode}")
    print(f"State groups: {', '.join(loaded.state_groups)}")
    print(f"Claim boundary: {CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
