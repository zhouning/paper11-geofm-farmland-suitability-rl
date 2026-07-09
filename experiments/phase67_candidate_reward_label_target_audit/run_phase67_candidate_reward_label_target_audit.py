from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase67_candidate_reward_label_target_audit import (
    run_phase67_candidate_reward_label_target_audit,
    write_phase67_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = run_phase67_candidate_reward_label_target_audit(
            phase2_output_dir=args.phase2_output_dir,
            phase8_output_dir=args.phase8_output_dir,
            phase61_output_dir=args.phase61_output_dir,
            tile_index_csv=args.tile_index_csv,
            phase10_json=args.phase10_json,
            phase18_json=args.phase18_json,
            phase66_json=args.phase66_json,
            phase39_json=args.phase39_json,
            phase40_json=args.phase40_json,
            variants=args.variants,
            label_columns=args.label_columns,
            top_k_values=args.top_k_values,
        )
        paths = write_phase67_artifacts(analysis, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    gate = analysis["candidate_target_gate"]
    print(f"Phase 67 status: {gate['phase67_status']}")
    print(f"Audit JSON: {paths['audit_json']}")
    print(f"Inventory CSV: {paths['inventory_csv']}")
    print(f"Gate Audit CSV: {paths['gate_audit_csv']}")
    print(f"Information Gain CSV: {paths['information_gain_csv']}")
    print(f"Audit Markdown: {paths['audit_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 67 candidate reward/label target audit."
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--phase8-output-dir", type=Path, default=None)
    parser.add_argument("--phase61-output-dir", type=Path, default=None)
    parser.add_argument("--tile-index-csv", type=Path, default=None)
    parser.add_argument("--phase10-json", type=Path, required=True)
    parser.add_argument("--phase18-json", type=Path, required=True)
    parser.add_argument("--phase66-json", type=Path, required=True)
    parser.add_argument("--phase39-json", type=Path, default=None)
    parser.add_argument("--phase40-json", type=Path, default=None)
    parser.add_argument("--variants", default="B0,D4P8,D4P16,D6R8,D6R16")
    parser.add_argument(
        "--label-columns",
        default="current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label",
    )
    parser.add_argument("--top-k-values", default="8,16,32")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
