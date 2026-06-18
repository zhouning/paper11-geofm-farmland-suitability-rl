from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase27_stability_diagnosis import (
    build_phase27_stability_diagnosis,
    write_phase27_stability_diagnosis_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose B0/B1 learned-policy stability across existing Phase 26 "
            "comparison JSON artifacts."
        )
    )
    parser.add_argument(
        "--phase26-comparison-json",
        type=Path,
        action="append",
        required=True,
        help="Repeat for each Phase 26 phase26_main_comparison.json input.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if len(args.phase26_comparison_json) < 2:
            raise ValueError(
                "Phase 27 requires at least two --phase26-comparison-json inputs"
            )
        analysis = build_phase27_stability_diagnosis(args.phase26_comparison_json)
        paths = write_phase27_stability_diagnosis_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    transition_rows = analysis["budget_transition_rows"]
    stability_counts = analysis["stability_counts"]
    print(f"Phase 27 diagnostic status: {analysis['phase27_diagnostic_status']}")
    for row in transition_rows:
        print(
            f"Budget {row['train_timesteps']} steps: "
            f"B1-B0 mean delta {row['b1_minus_b0_mean_reward']}, "
            f"positive tile-seed count {row['positive_tile_seed_count']} / "
            f"{row['total_tile_seed_count']}"
        )
    print(
        "Stability counts: "
        f"stable_positive={stability_counts['stable_positive']}, "
        f"stable_negative={stability_counts['stable_negative']}, "
        f"flip_to_positive={stability_counts['flip_to_positive']}, "
        f"flip_to_negative={stability_counts['flip_to_negative']}, "
        f"incomplete={stability_counts['incomplete']}"
    )
    print(f"Budget transition CSV: {paths['budget_transition_csv']}")
    print(f"Tile-seed stability CSV: {paths['tile_seed_stability_csv']}")
    print(f"Diagnostic summary JSON: {paths['diagnostic_summary_json']}")
    print(f"Diagnostic readiness Markdown: {paths['diagnostic_readiness_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
