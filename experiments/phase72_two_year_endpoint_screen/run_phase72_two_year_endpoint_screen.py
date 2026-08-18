from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase72_two_year_endpoint_screen import (  # noqa: E402
    confirm_phase72_two_year_endpoint_screen,
    fit_freeze_phase72_two_year_models,
    prepare_phase72_two_year_endpoint_screen,
    write_phase72_two_year_confirmation_artifacts,
    write_phase72_two_year_prepared_artifacts,
)


def _required(value: Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} is required for the selected mode")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Paper11 Phase 72 two-year endpoint screen."
    )
    parser.add_argument(
        "--mode", choices=("prepare", "fit-freeze", "confirm"), required=True
    )
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--phase72a-package-dir", type=Path, required=True)
    parser.add_argument("--phase72b-prepared-dir", type=Path, required=True)
    parser.add_argument("--phase72b-reference-frozen-dir", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--frozen-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.mode == "prepare":
            package = prepare_phase72_two_year_endpoint_screen(
                protocol_path=_required(args.protocol, "--protocol"),
                phase72a_package_dir=args.phase72a_package_dir,
                phase72b_prepared_dir=args.phase72b_prepared_dir,
            )
            paths = write_phase72_two_year_prepared_artifacts(
                package, args.output_dir
            )
            print("Phase 72 two-year prepare status: phase72_two_year_inputs_prepared")
            print(f"Endpoint counts: {package['endpoint_counts']}")
            print(
                "Prepared SHA256: "
                f"{paths['manifest_sha256'].read_text(encoding='ascii').strip()}"
            )
        elif args.mode == "fit-freeze":
            selected, paths = fit_freeze_phase72_two_year_models(
                prepared_dir=_required(args.prepared_dir, "--prepared-dir"),
                phase72a_package_dir=args.phase72a_package_dir,
                phase72b_prepared_dir=args.phase72b_prepared_dir,
                phase72b_reference_frozen_dir=_required(
                    args.phase72b_reference_frozen_dir,
                    "--phase72b-reference-frozen-dir",
                ),
                output_dir=args.output_dir,
            )
            print(f"Phase 72 two-year fit status: {selected['status']}")
            print(f"Bundle count: {selected['bundle_count']}")
            print(
                "Selected-model SHA256: "
                f"{paths['selected_models_sha256'].read_text(encoding='ascii').strip()}"
            )
        else:
            output = args.output_dir
            if output.exists() and any(output.iterdir()):
                raise ValueError(
                    "--output-dir must be new or empty before confirmation targets are opened"
                )
            result = confirm_phase72_two_year_endpoint_screen(
                prepared_dir=_required(args.prepared_dir, "--prepared-dir"),
                phase72a_package_dir=args.phase72a_package_dir,
                phase72b_prepared_dir=args.phase72b_prepared_dir,
                frozen_dir=_required(args.frozen_dir, "--frozen-dir"),
            )
            paths = write_phase72_two_year_confirmation_artifacts(
                result, output
            )
            print(
                "Phase 72 two-year confirmation status: "
                f"{result['phase72_two_year_status']}"
            )
            for endpoint, endpoint_result in result["endpoint_results"].items():
                print(f"{endpoint}: {endpoint_result['phase72b_status']}")
            print(f"Receipt: {paths['receipt']}")
            print(f"Next action: {result['next_action']}")
            if result["phase72_two_year_status"] == (
                "phase72_two_year_inputs_not_ready"
            ):
                return 1
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
