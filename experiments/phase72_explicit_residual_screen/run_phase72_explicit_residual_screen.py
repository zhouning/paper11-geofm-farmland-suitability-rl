from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase72_explicit_residual_screen import (  # noqa: E402
    confirm_phase72_explicit_residual_screen,
    fit_freeze_phase72_explicit_residual_models,
    prepare_phase72_explicit_residual_screen,
    write_phase72_explicit_residual_confirmation_artifacts,
    write_phase72_explicit_residual_prepared_artifacts,
)


def _required(value: Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} is required for the selected mode")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Paper11 Phase 72 explicit residual exhaustion screen."
    )
    parser.add_argument(
        "--mode", choices=("prepare", "fit-freeze", "confirm"), required=True
    )
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--phase72a-package-dir", type=Path, required=True)
    parser.add_argument("--phase72b-prepared-dir", type=Path, required=True)
    parser.add_argument("--phase72b-frozen-dir", type=Path, required=True)
    parser.add_argument("--phase72b-confirmation-dir", type=Path, required=True)
    parser.add_argument(
        "--phase72-two-year-prepared-dir", type=Path, required=True
    )
    parser.add_argument(
        "--phase72-two-year-frozen-dir", type=Path, required=True
    )
    parser.add_argument(
        "--phase72-two-year-confirmation-dir", type=Path, required=True
    )
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--frozen-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    common = {
        "phase72a_package_dir": args.phase72a_package_dir,
        "phase72b_prepared_dir": args.phase72b_prepared_dir,
        "phase72b_frozen_dir": args.phase72b_frozen_dir,
        "phase72b_confirmation_dir": args.phase72b_confirmation_dir,
        "phase72_two_year_prepared_dir": (
            args.phase72_two_year_prepared_dir
        ),
        "phase72_two_year_frozen_dir": args.phase72_two_year_frozen_dir,
        "phase72_two_year_confirmation_dir": (
            args.phase72_two_year_confirmation_dir
        ),
    }
    try:
        if args.mode == "prepare":
            package = prepare_phase72_explicit_residual_screen(
                protocol_path=_required(args.protocol, "--protocol"),
                **common,
            )
            paths = write_phase72_explicit_residual_prepared_artifacts(
                package, args.output_dir
            )
            print(f"Phase 72 residual prepare status: {package['status']}")
            print(f"Endpoint counts: {package['endpoint_counts']}")
            print(
                "Prepared SHA256: "
                f"{paths['manifest_sha256'].read_text(encoding='ascii').strip()}"
            )
        elif args.mode == "fit-freeze":
            selected, paths = fit_freeze_phase72_explicit_residual_models(
                prepared_dir=_required(args.prepared_dir, "--prepared-dir"),
                output_dir=args.output_dir,
                **common,
            )
            print(f"Phase 72 residual fit status: {selected['status']}")
            print(f"Bundle count: {selected['bundle_count']}")
            print(
                "Selected-model SHA256: "
                f"{paths['selected_models_sha256'].read_text(encoding='ascii').strip()}"
            )
        else:
            if args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise ValueError(
                    "--output-dir must be new or empty before confirmation targets are opened"
                )
            result = confirm_phase72_explicit_residual_screen(
                prepared_dir=_required(args.prepared_dir, "--prepared-dir"),
                frozen_dir=_required(args.frozen_dir, "--frozen-dir"),
                **common,
            )
            paths = write_phase72_explicit_residual_confirmation_artifacts(
                result, args.output_dir
            )
            print(
                "Phase 72 residual confirmation status: "
                f"{result['phase72_explicit_residual_status']}"
            )
            for endpoint, endpoint_result in result["endpoint_results"].items():
                print(f"{endpoint}: {endpoint_result['phase72b_status']}")
            print(f"Receipt: {paths['receipt']}")
            print(f"Next action: {result['next_action']}")
            if result["phase72_explicit_residual_status"] == (
                "phase72_explicit_residual_inputs_not_ready"
            ):
                return 1
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
