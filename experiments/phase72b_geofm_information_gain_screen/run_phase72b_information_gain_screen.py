from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase72b_information_gain_screen import (  # noqa: E402
    confirm_phase72b_information_gain_screen,
    prepare_phase72b_information_gain_screen,
    write_phase72b_confirmation_artifacts,
    write_phase72b_prepared_artifacts,
)
from paper11_geofm.phase72b_models import (  # noqa: E402
    fit_freeze_phase72b_models,
)


def _parse_region_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected region=path, got {value}")
        region, raw_path = value.split("=", 1)
        region = region.strip().lower()
        if not region or region in result:
            raise ValueError(
                f"region mapping must be nonblank and unique: {region}"
            )
        result[region] = Path(raw_path)
    return result


def _required(value: Path | None, name: str) -> Path:
    if value is None:
        raise ValueError(f"{name} is required for the selected mode")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 72B GeoFM information-gain screen"
    )
    parser.add_argument(
        "--mode", choices=("prepare", "fit-freeze", "confirm"), required=True
    )
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--phase72a-region-config", type=Path)
    parser.add_argument("--phase72a-package-dir", type=Path)
    parser.add_argument("--embedding-dir", action="append", default=[])
    parser.add_argument("--label-dir", action="append", default=[])
    parser.add_argument("--terrain-dir", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--frozen-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.mode == "prepare":
            package = prepare_phase72b_information_gain_screen(
                protocol_path=_required(args.protocol, "--protocol"),
                phase72a_region_config=_required(
                    args.phase72a_region_config, "--phase72a-region-config"
                ),
                phase72a_package_dir=_required(
                    args.phase72a_package_dir, "--phase72a-package-dir"
                ),
                embedding_dirs=_parse_region_paths(args.embedding_dir),
                label_dirs=_parse_region_paths(args.label_dir),
                terrain_dir=_required(args.terrain_dir, "--terrain-dir"),
            )
            paths = write_phase72b_prepared_artifacts(package, args.output_dir)
            print("Phase 72B prepare status: phase72b_inputs_prepared")
            print(
                "Row counts: "
                f"development={len(package['development_targets']['sample_index'])}, "
                f"confirmation={len(package['confirmation_targets']['sample_index'])}"
            )
            print(
                "Frozen protocol SHA256: "
                f"{paths['protocol_hash'].read_text(encoding='ascii').strip()}"
            )
            print(f"Claim boundary: {package['claim_boundary']}")
        elif args.mode == "fit-freeze":
            selected, paths = fit_freeze_phase72b_models(
                prepared_dir=_required(args.prepared_dir, "--prepared-dir"),
                output_dir=args.output_dir,
            )
            print(f"Phase 72B fit-freeze status: {selected['status']}")
            print(f"Bundle rows: {len(selected['bundle_records'])}")
            print(
                "Selected models SHA256: "
                f"{paths['selected_models_hash'].read_text(encoding='ascii').strip()}"
            )
            print(f"Claim boundary: {selected['claim_boundary']}")
        else:
            result = confirm_phase72b_information_gain_screen(
                prepared_dir=_required(args.prepared_dir, "--prepared-dir"),
                frozen_dir=_required(args.frozen_dir, "--frozen-dir"),
            )
            paths = write_phase72b_confirmation_artifacts(
                result, args.output_dir
            )
            print(f"Phase 72B confirmation status: {result['phase72b_status']}")
            print(f"Row counts: {result['counts']}")
            print(
                "Frozen hashes: "
                f"protocol={result['frozen_protocol_sha256']}, "
                f"selected_models={result['selected_models_sha256']}"
            )
            for key, path in paths.items():
                print(f"{key}: {path}")
            print(f"Blockers: {result['blockers']}")
            print(f"Next action: {result['next_action']}")
            print(f"Claim boundary: {result['claim_boundary']}")
            if result["phase72b_status"] == "phase72b_inputs_not_ready":
                return 1
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
