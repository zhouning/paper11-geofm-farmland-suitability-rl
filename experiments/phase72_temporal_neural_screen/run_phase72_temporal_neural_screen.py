from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase72_temporal_neural_screen import (  # noqa: E402
    benchmark_phase72_temporal_neural_model,
    confirm_phase72_temporal_neural_screen,
    fit_freeze_phase72_temporal_neural_models,
    prepare_phase72_temporal_neural_screen,
    write_phase72_temporal_neural_confirmation_artifacts,
    write_phase72_temporal_neural_prepared_artifacts,
)
from paper11_geofm.phase72b_protocol import write_hashed_json  # noqa: E402


def _required(value: Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} is required for the selected mode")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Paper11 Phase 72 compact temporal neural exhaustion screen."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("prepare", "benchmark", "fit-freeze", "confirm"),
        required=True,
    )
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--phase72b-prepared-dir", type=Path, required=True)
    parser.add_argument("--phase72b-frozen-dir", type=Path, required=True)
    parser.add_argument("--phase72b-confirmation-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--frozen-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    common = {
        "phase72b_prepared_dir": args.phase72b_prepared_dir,
        "phase72b_frozen_dir": args.phase72b_frozen_dir,
        "phase72b_confirmation_dir": args.phase72b_confirmation_dir,
    }
    try:
        if args.mode == "prepare":
            package = prepare_phase72_temporal_neural_screen(
                protocol_path=_required(args.protocol, "--protocol"),
                **common,
            )
            paths = write_phase72_temporal_neural_prepared_artifacts(
                package, args.output_dir
            )
            print(f"Phase 72 temporal neural prepare status: {package['status']}")
            print(f"Counts: {package['counts']}")
            print(
                "Prepared SHA256: "
                f"{paths['manifest_sha256'].read_text(encoding='ascii').strip()}"
            )
        elif args.mode == "benchmark":
            if args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise ValueError("--output-dir must be new or empty for benchmark")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            result = benchmark_phase72_temporal_neural_model(
                prepared_dir=_required(args.prepared_dir, "--prepared-dir"),
                **common,
            )
            result_path, result_hash = write_hashed_json(
                args.output_dir / "phase72_temporal_neural_runtime_benchmark.json",
                result,
            )
            print(f"Phase 72 temporal neural benchmark status: {result['status']}")
            print(f"Rows: {result['train_rows']} train / {result['validation_rows']} validation")
            print(f"Parameters: {result['parameter_count']}")
            print(f"Elapsed seconds: {result['elapsed_seconds']:.3f}")
            print(
                "Benchmark SHA256: "
                f"{result_hash.read_text(encoding='ascii').strip()}"
            )
            assert result_path.exists()
        elif args.mode == "fit-freeze":
            selected, paths = fit_freeze_phase72_temporal_neural_models(
                prepared_dir=_required(args.prepared_dir, "--prepared-dir"),
                output_dir=args.output_dir,
                **common,
            )
            print(f"Phase 72 temporal neural fit status: {selected['status']}")
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
            result = confirm_phase72_temporal_neural_screen(
                prepared_dir=_required(args.prepared_dir, "--prepared-dir"),
                frozen_dir=_required(args.frozen_dir, "--frozen-dir"),
                **common,
            )
            paths = write_phase72_temporal_neural_confirmation_artifacts(
                result, args.output_dir
            )
            print(
                "Phase 72 temporal neural confirmation status: "
                f"{result['phase72_temporal_neural_status']}"
            )
            print(
                "Endpoint gate: "
                f"{result['endpoint_result']['phase72b_status']}"
            )
            print(f"Receipt: {paths['receipt']}")
            print(f"Next action: {result['next_action']}")
            if result["phase72_temporal_neural_status"] == (
                "phase72_temporal_neural_inputs_not_ready"
            ):
                return 1
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
