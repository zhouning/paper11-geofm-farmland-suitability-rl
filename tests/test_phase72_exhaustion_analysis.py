from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper11_geofm.phase72b_protocol import canonical_json_sha256


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    confirmation = tmp_path / "confirmation"
    artifact_names = [
        "phase72b_metrics.csv",
        "phase72b_predictions.csv",
        "phase72b_calibration.csv",
        "phase72b_bootstrap_deltas.csv",
        "phase72b_control_comparison.csv",
        "phase72b_confirmation_control_manifest.csv",
        "phase72b_transfer_summary.csv",
        "phase72b_information_gain_screen.json",
        "phase72b_information_gain_screen.md",
    ]
    receipt_artifacts = []
    for name in artifact_names:
        path = confirmation / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        receipt_artifacts.append(
            {"name": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )

    phase72a = _write_json(
        tmp_path / "phase72a.json",
        {
            "phase72a_status": "phase72a_label_inputs_ready",
            "region_audits": [
                {
                    "file_manifest_rows": [
                        {
                            "asset_type": "label",
                            "source_id": "esri",
                            "independent_label": True,
                        },
                        {
                            "asset_type": "label",
                            "source_id": "weak_dltb",
                            "independent_label": False,
                        },
                        {"asset_type": "embedding", "source_id": "alphaearth"},
                    ]
                }
            ],
            "row_counts": {"sample_rows": 10},
        },
    )
    summary = _write_csv(
        tmp_path / "summary.csv",
        ["region_id", "horizon", "eligible_rows", "positive_rows"],
        [
            {"region_id": "bishan", "horizon": "1y", "eligible_rows": 10, "positive_rows": 6},
            {"region_id": "bishan", "horizon": "2y", "eligible_rows": 8, "positive_rows": 5},
            {"region_id": "bishan", "horizon": "continuous_2y", "eligible_rows": 8, "positive_rows": 4},
        ],
    )
    review = _write_csv(
        tmp_path / "review.csv",
        ["review_label", "review_source", "review_source_id", "review_date", "review_confidence"],
        [{"review_label": "", "review_source": "", "review_source_id": "", "review_date": "", "review_confidence": ""}],
    )
    phase72b = _write_json(
        tmp_path / "phase72b.json",
        {
            "phase72b_status": "geofm_information_not_supported",
            "counts": {"confirmation_rows": 2},
            "evidence": {
                "controls": [
                    {"control_id": "temporal_order_shuffle", "passed": False},
                    {"control_id": "spatial_shuffle", "passed": False},
                    {"control_id": "random_projection", "passed": True},
                ],
                "transfers": [
                    {"axis_id": "bishan_to_dongxing", "passed": False},
                    {"axis_id": "dongxing_to_bishan", "passed": False},
                ],
                "spatial_regions": [
                    {"region_id": "bishan", "passed": False},
                    {"region_id": "dongxing", "passed": True},
                ],
                "spatial_folds": [{"axis_id": "spatial_bishan_fold0"}],
            },
        },
    )
    protocol = _write_json(
        tmp_path / "protocol.json",
        {
            "years": {"train": [2017], "validation": [2022], "test": [2023]},
            "spatial": {"folds": 5, "buffer_rings": 1},
            "controls": {"seeds": [72, 73, 74, 75, 76]},
            "variants": ["explicit_history", "explicit_plus_geofm_temporal_full"],
        },
    )
    metrics = _write_csv(
        tmp_path / "metrics.csv",
        ["variant_id", "model_family"],
        [
            {"variant_id": "explicit_history", "model_family": "logistic"},
            {"variant_id": "explicit_plus_geofm_temporal_full", "model_family": "hgb"},
        ],
    )
    control = _write_csv(tmp_path / "control.csv", ["control_id"], [{"control_id": "temporal_order_shuffle"}])
    transfer = _write_csv(tmp_path / "transfer.csv", ["axis_id"], [{"axis_id": "bishan_to_dongxing"}])
    receipt = _write_json(
        tmp_path / "receipt.json",
        {
            "phase72b_status": "geofm_information_not_supported",
            "artifacts": receipt_artifacts,
        },
    )
    receipt_hash = tmp_path / "receipt.sha256"
    receipt_hash.write_text(
        canonical_json_sha256(json.loads(receipt.read_text(encoding="utf-8"))) + "\n",
        encoding="ascii",
    )
    return {
        "phase72a_json": phase72a,
        "phase72a_summary_csv": summary,
        "phase72a_review_csv": review,
        "phase72b_json": phase72b,
        "phase72b_protocol_json": protocol,
        "phase72b_metrics_csv": metrics,
        "phase72b_control_csv": control,
        "phase72b_transfer_csv": transfer,
        "phase72b_receipt_json": receipt,
        "phase72b_receipt_sha256": receipt_hash,
        "phase72b_confirmation_dir": confirmation,
    }


def _with_two_year_evidence(
    tmp_path: Path, paths: dict[str, Path]
) -> dict[str, Path]:
    confirmation = tmp_path / "two-year-confirmation"
    artifact_names = [
        "phase72_two_year_metrics.csv",
        "phase72_two_year_predictions.csv",
        "phase72_two_year_bootstrap_deltas.csv",
        "phase72_two_year_control_comparison.csv",
        "phase72_two_year_transfer_summary.csv",
        "phase72_two_year_spatial_summary.csv",
        "phase72_two_year_endpoint_screen.json",
        "phase72_two_year_endpoint_screen.md",
    ]
    receipt_artifacts = []
    for name in artifact_names:
        path = confirmation / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        receipt_artifacts.append(
            {
                "name": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    two_year_json = _write_json(
        tmp_path / "two-year.json",
        {
            "phase72_two_year_status": (
                "two_year_geofm_information_not_supported"
            ),
            "endpoint_results": {
                "conversion_2y": {
                    "phase72b_status": "geofm_information_not_supported"
                },
                "noncontinuous_persistence_2y": {
                    "phase72b_status": "geofm_information_not_supported"
                },
            },
            "counts": {"endpoints": 2, "bundle_count": 142},
        },
    )
    receipt = _write_json(
        tmp_path / "two-year-receipt.json",
        {
            "phase72_two_year_status": (
                "two_year_geofm_information_not_supported"
            ),
            "artifacts": receipt_artifacts,
        },
    )
    receipt_hash = tmp_path / "two-year-receipt.sha256"
    receipt_hash.write_text(
        canonical_json_sha256(
            json.loads(receipt.read_text(encoding="utf-8"))
        )
        + "\n",
        encoding="ascii",
    )
    return {
        **paths,
        "phase72_two_year_json": two_year_json,
        "phase72_two_year_receipt_json": receipt,
        "phase72_two_year_receipt_sha256": receipt_hash,
        "phase72_two_year_confirmation_dir": confirmation,
    }


def _with_residual_evidence(
    tmp_path: Path, paths: dict[str, Path]
) -> dict[str, Path]:
    confirmation = tmp_path / "residual-confirmation"
    artifact_names = [
        "phase72_explicit_residual_metrics.csv",
        "phase72_explicit_residual_predictions.csv",
        "phase72_explicit_residual_bootstrap_deltas.csv",
        "phase72_explicit_residual_control_comparison.csv",
        "phase72_explicit_residual_transfer_summary.csv",
        "phase72_explicit_residual_spatial_summary.csv",
        "phase72_explicit_residual_screen.json",
        "phase72_explicit_residual_screen.md",
    ]
    receipt_artifacts = []
    for name in artifact_names:
        path = confirmation / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        receipt_artifacts.append(
            {
                "name": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    residual_json = _write_json(
        confirmation / "phase72_explicit_residual_screen.json",
        {
            "phase72_explicit_residual_status": (
                "explicit_residual_information_mixed"
            ),
            "phase72c_allowed": False,
            "prepared_sha256": "1" * 64,
            "selected_models_sha256": "2" * 64,
            "endpoint_results": {
                "conversion_1y": {
                    "phase72b_status": "geofm_information_mixed"
                },
                "conversion_2y": {
                    "phase72b_status": "geofm_information_not_supported"
                },
                "noncontinuous_persistence_2y": {
                    "phase72b_status": "geofm_information_not_supported"
                },
            },
            "counts": {"endpoints": 3, "bundle_count": 123},
        },
    )
    for entry in receipt_artifacts:
        if entry["name"] == "phase72_explicit_residual_screen.json":
            entry["sha256"] = hashlib.sha256(
                residual_json.read_bytes()
            ).hexdigest()
    receipt = _write_json(
        tmp_path / "residual-receipt.json",
        {
            "phase72_explicit_residual_status": (
                "explicit_residual_information_mixed"
            ),
            "phase72c_allowed": False,
            "prepared_sha256": "1" * 64,
            "selected_models_sha256": "2" * 64,
            "artifacts": receipt_artifacts,
        },
    )
    receipt_hash = tmp_path / "residual-receipt.sha256"
    receipt_hash.write_text(
        canonical_json_sha256(
            json.loads(receipt.read_text(encoding="utf-8"))
        )
        + "\n",
        encoding="ascii",
    )
    return {
        **paths,
        "phase72_residual_json": residual_json,
        "phase72_residual_receipt_json": receipt,
        "phase72_residual_receipt_sha256": receipt_hash,
        "phase72_residual_confirmation_dir": confirmation,
    }


def test_phase72_exhaustion_separates_negative_from_unresolved(tmp_path):
    from paper11_geofm.phase72_exhaustion_analysis import (
        build_phase72_exhaustion_analysis,
        write_phase72_exhaustion_analysis_artifacts,
    )

    analysis = build_phase72_exhaustion_analysis(**_fixture_paths(tmp_path))

    assert analysis["phase72_exhaustion_status"] == "phase72_exhaustion_criteria_not_fully_evaluated"
    assert analysis["route_decision"] == "phase72_route_closed_at_phase72b_gate"
    assert analysis["phase72c_allowed"] is False
    assert analysis["counts"]["independent_label_sources"] == 1
    assert analysis["counts"]["receipt_hash_rows"] == 10
    criteria = {row["criterion_id"]: row for row in analysis["criteria_rows"]}
    assert criteria["prediction_outcome_gate"]["criterion_status"] == "evaluated_negative"
    assert criteria["strict_temporal_and_representation_controls"]["criterion_status"] == "evaluated_negative"
    assert criteria["independent_annual_products"]["criterion_status"] == "data_gap"
    assert criteria["temporal_neural_model"]["criterion_status"] == "not_evaluated"
    assert criteria["constrained_planning_outcomes"]["criterion_status"] == "not_evaluated"
    claims = {row["claim_id"]: row for row in analysis["claim_boundary_rows"]}
    assert claims["phase72b_low_cost_screen_negative"]["claim_status"] == "allowed"
    assert claims["complete_future_aware_geofm_exhaustion"]["claim_status"] == "blocked"

    artifacts = write_phase72_exhaustion_analysis_artifacts(analysis, tmp_path / "outputs")
    assert {path.name for path in artifacts.values()} == {
        "phase72_exhaustion_criteria.csv",
        "phase72_exhaustion_claim_boundary.csv",
        "phase72_exhaustion_artifact_hashes.csv",
        "phase72_exhaustion_analysis.json",
        "phase72_exhaustion_analysis.md",
    }
    assert "phase72_exhaustion_criteria_not_fully_evaluated" in (
        tmp_path / "outputs" / "phase72_exhaustion_analysis.md"
    ).read_text(encoding="utf-8")


def test_phase72_exhaustion_rejects_tampered_receipt_artifact(tmp_path):
    from paper11_geofm.phase72_exhaustion_analysis import build_phase72_exhaustion_analysis

    paths = _fixture_paths(tmp_path)
    (paths["phase72b_confirmation_dir"] / "phase72b_metrics.csv").write_text("tampered", encoding="utf-8")
    analysis = build_phase72_exhaustion_analysis(**paths)

    assert analysis["phase72_exhaustion_status"] == "phase72_exhaustion_inputs_not_ready"
    assert any("phase72b_metrics.csv" in item for item in analysis["integrity_blockers"])


def test_phase72_exhaustion_integrates_negative_two_year_screen(tmp_path):
    from paper11_geofm.phase72_exhaustion_analysis import (
        build_phase72_exhaustion_analysis,
    )

    paths = _with_two_year_evidence(tmp_path, _fixture_paths(tmp_path))
    analysis = build_phase72_exhaustion_analysis(**paths)

    criteria = {row["criterion_id"]: row for row in analysis["criteria_rows"]}
    assert criteria["one_and_two_year_endpoints"]["criterion_status"] == (
        "evaluated_complete"
    )
    assert criteria["two_year_prediction_outcome_gate"][
        "criterion_status"
    ] == "evaluated_negative"
    assert "one_and_two_year_endpoints" not in analysis["unresolved_criteria"]
    assert analysis["two_year_evidence"]["status"] == (
        "two_year_geofm_information_not_supported"
    )
    assert analysis["counts"]["receipt_hash_rows"] == 19
    assert analysis["counts"]["integrity_blockers"] == 0


def test_phase72_exhaustion_integrates_mixed_explicit_residual_screen(tmp_path):
    from paper11_geofm.phase72_exhaustion_analysis import (
        build_phase72_exhaustion_analysis,
    )

    paths = _with_residual_evidence(
        tmp_path,
        _with_two_year_evidence(tmp_path, _fixture_paths(tmp_path)),
    )
    analysis = build_phase72_exhaustion_analysis(**paths)

    criteria = {row["criterion_id"]: row for row in analysis["criteria_rows"]}
    assert criteria["explicit_residual_model"]["criterion_status"] == (
        "evaluated_mixed"
    )
    assert "explicit_residual_model" not in analysis["unresolved_criteria"]
    assert analysis["residual_evidence"]["status"] == (
        "explicit_residual_information_mixed"
    )
    assert analysis["counts"]["receipt_hash_rows"] == 28
    assert analysis["counts"]["unresolved_criteria"] == 4
    assert analysis["counts"]["negative_or_mixed_criteria"] == 6
    assert analysis["counts"]["integrity_blockers"] == 0


def test_phase72_exhaustion_rejects_residual_binding_mismatch(tmp_path):
    from paper11_geofm.phase72_exhaustion_analysis import (
        build_phase72_exhaustion_analysis,
    )

    paths = _with_residual_evidence(tmp_path, _fixture_paths(tmp_path))
    residual = json.loads(paths["phase72_residual_json"].read_text())
    residual["selected_models_sha256"] = "3" * 64
    paths["phase72_residual_json"].write_text(json.dumps(residual))

    analysis = build_phase72_exhaustion_analysis(**paths)

    assert analysis["phase72_exhaustion_status"] == (
        "phase72_exhaustion_inputs_not_ready"
    )
    assert any(
        "residual receipt binding" in blocker
        for blocker in analysis["integrity_blockers"]
    )


def test_phase72_exhaustion_rejects_tampered_receipt_sidecar(tmp_path):
    from paper11_geofm.phase72_exhaustion_analysis import build_phase72_exhaustion_analysis

    paths = _fixture_paths(tmp_path)
    paths["phase72b_receipt_sha256"].write_text("0" * 64 + "\n", encoding="ascii")
    analysis = build_phase72_exhaustion_analysis(**paths)

    assert analysis["phase72_exhaustion_status"] == "phase72_exhaustion_inputs_not_ready"
    assert any("receipt canonical hash" in item for item in analysis["integrity_blockers"])


def test_phase72_exhaustion_requires_all_status_inputs(tmp_path):
    from paper11_geofm.phase72_exhaustion_analysis import build_phase72_exhaustion_analysis

    paths = _fixture_paths(tmp_path)
    phase72b = json.loads(paths["phase72b_json"].read_text(encoding="utf-8"))
    phase72b.pop("phase72b_status")
    paths["phase72b_json"].write_text(json.dumps(phase72b), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required status field: phase72b_status"):
        build_phase72_exhaustion_analysis(**paths)


def test_phase72_exhaustion_runner_cli_writes_artifacts(tmp_path):
    paths = _fixture_paths(tmp_path)
    output_dir = tmp_path / "cli-output"
    script = ROOT / "experiments" / "phase72_exhaustion_analysis" / "run_phase72_exhaustion_analysis.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            *sum(([f"--{key.replace('_', '-')}", str(value)] for key, value in paths.items()), []),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "phase72_exhaustion_criteria_not_fully_evaluated" in result.stdout
    assert (output_dir / "phase72_exhaustion_analysis.json").is_file()
