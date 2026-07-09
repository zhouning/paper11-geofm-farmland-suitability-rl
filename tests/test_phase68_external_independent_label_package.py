import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _phase2_rows(count: int = 12) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        rows.append(
            {
                "block_id": f"b{index:03d}",
                "split": "train" if index < 8 else "test",
                "explicit_feature_00": index,
            }
        )
    return rows


def _phase2_dir(tmp_path: Path, rows: list[dict[str, object]] | None = None) -> Path:
    if rows is None:
        rows = _phase2_rows()
    return _write_csv(
        tmp_path / "phase2" / "block_geofm_features.csv",
        rows,
        ["block_id", "split", "explicit_feature_00"],
    ).parent


def _external_labels(
    path: Path,
    values: list[object],
    label_column: str = "external_irrigation_label",
    block_ids: list[str] | None = None,
) -> Path:
    if block_ids is None:
        block_ids = [f"b{index:03d}" for index in range(len(values))]
    rows = [
        {"block_id": block_id, label_column: value}
        for block_id, value in zip(block_ids, values)
    ]
    return _write_csv(path, rows, ["block_id", label_column])


def _registry(
    path: Path,
    label_column: str = "external_irrigation_label",
    source_type: str = "external_irrigation",
    independence_level: str = "independent",
) -> Path:
    rows = [
        {
            "label_column": label_column,
            "label_source": "synthetic external fixture",
            "source_type": source_type,
            "independence_level": independence_level,
            "allowed_eval_roles": "test,validation,eval",
            "provenance_note": "not derived from DLTB, slope, source metadata, or GeoFM",
            "license_or_access": "test fixture",
            "expected_positive_definition": "1",
            "source_owner": "fixture owner",
            "collection_date_or_period": "2026 fixture",
            "spatial_join_method": "block_id fixture join",
            "original_unit": "block",
            "label_scale": "binary",
            "missing_value_policy": "blank means missing",
            "known_overlap_with_dltb_slope_or_source_metadata": "none",
            "contact_or_access_note": "fixture only",
        }
    ]
    return _write_csv(
        path,
        rows,
        [
            "label_column",
            "label_source",
            "source_type",
            "independence_level",
            "allowed_eval_roles",
            "provenance_note",
            "license_or_access",
            "expected_positive_definition",
            "source_owner",
            "collection_date_or_period",
            "spatial_join_method",
            "original_unit",
            "label_scale",
            "missing_value_policy",
            "known_overlap_with_dltb_slope_or_source_metadata",
            "contact_or_access_note",
        ],
    )


def test_phase68_template_only_generates_package_ready_status_and_templates(tmp_path):
    from paper11_geofm.phase68_external_independent_label_package import (
        build_phase68_external_independent_label_package,
        write_phase68_external_independent_label_package_artifacts,
    )

    analysis = build_phase68_external_independent_label_package(
        phase2_output_dir=_phase2_dir(tmp_path),
    )

    assert analysis["phase"] == "phase68_external_independent_label_package"
    assert analysis["phase68_status"] == "external_label_package_ready"
    assert analysis["row_counts"]["phase2_block_rows"] == 12
    assert analysis["row_counts"]["template_rows"] == 12
    assert analysis["label_preflight_rows"] == []
    assert "does not train" in analysis["claim_boundary"]

    template_rows = analysis["external_label_template_rows"]
    assert template_rows[0]["block_id"] == "b000"
    assert "external_independent_label" in template_rows[0]
    registry_rows = analysis["registry_template_rows"]
    assert registry_rows[0]["source_type"] == "external_soil"

    artifacts = write_phase68_external_independent_label_package_artifacts(
        analysis,
        tmp_path / "outputs",
    )
    expected_names = {
        "phase68_external_label_template.csv",
        "phase68_label_registry_template.csv",
        "phase68_external_label_package_readme.md",
        "phase68_label_preflight.csv",
        "phase68_package_summary.csv",
        "phase68_external_independent_label_package.json",
        "phase68_external_independent_label_package.md",
    }
    assert {path.name for path in artifacts.values()} == expected_names
    readme = (
        tmp_path / "outputs" / "phase68_external_label_package_readme.md"
    ).read_text(encoding="utf-8")
    assert "block_id" in readme
    assert "Phase 40" in readme

def test_phase68_validation_mode_without_inputs_reports_missing(tmp_path):
    from paper11_geofm.phase68_external_independent_label_package import (
        build_phase68_external_independent_label_package,
    )

    analysis = build_phase68_external_independent_label_package(
        phase2_output_dir=_phase2_dir(tmp_path),
        validation_mode=True,
    )

    assert analysis["phase68_status"] == "external_label_inputs_missing"
    assert analysis["row_counts"]["label_preflight_rows"] == 0
    assert "missing" in analysis["recommended_next_step"].lower()


def test_phase68_external_csv_rejects_blank_block_id(tmp_path):
    from paper11_geofm.phase68_external_independent_label_package import (
        build_phase68_external_independent_label_package,
    )

    labels = _external_labels(
        tmp_path / "labels.csv",
        [1, 0],
        block_ids=["b000", ""],
    )
    registry = _registry(tmp_path / "registry.csv")

    try:
        build_phase68_external_independent_label_package(
            phase2_output_dir=_phase2_dir(tmp_path),
            external_label_csvs=labels,
            label_registry=registry,
            validation_mode=True,
        )
    except ValueError as exc:
        assert "blank block_id" in str(exc)
    else:
        raise AssertionError("Expected blank block_id to raise ValueError")


def test_phase68_external_csv_rejects_duplicate_block_id(tmp_path):
    from paper11_geofm.phase68_external_independent_label_package import (
        build_phase68_external_independent_label_package,
    )

    labels = _external_labels(
        tmp_path / "labels.csv",
        [1, 0],
        block_ids=["b000", "b000"],
    )
    registry = _registry(tmp_path / "registry.csv")

    try:
        build_phase68_external_independent_label_package(
            phase2_output_dir=_phase2_dir(tmp_path),
            external_label_csvs=labels,
            label_registry=registry,
            validation_mode=True,
        )
    except ValueError as exc:
        assert "duplicate block_id b000" in str(exc)
    else:
        raise AssertionError("Expected duplicate block_id to raise ValueError")

def test_phase68_diagnostic_label_is_blocked_from_phase40_ready_status(tmp_path):
    from paper11_geofm.phase68_external_independent_label_package import (
        build_phase68_external_independent_label_package,
    )

    labels = _external_labels(
        tmp_path / "labels.csv",
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    )
    registry = _registry(
        tmp_path / "registry.csv",
        source_type="dltb_derived",
        independence_level="leakage_risk",
    )

    analysis = build_phase68_external_independent_label_package(
        phase2_output_dir=_phase2_dir(tmp_path),
        external_label_csvs=labels,
        label_registry=registry,
        validation_mode=True,
        min_valid_count=10,
        min_split_valid_count=2,
    )

    assert analysis["phase68_status"] == "independent_label_route_blocked"
    row = analysis["label_preflight_rows"][0]
    assert row["label_preflight_status"] == "label_diagnostic_only"
    assert "not independent enough" in row["decision_reason"]


def test_phase68_valid_independent_label_is_ready_for_phase40(tmp_path):
    from paper11_geofm.phase68_external_independent_label_package import (
        build_phase68_external_independent_label_package,
    )

    labels = _external_labels(
        tmp_path / "labels.csv",
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    )
    registry = _registry(tmp_path / "registry.csv")

    analysis = build_phase68_external_independent_label_package(
        phase2_output_dir=_phase2_dir(tmp_path),
        external_label_csvs=labels,
        label_registry=registry,
        validation_mode=True,
        min_valid_count=10,
        min_split_valid_count=2,
    )

    assert analysis["phase68_status"] == "phase40_ready_to_rerun_with_external_label"
    row = analysis["label_preflight_rows"][0]
    assert row["label_preflight_status"] == "label_ready_for_phase40"
    assert row["valid_label_count"] == 12
    assert row["missing_count"] == 0
    assert row["positive_count"] == 6
    assert row["negative_count"] == 6
    assert row["train_positive_count"] == 4
    assert row["eval_positive_count"] == 2

def test_phase68_runner_template_only_cli(tmp_path):
    script = (
        ROOT
        / "experiments"
        / "phase68_external_independent_label_package"
        / "run_phase68_external_independent_label_package.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--phase2-output-dir",
            str(_phase2_dir(tmp_path)),
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Phase 68 external-label package status: external_label_package_ready" in result.stdout
    assert (tmp_path / "outputs" / "phase68_external_independent_label_package.json").exists()
