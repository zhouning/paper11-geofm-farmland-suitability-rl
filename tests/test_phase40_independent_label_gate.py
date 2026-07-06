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


def _phase2_dir(tmp_path: Path, rows: list[dict[str, object]] | None = None) -> Path:
    if rows is None:
        rows = []
        for index in range(12):
            rows.append(
                {
                    "block_id": f"b{index:03d}",
                    "split": "train" if index < 8 else "test",
                    "independent_irrigation_label": 1 if index in {0, 1, 8} else 0,
                    "diagnostic_internal_label": 1 if index % 2 == 0 else 0,
                    "single_class_label": 1,
                }
            )
    return _write_csv(
        tmp_path / "phase2" / "block_geofm_features.csv",
        rows,
        [
            "block_id",
            "split",
            "independent_irrigation_label",
            "diagnostic_internal_label",
            "single_class_label",
        ],
    ).parent


def _registry_csv(path: Path, rows: list[dict[str, object]]) -> Path:
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
        ],
    )


def _independent_registry_row(label_column: str = "independent_irrigation_label") -> dict[str, object]:
    return {
        "label_column": label_column,
        "label_source": "synthetic independent irrigation fixture",
        "source_type": "external_irrigation",
        "independence_level": "independent",
        "allowed_eval_roles": "test,validation,eval",
        "provenance_note": "not derived from DLTB, slope, source metadata, or GeoFM features",
        "license_or_access": "test fixture",
        "expected_positive_definition": "1",
    }


def test_phase40_no_registry_returns_missing_inputs(tmp_path):
    from paper11_geofm.phase40_independent_label_gate import (
        run_phase40_independent_label_gate,
    )

    result = run_phase40_independent_label_gate(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=None,
    )

    assert result["phase"] == "phase40_independent_label_gate"
    assert result["phase40_independent_label_gate_status"] == "independent_label_inputs_missing"
    assert result["row_counts"]["registry_rows"] == 0
    assert "go/no-go" in result["claim_boundary"]


def test_phase40_empty_registry_returns_missing_inputs(tmp_path):
    from paper11_geofm.phase40_independent_label_gate import (
        run_phase40_independent_label_gate,
    )

    registry = _registry_csv(tmp_path / "registry.csv", [])
    result = run_phase40_independent_label_gate(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=registry,
    )

    assert result["phase40_independent_label_gate_status"] == "independent_label_inputs_missing"
    assert result["row_counts"]["registry_rows"] == 0


def test_phase40_independent_csv_registry_can_pass_gate(tmp_path):
    from paper11_geofm.phase40_independent_label_gate import (
        run_phase40_independent_label_gate,
    )

    registry = _registry_csv(tmp_path / "registry.csv", [_independent_registry_row()])
    result = run_phase40_independent_label_gate(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=registry,
        min_valid_count=10,
        min_split_valid_count=2,
    )

    assert result["phase40_independent_label_gate_status"] == "independent_label_gate_passed"
    row = result["label_gate_rows"][0]
    assert row["label_column"] == "independent_irrigation_label"
    assert row["label_gate_status"] == "label_gate_passed"
    assert row["valid_label_count"] == 12
    assert row["train_valid_count"] == 8
    assert row["eval_valid_count"] == 4
    assert row["positive_rate"] == 0.25


def test_phase40_json_registry_has_csv_semantics(tmp_path):
    from paper11_geofm.phase40_independent_label_gate import (
        run_phase40_independent_label_gate,
    )

    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps([_independent_registry_row()]), encoding="utf-8")

    result = run_phase40_independent_label_gate(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=registry,
        min_valid_count=10,
        min_split_valid_count=2,
    )

    assert result["phase40_independent_label_gate_status"] == "independent_label_gate_passed"
    assert result["label_gate_rows"][0]["label_gate_status"] == "label_gate_passed"


def test_phase40_internal_label_is_diagnostic_only(tmp_path):
    from paper11_geofm.phase40_independent_label_gate import (
        run_phase40_independent_label_gate,
    )

    row = _independent_registry_row("diagnostic_internal_label")
    row["source_type"] = "diagnostic_internal"
    row["independence_level"] = "diagnostic_only"
    registry = _registry_csv(tmp_path / "registry.csv", [row])

    result = run_phase40_independent_label_gate(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=registry,
        min_valid_count=10,
        min_split_valid_count=2,
    )

    assert result["phase40_independent_label_gate_status"] == "independent_label_gate_diagnostic_only"
    assert result["label_gate_rows"][0]["label_gate_status"] == "label_gate_diagnostic_only"
    assert "not independent enough" in result["label_gate_rows"][0]["decision_reason"]


def test_phase40_missing_label_column_blocks_gate(tmp_path):
    from paper11_geofm.phase40_independent_label_gate import (
        run_phase40_independent_label_gate,
    )

    registry = _registry_csv(
        tmp_path / "registry.csv",
        [_independent_registry_row("missing_external_label")],
    )

    result = run_phase40_independent_label_gate(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=registry,
    )

    assert result["phase40_independent_label_gate_status"] == "independent_label_gate_blocked"
    assert result["label_gate_rows"][0]["label_gate_status"] == "label_missing"


def test_phase40_single_class_label_blocks_gate(tmp_path):
    from paper11_geofm.phase40_independent_label_gate import (
        run_phase40_independent_label_gate,
    )

    registry = _registry_csv(
        tmp_path / "registry.csv",
        [_independent_registry_row("single_class_label")],
    )

    result = run_phase40_independent_label_gate(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=registry,
        min_valid_count=10,
        min_split_valid_count=2,
    )

    assert result["phase40_independent_label_gate_status"] == "independent_label_gate_blocked"
    assert result["label_gate_rows"][0]["label_gate_status"] == "label_gate_blocked"
    assert "positive_rate" in result["label_gate_rows"][0]["decision_reason"]


def test_phase40_artifact_writer_creates_csv_json_markdown(tmp_path):
    from paper11_geofm.phase40_independent_label_gate import (
        run_phase40_independent_label_gate,
        write_phase40_independent_label_gate_artifacts,
    )

    result = run_phase40_independent_label_gate(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=_registry_csv(tmp_path / "registry.csv", [_independent_registry_row()]),
        min_valid_count=10,
        min_split_valid_count=2,
    )
    artifacts = write_phase40_independent_label_gate_artifacts(result, tmp_path / "outputs")

    assert {path.name for path in artifacts.values()} == {
        "phase40_label_gate_summary.csv",
        "phase40_independent_label_gate.json",
        "phase40_independent_label_gate.md",
    }
    saved = json.loads(artifacts["diagnosis_json"].read_text(encoding="utf-8"))
    assert saved["phase40_independent_label_gate_status"] == "independent_label_gate_passed"
    markdown = artifacts["diagnosis_md"].read_text(encoding="utf-8")
    assert "Phase 40 Independent Label Gate" in markdown
    assert "independent_label_gate_passed" in markdown


def test_phase40_cli_writes_outputs(tmp_path):
    phase2_dir = _phase2_dir(tmp_path)
    registry = _registry_csv(tmp_path / "registry.csv", [_independent_registry_row()])
    output_dir = tmp_path / "outputs"
    runner = ROOT / "experiments" / "phase40_independent_label_gate" / (
        "run_phase40_independent_label_gate.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--phase2-output-dir",
            str(phase2_dir),
            "--label-registry",
            str(registry),
            "--output-dir",
            str(output_dir),
            "--min-valid-count",
            "10",
            "--min-split-valid-count",
            "2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Phase 40 independent-label gate status:" in result.stdout
    assert "independent_label_gate_passed" in result.stdout
    assert (output_dir / "phase40_independent_label_gate.json").exists()
