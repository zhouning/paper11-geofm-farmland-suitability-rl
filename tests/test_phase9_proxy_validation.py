import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _run_phase2_fixture(output_dir: Path) -> Path:
    runner_path = (
        ROOT / "experiments" / "phase2_block_geofm_features" / "run_phase2.py"
    )
    spec = importlib.util.spec_from_file_location("phase2_runner_phase9", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    fixture_dir = ROOT / "data" / "bishan_phase2_csv_sample"
    exit_code = module.main(
        [
            "--mapping-csv",
            str(fixture_dir / "block_pixel_mapping.csv"),
            "--attributes-csv",
            str(fixture_dir / "block_attributes.csv"),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    return output_dir


def test_phase9_builds_proxy_validation_report_from_phase2_fixture(tmp_path):
    from paper11_geofm.proxy_validation import (
        PHASE9_CLAIM_BOUNDARY,
        build_phase9_proxy_validation_report,
    )

    phase2_dir = _run_phase2_fixture(tmp_path / "phase2")

    report = build_phase9_proxy_validation_report(
        phase2_dir,
        label_columns=(
            "stable_farmland_label",
            "high_standard_farmland_label",
            "missing_label",
        ),
    )

    assert report["phase"] == "phase9_proxy_validation_report"
    assert report["block_table"] == "block_geofm_features.csv"
    assert report["n_blocks"] == 4
    assert report["label_columns_requested"] == [
        "stable_farmland_label",
        "high_standard_farmland_label",
        "missing_label",
    ]
    assert report["label_columns_available"] == [
        "stable_farmland_label",
        "high_standard_farmland_label",
    ]
    assert report["label_columns_missing"] == ["missing_label"]
    assert report["suitability_summary"]["count"] == 4
    assert set(report["suitability_summary"]) == {
        "count",
        "min",
        "max",
        "mean",
        "std",
        "q25",
        "median",
        "q75",
    }
    stable = report["labels"]["stable_farmland_label"]
    assert stable["validation_available"] is True
    assert stable["positive_count"] == 2
    assert stable["negative_count"] == 2
    assert stable["valid_label_count"] == 4
    assert stable["missing_label_count"] == 0
    assert stable["mean_difference"] is not None
    assert stable["rank_auc"] is not None
    assert stable["interpretation"] in {
        "positive_alignment",
        "negative_or_no_alignment",
    }
    assert report["labels"]["missing_label"]["interpretation"] == "label_unavailable"
    assert report["claim_boundary"] == PHASE9_CLAIM_BOUNDARY


def test_phase9_missing_block_table_raises(tmp_path):
    from paper11_geofm.proxy_validation import build_phase9_proxy_validation_report

    with pytest.raises(FileNotFoundError, match="Missing Phase 2 block feature table"):
        build_phase9_proxy_validation_report(tmp_path)


def test_phase9_unusable_suitability_column_raises(tmp_path):
    from paper11_geofm.proxy_validation import build_phase9_proxy_validation_report

    phase2_dir = tmp_path / "phase2"
    phase2_dir.mkdir()
    (phase2_dir / "block_geofm_features.csv").write_text(
        "block_id,suitability_proxy,stable_farmland_label\n"
        "b0,not_numeric,1\n"
        "b1,,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="numeric suitability_proxy"):
        build_phase9_proxy_validation_report(phase2_dir)


def test_phase9_proxy_validation_report_is_written(tmp_path):
    from paper11_geofm.proxy_validation import (
        build_phase9_proxy_validation_report,
        write_phase9_proxy_validation_report,
    )

    phase2_dir = _run_phase2_fixture(tmp_path / "phase2")
    report = build_phase9_proxy_validation_report(phase2_dir)

    report_path = write_phase9_proxy_validation_report(report, tmp_path / "phase9")

    assert report_path.name == "phase9_proxy_validation_report.json"
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["phase"] == "phase9_proxy_validation_report"
    assert written["n_blocks"] == 4
    assert written["claim_boundary"] == report["claim_boundary"]


def test_phase9_cli_writes_report_and_prints_summary(tmp_path, capsys):
    phase2_dir = _run_phase2_fixture(tmp_path / "phase2")
    output_dir = tmp_path / "phase9"
    runner_path = (
        ROOT
        / "experiments"
        / "phase9_proxy_validation"
        / "run_phase9_proxy_validation.py"
    )
    spec = importlib.util.spec_from_file_location("phase9_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--phase2-output-dir",
            str(phase2_dir),
            "--output-dir",
            str(output_dir),
            "--label-columns",
            "stable_farmland_label,high_standard_farmland_label,missing_label",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "Report:" in stdout
    assert "Blocks: 4" in stdout
    assert (
        "Available labels: stable_farmland_label,high_standard_farmland_label"
        in stdout
    )
    assert "stable_farmland_label rank_auc:" in stdout
    assert "missing_label: label_unavailable" in stdout
    assert "Claim boundary: Phase 9 is a weak-label proxy-validation report" in stdout
    assert (output_dir / "phase9_proxy_validation_report.json").exists()
