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
    spec = importlib.util.spec_from_file_location("phase2_runner_phase10", runner_path)
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


def _write_phase9_fixture_report(tmp_path: Path) -> Path:
    from paper11_geofm.proxy_validation import (
        build_phase9_proxy_validation_report,
        write_phase9_proxy_validation_report,
    )

    phase2_dir = _run_phase2_fixture(tmp_path / "phase2")
    report = build_phase9_proxy_validation_report(phase2_dir)
    return write_phase9_proxy_validation_report(report, tmp_path / "phase9")


def test_phase10_marks_fixture_not_ready_for_suitability_reward(tmp_path):
    from paper11_geofm.reward_readiness import (
        PHASE10_CLAIM_BOUNDARY,
        build_phase10_reward_readiness_gate,
    )

    phase9_report_path = _write_phase9_fixture_report(tmp_path)

    gate = build_phase10_reward_readiness_gate(phase9_report_path)

    assert gate["phase"] == "phase10_reward_readiness_gate"
    assert gate["phase9_report"] == str(phase9_report_path)
    assert gate["required_labels"] == [
        "stable_farmland_label",
        "high_standard_farmland_label",
    ]
    assert gate["status"] == "not_ready_for_suitability_reward"
    assert gate["recommendation"] == "do_not_enable_suitability_reward"
    assert gate["passing_label_count"] == 0
    assert gate["failing_label_count"] == 2
    assert gate["insufficient_label_count"] == 0
    assert gate["labels"]["stable_farmland_label"]["passes_gate"] is False
    assert gate["labels"]["stable_farmland_label"]["interpretation"] == (
        "negative_or_no_alignment"
    )
    assert gate["labels"]["high_standard_farmland_label"]["passes_gate"] is False
    assert "failed suitability proxy alignment gate" in gate["reasons"][0]
    assert gate["claim_boundary"] == PHASE10_CLAIM_BOUNDARY


def _write_report(path: Path, labels: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "phase9_proxy_validation_report",
        "labels": labels,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _positive_label() -> dict[str, object]:
    return {
        "validation_available": True,
        "interpretation": "positive_alignment",
        "rank_auc": 0.75,
        "mean_difference": 0.2,
        "positive_count": 3,
        "negative_count": 3,
    }


def test_phase10_marks_positive_synthetic_report_ready(tmp_path):
    from paper11_geofm.reward_readiness import build_phase10_reward_readiness_gate

    report_path = _write_report(
        tmp_path / "phase9" / "phase9_proxy_validation_report.json",
        {
            "stable_farmland_label": _positive_label(),
            "high_standard_farmland_label": _positive_label(),
        },
    )

    gate = build_phase10_reward_readiness_gate(report_path)

    assert gate["status"] == "ready_for_suitability_reward_smoke"
    assert gate["recommendation"] == "allow_bounded_suitability_reward_smoke"
    assert gate["passing_label_count"] == 2
    assert gate["failing_label_count"] == 0
    assert gate["insufficient_label_count"] == 0


def test_phase10_marks_missing_and_one_class_labels_insufficient(tmp_path):
    from paper11_geofm.reward_readiness import build_phase10_reward_readiness_gate

    report_path = _write_report(
        tmp_path / "phase9" / "phase9_proxy_validation_report.json",
        {
            "stable_farmland_label": {
                "validation_available": False,
                "interpretation": "insufficient_label_variation",
                "rank_auc": None,
                "mean_difference": None,
                "positive_count": 4,
                "negative_count": 0,
            }
        },
    )

    gate = build_phase10_reward_readiness_gate(report_path)

    assert gate["status"] == "insufficient_evidence"
    assert gate["recommendation"] == "collect_or_rebuild_weak_labels_before_reward_use"
    assert gate["insufficient_label_count"] == 2
    assert gate["labels"]["high_standard_farmland_label"]["available"] is False


def test_phase10_invalid_phase9_report_raises(tmp_path):
    from paper11_geofm.reward_readiness import build_phase10_reward_readiness_gate

    report_path = tmp_path / "bad_report.json"
    report_path.write_text(
        json.dumps({"phase": "wrong", "labels": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Phase 9 proxy-validation report"):
        build_phase10_reward_readiness_gate(report_path)


def test_phase10_reward_readiness_gate_is_written(tmp_path):
    from paper11_geofm.reward_readiness import (
        build_phase10_reward_readiness_gate,
        write_phase10_reward_readiness_gate,
    )

    report_path = _write_report(
        tmp_path / "phase9" / "phase9_proxy_validation_report.json",
        {
            "stable_farmland_label": _positive_label(),
            "high_standard_farmland_label": _positive_label(),
        },
    )
    gate = build_phase10_reward_readiness_gate(report_path)

    gate_path = write_phase10_reward_readiness_gate(gate, tmp_path / "phase10")

    assert gate_path.name == "phase10_reward_readiness_gate.json"
    written = json.loads(gate_path.read_text(encoding="utf-8"))
    assert written["phase"] == "phase10_reward_readiness_gate"
    assert written["status"] == "ready_for_suitability_reward_smoke"


def test_phase10_cli_writes_gate_and_prints_summary(tmp_path, capsys):
    phase9_report_path = _write_phase9_fixture_report(tmp_path)
    output_dir = tmp_path / "phase10"
    runner_path = (
        ROOT
        / "experiments"
        / "phase10_reward_readiness"
        / "run_phase10_reward_readiness.py"
    )
    spec = importlib.util.spec_from_file_location("phase10_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--phase9-report",
            str(phase9_report_path),
            "--output-dir",
            str(output_dir),
            "--required-labels",
            "stable_farmland_label,high_standard_farmland_label",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "Gate:" in stdout
    assert "Status: not_ready_for_suitability_reward" in stdout
    assert "Recommendation: do_not_enable_suitability_reward" in stdout
    assert "stable_farmland_label:" in stdout
    assert "Claim boundary: Phase 10 is a reward-readiness gate" in stdout
    assert (output_dir / "phase10_reward_readiness_gate.json").exists()
