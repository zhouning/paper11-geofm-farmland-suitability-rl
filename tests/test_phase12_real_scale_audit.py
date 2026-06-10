import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _artifact_fixture(tmp_path: Path) -> dict[str, Path]:
    phase11 = _write_json(
        tmp_path / "phase11" / "phase11_bishan_dltb_adapter_summary.json",
        {
            "phase": "phase11_bishan_dltb_real_adapter",
            "rows_exported": 10,
            "rows_read_in_bbox": 12,
            "category_counts": {"Farmland": 4, "Other": 6},
            "label_positive_counts": {
                "current_farmland_label": 4,
                "low_slope_farmland_label": 2,
                "farmland_or_orchard_label": 5,
            },
        },
    )
    phase2_dir = tmp_path / "phase2"
    _write_json(
        phase2_dir / "summary.json",
        {
            "n_blocks": 10,
            "feature_groups_present": [
                "explicit_planning_features",
                "geofm_embedding",
                "suitability_proxy",
            ],
            "feature_readiness": {
                "B0": {"ready": True, "missing": []},
                "B1": {"ready": True, "missing": []},
                "B2": {"ready": True, "missing": []},
                "B3": {"ready": True, "missing": []},
            },
        },
    )
    variants = {
        "B0": {
            "ready": True,
            "missing": [],
            "row_count": 10,
            "required_columns": ["explicit_feature_00", "explicit_feature_01"],
            "reward": "base_planning_reward",
            "feature_table": "variant_B0_features.csv",
            "state_groups": ["explicit_planning_features"],
        },
        "B1": {
            "ready": True,
            "missing": [],
            "row_count": 10,
            "required_columns": [
                "explicit_feature_00",
                "explicit_feature_01",
                "embedding_mean_00",
            ],
            "reward": "base_planning_reward",
            "feature_table": "variant_B1_features.csv",
            "state_groups": ["explicit_planning_features", "geofm_embedding"],
        },
        "B2": {
            "ready": True,
            "missing": [],
            "row_count": 10,
            "required_columns": [
                "explicit_feature_00",
                "explicit_feature_01",
                "suitability_proxy",
            ],
            "reward": "base_plus_suitability_reward",
            "feature_table": "variant_B2_features.csv",
            "state_groups": ["explicit_planning_features", "suitability_proxy"],
        },
        "B3": {
            "ready": True,
            "missing": [],
            "row_count": 10,
            "required_columns": [
                "explicit_feature_00",
                "explicit_feature_01",
                "embedding_mean_00",
                "suitability_proxy",
            ],
            "reward": "base_plus_suitability_reward",
            "feature_table": "variant_B3_features.csv",
            "state_groups": [
                "explicit_planning_features",
                "geofm_embedding",
                "suitability_proxy",
            ],
        },
    }
    _write_json(phase2_dir / "experiment_variants.json", {"variants": variants})
    phase9 = _write_json(
        tmp_path / "phase9" / "phase9_proxy_validation_report.json",
        {
            "n_blocks": 10,
            "labels": {
                "current_farmland_label": {
                    "interpretation": "positive_alignment",
                    "rank_auc": 0.51,
                    "mean_difference": 0.01,
                },
                "low_slope_farmland_label": {
                    "interpretation": "negative_or_no_alignment",
                    "rank_auc": 0.49,
                    "mean_difference": -0.01,
                },
            },
        },
    )
    phase10 = _write_json(
        tmp_path / "phase10" / "phase10_reward_readiness_gate.json",
        {
            "status": "not_ready_for_suitability_reward",
            "recommendation": "do_not_enable_suitability_reward",
            "passing_label_count": 1,
            "failing_label_count": 1,
            "insufficient_label_count": 0,
            "labels": {
                "current_farmland_label": {"passes_gate": True},
                "low_slope_farmland_label": {"passes_gate": False},
            },
        },
    )
    return {
        "phase11": phase11,
        "phase2_dir": phase2_dir,
        "phase9": phase9,
        "phase10": phase10,
    }


def test_phase12_audit_blocks_reward_and_flat_training_when_gate_fails(tmp_path):
    from paper11_geofm.real_scale_audit import (
        PHASE12_CLAIM_BOUNDARY,
        build_phase12_real_scale_audit,
    )

    paths = _artifact_fixture(tmp_path)
    report = build_phase12_real_scale_audit(
        paths["phase11"],
        paths["phase2_dir"],
        paths["phase9"],
        paths["phase10"],
        flat_observation_threshold=20,
    )

    assert report["n_blocks"] == 10
    assert report["real_feature_tables_ready"] is True
    assert report["representation_only_smoke_allowed"] is True
    assert report["suitability_reward_allowed"] is False
    assert report["flat_full_scale_training_ready"] is False
    assert report["requires_tiled_or_hierarchical_env"] is True
    assert report["variants"]["B3"]["observation_dimension"] == 43
    assert report["max_observation_dimension"] == 43
    assert report["phase10"]["status"] == "not_ready_for_suitability_reward"
    assert "keep_suitability_reward_disabled" in report["recommendation"]
    assert report["claim_boundary"] == PHASE12_CLAIM_BOUNDARY


def test_phase12_can_pass_flat_training_gate_when_reward_ready_and_threshold_high(
    tmp_path,
):
    from paper11_geofm.real_scale_audit import build_phase12_real_scale_audit

    paths = _artifact_fixture(tmp_path)
    phase10_payload = json.loads(paths["phase10"].read_text(encoding="utf-8"))
    phase10_payload["status"] = "ready_for_suitability_reward"
    phase10_payload["recommendation"] = "enable_bounded_suitability_reward_smoke"
    paths["phase10"].write_text(json.dumps(phase10_payload), encoding="utf-8")

    report = build_phase12_real_scale_audit(
        paths["phase11"],
        paths["phase2_dir"],
        paths["phase9"],
        paths["phase10"],
        flat_observation_threshold=100,
    )

    assert report["suitability_reward_allowed"] is True
    assert report["requires_tiled_or_hierarchical_env"] is False
    assert report["flat_full_scale_training_ready"] is True


def test_phase12_writer_outputs_json_report(tmp_path):
    from paper11_geofm.real_scale_audit import (
        build_phase12_real_scale_audit,
        write_phase12_real_scale_audit,
    )

    paths = _artifact_fixture(tmp_path)
    report = build_phase12_real_scale_audit(
        paths["phase11"],
        paths["phase2_dir"],
        paths["phase9"],
        paths["phase10"],
        flat_observation_threshold=20,
    )
    output_path = write_phase12_real_scale_audit(report, tmp_path / "outputs")

    assert output_path.name == "phase12_real_dltb_scale_audit.json"
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["phase"] == "phase12_real_dltb_scale_audit"
    assert written["variants"]["B0"]["n_features"] == 2


def test_phase12_rejects_non_positive_threshold(tmp_path):
    from paper11_geofm.real_scale_audit import build_phase12_real_scale_audit

    paths = _artifact_fixture(tmp_path)
    try:
        build_phase12_real_scale_audit(
            paths["phase11"],
            paths["phase2_dir"],
            paths["phase9"],
            paths["phase10"],
            flat_observation_threshold=0,
        )
    except ValueError as exc:
        assert "flat_observation_threshold must be positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_phase12_cli_writes_audit_report(tmp_path, capsys):
    paths = _artifact_fixture(tmp_path)
    runner_path = (
        ROOT
        / "experiments"
        / "phase12_real_scale_audit"
        / "run_phase12_real_scale_audit.py"
    )
    spec = importlib.util.spec_from_file_location("phase12_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--phase11-summary",
            str(paths["phase11"]),
            "--phase2-output-dir",
            str(paths["phase2_dir"]),
            "--phase9-report",
            str(paths["phase9"]),
            "--phase10-gate",
            str(paths["phase10"]),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--flat-observation-threshold",
            "20",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Real feature tables ready: True" in stdout
    assert "Suitability reward allowed: False" in stdout
    assert "Flat full-scale training ready: False" in stdout
    assert "phase12_real_dltb_scale_audit.json" in stdout
