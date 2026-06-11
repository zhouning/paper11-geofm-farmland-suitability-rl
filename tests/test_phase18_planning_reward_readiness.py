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
    phase2_dir = tmp_path / "phase2"
    variants = {
        "B0": {
            "ready": True,
            "missing": [],
            "row_count": 64984,
            "reward": "base_planning_reward",
            "feature_table": "variant_B0_features.csv",
            "state_groups": ["explicit_planning_features"],
            "required_columns": ["explicit_feature_00"],
        },
        "B1": {
            "ready": True,
            "missing": [],
            "row_count": 64984,
            "reward": "base_planning_reward",
            "feature_table": "variant_B1_features.csv",
            "state_groups": ["explicit_planning_features", "geofm_embedding"],
            "required_columns": ["explicit_feature_00", "embedding_mean_00"],
        },
        "B2": {
            "ready": True,
            "missing": [],
            "row_count": 64984,
            "reward": "base_plus_suitability_reward",
            "feature_table": "variant_B2_features.csv",
            "state_groups": ["explicit_planning_features", "suitability_proxy"],
            "required_columns": ["explicit_feature_00", "suitability_proxy"],
        },
        "B3": {
            "ready": True,
            "missing": [],
            "row_count": 64984,
            "reward": "base_plus_suitability_reward",
            "feature_table": "variant_B3_features.csv",
            "state_groups": [
                "explicit_planning_features",
                "geofm_embedding",
                "suitability_proxy",
            ],
            "required_columns": [
                "explicit_feature_00",
                "embedding_mean_00",
                "suitability_proxy",
            ],
        },
    }
    _write_json(phase2_dir / "experiment_variants.json", {"variants": variants})
    phase10 = _write_json(
        tmp_path / "phase10" / "phase10_reward_readiness_gate.json",
        {
            "phase": "phase10_reward_readiness_gate",
            "status": "not_ready_for_suitability_reward",
            "recommendation": "do_not_enable_suitability_reward",
            "passing_label_count": 2,
            "failing_label_count": 1,
            "insufficient_label_count": 0,
            "labels": {
                "current_farmland_label": {"passes_gate": True},
                "low_slope_farmland_label": {"passes_gate": False},
                "farmland_or_orchard_label": {"passes_gate": True},
            },
        },
    )
    phase12 = _write_json(
        tmp_path / "phase12" / "phase12_real_dltb_scale_audit.json",
        {
            "phase": "phase12_real_dltb_scale_audit",
            "n_blocks": 64984,
            "max_observation_dimension": 5328691,
            "real_feature_tables_ready": True,
            "representation_only_smoke_allowed": True,
            "suitability_reward_allowed": False,
            "flat_full_scale_training_ready": False,
            "requires_tiled_or_hierarchical_env": True,
        },
    )
    phase17 = _write_json(
        tmp_path
        / "phase17"
        / "phase17_tiled_maskableppo_readiness.json",
        {
            "phase": "phase17_tiled_maskableppo_readiness",
            "tile_id": "tile_r003_c003",
            "variant_id": "B1",
            "n_blocks": 2234,
            "n_features": 81,
            "observation_shape": 180957,
            "readiness_status": "passed_tiled_maskableppo_smoke",
            "masking_supported": True,
            "predicted_action_valid": True,
            "reward_mode": "base_planning_reward",
        },
    )
    return {
        "phase2_dir": phase2_dir,
        "phase10": phase10,
        "phase12": phase12,
        "phase17": phase17,
    }


def test_phase18_keeps_performance_blocked_after_base_reward_implementation(tmp_path):
    from paper11_geofm.planning_reward_readiness import (
        PHASE18_CLAIM_BOUNDARY,
        build_phase18_planning_reward_readiness,
    )

    paths = _artifact_fixture(tmp_path)
    report = build_phase18_planning_reward_readiness(
        paths["phase2_dir"],
        paths["phase10"],
        paths["phase12"],
        phase17_readiness_path=paths["phase17"],
    )

    assert report["phase"] == "phase18_planning_reward_readiness"
    assert report["base_planning_reward_implemented"] is True
    assert report["base_reward_modes"] == {
        "B0": "base_planning_reward",
        "B1": "base_planning_reward",
    }
    assert "bounded weighted score" in report["base_planning_reward_evidence"]
    assert report["suitability_reward_allowed"] is False
    assert report["flat_full_scale_training_ready"] is False
    assert report["tiled_maskableppo_api_ready"] is True
    assert report["performance_experiment_ready"] is False
    assert "base_planning_reward_not_implemented" not in report["blocked_reasons"]
    assert "suitability_reward_not_allowed" in report["blocked_reasons"]
    assert "flat_full_scale_training_not_ready" in report["blocked_reasons"]
    assert (
        report["recommended_next_step"]
        == "resolve_suitability_reward_gate_before_suitability_reward_experiments"
    )
    assert report["claim_boundary"] == PHASE18_CLAIM_BOUNDARY


def test_phase18_marks_tiled_api_not_supplied_when_phase17_missing(tmp_path):
    from paper11_geofm.planning_reward_readiness import (
        build_phase18_planning_reward_readiness,
    )

    paths = _artifact_fixture(tmp_path)
    report = build_phase18_planning_reward_readiness(
        paths["phase2_dir"],
        paths["phase10"],
        paths["phase12"],
    )

    assert report["phase17_readiness"] is None
    assert report["tiled_maskableppo_status"] == "not_supplied"
    assert report["tiled_maskableppo_api_ready"] is False
    assert "tiled_maskableppo_api_not_ready" in report["blocked_reasons"]


def test_phase18_writer_outputs_json_report(tmp_path):
    from paper11_geofm.planning_reward_readiness import (
        build_phase18_planning_reward_readiness,
        write_phase18_planning_reward_readiness,
    )

    paths = _artifact_fixture(tmp_path)
    report = build_phase18_planning_reward_readiness(
        paths["phase2_dir"],
        paths["phase10"],
        paths["phase12"],
        phase17_readiness_path=paths["phase17"],
    )

    output_path = write_phase18_planning_reward_readiness(
        report,
        tmp_path / "outputs",
    )

    assert output_path.name == "phase18_planning_reward_readiness.json"
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["phase"] == "phase18_planning_reward_readiness"
    assert written["performance_experiment_ready"] is False


def test_phase18_cli_writes_report_and_prints_summary(tmp_path, capsys):
    paths = _artifact_fixture(tmp_path)
    runner_path = (
        ROOT
        / "experiments"
        / "phase18_planning_reward_readiness"
        / "run_phase18_planning_reward_readiness.py"
    )
    spec = importlib.util.spec_from_file_location("phase18_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--phase2-output-dir",
            str(paths["phase2_dir"]),
            "--phase10-gate",
            str(paths["phase10"]),
            "--phase12-audit",
            str(paths["phase12"]),
            "--phase17-readiness",
            str(paths["phase17"]),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Base planning reward implemented: True" in stdout
    assert "Suitability reward allowed: False" in stdout
    assert "Flat full-scale training ready: False" in stdout
    assert "Tiled MaskablePPO API ready: True" in stdout
    assert "Performance experiment ready: False" in stdout
    assert (
        "Recommendation: "
        "resolve_suitability_reward_gate_before_suitability_reward_experiments"
    ) in stdout
    assert "phase18_planning_reward_readiness.json" in stdout
    assert "Claim boundary: Phase 18 is a planning-reward readiness audit" in stdout
