import importlib.util
import faulthandler
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


pytestmark = pytest.mark.filterwarnings("ignore:XPU device count is zero!:UserWarning")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id, suitability):
    row = {"block_id": block_id, "suitability_proxy": suitability}
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(dim)
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = float(idx)
    return row


def _phase2_test_summary():
    return {
        "metadata_source": "test",
        "base_year_requested": 2020,
        "base_year_used": 2020,
        "years": [2020],
        "grid_shape": [2, 2],
        "embedding_dim": 64,
        "mapping_mode": "test",
    }


def _write_ready_phase2_outputs(output_dir):
    from paper11_geofm.artifacts import write_phase2_artifacts

    rows = [
        _complete_phase2_feature_row("sample_block_00", 0.25),
        _complete_phase2_feature_row("sample_block_01", 0.50),
        _complete_phase2_feature_row("sample_block_02", 0.75),
        _complete_phase2_feature_row("sample_block_03", 1.00),
    ]
    return write_phase2_artifacts(rows, output_dir, _phase2_test_summary())


@contextmanager
def _torch_windows_faulthandler_guard():
    was_enabled = faulthandler.is_enabled()
    if was_enabled:
        faulthandler.disable()
    try:
        yield
    finally:
        if was_enabled:
            faulthandler.enable(file=sys.__stderr__)


def _require_maskableppo_dependencies():
    with _torch_windows_faulthandler_guard():
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("sb3_contrib")


def test_phase7_maskableppo_smoke_runs_tiny_learn_and_predict(tmp_path):
    _require_maskableppo_dependencies()
    from paper11_geofm.maskableppo_smoke import (
        PHASE7_CLAIM_BOUNDARY,
        run_phase7_maskableppo_smoke,
    )

    _write_ready_phase2_outputs(tmp_path)

    with _torch_windows_faulthandler_guard():
        summary = run_phase7_maskableppo_smoke(
            tmp_path,
            variant_id="B3",
            total_timesteps=8,
            seed=0,
        )

    assert summary["phase"] == "phase7_maskableppo_smoke"
    assert summary["variant_id"] == "B3"
    assert summary["seed"] == 0
    assert summary["n_blocks"] == 4
    assert summary["n_features"] == 82
    assert summary["observation_shape"] == 331
    assert summary["action_space_n"] == 4
    assert summary["reward_mode"] == "base_plus_suitability_reward"
    assert summary["masking_supported"] is True
    assert summary["initial_valid_actions"] == 4
    assert summary["learn_timesteps"] == 8
    assert summary["device"] == "cpu"
    assert summary["predicted_action_valid"] is True
    assert 0 <= summary["predicted_action"] < 4
    assert str(summary["selected_block_id"]).startswith("sample_block_")
    assert summary["claim_boundary"] == PHASE7_CLAIM_BOUNDARY

    dependencies = summary["dependencies"]
    assert dependencies["stable_baselines3"]["available"] is True
    assert isinstance(dependencies["stable_baselines3"]["version"], str)
    assert dependencies["stable_baselines3"]["version"]
    assert dependencies["sb3_contrib"]["available"] is True
    assert isinstance(dependencies["sb3_contrib"]["version"], str)
    assert dependencies["sb3_contrib"]["version"]


def test_phase7_maskableppo_artifact_is_written(tmp_path):
    from paper11_geofm.maskableppo_smoke import (
        PHASE7_CLAIM_BOUNDARY,
        write_phase7_maskableppo_artifact,
    )

    summary = {
        "phase": "phase7_maskableppo_smoke",
        "variant_id": "B3",
        "masking_supported": True,
        "predicted_action_valid": True,
        "claim_boundary": PHASE7_CLAIM_BOUNDARY,
    }

    artifact_path = write_phase7_maskableppo_artifact(
        summary,
        tmp_path / "nested" / "phase7",
    )

    assert artifact_path.name == "phase7_maskableppo_smoke.json"
    assert artifact_path.exists()
    saved = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert saved == summary


def test_phase7_maskableppo_cli_prints_smoke_summary_and_artifact(
    tmp_path,
    capsys,
):
    _require_maskableppo_dependencies()
    spec = importlib.util.spec_from_file_location(
        "run_phase7_maskableppo_smoke",
        ROOT
        / "experiments"
        / "phase7_maskableppo_smoke"
        / "run_phase7_maskableppo_smoke.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase7"
    _write_ready_phase2_outputs(phase2_dir)

    with _torch_windows_faulthandler_guard():
        exit_code = module.main(
            [
                "--phase2-output-dir",
                str(phase2_dir),
                "--output-dir",
                str(output_dir),
                "--variant",
                "B3",
                "--total-timesteps",
                "8",
                "--seed",
                "0",
            ]
        )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Variant: B3" in stdout
    assert "Observation shape: 331" in stdout
    assert "Action space: Discrete(4)" in stdout
    assert "Masking supported: True" in stdout
    assert "Predicted action valid: True" in stdout
    assert "Artifact:" in stdout
    assert (
        "Claim boundary: Phase 7 is a MaskablePPO compatibility smoke check"
        in stdout
    )
    assert (output_dir / "phase7_maskableppo_smoke.json").exists()
