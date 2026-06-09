import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id):
    row = {"block_id": block_id, "suitability_proxy": 0.75}
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


def _write_ready_phase2_outputs(output_dir, block_ids=None):
    from paper11_geofm.artifacts import write_phase2_artifacts

    if block_ids is None:
        block_ids = ["sample_block_00", "sample_block_01"]

    return write_phase2_artifacts(
        [_complete_phase2_feature_row(block_id) for block_id in block_ids],
        output_dir,
        _phase2_test_summary(),
    )


def test_phase4_env_wraps_b3_variant_as_gym_contract(tmp_path):
    from paper11_geofm.drl_inputs import load_variant_input
    from paper11_geofm.drl_smoke_env import (
        PHASE4_CLAIM_BOUNDARY,
        Phase4InputContractEnv,
    )

    _write_ready_phase2_outputs(tmp_path)
    loaded = load_variant_input(tmp_path, "B3")

    env = Phase4InputContractEnv(loaded)

    assert env.variant_id == "B3"
    assert env.n_blocks == 2
    assert env.n_features == 82
    assert env.action_space.n == 2
    assert env.observation_space.shape == (167,)

    obs, info = env.reset(seed=123)

    assert obs.dtype == np.float32
    assert obs.shape == (167,)
    np.testing.assert_allclose(obs[:164], loaded.state_matrix.reshape(-1))
    np.testing.assert_allclose(obs[-3:], np.array([1.0, 0.0, 1.0], dtype=np.float32))
    assert info["variant_id"] == "B3"
    assert info["n_blocks"] == 2
    assert info["n_features"] == 82
    assert info["reward_mode"] == "base_plus_suitability_reward"
    assert info["claim_boundary"] == PHASE4_CLAIM_BOUNDARY
    assert env.action_masks().tolist() == [True, True]

    next_obs, reward, terminated, truncated, step_info = env.step(0)

    assert next_obs.dtype == np.float32
    assert next_obs.shape == (167,)
    np.testing.assert_allclose(
        next_obs[-3:], np.array([0.5, 0.5, 0.5], dtype=np.float32)
    )
    assert reward == 0.75
    assert terminated is False
    assert truncated is False
    assert step_info["action"] == 0
    assert step_info["selected_block_id"] == "sample_block_00"
    assert step_info["step"] == 1
    assert step_info["valid_actions"] == 1
    assert step_info["claim_boundary"] == PHASE4_CLAIM_BOUNDARY
    assert env.action_masks().tolist() == [False, True]


def test_phase4_env_returns_zero_contract_reward_for_base_reward_variant(tmp_path):
    from paper11_geofm.drl_smoke_env import make_phase4_smoke_env

    _write_ready_phase2_outputs(tmp_path)
    env = make_phase4_smoke_env(tmp_path, "B1")
    env.reset()

    _, reward, _, _, info = env.step(0)

    assert reward == 0.0
    assert info["reward_mode"] == "base_planning_reward"


def test_phase4_env_rejects_invalid_or_repeated_actions(tmp_path):
    from paper11_geofm.drl_smoke_env import make_phase4_smoke_env

    _write_ready_phase2_outputs(tmp_path)
    env = make_phase4_smoke_env(tmp_path, "B3")
    env.reset()

    with pytest.raises(ValueError, match="out of range"):
        env.step(9)

    env.step(0)
    with pytest.raises(ValueError, match="already selected"):
        env.step(0)


def test_run_phase4_smoke_cli_prints_one_step_summary(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "run_phase4_smoke",
        ROOT / "experiments" / "phase4_drl_smoke_env" / "run_phase4_smoke.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    _write_ready_phase2_outputs(
        tmp_path,
        [
            "sample_block_00",
            "sample_block_01",
            "sample_block_02",
            "sample_block_03",
        ],
    )

    exit_code = module.main(
        [
            "--phase2-output-dir",
            str(tmp_path),
            "--variant",
            "B3",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Variant: B3" in stdout
    assert "Observation shape: 331" in stdout
    assert "Action space: Discrete(4)" in stdout
    assert "Initial valid actions: 4" in stdout
    assert "Selected block: sample_block_00" in stdout
    assert "Step reward: 0.750000" in stdout
    assert "Claim boundary: Phase 4 is a DRL input-contract smoke environment" in stdout
