import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _complete_phase2_feature_row(block_id: str) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "block_id": block_id,
        "suitability_proxy": 0.75,
    }
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(dim)
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = 0.0
    row.update(
        {
            "explicit_feature_00": 2.5,
            "explicit_feature_01": 10.0,
            "explicit_feature_02": 28.0,
            "explicit_feature_04": 1.0,
            "explicit_feature_07": 0.0,
            "explicit_feature_09": 0.0,
            "explicit_feature_10": 0.0,
            "explicit_feature_13": 1.0,
            "explicit_feature_16": 1.0,
        }
    )
    return row


def _phase2_test_summary() -> dict[str, object]:
    return {
        "metadata_source": "test",
        "base_year_requested": 2020,
        "base_year_used": 2020,
        "years": [2020],
        "grid_shape": [2, 2],
        "embedding_dim": 64,
        "mapping_mode": "test",
    }


def _write_ready_phase2_outputs(output_dir: Path) -> None:
    from paper11_geofm.artifacts import write_phase2_artifacts

    write_phase2_artifacts(
        [
            _complete_phase2_feature_row("sample_block_00"),
            _complete_phase2_feature_row("sample_block_01"),
        ],
        output_dir,
        _phase2_test_summary(),
    )


def _phase18_artifact_fixture(tmp_path: Path) -> dict[str, Path]:
    phase2_dir = tmp_path / "phase2"
    variants = {
        "B0": {
            "ready": True,
            "missing": [],
            "row_count": 2,
            "reward": "base_planning_reward",
            "feature_table": "variant_B0_features.csv",
            "state_groups": ["explicit_planning_features"],
            "required_columns": [f"explicit_feature_{idx:02d}" for idx in range(17)],
        },
        "B1": {
            "ready": True,
            "missing": [],
            "row_count": 2,
            "reward": "base_planning_reward",
            "feature_table": "variant_B1_features.csv",
            "state_groups": ["explicit_planning_features", "geofm_embedding"],
            "required_columns": [
                *[f"explicit_feature_{idx:02d}" for idx in range(17)],
                *[f"embedding_mean_{idx:02d}" for idx in range(64)],
            ],
        },
    }
    _write_json(phase2_dir / "experiment_variants.json", {"variants": variants})
    phase10 = _write_json(
        tmp_path / "phase10" / "phase10_reward_readiness_gate.json",
        {
            "status": "not_ready_for_suitability_reward",
            "recommendation": "do_not_enable_suitability_reward",
        },
    )
    phase12 = _write_json(
        tmp_path / "phase12" / "phase12_real_dltb_scale_audit.json",
        {
            "n_blocks": 2,
            "max_observation_dimension": 165,
            "real_feature_tables_ready": True,
            "representation_only_smoke_allowed": True,
            "suitability_reward_allowed": False,
            "flat_full_scale_training_ready": False,
            "requires_tiled_or_hierarchical_env": True,
        },
    )
    phase17 = _write_json(
        tmp_path / "phase17" / "phase17_tiled_maskableppo_readiness.json",
        {
            "readiness_status": "passed_tiled_maskableppo_smoke",
            "masking_supported": True,
            "predicted_action_valid": True,
        },
    )
    return {
        "phase2_dir": phase2_dir,
        "phase10": phase10,
        "phase12": phase12,
        "phase17": phase17,
    }


def test_base_planning_reward_matches_weighted_formula():
    from paper11_geofm.planning_reward import (
        BASE_PLANNING_REWARD_REQUIRED_COLUMNS,
        compute_base_planning_reward,
    )

    row = {column: 0.0 for column in BASE_PLANNING_REWARD_REQUIRED_COLUMNS}
    row.update(
        {
            "explicit_feature_00": 2.5,
            "explicit_feature_01": 10.0,
            "explicit_feature_02": 28.0,
            "explicit_feature_04": 1.0,
            "explicit_feature_07": 0.0,
            "explicit_feature_09": 0.0,
            "explicit_feature_10": 0.0,
            "explicit_feature_13": 1.0,
            "explicit_feature_16": 1.0,
        }
    )

    reward = compute_base_planning_reward(row)

    expected = 0.35 + 0.20 + 0.10 + 0.05 - 0.06 - 0.04
    assert reward == round(expected, 10)


def test_base_planning_reward_rejects_missing_explicit_feature():
    from paper11_geofm.planning_reward import compute_base_planning_reward

    with pytest.raises(ValueError, match="explicit_feature_16"):
        compute_base_planning_reward({"explicit_feature_00": 1.0})


def test_base_planning_reward_clips_area_and_slope_terms():
    from paper11_geofm.planning_reward import (
        BASE_PLANNING_REWARD_REQUIRED_COLUMNS,
        compute_base_planning_reward,
    )

    row = {column: 0.0 for column in BASE_PLANNING_REWARD_REQUIRED_COLUMNS}
    row.update(
        {
            "explicit_feature_00": 50.0,
            "explicit_feature_01": 250.0,
            "explicit_feature_02": 350.0,
        }
    )

    assert compute_base_planning_reward(row) == -0.10


def test_phase4_env_uses_base_planning_reward_for_b1(tmp_path):
    from paper11_geofm.drl_smoke_env import make_phase4_smoke_env

    _write_ready_phase2_outputs(tmp_path)
    env = make_phase4_smoke_env(tmp_path, "B1")
    env.reset()

    _, reward, _, _, info = env.step(0)

    assert reward == 0.6
    assert info["reward_mode"] == "base_planning_reward"


def test_phase4_env_adds_base_reward_to_suitability_proxy_for_b3(tmp_path):
    from paper11_geofm.drl_smoke_env import make_phase4_smoke_env

    _write_ready_phase2_outputs(tmp_path)
    env = make_phase4_smoke_env(tmp_path, "B3")
    env.reset()

    _, reward, _, _, _ = env.step(0)

    assert reward == 1.35


def test_phase4_base_reward_requires_explicit_feature_columns():
    from paper11_geofm.drl_inputs import VariantInput
    from paper11_geofm.drl_smoke_env import Phase4InputContractEnv

    variant_input = VariantInput(
        variant_id="B0",
        block_ids=("sample_block_00",),
        feature_columns=("explicit_feature_00",),
        state_matrix=np.asarray([[1.0]], dtype=np.float32),
        reward_mode="base_planning_reward",
        state_groups=("explicit_planning_features",),
        source_table=Path("variant_B0_features.csv"),
    )
    env = Phase4InputContractEnv(variant_input)
    env.reset()

    with pytest.raises(ValueError, match="explicit_feature_16"):
        env.step(0)


def test_phase18_reads_base_reward_metadata_after_phase19(tmp_path):
    from paper11_geofm.planning_reward_readiness import (
        build_phase18_planning_reward_readiness,
    )

    paths = _phase18_artifact_fixture(tmp_path)

    report = build_phase18_planning_reward_readiness(
        paths["phase2_dir"],
        paths["phase10"],
        paths["phase12"],
        phase17_readiness_path=paths["phase17"],
    )

    assert report["base_planning_reward_implemented"] is True
    assert "bounded weighted score" in report["base_planning_reward_evidence"]
    assert "base_planning_reward_not_implemented" not in report["blocked_reasons"]
    assert report["performance_experiment_ready"] is False
