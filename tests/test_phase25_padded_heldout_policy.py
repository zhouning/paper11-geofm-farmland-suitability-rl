import csv
import faulthandler
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.filterwarnings("ignore:XPU device count is zero!:UserWarning")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id, slope_mean, farmland, suitability=0.75):
    row = {"block_id": block_id, "suitability_proxy": suitability}
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(dim) / 100.0
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = 0.0
    row.update(
        {
            "explicit_feature_00": 2.0,
            "explicit_feature_01": float(slope_mean),
            "explicit_feature_02": float(slope_mean) + 5.0,
            "explicit_feature_04": float(farmland),
            "explicit_feature_07": 0.0,
            "explicit_feature_09": 0.0,
            "explicit_feature_10": 0.0,
            "explicit_feature_13": 1.0 if slope_mean <= 15.0 else 0.0,
            "explicit_feature_16": float(farmland),
        }
    )
    return row


def _write_ready_phase2_outputs(output_dir: Path):
    from paper11_geofm.artifacts import write_phase2_artifacts

    return write_phase2_artifacts(
        [
            _complete_phase2_feature_row("b1", slope_mean=8.0, farmland=1.0),
            _complete_phase2_feature_row("b2", slope_mean=30.0, farmland=0.0),
            _complete_phase2_feature_row("b3", slope_mean=12.0, farmland=1.0),
            _complete_phase2_feature_row("b4", slope_mean=25.0, farmland=0.0),
            _complete_phase2_feature_row("b5", slope_mean=6.0, farmland=1.0),
            _complete_phase2_feature_row("b6", slope_mean=22.0, farmland=0.0),
        ],
        output_dir,
        {
            "metadata_source": "test",
            "base_year_requested": 2020,
            "base_year_used": 2020,
            "years": [2020],
            "grid_shape": [2, 3],
            "embedding_dim": 64,
            "mapping_mode": "test",
        },
    )


def _write_tile_index(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tile_id", "tile_row", "tile_col", "n_blocks", "block_ids"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "tile_id": "tile_r000_c000",
                "tile_row": 0,
                "tile_col": 0,
                "n_blocks": 1,
                "block_ids": "b6",
            }
        )
        writer.writerow(
            {
                "tile_id": "tile_r000_c001",
                "tile_row": 0,
                "tile_col": 1,
                "n_blocks": 3,
                "block_ids": "b1;b3;b5",
            }
        )
        writer.writerow(
            {
                "tile_id": "tile_r000_c002",
                "tile_row": 0,
                "tile_col": 2,
                "n_blocks": 2,
                "block_ids": "b2;b4",
            }
        )
    return path


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


def test_phase25_padded_env_uses_fixed_shape_and_masks_padded_rows(tmp_path):
    from paper11_geofm.padded_heldout_policy import Phase25PaddedTileEnv
    from paper11_geofm.tiled_inputs import load_tiled_variant_input

    _write_ready_phase2_outputs(tmp_path / "phase2")
    tile_index = _write_tile_index(tmp_path / "phase13_tile_index.csv")
    tiled = load_tiled_variant_input(
        tmp_path / "phase2",
        tile_index,
        "tile_r000_c002",
        variant_id="B0",
    )

    env = Phase25PaddedTileEnv(tiled, max_blocks=3, max_steps=2)
    obs, info = env.reset(seed=0)

    assert info["variant_id"] == "B0"
    assert info["tile_id"] == "tile_r000_c002"
    assert info["n_blocks"] == 2
    assert info["max_blocks"] == 3
    assert obs.shape == (3 * 17 + 3 + 3 + 5,)
    assert env.observation_space.shape == obs.shape
    assert env.action_space.n == 3
    assert env.action_masks().tolist() == [True, True, False]

    state_part = obs[: 3 * 17].reshape(3, 17)
    selected_mask = obs[3 * 17 : 3 * 17 + 3]
    valid_block_mask = obs[3 * 17 + 3 : 3 * 17 + 6]
    np.testing.assert_array_equal(state_part[2], np.zeros_like(state_part[2]))
    np.testing.assert_array_equal(selected_mask, np.array([0, 0, 0]))
    np.testing.assert_array_equal(valid_block_mask, np.array([1, 1, 0]))


def test_phase25_padded_env_rejects_padded_and_repeated_actions(tmp_path):
    from paper11_geofm.padded_heldout_policy import Phase25PaddedTileEnv
    from paper11_geofm.tiled_inputs import load_tiled_variant_input

    _write_ready_phase2_outputs(tmp_path / "phase2")
    tile_index = _write_tile_index(tmp_path / "phase13_tile_index.csv")
    tiled = load_tiled_variant_input(
        tmp_path / "phase2",
        tile_index,
        "tile_r000_c002",
        variant_id="B0",
    )

    env = Phase25PaddedTileEnv(tiled, max_blocks=3, max_steps=2)
    obs, info = env.reset(seed=0)

    with pytest.raises(ValueError, match="padded action"):
        env.step(2)

    next_obs, reward, terminated, truncated, next_info = env.step(0)
    float(reward)
    assert terminated is False
    assert truncated is False
    assert next_info["selected_block_id"] == "b2"
    assert next_info["action_valid"] is True
    assert env.action_masks().tolist() == [False, True, False]
    selected_mask_start = 3 * 17
    selected_mask = next_obs[selected_mask_start : selected_mask_start + 3]
    np.testing.assert_allclose(selected_mask, [1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="already selected"):
        env.step(0)
