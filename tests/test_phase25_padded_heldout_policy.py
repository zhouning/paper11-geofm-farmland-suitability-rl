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


def test_phase25_contract_selects_largest_train_and_distinct_eval_tiles(tmp_path):
    from paper11_geofm.padded_heldout_policy import (
        PHASE25_CLAIM_BOUNDARY,
        build_phase25_padded_heldout_policy_contract,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    contract = build_phase25_padded_heldout_policy_contract(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variants=("B0", "B1"),
        total_timesteps=8,
        eval_max_steps=2,
        seeds="0,1",
        max_eval_tiles=2,
    )

    assert contract["phase"] == "phase25_padded_heldout_policy"
    assert contract["variants"] == ["B0", "B1"]
    assert contract["train_tile_id"] == "tile_r000_c001"
    assert contract["eval_tile_ids"] == ["tile_r000_c002", "tile_r000_c000"]
    assert contract["eval_tile_ranks"] == {
        "tile_r000_c002": 1,
        "tile_r000_c000": 2,
    }
    assert contract["train_tile_selection"] == "largest"
    assert contract["eval_tile_selection"] == "largest_distinct"
    assert contract["padded_policy_status"] == "enabled_distinct_heldout_tiles"
    assert contract["max_blocks"] == 3
    assert contract["total_timesteps"] == 8
    assert contract["eval_max_steps"] == 2
    assert contract["seeds"] == [0, 1]
    assert contract["claim_boundary"] == PHASE25_CLAIM_BOUNDARY


def test_phase25_contract_rejects_suitability_variants_and_train_eval_overlap(tmp_path):
    from paper11_geofm.padded_heldout_policy import (
        build_phase25_padded_heldout_policy_contract,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    tile_index = _write_tile_index(tmp_path / "phase13_tile_index.csv")
    with pytest.raises(ValueError, match="B0/B1"):
        build_phase25_padded_heldout_policy_contract(
            tmp_path / "phase2",
            tile_index,
            variants=("B3",),
        )

    with pytest.raises(ValueError, match="must be distinct"):
        build_phase25_padded_heldout_policy_contract(
            tmp_path / "phase2",
            tile_index,
            train_tile_id="tile_r000_c001",
            eval_tile_ids=["tile_r000_c001"],
        )


def test_phase25_runs_padded_heldout_policy_training_and_comparison(tmp_path):
    _require_maskableppo_dependencies()
    from paper11_geofm.padded_heldout_policy import (
        PHASE25_CLAIM_BOUNDARY,
        run_phase25_padded_heldout_policy,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with _torch_windows_faulthandler_guard():
        protocol = run_phase25_padded_heldout_policy(
            tmp_path / "phase2",
            _write_tile_index(tmp_path / "phase13_tile_index.csv"),
            variants=("B0", "B1"),
            total_timesteps=8,
            eval_max_steps=2,
            seeds=(0, 1),
            max_eval_tiles=2,
        )

    assert protocol["phase"] == "phase25_padded_heldout_policy"
    assert protocol["train_tile_id"] == "tile_r000_c001"
    assert protocol["eval_tile_ids"] == ["tile_r000_c002", "tile_r000_c000"]
    assert protocol["training_completed"] is True
    assert protocol["all_evaluations_completed"] is True
    assert protocol["summary_count"] == 24
    assert len(protocol["summaries"]) == 24
    assert all(row["max_blocks"] == 3 for row in protocol["summaries"])
    assert all(row["action_space_n"] == 3 for row in protocol["summaries"])
    assert all(row["all_actions_valid"] is True for row in protocol["summaries"])
    assert all(row["invalid_action_count"] == 0 for row in protocol["summaries"])
    assert all(row["claim_boundary"] == PHASE25_CLAIM_BOUNDARY for row in protocol["summaries"])
    assert protocol["comparison"]["learned_policy"]["B1_minus_B0_mean_reward"] is not None
    assert protocol["comparison"]["learned_policy"][
        "heldout_tile_B1_minus_B0_mean_reward"
    ]
    assert protocol["comparison"]["pilot_result_status"] in {
        "B1_improves_B0",
        "B1_matches_B0",
        "B1_underperforms_B0",
    }
    assert protocol["traces"]["trained_policy"]["B0"]["tile_r000_c002"]["0"]
    assert protocol["traces"]["seeded_random"]["B1"]["tile_r000_c000"]["1"]


def test_phase25_writer_outputs_summary_trace_and_comparison(tmp_path):
    from paper11_geofm.padded_heldout_policy import (
        PHASE25_CLAIM_BOUNDARY,
        write_phase25_padded_heldout_policy_artifacts,
    )

    protocol = {
        "phase": "phase25_padded_heldout_policy",
        "summaries": [
            {
                "row_type": "trained_policy",
                "variant_id": "B0",
                "train_tile_id": "tile_r000_c001",
                "eval_tile_id": "tile_r000_c002",
                "eval_tile_rank": 1,
                "seed": 0,
                "max_blocks": 3,
                "eval_max_steps": 2,
                "n_blocks": 3,
                "n_features": 17,
                "observation_shape": 62,
                "action_space_n": 3,
                "episode_steps": 2,
                "terminated": True,
                "truncated": False,
                "total_contract_reward": 1.2,
                "selected_block_ids": ["b2", "b4"],
                "all_actions_valid": True,
                "invalid_action_count": 0,
                "claim_boundary": PHASE25_CLAIM_BOUNDARY,
            }
        ],
        "traces": {"trained_policy": {"B0": {"tile_r000_c002": {"0": []}}}},
        "comparison": {
            "learned_policy": {"B1_minus_B0_mean_reward": None},
            "pilot_result_status": "B1_matches_B0",
            "claim_boundary": PHASE25_CLAIM_BOUNDARY,
        },
        "claim_boundary": PHASE25_CLAIM_BOUNDARY,
    }

    paths = write_phase25_padded_heldout_policy_artifacts(
        protocol,
        tmp_path / "outputs",
    )

    assert paths["summary_csv"].name == "phase25_padded_heldout_policy_summary.csv"
    assert paths["traces_json"].name == "phase25_padded_heldout_policy_traces.json"
    assert paths["comparison_json"].name == "phase25_padded_heldout_policy_comparison.json"
    rows = list(csv.DictReader(paths["summary_csv"].open("r", encoding="utf-8")))
    assert rows[0]["selected_block_ids"] == "b2;b4"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["pilot_result_status"] == "B1_matches_B0"


def test_phase25_cli_writes_outputs_and_prints_summary(tmp_path, capsys):
    _require_maskableppo_dependencies()
    runner_path = (
        ROOT
        / "experiments"
        / "phase25_padded_heldout_policy"
        / "run_phase25_padded_heldout_policy.py"
    )
    spec = importlib.util.spec_from_file_location("phase25_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with _torch_windows_faulthandler_guard():
        exit_code = module.main(
            [
                "--phase2-output-dir",
                str(tmp_path / "phase2"),
                "--tile-index-csv",
                str(_write_tile_index(tmp_path / "phase13_tile_index.csv")),
                "--variants",
                "B0,B1",
                "--total-timesteps",
                "8",
                "--eval-max-steps",
                "2",
                "--seeds",
                "0,1",
                "--max-eval-tiles",
                "2",
                "--output-dir",
                str(tmp_path / "outputs"),
            ]
        )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Train tile: tile_r000_c001" in stdout
    assert "Held-out evaluation tiles: tile_r000_c002, tile_r000_c000" in stdout
    assert "Padded max blocks: 3" in stdout
    assert "Seeds: 0, 1" in stdout
    assert "Variants: B0, B1" in stdout
    assert "Summary rows: 24" in stdout
    assert "B1-B0 held-out learned-policy mean reward delta:" in stdout
    assert "phase25_padded_heldout_policy_comparison.json" in stdout
    assert (
        "Claim boundary: Phase 25 is a bounded padded variable-size held-out-tile"
    ) in stdout
