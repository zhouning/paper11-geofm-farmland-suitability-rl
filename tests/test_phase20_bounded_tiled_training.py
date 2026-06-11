import csv
import faulthandler
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


pytestmark = pytest.mark.filterwarnings("ignore:XPU device count is zero!:UserWarning")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id, suitability=0.75):
    row = {"block_id": block_id, "suitability_proxy": suitability}
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


def _write_ready_phase2_outputs(output_dir: Path):
    from paper11_geofm.artifacts import write_phase2_artifacts

    return write_phase2_artifacts(
        [
            _complete_phase2_feature_row("b1", 0.25),
            _complete_phase2_feature_row("b2", 0.50),
            _complete_phase2_feature_row("b3", 0.75),
            _complete_phase2_feature_row("b4", 1.00),
        ],
        output_dir,
        {
            "metadata_source": "test",
            "base_year_requested": 2020,
            "base_year_used": 2020,
            "years": [2020],
            "grid_shape": [2, 2],
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
                "block_ids": "b2",
            }
        )
        writer.writerow(
            {
                "tile_id": "tile_r000_c001",
                "tile_row": 0,
                "tile_col": 1,
                "n_blocks": 3,
                "block_ids": "b1;b3;b4",
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


def test_phase20_contract_selects_largest_train_and_same_eval_tile_by_default(tmp_path):
    from paper11_geofm.bounded_tiled_training import (
        PHASE20_CLAIM_BOUNDARY,
        build_phase20_bounded_training_contract,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    contract = build_phase20_bounded_training_contract(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variants=("B0", "B1"),
        total_timesteps=8,
        eval_max_steps=2,
        seed=0,
    )

    assert contract["phase"] == "phase20_bounded_tiled_training"
    assert contract["variants"] == ["B0", "B1"]
    assert contract["train_tile_id"] == "tile_r000_c001"
    assert contract["eval_tile_id"] == "tile_r000_c001"
    assert contract["train_tile_selection"] == "largest"
    assert contract["eval_tile_selection"] == "same_as_train_default"
    assert contract["learned_policy_evaluation_scope"] == "same_tile_bounded_pilot"
    assert (
        contract["cross_tile_evaluation_status"]
        == "blocked_variable_observation_shape"
    )
    assert contract["total_timesteps"] == 8
    assert contract["eval_max_steps"] == 2
    assert contract["seed"] == 0
    assert contract["claim_boundary"] == PHASE20_CLAIM_BOUNDARY


def test_phase20_rejects_distinct_eval_tile_until_variable_shape_policy(tmp_path):
    from paper11_geofm.bounded_tiled_training import (
        build_phase20_bounded_training_contract,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with pytest.raises(ValueError, match="same train/evaluation tile"):
        build_phase20_bounded_training_contract(
            tmp_path / "phase2",
            _write_tile_index(tmp_path / "phase13_tile_index.csv"),
            train_tile_id="tile_r000_c001",
            eval_tile_id="tile_r000_c000",
        )


def test_phase20_rejects_suitability_reward_variants(tmp_path):
    from paper11_geofm.bounded_tiled_training import (
        build_phase20_bounded_training_contract,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with pytest.raises(ValueError, match="B0/B1"):
        build_phase20_bounded_training_contract(
            tmp_path / "phase2",
            _write_tile_index(tmp_path / "phase13_tile_index.csv"),
            variants=("B3",),
        )


def test_phase20_runs_tiny_b0_b1_training_and_evaluation(tmp_path):
    _require_maskableppo_dependencies()
    from paper11_geofm.bounded_tiled_training import (
        PHASE20_CLAIM_BOUNDARY,
        run_phase20_bounded_tiled_training,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with _torch_windows_faulthandler_guard():
        protocol = run_phase20_bounded_tiled_training(
            tmp_path / "phase2",
            _write_tile_index(tmp_path / "phase13_tile_index.csv"),
            variants=("B0", "B1"),
            total_timesteps=8,
            eval_max_steps=2,
            seed=0,
        )

    assert protocol["phase"] == "phase20_bounded_tiled_training"
    assert protocol["train_tile_id"] == "tile_r000_c001"
    assert protocol["eval_tile_id"] == "tile_r000_c001"
    assert protocol["eval_tile_selection"] == "same_as_train_default"
    assert (
        protocol["cross_tile_evaluation_status"]
        == "blocked_variable_observation_shape"
    )
    assert protocol["variants"] == ["B0", "B1"]
    assert protocol["training_completed"] is True
    assert protocol["all_evaluations_completed"] is True
    assert len(protocol["summaries"]) == 6
    row_types = {(row["row_type"], row["variant_id"]) for row in protocol["summaries"]}
    assert ("trained_policy", "B0") in row_types
    assert ("trained_policy", "B1") in row_types
    assert ("first_valid", "B0") in row_types
    assert ("seeded_random", "B1") in row_types
    assert all(row["claim_boundary"] == PHASE20_CLAIM_BOUNDARY for row in protocol["summaries"])
    assert protocol["dependencies"]["sb3_contrib"]["available"] is True


def test_phase20_writer_outputs_csv_and_json(tmp_path):
    from paper11_geofm.bounded_tiled_training import (
        PHASE20_CLAIM_BOUNDARY,
        write_phase20_bounded_tiled_training_artifacts,
    )

    protocol = {
        "phase": "phase20_bounded_tiled_training",
        "summaries": [
            {
                "row_type": "trained_policy",
                "variant_id": "B0",
                "train_tile_id": "tile_r000_c001",
                "eval_tile_id": "tile_r000_c001",
                "seed": 0,
                "train_timesteps": 8,
                "eval_max_steps": 2,
                "n_blocks": 1,
                "n_features": 17,
                "observation_shape": 20,
                "action_space_n": 1,
                "episode_steps": 1,
                "terminated": True,
                "truncated": False,
                "total_contract_reward": 0.6,
                "selected_block_ids": ["b2"],
                "claim_boundary": PHASE20_CLAIM_BOUNDARY,
            }
        ],
        "traces": {"trained_policy": {"B0": []}},
        "claim_boundary": PHASE20_CLAIM_BOUNDARY,
    }

    paths = write_phase20_bounded_tiled_training_artifacts(
        protocol,
        tmp_path / "outputs",
    )

    assert paths["summary_csv"].name == "phase20_bounded_tiled_training_summary.csv"
    assert paths["traces_json"].name == "phase20_bounded_tiled_training_traces.json"
    rows = list(csv.DictReader(paths["summary_csv"].open("r", encoding="utf-8")))
    assert rows[0]["row_type"] == "trained_policy"
    assert rows[0]["selected_block_ids"] == "b2"
    saved = json.loads(paths["traces_json"].read_text(encoding="utf-8"))
    assert saved["phase"] == "phase20_bounded_tiled_training"


def test_phase20_cli_writes_outputs_and_prints_summary(tmp_path, capsys):
    _require_maskableppo_dependencies()
    runner_path = (
        ROOT
        / "experiments"
        / "phase20_bounded_tiled_training"
        / "run_phase20_bounded_tiled_training.py"
    )
    spec = importlib.util.spec_from_file_location("phase20_runner", runner_path)
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
                "--seed",
                "0",
                "--output-dir",
                str(tmp_path / "outputs"),
            ]
        )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Train tile: tile_r000_c001" in stdout
    assert "Evaluation tile: tile_r000_c001" in stdout
    assert "Cross-tile learned-policy status: blocked_variable_observation_shape" in stdout
    assert "Variants: B0, B1" in stdout
    assert "Summary rows: 6" in stdout
    assert "phase20_bounded_tiled_training_summary.csv" in stdout
    assert "Claim boundary: Phase 20 is a bounded same-tile B0/B1 training pilot" in stdout
