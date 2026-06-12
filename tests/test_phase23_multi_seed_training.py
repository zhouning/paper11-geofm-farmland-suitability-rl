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


def test_phase23_contract_selects_largest_same_tile_and_normalizes_seeds(tmp_path):
    from paper11_geofm.multi_seed_training import (
        PHASE23_CLAIM_BOUNDARY,
        build_phase23_multi_seed_training_contract,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    contract = build_phase23_multi_seed_training_contract(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variants=("B0", "B1"),
        total_timesteps=8,
        eval_max_steps=2,
        seeds="0,1",
    )

    assert contract["phase"] == "phase23_multi_seed_training"
    assert contract["variants"] == ["B0", "B1"]
    assert contract["train_tile_id"] == "tile_r000_c001"
    assert contract["eval_tile_id"] == "tile_r000_c001"
    assert contract["train_tile_selection"] == "largest"
    assert contract["eval_tile_selection"] == "same_as_train_default"
    assert contract["seeds"] == [0, 1]
    assert contract["seed_ranks"] == {"0": 1, "1": 2}
    assert contract["total_timesteps"] == 8
    assert contract["eval_max_steps"] == 2
    assert contract["learned_policy_evaluation_scope"] == (
        "multi_seed_same_tile_b0_b1_training_pilot"
    )
    assert contract["cross_tile_evaluation_status"] == "blocked_variable_observation_shape"
    assert contract["claim_boundary"] == PHASE23_CLAIM_BOUNDARY


def test_phase23_rejects_suitability_reward_variants_and_distinct_eval_tile(tmp_path):
    from paper11_geofm.multi_seed_training import (
        build_phase23_multi_seed_training_contract,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    tile_index = _write_tile_index(tmp_path / "phase13_tile_index.csv")
    with pytest.raises(ValueError, match="B0/B1"):
        build_phase23_multi_seed_training_contract(
            tmp_path / "phase2",
            tile_index,
            variants=("B3",),
        )

    with pytest.raises(ValueError, match="same train/evaluation tile"):
        build_phase23_multi_seed_training_contract(
            tmp_path / "phase2",
            tile_index,
            train_tile_id="tile_r000_c001",
            eval_tile_id="tile_r000_c000",
        )


def test_phase23_runs_multi_seed_b0_b1_training_and_comparison(tmp_path):
    _require_maskableppo_dependencies()
    from paper11_geofm.multi_seed_training import (
        PHASE23_CLAIM_BOUNDARY,
        run_phase23_multi_seed_training,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with _torch_windows_faulthandler_guard():
        protocol = run_phase23_multi_seed_training(
            tmp_path / "phase2",
            _write_tile_index(tmp_path / "phase13_tile_index.csv"),
            variants=("B0", "B1"),
            total_timesteps=8,
            eval_max_steps=2,
            seeds=(0, 1),
        )

    assert protocol["phase"] == "phase23_multi_seed_training"
    assert protocol["train_tile_id"] == "tile_r000_c001"
    assert protocol["eval_tile_id"] == "tile_r000_c001"
    assert protocol["training_completed"] is True
    assert protocol["all_evaluations_completed"] is True
    assert protocol["summary_count"] == 12
    assert len(protocol["summaries"]) == 12
    row_keys = {
        (
            row["row_type"],
            row["variant_id"],
            row["seed"],
            row["phase23_seed_rank"],
        )
        for row in protocol["summaries"]
    }
    assert ("trained_policy", "B0", 0, 1) in row_keys
    assert ("trained_policy", "B1", 1, 2) in row_keys
    assert ("first_valid", "B0", 0, 1) in row_keys
    assert ("seeded_random", "B1", 1, 2) in row_keys
    assert all(row["claim_boundary"] == PHASE23_CLAIM_BOUNDARY for row in protocol["summaries"])
    assert protocol["comparison"]["learned_policy"]["B1_minus_B0_mean_reward"] is not None
    assert protocol["comparison"]["remaining_evidence_gaps"]
    assert protocol["traces"]["trained_policy"]["B0"]["0"]
    assert protocol["traces"]["seeded_random"]["B1"]["1"]


def test_phase23_writer_outputs_csv_json_and_comparison(tmp_path):
    from paper11_geofm.multi_seed_training import (
        PHASE23_CLAIM_BOUNDARY,
        write_phase23_multi_seed_training_artifacts,
    )

    protocol = {
        "phase": "phase23_multi_seed_training",
        "summaries": [
            {
                "row_type": "trained_policy",
                "variant_id": "B0",
                "train_tile_id": "tile_r000_c001",
                "eval_tile_id": "tile_r000_c001",
                "seed": 0,
                "phase23_seed_rank": 1,
                "train_timesteps": 8,
                "eval_max_steps": 2,
                "n_blocks": 3,
                "n_features": 17,
                "observation_shape": 54,
                "action_space_n": 3,
                "episode_steps": 2,
                "terminated": True,
                "truncated": False,
                "total_contract_reward": 1.2,
                "selected_block_ids": ["b1", "b3"],
                "claim_boundary": PHASE23_CLAIM_BOUNDARY,
            }
        ],
        "traces": {"trained_policy": {"B0": {"0": []}}},
        "comparison": {
            "learned_policy": {"B1_minus_B0_mean_reward": None},
            "claim_boundary": PHASE23_CLAIM_BOUNDARY,
        },
        "claim_boundary": PHASE23_CLAIM_BOUNDARY,
    }

    paths = write_phase23_multi_seed_training_artifacts(
        protocol,
        tmp_path / "outputs",
    )

    assert paths["summary_csv"].name == "phase23_multi_seed_training_summary.csv"
    assert paths["traces_json"].name == "phase23_multi_seed_training_traces.json"
    assert paths["comparison_json"].name == "phase23_multi_seed_training_comparison.json"
    rows = list(csv.DictReader(paths["summary_csv"].open("r", encoding="utf-8")))
    assert rows[0]["phase23_seed_rank"] == "1"
    assert rows[0]["selected_block_ids"] == "b1;b3"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["claim_boundary"] == PHASE23_CLAIM_BOUNDARY


def test_phase23_cli_writes_outputs_and_prints_summary(tmp_path, capsys):
    _require_maskableppo_dependencies()
    runner_path = (
        ROOT
        / "experiments"
        / "phase23_multi_seed_training"
        / "run_phase23_multi_seed_training.py"
    )
    spec = importlib.util.spec_from_file_location("phase23_runner", runner_path)
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
                "--output-dir",
                str(tmp_path / "outputs"),
            ]
        )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Train tile: tile_r000_c001" in stdout
    assert "Evaluation tile: tile_r000_c001" in stdout
    assert "Seeds: 0, 1" in stdout
    assert "Variants: B0, B1" in stdout
    assert "Summary rows: 12" in stdout
    assert "B1-B0 learned-policy mean reward delta:" in stdout
    assert "phase23_multi_seed_training_comparison.json" in stdout
    assert (
        "Claim boundary: Phase 23 is a bounded multi-seed same-tile B0/B1 "
        "MaskablePPO training pilot"
    ) in stdout
