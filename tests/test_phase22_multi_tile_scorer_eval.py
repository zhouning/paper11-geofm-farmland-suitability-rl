import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


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


def test_phase22_contract_selects_largest_train_multi_eval_tiles_and_seeds(tmp_path):
    from paper11_geofm.multi_tile_scorer_eval import (
        PHASE22_CLAIM_BOUNDARY,
        build_phase22_multi_tile_scorer_eval_contract,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    contract = build_phase22_multi_tile_scorer_eval_contract(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variants=("B0", "B1"),
        ridge_alpha=1e-6,
        eval_max_steps=2,
        seeds="0,1",
        max_eval_tiles=2,
    )

    assert contract["phase"] == "phase22_multi_tile_scorer_eval"
    assert contract["variants"] == ["B0", "B1"]
    assert contract["train_tile_id"] == "tile_r000_c001"
    assert contract["eval_tile_ids"] == ["tile_r000_c002", "tile_r000_c000"]
    assert contract["eval_tile_ranks"] == {
        "tile_r000_c002": 1,
        "tile_r000_c000": 2,
    }
    assert contract["train_tile_selection"] == "largest"
    assert contract["eval_tile_selection"] == "largest_distinct"
    assert contract["learned_policy_evaluation_scope"] == (
        "multi_tile_multi_seed_per_block_scorer_pilot"
    )
    assert contract["multi_tile_evaluation_status"] == "executed_distinct_tiles"
    assert contract["ridge_alpha"] == 1e-6
    assert contract["eval_max_steps"] == 2
    assert contract["seeds"] == [0, 1]
    assert contract["claim_boundary"] == PHASE22_CLAIM_BOUNDARY


def test_phase22_rejects_suitability_reward_variants(tmp_path):
    from paper11_geofm.multi_tile_scorer_eval import (
        build_phase22_multi_tile_scorer_eval_contract,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with pytest.raises(ValueError, match="B0/B1"):
        build_phase22_multi_tile_scorer_eval_contract(
            tmp_path / "phase2",
            _write_tile_index(tmp_path / "phase13_tile_index.csv"),
            variants=("B3",),
        )


def test_phase22_runs_multi_tile_multi_seed_scorer_and_baselines(tmp_path):
    from paper11_geofm.multi_tile_scorer_eval import (
        PHASE22_CLAIM_BOUNDARY,
        run_phase22_multi_tile_scorer_eval,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    protocol = run_phase22_multi_tile_scorer_eval(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variants=("B0", "B1"),
        ridge_alpha=1e-6,
        eval_max_steps=2,
        seeds=(0, 1),
        max_eval_tiles=2,
    )

    assert protocol["phase"] == "phase22_multi_tile_scorer_eval"
    assert protocol["train_tile_id"] == "tile_r000_c001"
    assert protocol["eval_tile_ids"] == ["tile_r000_c002", "tile_r000_c000"]
    assert protocol["multi_tile_evaluation_status"] == "executed_distinct_tiles"
    assert protocol["all_evaluations_completed"] is True
    assert protocol["summary_count"] == 24
    assert len(protocol["summaries"]) == 24
    row_keys = {
        (
            row["row_type"],
            row["variant_id"],
            row["eval_tile_id"],
            row["eval_tile_rank"],
            row["seed"],
        )
        for row in protocol["summaries"]
    }
    assert ("learned_block_scorer", "B0", "tile_r000_c002", 1, 0) in row_keys
    assert ("first_valid", "B1", "tile_r000_c000", 2, 1) in row_keys
    assert ("seeded_random", "B1", "tile_r000_c002", 1, 1) in row_keys
    assert all(row["train_n_blocks"] == 3 for row in protocol["summaries"])
    assert all(row["eval_tile_rank"] in {1, 2} for row in protocol["summaries"])
    assert all(row["claim_boundary"] == PHASE22_CLAIM_BOUNDARY for row in protocol["summaries"])
    assert protocol["model_metadata"]["B0"]["model_type"] == "standardized_ridge_linear"
    assert protocol["traces"]["learned_block_scorer"]["B0"]["tile_r000_c002"]["0"]
    assert protocol["traces"]["seeded_random"]["B1"]["tile_r000_c000"]["1"]


def test_phase22_writer_outputs_csv_and_json(tmp_path):
    from paper11_geofm.multi_tile_scorer_eval import (
        PHASE22_CLAIM_BOUNDARY,
        write_phase22_multi_tile_scorer_eval_artifacts,
    )

    protocol = {
        "phase": "phase22_multi_tile_scorer_eval",
        "summaries": [
            {
                "row_type": "learned_block_scorer",
                "variant_id": "B0",
                "train_tile_id": "tile_r000_c001",
                "eval_tile_id": "tile_r000_c002",
                "eval_tile_rank": 1,
                "seed": 0,
                "ridge_alpha": 1e-6,
                "eval_max_steps": 2,
                "train_n_blocks": 3,
                "eval_n_blocks": 2,
                "n_features": 17,
                "eval_observation_shape": 37,
                "action_space_n": 2,
                "episode_steps": 2,
                "terminated": True,
                "truncated": False,
                "total_contract_reward": -0.2,
                "selected_block_ids": ["b2", "b4"],
                "claim_boundary": PHASE22_CLAIM_BOUNDARY,
            }
        ],
        "traces": {
            "learned_block_scorer": {"B0": {"tile_r000_c002": {"0": []}}},
        },
        "model_metadata": {"B0": {"model_type": "standardized_ridge_linear"}},
        "claim_boundary": PHASE22_CLAIM_BOUNDARY,
    }

    paths = write_phase22_multi_tile_scorer_eval_artifacts(
        protocol,
        tmp_path / "outputs",
    )

    assert paths["summary_csv"].name == "phase22_multi_tile_scorer_eval_summary.csv"
    assert paths["traces_json"].name == "phase22_multi_tile_scorer_eval_traces.json"
    rows = list(csv.DictReader(paths["summary_csv"].open("r", encoding="utf-8")))
    assert rows[0]["eval_tile_rank"] == "1"
    assert rows[0]["selected_block_ids"] == "b2;b4"
    saved = json.loads(paths["traces_json"].read_text(encoding="utf-8"))
    assert saved["phase"] == "phase22_multi_tile_scorer_eval"


def test_phase22_cli_writes_outputs_and_prints_summary(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase22_multi_tile_scorer_eval"
        / "run_phase22_multi_tile_scorer_eval.py"
    )
    spec = importlib.util.spec_from_file_location("phase22_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    _write_ready_phase2_outputs(tmp_path / "phase2")
    exit_code = module.main(
        [
            "--phase2-output-dir",
            str(tmp_path / "phase2"),
            "--tile-index-csv",
            str(_write_tile_index(tmp_path / "phase13_tile_index.csv")),
            "--variants",
            "B0,B1",
            "--ridge-alpha",
            "1e-6",
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
    assert "Evaluation tiles: tile_r000_c002, tile_r000_c000" in stdout
    assert "Seeds: 0, 1" in stdout
    assert "Multi-tile learned-policy status: executed_distinct_tiles" in stdout
    assert "Variants: B0, B1" in stdout
    assert "Summary rows: 24" in stdout
    assert "phase22_multi_tile_scorer_eval_summary.csv" in stdout
    assert (
        "Claim boundary: Phase 22 is a bounded multi-tile, multi-seed "
        "per-block scorer evaluation pilot"
    ) in stdout
