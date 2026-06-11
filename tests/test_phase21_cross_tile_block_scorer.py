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


def test_phase21_contract_selects_largest_train_and_distinct_eval_tile(tmp_path):
    from paper11_geofm.cross_tile_block_scorer import (
        PHASE21_CLAIM_BOUNDARY,
        build_phase21_cross_tile_scorer_contract,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    contract = build_phase21_cross_tile_scorer_contract(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variants=("B0", "B1"),
        ridge_alpha=1e-6,
        eval_max_steps=2,
        seed=0,
    )

    assert contract["phase"] == "phase21_cross_tile_block_scorer"
    assert contract["variants"] == ["B0", "B1"]
    assert contract["train_tile_id"] == "tile_r000_c001"
    assert contract["eval_tile_id"] == "tile_r000_c000"
    assert contract["train_tile_selection"] == "largest"
    assert contract["eval_tile_selection"] == "next_largest_distinct"
    assert contract["learned_policy_evaluation_scope"] == "cross_tile_per_block_scorer_pilot"
    assert contract["cross_tile_evaluation_status"] == "executed_distinct_tile"
    assert contract["ridge_alpha"] == 1e-6
    assert contract["eval_max_steps"] == 2
    assert contract["seed"] == 0
    assert contract["claim_boundary"] == PHASE21_CLAIM_BOUNDARY


def test_phase21_rejects_suitability_reward_variants(tmp_path):
    from paper11_geofm.cross_tile_block_scorer import (
        build_phase21_cross_tile_scorer_contract,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with pytest.raises(ValueError, match="B0/B1"):
        build_phase21_cross_tile_scorer_contract(
            tmp_path / "phase2",
            _write_tile_index(tmp_path / "phase13_tile_index.csv"),
            variants=("B3",),
        )


def test_phase21_rejects_same_train_and_eval_tile(tmp_path):
    from paper11_geofm.cross_tile_block_scorer import (
        build_phase21_cross_tile_scorer_contract,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with pytest.raises(ValueError, match="distinct"):
        build_phase21_cross_tile_scorer_contract(
            tmp_path / "phase2",
            _write_tile_index(tmp_path / "phase13_tile_index.csv"),
            train_tile_id="tile_r000_c001",
            eval_tile_id="tile_r000_c001",
        )


def test_phase21_runs_cross_tile_b0_b1_scorer_and_baselines(tmp_path):
    from paper11_geofm.cross_tile_block_scorer import (
        PHASE21_CLAIM_BOUNDARY,
        run_phase21_cross_tile_block_scorer,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    protocol = run_phase21_cross_tile_block_scorer(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variants=("B0", "B1"),
        ridge_alpha=1e-6,
        eval_max_steps=2,
        seed=0,
    )

    assert protocol["phase"] == "phase21_cross_tile_block_scorer"
    assert protocol["train_tile_id"] == "tile_r000_c001"
    assert protocol["eval_tile_id"] == "tile_r000_c000"
    assert protocol["cross_tile_evaluation_status"] == "executed_distinct_tile"
    assert protocol["all_evaluations_completed"] is True
    assert len(protocol["summaries"]) == 6
    row_types = {(row["row_type"], row["variant_id"]) for row in protocol["summaries"]}
    assert ("learned_block_scorer", "B0") in row_types
    assert ("learned_block_scorer", "B1") in row_types
    assert ("first_valid", "B0") in row_types
    assert ("seeded_random", "B1") in row_types
    assert all(row["train_n_blocks"] == 3 for row in protocol["summaries"])
    assert all(row["eval_n_blocks"] == 1 for row in protocol["summaries"])
    assert all(row["selected_block_ids"] == ["b2"] for row in protocol["summaries"])
    assert all(row["claim_boundary"] == PHASE21_CLAIM_BOUNDARY for row in protocol["summaries"])
    assert protocol["model_metadata"]["B0"]["model_type"] == "standardized_ridge_linear"


def test_phase21_writer_outputs_csv_and_json(tmp_path):
    from paper11_geofm.cross_tile_block_scorer import (
        PHASE21_CLAIM_BOUNDARY,
        write_phase21_cross_tile_block_scorer_artifacts,
    )

    protocol = {
        "phase": "phase21_cross_tile_block_scorer",
        "summaries": [
            {
                "row_type": "learned_block_scorer",
                "variant_id": "B0",
                "train_tile_id": "tile_r000_c001",
                "eval_tile_id": "tile_r000_c000",
                "seed": 0,
                "ridge_alpha": 1e-6,
                "eval_max_steps": 2,
                "train_n_blocks": 3,
                "eval_n_blocks": 1,
                "n_features": 17,
                "eval_observation_shape": 20,
                "action_space_n": 1,
                "episode_steps": 1,
                "terminated": True,
                "truncated": False,
                "total_contract_reward": -0.2,
                "selected_block_ids": ["b2"],
                "claim_boundary": PHASE21_CLAIM_BOUNDARY,
            }
        ],
        "traces": {"learned_block_scorer": {"B0": []}},
        "model_metadata": {"B0": {"model_type": "standardized_ridge_linear"}},
        "claim_boundary": PHASE21_CLAIM_BOUNDARY,
    }

    paths = write_phase21_cross_tile_block_scorer_artifacts(
        protocol,
        tmp_path / "outputs",
    )

    assert paths["summary_csv"].name == "phase21_cross_tile_block_scorer_summary.csv"
    assert paths["traces_json"].name == "phase21_cross_tile_block_scorer_traces.json"
    rows = list(csv.DictReader(paths["summary_csv"].open("r", encoding="utf-8")))
    assert rows[0]["row_type"] == "learned_block_scorer"
    assert rows[0]["selected_block_ids"] == "b2"
    saved = json.loads(paths["traces_json"].read_text(encoding="utf-8"))
    assert saved["phase"] == "phase21_cross_tile_block_scorer"


def test_phase21_cli_writes_outputs_and_prints_summary(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase21_cross_tile_block_scorer"
        / "run_phase21_cross_tile_block_scorer.py"
    )
    spec = importlib.util.spec_from_file_location("phase21_runner", runner_path)
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
            "--seed",
            "0",
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Train tile: tile_r000_c001" in stdout
    assert "Evaluation tile: tile_r000_c000" in stdout
    assert "Cross-tile learned-policy status: executed_distinct_tile" in stdout
    assert "Variants: B0, B1" in stdout
    assert "Summary rows: 6" in stdout
    assert "phase21_cross_tile_block_scorer_summary.csv" in stdout
    assert "Claim boundary: Phase 21 is a bounded cross-tile per-block scorer pilot" in stdout
