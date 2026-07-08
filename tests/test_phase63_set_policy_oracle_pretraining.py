import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _required_feature_columns() -> tuple[str, ...]:
    return (
        "explicit_feature_00",
        "explicit_feature_01",
        "explicit_feature_02",
        "explicit_feature_04",
        "explicit_feature_07",
        "explicit_feature_09",
        "explicit_feature_10",
        "explicit_feature_13",
        "explicit_feature_16",
    )


def _tiled_input(
    block_ids=("b2", "b1", "b3"),
    scores=(0.5, 0.5, 0.3),
    variant_id="B0",
):
    from paper11_geofm.tiled_inputs import TiledVariantInput

    columns = _required_feature_columns()
    matrix = np.zeros((len(block_ids), len(columns)), dtype=np.float32)
    slope_farmland_index = columns.index("explicit_feature_16")
    for row_index, score in enumerate(scores):
        matrix[row_index, slope_farmland_index] = float(score)
    return TiledVariantInput(
        tile_id="tile_eval",
        variant_id=variant_id,
        block_ids=tuple(block_ids),
        feature_columns=columns,
        state_matrix=matrix,
        reward_mode="base_planning_reward",
        state_groups=("explicit_planning_features",),
        source_table=Path("variant_B0_features.csv"),
        tile_index_csv=Path("tiles.csv"),
    )


def test_phase63_contract_routes_b0_d4_and_d6_variants(tmp_path):
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        build_phase63_set_policy_contract,
    )

    tile_index = _write_csv(
        tmp_path / "tiles.csv",
        [
            {"tile_id": "tile_train", "block_ids": "b1;b2;b3;b4"},
            {"tile_id": "tile_eval", "block_ids": "b1;b2;b3"},
        ],
    )
    contract = build_phase63_set_policy_contract(
        phase2_output_dir=tmp_path / "phase2",
        phase8_output_dir=tmp_path / "phase8",
        phase61_output_dir=tmp_path / "phase61",
        tile_index_csv=tile_index,
        train_tile_id="tile_train",
        eval_tile_ids="tile_eval",
        variants="B0,D4P8,D4P16,D6R8,D6R16",
        seeds="0,1",
        eval_max_steps=2,
        bc_epochs=5,
        learning_rate=0.01,
        hidden_dim=16,
        top_k=2,
    )

    assert contract["phase"] == "phase63_set_policy_oracle_pretraining"
    assert contract["variants"] == ["B0", "D4P8", "D4P16", "D6R8", "D6R16"]
    assert contract["variant_source_dirs"]["B0"].endswith("phase2")
    assert contract["variant_source_dirs"]["D4P8"].endswith("phase8")
    assert contract["variant_source_dirs"]["D6R16"].endswith("phase61")
    assert contract["eval_tile_ids"] == ["tile_eval"]
    assert contract["seeds"] == [0, 1]
    assert contract["eval_max_steps"] == 2
    assert contract["bc_epochs"] == 5
    assert contract["top_k"] == 2


def test_phase63_oracle_uses_reward_descending_then_block_id_tiebreak():
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        build_phase63_oracle_trajectory,
    )

    trajectory = build_phase63_oracle_trajectory(
        _tiled_input(),
        eval_max_steps=3,
    )

    assert trajectory["action_indices"] == [1, 0, 2]
    assert trajectory["selected_block_ids"] == ["b1", "b2", "b3"]
    assert trajectory["step_rewards"] == [0.175, 0.175, 0.105]
    assert trajectory["total_oracle_reward"] == 0.455
    assert trajectory["top_k_reward_ceiling"] == 0.455


def test_phase63_oracle_stops_at_eval_max_steps():
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        build_phase63_oracle_trajectory,
    )

    trajectory = build_phase63_oracle_trajectory(
        _tiled_input(block_ids=("b3", "b1", "b2"), scores=(0.1, 0.9, 0.8)),
        eval_max_steps=2,
    )

    assert trajectory["action_indices"] == [1, 2]
    assert trajectory["selected_block_ids"] == ["b1", "b2"]
    assert trajectory["episode_steps"] == 2
    assert trajectory["terminated"] is False


def test_phase63_model_inputs_encode_valid_selected_and_available_masks():
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        build_phase63_model_inputs,
    )

    inputs = build_phase63_model_inputs(
        _tiled_input(block_ids=("b1", "b2", "b3"), scores=(0.9, 0.4, 0.2)),
        selected_indices=(1,),
    )

    assert inputs["block_features"].shape == (3, 9)
    assert inputs["valid_mask"].tolist() == [True, True, True]
    assert inputs["selected_mask"].tolist() == [False, True, False]
    assert inputs["available_mask"].tolist() == [True, False, True]


def test_phase63_set_policy_scorer_masks_selected_and_invalid_actions():
    import torch
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        Phase63SetPolicyScorer,
    )

    torch.manual_seed(63)
    model = Phase63SetPolicyScorer(n_features=9, hidden_dim=12)
    block_features = torch.zeros((1, 4, 9), dtype=torch.float32)
    valid_mask = torch.tensor([[True, True, False, False]])
    selected_mask = torch.tensor([[False, True, False, False]])

    logits = model(block_features, valid_mask, selected_mask)

    assert logits.shape == (1, 4)
    assert torch.isfinite(logits[0, 0])
    assert logits[0, 1].item() < -1e8
    assert logits[0, 2].item() < -1e8
    assert logits[0, 3].item() < -1e8


def test_phase63_behavior_cloning_loss_decreases_on_tiny_tile():
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        train_phase63_behavior_cloner,
    )

    model, history = train_phase63_behavior_cloner(
        _tiled_input(block_ids=("b1", "b2", "b3", "b4"), scores=(0.9, 0.7, 0.2, 0.1)),
        seed=63,
        eval_max_steps=3,
        epochs=25,
        learning_rate=0.01,
        hidden_dim=16,
        top_k=2,
    )

    assert model.n_features == 9
    assert len(history) == 25
    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["top1_accuracy"] >= history[0]["top1_accuracy"]


def test_phase63_greedy_rollout_never_selects_invalid_or_repeated_actions():
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        rollout_phase63_greedy_policy,
        train_phase63_behavior_cloner,
    )

    tiled = _tiled_input(
        block_ids=("b3", "b1", "b2", "b4"),
        scores=(0.2, 0.9, 0.7, 0.1),
    )
    model, _history = train_phase63_behavior_cloner(
        tiled,
        seed=63,
        eval_max_steps=3,
        epochs=30,
        learning_rate=0.01,
        hidden_dim=16,
        top_k=2,
    )
    rollout = rollout_phase63_greedy_policy(
        model,
        tiled,
        train_tile_id="tile_train",
        eval_tile_rank=1,
        seed=63,
        phase63_seed_rank=1,
        eval_max_steps=3,
    )

    assert rollout["row_type"] == "bc_greedy_policy"
    assert rollout["all_actions_valid"] is True
    assert rollout["invalid_action_count"] == 0
    assert len(rollout["selected_action_indices"].split(";")) == 3
    assert len(set(rollout["selected_action_indices"].split(";"))) == 3
    assert float(rollout["total_contract_reward"]) > 0.0


def _rollout_row(variant_id, reward, oracle=1.0, tile_id="tile_a", seed=0):
    return {
        "row_type": "bc_greedy_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": tile_id,
        "eval_tile_rank": 1 if tile_id == "tile_a" else 2,
        "seed": seed,
        "phase63_seed_rank": seed + 1,
        "eval_max_steps": 8,
        "n_blocks": 4,
        "n_features": 9,
        "episode_steps": 3,
        "terminated": False,
        "truncated": True,
        "all_actions_valid": True,
        "invalid_action_count": 0,
        "total_contract_reward": reward,
        "oracle_total_reward": oracle,
        "oracle_gap": oracle - reward,
        "oracle_gap_fraction": (oracle - reward) / oracle,
        "selected_block_ids": "b1;b2;b3",
        "selected_action_indices": "0;1;2",
        "claim_boundary": "fixture",
    }


def _flattened_row(variant_id, reward, tile_id="tile_a", seed=0):
    return {
        "row_type": "trained_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": tile_id,
        "seed": seed,
        "total_contract_reward": reward,
    }


def test_phase63_analysis_reports_architecture_improvement_with_complete_baseline():
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        build_phase63_set_policy_analysis,
    )

    pairs = [("tile_a", 0), ("tile_a", 1), ("tile_b", 0), ("tile_b", 1)]
    rollout_rows = []
    flattened_rows = []
    for tile_id, seed in pairs:
        rollout_rows.extend(
            [
                _rollout_row("B0", 1.10, tile_id=tile_id, seed=seed),
                _rollout_row("D4P8", 1.30, tile_id=tile_id, seed=seed),
                _rollout_row("D4P16", 1.35, tile_id=tile_id, seed=seed),
                _rollout_row("D6R8", 1.25, tile_id=tile_id, seed=seed),
                _rollout_row("D6R16", 1.28, tile_id=tile_id, seed=seed),
            ]
        )
        flattened_rows.extend(
            [
                _flattened_row("B0", 0.90, tile_id=tile_id, seed=seed),
                _flattened_row("D4P8", 1.00, tile_id=tile_id, seed=seed),
                _flattened_row("D4P16", 1.00, tile_id=tile_id, seed=seed),
                _flattened_row("D6R8", 1.00, tile_id=tile_id, seed=seed),
                _flattened_row("D6R16", 1.00, tile_id=tile_id, seed=seed),
            ]
        )
    analysis = build_phase63_set_policy_analysis(
        rollout_rows,
        existing_flattened_rows=flattened_rows,
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
    )

    assert analysis["phase63_set_policy_status"] == "geofm_set_policy_advantage"
    assert analysis["architecture_delta_summary"]["mean_delta"] > 0
    assert analysis["d4_b0_delta_summary"]["positive_count"] == 8
    assert analysis["coverage_issues"]["missing_rollout_rows"] == []


def test_phase63_writer_outputs_json_csv_and_markdown(tmp_path):
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        build_phase63_set_policy_analysis,
        write_phase63_set_policy_artifacts,
    )

    rollout_rows = [_rollout_row("B0", 1.0), _rollout_row("D4P8", 1.2)]
    flattened_rows = [_flattened_row("B0", 0.8), _flattened_row("D4P8", 0.9)]
    analysis = build_phase63_set_policy_analysis(
        rollout_rows,
        existing_flattened_rows=flattened_rows,
        metadata={"eval_tile_ids": ["tile_a"], "seeds": [0], "variants": ["B0", "D4P8"]},
    )
    paths = write_phase63_set_policy_artifacts(
        {
            **analysis,
            "oracle_trajectories": [],
            "oracle_summary_rows": [],
            "history_rows": [],
            "rollout_rows": rollout_rows,
        },
        tmp_path / "outputs",
    )

    assert paths["oracle_json"].name == "phase63_oracle_trajectories.json"
    assert paths["oracle_summary_csv"].name == "phase63_oracle_summary.csv"
    assert paths["history_csv"].name == "phase63_bc_training_history.csv"
    assert paths["rollout_csv"].name == "phase63_bc_rollout_summary.csv"
    assert paths["comparison_json"].name == "phase63_set_policy_comparison.json"
    assert paths["delta_csv"].name == "phase63_set_policy_delta_table.csv"
    assert paths["readiness_md"].name == "phase63_set_policy_oracle_pretraining.md"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase"] == "phase63_set_policy_analysis"
    markdown = paths["readiness_md"].read_text(encoding="utf-8")
    assert "Phase 63 Set-Policy Oracle Pretraining" in markdown
    assert "does not enable suitability reward" in markdown


def test_phase63_cli_analyze_only(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase63_set_policy_oracle_pretraining"
        / "run_phase63_set_policy_oracle_pretraining.py"
    )
    spec = importlib.util.spec_from_file_location("phase63_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    rollout_csv = _write_csv(tmp_path / "rollout.csv", [_rollout_row("B0", 1.0)])
    flattened_csv = _write_csv(tmp_path / "flat.csv", [_flattened_row("B0", 0.8)])
    exit_code = module.main(
        [
            "--mode",
            "analyze-only",
            "--existing-rollout-csv",
            str(rollout_csv),
            "--existing-flattened-summary-csvs",
            str(flattened_csv),
            "--output-dir",
            str(tmp_path / "analysis"),
            "--eval-tile-ids",
            "tile_a",
            "--seeds",
            "0",
            "--variants",
            "B0",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Phase 63 set-policy status:" in stdout
    assert "phase63_set_policy_comparison.json" in stdout


def test_phase63_cli_rollout_only_parser_accepts_core_inputs():
    runner_path = (
        ROOT
        / "experiments"
        / "phase63_set_policy_oracle_pretraining"
        / "run_phase63_set_policy_oracle_pretraining.py"
    )
    spec = importlib.util.spec_from_file_location("phase63_runner_args", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "rollout-only",
            "--phase2-output-dir",
            "phase2",
            "--phase8-output-dir",
            "phase8",
            "--phase61-output-dir",
            "phase61",
            "--tile-index-csv",
            "tiles.csv",
            "--variants",
            "B0,D4P8",
            "--output-dir",
            "outputs",
        ]
    )

    assert args.mode == "rollout-only"
    assert args.variants == "B0,D4P8"
