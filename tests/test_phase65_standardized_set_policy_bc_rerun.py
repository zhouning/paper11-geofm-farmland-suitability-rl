import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _tiled_input(matrix, variant_id="D4P8", tile_id="tile_train"):
    from paper11_geofm.tiled_inputs import TiledVariantInput

    array = np.asarray(matrix, dtype=np.float32)
    feature_columns = tuple(f"feature_{index:02d}" for index in range(array.shape[1]))
    block_ids = tuple(f"b{index}" for index in range(array.shape[0]))
    return TiledVariantInput(
        tile_id=tile_id,
        variant_id=variant_id,
        block_ids=block_ids,
        feature_columns=feature_columns,
        state_matrix=array,
        reward_mode="base_planning_reward",
        state_groups=("synthetic",),
        source_table=Path(f"variant_{variant_id}_features.csv"),
        tile_index_csv=Path("tiles.csv"),
    )


def test_phase65_standardizer_fits_train_tile_and_applies_to_eval_without_eval_stats():
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        apply_phase65_standardizer,
        fit_phase65_train_tile_standardizer,
    )

    train = _tiled_input(
        [
            [1.0, 10.0, 5.0],
            [3.0, 14.0, 5.0],
            [5.0, 18.0, 5.0],
        ],
        variant_id="D4P8",
        tile_id="tile_train",
    )
    eval_tile = _tiled_input(
        [
            [7.0, 22.0, 5.0],
            [9.0, 26.0, 5.0],
        ],
        variant_id="D4P8",
        tile_id="tile_eval",
    )

    transform = fit_phase65_train_tile_standardizer(train)
    standardized_train = apply_phase65_standardizer(train, transform)
    standardized_eval = apply_phase65_standardizer(eval_tile, transform)

    np.testing.assert_allclose(transform.mean, np.array([3.0, 14.0, 5.0]))
    np.testing.assert_allclose(transform.safe_std[2], 1.0)
    np.testing.assert_allclose(
        standardized_train.state_matrix.mean(axis=0),
        np.array([0.0, 0.0, 0.0]),
        atol=1.0e-6,
    )
    expected_eval_first = np.array(
        [
            (7.0 - transform.mean[0]) / transform.safe_std[0],
            (22.0 - transform.mean[1]) / transform.safe_std[1],
            0.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(standardized_eval.state_matrix[0], expected_eval_first)
    assert standardized_eval.tile_id == "tile_eval"
    assert standardized_eval.variant_id == "D4P8"
    assert standardized_eval.block_ids == eval_tile.block_ids
    assert standardized_eval.feature_columns == eval_tile.feature_columns


def test_phase65_standardizer_rejects_mismatched_variant_and_columns():
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        apply_phase65_standardizer,
        fit_phase65_train_tile_standardizer,
    )

    transform = fit_phase65_train_tile_standardizer(
        _tiled_input([[1.0, 2.0], [3.0, 4.0]], variant_id="D4P8")
    )
    mismatched_variant = _tiled_input(
        [[1.0, 2.0], [3.0, 4.0]],
        variant_id="D6R8",
        tile_id="tile_eval",
    )
    mismatched_columns = _tiled_input(
        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
        variant_id="D4P8",
        tile_id="tile_eval",
    )

    try:
        apply_phase65_standardizer(mismatched_variant, transform)
    except ValueError as exc:
        assert "variant" in str(exc)
    else:
        raise AssertionError("Expected variant mismatch to fail")

    try:
        apply_phase65_standardizer(mismatched_columns, transform)
    except ValueError as exc:
        assert "feature columns" in str(exc)
    else:
        raise AssertionError("Expected feature-column mismatch to fail")


def test_phase65_standardized_inputs_do_not_change_raw_reward_or_oracle_targets():
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        build_phase63_oracle_trajectory,
    )
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        apply_phase65_standardizer,
        build_phase65_bc_examples,
        fit_phase65_train_tile_standardizer,
    )
    from paper11_geofm.tiled_inputs import TiledVariantInput

    columns = (
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
    matrix = np.zeros((3, len(columns)), dtype=np.float32)
    score_index = columns.index("explicit_feature_16")
    matrix[:, score_index] = np.array([0.9, 0.5, 0.1], dtype=np.float32)
    matrix[:, columns.index("explicit_feature_00")] = np.array(
        [100.0, 200.0, 300.0],
        dtype=np.float32,
    )
    raw = TiledVariantInput(
        tile_id="tile_train",
        variant_id="D4P8",
        block_ids=("b1", "b2", "b3"),
        feature_columns=columns,
        state_matrix=matrix,
        reward_mode="base_planning_reward",
        state_groups=("synthetic",),
        source_table=Path("variant_D4P8_features.csv"),
        tile_index_csv=Path("tiles.csv"),
    )
    transform = fit_phase65_train_tile_standardizer(raw)
    standardized = apply_phase65_standardizer(raw, transform)

    raw_oracle = build_phase63_oracle_trajectory(raw, eval_max_steps=2)
    examples = build_phase65_bc_examples(raw, transform, eval_max_steps=2)

    assert not np.allclose(standardized.state_matrix, raw.state_matrix)
    assert [example["target_action"] for example in examples] == raw_oracle["action_indices"]


def _reward_tiled_input(
    block_ids=("b3", "b1", "b2", "b4"),
    scores=(0.2, 0.9, 0.7, 0.1),
    scale_feature=(100.0, 200.0, 300.0, 400.0),
    variant_id="D4P8",
    tile_id="tile_eval",
):
    columns = (
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
    matrix = np.zeros((len(block_ids), len(columns)), dtype=np.float32)
    score_index = columns.index("explicit_feature_16")
    scale_index = columns.index("explicit_feature_00")
    for row_index, score in enumerate(scores):
        matrix[row_index, score_index] = float(score)
        matrix[row_index, scale_index] = float(scale_feature[row_index])
    from paper11_geofm.tiled_inputs import TiledVariantInput

    return TiledVariantInput(
        tile_id=tile_id,
        variant_id=variant_id,
        block_ids=tuple(block_ids),
        feature_columns=columns,
        state_matrix=matrix,
        reward_mode="base_planning_reward",
        state_groups=("synthetic",),
        source_table=Path(f"variant_{variant_id}_features.csv"),
        tile_index_csv=Path("tiles.csv"),
    )


def test_phase65_behavior_cloning_loss_decreases_with_standardized_inputs():
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        fit_phase65_train_tile_standardizer,
        train_phase65_behavior_cloner,
    )

    raw = _reward_tiled_input()
    transform = fit_phase65_train_tile_standardizer(raw)
    model, history = train_phase65_behavior_cloner(
        raw,
        transform,
        seed=65,
        eval_max_steps=3,
        epochs=30,
        learning_rate=0.01,
        hidden_dim=16,
        top_k=2,
    )

    assert model.n_features == len(raw.feature_columns)
    assert len(history) == 30
    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["topk_hit_rate"] >= history[0]["topk_hit_rate"]
    assert history[-1]["claim_boundary"].startswith("Phase 65")


def test_phase65_rollout_uses_standardized_logits_and_raw_rewards():
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        build_phase63_oracle_trajectory,
    )
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        fit_phase65_train_tile_standardizer,
        rollout_phase65_greedy_policy,
        train_phase65_behavior_cloner,
    )

    raw = _reward_tiled_input()
    transform = fit_phase65_train_tile_standardizer(raw)
    model, _history = train_phase65_behavior_cloner(
        raw,
        transform,
        seed=65,
        eval_max_steps=3,
        epochs=35,
        learning_rate=0.01,
        hidden_dim=16,
        top_k=2,
    )
    rollout = rollout_phase65_greedy_policy(
        model,
        raw_tiled_input=raw,
        standardizer=transform,
        train_tile_id="tile_train",
        eval_tile_rank=1,
        seed=65,
        phase65_seed_rank=1,
        eval_max_steps=3,
    )
    oracle = build_phase63_oracle_trajectory(raw, eval_max_steps=3)

    assert rollout["row_type"] == "bc_greedy_policy"
    assert rollout["all_actions_valid"] is True
    assert rollout["invalid_action_count"] == 0
    assert rollout["oracle_total_reward"] == oracle["total_oracle_reward"]
    assert float(rollout["total_contract_reward"]) > 0.0
    assert rollout["claim_boundary"].startswith("Phase 65")


def _rollout_row(variant_id, reward, tile_id="tile_a", seed=0, gap_fraction=0.1):
    return {
        "row_type": "bc_greedy_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": tile_id,
        "eval_tile_rank": 1,
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
        "oracle_total_reward": 1.5,
        "oracle_gap": 1.5 - reward,
        "oracle_gap_fraction": gap_fraction,
        "selected_block_ids": "b1;b2;b3",
        "selected_action_indices": "0;1;2",
        "claim_boundary": "fixture",
    }


def test_phase65_pairwise_delta_reports_standardized_minus_unstandardized():
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        build_phase65_standardization_pairwise_rows,
    )

    standardized = [_rollout_row("D4P8", 1.30), _rollout_row("B0", 1.10)]
    unstandardized = [_rollout_row("D4P8", 1.00), _rollout_row("B0", 1.20)]

    rows, coverage = build_phase65_standardization_pairwise_rows(
        standardized,
        unstandardized,
        variants=["B0", "D4P8"],
        eval_tile_ids=["tile_a"],
        seeds=[0],
    )

    assert coverage["missing_standardized_rows"] == []
    assert coverage["missing_unstandardized_rows"] == []
    d4 = [row for row in rows if row["variant_id"] == "D4P8"][0]
    assert d4["standardized_minus_unstandardized_reward"] == 0.3
    assert d4["self_improves_unstandardized"] is True


def test_phase65_status_gate_covers_supported_all_variant_not_helpful_and_insufficient():
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        build_phase65_standardization_comparison,
    )

    variants = ["B0", "D4P8", "D4P16", "D6R8", "D6R16"]
    old_rows = [_rollout_row(variant, 1.0) for variant in variants]
    geofm_rows = [
        _rollout_row("B0", 1.05),
        _rollout_row("D4P8", 1.40),
        _rollout_row("D4P16", 1.35),
        _rollout_row("D6R8", 1.20),
        _rollout_row("D6R16", 1.25),
    ]
    all_variant_rows = [
        _rollout_row("B0", 1.30),
        _rollout_row("D4P8", 1.10),
        _rollout_row("D4P16", 1.12),
        _rollout_row("D6R8", 1.18),
        _rollout_row("D6R16", 1.19),
    ]
    not_helpful_rows = [_rollout_row(variant, 0.90) for variant in variants]

    geofm = build_phase65_standardization_comparison(
        geofm_rows,
        old_rows,
        variants=variants,
        eval_tile_ids=["tile_a"],
        seeds=[0],
    )
    all_variant = build_phase65_standardization_comparison(
        all_variant_rows,
        old_rows,
        variants=variants,
        eval_tile_ids=["tile_a"],
        seeds=[0],
    )
    not_helpful = build_phase65_standardization_comparison(
        not_helpful_rows,
        old_rows,
        variants=variants,
        eval_tile_ids=["tile_a"],
        seeds=[0],
    )
    insufficient = build_phase65_standardization_comparison(
        geofm_rows[:-1],
        old_rows,
        variants=variants,
        eval_tile_ids=["tile_a"],
        seeds=[0],
    )

    assert geofm["phase65_status"] == "standardization_improves_geofm_set_policy"
    assert all_variant["phase65_status"] == "standardization_improves_all_variants_no_geofm_advantage"
    assert not_helpful["phase65_status"] == "standardization_not_helpful"
    assert insufficient["phase65_status"] == "insufficient"


def test_phase65_writer_outputs_json_csv_and_markdown(tmp_path):
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        build_phase65_standardization_comparison,
        write_phase65_artifacts,
    )

    variants = ["B0", "D4P8"]
    standardized_rows = [_rollout_row("B0", 1.1), _rollout_row("D4P8", 1.3)]
    unstandardized_rows = [_rollout_row("B0", 1.0), _rollout_row("D4P8", 1.0)]
    comparison = build_phase65_standardization_comparison(
        standardized_rows,
        unstandardized_rows,
        variants=variants,
        eval_tile_ids=["tile_a"],
        seeds=[0],
    )
    analysis = {
        "phase": "phase65_standardized_set_policy_bc_rerun",
        "standardization_stats": [
            {
                "variant_id": "D4P8",
                "train_tile_id": "tile_train",
                "zero_variance_feature_count": 0,
                "claim_boundary": "Phase 65",
            }
        ],
        "history_rows": [],
        "rollout_rows": standardized_rows,
        "phase63_style_analysis": {
            "mean_bc_reward_by_variant": {"B0": 1.1, "D4P8": 1.3},
            "oracle_gap_fraction_summary": {"mean_delta": 0.1},
        },
        "standardization_comparison": comparison,
        "claim_boundary": "Phase 65",
    }

    paths = write_phase65_artifacts(analysis, tmp_path / "outputs")

    assert paths["standardization_stats_json"].name == "phase65_standardization_stats.json"
    assert paths["history_csv"].name == "phase65_bc_training_history.csv"
    assert paths["rollout_csv"].name == "phase65_bc_rollout_summary.csv"
    assert paths["comparison_json"].name == "phase65_set_policy_comparison.json"
    assert paths["pairwise_delta_csv"].name == "phase65_standardization_pairwise_delta.csv"
    assert paths["readiness_md"].name == "phase65_standardized_set_policy_bc_rerun.md"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase65_status"] in {
        "standardization_improves_geofm_set_policy",
        "standardization_improves_all_variants_no_geofm_advantage",
        "standardization_not_helpful",
        "standardization_hurts_or_inconclusive",
        "insufficient",
    }
    markdown = paths["readiness_md"].read_text(encoding="utf-8")
    assert "Phase 65 Standardized Set-Policy BC Rerun" in markdown


def test_phase65_cli_parser_accepts_required_inputs():
    runner_path = (
        ROOT
        / "experiments"
        / "phase65_standardized_set_policy_bc_rerun"
        / "run_phase65_standardized_set_policy_bc_rerun.py"
    )
    spec = importlib.util.spec_from_file_location("phase65_runner_args", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--phase63-comparison-json",
            "phase63_set_policy_comparison.json",
            "--phase63-rollout-csv",
            "phase63_bc_rollout_summary.csv",
            "--existing-flattened-summary-csvs",
            "phase52.csv,phase62.csv",
            "--output-dir",
            "outputs",
        ]
    )

    assert args.phase63_comparison_json == Path("phase63_set_policy_comparison.json")
    assert args.phase63_rollout_csv == Path("phase63_bc_rollout_summary.csv")
    assert args.output_dir == Path("outputs")
