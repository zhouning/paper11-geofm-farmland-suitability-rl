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


def _history_row(
    variant_id="B0",
    seed=0,
    epoch=1,
    loss=1.0,
    top1=0.0,
    topk=0.0,
    train_tile_id="tile_train",
):
    return {
        "variant_id": variant_id,
        "train_tile_id": train_tile_id,
        "seed": seed,
        "epoch": epoch,
        "loss": loss,
        "top1_accuracy": top1,
        "topk_hit_rate": topk,
        "learning_rate": 0.001,
        "hidden_dim": 64,
        "claim_boundary": "phase63",
    }


def test_phase64_splits_semicolon_values_and_summarizes_convergence():
    from paper11_geofm.phase64_set_policy_error_diagnosis import (
        _split_semicolon_values,
        build_phase64_convergence_summary,
    )

    assert _split_semicolon_values(" b2 ; b1;;b3 ") == ["b2", "b1", "b3"]
    assert _split_semicolon_values("") == []

    history_rows = [
        _history_row("B0", 0, 1, loss=4.0, top1=0.0, topk=0.0),
        _history_row("B0", 0, 2, loss=2.0, top1=0.25, topk=0.50),
        _history_row("B0", 0, 3, loss=2.5, top1=0.20, topk=0.75),
        _history_row("D4P8", 1, 1, loss=3.0, top1=0.10, topk=0.25),
        _history_row("D4P8", 1, 2, loss=1.5, top1=0.40, topk=0.50),
    ]

    summary = build_phase64_convergence_summary(history_rows)

    assert len(summary) == 2
    b0 = summary[0]
    assert b0["variant_id"] == "B0"
    assert b0["seed"] == 0
    assert b0["first_epoch"] == 1
    assert b0["final_epoch"] == 3
    assert b0["best_epoch"] == 2
    assert b0["first_loss"] == 4.0
    assert b0["final_loss"] == 2.5
    assert b0["best_loss"] == 2.0
    assert b0["final_top1_accuracy"] == 0.2
    assert b0["best_top1_accuracy"] == 0.25
    assert b0["final_topk_hit_rate"] == 0.75
    assert b0["best_topk_hit_rate"] == 0.75
    assert b0["loss_delta"] == -1.5


def _rollout_row(
    variant_id="B0",
    eval_tile_id="tile_eval",
    seed=0,
    selected="b1;b3",
    reward=1.0,
    oracle=1.5,
    gap=0.5,
    gap_fraction=0.3333333333,
    eval_max_steps=3,
):
    return {
        "row_type": "bc_greedy_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": eval_tile_id,
        "eval_tile_rank": 1,
        "seed": seed,
        "phase63_seed_rank": 1,
        "eval_max_steps": eval_max_steps,
        "n_blocks": 4,
        "n_features": 9,
        "episode_steps": 2,
        "terminated": False,
        "truncated": True,
        "all_actions_valid": True,
        "invalid_action_count": 0,
        "total_contract_reward": reward,
        "oracle_total_reward": oracle,
        "oracle_gap": gap,
        "oracle_gap_fraction": gap_fraction,
        "selected_block_ids": selected,
        "selected_action_indices": "0;2",
        "claim_boundary": "phase63",
    }


def _oracle_row(
    variant_id="B0",
    tile_id="tile_eval",
    seed=0,
    selected="b1;b2;b4",
    action_indices="0;1;3",
    eval_max_steps=3,
    oracle=1.5,
):
    return {
        "variant_id": variant_id,
        "tile_role": "eval",
        "tile_id": tile_id,
        "seed": seed,
        "eval_max_steps": eval_max_steps,
        "n_blocks": 4,
        "n_features": 9,
        "episode_steps": 3,
        "terminated": False,
        "total_oracle_reward": oracle,
        "top_k_reward_ceiling": oracle,
        "selected_block_ids": selected,
        "action_indices": action_indices,
        "claim_boundary": "phase63",
    }


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
    block_ids=("b1", "b2", "b3", "b4"),
    scores=(0.9, 0.8, 0.2, 0.7),
    variant_id="B0",
    tile_id="tile_eval",
):
    from paper11_geofm.tiled_inputs import TiledVariantInput

    columns = _required_feature_columns()
    matrix = np.zeros((len(block_ids), len(columns)), dtype=np.float32)
    score_index = columns.index("explicit_feature_16")
    for row_index, score in enumerate(scores):
        matrix[row_index, score_index] = float(score)
    return TiledVariantInput(
        tile_id=tile_id,
        variant_id=variant_id,
        block_ids=tuple(block_ids),
        feature_columns=columns,
        state_matrix=matrix,
        reward_mode="base_planning_reward",
        state_groups=("explicit_planning_features",),
        source_table=Path(f"variant_{variant_id}_features.csv"),
        tile_index_csv=Path("tiles.csv"),
    )


def test_phase64_rollout_overlap_tracks_prefix_jaccard_and_missed_oracle():
    from paper11_geofm.phase64_set_policy_error_diagnosis import (
        build_phase64_rollout_overlap,
    )

    rows = build_phase64_rollout_overlap(
        [_rollout_row(selected="b1;b3;b3")],
        [_oracle_row(selected="b1;b2;b4")],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["selected_overlap_count"] == 1
    assert row["selected_overlap_fraction"] == 0.3333333333
    assert row["prefix_overlap_count"] == 1
    assert row["jaccard_similarity"] == 0.25
    assert row["duplicate_selection_count"] == 1
    assert row["missed_oracle_block_ids"] == "b2;b4"
    assert row["extra_selected_block_ids"] == "b3"


def test_phase64_oracle_rank_gap_reports_missed_blocks_and_rank_losses():
    from paper11_geofm.phase64_set_policy_error_diagnosis import (
        build_phase64_oracle_rank_gap,
    )

    tiled = _tiled_input()
    rows = build_phase64_oracle_rank_gap(
        [_rollout_row(selected="b1;b3", eval_max_steps=3)],
        {("B0", "tile_eval"): tiled},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["selected_rank_values"] == "1;4"
    assert row["missed_oracle_block_ids"] == "b2;b4"
    assert row["worst_selected_rank"] == 4
    assert row["selected_outside_top_eval_max_steps"] == 1
    assert row["selected_outside_top16"] == 0
    assert row["selected_outside_top32"] == 0
    assert row["reward_loss_from_missed_oracle"] > 0.0


def _matrix_tiled_input(matrix, variant_id="D4P8", tile_id="tile_train"):
    from paper11_geofm.tiled_inputs import TiledVariantInput

    feature_columns = tuple(f"feature_{index:02d}" for index in range(np.asarray(matrix).shape[1]))
    block_ids = tuple(f"b{index}" for index in range(np.asarray(matrix).shape[0]))
    return TiledVariantInput(
        tile_id=tile_id,
        variant_id=variant_id,
        block_ids=block_ids,
        feature_columns=feature_columns,
        state_matrix=np.asarray(matrix, dtype=np.float32),
        reward_mode="base_planning_reward",
        state_groups=("synthetic",),
        source_table=Path(f"variant_{variant_id}_features.csv"),
        tile_index_csv=Path("tiles.csv"),
    )


def test_phase64_feature_diagnostics_detect_scale_shift_and_low_rank():
    from paper11_geofm.phase64_set_policy_error_diagnosis import (
        build_phase64_feature_diagnostics,
    )

    train = _matrix_tiled_input(
        [
            [1.0, 10.0, 0.0],
            [2.0, 20.0, 0.0],
            [3.0, 30.0, 0.0],
            [4.0, 40.0, 0.0],
        ],
        variant_id="D4P8",
        tile_id="tile_train",
    )
    eval_tile = _matrix_tiled_input(
        [
            [11.0, 100.0, 0.0],
            [12.0, 120.0, 0.0],
        ],
        variant_id="D4P8",
        tile_id="tile_eval",
    )

    diagnostics = build_phase64_feature_diagnostics(
        [("train", train), ("eval", eval_tile)],
        {"D4P8": "tile_train"},
    )

    feature_rows = diagnostics["feature_scale_rows"]
    rank_rows = diagnostics["feature_effective_rank_rows"]
    assert len(feature_rows) == 6
    train_feature0 = [
        row for row in feature_rows
        if row["tile_role"] == "train" and row["feature_name"] == "feature_00"
    ][0]
    assert train_feature0["mean"] == 2.5
    assert train_feature0["zero_variance"] is False
    eval_feature0 = [
        row for row in feature_rows
        if row["tile_role"] == "eval" and row["feature_name"] == "feature_00"
    ][0]
    assert eval_feature0["eval_mean_z_shift"] > 3.0

    eval_rank = [row for row in rank_rows if row["tile_role"] == "eval"][0]
    assert eval_rank["zero_variance_feature_count"] == 1
    assert eval_rank["rank_flag"] is True
    assert eval_rank["shift_flag"] is True


def _comparison(
    d4_b0_mean=-0.1,
    d4_d6_mean=-0.05,
    missing=None,
    duplicate=None,
    unexpected=None,
):
    return {
        "coverage_issues": {
            "missing_rollout_rows": [] if missing is None else missing,
            "duplicate_rollout_rows": [] if duplicate is None else duplicate,
            "unexpected_rollout_rows": [] if unexpected is None else unexpected,
        },
        "d4_b0_delta_summary": {"mean_delta": d4_b0_mean, "positive_count": 0, "total_count": 4},
        "d4_d6_delta_summary": {"mean_delta": d4_d6_mean, "positive_count": 0, "total_count": 4},
        "oracle_gap_fraction_summary": {"mean_delta": 0.08, "positive_count": 4, "total_count": 4},
        "d4_b0_delta_rows": [
            {
                "left_variant_id": "D4P8",
                "right_variant_id": "B0",
                "eval_tile_id": "tile_eval",
                "seed": 0,
                "left_minus_right_reward": -0.2,
            }
        ],
        "d4_d6_delta_rows": [],
    }


def test_phase64_standardization_gate_reports_supported_capacity_not_helpful_and_inconclusive():
    from paper11_geofm.phase64_set_policy_error_diagnosis import (
        build_phase64_standardization_gate,
    )

    strong_convergence = [
        {"variant_id": "B0", "best_top1_accuracy": 0.6, "best_topk_hit_rate": 0.9},
        {"variant_id": "D4P8", "best_top1_accuracy": 0.5, "best_topk_hit_rate": 0.8},
    ]
    weak_convergence = [
        {"variant_id": "B0", "best_top1_accuracy": 0.1, "best_topk_hit_rate": 0.2},
        {"variant_id": "D4P8", "best_top1_accuracy": 0.1, "best_topk_hit_rate": 0.2},
    ]
    flagged_rank = [
        {
            "variant_id": "D4P8",
            "scale_flag": True,
            "shift_flag": False,
            "rank_flag": False,
            "tile_role": "eval",
        }
    ]
    clean_rank = [
        {
            "variant_id": "D4P8",
            "scale_flag": False,
            "shift_flag": False,
            "rank_flag": False,
            "tile_role": "eval",
        }
    ]

    supported = build_phase64_standardization_gate(
        _comparison(),
        strong_convergence,
        flagged_rank,
    )
    assert supported["phase64_status"] == "standardization_route_supported"
    assert supported["recommend_standardized_rerun"] is True

    capacity = build_phase64_standardization_gate(
        _comparison(),
        weak_convergence,
        flagged_rank,
    )
    assert capacity["phase64_status"] == "bc_training_capacity_limited"

    not_helpful = build_phase64_standardization_gate(
        _comparison(),
        strong_convergence,
        clean_rank,
    )
    assert not_helpful["phase64_status"] == "geofm_features_not_helpful_under_set_policy"

    inconclusive = build_phase64_standardization_gate(
        _comparison(missing=["B0:tile_eval:0"]),
        strong_convergence,
        flagged_rank,
    )
    assert inconclusive["phase64_status"] == "diagnostic_inconclusive"


def test_phase64_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase64_set_policy_error_diagnosis import (
        build_phase64_failure_cases,
        build_phase64_standardization_gate,
        write_phase64_artifacts,
    )

    convergence = [
        {
            "variant_id": "D4P8",
            "train_tile_id": "tile_train",
            "seed": 0,
            "best_top1_accuracy": 0.5,
            "best_topk_hit_rate": 0.75,
            "final_loss": 1.0,
            "claim_boundary": "phase64",
        }
    ]
    overlap = [
        {
            "variant_id": "D4P8",
            "eval_tile_id": "tile_eval",
            "seed": 0,
            "oracle_gap_fraction": 0.4,
            "selected_overlap_fraction": 0.25,
            "missed_oracle_block_ids": "b2",
            "selected_block_ids": "b1",
        }
    ]
    rank_gap = [
        {
            "variant_id": "D4P8",
            "eval_tile_id": "tile_eval",
            "seed": 0,
            "reward_loss_from_missed_oracle": 0.2,
            "worst_selected_rank": 4,
        }
    ]
    effective_rank = [
        {
            "variant_id": "D4P8",
            "tile_role": "eval",
            "tile_id": "tile_eval",
            "scale_flag": True,
            "shift_flag": False,
            "rank_flag": False,
        }
    ]
    gate = build_phase64_standardization_gate(_comparison(), convergence, effective_rank)
    failure_cases = build_phase64_failure_cases(
        _comparison(),
        overlap,
        rank_gap,
        convergence,
        effective_rank,
        limit=3,
    )
    analysis = {
        "phase": "phase64_set_policy_error_diagnosis",
        "convergence_rows": convergence,
        "overlap_rows": overlap,
        "oracle_rank_gap_rows": rank_gap,
        "feature_scale_rows": [],
        "feature_effective_rank_rows": effective_rank,
        "failure_case_rows": failure_cases,
        "standardization_gate": gate,
        "claim_boundary": "phase64",
    }

    paths = write_phase64_artifacts(analysis, tmp_path / "outputs")

    assert paths["convergence_csv"].name == "phase64_convergence_summary.csv"
    assert paths["overlap_csv"].name == "phase64_rollout_overlap.csv"
    assert paths["oracle_rank_csv"].name == "phase64_oracle_rank_gap.csv"
    assert paths["gate_json"].name == "phase64_standardization_gate.json"
    saved = json.loads(paths["gate_json"].read_text(encoding="utf-8"))
    assert saved["phase64_status"] == "standardization_route_supported"
    markdown = paths["diagnosis_md"].read_text(encoding="utf-8")
    assert "Phase 64 Set-Policy Error Diagnosis" in markdown
    assert "standardization_route_supported" in markdown
