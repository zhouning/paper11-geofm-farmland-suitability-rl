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
