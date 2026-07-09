import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _reward_columns() -> tuple[str, ...]:
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
    matrix=None,
    columns=None,
    variant_id="D4P8",
    tile_id="tile_eval",
):
    from paper11_geofm.tiled_inputs import TiledVariantInput

    feature_columns = tuple(columns or _reward_columns())
    values = np.asarray(
        matrix
        if matrix is not None
        else [
            [5.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.8, 0.9],
            [4.0, 5.0, 7.0, 0.2, 0.6, 0.0, 0.0, 0.7, 0.8],
            [3.0, 15.0, 20.0, 0.0, 0.1, 0.2, 0.0, 0.2, 0.3],
            [1.0, 25.0, 35.0, 0.0, 0.0, 0.5, 0.4, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    return TiledVariantInput(
        tile_id=tile_id,
        variant_id=variant_id,
        block_ids=tuple(block_ids),
        feature_columns=feature_columns,
        state_matrix=values,
        reward_mode="base_planning_reward",
        state_groups=("synthetic",),
        source_table=Path(f"variant_{variant_id}_features.csv"),
        tile_index_csv=Path("tiles.csv"),
    )


def test_phase66_reward_components_sum_to_base_reward():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        decompose_phase66_base_reward_components,
    )
    from paper11_geofm.planning_reward import (
        compute_base_planning_reward_from_matrix_row,
    )

    tiled = _tiled_input()
    row = decompose_phase66_base_reward_components(
        tiled.feature_columns,
        tiled.state_matrix[0],
    )
    expected = compute_base_planning_reward_from_matrix_row(
        tiled.feature_columns,
        tiled.state_matrix[0],
    )

    assert row["low_slope_farmland_or_orchard_component"] == 0.315
    assert row["current_farmland_or_orchard_component"] == 0.08
    assert row["low_slope_component"] == 0.08
    assert row["area_component"] == 0.1
    assert row["mean_slope_penalty_component"] == -0.0
    assert row["max_slope_penalty_component"] == -0.0
    assert row["built_up_penalty_component"] == -0.0
    assert row["water_penalty_component"] == -0.0
    assert row["total_reward"] == expected


def test_phase66_reward_components_reject_missing_required_columns():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        decompose_phase66_base_reward_components,
    )

    try:
        decompose_phase66_base_reward_components(
            ("explicit_feature_00", "explicit_feature_16"),
            [1.0, 0.9],
        )
    except ValueError as exc:
        assert "explicit feature columns" in str(exc)
        assert "explicit_feature_01" in str(exc)
    else:
        raise AssertionError("Expected missing base-reward columns to fail")


def test_phase66_block_reward_table_ranks_blocks_by_reward_descending():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        build_phase66_block_reward_table,
    )

    table = build_phase66_block_reward_table(_tiled_input())

    assert [row["block_id"] for row in table] == ["b1", "b2", "b3", "b4"]
    assert [row["reward_rank"] for row in table] == [1, 2, 3, 4]
    assert table[0]["total_reward"] > table[1]["total_reward"]
    assert table[0]["variant_id"] == "D4P8"
    assert table[0]["tile_id"] == "tile_eval"


def _rollout_row(
    variant_id="D4P8",
    eval_tile_id="tile_eval",
    seed=0,
    selected="b1;b3",
    reward=0.0,
    oracle=0.0,
    gap=0.0,
    gap_fraction=0.0,
):
    return {
        "row_type": "bc_greedy_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": eval_tile_id,
        "eval_tile_rank": 1,
        "seed": seed,
        "phase63_seed_rank": seed + 1,
        "eval_max_steps": 2,
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
        "claim_boundary": "fixture",
    }


def _oracle_row(
    variant_id="D4P8",
    tile_id="tile_eval",
    seed=0,
    selected="b1;b2",
    oracle=0.0,
):
    return {
        "variant_id": variant_id,
        "tile_role": "eval",
        "tile_id": tile_id,
        "seed": seed,
        "eval_max_steps": 2,
        "n_blocks": 4,
        "n_features": 9,
        "episode_steps": 2,
        "terminated": False,
        "total_oracle_reward": oracle,
        "top_k_reward_ceiling": oracle,
        "selected_block_ids": selected,
        "action_indices": "0;1",
        "claim_boundary": "fixture",
    }


def test_phase66_selected_block_atlas_reports_overlap_ranks_and_equivalence():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        build_phase66_selected_block_atlas,
    )

    tiled = _tiled_input(
        block_ids=("b1", "b2", "b3", "b4"),
        matrix=[
            [5.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.8, 0.90],
            [5.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.8, 0.86],
            [5.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.8, 0.84],
            [1.0, 25.0, 35.0, 0.0, 0.0, 0.5, 0.4, 0.1, 0.10],
        ],
    )

    rows = build_phase66_selected_block_atlas(
        phase63_rollout_rows=[_rollout_row(selected="b1;b3")],
        phase65_rollout_rows=[_rollout_row(selected="b1;b2")],
        oracle_rows=[_oracle_row(selected="b1;b2")],
        tiled_inputs={("D4P8", "tile_eval"): tiled},
        reward_tolerance=0.02,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["phase63_oracle_overlap_count"] == 1
    assert row["phase65_oracle_overlap_count"] == 2
    assert row["phase63_oracle_jaccard"] == 0.3333333333
    assert row["phase65_oracle_jaccard"] == 1.0
    assert row["phase63_selected_rank_values"] == "1;3"
    assert row["phase65_selected_rank_values"] == "1;2"
    assert row["phase63_reward_equivalent_substitution"] is True
    assert row["phase65_reward_equivalent_substitution"] is True
    assert row["phase63_extra_selected_block_ids"] == "b3"
    assert row["phase63_missed_oracle_block_ids"] == "b2"


def test_phase66_selected_block_atlas_rejects_missing_rollout_rows():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        build_phase66_selected_block_atlas,
    )

    try:
        build_phase66_selected_block_atlas(
            phase63_rollout_rows=[],
            phase65_rollout_rows=[_rollout_row()],
            oracle_rows=[_oracle_row()],
            tiled_inputs={("D4P8", "tile_eval"): _tiled_input()},
        )
    except ValueError as exc:
        assert "missing Phase 63 rollout row" in str(exc)
    else:
        raise AssertionError("Expected missing Phase 63 row to fail")


def test_phase66_rank_metric_handles_ties_and_constant_columns():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        phase66_spearman_abs,
        phase66_topk_enrichment,
    )

    assert phase66_spearman_abs([1.0, 2.0, 2.0, 4.0], [0.1, 0.2, 0.2, 0.4]) == 1.0
    assert phase66_spearman_abs([1.0, 1.0, 1.0], [0.1, 0.2, 0.3]) == 0.0
    assert phase66_topk_enrichment([0.9, 0.8, 0.1, 0.0], [1.0, 0.7, 0.2, 0.1], top_k=2) == 1.0
    assert phase66_topk_enrichment([0.0, 0.1, 0.8, 0.9], [1.0, 0.7, 0.2, 0.1], top_k=2) == 1.0


def test_phase66_representation_rank_alignment_separates_explicit_and_extra_columns():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        build_phase66_representation_rank_alignment,
    )

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
        "embedding_pca_00",
        "embedding_pca_01",
    )
    matrix = np.asarray(
        [
            [5.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.8, 0.9, 0.9, 0.0],
            [4.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.7, 0.8, 0.8, 0.0],
            [3.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.2, 0.3, 0.3, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.1, 0.1, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    tiled = _tiled_input(columns=columns, matrix=matrix, variant_id="D4P8")

    rows = build_phase66_representation_rank_alignment(
        tiled_inputs={("D4P8", "tile_eval"): tiled},
        eval_max_steps=2,
    )
    by_group = {row["feature_group"]: row for row in rows}

    assert by_group["reward_explicit"]["n_columns"] == 9
    assert by_group["representation_extra"]["n_columns"] == 2
    assert by_group["representation_extra"]["max_abs_spearman"] == 1.0
    assert by_group["representation_extra"]["best_topk_enrichment"] == 1.0
    assert by_group["representation_extra"]["proxy_r2"] > 0.9


def test_phase66_representation_alignment_rejects_geofm_variant_without_extra_columns():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        build_phase66_representation_rank_alignment,
    )

    try:
        build_phase66_representation_rank_alignment(
            tiled_inputs={("D4P8", "tile_eval"): _tiled_input(variant_id="D4P8")},
            eval_max_steps=2,
        )
    except ValueError as exc:
        assert "representation columns" in str(exc)
    else:
        raise AssertionError("Expected D4/D6 without representation columns to fail")


def test_phase66_failure_modes_cover_reward_equivalent_component_miss_and_standardization_hurt():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        build_phase66_failure_mode_summary,
    )

    atlas_rows = [
        {
            "variant_id": "D4P8",
            "eval_tile_id": "tile_a",
            "seed": 0,
            "phase63_oracle_jaccard": 0.2,
            "phase65_oracle_jaccard": 0.9,
            "phase63_reward_equivalent_substitution": True,
            "phase65_reward_equivalent_substitution": True,
            "phase63_missed_oracle_block_ids": "b2",
            "phase63_extra_selected_block_ids": "b3",
        },
        {
            "variant_id": "D4P16",
            "eval_tile_id": "tile_b",
            "seed": 1,
            "phase63_oracle_jaccard": 0.1,
            "phase65_oracle_jaccard": 0.1,
            "phase63_reward_equivalent_substitution": False,
            "phase65_reward_equivalent_substitution": False,
            "phase63_missed_oracle_block_ids": "b2",
            "phase63_extra_selected_block_ids": "b4",
        },
    ]
    alignment_rows = [
        {"variant_id": "D4P8", "feature_group": "representation_extra", "proxy_r2": 0.10, "max_abs_spearman": 0.20},
        {"variant_id": "D4P8", "feature_group": "reward_explicit", "proxy_r2": 0.90, "max_abs_spearman": 0.95},
        {"variant_id": "D4P16", "feature_group": "representation_extra", "proxy_r2": 0.10, "max_abs_spearman": 0.20},
        {"variant_id": "D4P16", "feature_group": "reward_explicit", "proxy_r2": 0.90, "max_abs_spearman": 0.95},
    ]
    phase65_pairwise_rows = [
        {
            "variant_id": "D4P16",
            "eval_tile_id": "tile_b",
            "seed": 1,
            "standardized_minus_unstandardized_reward": -0.5,
        }
    ]

    rows = build_phase66_failure_mode_summary(
        atlas_rows,
        alignment_rows,
        phase65_pairwise_rows,
    )
    modes = {row["failure_mode"]: row for row in rows}

    assert modes["near_oracle_reward_equivalent"]["case_count"] == 1
    assert modes["misses_explicit_reward_components"]["case_count"] == 1
    assert modes["representation_not_aligned_with_base_reward"]["case_count"] == 2
    assert modes["standardization_hurts_rank_geometry"]["case_count"] == 1


def test_phase66_diagnostic_gate_covers_all_statuses():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        build_phase66_diagnostic_gate,
    )

    strong_alignment = [
        {"variant_id": "B0", "feature_group": "reward_explicit", "proxy_r2": 0.60, "max_abs_spearman": 0.60, "best_topk_enrichment": 0.50},
        {"variant_id": "D4P8", "feature_group": "reward_explicit", "proxy_r2": 0.60, "max_abs_spearman": 0.60, "best_topk_enrichment": 0.50},
        {"variant_id": "D4P8", "feature_group": "representation_extra", "proxy_r2": 0.80, "max_abs_spearman": 0.80, "best_topk_enrichment": 0.75},
    ]
    redundant_alignment = [
        {"variant_id": "B0", "feature_group": "reward_explicit", "proxy_r2": 0.85, "max_abs_spearman": 0.90, "best_topk_enrichment": 1.00},
        {"variant_id": "D4P8", "feature_group": "reward_explicit", "proxy_r2": 0.86, "max_abs_spearman": 0.91, "best_topk_enrichment": 1.00},
        {"variant_id": "D4P8", "feature_group": "representation_extra", "proxy_r2": 0.84, "max_abs_spearman": 0.89, "best_topk_enrichment": 1.00},
    ]
    failure_summary = [
        {"failure_mode": "misses_explicit_reward_components", "case_count": 5},
        {"failure_mode": "representation_not_aligned_with_base_reward", "case_count": 5},
    ]

    assert build_phase66_diagnostic_gate([], strong_alignment, [], {})["phase66_status"] == "representation_adds_reward_ranking_signal"
    assert build_phase66_diagnostic_gate([], redundant_alignment, [], {})["phase66_status"] == "representation_signal_redundant_with_explicit_reward"
    assert build_phase66_diagnostic_gate([], redundant_alignment, failure_summary, {"phase10_status": "not_ready_for_suitability_reward"})["phase66_status"] == "base_reward_target_masks_geofm_signal"
    assert build_phase66_diagnostic_gate(["missing row"], redundant_alignment, failure_summary, {})["phase66_status"] == "insufficient"
