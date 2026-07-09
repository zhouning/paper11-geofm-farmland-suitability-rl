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
