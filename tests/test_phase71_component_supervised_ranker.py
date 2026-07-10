import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


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
    matrix: np.ndarray,
    *,
    tile_id: str = "tile_train_a",
    variant_id: str = "B0",
    block_ids: tuple[str, ...] | None = None,
):
    from paper11_geofm.tiled_inputs import TiledVariantInput

    if block_ids is None:
        block_ids = tuple(f"b{index}" for index in range(matrix.shape[0]))
    return TiledVariantInput(
        tile_id=tile_id,
        variant_id=variant_id,
        block_ids=tuple(block_ids),
        feature_columns=_required_feature_columns()[: matrix.shape[1]],
        state_matrix=matrix.astype(np.float32, copy=True),
        reward_mode="base_planning_reward",
        state_groups=("synthetic",),
        source_table=Path(f"variant_{variant_id}_features.csv"),
        tile_index_csv=Path("tiles.csv"),
        claim_boundary="fixture boundary",
    )


def test_phase71_reward_components_sum_to_base_reward_and_fold_standardization_is_train_only():
    from paper11_geofm.planning_reward import compute_base_planning_reward_from_matrix_row
    from paper11_geofm.phase71_component_supervised_ranker import (
        apply_phase71_fold_standardization,
        build_phase71_component_targets,
        fit_phase71_fold_standardization,
    )

    train_a = _tiled_input(
        np.array(
            [
                [5.0, 5.0, 7.0, 0.2, 0.4, 0.1, 0.0, 0.8, 0.9],
                [2.5, 10.0, 14.0, 0.1, 0.3, 0.0, 0.2, 0.5, 0.7],
            ],
            dtype=np.float32,
        ),
        tile_id="tile_train_a",
        block_ids=("b1", "b2"),
    )
    train_b = _tiled_input(
        np.array(
            [
                [1.0, 15.0, 21.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.2],
                [3.0, 20.0, 28.0, 0.4, 0.4, 0.0, 0.1, 0.9, 0.8],
            ],
            dtype=np.float32,
        ),
        tile_id="tile_train_b",
        block_ids=("b3", "b4"),
    )
    eval_tile = _tiled_input(
        np.array(
            [[101.0, 99.0, 105.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]],
            dtype=np.float32,
        ),
        tile_id="tile_eval",
        block_ids=("be",),
    )

    targets = build_phase71_component_targets(train_a)
    first = targets[0]
    expected_total = compute_base_planning_reward_from_matrix_row(
        train_a.feature_columns,
        train_a.state_matrix[0],
    )
    assert first["block_id"] == "b1"
    assert first["reward_total"] == expected_total
    assert round(sum(first["components"].values()), 10) == expected_total
    assert first["components"]["low_slope_farmland_or_orchard"] == 0.315
    assert first["components"]["mean_slope_penalty"] == -0.03

    params = fit_phase71_fold_standardization(
        [train_a, train_b],
        variant_id="B0",
        fold_id="tile_eval",
    )
    standardized_train = apply_phase71_fold_standardization(train_a, params)
    standardized_eval = apply_phase71_fold_standardization(eval_tile, params)
    stacked_train = np.vstack([train_a.state_matrix, train_b.state_matrix])

    assert params["variant_id"] == "B0"
    assert params["fold_id"] == "tile_eval"
    assert params["means"] == [
        round(float(value), 10) for value in stacked_train.mean(axis=0)
    ]
    assert standardized_eval.reward_matrix[0, 0] == 101.0
    np.testing.assert_allclose(
        standardized_train.model_matrix[0, 0],
        (train_a.state_matrix[0, 0] - stacked_train[:, 0].mean())
        / params["scales"][0],
        atol=1.0e-6,
    )
