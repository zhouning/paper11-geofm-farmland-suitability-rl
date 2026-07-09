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
    tile_id: str = "tile_train",
    variant_id: str = "D4P8",
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


def test_phase70_standardization_uses_train_tile_only_and_safe_scales():
    from paper11_geofm.phase70_standardized_set_policy_rerun import (
        apply_phase70_standardization,
        fit_phase70_standardization,
    )

    train = _tiled_input(
        np.array(
            [
                [1.0, 10.0, 5.0],
                [3.0, 10.0, 9.0],
                [5.0, 10.0, 13.0],
            ],
            dtype=np.float32,
        )
    )
    eval_tile = _tiled_input(
        np.array(
            [
                [101.0, 99.0, 105.0],
                [103.0, 99.0, 109.0],
            ],
            dtype=np.float32,
        ),
        tile_id="tile_eval",
    )

    params = fit_phase70_standardization(train)
    transformed_train = apply_phase70_standardization(train, params)
    transformed_eval = apply_phase70_standardization(eval_tile, params)

    assert params["variant_id"] == "D4P8"
    assert params["tile_id"] == "tile_train"
    assert params["feature_columns"] == list(train.feature_columns)
    assert params["means"] == [3.0, 10.0, 9.0]
    assert params["scales"][1] == 1.0
    np.testing.assert_allclose(
        transformed_train.model_matrix.mean(axis=0),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        transformed_eval.model_matrix[:, 0],
        (eval_tile.state_matrix[:, 0] - 3.0) / params["scales"][0],
    )
    assert transformed_eval.reward_matrix[0, 0] == 101.0
    assert transformed_eval.tiled_input.block_ids == eval_tile.block_ids
    assert transformed_eval.tiled_input.feature_columns == eval_tile.feature_columns
    assert transformed_eval.tiled_input.reward_mode == "base_planning_reward"