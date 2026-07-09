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
