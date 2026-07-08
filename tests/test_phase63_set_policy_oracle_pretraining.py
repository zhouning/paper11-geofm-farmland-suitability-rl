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
