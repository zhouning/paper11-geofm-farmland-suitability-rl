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


def test_phase71_listwise_ranker_rollout_scores_original_reward_matrix():
    from paper11_geofm.phase71_component_supervised_ranker import (
        apply_phase71_fold_standardization,
        build_phase71_listwise_training_tile,
        fit_phase71_fold_standardization,
        rollout_phase71_ranker,
        train_phase71_component_ranker,
    )

    train_a = _tiled_input(
        np.array(
            [
                [5.0, 5.0, 7.0, 0.2, 0.4, 0.1, 0.0, 0.8, 0.9],
                [2.5, 10.0, 14.0, 0.1, 0.3, 0.0, 0.2, 0.5, 0.7],
                [1.0, 15.0, 21.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.2],
                [3.0, 20.0, 28.0, 0.4, 0.4, 0.0, 0.1, 0.9, 0.8],
            ],
            dtype=np.float32,
        ),
        tile_id="tile_train_a",
        block_ids=("b1", "b2", "b3", "b4"),
    )
    train_b = _tiled_input(
        np.array(
            [
                [5.0, 4.0, 6.0, 0.2, 0.5, 0.0, 0.0, 0.9, 0.95],
                [2.0, 9.0, 12.0, 0.1, 0.2, 0.1, 0.2, 0.6, 0.65],
                [1.0, 18.0, 30.0, 0.0, 0.0, 0.2, 0.4, 0.3, 0.1],
                [3.0, 21.0, 31.0, 0.4, 0.3, 0.0, 0.1, 0.8, 0.75],
            ],
            dtype=np.float32,
        ),
        tile_id="tile_train_b",
        block_ids=("c1", "c2", "c3", "c4"),
    )
    eval_tile = _tiled_input(
        train_a.state_matrix.copy(),
        tile_id="tile_eval",
        block_ids=("e1", "e2", "e3", "e4"),
    )
    params = fit_phase71_fold_standardization(
        [train_a, train_b],
        variant_id="B0",
        fold_id="tile_eval",
    )
    prepared_train = [
        apply_phase71_fold_standardization(train_a, params),
        apply_phase71_fold_standardization(train_b, params),
    ]
    prepared_eval = apply_phase71_fold_standardization(eval_tile, params)

    example = build_phase71_listwise_training_tile(prepared_train[0])
    assert example["reward_targets"][0] > example["reward_targets"][2]
    assert example["component_targets"].shape == (4, 8)

    model, history = train_phase71_component_ranker(
        prepared_train,
        seed=71,
        epochs=60,
        learning_rate=0.01,
        hidden_dim=24,
        component_weight=0.05,
        top_k=3,
    )
    rollout = rollout_phase71_ranker(
        model,
        prepared_eval,
        train_tile_ids=("tile_train_a", "tile_train_b"),
        eval_tile_rank=1,
        seed=71,
        phase71_seed_rank=1,
        eval_max_steps=3,
    )

    assert history[-1]["loss"] <= history[0]["loss"]
    assert rollout["row_type"] == "component_ranker_policy"
    assert rollout["phase71_component_supervised"] is True
    assert rollout["all_actions_valid"] is True
    assert rollout["invalid_action_count"] == 0
    assert rollout["selected_block_ids"].split(";")[0] == "e1"
    assert float(rollout["oracle_total_reward"]) >= float(
        rollout["total_contract_reward"]
    )
    assert float(rollout["total_contract_reward"]) > 0.0


def _phase71_rollout_row(variant_id, reward, oracle=2.0, tile_id="tile_a", seed=0):
    return {
        "row_type": "component_ranker_policy",
        "variant_id": variant_id,
        "train_tile_ids": "tile_train_a;tile_train_b",
        "eval_tile_id": tile_id,
        "eval_tile_rank": 1 if tile_id == "tile_a" else 2,
        "seed": seed,
        "phase71_seed_rank": seed + 1,
        "eval_max_steps": 3,
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
        "topk_oracle_overlap_count": 3,
        "topk_oracle_overlap_fraction": 1.0,
        "worst_selected_oracle_rank": 3,
        "selected_block_ids": "b1;b2;b3",
        "selected_action_indices": "0;1;2",
        "selected_model_scores": "3.0;2.0;1.0",
        "phase71_component_supervised": True,
        "claim_boundary": "fixture",
    }


def _reference_row(variant_id, reward, tile_id="tile_a", seed=0):
    return {
        "row_type": "bc_greedy_policy",
        "variant_id": variant_id,
        "eval_tile_id": tile_id,
        "seed": seed,
        "total_contract_reward": reward,
        "oracle_gap_fraction": 0.1,
    }


def test_phase71_comparison_statuses_and_writer_outputs(tmp_path):
    from paper11_geofm.phase71_component_supervised_ranker import (
        build_phase71_component_ranker_comparison,
        write_phase71_component_ranker_artifacts,
    )

    pairs = [("tile_a", 0), ("tile_a", 1), ("tile_b", 0), ("tile_b", 1)]
    phase63_rows = []
    phase70_rows = []
    target_masked_rows = []
    geofm_rows = []
    weak_rows = []
    for tile_id, seed in pairs:
        for variant_id in ("B0", "D4P8", "D4P16", "D6R8", "D6R16"):
            phase63_rows.append(
                _reference_row(variant_id, 1.00, tile_id=tile_id, seed=seed)
            )
            phase70_rows.append(
                _reference_row(variant_id, 1.05, tile_id=tile_id, seed=seed)
            )
        target_masked_rows.extend(
            [
                _phase71_rollout_row("B0", 1.40, tile_id=tile_id, seed=seed),
                _phase71_rollout_row("D4P8", 1.20, tile_id=tile_id, seed=seed),
                _phase71_rollout_row("D4P16", 1.22, tile_id=tile_id, seed=seed),
                _phase71_rollout_row("D6R8", 1.30, tile_id=tile_id, seed=seed),
                _phase71_rollout_row("D6R16", 1.31, tile_id=tile_id, seed=seed),
            ]
        )
        geofm_rows.extend(
            [
                _phase71_rollout_row("B0", 1.20, tile_id=tile_id, seed=seed),
                _phase71_rollout_row("D4P8", 1.45, tile_id=tile_id, seed=seed),
                _phase71_rollout_row("D4P16", 1.46, tile_id=tile_id, seed=seed),
                _phase71_rollout_row("D6R8", 1.30, tile_id=tile_id, seed=seed),
                _phase71_rollout_row("D6R16", 1.32, tile_id=tile_id, seed=seed),
            ]
        )
        weak_rows.extend(
            [
                _phase71_rollout_row("B0", 0.90, tile_id=tile_id, seed=seed),
                _phase71_rollout_row("D4P8", 0.88, tile_id=tile_id, seed=seed),
                _phase71_rollout_row("D4P16", 0.87, tile_id=tile_id, seed=seed),
                _phase71_rollout_row("D6R8", 0.89, tile_id=tile_id, seed=seed),
                _phase71_rollout_row("D6R16", 0.90, tile_id=tile_id, seed=seed),
            ]
        )
    metadata = {
        "variants": ["B0", "D4P8", "D4P16", "D6R8", "D6R16"],
        "eval_tile_ids": ["tile_a", "tile_b"],
        "seeds": [0, 1],
    }
    target_masked = build_phase71_component_ranker_comparison(
        target_masked_rows,
        phase63_rows,
        phase70_rows,
        metadata=metadata,
    )
    geofm = build_phase71_component_ranker_comparison(
        geofm_rows,
        phase63_rows,
        phase70_rows,
        metadata=metadata,
    )
    weak = build_phase71_component_ranker_comparison(
        weak_rows,
        phase63_rows,
        phase70_rows,
        metadata=metadata,
    )
    incomplete = build_phase71_component_ranker_comparison(
        target_masked_rows[:-1],
        phase63_rows,
        phase70_rows,
        metadata=metadata,
    )

    assert target_masked["phase71_status"] == "ranker_improves_but_target_masks_geofm"
    assert target_masked["phase71_minus_phase63_summary"]["mean_delta"] > 0.0
    assert target_masked["phase71_minus_phase70_summary"]["mean_delta"] > 0.0
    assert target_masked["d4_b0_delta_summary"]["mean_delta"] < 0.0
    assert geofm["phase71_status"] == "ranker_supports_geofm_followup"
    assert weak["phase71_status"] == "ranker_not_sufficient"
    assert incomplete["phase71_status"] == "ranker_incomplete"

    paths = write_phase71_component_ranker_artifacts(
        {
            **target_masked,
            "history_rows": [],
            "rollout_rows": target_masked_rows,
            "oracle_summary_rows": [],
            "component_diagnostic_rows": [],
        },
        tmp_path / "outputs",
    )
    assert paths["comparison_json"].name == "phase71_component_supervised_ranker.json"
    assert paths["rollout_csv"].name == "phase71_ranker_rollout_summary.csv"
    assert "ranker_improves_but_target_masks_geofm" in paths[
        "readiness_md"
    ].read_text(encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_variant_fixture(
    output_dir: Path,
    variant_id: str,
    rows: list[dict[str, object]],
    columns: tuple[str, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table = output_dir / f"variant_{variant_id}_features.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=["block_id", *columns])
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "variants": {
            variant_id: {
                "ready": True,
                "feature_table": table.name,
                "required_columns": list(columns),
                "reward": "base_planning_reward",
                "state_groups": ["synthetic"],
            }
        }
    }
    (output_dir / "experiment_variants.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_phase71_run_wrapper_and_cli_succeed_on_fixture(tmp_path):
    from paper11_geofm.phase71_component_supervised_ranker import (
        run_phase71_component_supervised_ranker,
    )

    columns = _required_feature_columns()
    blocks = ("b1", "b2", "b3", "b4")
    rows = [
        {**{"block_id": block_id}, **{column: 0.0 for column in columns}}
        for block_id in blocks
    ]
    for row, value in zip(rows, (0.90, 0.70, 0.20, 0.10)):
        row["explicit_feature_00"] = 5.0
        row["explicit_feature_13"] = value
        row["explicit_feature_16"] = value
    phase2 = tmp_path / "phase2"
    phase8 = tmp_path / "phase8"
    phase61 = tmp_path / "phase61"
    _write_variant_fixture(phase2, "B0", rows, columns)
    _write_variant_fixture(phase8, "D4P8", rows, columns)
    _write_variant_fixture(phase61, "D6R8", rows, columns)
    tile_index = _write_csv(
        tmp_path / "tiles.csv",
        [
            {"tile_id": "tile_train", "block_ids": ";".join(blocks)},
            {"tile_id": "tile_eval_a", "block_ids": ";".join(blocks)},
        ],
    )
    phase63_csv = _write_csv(
        tmp_path / "phase63_rollout.csv",
        [
            _reference_row("B0", 0.50, tile_id="tile_eval_a", seed=0),
            _reference_row("D4P8", 0.45, tile_id="tile_eval_a", seed=0),
            _reference_row("D6R8", 0.48, tile_id="tile_eval_a", seed=0),
        ],
    )
    phase70_csv = _write_csv(
        tmp_path / "phase70_rollout.csv",
        [
            _reference_row("B0", 0.55, tile_id="tile_eval_a", seed=0),
            _reference_row("D4P8", 0.50, tile_id="tile_eval_a", seed=0),
            _reference_row("D6R8", 0.51, tile_id="tile_eval_a", seed=0),
        ],
    )

    analysis = run_phase71_component_supervised_ranker(
        phase2_output_dir=phase2,
        phase8_output_dir=phase8,
        phase61_output_dir=phase61,
        tile_index_csv=tile_index,
        phase63_rollout_csv=phase63_csv,
        phase70_rollout_csv=phase70_csv,
        variants="B0,D4P8,D6R8",
        train_tile_id="tile_train",
        eval_tile_ids="tile_eval_a",
        seeds="0",
        eval_max_steps=3,
        ranker_epochs=8,
        learning_rate=0.01,
        hidden_dim=12,
        component_weight=0.05,
        top_k=2,
    )

    assert analysis["phase"] == "phase71_component_supervised_ranker"
    assert len(analysis["rollout_rows"]) == 3
    assert len(analysis["history_rows"]) == 24

    runner_path = (
        ROOT
        / "experiments"
        / "phase71_component_supervised_ranker"
        / "run_phase71_component_supervised_ranker.py"
    )
    spec = importlib.util.spec_from_file_location("phase71_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    exit_code = module.main(
        [
            "--phase2-output-dir",
            str(phase2),
            "--phase8-output-dir",
            str(phase8),
            "--phase61-output-dir",
            str(phase61),
            "--tile-index-csv",
            str(tile_index),
            "--phase63-rollout-csv",
            str(phase63_csv),
            "--phase70-rollout-csv",
            str(phase70_csv),
            "--variants",
            "B0,D4P8,D6R8",
            "--train-tile-id",
            "tile_train",
            "--eval-tile-ids",
            "tile_eval_a",
            "--seeds",
            "0",
            "--eval-max-steps",
            "3",
            "--ranker-epochs",
            "8",
            "--learning-rate",
            "0.01",
            "--hidden-dim",
            "12",
            "--component-weight",
            "0.05",
            "--top-k",
            "2",
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "outputs" / "phase71_component_supervised_ranker.json").exists()
