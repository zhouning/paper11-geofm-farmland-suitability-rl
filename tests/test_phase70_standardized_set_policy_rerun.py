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

def test_phase70_dual_matrix_rollout_scores_original_reward_matrix():
    from paper11_geofm.phase70_standardized_set_policy_rerun import (
        apply_phase70_standardization,
        build_phase70_oracle_trajectory,
        fit_phase70_standardization,
        rollout_phase70_standardized_greedy_policy,
        train_phase70_standardized_behavior_cloner,
    )

    train = _tiled_input(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
            ],
            dtype=np.float32,
        ),
        block_ids=("b1", "b2", "b3", "b4"),
    )
    params = fit_phase70_standardization(train)
    standardized_train = apply_phase70_standardization(train, params)

    oracle = build_phase70_oracle_trajectory(standardized_train, eval_max_steps=3)
    model, history = train_phase70_standardized_behavior_cloner(
        standardized_train,
        seed=70,
        eval_max_steps=3,
        epochs=35,
        learning_rate=0.01,
        hidden_dim=16,
        top_k=2,
    )
    rollout = rollout_phase70_standardized_greedy_policy(
        model,
        standardized_train,
        train_tile_id="tile_train",
        eval_tile_rank=1,
        seed=70,
        phase70_seed_rank=1,
        eval_max_steps=3,
    )

    assert oracle["selected_block_ids"] == ["b1", "b2", "b3"]
    assert oracle["total_oracle_reward"] == 0.63
    assert history[-1]["loss"] <= history[0]["loss"]
    assert rollout["row_type"] == "bc_greedy_policy"
    assert rollout["phase70_standardized_input"] is True
    assert rollout["all_actions_valid"] is True
    assert rollout["invalid_action_count"] == 0
    assert float(rollout["oracle_total_reward"]) == 0.63
    assert float(rollout["total_contract_reward"]) > 0.0

def _rollout_row(variant_id, reward, oracle=2.0, tile_id="tile_a", seed=0, boundary="fixture"):
    return {
        "row_type": "bc_greedy_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": tile_id,
        "eval_tile_rank": 1 if tile_id == "tile_a" else 2,
        "seed": seed,
        "phase70_seed_rank": seed + 1,
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
        "selected_block_ids": "b1;b2;b3",
        "selected_action_indices": "0;1;2",
        "claim_boundary": boundary,
    }


def test_phase70_comparison_statuses_distinguish_geofm_architecture_and_incomplete(tmp_path):
    from paper11_geofm.phase70_standardized_set_policy_rerun import (
        build_phase70_standardized_set_policy_comparison,
        write_phase70_standardized_set_policy_artifacts,
    )

    pairs = [("tile_a", 0), ("tile_a", 1), ("tile_b", 0), ("tile_b", 1)]
    baseline = []
    geofm_rows = []
    architecture_rows = []
    weak_rows = []
    for tile_id, seed in pairs:
        baseline.extend(
            [
                _rollout_row("B0", 1.10, tile_id=tile_id, seed=seed),
                _rollout_row("D4P8", 0.90, tile_id=tile_id, seed=seed),
                _rollout_row("D4P16", 0.95, tile_id=tile_id, seed=seed),
                _rollout_row("D6R8", 1.00, tile_id=tile_id, seed=seed),
                _rollout_row("D6R16", 1.02, tile_id=tile_id, seed=seed),
            ]
        )
        geofm_rows.extend(
            [
                _rollout_row("B0", 1.20, tile_id=tile_id, seed=seed),
                _rollout_row("D4P8", 1.35, tile_id=tile_id, seed=seed),
                _rollout_row("D4P16", 1.40, tile_id=tile_id, seed=seed),
                _rollout_row("D6R8", 1.25, tile_id=tile_id, seed=seed),
                _rollout_row("D6R16", 1.27, tile_id=tile_id, seed=seed),
            ]
        )
        architecture_rows.extend(
            [
                _rollout_row("B0", 1.40, tile_id=tile_id, seed=seed),
                _rollout_row("D4P8", 1.05, tile_id=tile_id, seed=seed),
                _rollout_row("D4P16", 1.06, tile_id=tile_id, seed=seed),
                _rollout_row("D6R8", 1.30, tile_id=tile_id, seed=seed),
                _rollout_row("D6R16", 1.31, tile_id=tile_id, seed=seed),
            ]
        )
        weak_rows.extend(
            [
                _rollout_row("B0", 1.0, tile_id=tile_id, seed=seed),
                _rollout_row("D4P8", 0.85, tile_id=tile_id, seed=seed),
                _rollout_row("D4P16", 0.86, tile_id=tile_id, seed=seed),
                _rollout_row("D6R8", 1.0, tile_id=tile_id, seed=seed),
                _rollout_row("D6R16", 1.0, tile_id=tile_id, seed=seed),
            ]
        )

    metadata = {
        "variants": ["B0", "D4P8", "D4P16", "D6R8", "D6R16"],
        "eval_tile_ids": ["tile_a", "tile_b"],
        "seeds": [0, 1],
    }
    geofm = build_phase70_standardized_set_policy_comparison(geofm_rows, baseline, metadata=metadata)
    architecture = build_phase70_standardized_set_policy_comparison(architecture_rows, baseline, metadata=metadata)
    weak = build_phase70_standardized_set_policy_comparison(weak_rows, baseline, metadata=metadata)
    incomplete = build_phase70_standardized_set_policy_comparison(geofm_rows[:-1], baseline, metadata=metadata)

    assert geofm["phase70_status"] == "standardization_improves_geofm_set_policy_route"
    assert geofm["d4_b0_delta_summary"]["mean_delta"] > 0.0
    assert geofm["d4_d6_delta_summary"]["mean_delta"] > 0.0
    assert architecture["phase70_status"] == "standardization_improves_architecture_not_geofm"
    assert architecture["standardized_minus_phase63_summary"]["mean_delta"] > 0.0
    assert weak["phase70_status"] == "standardization_not_sufficient"
    assert incomplete["phase70_status"] == "standardized_rerun_incomplete"

    paths = write_phase70_standardized_set_policy_artifacts(
        {
            **geofm,
            "standardization_parameter_rows": [
                {"variant_id": "D4P8", "tile_id": "tile_train", "feature_name": "f0", "mean": 1.0, "scale": 2.0}
            ],
            "history_rows": [],
            "rollout_rows": geofm_rows,
            "oracle_summary_rows": [],
        },
        tmp_path / "outputs",
    )
    assert paths["comparison_json"].name == "phase70_standardized_set_policy_comparison.json"
    assert paths["delta_csv"].name == "phase70_standardized_delta_table.csv"
    assert "standardization_improves_geofm_set_policy_route" in paths["readiness_md"].read_text(encoding="utf-8")

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


def _phase70_fixture_roots(tmp_path: Path):
    columns = _required_feature_columns()
    blocks = ("b1", "b2", "b3", "b4")
    base_rows = [
        {**{"block_id": block_id}, **{column: 0.0 for column in columns}}
        for block_id in blocks
    ]
    for row, value in zip(base_rows, (0.90, 0.70, 0.20, 0.10)):
        row["explicit_feature_16"] = value

    phase2 = tmp_path / "phase2"
    phase8 = tmp_path / "phase8"
    phase61 = tmp_path / "phase61"
    _write_variant_fixture(phase2, "B0", base_rows, columns)
    _write_variant_fixture(phase8, "D4P8", base_rows, columns)
    _write_variant_fixture(phase61, "D6R8", base_rows, columns)
    tile_index = _write_csv(
        tmp_path / "tiles.csv",
        [
            {"tile_id": "tile_train", "block_ids": ";".join(blocks)},
            {"tile_id": "tile_eval", "block_ids": ";".join(blocks)},
        ],
    )
    baseline = [
        _rollout_row("B0", 0.50, tile_id="tile_eval", seed=0),
        _rollout_row("D4P8", 0.45, tile_id="tile_eval", seed=0),
        _rollout_row("D6R8", 0.48, tile_id="tile_eval", seed=0),
    ]
    baseline_csv = _write_csv(tmp_path / "phase63_rollout.csv", baseline)
    return phase2, phase8, phase61, tile_index, baseline_csv


def test_phase70_run_wrapper_and_cli_succeed_on_fixture(tmp_path):
    from paper11_geofm.phase70_standardized_set_policy_rerun import (
        run_phase70_standardized_set_policy_rerun,
    )

    phase2, phase8, phase61, tile_index, baseline_csv = _phase70_fixture_roots(tmp_path)
    analysis = run_phase70_standardized_set_policy_rerun(
        phase2_output_dir=phase2,
        phase8_output_dir=phase8,
        phase61_output_dir=phase61,
        tile_index_csv=tile_index,
        phase63_rollout_csv=baseline_csv,
        variants="B0,D4P8,D6R8",
        train_tile_id="tile_train",
        eval_tile_ids="tile_eval",
        seeds="0",
        eval_max_steps=3,
        bc_epochs=8,
        learning_rate=0.01,
        hidden_dim=12,
        top_k=2,
    )

    assert analysis["phase"] == "phase70_standardized_set_policy_rerun"
    assert len(analysis["rollout_rows"]) == 3
    assert len(analysis["history_rows"]) == 24
    assert len(analysis["standardization_parameter_rows"]) == 27
    assert analysis["phase70_status"] in {
        "standardization_improves_geofm_set_policy_route",
        "standardization_improves_architecture_not_geofm",
        "standardization_not_sufficient",
    }

    runner_path = (
        ROOT
        / "experiments"
        / "phase70_standardized_set_policy_rerun"
        / "run_phase70_standardized_set_policy_rerun.py"
    )
    spec = importlib.util.spec_from_file_location("phase70_runner", runner_path)
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
            str(baseline_csv),
            "--variants",
            "B0,D4P8,D6R8",
            "--train-tile-id",
            "tile_train",
            "--eval-tile-ids",
            "tile_eval",
            "--seeds",
            "0",
            "--eval-max-steps",
            "3",
            "--bc-epochs",
            "8",
            "--learning-rate",
            "0.01",
            "--hidden-dim",
            "12",
            "--top-k",
            "2",
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "outputs" / "phase70_standardized_set_policy_comparison.json").exists()