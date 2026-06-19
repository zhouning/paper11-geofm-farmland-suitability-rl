import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _explicit_row(block_id, area, slope_mean, farmland):
    row = {"block_id": block_id}
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = 0.0
    row.update(
        {
            "explicit_feature_00": area,
            "explicit_feature_01": slope_mean,
            "explicit_feature_02": slope_mean + 5.0,
            "explicit_feature_04": farmland,
            "explicit_feature_07": 0.0,
            "explicit_feature_09": 0.0,
            "explicit_feature_10": 0.0,
            "explicit_feature_13": 1.0 if slope_mean <= 15.0 else 0.0,
            "explicit_feature_16": farmland,
        }
    )
    return row


def _phase30_feature_rows():
    rows = [
        _explicit_row("b1", area=1.0, slope_mean=8.0, farmland=1.0),
        _explicit_row("b2", area=2.0, slope_mean=28.0, farmland=0.0),
        _explicit_row("b3", area=3.0, slope_mean=10.0, farmland=1.0),
        _explicit_row("b4", area=4.0, slope_mean=30.0, farmland=0.0),
    ]
    embedding_values = [
        (1.0, 1.0, 0.0),
        (2.0, 3.0, 0.0),
        (4.0, 5.0, 0.0),
        (8.0, 9.0, 0.0),
    ]
    for row, values in zip(rows, embedding_values):
        for dim in range(64):
            row[f"embedding_mean_{dim:02d}"] = 0.0
        for dim, value in enumerate(values):
            row[f"embedding_mean_{dim:02d}"] = value
    return rows


def _write_feature_csv(path: Path, rows: list[dict[str, object]], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["block_id", *columns])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in ["block_id", *columns]})
    return path


def _write_phase2_outputs(output_dir: Path) -> Path:
    rows = _phase30_feature_rows()
    explicit = [f"explicit_feature_{idx:02d}" for idx in range(17)]
    embedding = [f"embedding_mean_{idx:02d}" for idx in range(64)]
    _write_feature_csv(output_dir / "variant_B0_features.csv", rows, explicit)
    _write_feature_csv(output_dir / "variant_B1_features.csv", rows, explicit + embedding)
    manifest = {
        "claim_boundary": "fixture",
        "variants": {
            "B0": {
                "description": "Explicit planning feature baseline.",
                "state_groups": ["explicit_planning_features"],
                "reward": "base_planning_reward",
                "required_columns": explicit,
                "ready": True,
                "missing": [],
                "feature_table": "variant_B0_features.csv",
                "row_count": len(rows),
            },
            "B1": {
                "description": "Explicit planning features plus raw GeoFM embeddings.",
                "state_groups": ["explicit_planning_features", "geofm_embedding"],
                "reward": "base_planning_reward",
                "required_columns": explicit + embedding,
                "ready": True,
                "missing": [],
                "feature_table": "variant_B1_features.csv",
                "row_count": len(rows),
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_variants.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_dir


def _write_phase8_outputs(phase2_dir: Path, output_dir: Path) -> Path:
    from paper11_geofm.ablation_controls import (
        build_phase8_ablation_controls,
        write_phase8_ablation_artifacts,
    )

    protocol = build_phase8_ablation_controls(phase2_dir, seed=0, pca_dimensions=(8, 16))
    write_phase8_ablation_artifacts(protocol, output_dir)
    return output_dir


def _write_tile_index(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tile_id", "tile_row", "tile_col", "n_blocks", "block_ids"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "tile_id": "tile_train",
                "tile_row": 0,
                "tile_col": 0,
                "n_blocks": 2,
                "block_ids": "b1;b3",
            }
        )
        writer.writerow(
            {
                "tile_id": "tile_eval_a",
                "tile_row": 0,
                "tile_col": 1,
                "n_blocks": 2,
                "block_ids": "b2;b4",
            }
        )
    return path


def _summary_row(variant_id, reward, tile_id="tile_eval_a", seed=0):
    return {
        "row_type": "trained_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": tile_id,
        "eval_tile_rank": 1,
        "seed": seed,
        "phase25_seed_rank": seed + 1,
        "train_timesteps": 128,
        "eval_max_steps": 4,
        "max_blocks": 4,
        "train_n_blocks": 4,
        "eval_n_blocks": 2,
        "n_features": 81,
        "observation_shape": 333,
        "action_space_n": 4,
        "episode_steps": 2,
        "terminated": True,
        "truncated": False,
        "all_actions_valid": True,
        "invalid_action_count": 0,
        "total_contract_reward": reward,
        "selected_block_ids": "b1;b3",
        "claim_boundary": "fixture",
    }


def _phase30_summary_rows(case="normalization_beats_raw"):
    rewards = {
        "normalization_beats_raw": {
            "B0": 1.0,
            "B1": 0.7,
            "N1Z": 1.2,
            "N1ZR": 0.9,
            "D2": 0.6,
            "D3": 0.8,
            "D4P8": 1.1,
            "D4P16": 1.05,
        },
        "normalization_no_gain": {
            "B0": 1.0,
            "B1": 0.9,
            "N1Z": 0.85,
            "N1ZR": 0.88,
            "D2": 0.6,
            "D3": 0.8,
            "D4P8": 1.1,
            "D4P16": 1.05,
        },
    }[case]
    rows = []
    for tile_id in ("tile_eval_a", "tile_eval_b"):
        for seed in (0, 1):
            tile_offset = 0.1 if tile_id == "tile_eval_b" else 0.0
            seed_offset = 0.01 * seed
            for variant_id, reward in rewards.items():
                rows.append(
                    _summary_row(
                        variant_id,
                        reward + tile_offset + seed_offset,
                        tile_id=tile_id,
                        seed=seed,
                    )
                )
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(_summary_row("B0", 1.0).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_phase30_builds_true_zscore_and_zscore_row_l2_controls(tmp_path):
    from paper11_geofm.drl_inputs import load_variant_input
    from paper11_geofm.phase30_normalized_b1_ablation import (
        PHASE30_CLAIM_BOUNDARY,
        build_phase30_normalized_b1_controls,
        write_phase30_normalized_b1_controls,
    )

    phase2_dir = _write_phase2_outputs(tmp_path / "phase2")

    controls = build_phase30_normalized_b1_controls(phase2_dir)
    paths = write_phase30_normalized_b1_controls(controls, tmp_path / "normalized")

    assert controls["phase"] == "phase30_normalized_b1_controls"
    assert controls["claim_boundary"] == PHASE30_CLAIM_BOUNDARY
    assert set(controls["variant_tables"]) == {"N1Z", "N1ZR"}
    assert paths["manifest"].name == "experiment_variants.json"

    n1z = load_variant_input(tmp_path / "normalized", "N1Z")
    n1zr = load_variant_input(tmp_path / "normalized", "N1ZR")
    assert n1z.block_ids == ("b1", "b2", "b3", "b4")
    assert n1zr.block_ids == n1z.block_ids
    assert n1z.feature_columns[:17] == tuple(f"explicit_feature_{idx:02d}" for idx in range(17))
    assert n1z.feature_columns[17:] == tuple(f"embedding_mean_{idx:02d}" for idx in range(64))

    embedding = n1z.state_matrix[:, 17:].astype(float)
    assert np.allclose(embedding[:, 0].mean(), 0.0)
    assert np.allclose(embedding[:, 0].std(), 1.0)
    assert np.allclose(embedding[:, 1].mean(), 0.0)
    assert np.allclose(embedding[:, 1].std(), 1.0)
    assert np.allclose(embedding[:, 2], 0.0)

    n1zr_embedding = n1zr.state_matrix[:, 17:].astype(float)
    assert np.allclose(np.linalg.norm(n1zr_embedding, axis=1), 1.0)
    assert n1z.reward_mode == "base_planning_reward"
    assert n1zr.state_groups == (
        "explicit_planning_features",
        "column_zscore_row_l2_geofm_embedding",
    )


def test_phase30_analysis_reports_normalized_delta_status_and_artifacts(tmp_path):
    from paper11_geofm.phase30_normalized_b1_ablation import (
        PHASE30_CLAIM_BOUNDARY,
        build_phase30_normalized_b1_analysis,
        write_phase30_normalized_b1_artifacts,
    )

    analysis = build_phase30_normalized_b1_analysis(
        _phase30_summary_rows("normalization_beats_raw"),
        metadata={
            "variants": ["B0", "B1", "N1Z", "N1ZR", "D2", "D3", "D4P8", "D4P16"],
            "eval_tile_ids": ["tile_eval_a", "tile_eval_b"],
            "seeds": [0, 1],
            "train_timesteps": 128,
            "eval_max_steps": 4,
        },
    )

    assert analysis["phase"] == "phase30_normalized_b1_analysis"
    assert analysis["phase30_normalized_b1_status"] == (
        "normalized_b1_matches_or_exceeds_compressed_controls"
    )
    assert analysis["claim_boundary"] == PHASE30_CLAIM_BOUNDARY
    assert analysis["learned_policy"]["mean_reward_by_variant"]["N1Z"] == 1.255
    assert analysis["learned_policy"]["focal_deltas"]["N1Z_minus_B1"]["mean_reward_delta"] == 0.5
    assert analysis["learned_policy"]["focal_deltas"]["N1Z_minus_D4P8"]["positive_tile_seed_count"] == 4

    paths = write_phase30_normalized_b1_artifacts(
        {**analysis, "summaries": _phase30_summary_rows("normalization_beats_raw"), "traces": {}},
        tmp_path / "outputs",
    )
    assert paths["summary_csv"].name == "phase30_normalized_b1_summary.csv"
    assert paths["delta_csv"].name == "phase30_normalized_b1_delta_table.csv"
    assert paths["comparison_json"].name == "phase30_normalized_b1_comparison.json"
    assert paths["readiness_md"].name == "phase30_normalized_b1_readiness.md"

    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase30_normalized_b1_status"] == analysis["phase30_normalized_b1_status"]
    markdown = paths["readiness_md"].read_text(encoding="utf-8")
    assert "bounded representation-only ablation" in markdown
    assert "does not support submission-level planning-performance claims" in markdown


def test_phase30_run_with_existing_control_summary_trains_only_normalized_variants(
    tmp_path,
    monkeypatch,
):
    from paper11_geofm import phase30_normalized_b1_ablation as phase30

    class FakeModel:
        def predict(self, obs, deterministic=True, action_masks=None):
            valid_actions = [
                index
                for index, valid in enumerate(action_masks.tolist())
                if bool(valid)
            ]
            return valid_actions[0], None

    phase2_dir = _write_phase2_outputs(tmp_path / "phase2")
    phase8_dir = _write_phase8_outputs(phase2_dir, tmp_path / "phase8")
    tile_index = _write_tile_index(tmp_path / "phase13_tile_index.csv")
    existing_summary = _write_summary_csv(
        tmp_path / "phase28_control_summary.csv",
        [
            row
            for row in _phase30_summary_rows("normalization_beats_raw")
            if row["variant_id"] not in {"N1Z", "N1ZR"}
            and row["eval_tile_id"] == "tile_eval_a"
            and row["seed"] == 0
        ],
    )

    train_calls = []

    def _fake_train(train_env, seed, total_timesteps):
        train_calls.append(
            {
                "variant_id": train_env.unwrapped.variant_id,
                "seed": seed,
                "total_timesteps": total_timesteps,
            }
        )
        return FakeModel()

    monkeypatch.setattr(phase30, "_train_maskable_ppo_model", _fake_train)

    protocol = phase30.run_phase30_normalized_b1_ablation(
        phase2_output_dir=phase2_dir,
        phase8_output_dir=phase8_dir,
        tile_index_csv=tile_index,
        output_dir=tmp_path / "outputs",
        existing_control_summary_csv=existing_summary,
        variants=("B0", "B1", "N1Z", "N1ZR", "D2", "D3", "D4P8", "D4P16"),
        eval_tile_ids=("tile_eval_a",),
        total_timesteps=8,
        eval_max_steps=2,
        seeds=(0,),
    )

    assert train_calls == [
        {"variant_id": "N1Z", "seed": 0, "total_timesteps": 8},
        {"variant_id": "N1ZR", "seed": 0, "total_timesteps": 8},
    ]
    trained_rows = [
        row for row in protocol["summaries"] if row["row_type"] == "trained_policy"
    ]
    assert {"B0", "B1", "N1Z", "N1ZR", "D2", "D3", "D4P8", "D4P16"} == {
        row["variant_id"] for row in trained_rows
    }
    assert protocol["existing_control_summary_csv"] == str(existing_summary)


def test_phase30_cli_analyze_only_writes_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase30_normalized_b1_ablation"
        / "run_phase30_normalized_b1_ablation.py"
    )
    spec = importlib.util.spec_from_file_location("phase30_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    summary_csv = tmp_path / "phase30_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(_summary_row("B0", 1.0).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_phase30_summary_rows("normalization_beats_raw"))

    exit_code = module.main(
        [
            "--mode",
            "analyze-only",
            "--existing-summary-csv",
            str(summary_csv),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "Phase 30 normalized-B1 status: "
        "normalized_b1_matches_or_exceeds_compressed_controls"
    ) in stdout
    assert "phase30_normalized_b1_comparison.json" in stdout
