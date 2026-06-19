import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _explicit_row(
    block_id,
    *,
    low_slope_farmland,
    farmland,
    low_slope,
    area,
    slope_mean,
    slope_max,
    built_up,
    water,
):
    row = {"block_id": block_id}
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = 0.0
    row.update(
        {
            "explicit_feature_00": area,
            "explicit_feature_01": slope_mean,
            "explicit_feature_02": slope_max,
            "explicit_feature_04": farmland,
            "explicit_feature_07": 0.0,
            "explicit_feature_09": built_up,
            "explicit_feature_10": water,
            "explicit_feature_13": low_slope,
            "explicit_feature_16": low_slope_farmland,
        }
    )
    return row


def _feature_rows():
    rows = [
        _explicit_row(
            "b1",
            low_slope_farmland=0.0,
            farmland=0.0,
            low_slope=0.0,
            area=1.0,
            slope_mean=25.0,
            slope_max=35.0,
            built_up=1.0,
            water=0.5,
        ),
        _explicit_row(
            "b2",
            low_slope_farmland=0.0,
            farmland=0.5,
            low_slope=0.0,
            area=2.0,
            slope_mean=20.0,
            slope_max=30.0,
            built_up=0.5,
            water=0.0,
        ),
        _explicit_row(
            "b3",
            low_slope_farmland=1.0,
            farmland=1.0,
            low_slope=1.0,
            area=5.0,
            slope_mean=5.0,
            slope_max=10.0,
            built_up=0.0,
            water=0.0,
        ),
        _explicit_row(
            "b4",
            low_slope_farmland=1.0,
            farmland=1.0,
            low_slope=1.0,
            area=4.0,
            slope_mean=10.0,
            slope_max=15.0,
            built_up=0.0,
            water=0.0,
        ),
    ]
    for row_index, row in enumerate(rows):
        for dim in range(64):
            row[f"embedding_mean_{dim:02d}"] = 0.0
        row["embedding_mean_00"] = 1.0 if row_index in {1, 3} else 0.0
        row["embedding_mean_01"] = 1.0 if row_index in {2, 3} else 0.0
        for dim in range(8):
            row[f"embedding_pca_{dim:02d}"] = float(row_index + dim)
        for dim in range(8, 16):
            row[f"embedding_pca_{dim:02d}"] = float((row_index + 1) * dim)
    return rows


def _write_feature_csv(path: Path, rows: list[dict[str, object]], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["block_id", *columns])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in ["block_id", *columns]})
    return path


def _write_fixture_feature_tables(tmp_path):
    rows = _feature_rows()
    explicit = [f"explicit_feature_{idx:02d}" for idx in range(17)]
    embedding = [f"embedding_mean_{idx:02d}" for idx in range(64)]
    pca8 = [f"embedding_pca_{idx:02d}" for idx in range(8)]
    pca16 = [f"embedding_pca_{idx:02d}" for idx in range(16)]
    return {
        "B1": _write_feature_csv(tmp_path / "variant_B1_features.csv", rows, explicit + embedding),
        "D4P8": _write_feature_csv(tmp_path / "variant_D4P8_features.csv", rows, explicit + pca8),
        "D4P16": _write_feature_csv(tmp_path / "variant_D4P16_features.csv", rows, explicit + pca16),
    }


def _summary_row(variant_id, tile_id, seed, reward, selected_block_ids):
    return {
        "row_type": "trained_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": tile_id,
        "eval_tile_rank": 1,
        "seed": seed,
        "phase25_seed_rank": int(seed) + 1,
        "train_timesteps": 4096,
        "eval_max_steps": 8,
        "max_blocks": 4,
        "train_n_blocks": 4,
        "eval_n_blocks": 4,
        "n_features": 81,
        "observation_shape": 100,
        "action_space_n": 4,
        "episode_steps": 2,
        "terminated": True,
        "truncated": False,
        "all_actions_valid": True,
        "invalid_action_count": 0,
        "total_contract_reward": reward,
        "selected_block_ids": selected_block_ids,
        "claim_boundary": "phase28 fixture",
    }


def _write_summary_csv(path: Path) -> Path:
    from paper11_geofm.padded_heldout_policy import SUMMARY_FIELDNAMES

    rows = []
    for tile_id, seed, b1_reward in [
        ("tile_a", 0, 1.0),
        ("tile_a", 1, 1.2),
    ]:
        rows.extend(
            [
                _summary_row("B1", tile_id, seed, b1_reward, "b1;b2"),
                _summary_row("B0", tile_id, seed, b1_reward + 0.2, "b3;b4"),
                _summary_row("D2", tile_id, seed, b1_reward - 0.2, "b1;b3"),
                _summary_row("D3", tile_id, seed, b1_reward + 0.1, "b2;b3"),
                _summary_row("D4P8", tile_id, seed, b1_reward + 0.5, "b3;b4"),
                _summary_row("D4P16", tile_id, seed, b1_reward + 0.4, "b3;b4"),
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_phase28_compression_diagnosis_builds_overlap_reward_and_pca_tables(tmp_path):
    from paper11_geofm.phase28_compression_diagnosis import (
        PHASE28_COMPRESSION_CLAIM_BOUNDARY,
        build_phase28_compression_diagnosis,
    )

    feature_paths = _write_fixture_feature_tables(tmp_path)
    summary_csv = _write_summary_csv(tmp_path / "phase28_summary.csv")

    analysis = build_phase28_compression_diagnosis(
        summary_csv=summary_csv,
        phase2_b1_features_csv=feature_paths["B1"],
        d4p8_features_csv=feature_paths["D4P8"],
        d4p16_features_csv=feature_paths["D4P16"],
    )

    assert analysis["phase"] == "phase28_compression_diagnosis"
    assert analysis["phase28_compression_diagnostic_status"] == (
        "compressed_controls_select_distinct_higher_reward_blocks"
    )
    assert analysis["claim_boundary"] == PHASE28_COMPRESSION_CLAIM_BOUNDARY

    overlap = {
        row["comparator_variant_id"]: row
        for row in analysis["selection_overlap_rows"]
    }
    assert overlap["D2"]["mean_jaccard_overlap"] == 0.3333333333
    assert overlap["D4P8"]["mean_jaccard_overlap"] == 0.0
    assert overlap["D4P8"]["mean_b1_minus_comparator_reward"] == -0.5
    assert overlap["D4P16"]["mean_shared_selected_blocks"] == 0.0

    components = {
        row["variant_id"]: row
        for row in analysis["reward_component_rows"]
    }
    assert components["B1"]["selected_block_count"] == 4
    assert components["B1"]["low_slope_farmland_or_orchard"] == 0.0
    assert components["D4P8"]["low_slope_farmland_or_orchard"] == 0.35
    assert components["D4P16"]["mean_slope_penalty"] > components["B1"]["mean_slope_penalty"]

    pca = analysis["pca_diagnostics"]
    assert pca["raw_embedding_rank_threshold"] == 1e-12
    assert pca["raw_embedding_numerical_rank"] == 2
    assert pca["top8_pca_variance_ratio"] == 1.0
    assert pca["top16_pca_variance_ratio"] == 1.0
    assert pca["mean_d4p8_component_std"] > 0.0


def test_phase28_compression_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase28_compression_diagnosis import (
        build_phase28_compression_diagnosis,
        write_phase28_compression_diagnosis_artifacts,
    )

    feature_paths = _write_fixture_feature_tables(tmp_path)
    summary_csv = _write_summary_csv(tmp_path / "phase28_summary.csv")
    analysis = build_phase28_compression_diagnosis(
        summary_csv=summary_csv,
        phase2_b1_features_csv=feature_paths["B1"],
        d4p8_features_csv=feature_paths["D4P8"],
        d4p16_features_csv=feature_paths["D4P16"],
    )

    paths = write_phase28_compression_diagnosis_artifacts(
        analysis,
        tmp_path / "outputs",
    )

    assert paths["overlap_csv"].name == "phase28_compression_overlap.csv"
    assert paths["reward_components_csv"].name == "phase28_compression_reward_components.csv"
    assert paths["diagnosis_json"].name == "phase28_compression_diagnosis.json"
    assert paths["diagnosis_md"].name == "phase28_compression_diagnosis.md"
    assert all(path.exists() for path in paths.values())

    overlap_rows = list(
        csv.DictReader(paths["overlap_csv"].open("r", encoding="utf-8"))
    )
    assert {row["comparator_variant_id"] for row in overlap_rows} == {
        "B0",
        "D2",
        "D3",
        "D4P8",
        "D4P16",
    }
    saved = json.loads(paths["diagnosis_json"].read_text(encoding="utf-8"))
    assert saved["phase28_compression_diagnostic_status"] == (
        "compressed_controls_select_distinct_higher_reward_blocks"
    )
    markdown = paths["diagnosis_md"].read_text(encoding="utf-8")
    assert "Selection overlap:" in markdown
    assert "Reward components:" in markdown
    assert "Compression-scale diagnostics:" in markdown
    assert "does not prove that PCA is intrinsically superior" in markdown


def test_phase28_compression_cli_writes_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase28_compression_diagnosis"
        / "run_phase28_compression_diagnosis.py"
    )
    spec = importlib.util.spec_from_file_location("phase28_compression_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    feature_paths = _write_fixture_feature_tables(tmp_path)
    summary_csv = _write_summary_csv(tmp_path / "phase28_summary.csv")

    exit_code = module.main(
        [
            "--summary-csv",
            str(summary_csv),
            "--phase2-b1-features-csv",
            str(feature_paths["B1"]),
            "--d4p8-features-csv",
            str(feature_paths["D4P8"]),
            "--d4p16-features-csv",
            str(feature_paths["D4P16"]),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "Phase 28 compression diagnostic status: "
        "compressed_controls_select_distinct_higher_reward_blocks"
    ) in stdout
    assert "phase28_compression_diagnosis.json" in stdout
