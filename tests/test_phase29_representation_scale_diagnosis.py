import csv
import importlib.util
import json
import math
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


def _write_tile_index(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tile_id",
                "tile_row",
                "tile_col",
                "n_blocks",
                "min_grid_row",
                "max_grid_row",
                "min_grid_col",
                "max_grid_col",
                "block_ids",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "tile_id": "tile_a",
                "tile_row": 0,
                "tile_col": 0,
                "n_blocks": 2,
                "min_grid_row": 0,
                "max_grid_row": 0,
                "min_grid_col": 0,
                "max_grid_col": 1,
                "block_ids": "b1;b2",
            }
        )
        writer.writerow(
            {
                "tile_id": "tile_b",
                "tile_row": 0,
                "tile_col": 1,
                "n_blocks": 2,
                "min_grid_row": 0,
                "max_grid_row": 0,
                "min_grid_col": 2,
                "max_grid_col": 3,
                "block_ids": "b3;b4",
            }
        )
    return path


def test_phase29_representation_scale_diagnosis_builds_variant_tile_and_normalization_tables(
    tmp_path,
):
    from paper11_geofm.phase29_representation_scale_diagnosis import (
        PHASE29_REPRESENTATION_SCALE_CLAIM_BOUNDARY,
        build_phase29_representation_scale_diagnosis,
    )

    feature_paths = _write_fixture_feature_tables(tmp_path)
    tile_index_csv = _write_tile_index(tmp_path / "phase13_tile_index.csv")

    analysis = build_phase29_representation_scale_diagnosis(
        phase2_b1_features_csv=feature_paths["B1"],
        d4p8_features_csv=feature_paths["D4P8"],
        d4p16_features_csv=feature_paths["D4P16"],
        tile_index_csv=tile_index_csv,
    )

    assert analysis["phase"] == "phase29_representation_scale_diagnosis"
    assert analysis["phase29_representation_scale_status"] == (
        "raw_b1_scale_may_affect_optimization"
    )
    assert analysis["claim_boundary"] == PHASE29_REPRESENTATION_SCALE_CLAIM_BOUNDARY

    variant_rows = {
        row["variant_id"]: row
        for row in analysis["variant_scale_rows"]
    }
    assert set(variant_rows) == {"B1", "D4P8", "D4P16"}
    assert variant_rows["B1"]["n_blocks"] == 4
    assert variant_rows["B1"]["n_dimensions"] == 64
    assert variant_rows["B1"]["mean_row_l2_norm"] == 0.8535533906
    assert variant_rows["B1"]["mean_column_std"] == 0.015625
    assert variant_rows["D4P8"]["mean_column_std"] > variant_rows["B1"]["mean_column_std"]
    assert variant_rows["D4P16"]["mean_column_std"] > variant_rows["B1"]["mean_column_std"]

    tile_rows = {
        row["tile_id"]: row
        for row in analysis["tile_scale_rows"]
    }
    assert set(tile_rows) == {"tile_a", "tile_b"}
    assert tile_rows["tile_a"]["mean_b1_row_l2_norm"] == 0.5
    assert tile_rows["tile_b"]["mean_b1_row_l2_norm"] == 1.2071067812

    normalization_rows = {
        row["profile_id"]: row
        for row in analysis["b1_normalization_profile_rows"]
    }
    assert set(normalization_rows) == {
        "raw",
        "column_zscore",
        "row_l2",
        "column_zscore_row_l2",
    }
    assert normalization_rows["raw"]["mean_row_l2_norm"] == 0.8535533906
    assert normalization_rows["column_zscore"]["mean_row_l2_norm"] == 1.4142135624
    assert normalization_rows["column_zscore"]["std_row_l2_norm"] == 0.0
    assert normalization_rows["row_l2"]["mean_row_l2_norm"] == 0.75
    assert normalization_rows["column_zscore_row_l2"]["mean_row_l2_norm"] == 1.0
    assert normalization_rows["column_zscore_row_l2"]["std_row_l2_norm"] == 0.0

    pca = analysis["pca_diagnostics"]
    assert pca["raw_embedding_numerical_rank"] == 2
    assert pca["top8_pca_variance_ratio"] == 1.0
    assert pca["top16_pca_variance_ratio"] == 1.0
    assert pca["mean_d4p8_component_std"] > pca["mean_raw_embedding_std"]


def test_phase29_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase29_representation_scale_diagnosis import (
        build_phase29_representation_scale_diagnosis,
        write_phase29_representation_scale_diagnosis_artifacts,
    )

    feature_paths = _write_fixture_feature_tables(tmp_path)
    tile_index_csv = _write_tile_index(tmp_path / "phase13_tile_index.csv")
    analysis = build_phase29_representation_scale_diagnosis(
        phase2_b1_features_csv=feature_paths["B1"],
        d4p8_features_csv=feature_paths["D4P8"],
        d4p16_features_csv=feature_paths["D4P16"],
        tile_index_csv=tile_index_csv,
    )

    paths = write_phase29_representation_scale_diagnosis_artifacts(
        analysis,
        tmp_path / "outputs",
    )

    assert paths["variant_scale_csv"].name == "phase29_variant_scale_summary.csv"
    assert paths["tile_scale_csv"].name == "phase29_tile_scale_summary.csv"
    assert paths["normalization_profiles_csv"].name == "phase29_b1_normalization_profiles.csv"
    assert paths["diagnosis_json"].name == "phase29_representation_scale_diagnosis.json"
    assert paths["diagnosis_md"].name == "phase29_representation_scale_diagnosis.md"
    assert all(path.exists() for path in paths.values())

    variant_rows = list(
        csv.DictReader(paths["variant_scale_csv"].open("r", encoding="utf-8"))
    )
    assert {row["variant_id"] for row in variant_rows} == {"B1", "D4P8", "D4P16"}

    saved = json.loads(paths["diagnosis_json"].read_text(encoding="utf-8"))
    assert saved["phase29_representation_scale_status"] == (
        "raw_b1_scale_may_affect_optimization"
    )
    markdown = paths["diagnosis_md"].read_text(encoding="utf-8")
    assert "Variant-scale summary:" in markdown
    assert "B1 normalization profiles:" in markdown
    assert "does not prove that normalization would improve PPO performance" in markdown


def test_phase29_cli_writes_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase29_representation_scale_diagnosis"
        / "run_phase29_representation_scale_diagnosis.py"
    )
    spec = importlib.util.spec_from_file_location("phase29_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    feature_paths = _write_fixture_feature_tables(tmp_path)
    tile_index_csv = _write_tile_index(tmp_path / "phase13_tile_index.csv")

    exit_code = module.main(
        [
            "--phase2-b1-features-csv",
            str(feature_paths["B1"]),
            "--d4p8-features-csv",
            str(feature_paths["D4P8"]),
            "--d4p16-features-csv",
            str(feature_paths["D4P16"]),
            "--tile-index-csv",
            str(tile_index_csv),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "Phase 29 representation-scale status: "
        "raw_b1_scale_may_affect_optimization"
    ) in stdout
    assert "phase29_representation_scale_diagnosis.json" in stdout
