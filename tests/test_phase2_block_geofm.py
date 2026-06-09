import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_validate_block_pixel_mapping_rejects_out_of_range_pixels():
    from paper11_geofm.block_mapping import validate_block_pixel_mapping

    rows = [
        {"block_id": "b0", "row": 0, "col": 0},
        {"block_id": "b1", "row": 67, "col": 0},
    ]

    with pytest.raises(ValueError, match="outside grid_shape"):
        validate_block_pixel_mapping(rows, (67, 70))


def test_validate_block_pixel_mapping_defaults_weights_and_preserves_order():
    from paper11_geofm.block_mapping import validate_block_pixel_mapping

    rows = [
        {"block_id": "b0", "row": 0, "col": 0},
        {"block_id": "b0", "row": 0, "col": 1, "weight": 2.5},
        {"block_id": "b1", "row": 1, "col": 0},
    ]

    mapping = validate_block_pixel_mapping(rows, (67, 70))

    assert [entry["block_id"] for entry in mapping] == ["b0", "b0", "b1"]
    assert [entry["weight"] for entry in mapping] == [1.0, 2.5, 1.0]
    assert all(isinstance(entry["row"], int) for entry in mapping)
    assert all(isinstance(entry["col"], int) for entry in mapping)


def _tiny_embedding_grid():
    grid = np.zeros((2, 2, 64), dtype=np.float64)
    grid[0, 0, :] = 1.0
    grid[0, 1, :] = 3.0
    grid[1, 0, :] = 10.0
    grid[1, 1, :] = 20.0
    return grid


def test_compute_block_geofm_features_uses_weighted_pixel_means():
    from paper11_geofm.block_features import compute_block_geofm_features
    from paper11_geofm.block_mapping import validate_block_pixel_mapping

    base_embedding = _tiny_embedding_grid()
    mapping = validate_block_pixel_mapping(
        [
            {"block_id": "b0", "row": 0, "col": 0, "weight": 1.0},
            {"block_id": "b0", "row": 0, "col": 1, "weight": 3.0},
            {"block_id": "b1", "row": 1, "col": 0},
        ],
        (2, 2),
    )

    rows = compute_block_geofm_features(base_embedding, mapping)

    assert [row["block_id"] for row in rows] == ["b0", "b1"]
    assert rows[0]["pixel_count"] == 2
    assert rows[0]["pixel_weight_sum"] == 4.0
    assert rows[0]["row_min"] == 0
    assert rows[0]["row_max"] == 0
    assert rows[0]["col_min"] == 0
    assert rows[0]["col_max"] == 1
    assert rows[0]["embedding_mean_00"] == 2.5
    assert rows[0]["embedding_mean_63"] == 2.5
    assert rows[1]["embedding_mean_00"] == 10.0
    assert "embedding_std_mean" in rows[0]
    assert "temporal_stability" in rows[0]


def test_compute_block_geofm_features_uses_annual_temporal_stability():
    from paper11_geofm.block_features import compute_block_geofm_features
    from paper11_geofm.block_mapping import validate_block_pixel_mapping

    base_embedding = _tiny_embedding_grid()
    annual_embeddings = {
        2020: base_embedding,
        2021: base_embedding + 1.0,
    }
    mapping = validate_block_pixel_mapping(
        [{"block_id": "b0", "row": 0, "col": 0}],
        (2, 2),
    )

    rows = compute_block_geofm_features(base_embedding, mapping, annual_embeddings)

    assert 0.0 < rows[0]["temporal_stability"] < 1.0


def test_attach_optional_block_attributes_preserves_explicit_features_and_reports_readiness():
    from paper11_geofm.block_features import attach_optional_block_attributes
    from paper11_geofm.block_schema import summarize_phase2_readiness

    rows = [
        {
            "block_id": "b0",
            "pixel_count": 1,
            "pixel_weight_sum": 1.0,
            "embedding_mean_00": 1.0,
            "embedding_mean_63": 1.0,
            "suitability_proxy": 0.5,
        }
    ]
    attributes = [
        {
            "block_id": "b0",
            "explicit_feature_00": 7.0,
            "explicit_feature_16": 9.0,
            "stable_farmland_label": 1,
            "split": "train",
        }
    ]

    joined = attach_optional_block_attributes(rows, attributes)
    readiness = summarize_phase2_readiness(joined)

    assert joined[0]["explicit_feature_00"] == 7.0
    assert joined[0]["explicit_feature_16"] == 9.0
    assert joined[0]["stable_farmland_label"] == 1
    assert joined[0]["split"] == "train"
    assert readiness["B0"]["ready"] is False
    assert readiness["B1"]["ready"] is False
    assert readiness["B2"]["ready"] is False
    assert readiness["B3"]["ready"] is False
    assert "explicit_features_incomplete" in readiness["B0"]["missing"]


def test_phase2_readiness_requires_required_columns_on_every_block():
    from paper11_geofm.block_schema import summarize_phase2_readiness

    complete_row = {
        "block_id": "b0",
        "suitability_proxy": 0.8,
    }
    incomplete_row = {
        "block_id": "b1",
        "suitability_proxy": 0.4,
    }
    for dim in range(64):
        complete_row[f"embedding_mean_{dim:02d}"] = float(dim)
        incomplete_row[f"embedding_mean_{dim:02d}"] = float(dim)
    for idx in range(17):
        complete_row[f"explicit_feature_{idx:02d}"] = float(idx)
        if idx < 16:
            incomplete_row[f"explicit_feature_{idx:02d}"] = float(idx)

    readiness = summarize_phase2_readiness([complete_row, incomplete_row])

    assert readiness["B0"]["ready"] is False
    assert readiness["B1"]["ready"] is False
    assert readiness["B2"]["ready"] is False
    assert readiness["B3"]["ready"] is False
    assert "explicit_features_incomplete" in readiness["B3"]["missing"]


def test_phase2_readiness_marks_complete_b3_table_ready():
    from paper11_geofm.block_schema import summarize_phase2_readiness

    row = {
        "block_id": "b0",
        "suitability_proxy": 0.8,
    }
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(dim)
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = float(idx)

    readiness = summarize_phase2_readiness([row])

    assert readiness["B0"] == {"ready": True, "missing": []}
    assert readiness["B1"] == {"ready": True, "missing": []}
    assert readiness["B2"] == {"ready": True, "missing": []}
    assert readiness["B3"] == {"ready": True, "missing": []}


def test_phase2_artifacts_are_written_with_readiness_and_claim_boundary(tmp_path):
    from paper11_geofm.artifacts import write_phase2_artifacts

    row = {
        "block_id": "b0",
        "pixel_count": 1,
        "pixel_weight_sum": 1.0,
        "row_min": 0,
        "row_max": 0,
        "col_min": 0,
        "col_max": 0,
        "embedding_std_mean": 0.0,
        "temporal_stability": 1.0,
        "suitability_proxy": 0.75,
    }
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(dim)

    paths = write_phase2_artifacts(
        [row],
        tmp_path,
        {
            "metadata_source": "test",
            "base_year_requested": 2020,
            "base_year_used": 2020,
            "years": [2020],
            "grid_shape": [2, 2],
            "embedding_dim": 64,
            "mapping_mode": "test",
        },
    )

    with paths["block_table"].open("r", encoding="utf-8", newline="") as handle:
        record = next(csv.DictReader(handle))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

    assert record["block_id"] == "b0"
    assert paths["block_table"].name == "block_geofm_features.csv"
    assert summary["n_blocks"] == 1
    assert summary["block_table"] == "block_geofm_features.csv"
    assert summary["feature_readiness"]["B1"]["ready"] is False
    assert "does not directly measure soil" in summary["claim_boundary"].lower()


def test_phase2_runner_writes_block_feature_artifacts(tmp_path):
    runner_path = (
        ROOT / "experiments" / "phase2_block_geofm_features" / "run_phase2.py"
    )
    spec = importlib.util.spec_from_file_location("phase2_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(["--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "block_geofm_features.csv").exists()
    assert (tmp_path / "summary.json").exists()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_blocks"] == 25
    assert summary["mapping_mode"] == "generated_grid"
    assert summary["feature_readiness"]["B3"]["ready"] is False


def test_phase2_runner_accepts_mapping_csv(tmp_path):
    runner_path = (
        ROOT / "experiments" / "phase2_block_geofm_features" / "run_phase2.py"
    )
    mapping_csv = tmp_path / "mapping.csv"
    mapping_csv.write_text(
        "block_id,row,col,weight\n"
        "block_a,0,0,1.0\n"
        "block_a,0,1,2.0\n"
        "block_b,1,0,1.0\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"

    spec = importlib.util.spec_from_file_location("phase2_runner_csv", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--mapping-csv",
            str(mapping_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    with (output_dir / "block_geofm_features.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        records = list(csv.DictReader(handle))

    assert summary["n_blocks"] == 2
    assert summary["mapping_mode"] == "mapping_csv"
    assert summary["mapping_csv"] == str(mapping_csv)
    assert [record["block_id"] for record in records] == ["block_a", "block_b"]
    assert records[0]["pixel_weight_sum"] == "3.0"


def test_phase2_runner_accepts_attributes_csv_and_marks_b3_ready(tmp_path):
    runner_path = (
        ROOT / "experiments" / "phase2_block_geofm_features" / "run_phase2.py"
    )
    mapping_csv = tmp_path / "mapping.csv"
    mapping_csv.write_text(
        "block_id,row,col\n"
        "block_a,0,0\n"
        "block_b,1,0\n",
        encoding="utf-8",
    )
    attribute_columns = [f"explicit_feature_{idx:02d}" for idx in range(17)]
    attributes_csv = tmp_path / "attributes.csv"
    attributes_csv.write_text(
        ",".join(["block_id", *attribute_columns, "stable_farmland_label", "split"])
        + "\n"
        + ",".join(["block_a", *["1.0" for _ in attribute_columns], "1", "train"])
        + "\n"
        + ",".join(["block_b", *["2.0" for _ in attribute_columns], "0", "test"])
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"

    spec = importlib.util.spec_from_file_location("phase2_runner_attrs", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--mapping-csv",
            str(mapping_csv),
            "--attributes-csv",
            str(attributes_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    with (output_dir / "block_geofm_features.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        records = list(csv.DictReader(handle))

    assert summary["mapping_mode"] == "mapping_csv"
    assert summary["attributes_csv"] == str(attributes_csv)
    assert "explicit_planning_features" in summary["feature_groups_present"]
    assert "weak_labels" in summary["feature_groups_present"]
    assert summary["feature_readiness"]["B3"] == {"ready": True, "missing": []}
    assert records[0]["explicit_feature_00"] == "1.0"
    assert records[1]["split"] == "test"


def test_included_phase2_csv_fixtures_have_expected_schema():
    fixture_dir = ROOT / "data" / "bishan_phase2_csv_sample"
    mapping_csv = fixture_dir / "block_pixel_mapping.csv"
    attributes_csv = fixture_dir / "block_attributes.csv"

    assert mapping_csv.exists()
    assert attributes_csv.exists()

    with mapping_csv.open("r", encoding="utf-8", newline="") as handle:
        mapping_rows = list(csv.DictReader(handle))
    with attributes_csv.open("r", encoding="utf-8", newline="") as handle:
        attribute_rows = list(csv.DictReader(handle))

    assert len(mapping_rows) == 8
    assert len(attribute_rows) == 4
    assert {"block_id", "row", "col", "weight"}.issubset(mapping_rows[0])
    assert {row["block_id"] for row in mapping_rows} == {
        "sample_block_00",
        "sample_block_01",
        "sample_block_02",
        "sample_block_03",
    }

    required_attributes = {
        "block_id",
        "stable_farmland_label",
        "high_standard_farmland_label",
        "split",
    }
    required_attributes.update(f"explicit_feature_{idx:02d}" for idx in range(17))
    assert required_attributes.issubset(attribute_rows[0])


def test_phase2_runner_accepts_included_csv_fixtures(tmp_path):
    runner_path = (
        ROOT / "experiments" / "phase2_block_geofm_features" / "run_phase2.py"
    )
    fixture_dir = ROOT / "data" / "bishan_phase2_csv_sample"
    mapping_csv = fixture_dir / "block_pixel_mapping.csv"
    attributes_csv = fixture_dir / "block_attributes.csv"
    output_dir = tmp_path / "outputs"

    spec = importlib.util.spec_from_file_location("phase2_runner_fixture", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--mapping-csv",
            str(mapping_csv),
            "--attributes-csv",
            str(attributes_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    with (output_dir / "block_geofm_features.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        records = list(csv.DictReader(handle))

    assert summary["n_blocks"] == 4
    assert summary["mapping_mode"] == "mapping_csv"
    assert summary["feature_readiness"]["B3"] == {"ready": True, "missing": []}
    assert [record["block_id"] for record in records] == [
        "sample_block_00",
        "sample_block_01",
        "sample_block_02",
        "sample_block_03",
    ]
