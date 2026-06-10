import csv
import importlib.util
import json
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _metadata_path(tmp_path: Path) -> Path:
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "bbox": [0.0, 0.0, 2.0, 2.0],
                "grid_shape": [2, 2],
                "scale_m": 500,
                "embedding_dim": 64,
            }
        ),
        encoding="utf-8",
    )
    return path


def _tiny_dltb_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_dltb.gpkg"
    gdf = gpd.GeoDataFrame(
        [
            {
                "BSM": 1,
                "DLBM": "011",
                "DLMC": "paddy_field",
                "TBMJ": 10000.0,
                "category": "Farmland",
                "slope_mean": 5.0,
                "slope_max": 8.0,
                "slope_pixel_count": 4,
                "geometry": Polygon(
                    [(0.1, 1.6), (0.3, 1.6), (0.3, 1.8), (0.1, 1.8)]
                ),
            },
            {
                "BSM": 2,
                "DLBM": "031",
                "DLMC": "forest_land",
                "TBMJ": 20000.0,
                "category": "Forest",
                "slope_mean": 18.0,
                "slope_max": 25.0,
                "slope_pixel_count": 7,
                "geometry": Polygon(
                    [(1.5, 0.1), (1.7, 0.1), (1.7, 0.3), (1.5, 0.3)]
                ),
            },
            {
                "BSM": 3,
                "DLBM": "023",
                "DLMC": "other_orchard",
                "TBMJ": 15000.0,
                "category": "Orchard",
                "slope_mean": 4.0,
                "slope_max": 6.0,
                "slope_pixel_count": 3,
                "geometry": Polygon(
                    [(0.6, 0.6), (0.8, 0.6), (0.8, 0.8), (0.6, 0.8)]
                ),
            },
        ],
        crs="EPSG:4326",
    )
    gdf.to_file(path, layer="DLTB", driver="GPKG")
    return path


def test_phase11_builds_mapping_attributes_and_summary(tmp_path):
    from paper11_geofm.dltb_adapter import (
        PHASE11_CLAIM_BOUNDARY,
        build_bishan_dltb_phase2_inputs,
    )

    payload = build_bishan_dltb_phase2_inputs(
        _tiny_dltb_path(tmp_path),
        _metadata_path(tmp_path),
    )

    mapping_rows = payload["mapping_rows"]
    attribute_rows = payload["attribute_rows"]
    summary = payload["summary"]

    assert len(mapping_rows) == 3
    assert mapping_rows[0] == {
        "block_id": "dltb_1",
        "row": 0,
        "col": 0,
        "weight": 1.0,
    }
    assert mapping_rows[1]["row"] == 1
    assert mapping_rows[1]["col"] == 1
    assert attribute_rows[0]["current_farmland_label"] == 1
    assert attribute_rows[0]["low_slope_farmland_label"] == 1
    assert attribute_rows[1]["current_farmland_label"] == 0
    assert attribute_rows[1]["explicit_feature_15"] == 1.0
    assert attribute_rows[2]["farmland_or_orchard_label"] == 1
    assert all(f"explicit_feature_{idx:02d}" in attribute_rows[0] for idx in range(17))
    assert summary["rows_exported"] == 3
    assert summary["label_positive_counts"]["current_farmland_label"] == 1
    assert summary["claim_boundary"] == PHASE11_CLAIM_BOUNDARY


def test_phase11_max_blocks_caps_rows_deterministically(tmp_path):
    from paper11_geofm.dltb_adapter import build_bishan_dltb_phase2_inputs

    payload = build_bishan_dltb_phase2_inputs(
        _tiny_dltb_path(tmp_path),
        _metadata_path(tmp_path),
        max_blocks=2,
    )

    assert [row["block_id"] for row in payload["mapping_rows"]] == [
        "dltb_1",
        "dltb_2",
    ]
    assert payload["summary"]["rows_exported"] == 2


def test_phase11_writes_phase2_input_csvs_and_summary(tmp_path):
    from paper11_geofm.dltb_adapter import (
        build_bishan_dltb_phase2_inputs,
        write_bishan_dltb_phase2_inputs,
    )

    payload = build_bishan_dltb_phase2_inputs(
        _tiny_dltb_path(tmp_path),
        _metadata_path(tmp_path),
    )

    paths = write_bishan_dltb_phase2_inputs(payload, tmp_path / "outputs")

    assert paths["mapping_csv"].name == "block_pixel_mapping.csv"
    assert paths["attributes_csv"].name == "block_attributes.csv"
    assert paths["summary"].name == "phase11_bishan_dltb_adapter_summary.json"
    with paths["mapping_csv"].open("r", encoding="utf-8", newline="") as handle:
        mapping_records = list(csv.DictReader(handle))
    with paths["attributes_csv"].open("r", encoding="utf-8", newline="") as handle:
        attribute_records = list(csv.DictReader(handle))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert mapping_records[0]["block_id"] == "dltb_1"
    assert attribute_records[0]["current_farmland_label"] == "1"
    assert summary["rows_exported"] == 3


def test_phase11_cli_writes_adapter_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase11_bishan_dltb_real"
        / "run_phase11_bishan_dltb_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("phase11_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--dltb-path",
            str(_tiny_dltb_path(tmp_path)),
            "--metadata-path",
            str(_metadata_path(tmp_path)),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "Rows exported: 3" in stdout
    assert "block_pixel_mapping.csv" in stdout
    assert "block_attributes.csv" in stdout
    assert "Claim boundary: Phase 11 builds real Bishan DLTB-derived" in stdout
