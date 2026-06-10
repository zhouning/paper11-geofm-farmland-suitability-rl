import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_mapping(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"block_id": "b1", "row": 0, "col": 0, "weight": 1.0},
        {"block_id": "b2", "row": 0, "col": 1, "weight": 1.0},
        {"block_id": "b3", "row": 3, "col": 0, "weight": 1.0},
        {"block_id": "b4", "row": 5, "col": 5, "weight": 1.0},
        {"block_id": "b5", "row": 6, "col": 5, "weight": 1.0},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["block_id", "row", "col", "weight"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_manifest(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    variants = {
        "B0": {
            "ready": True,
            "missing": [],
            "required_columns": ["explicit_feature_00", "explicit_feature_01"],
            "reward": "base_planning_reward",
            "feature_table": "variant_B0_features.csv",
            "state_groups": ["explicit_planning_features"],
        },
        "B1": {
            "ready": True,
            "missing": [],
            "required_columns": [
                "explicit_feature_00",
                "explicit_feature_01",
                "embedding_mean_00",
            ],
            "reward": "base_planning_reward",
            "feature_table": "variant_B1_features.csv",
            "state_groups": ["explicit_planning_features", "geofm_embedding"],
        },
        "B2": {
            "ready": True,
            "missing": [],
            "required_columns": [
                "explicit_feature_00",
                "explicit_feature_01",
                "suitability_proxy",
            ],
            "reward": "base_plus_suitability_reward",
            "feature_table": "variant_B2_features.csv",
            "state_groups": ["explicit_planning_features", "suitability_proxy"],
        },
        "B3": {
            "ready": True,
            "missing": [],
            "required_columns": [
                "explicit_feature_00",
                "explicit_feature_01",
                "embedding_mean_00",
                "suitability_proxy",
            ],
            "reward": "base_plus_suitability_reward",
            "feature_table": "variant_B3_features.csv",
            "state_groups": [
                "explicit_planning_features",
                "geofm_embedding",
                "suitability_proxy",
            ],
        },
    }
    path.write_text(json.dumps({"variants": variants}, indent=2), encoding="utf-8")
    return path


def test_phase13_builds_tile_index_and_contract_summary(tmp_path):
    from paper11_geofm.tiled_contract import (
        PHASE13_CLAIM_BOUNDARY,
        build_phase13_tiled_contract,
    )

    report = build_phase13_tiled_contract(
        _write_mapping(tmp_path / "mapping.csv"),
        _write_manifest(tmp_path / "experiment_variants.json"),
        tile_rows=4,
        tile_cols=4,
        observation_threshold=20,
    )

    assert report["total_blocks"] == 5
    assert report["tile_count"] == 2
    assert report["block_count_summary"]["max"] == 3
    assert report["variants"]["B3"]["n_features"] == 4
    assert report["variants"]["B3"]["max_tile_observation_dimension"] == 15
    assert report["all_tiles_within_observation_threshold"] is True
    assert report["tiled_contract_ready"] is True
    assert report["tiles"][0]["tile_id"] == "tile_r000_c000"
    assert report["tiles"][0]["block_ids"] == ["b1", "b2", "b3"]
    assert report["claim_boundary"] == PHASE13_CLAIM_BOUNDARY


def test_phase13_threshold_blocks_contract_when_tile_observation_is_too_large(
    tmp_path,
):
    from paper11_geofm.tiled_contract import build_phase13_tiled_contract

    report = build_phase13_tiled_contract(
        _write_mapping(tmp_path / "mapping.csv"),
        _write_manifest(tmp_path / "experiment_variants.json"),
        tile_rows=4,
        tile_cols=4,
        observation_threshold=14,
    )

    assert report["all_tiles_within_observation_threshold"] is False
    assert report["tiled_contract_ready"] is False
    assert "increase_tile_partitioning" in report["recommendation"]


def test_phase13_writer_outputs_tile_csv_and_json(tmp_path):
    from paper11_geofm.tiled_contract import (
        build_phase13_tiled_contract,
        write_phase13_tiled_contract,
    )

    report = build_phase13_tiled_contract(
        _write_mapping(tmp_path / "mapping.csv"),
        _write_manifest(tmp_path / "experiment_variants.json"),
        tile_rows=4,
        tile_cols=4,
    )
    paths = write_phase13_tiled_contract(report, tmp_path / "outputs")

    assert paths["tile_index"].name == "phase13_tile_index.csv"
    assert paths["summary"].name == "phase13_tiled_real_contract.json"
    with paths["tile_index"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert rows[0]["tile_id"] == "tile_r000_c000"
    assert rows[0]["block_ids"] == "b1;b2;b3"
    assert summary["tile_count"] == 2


def test_phase13_cli_writes_tiled_contract_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase13_tiled_real_contract"
        / "run_phase13_tiled_real_contract.py"
    )
    spec = importlib.util.spec_from_file_location("phase13_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--mapping-csv",
            str(_write_mapping(tmp_path / "mapping.csv")),
            "--variant-manifest",
            str(_write_manifest(tmp_path / "experiment_variants.json")),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--tile-rows",
            "4",
            "--tile-cols",
            "4",
            "--observation-threshold",
            "20",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Tiles: 2" in stdout
    assert "Tiled contract ready: True" in stdout
    assert "phase13_tile_index.csv" in stdout
