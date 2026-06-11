import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id):
    row = {"block_id": block_id, "suitability_proxy": 0.75}
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(dim)
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = 0.0
    row.update(
        {
            "explicit_feature_00": 2.5,
            "explicit_feature_01": 10.0,
            "explicit_feature_02": 28.0,
            "explicit_feature_04": 1.0,
            "explicit_feature_07": 0.0,
            "explicit_feature_09": 0.0,
            "explicit_feature_10": 0.0,
            "explicit_feature_13": 1.0,
            "explicit_feature_16": 1.0,
        }
    )
    return row


def _write_ready_phase2_outputs(output_dir):
    from paper11_geofm.artifacts import write_phase2_artifacts

    return write_phase2_artifacts(
        [
            _complete_phase2_feature_row(block_id)
            for block_id in ["b1", "b2", "b3", "b4"]
        ],
        output_dir,
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
                "tile_id": "tile_r000_c000",
                "tile_row": 0,
                "tile_col": 0,
                "n_blocks": 3,
                "block_ids": "b1;b3;b4",
            }
        )
        writer.writerow(
            {
                "tile_id": "tile_r000_c001",
                "tile_row": 0,
                "tile_col": 1,
                "n_blocks": 1,
                "block_ids": "b2",
            }
        )
    return path


def test_phase15_runs_batch_smoke_for_all_tiles(tmp_path):
    from paper11_geofm.tiled_batch_smoke import (
        PHASE15_CLAIM_BOUNDARY,
        run_phase15_tiled_batch_smoke,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    report = run_phase15_tiled_batch_smoke(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variant_id="B1",
    )

    assert report["tile_count"] == 2
    assert report["total_blocks"] == 4
    assert report["block_count_summary"]["max"] == 3
    assert report["max_observation_shape"] == 246
    assert report["all_tile_smokes_passed"] is True
    assert report["rows"][0]["tile_id"] == "tile_r000_c000"
    assert report["rows"][0]["selected_block_id"] == "b1"
    assert report["rows"][0]["step_reward"] == 0.6
    assert report["claim_boundary"] == PHASE15_CLAIM_BOUNDARY


def test_phase15_max_tiles_caps_batch(tmp_path):
    from paper11_geofm.tiled_batch_smoke import run_phase15_tiled_batch_smoke

    _write_ready_phase2_outputs(tmp_path / "phase2")
    report = run_phase15_tiled_batch_smoke(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variant_id="B1",
        max_tiles=1,
    )

    assert report["tile_count"] == 1
    assert report["total_blocks"] == 3
    assert [row["tile_id"] for row in report["rows"]] == ["tile_r000_c000"]


def test_phase15_rejects_suitability_reward_variant_by_default(tmp_path):
    from paper11_geofm.tiled_batch_smoke import run_phase15_tiled_batch_smoke

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with pytest.raises(ValueError, match="suitability reward variants are disabled"):
        run_phase15_tiled_batch_smoke(
            tmp_path / "phase2",
            _write_tile_index(tmp_path / "phase13_tile_index.csv"),
            variant_id="B3",
        )


def test_phase15_writer_outputs_csv_and_json(tmp_path):
    from paper11_geofm.tiled_batch_smoke import (
        run_phase15_tiled_batch_smoke,
        write_phase15_tiled_batch_smoke,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    report = run_phase15_tiled_batch_smoke(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variant_id="B1",
    )
    paths = write_phase15_tiled_batch_smoke(report, tmp_path / "outputs")

    assert paths["summary_csv"].name == "phase15_tiled_batch_smoke_summary.csv"
    assert paths["report_json"].name == "phase15_tiled_batch_smoke_report.json"
    with paths["summary_csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    written = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    assert rows[0]["tile_id"] == "tile_r000_c000"
    assert rows[0]["status"] == "passed"
    assert written["tile_count"] == 2


def test_phase15_cli_writes_batch_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase15_tiled_batch_smoke"
        / "run_phase15_tiled_batch_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("phase15_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    _write_ready_phase2_outputs(tmp_path / "phase2")
    exit_code = module.main(
        [
            "--phase2-output-dir",
            str(tmp_path / "phase2"),
            "--tile-index-csv",
            str(_write_tile_index(tmp_path / "phase13_tile_index.csv")),
            "--variant",
            "B1",
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Tiles processed: 2" in stdout
    assert "All passed: True" in stdout
    assert "phase15_tiled_batch_smoke_summary.csv" in stdout
