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
        row[f"explicit_feature_{idx:02d}"] = float(idx)
    return row


def _write_ready_phase2_outputs(output_dir):
    from paper11_geofm.artifacts import write_phase2_artifacts

    return write_phase2_artifacts(
        [
            _complete_phase2_feature_row("b1"),
            _complete_phase2_feature_row("b2"),
            _complete_phase2_feature_row("b3"),
            _complete_phase2_feature_row("b4"),
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
                "tile_id": "tile_r000_c000",
                "tile_row": 0,
                "tile_col": 0,
                "n_blocks": 3,
                "min_grid_row": 0,
                "max_grid_row": 3,
                "min_grid_col": 0,
                "max_grid_col": 3,
                "block_ids": "b1;b3;b4",
            }
        )
    return path


def test_phase14_loads_tiled_b1_variant_subset_in_tile_order(tmp_path):
    from paper11_geofm.tiled_inputs import (
        PHASE14_CLAIM_BOUNDARY,
        load_tiled_variant_input,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    loaded = load_tiled_variant_input(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        "tile_r000_c000",
        "B1",
    )

    assert loaded.tile_id == "tile_r000_c000"
    assert loaded.variant_id == "B1"
    assert loaded.block_ids == ("b1", "b3", "b4")
    assert loaded.state_matrix.shape == (3, 81)
    assert loaded.reward_mode == "base_planning_reward"
    assert loaded.claim_boundary == PHASE14_CLAIM_BOUNDARY


def test_phase14_runs_one_step_tiled_b1_smoke(tmp_path):
    from paper11_geofm.tiled_inputs import run_phase14_tiled_smoke

    _write_ready_phase2_outputs(tmp_path / "phase2")
    summary = run_phase14_tiled_smoke(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        "tile_r000_c000",
        variant_id="B1",
    )

    assert summary["tile_id"] == "tile_r000_c000"
    assert summary["variant_id"] == "B1"
    assert summary["n_blocks"] == 3
    assert summary["n_features"] == 81
    assert summary["observation_shape"] == 246
    assert summary["action_space_n"] == 3
    assert summary["selected_block_id"] == "b1"
    assert summary["step_reward"] == 0.0
    assert summary["reward_mode"] == "base_planning_reward"


def test_phase14_rejects_suitability_reward_variants_by_default(tmp_path):
    from paper11_geofm.tiled_inputs import load_tiled_variant_input

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with pytest.raises(ValueError, match="suitability reward variants are disabled"):
        load_tiled_variant_input(
            tmp_path / "phase2",
            _write_tile_index(tmp_path / "phase13_tile_index.csv"),
            "tile_r000_c000",
            "B3",
        )


def test_phase14_writer_outputs_summary_json(tmp_path):
    from paper11_geofm.tiled_inputs import (
        run_phase14_tiled_smoke,
        write_phase14_tiled_smoke_summary,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    summary = run_phase14_tiled_smoke(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        "tile_r000_c000",
    )
    path = write_phase14_tiled_smoke_summary(summary, tmp_path / "outputs")

    assert path.name == "phase14_tiled_smoke_summary.json"
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["tile_id"] == "tile_r000_c000"
    assert written["observation_shape"] == 246


def test_phase14_cli_writes_tiled_smoke_summary(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase14_tiled_smoke_env"
        / "run_phase14_tiled_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("phase14_runner", runner_path)
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
            "--tile-id",
            "tile_r000_c000",
            "--variant",
            "B1",
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Tile: tile_r000_c000" in stdout
    assert "Variant: B1" in stdout
    assert "Rows: 3" in stdout
    assert "phase14_tiled_smoke_summary.json" in stdout
