import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id, suitability=0.75):
    row = {"block_id": block_id, "suitability_proxy": suitability}
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
            _complete_phase2_feature_row("b1", 0.25),
            _complete_phase2_feature_row("b2", 0.50),
            _complete_phase2_feature_row("b3", 0.75),
            _complete_phase2_feature_row("b4", 1.00),
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


def test_phase16_runs_default_policies_for_all_tiles(tmp_path):
    from paper11_geofm.tiled_baseline_protocol import (
        PHASE16_CLAIM_BOUNDARY,
        run_phase16_tiled_baseline_protocol,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    protocol = run_phase16_tiled_baseline_protocol(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variant_id="B1",
        max_steps=2,
        seed=0,
    )

    assert protocol["tile_count"] == 2
    assert protocol["policy_ids"] == ["first_valid", "seeded_random"]
    assert protocol["summary_count"] == 4
    assert protocol["total_blocks"] == 4
    assert protocol["max_observation_shape"] == 246
    assert protocol["all_rollouts_completed"] is True
    assert protocol["summaries"][0]["policy_id"] == "first_valid"
    assert protocol["summaries"][0]["tile_id"] == "tile_r000_c000"
    assert protocol["summaries"][0]["episode_steps"] == 2
    assert protocol["summaries"][0]["selected_block_ids"] == ["b1", "b3"]
    assert protocol["summaries"][0]["total_contract_reward"] == 1.2
    assert protocol["claim_boundary"] == PHASE16_CLAIM_BOUNDARY


def test_phase16_seeded_random_is_reproducible_and_seed_sensitive(tmp_path):
    from paper11_geofm.tiled_baseline_protocol import (
        run_phase16_tiled_baseline_protocol,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    tile_index_csv = _write_tile_index(tmp_path / "phase13_tile_index.csv")

    first = run_phase16_tiled_baseline_protocol(
        tmp_path / "phase2",
        tile_index_csv,
        policy_ids=("seeded_random",),
        max_steps=3,
        seed=0,
    )
    repeated = run_phase16_tiled_baseline_protocol(
        tmp_path / "phase2",
        tile_index_csv,
        policy_ids=("seeded_random",),
        max_steps=3,
        seed=0,
    )
    changed = run_phase16_tiled_baseline_protocol(
        tmp_path / "phase2",
        tile_index_csv,
        policy_ids=("seeded_random",),
        max_steps=3,
        seed=1,
    )

    first_order = [
        step["selected_block_id"]
        for step in first["traces"]["seeded_random"]["tile_r000_c000"]
    ]
    repeated_order = [
        step["selected_block_id"]
        for step in repeated["traces"]["seeded_random"]["tile_r000_c000"]
    ]
    changed_order = [
        step["selected_block_id"]
        for step in changed["traces"]["seeded_random"]["tile_r000_c000"]
    ]

    assert first_order == repeated_order
    assert first_order != changed_order


def test_phase16_max_tiles_caps_tile_rows_not_policies(tmp_path):
    from paper11_geofm.tiled_baseline_protocol import (
        run_phase16_tiled_baseline_protocol,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    protocol = run_phase16_tiled_baseline_protocol(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        max_tiles=1,
    )

    assert protocol["tile_count"] == 1
    assert protocol["summary_count"] == 2
    assert protocol["total_blocks"] == 3
    assert [row["tile_id"] for row in protocol["summaries"]] == [
        "tile_r000_c000",
        "tile_r000_c000",
    ]


def test_phase16_rejects_suitability_reward_variant_by_default(tmp_path):
    from paper11_geofm.tiled_baseline_protocol import (
        run_phase16_tiled_baseline_protocol,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with pytest.raises(ValueError, match="suitability reward variants are disabled"):
        run_phase16_tiled_baseline_protocol(
            tmp_path / "phase2",
            _write_tile_index(tmp_path / "phase13_tile_index.csv"),
            variant_id="B3",
        )


def test_phase16_writer_outputs_summary_csv_and_trace_json(tmp_path):
    from paper11_geofm.tiled_baseline_protocol import (
        PHASE16_CLAIM_BOUNDARY,
        run_phase16_tiled_baseline_protocol,
        write_phase16_tiled_baseline_artifacts,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    protocol = run_phase16_tiled_baseline_protocol(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        max_steps=2,
    )
    paths = write_phase16_tiled_baseline_artifacts(protocol, tmp_path / "outputs")

    assert paths["summary_csv"].name == "phase16_tiled_baseline_summary.csv"
    assert paths["traces_json"].name == "phase16_tiled_baseline_traces.json"
    assert paths["summary_csv"].exists()
    assert paths["traces_json"].exists()

    with paths["summary_csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    saved = json.loads(paths["traces_json"].read_text(encoding="utf-8"))

    assert len(rows) == 4
    assert rows[0]["policy_id"] == "first_valid"
    assert rows[0]["tile_id"] == "tile_r000_c000"
    assert rows[0]["selected_block_ids"] == "b1;b3"
    assert rows[0]["claim_boundary"] == PHASE16_CLAIM_BOUNDARY
    assert saved["summary_count"] == 4
    assert saved["traces"]["first_valid"]["tile_r000_c000"][1][
        "selected_block_id"
    ] == "b3"


def test_phase16_cli_writes_outputs_and_prints_summary(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase16_tiled_baseline_protocol"
        / "run_phase16_tiled_baselines.py"
    )
    spec = importlib.util.spec_from_file_location("phase16_runner", runner_path)
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
            "--policies",
            "first_valid,seeded_random",
            "--max-steps",
            "2",
            "--seed",
            "0",
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Tiles processed: 2" in stdout
    assert "Policies: 2" in stdout
    assert "Summary rows: 4" in stdout
    assert "All completed: True" in stdout
    assert "phase16_tiled_baseline_summary.csv" in stdout
    assert "phase16_tiled_baseline_traces.json" in stdout
