import csv
import faulthandler
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


pytestmark = pytest.mark.filterwarnings("ignore:XPU device count is zero!:UserWarning")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id, suitability=0.75):
    row = {"block_id": block_id, "suitability_proxy": suitability}
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(dim)
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = float(idx)
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
                "n_blocks": 1,
                "block_ids": "b2",
            }
        )
        writer.writerow(
            {
                "tile_id": "tile_r000_c001",
                "tile_row": 0,
                "tile_col": 1,
                "n_blocks": 3,
                "block_ids": "b1;b3;b4",
            }
        )
    return path


def test_phase17_selects_largest_tile_and_reports_contract(tmp_path):
    from paper11_geofm.tiled_maskableppo_readiness import (
        PHASE17_CLAIM_BOUNDARY,
        build_phase17_tiled_contract_summary,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    summary = build_phase17_tiled_contract_summary(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variant_id="B1",
        tile_selection="largest",
        seed=0,
        total_timesteps=8,
    )

    assert summary["phase"] == "phase17_tiled_maskableppo_readiness"
    assert summary["tile_id"] == "tile_r000_c001"
    assert summary["tile_selection"] == "largest"
    assert summary["variant_id"] == "B1"
    assert summary["seed"] == 0
    assert summary["learn_timesteps"] == 8
    assert summary["n_blocks"] == 3
    assert summary["n_features"] == 81
    assert summary["observation_shape"] == 246
    assert summary["action_space_n"] == 3
    assert summary["reward_mode"] == "base_planning_reward"
    assert summary["initial_valid_actions"] == 3
    assert summary["claim_boundary"] == PHASE17_CLAIM_BOUNDARY


def test_phase17_tile_id_override_selects_requested_tile(tmp_path):
    from paper11_geofm.tiled_maskableppo_readiness import (
        build_phase17_tiled_contract_summary,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    summary = build_phase17_tiled_contract_summary(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variant_id="B1",
        tile_id="tile_r000_c000",
    )

    assert summary["tile_id"] == "tile_r000_c000"
    assert summary["tile_selection"] == "explicit"
    assert summary["n_blocks"] == 1
    assert summary["observation_shape"] == 84


def test_phase17_rejects_suitability_reward_variant_by_default(tmp_path):
    from paper11_geofm.tiled_maskableppo_readiness import (
        build_phase17_tiled_contract_summary,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with pytest.raises(ValueError, match="suitability reward variants are disabled"):
        build_phase17_tiled_contract_summary(
            tmp_path / "phase2",
            _write_tile_index(tmp_path / "phase13_tile_index.csv"),
            variant_id="B3",
        )


@contextmanager
def _torch_windows_faulthandler_guard():
    was_enabled = faulthandler.is_enabled()
    if was_enabled:
        faulthandler.disable()
    try:
        yield
    finally:
        if was_enabled:
            faulthandler.enable(file=sys.__stderr__)


def _require_maskableppo_dependencies():
    with _torch_windows_faulthandler_guard():
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("sb3_contrib")


def test_phase17_runs_tiny_tiled_maskableppo_smoke(tmp_path):
    _require_maskableppo_dependencies()
    from paper11_geofm.tiled_maskableppo_readiness import (
        PHASE17_CLAIM_BOUNDARY,
        run_phase17_tiled_maskableppo_readiness,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with _torch_windows_faulthandler_guard():
        summary = run_phase17_tiled_maskableppo_readiness(
            tmp_path / "phase2",
            _write_tile_index(tmp_path / "phase13_tile_index.csv"),
            variant_id="B1",
            total_timesteps=8,
            seed=0,
        )

    assert summary["masking_supported"] is True
    assert summary["learn_timesteps"] == 8
    assert summary["device"] == "cpu"
    assert summary["predicted_action_valid"] is True
    assert 0 <= summary["predicted_action"] < summary["action_space_n"]
    assert str(summary["selected_block_id"]) in {"b1", "b3", "b4"}
    assert summary["readiness_status"] == "passed_tiled_maskableppo_smoke"
    assert (
        summary["recommendation"]
        == "tiled_maskableppo_contract_ready_for_larger_controlled_smokes"
    )
    assert summary["dependencies"]["stable_baselines3"]["available"] is True
    assert summary["dependencies"]["sb3_contrib"]["available"] is True
    assert summary["claim_boundary"] == PHASE17_CLAIM_BOUNDARY


def test_phase17_writer_outputs_json(tmp_path):
    from paper11_geofm.tiled_maskableppo_readiness import (
        PHASE17_CLAIM_BOUNDARY,
        write_phase17_tiled_maskableppo_readiness_artifact,
    )

    summary = {
        "phase": "phase17_tiled_maskableppo_readiness",
        "tile_id": "tile_r000_c001",
        "readiness_status": "passed_tiled_maskableppo_smoke",
        "claim_boundary": PHASE17_CLAIM_BOUNDARY,
    }
    path = write_phase17_tiled_maskableppo_readiness_artifact(
        summary,
        tmp_path / "outputs",
    )

    assert path.name == "phase17_tiled_maskableppo_readiness.json"
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == summary


def test_phase17_cli_writes_outputs_and_prints_summary(tmp_path, capsys):
    _require_maskableppo_dependencies()
    runner_path = (
        ROOT
        / "experiments"
        / "phase17_tiled_maskableppo_readiness"
        / "run_phase17_tiled_maskableppo_readiness.py"
    )
    spec = importlib.util.spec_from_file_location("phase17_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with _torch_windows_faulthandler_guard():
        exit_code = module.main(
            [
                "--phase2-output-dir",
                str(tmp_path / "phase2"),
                "--tile-index-csv",
                str(_write_tile_index(tmp_path / "phase13_tile_index.csv")),
                "--variant",
                "B1",
                "--total-timesteps",
                "8",
                "--seed",
                "0",
                "--output-dir",
                str(tmp_path / "outputs"),
            ]
        )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Tile: tile_r000_c001" in stdout
    assert "Variant: B1" in stdout
    assert "Masking supported: True" in stdout
    assert "Predicted action valid: True" in stdout
    assert "Readiness status: passed_tiled_maskableppo_smoke" in stdout
    assert "phase17_tiled_maskableppo_readiness.json" in stdout
