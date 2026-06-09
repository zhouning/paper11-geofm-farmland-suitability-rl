import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id, suitability):
    row = {"block_id": block_id, "suitability_proxy": suitability}
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(dim)
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = float(idx)
    return row


def _phase2_test_summary():
    return {
        "metadata_source": "test",
        "base_year_requested": 2020,
        "base_year_used": 2020,
        "years": [2020],
        "grid_shape": [2, 2],
        "embedding_dim": 64,
        "mapping_mode": "test",
    }


def _write_ready_phase2_outputs(output_dir):
    from paper11_geofm.artifacts import write_phase2_artifacts

    rows = [
        _complete_phase2_feature_row("sample_block_00", 0.25),
        _complete_phase2_feature_row("sample_block_01", 0.50),
        _complete_phase2_feature_row("sample_block_02", 0.75),
        _complete_phase2_feature_row("sample_block_03", 1.00),
    ]
    return write_phase2_artifacts(rows, output_dir, _phase2_test_summary())


def _summaries_by_variant(protocol):
    return {row["variant_id"]: row for row in protocol["summaries"]}


def test_phase5_rollout_protocol_runs_all_ready_variants(tmp_path):
    from paper11_geofm.rollout_smoke import (
        PHASE5_CLAIM_BOUNDARY,
        run_phase5_rollout_protocol,
    )

    _write_ready_phase2_outputs(tmp_path)

    protocol = run_phase5_rollout_protocol(tmp_path)

    assert protocol["claim_boundary"] == PHASE5_CLAIM_BOUNDARY
    assert protocol["variant_ids"] == ["B0", "B1", "B2", "B3"]
    summaries = _summaries_by_variant(protocol)
    assert set(summaries) == {"B0", "B1", "B2", "B3"}
    assert summaries["B0"]["n_features"] == 17
    assert summaries["B1"]["n_features"] == 81
    assert summaries["B2"]["n_features"] == 18
    assert summaries["B3"]["n_features"] == 82
    assert summaries["B3"]["observation_shape"] == 331
    assert summaries["B3"]["action_space_n"] == 4

    for summary in summaries.values():
        assert summary["n_blocks"] == 4
        assert summary["episode_steps"] == 4
        assert summary["max_steps"] == 4
        assert summary["terminated"] is True
        assert summary["truncated"] is False
        assert summary["valid_action_rate"] == 1.0
        assert summary["selected_block_ids"] == [
            "sample_block_00",
            "sample_block_01",
            "sample_block_02",
            "sample_block_03",
        ]
        assert summary["claim_boundary"] == PHASE5_CLAIM_BOUNDARY

    assert summaries["B0"]["total_contract_reward"] == 0.0
    assert summaries["B1"]["total_contract_reward"] == 0.0
    assert summaries["B2"]["total_contract_reward"] == 2.5
    assert summaries["B3"]["total_contract_reward"] == 2.5
    assert protocol["steps"]["B3"][0]["selected_block_id"] == "sample_block_00"
    assert protocol["steps"]["B3"][0]["valid_actions_before"] == 4
    assert protocol["steps"]["B3"][0]["valid_actions_after"] == 3


def test_phase5_rollout_respects_max_steps(tmp_path):
    from paper11_geofm.rollout_smoke import run_phase5_rollout_protocol

    _write_ready_phase2_outputs(tmp_path)

    protocol = run_phase5_rollout_protocol(tmp_path, variant_ids=("B3",), max_steps=2)

    summary = protocol["summaries"][0]
    assert summary["variant_id"] == "B3"
    assert summary["max_steps"] == 2
    assert summary["episode_steps"] == 2
    assert summary["terminated"] is True
    assert summary["selected_block_ids"] == ["sample_block_00", "sample_block_01"]
    assert summary["total_contract_reward"] == 0.75
    assert len(protocol["steps"]["B3"]) == 2


def test_phase5_rollout_rejects_empty_variant_list(tmp_path):
    import pytest

    from paper11_geofm.rollout_smoke import run_phase5_rollout_protocol

    _write_ready_phase2_outputs(tmp_path)

    with pytest.raises(ValueError, match="At least one"):
        run_phase5_rollout_protocol(tmp_path, variant_ids=())


def test_phase5_rollout_artifacts_are_written(tmp_path):
    import csv

    from paper11_geofm.rollout_smoke import (
        PHASE5_CLAIM_BOUNDARY,
        run_phase5_rollout_protocol,
        write_phase5_rollout_artifacts,
    )

    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase5"
    _write_ready_phase2_outputs(phase2_dir)
    protocol = run_phase5_rollout_protocol(phase2_dir, variant_ids=("B2", "B3"))

    paths = write_phase5_rollout_artifacts(protocol, output_dir)

    assert paths["summary_csv"].name == "phase5_rollout_summary.csv"
    assert paths["steps_json"].name == "phase5_rollout_steps.json"
    assert paths["summary_csv"].exists()
    assert paths["steps_json"].exists()

    with paths["summary_csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["variant_id"] for row in rows] == ["B2", "B3"]
    assert rows[0]["selected_block_ids"] == (
        "sample_block_00;sample_block_01;sample_block_02;sample_block_03"
    )
    assert rows[0]["claim_boundary"] == PHASE5_CLAIM_BOUNDARY

    saved = json.loads(paths["steps_json"].read_text(encoding="utf-8"))
    assert saved["claim_boundary"] == PHASE5_CLAIM_BOUNDARY
    assert saved["variant_ids"] == ["B2", "B3"]
    assert saved["steps"]["B3"][3]["selected_block_id"] == "sample_block_03"


def test_phase5_rollout_cli_prints_summary_and_artifacts(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "run_phase5_rollout",
        ROOT / "experiments" / "phase5_rollout_protocol" / "run_phase5_rollout.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase5"
    _write_ready_phase2_outputs(phase2_dir)

    exit_code = module.main(
        [
            "--phase2-output-dir",
            str(phase2_dir),
            "--output-dir",
            str(output_dir),
            "--variants",
            "B0,B3",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Variant B0: steps=4 features=17 total_contract_reward=0.000000" in stdout
    assert "Variant B3: steps=4 features=82 total_contract_reward=2.500000" in stdout
    assert "Summary CSV:" in stdout
    assert "Steps JSON:" in stdout
    assert (
        "Claim boundary: Phase 5 is a deterministic rollout-protocol smoke check"
        in stdout
    )
    assert (output_dir / "phase5_rollout_summary.csv").exists()
    assert (output_dir / "phase5_rollout_steps.json").exists()
