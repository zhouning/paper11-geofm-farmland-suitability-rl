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


def _summaries_by_policy_variant(protocol):
    return {
        (row["policy_id"], row["variant_id"]): row
        for row in protocol["summaries"]
    }


def test_phase6_runs_default_policies_for_all_ready_variants(tmp_path):
    from paper11_geofm.baseline_eval import (
        PHASE6_CLAIM_BOUNDARY,
        run_phase6_baseline_evaluator,
    )

    _write_ready_phase2_outputs(tmp_path)

    protocol = run_phase6_baseline_evaluator(tmp_path)

    assert protocol["claim_boundary"] == PHASE6_CLAIM_BOUNDARY
    assert protocol["variant_ids"] == ["B0", "B1", "B2", "B3"]
    assert protocol["policy_ids"] == ["first_valid", "seeded_random"]
    assert protocol["seed"] == 0
    assert len(protocol["summaries"]) == 8

    summaries = _summaries_by_policy_variant(protocol)
    assert summaries[("first_valid", "B0")]["n_features"] == 17
    assert summaries[("first_valid", "B1")]["n_features"] == 81
    assert summaries[("first_valid", "B2")]["n_features"] == 18
    assert summaries[("first_valid", "B3")]["n_features"] == 82
    assert summaries[("first_valid", "B3")]["observation_shape"] == 331
    assert summaries[("first_valid", "B3")]["action_space_n"] == 4

    expected_order = [
        "sample_block_00",
        "sample_block_01",
        "sample_block_02",
        "sample_block_03",
    ]
    assert summaries[("first_valid", "B3")]["selected_block_ids"] == expected_order
    assert summaries[("first_valid", "B0")]["total_contract_reward"] == 2.4
    assert summaries[("first_valid", "B1")]["total_contract_reward"] == 2.4
    assert summaries[("first_valid", "B2")]["total_contract_reward"] == 4.9
    assert summaries[("first_valid", "B3")]["total_contract_reward"] == 4.9

    for summary in summaries.values():
        assert summary["seed"] == 0
        assert summary["n_blocks"] == 4
        assert summary["episode_steps"] == 4
        assert summary["max_steps"] == 4
        assert summary["terminated"] is True
        assert summary["truncated"] is False
        assert summary["valid_action_rate"] == 1.0
        assert summary["claim_boundary"] == PHASE6_CLAIM_BOUNDARY

    assert protocol["traces"]["first_valid"]["B3"][0]["selected_block_id"] == (
        "sample_block_00"
    )
    assert protocol["traces"]["first_valid"]["B3"][0]["valid_actions_before"] == 4
    assert protocol["traces"]["first_valid"]["B3"][0]["valid_actions_after"] == 3


def test_phase6_seeded_random_is_reproducible_and_seed_sensitive(tmp_path):
    from paper11_geofm.baseline_eval import run_phase6_baseline_evaluator

    _write_ready_phase2_outputs(tmp_path)

    first = run_phase6_baseline_evaluator(
        tmp_path,
        variant_ids=("B3",),
        policy_ids=("seeded_random",),
        seed=0,
    )
    repeated = run_phase6_baseline_evaluator(
        tmp_path,
        variant_ids=("B3",),
        policy_ids=("seeded_random",),
        seed=0,
    )
    changed = run_phase6_baseline_evaluator(
        tmp_path,
        variant_ids=("B3",),
        policy_ids=("seeded_random",),
        seed=1,
    )

    first_order = [
        step["selected_block_id"] for step in first["traces"]["seeded_random"]["B3"]
    ]
    repeated_order = [
        step["selected_block_id"] for step in repeated["traces"]["seeded_random"]["B3"]
    ]
    changed_order = [
        step["selected_block_id"] for step in changed["traces"]["seeded_random"]["B3"]
    ]

    assert first_order == repeated_order
    assert first_order != changed_order


def test_phase6_baselines_respect_max_steps(tmp_path):
    from paper11_geofm.baseline_eval import run_phase6_baseline_evaluator

    _write_ready_phase2_outputs(tmp_path)

    protocol = run_phase6_baseline_evaluator(
        tmp_path,
        variant_ids=("B3",),
        policy_ids=("first_valid", "seeded_random"),
        max_steps=2,
        seed=0,
    )

    summaries = _summaries_by_policy_variant(protocol)
    for summary in summaries.values():
        assert summary["max_steps"] == 2
        assert summary["episode_steps"] == 2
        assert summary["terminated"] is True
        assert len(summary["selected_block_ids"]) == 2
    assert summaries[("first_valid", "B3")]["selected_block_ids"] == [
        "sample_block_00",
        "sample_block_01",
    ]
    assert summaries[("first_valid", "B3")]["total_contract_reward"] == 1.95


def test_phase6_rejects_unknown_policy(tmp_path):
    import pytest

    from paper11_geofm.baseline_eval import run_phase6_baseline_evaluator

    _write_ready_phase2_outputs(tmp_path)

    with pytest.raises(ValueError, match="Unknown Phase 6 policy"):
        run_phase6_baseline_evaluator(tmp_path, policy_ids=("first_valid", "learned"))


def test_phase6_baseline_artifacts_are_written(tmp_path):
    import csv

    from paper11_geofm.baseline_eval import (
        PHASE6_CLAIM_BOUNDARY,
        run_phase6_baseline_evaluator,
        write_phase6_baseline_artifacts,
    )

    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase6"
    _write_ready_phase2_outputs(phase2_dir)
    protocol = run_phase6_baseline_evaluator(
        phase2_dir,
        variant_ids=("B2", "B3"),
        policy_ids=("first_valid", "seeded_random"),
        seed=0,
    )

    paths = write_phase6_baseline_artifacts(protocol, output_dir)

    assert paths["summary_csv"].name == "phase6_baseline_summary.csv"
    assert paths["traces_json"].name == "phase6_baseline_traces.json"
    assert paths["summary_csv"].exists()
    assert paths["traces_json"].exists()

    with paths["summary_csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["policy_id"] == "first_valid"
    assert rows[0]["variant_id"] == "B2"
    assert rows[0]["selected_block_ids"] == (
        "sample_block_00;sample_block_01;sample_block_02;sample_block_03"
    )
    assert rows[0]["claim_boundary"] == PHASE6_CLAIM_BOUNDARY

    saved = json.loads(paths["traces_json"].read_text(encoding="utf-8"))
    assert saved["claim_boundary"] == PHASE6_CLAIM_BOUNDARY
    assert saved["policy_ids"] == ["first_valid", "seeded_random"]
    assert saved["variant_ids"] == ["B2", "B3"]
    assert saved["traces"]["first_valid"]["B3"][3]["selected_block_id"] == (
        "sample_block_03"
    )


def test_phase6_baseline_cli_prints_summary_and_artifacts(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "run_phase6_baselines",
        ROOT
        / "experiments"
        / "phase6_masked_baselines"
        / "run_phase6_baselines.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase6"
    _write_ready_phase2_outputs(phase2_dir)

    exit_code = module.main(
        [
            "--phase2-output-dir",
            str(phase2_dir),
            "--output-dir",
            str(output_dir),
            "--variants",
            "B0,B3",
            "--policies",
            "first_valid,seeded_random",
            "--seed",
            "0",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "Policy first_valid / Variant B0: "
        "steps=4 features=17 total_contract_reward=2.400000"
    ) in stdout
    assert "Policy seeded_random / Variant B3:" in stdout
    assert "Summary CSV:" in stdout
    assert "Trace JSON:" in stdout
    assert (
        "Claim boundary: Phase 6 is a non-learning masked baseline evaluator"
        in stdout
    )
    assert (output_dir / "phase6_baseline_summary.csv").exists()
    assert (output_dir / "phase6_baseline_traces.json").exists()
