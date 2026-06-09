import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
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

    return write_phase2_artifacts(
        [
            _complete_phase2_feature_row("sample_block_00"),
            _complete_phase2_feature_row("sample_block_01"),
        ],
        output_dir,
        _phase2_test_summary(),
    )


def test_load_variant_input_reads_ready_b3_matrix(tmp_path):
    from paper11_geofm.drl_inputs import load_variant_input

    _write_ready_phase2_outputs(tmp_path)

    loaded = load_variant_input(tmp_path, "b3")

    assert loaded.variant_id == "B3"
    assert loaded.block_ids == ("sample_block_00", "sample_block_01")
    assert loaded.feature_columns[0] == "explicit_feature_00"
    assert loaded.feature_columns[-1] == "suitability_proxy"
    assert loaded.reward_mode == "base_plus_suitability_reward"
    assert loaded.state_groups == (
        "explicit_planning_features",
        "geofm_embedding",
        "suitability_proxy",
    )
    assert loaded.source_table.name == "variant_B3_features.csv"
    assert loaded.state_matrix.dtype == np.float32
    assert loaded.state_matrix.shape == (2, 82)
    assert loaded.state_matrix[0, 0] == np.float32(0.0)
    assert loaded.state_matrix[0, 80] == np.float32(63.0)
    assert loaded.state_matrix[0, 81] == np.float32(0.75)


def test_load_variant_input_reports_expected_dimensions(tmp_path):
    from paper11_geofm.drl_inputs import load_variant_input

    _write_ready_phase2_outputs(tmp_path)

    expected = {
        "B0": 17,
        "B1": 81,
        "B2": 18,
        "B3": 82,
    }
    manifest = json.loads(
        (tmp_path / "experiment_variants.json").read_text(encoding="utf-8")
    )
    for variant_id, feature_count in expected.items():
        loaded = load_variant_input(tmp_path, variant_id)
        assert loaded.feature_columns == tuple(
            manifest["variants"][variant_id]["required_columns"]
        )
        assert loaded.state_matrix.shape == (2, feature_count)


def test_load_variant_input_rejects_incomplete_variant(tmp_path):
    from paper11_geofm.artifacts import write_phase2_artifacts
    from paper11_geofm.drl_inputs import load_variant_input

    write_phase2_artifacts(
        [{"block_id": "b0", "suitability_proxy": 0.5}],
        tmp_path,
        _phase2_test_summary(),
    )

    with pytest.raises(ValueError, match="B3 is not ready"):
        load_variant_input(tmp_path, "B3")


def test_load_variant_input_rejects_duplicate_block_ids(tmp_path):
    from paper11_geofm.drl_inputs import load_variant_input

    _write_ready_phase2_outputs(tmp_path)
    table = tmp_path / "variant_B0_features.csv"
    rows = list(csv.DictReader(table.open("r", encoding="utf-8", newline="")))
    rows[1]["block_id"] = rows[0]["block_id"]
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Duplicate block_id"):
        load_variant_input(tmp_path, "B0")


def test_load_variant_input_rejects_non_numeric_required_values(tmp_path):
    from paper11_geofm.drl_inputs import load_variant_input

    _write_ready_phase2_outputs(tmp_path)
    table = tmp_path / "variant_B0_features.csv"
    rows = list(csv.DictReader(table.open("r", encoding="utf-8", newline="")))
    rows[0]["explicit_feature_00"] = "not-a-number"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Non-numeric value"):
        load_variant_input(tmp_path, "B0")


def test_inspect_variant_inputs_cli_prints_contract_summary(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "inspect_variant_inputs",
        ROOT / "experiments" / "phase3_drl_input_adapter" / "inspect_variant_inputs.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    _write_ready_phase2_outputs(tmp_path)

    exit_code = module.main(
        [
            "--phase2-output-dir",
            str(tmp_path),
            "--variant",
            "B3",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Variant: B3" in stdout
    assert "Rows: 2" in stdout
    assert "Features: 82" in stdout
    assert "Matrix shape: 2 x 82" in stdout
    assert "Reward mode: base_plus_suitability_reward" in stdout
    assert "Claim boundary: input contract only; no DRL policy is trained or evaluated." in stdout
