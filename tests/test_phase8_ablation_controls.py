import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id, suitability, block_offset):
    row = {"block_id": block_id, "suitability_proxy": suitability}
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(block_offset + dim)
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = float(block_offset / 100.0 + idx)
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
        _complete_phase2_feature_row("sample_block_00", 0.25, 0),
        _complete_phase2_feature_row("sample_block_01", 0.50, 100),
        _complete_phase2_feature_row("sample_block_02", 0.75, 200),
        _complete_phase2_feature_row("sample_block_03", 1.00, 300),
    ]
    return write_phase2_artifacts(rows, output_dir, _phase2_test_summary())


def _matrix_from_rows(rows, columns):
    return np.asarray(
        [[float(row[column]) for column in columns] for row in rows],
        dtype=float,
    )


def test_phase8_builds_expected_ablation_control_tables(tmp_path):
    from paper11_geofm.ablation_controls import (
        PHASE8_CLAIM_BOUNDARY,
        build_phase8_ablation_controls,
    )

    _write_ready_phase2_outputs(tmp_path)

    protocol = build_phase8_ablation_controls(tmp_path, seed=0)

    assert protocol["phase"] == "phase8_ablation_control_features"
    assert protocol["variant_ids"] == ["D2", "D3", "D4P8", "D4P16"]
    assert protocol["seed"] == 0
    assert protocol["source_variants"]["B0"]["n_features"] == 17
    assert protocol["source_variants"]["B1"]["n_features"] == 81
    assert protocol["summary"]["D2"]["n_features"] == 81
    assert protocol["summary"]["D3"]["n_features"] == 81
    assert protocol["summary"]["D4P8"]["n_features"] == 25
    assert protocol["summary"]["D4P16"]["n_features"] == 33
    assert protocol["manifest"]["variants"]["D2"]["ready"] is True
    assert protocol["manifest"]["variants"]["D4P16"]["row_count"] == 4
    assert protocol["claim_boundary"] == PHASE8_CLAIM_BOUNDARY

    assert len(protocol["variant_tables"]["D2"]) == 4
    assert len(protocol["variant_tables"]["D4P16"]) == 4


def test_phase8_random_control_is_reproducible_and_seed_sensitive(tmp_path):
    from paper11_geofm.ablation_controls import build_phase8_ablation_controls
    from paper11_geofm.block_schema import EMBEDDING_COLUMNS
    from paper11_geofm.drl_inputs import load_variant_input

    _write_ready_phase2_outputs(tmp_path)

    first = build_phase8_ablation_controls(tmp_path, seed=0)
    repeated = build_phase8_ablation_controls(tmp_path, seed=0)
    changed = build_phase8_ablation_controls(tmp_path, seed=1)

    first_values = _matrix_from_rows(
        first["variant_tables"]["D2"],
        EMBEDDING_COLUMNS,
    )
    repeated_values = _matrix_from_rows(
        repeated["variant_tables"]["D2"],
        EMBEDDING_COLUMNS,
    )
    changed_values = _matrix_from_rows(
        changed["variant_tables"]["D2"],
        EMBEDDING_COLUMNS,
    )
    source_b1 = load_variant_input(tmp_path, "B1")
    source_embedding_values = source_b1.state_matrix[:, 17:]

    np.testing.assert_allclose(first_values, repeated_values)
    assert not np.allclose(first_values, changed_values)
    assert not np.allclose(first_values, source_embedding_values)


def test_phase8_shuffled_control_uses_non_identity_permutation(tmp_path):
    from paper11_geofm.ablation_controls import build_phase8_ablation_controls
    from paper11_geofm.block_schema import EMBEDDING_COLUMNS
    from paper11_geofm.drl_inputs import load_variant_input

    _write_ready_phase2_outputs(tmp_path)

    protocol = build_phase8_ablation_controls(tmp_path, seed=0)

    permutation = protocol["summary"]["D3"]["shuffle_permutation"]
    assert len(permutation) == 4
    assert permutation != [0, 1, 2, 3]

    d3_values = _matrix_from_rows(
        protocol["variant_tables"]["D3"],
        EMBEDDING_COLUMNS,
    )
    source_b1 = load_variant_input(tmp_path, "B1")
    source_embedding_values = source_b1.state_matrix[:, 17:]
    np.testing.assert_allclose(d3_values, source_embedding_values[permutation])


def test_phase8_pca_controls_emit_requested_dimensions_with_padding(tmp_path):
    from paper11_geofm.ablation_controls import build_phase8_ablation_controls

    _write_ready_phase2_outputs(tmp_path)

    protocol = build_phase8_ablation_controls(tmp_path, seed=0)
    d4p8_columns = [
        column
        for column in protocol["manifest"]["variants"]["D4P8"]["required_columns"]
        if column.startswith("embedding_pca_")
    ]
    d4p16_columns = [
        column
        for column in protocol["manifest"]["variants"]["D4P16"]["required_columns"]
        if column.startswith("embedding_pca_")
    ]

    assert d4p8_columns == [f"embedding_pca_{index:02d}" for index in range(8)]
    assert d4p16_columns == [f"embedding_pca_{index:02d}" for index in range(16)]

    d4p16_values = _matrix_from_rows(
        protocol["variant_tables"]["D4P16"],
        d4p16_columns,
    )
    assert not np.allclose(d4p16_values[:, 0], 0.0)
    np.testing.assert_allclose(d4p16_values[:, 1:], 0.0)


def test_phase8_ablation_artifacts_are_written_and_loadable(tmp_path):
    from paper11_geofm.ablation_controls import (
        PHASE8_CLAIM_BOUNDARY,
        build_phase8_ablation_controls,
        write_phase8_ablation_artifacts,
    )
    from paper11_geofm.drl_inputs import load_variant_input

    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase8"
    _write_ready_phase2_outputs(phase2_dir)
    protocol = build_phase8_ablation_controls(phase2_dir, seed=0)

    paths = write_phase8_ablation_artifacts(protocol, output_dir)

    assert paths["manifest"].name == "experiment_variants.json"
    assert paths["summary"].name == "phase8_ablation_control_summary.json"
    assert paths["variant_tables"]["D2"].name == "variant_D2_features.csv"
    assert paths["variant_tables"]["D4P16"].name == "variant_D4P16_features.csv"

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert manifest["claim_boundary"] == PHASE8_CLAIM_BOUNDARY
    assert summary["claim_boundary"] == PHASE8_CLAIM_BOUNDARY
    assert "variant_tables" not in summary

    loaded_d2 = load_variant_input(output_dir, "D2")
    loaded_d4p16 = load_variant_input(output_dir, "D4P16")
    assert loaded_d2.state_matrix.shape == (4, 81)
    assert loaded_d4p16.state_matrix.shape == (4, 33)


def test_phase8_ablation_cli_prints_summary_and_artifacts(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "run_phase8_ablation_controls",
        ROOT
        / "experiments"
        / "phase8_ablation_controls"
        / "run_phase8_ablation_controls.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase8"
    _write_ready_phase2_outputs(phase2_dir)

    exit_code = module.main(
        [
            "--phase2-output-dir",
            str(phase2_dir),
            "--output-dir",
            str(output_dir),
            "--seed",
            "0",
            "--pca-dimensions",
            "8,16",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Generated variants: D2,D3,D4P8,D4P16" in stdout
    assert "D2 features: 81" in stdout
    assert "D3 features: 81" in stdout
    assert "D4P8 features: 25" in stdout
    assert "D4P16 features: 33" in stdout
    assert "Manifest:" in stdout
    assert "Summary:" in stdout
    assert (
        "Claim boundary: Phase 8 builds diagnostic ablation-control feature tables"
        in stdout
    )
    assert (output_dir / "experiment_variants.json").exists()
    assert (output_dir / "phase8_ablation_control_summary.json").exists()
