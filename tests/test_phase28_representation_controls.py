import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id, slope_mean, farmland, suitability=0.75):
    row = {"block_id": block_id, "suitability_proxy": suitability}
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(dim) / 100.0
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = 0.0
    row.update(
        {
            "explicit_feature_00": 2.0,
            "explicit_feature_01": float(slope_mean),
            "explicit_feature_02": float(slope_mean) + 5.0,
            "explicit_feature_04": float(farmland),
            "explicit_feature_07": 0.0,
            "explicit_feature_09": 0.0,
            "explicit_feature_10": 0.0,
            "explicit_feature_13": 1.0 if slope_mean <= 15.0 else 0.0,
            "explicit_feature_16": float(farmland),
        }
    )
    return row


def _write_ready_phase2_outputs(output_dir: Path):
    from paper11_geofm.artifacts import write_phase2_artifacts

    return write_phase2_artifacts(
        [
            _complete_phase2_feature_row("b1", slope_mean=8.0, farmland=1.0),
            _complete_phase2_feature_row("b2", slope_mean=30.0, farmland=0.0),
            _complete_phase2_feature_row("b3", slope_mean=12.0, farmland=1.0),
            _complete_phase2_feature_row("b4", slope_mean=25.0, farmland=0.0),
            _complete_phase2_feature_row("b5", slope_mean=6.0, farmland=1.0),
            _complete_phase2_feature_row("b6", slope_mean=22.0, farmland=0.0),
        ],
        output_dir,
        {
            "metadata_source": "test",
            "base_year_requested": 2020,
            "base_year_used": 2020,
            "years": [2020],
            "grid_shape": [2, 3],
            "embedding_dim": 64,
            "mapping_mode": "test",
        },
    )


def _write_phase8_outputs(phase2_dir: Path, output_dir: Path):
    from paper11_geofm.ablation_controls import (
        build_phase8_ablation_controls,
        write_phase8_ablation_artifacts,
    )

    protocol = build_phase8_ablation_controls(
        phase2_dir,
        seed=0,
        pca_dimensions=(8, 16),
    )
    return write_phase8_ablation_artifacts(protocol, output_dir)


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
                "block_ids": "b6",
            }
        )
        writer.writerow(
            {
                "tile_id": "tile_r000_c001",
                "tile_row": 0,
                "tile_col": 1,
                "n_blocks": 3,
                "block_ids": "b1;b3;b5",
            }
        )
        writer.writerow(
            {
                "tile_id": "tile_r000_c002",
                "tile_row": 0,
                "tile_col": 2,
                "n_blocks": 2,
                "block_ids": "b2;b4",
            }
        )
    return path


def _phase28_summary_rows(status_case="supported"):
    rewards_by_case = {
        "supported": {
            "B0": [1.0, 1.0, 1.0, 1.0],
            "B1": [1.6, 1.4, 1.5, 1.3],
            "D2": [1.0, 1.0, 1.0, 1.0],
            "D3": [0.9, 1.0, 0.8, 1.0],
            "D4P8": [1.2, 1.1, 1.0, 1.0],
            "D4P16": [1.2, 1.1, 1.0, 1.0],
        },
        "control_limited": {
            "B0": [1.6, 1.5, 1.4, 1.4],
            "B1": [1.2, 1.3, 1.2, 1.3],
            "D2": [1.0, 1.0, 1.1, 1.1],
            "D3": [1.1, 1.1, 1.2, 1.2],
            "D4P8": [1.0, 1.0, 1.1, 1.1],
            "D4P16": [1.0, 1.0, 1.1, 1.1],
        },
        "not_distinguishable": {
            "B0": [1.0, 1.0, 1.0, 1.0],
            "B1": [1.0, 1.0, 1.0, 1.0],
            "D2": [1.1, 1.1, 1.1, 1.1],
            "D3": [1.2, 1.2, 1.2, 1.2],
            "D4P8": [0.8, 0.8, 0.8, 0.8],
            "D4P16": [0.8, 0.8, 0.8, 0.8],
        },
        "compression_match": {
            "B0": [1.0, 1.0, 1.0, 1.0],
            "B1": [1.2, 1.2, 1.2, 1.2],
            "D2": [1.0, 1.0, 1.0, 1.0],
            "D3": [1.0, 1.0, 1.0, 1.0],
            "D4P8": [1.2, 1.2, 1.2, 1.2],
            "D4P16": [1.1, 1.1, 1.1, 1.1],
        },
    }
    rewards = rewards_by_case[status_case]
    rows = []
    pairs = [
        ("tile_eval_a", 0),
        ("tile_eval_a", 1),
        ("tile_eval_b", 0),
        ("tile_eval_b", 1),
    ]
    for pair_index, (tile_id, seed) in enumerate(pairs):
        for variant_id, values in rewards.items():
            rows.append(
                {
                    "row_type": "trained_policy",
                    "variant_id": variant_id,
                    "train_tile_id": "tile_train",
                    "eval_tile_id": tile_id,
                    "eval_tile_rank": 1 if tile_id == "tile_eval_a" else 2,
                    "seed": seed,
                    "phase25_seed_rank": seed + 1,
                    "train_timesteps": 128,
                    "eval_max_steps": 4,
                    "max_blocks": 3,
                    "train_n_blocks": 3,
                    "eval_n_blocks": 2,
                    "n_features": 17 if variant_id == "B0" else 81,
                    "observation_shape": 260,
                    "action_space_n": 3,
                    "episode_steps": 2,
                    "terminated": True,
                    "truncated": False,
                    "all_actions_valid": True,
                    "invalid_action_count": 0,
                    "total_contract_reward": values[pair_index],
                    "selected_block_ids": "b1;b2",
                    "claim_boundary": "phase28 fixture",
                }
            )
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    from paper11_geofm.padded_heldout_policy import SUMMARY_FIELDNAMES

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_phase28_contract_routes_b_and_d_variant_sources(tmp_path):
    from paper11_geofm.phase28_representation_controls import (
        PHASE28_CLAIM_BOUNDARY,
        build_phase28_representation_control_contract,
    )

    phase2_dir = tmp_path / "phase2"
    phase8_dir = tmp_path / "phase8"
    _write_ready_phase2_outputs(phase2_dir)
    _write_phase8_outputs(phase2_dir, phase8_dir)
    tile_index = _write_tile_index(tmp_path / "phase13_tile_index.csv")

    contract = build_phase28_representation_control_contract(
        phase2_output_dir=phase2_dir,
        phase8_output_dir=phase8_dir,
        tile_index_csv=tile_index,
        variants=("B0", "B1", "D2", "D3", "D4P8", "D4P16"),
        total_timesteps=8,
        eval_max_steps=2,
        seeds="0,1",
        max_eval_tiles=2,
    )

    assert contract["phase"] == "phase28_representation_control_evaluation"
    assert contract["variants"] == ["B0", "B1", "D2", "D3", "D4P8", "D4P16"]
    assert contract["variant_source_dirs"]["B0"] == str(phase2_dir)
    assert contract["variant_source_dirs"]["B1"] == str(phase2_dir)
    assert contract["variant_source_dirs"]["D2"] == str(phase8_dir)
    assert contract["variant_source_dirs"]["D4P16"] == str(phase8_dir)
    assert contract["train_tile_id"] == "tile_r000_c001"
    assert contract["eval_tile_ids"] == ["tile_r000_c002", "tile_r000_c000"]
    assert contract["max_blocks"] == 3
    assert contract["claim_boundary"] == PHASE28_CLAIM_BOUNDARY


def test_phase28_contract_rejects_unsupported_and_missing_b1(tmp_path):
    from paper11_geofm.phase28_representation_controls import (
        build_phase28_representation_control_contract,
    )

    phase2_dir = tmp_path / "phase2"
    phase8_dir = tmp_path / "phase8"
    _write_ready_phase2_outputs(phase2_dir)
    _write_phase8_outputs(phase2_dir, phase8_dir)
    tile_index = _write_tile_index(tmp_path / "phase13_tile_index.csv")

    with pytest.raises(ValueError, match="unsupported Phase 28 variants"):
        build_phase28_representation_control_contract(
            phase2_dir,
            phase8_dir,
            tile_index,
            variants=("B3",),
        )

    with pytest.raises(ValueError, match="requires B1"):
        build_phase28_representation_control_contract(
            phase2_dir,
            phase8_dir,
            tile_index,
            variants=("B0", "D2"),
        )


def test_phase28_analysis_computes_b1_control_deltas_and_supported_status(tmp_path):
    from paper11_geofm.phase28_representation_controls import (
        build_phase28_representation_control_analysis,
    )

    summary_csv = _write_summary_csv(
        tmp_path / "phase28_representation_control_summary.csv",
        _phase28_summary_rows("supported"),
    )

    analysis = build_phase28_representation_control_analysis(summary_csv)

    assert analysis["phase"] == "phase28_representation_control_analysis"
    assert analysis["phase28_diagnostic_status"] == "representation_signal_supported"
    assert analysis["learned_policy"]["mean_reward_by_variant"]["B1"] == 1.45
    assert analysis["learned_policy"]["comparator_deltas"]["B1_minus_D2"]["mean_reward_delta"] == 0.45
    assert analysis["learned_policy"]["comparator_deltas"]["B1_minus_D3"]["positive_tile_seed_count"] == 4
    assert len(analysis["tile_seed_delta_rows"]) == 20


@pytest.mark.parametrize(
    ("status_case", "expected_status"),
    [
        ("control_limited", "representation_signal_control_limited"),
        ("not_distinguishable", "representation_signal_not_distinguishable"),
        ("compression_match", "compression_matches_raw"),
    ],
)
def test_phase28_diagnostic_status_rules(tmp_path, status_case, expected_status):
    from paper11_geofm.phase28_representation_controls import (
        build_phase28_representation_control_analysis,
    )

    summary_csv = _write_summary_csv(
        tmp_path / f"{status_case}.csv",
        _phase28_summary_rows(status_case),
    )

    analysis = build_phase28_representation_control_analysis(summary_csv)

    assert analysis["phase28_diagnostic_status"] == expected_status


def test_phase28_reports_insufficient_for_missing_comparator_rows(tmp_path):
    from paper11_geofm.phase28_representation_controls import (
        build_phase28_representation_control_analysis,
    )

    rows = [
        row
        for row in _phase28_summary_rows("supported")
        if not (
            row["row_type"] == "trained_policy"
            and row["variant_id"] == "D3"
            and row["eval_tile_id"] == "tile_eval_b"
            and row["seed"] == 1
        )
    ]
    summary_csv = _write_summary_csv(tmp_path / "missing.csv", rows)

    analysis = build_phase28_representation_control_analysis(summary_csv)

    assert analysis["phase28_diagnostic_status"] == "insufficient"
    assert analysis["coverage_issues"]["missing_variant_rows"] == [
        {"eval_tile_id": "tile_eval_b", "seed": 1, "variant_id": "D3"}
    ]


def test_phase28_writer_outputs_summary_trace_comparison_delta_and_markdown(tmp_path):
    from paper11_geofm.phase28_representation_controls import (
        build_phase28_representation_control_analysis,
        write_phase28_representation_control_artifacts,
    )

    summary_rows = _phase28_summary_rows("supported")
    analysis = build_phase28_representation_control_analysis(summary_rows)
    protocol = {
        **analysis,
        "summaries": summary_rows,
        "traces": {"trained_policy": {"B1": {"tile_eval_a": {"0": []}}}},
    }

    paths = write_phase28_representation_control_artifacts(
        protocol,
        tmp_path / "outputs",
    )

    assert paths["summary_csv"].name == "phase28_representation_control_summary.csv"
    assert paths["traces_json"].name == "phase28_representation_control_traces.json"
    assert paths["comparison_json"].name == "phase28_representation_control_comparison.json"
    assert paths["tile_seed_delta_csv"].name == "phase28_tile_seed_delta_table.csv"
    assert paths["control_readiness_md"].name == "phase28_control_readiness.md"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase28_diagnostic_status"] == "representation_signal_supported"
    markdown = paths["control_readiness_md"].read_text(encoding="utf-8")
    assert "representation_signal_supported" in markdown
    assert "GeoFM improves planning decisions" not in markdown


def test_phase28_run_uses_fake_training_model_for_all_variants(tmp_path, monkeypatch):
    from paper11_geofm import phase28_representation_controls as phase28

    class FakeModel:
        def predict(self, obs, deterministic=True, action_masks=None):
            valid_actions = [
                index
                for index, valid in enumerate(action_masks.tolist())
                if bool(valid)
            ]
            return valid_actions[0], None

    phase2_dir = tmp_path / "phase2"
    phase8_dir = tmp_path / "phase8"
    _write_ready_phase2_outputs(phase2_dir)
    _write_phase8_outputs(phase2_dir, phase8_dir)
    tile_index = _write_tile_index(tmp_path / "phase13_tile_index.csv")
    monkeypatch.setattr(
        phase28,
        "_train_maskable_ppo_model",
        lambda train_env, seed, total_timesteps: FakeModel(),
    )

    protocol = phase28.run_phase28_representation_control_evaluation(
        phase2_output_dir=phase2_dir,
        phase8_output_dir=phase8_dir,
        tile_index_csv=tile_index,
        variants=("B1", "D2"),
        total_timesteps=8,
        eval_max_steps=2,
        seeds=(0,),
        max_eval_tiles=1,
    )

    assert protocol["training_completed"] is True
    assert protocol["summary_count"] == 6
    assert {row["variant_id"] for row in protocol["summaries"]} == {"B1", "D2"}
    assert protocol["phase28_diagnostic_status"] in {
        "representation_signal_control_limited",
        "representation_signal_not_distinguishable",
        "compression_matches_raw",
        "insufficient",
    }


def test_phase28_cli_analyze_only_writes_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase28_representation_controls"
        / "run_phase28_representation_controls.py"
    )
    spec = importlib.util.spec_from_file_location("phase28_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    summary_csv = _write_summary_csv(
        tmp_path / "summary.csv",
        _phase28_summary_rows("supported"),
    )

    exit_code = module.main(
        [
            "--mode",
            "analyze-only",
            "--existing-summary-csv",
            str(summary_csv),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Phase 28 diagnostic status: representation_signal_supported" in stdout
    assert "phase28_representation_control_comparison.json" in stdout


def test_phase28_cli_run_and_analyze_requires_explicit_training_settings(
    tmp_path,
    capsys,
):
    runner_path = (
        ROOT
        / "experiments"
        / "phase28_representation_controls"
        / "run_phase28_representation_controls.py"
    )
    spec = importlib.util.spec_from_file_location("phase28_runner_validation", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--mode",
            "run-and-analyze",
            "--phase2-output-dir",
            str(tmp_path / "phase2"),
            "--phase8-output-dir",
            str(tmp_path / "phase8"),
            "--tile-index-csv",
            str(tmp_path / "phase13_tile_index.csv"),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert "run-and-analyze requires" in stderr
    assert "--total-timesteps" in stderr
    assert "--eval-max-steps" in stderr
    assert "--seeds" in stderr
