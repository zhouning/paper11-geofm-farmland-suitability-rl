import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _feature_row(
    block_id,
    *,
    area,
    slope_mean,
    slope_max,
    farmland,
    low_slope_farmland,
    suitability,
):
    row = {
        "block_id": block_id,
        "suitability_proxy": suitability,
        "current_farmland_label": farmland,
        "farmland_or_orchard_label": farmland,
        "low_slope_farmland_label": low_slope_farmland,
        "slope_mean": slope_mean,
        "slope_max": slope_max,
        "area_m2": area * 10000.0,
    }
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = 0.0
    row.update(
        {
            "explicit_feature_00": area,
            "explicit_feature_01": slope_mean,
            "explicit_feature_02": slope_max,
            "explicit_feature_04": farmland,
            "explicit_feature_07": 0.0,
            "explicit_feature_09": 0.0,
            "explicit_feature_10": 0.0,
            "explicit_feature_13": 1.0 if slope_mean <= 15.0 else 0.0,
            "explicit_feature_16": low_slope_farmland,
        }
    )
    return row


def _write_phase32_fixture_inputs(tmp_path: Path) -> dict[str, Path]:
    ranked_cases = [
        {
            "case_id": "tile_overlap|2|N1ZR|B1",
            "case_rank": 1,
            "case_role": "failure_case",
            "eval_tile_id": "tile_overlap",
            "seed": 2,
            "variant_id": "N1ZR",
            "comparator_variant_id": "B1",
            "variant_reward": -0.1,
            "comparator_reward": 0.3,
            "variant_minus_comparator_reward": -0.4,
            "abs_variant_minus_comparator_reward": 0.4,
            "selected_block_jaccard": 1.0,
            "shared_selected_block_count": 3,
            "variant_selected_block_count": 3,
            "comparator_selected_block_count": 3,
            "trace_step_count": 3,
            "claim_boundary": "phase32 fixture",
        },
        {
            "case_id": "tile_distinct|1|N1ZR|B1",
            "case_rank": 2,
            "case_role": "strong_positive",
            "eval_tile_id": "tile_distinct",
            "seed": 1,
            "variant_id": "N1ZR",
            "comparator_variant_id": "B1",
            "variant_reward": 0.5,
            "comparator_reward": 0.1,
            "variant_minus_comparator_reward": 0.4,
            "abs_variant_minus_comparator_reward": 0.4,
            "selected_block_jaccard": 0.2,
            "shared_selected_block_count": 1,
            "variant_selected_block_count": 3,
            "comparator_selected_block_count": 3,
            "trace_step_count": 3,
            "claim_boundary": "phase32 fixture",
        },
    ]
    ranked_cases_csv = _write_csv(
        tmp_path / "phase31_ranked_cases.csv",
        ranked_cases,
        list(ranked_cases[0].keys()),
    )

    focal_traces = {
        "trained_policy": {
            "N1ZR": {
                "tile_overlap": {
                    "2": [
                        {"step": 1, "selected_block_id": "b_mid", "reward": 0.0},
                        {"step": 2, "selected_block_id": "b_bad", "reward": -0.3},
                        {"step": 3, "selected_block_id": "b_good", "reward": 0.2},
                    ]
                },
                "tile_distinct": {
                    "1": [
                        {"step": 1, "selected_block_id": "b_good", "reward": 0.2},
                        {"step": 2, "selected_block_id": "b_mid", "reward": 0.0},
                        {"step": 3, "selected_block_id": "b_extra", "reward": 0.1},
                    ]
                },
            }
        }
    }
    comparator_traces = {
        "trained_policy": {
            "B1": {
                "tile_overlap": {
                    "2": [
                        {"step": 1, "selected_block_id": "b_good", "reward": 0.2},
                        {"step": 2, "selected_block_id": "b_mid", "reward": 0.0},
                        {"step": 3, "selected_block_id": "b_bad", "reward": -0.3},
                    ]
                },
                "tile_distinct": {
                    "1": [
                        {"step": 1, "selected_block_id": "b_bad", "reward": -0.3},
                        {"step": 2, "selected_block_id": "b_mid", "reward": 0.0},
                        {"step": 3, "selected_block_id": "b_pool", "reward": -0.2},
                    ]
                },
            }
        }
    }
    focal_traces_json = tmp_path / "phase30_traces.json"
    comparator_traces_json = tmp_path / "phase28_traces.json"
    focal_traces_json.write_text(
        json.dumps(focal_traces, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    comparator_traces_json.write_text(
        json.dumps(comparator_traces, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    feature_rows = [
        _feature_row(
            "b_good",
            area=5.0,
            slope_mean=5.0,
            slope_max=8.0,
            farmland=1.0,
            low_slope_farmland=1.0,
            suitability=0.9,
        ),
        _feature_row(
            "b_mid",
            area=3.0,
            slope_mean=12.0,
            slope_max=15.0,
            farmland=1.0,
            low_slope_farmland=0.0,
            suitability=0.5,
        ),
        _feature_row(
            "b_bad",
            area=1.0,
            slope_mean=25.0,
            slope_max=35.0,
            farmland=0.0,
            low_slope_farmland=0.0,
            suitability=0.1,
        ),
        _feature_row(
            "b_extra",
            area=4.0,
            slope_mean=7.0,
            slope_max=10.0,
            farmland=1.0,
            low_slope_farmland=1.0,
            suitability=0.8,
        ),
        _feature_row(
            "b_pool",
            area=1.0,
            slope_mean=28.0,
            slope_max=36.0,
            farmland=0.0,
            low_slope_farmland=0.0,
            suitability=0.2,
        ),
    ]
    features_csv = _write_csv(
        tmp_path / "block_geofm_features.csv",
        feature_rows,
        list(feature_rows[0].keys()),
    )

    tile_rows = [
        {
            "tile_id": "tile_overlap",
            "tile_row": 2,
            "tile_col": 3,
            "n_blocks": 4,
            "min_grid_row": 16,
            "max_grid_row": 23,
            "min_grid_col": 24,
            "max_grid_col": 31,
            "block_ids": "b_good;b_mid;b_bad;b_pool",
        },
        {
            "tile_id": "tile_distinct",
            "tile_row": 5,
            "tile_col": 4,
            "n_blocks": 4,
            "min_grid_row": 40,
            "max_grid_row": 47,
            "min_grid_col": 32,
            "max_grid_col": 39,
            "block_ids": "b_good;b_mid;b_bad;b_extra",
        },
    ]
    tile_index_csv = _write_csv(
        tmp_path / "phase13_tile_index.csv",
        tile_rows,
        list(tile_rows[0].keys()),
    )

    return {
        "ranked_cases_csv": ranked_cases_csv,
        "focal_traces_json": focal_traces_json,
        "comparator_traces_json": comparator_traces_json,
        "phase2_features_csv": features_csv,
        "tile_index_csv": tile_index_csv,
    }


def test_phase32_builds_step_alignment_and_tile_pool_diagnostics(tmp_path):
    from paper11_geofm.phase32_action_order_diagnostics import (
        PHASE32_ACTION_ORDER_CLAIM_BOUNDARY,
        build_phase32_action_order_diagnostics,
    )

    paths = _write_phase32_fixture_inputs(tmp_path)
    analysis = build_phase32_action_order_diagnostics(
        ranked_cases_csv=paths["ranked_cases_csv"],
        focal_traces_json=paths["focal_traces_json"],
        comparator_traces_json=paths["comparator_traces_json"],
        phase2_features_csv=paths["phase2_features_csv"],
        tile_index_csv=paths["tile_index_csv"],
        top_k=2,
    )

    assert analysis["phase"] == "phase32_action_order_diagnostics"
    assert analysis["phase32_action_order_status"] == "action_order_diagnostics_ready"
    assert analysis["claim_boundary"] == PHASE32_ACTION_ORDER_CLAIM_BOUNDARY

    by_case = {
        row["case_id"]: row
        for row in analysis["case_summary_rows"]
    }
    overlap = by_case["tile_overlap|2|N1ZR|B1"]
    assert overlap["shared_block_count"] == 3
    assert overlap["mean_abs_shared_step_displacement"] == 1.3333333333
    assert overlap["focal_cumulative_reward"] == -0.1
    assert overlap["comparator_cumulative_reward"] == -0.1
    assert overlap["first_step_reward_gap"] == -0.2
    assert overlap["diagnostic_pattern"] == "same_blocks_reordered"

    step_rows = [
        row
        for row in analysis["step_alignment_rows"]
        if row["case_id"] == "tile_overlap|2|N1ZR|B1"
    ]
    assert [row["focal_block_id"] for row in step_rows] == ["b_mid", "b_bad", "b_good"]
    assert [row["comparator_block_id"] for row in step_rows] == [
        "b_good",
        "b_mid",
        "b_bad",
    ]
    assert step_rows[0]["focal_cumulative_reward"] == 0.0
    assert step_rows[0]["comparator_cumulative_reward"] == 0.2

    tile_pool = {
        row["case_id"]: row
        for row in analysis["tile_pool_composition_rows"]
    }
    assert tile_pool["tile_overlap|2|N1ZR|B1"]["tile_block_count"] == 4
    assert tile_pool["tile_overlap|2|N1ZR|B1"]["tile_low_slope_farmland_mean"] == 0.25
    assert tile_pool["tile_distinct|1|N1ZR|B1"]["focal_low_slope_farmland_mean"] == 0.6666666667


def test_phase32_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase32_action_order_diagnostics import (
        build_phase32_action_order_diagnostics,
        write_phase32_action_order_diagnostics_artifacts,
    )

    paths = _write_phase32_fixture_inputs(tmp_path)
    analysis = build_phase32_action_order_diagnostics(
        ranked_cases_csv=paths["ranked_cases_csv"],
        focal_traces_json=paths["focal_traces_json"],
        comparator_traces_json=paths["comparator_traces_json"],
        phase2_features_csv=paths["phase2_features_csv"],
        tile_index_csv=paths["tile_index_csv"],
        top_k=2,
    )
    artifact_paths = write_phase32_action_order_diagnostics_artifacts(
        analysis,
        tmp_path / "outputs",
    )

    assert artifact_paths["step_alignment_csv"].name == "phase32_step_alignment.csv"
    assert artifact_paths["case_summary_csv"].name == "phase32_case_summary.csv"
    assert artifact_paths["tile_pool_csv"].name == "phase32_tile_pool_composition.csv"
    assert artifact_paths["diagnosis_json"].name == "phase32_action_order_diagnostics.json"
    assert artifact_paths["diagnosis_md"].name == "phase32_action_order_diagnostics.md"
    assert all(path.exists() for path in artifact_paths.values())

    saved = json.loads(
        artifact_paths["diagnosis_json"].read_text(encoding="utf-8")
    )
    assert saved["phase32_action_order_status"] == "action_order_diagnostics_ready"
    markdown = artifact_paths["diagnosis_md"].read_text(encoding="utf-8")
    assert "Phase 32 Action-Order Diagnostics" in markdown
    assert "same_blocks_reordered" in markdown


def test_phase32_cli_writes_outputs(tmp_path):
    paths = _write_phase32_fixture_inputs(tmp_path)
    script = (
        ROOT
        / "experiments"
        / "phase32_action_order_diagnostics"
        / "run_phase32_action_order_diagnostics.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--ranked-cases-csv",
            str(paths["ranked_cases_csv"]),
            "--focal-traces-json",
            str(paths["focal_traces_json"]),
            "--comparator-traces-json",
            str(paths["comparator_traces_json"]),
            "--phase2-features-csv",
            str(paths["phase2_features_csv"]),
            "--tile-index-csv",
            str(paths["tile_index_csv"]),
            "--output-dir",
            str(tmp_path / "cli_outputs"),
            "--top-k",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "Phase 32 action-order status: action_order_diagnostics_ready"
        in result.stdout
    )
    assert (
        tmp_path
        / "cli_outputs"
        / "phase32_action_order_diagnostics.json"
    ).exists()
