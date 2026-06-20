import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _explicit_row(
    block_id,
    *,
    area,
    slope_mean,
    slope_max,
    farmland,
    orchard=0.0,
    low_slope=0.0,
    low_slope_farmland=0.0,
    built_up=0.0,
    water=0.0,
    suitability=0.5,
):
    row = {
        "block_id": block_id,
        "pixel_count": 1,
        "pixel_weight_sum": 1.0,
        "row_min": 0,
        "row_max": 0,
        "col_min": 0,
        "col_max": 0,
        "suitability_proxy": suitability,
        "current_farmland_label": int(farmland),
        "farmland_or_orchard_label": int(max(farmland, orchard)),
        "low_slope_farmland_label": int(low_slope_farmland),
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
            "explicit_feature_07": orchard,
            "explicit_feature_09": built_up,
            "explicit_feature_10": water,
            "explicit_feature_13": low_slope,
            "explicit_feature_16": low_slope_farmland,
        }
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _summary_row(variant_id, tile_id, seed, reward, selected_block_ids):
    return {
        "row_type": "trained_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": tile_id,
        "eval_tile_rank": 1,
        "seed": seed,
        "phase25_seed_rank": int(seed) + 1,
        "train_timesteps": 4096,
        "eval_max_steps": 8,
        "max_blocks": 4,
        "train_n_blocks": 4,
        "eval_n_blocks": 4,
        "n_features": 81,
        "observation_shape": 100,
        "action_space_n": 4,
        "episode_steps": 2,
        "terminated": True,
        "truncated": False,
        "all_actions_valid": True,
        "invalid_action_count": 0,
        "total_contract_reward": reward,
        "selected_block_ids": selected_block_ids,
        "claim_boundary": "phase31 fixture",
    }


def _write_phase31_fixture_inputs(tmp_path: Path) -> dict[str, Path]:
    from paper11_geofm.padded_heldout_policy import SUMMARY_FIELDNAMES

    feature_rows = [
        _explicit_row(
            "b_good_a",
            area=5.0,
            slope_mean=5.0,
            slope_max=8.0,
            farmland=1.0,
            low_slope=1.0,
            low_slope_farmland=1.0,
            suitability=0.9,
        ),
        _explicit_row(
            "b_good_b",
            area=4.0,
            slope_mean=8.0,
            slope_max=10.0,
            farmland=1.0,
            low_slope=1.0,
            low_slope_farmland=1.0,
            suitability=0.8,
        ),
        _explicit_row(
            "b_bad_a",
            area=1.0,
            slope_mean=24.0,
            slope_max=34.0,
            farmland=0.0,
            built_up=1.0,
            water=0.5,
            suitability=0.2,
        ),
        _explicit_row(
            "b_bad_b",
            area=2.0,
            slope_mean=20.0,
            slope_max=30.0,
            farmland=0.0,
            built_up=0.5,
            suitability=0.3,
        ),
    ]
    feature_fields = list(feature_rows[0].keys())
    features_csv = _write_csv(tmp_path / "block_geofm_features.csv", feature_rows, feature_fields)

    summary_rows = [
        _summary_row("B1", "tile_good", 1, 0.25, "b_bad_a;b_bad_b"),
        _summary_row("N1ZR", "tile_good", 1, 1.25, "b_good_a;b_good_b"),
        _summary_row("B1", "tile_bad", 2, 1.10, "b_good_a;b_good_b"),
        _summary_row("N1ZR", "tile_bad", 2, 0.70, "b_bad_a;b_bad_b"),
        _summary_row("B1", "tile_mid", 0, 0.60, "b_good_a;b_bad_a"),
        _summary_row("N1ZR", "tile_mid", 0, 0.65, "b_good_b;b_bad_b"),
    ]
    summary_csv = _write_csv(tmp_path / "phase30_summary.csv", summary_rows, SUMMARY_FIELDNAMES)

    traces = {
        "trained_policy": {
            "N1ZR": {
                "tile_good": {
                    "1": [
                        {
                            "step": 1,
                            "action": 0,
                            "selected_block_id": "b_good_a",
                            "reward": 0.1,
                        },
                        {
                            "step": 2,
                            "action": 1,
                            "selected_block_id": "b_good_b",
                            "reward": 0.2,
                        },
                    ]
                },
                "tile_bad": {
                    "2": [
                        {
                            "step": 1,
                            "action": 2,
                            "selected_block_id": "b_bad_a",
                            "reward": -0.1,
                        },
                        {
                            "step": 2,
                            "action": 3,
                            "selected_block_id": "b_bad_b",
                            "reward": -0.2,
                        },
                    ]
                },
            }
        }
    }
    traces_json = tmp_path / "phase30_traces.json"
    traces_json.write_text(json.dumps(traces, indent=2, sort_keys=True), encoding="utf-8")

    tile_rows = [
        {
            "tile_id": "tile_good",
            "tile_row": 5,
            "tile_col": 4,
            "n_blocks": 2,
            "min_grid_row": 40,
            "max_grid_row": 47,
            "min_grid_col": 32,
            "max_grid_col": 39,
            "block_ids": "b_good_a;b_good_b",
        },
        {
            "tile_id": "tile_bad",
            "tile_row": 2,
            "tile_col": 3,
            "n_blocks": 2,
            "min_grid_row": 16,
            "max_grid_row": 23,
            "min_grid_col": 24,
            "max_grid_col": 31,
            "block_ids": "b_bad_a;b_bad_b",
        },
        {
            "tile_id": "tile_mid",
            "tile_row": 3,
            "tile_col": 3,
            "n_blocks": 2,
            "min_grid_row": 24,
            "max_grid_row": 31,
            "min_grid_col": 24,
            "max_grid_col": 31,
            "block_ids": "b_good_a;b_bad_a",
        },
    ]
    tile_index_csv = _write_csv(
        tmp_path / "phase13_tile_index.csv",
        tile_rows,
        [
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

    mapping_rows = [
        {"block_id": "b_good_a", "row": 41, "col": 33, "weight": 1.0},
        {"block_id": "b_good_b", "row": 42, "col": 34, "weight": 1.0},
        {"block_id": "b_bad_a", "row": 18, "col": 25, "weight": 1.0},
        {"block_id": "b_bad_b", "row": 19, "col": 26, "weight": 1.0},
    ]
    mapping_csv = _write_csv(
        tmp_path / "block_pixel_mapping.csv",
        mapping_rows,
        ["block_id", "row", "col", "weight"],
    )

    return {
        "summary_csv": summary_csv,
        "traces_json": traces_json,
        "phase2_features_csv": features_csv,
        "tile_index_csv": tile_index_csv,
        "block_mapping_csv": mapping_csv,
    }


def test_phase31_builds_ranked_case_tables(tmp_path):
    from paper11_geofm.phase31_case_diagnostics import (
        PHASE31_CASE_DIAGNOSTICS_CLAIM_BOUNDARY,
        build_phase31_case_diagnostics,
    )

    paths = _write_phase31_fixture_inputs(tmp_path)
    analysis = build_phase31_case_diagnostics(
        summary_csv=paths["summary_csv"],
        traces_json=paths["traces_json"],
        phase2_features_csv=paths["phase2_features_csv"],
        tile_index_csv=paths["tile_index_csv"],
        block_mapping_csv=paths["block_mapping_csv"],
        top_k=2,
    )

    assert analysis["phase"] == "phase31_case_diagnostics"
    assert analysis["phase31_case_diagnostic_status"] == "case_diagnostics_ready"
    assert analysis["claim_boundary"] == PHASE31_CASE_DIAGNOSTICS_CLAIM_BOUNDARY
    assert [row["case_id"] for row in analysis["ranked_case_rows"]] == [
        "tile_good|1|N1ZR|B1",
        "tile_bad|2|N1ZR|B1",
    ]

    first_case = analysis["ranked_case_rows"][0]
    assert first_case["case_role"] == "strong_positive"
    assert first_case["variant_minus_comparator_reward"] == 1.0
    assert first_case["selected_block_jaccard"] == 0.0
    assert first_case["trace_step_count"] == 2

    selected = {
        (row["case_id"], row["variant_id"]): row
        for row in analysis["selected_block_summary_rows"]
    }
    assert selected[("tile_good|1|N1ZR|B1", "N1ZR")]["selected_block_count"] == 2
    assert selected[("tile_good|1|N1ZR|B1", "N1ZR")]["mean_low_slope_farmland_label"] == 1.0
    assert selected[("tile_bad|2|N1ZR|B1", "N1ZR")]["mean_current_farmland_label"] == 0.0

    geometry = {
        row["case_id"]: row
        for row in analysis["tile_geometry_rows"]
    }
    assert geometry["tile_good|1|N1ZR|B1"]["tile_row"] == 5
    assert geometry["tile_good|1|N1ZR|B1"]["selected_mapping_min_row"] == 41
    assert geometry["tile_bad|2|N1ZR|B1"]["selected_mapping_max_col"] == 26


def test_phase31_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase31_case_diagnostics import (
        build_phase31_case_diagnostics,
        write_phase31_case_diagnostics_artifacts,
    )

    paths = _write_phase31_fixture_inputs(tmp_path)
    analysis = build_phase31_case_diagnostics(
        summary_csv=paths["summary_csv"],
        traces_json=paths["traces_json"],
        phase2_features_csv=paths["phase2_features_csv"],
        tile_index_csv=paths["tile_index_csv"],
        block_mapping_csv=paths["block_mapping_csv"],
        top_k=2,
    )
    artifact_paths = write_phase31_case_diagnostics_artifacts(
        analysis,
        tmp_path / "outputs",
    )

    assert artifact_paths["ranked_cases_csv"].name == "phase31_ranked_cases.csv"
    assert artifact_paths["selected_blocks_csv"].name == "phase31_selected_blocks.csv"
    assert artifact_paths["tile_geometry_csv"].name == "phase31_tile_geometry.csv"
    assert artifact_paths["diagnosis_json"].name == "phase31_case_diagnostics.json"
    assert artifact_paths["diagnosis_md"].name == "phase31_case_diagnostics.md"
    assert all(path.exists() for path in artifact_paths.values())

    ranked_rows = list(
        csv.DictReader(
            artifact_paths["ranked_cases_csv"].open("r", encoding="utf-8")
        )
    )
    assert ranked_rows[0]["case_id"] == "tile_good|1|N1ZR|B1"
    saved = json.loads(
        artifact_paths["diagnosis_json"].read_text(encoding="utf-8")
    )
    assert saved["phase31_case_diagnostic_status"] == "case_diagnostics_ready"
    markdown = artifact_paths["diagnosis_md"].read_text(encoding="utf-8")
    assert "Phase 31 Case Diagnostics" in markdown
    assert "case_diagnostics_ready" in markdown


def test_phase31_cli_writes_outputs(tmp_path):
    paths = _write_phase31_fixture_inputs(tmp_path)
    script = (
        ROOT
        / "experiments"
        / "phase31_case_diagnostics"
        / "run_phase31_case_diagnostics.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--summary-csv",
            str(paths["summary_csv"]),
            "--traces-json",
            str(paths["traces_json"]),
            "--phase2-features-csv",
            str(paths["phase2_features_csv"]),
            "--tile-index-csv",
            str(paths["tile_index_csv"]),
            "--block-mapping-csv",
            str(paths["block_mapping_csv"]),
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
        "Phase 31 case diagnostic status: case_diagnostics_ready"
        in result.stdout
    )
    assert (tmp_path / "cli_outputs" / "phase31_case_diagnostics.json").exists()
