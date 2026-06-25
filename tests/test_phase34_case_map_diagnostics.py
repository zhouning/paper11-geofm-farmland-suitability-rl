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
    row_min,
    row_max,
    col_min,
    col_max,
    area,
    slope_mean,
    slope_max,
    farmland,
    low_slope_farmland,
    suitability,
    built_up=0.0,
    water=0.0,
):
    row = {
        "block_id": block_id,
        "pixel_count": 1,
        "pixel_weight_sum": 1.0,
        "row_min": row_min,
        "row_max": row_max,
        "col_min": col_min,
        "col_max": col_max,
        "suitability_proxy": suitability,
        "area_m2": area * 10000.0,
        "current_farmland_label": farmland,
        "farmland_or_orchard_label": farmland,
        "low_slope_farmland_label": low_slope_farmland,
        "slope_mean": slope_mean,
        "slope_max": slope_max,
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
            "explicit_feature_09": built_up,
            "explicit_feature_10": water,
            "explicit_feature_13": 1.0 if slope_mean <= 15.0 else 0.0,
            "explicit_feature_16": low_slope_farmland,
        }
    )
    return row


def _summary_row(variant_id, tile_id, seed, train_timesteps, reward, selected):
    from paper11_geofm.padded_heldout_policy import PHASE25_CLAIM_BOUNDARY

    return {
        "row_type": "trained_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": tile_id,
        "eval_tile_rank": 1,
        "seed": seed,
        "phase25_seed_rank": int(seed) + 1,
        "train_timesteps": train_timesteps,
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
        "selected_block_ids": selected,
        "claim_boundary": PHASE25_CLAIM_BOUNDARY,
    }


def _stability_row(variant_id, comparator_id, tile_id, seed, lower, higher, klass):
    return {
        "variant_id": variant_id,
        "comparator_variant_id": comparator_id,
        "eval_tile_id": tile_id,
        "seed": seed,
        "lower_budget_label": "4096_steps",
        "higher_budget_label": "5120_steps",
        "lower_train_timesteps": 4096,
        "higher_train_timesteps": 5120,
        "lower_delta": lower,
        "higher_delta": higher,
        "delta_change": higher - lower,
        "lower_positive": lower > 0.0,
        "higher_positive": higher > 0.0,
        "stability_class": klass,
    }


def _write_case_dir(
    root: Path,
    name: str,
    *,
    tile_id: str,
    seed: int,
    variant_selected: str,
    comparator_selected: str,
    variant_reward: float,
    comparator_reward: float,
    lower_delta: float,
    higher_delta: float,
    stability_class: str,
) -> Path:
    from paper11_geofm.padded_heldout_policy import SUMMARY_FIELDNAMES
    from paper11_geofm.phase33_budget_robustness import TILE_SEED_STABILITY_FIELDNAMES

    case_dir = root / name
    high_dir = case_dir / "phase30_high_budget"
    _write_csv(
        high_dir / "phase30_normalized_b1_summary.csv",
        [
            _summary_row(
                "N1ZR",
                tile_id,
                seed,
                5120,
                variant_reward,
                variant_selected,
            ),
            _summary_row(
                "D4P8",
                tile_id,
                seed,
                4096,
                comparator_reward,
                comparator_selected,
            ),
        ],
        SUMMARY_FIELDNAMES,
    )
    traces = {
        "trained_policy": {
            "N1ZR": {
                tile_id: {
                    str(seed): [
                        {
                            "step": index,
                            "action": index - 1,
                            "selected_block_id": block_id,
                            "reward": 0.1 * index,
                        }
                        for index, block_id in enumerate(
                            variant_selected.split(";"),
                            start=1,
                        )
                    ]
                }
            }
        }
    }
    (high_dir / "phase30_normalized_b1_traces.json").write_text(
        json.dumps(traces, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(
        case_dir / "phase33_tile_seed_stability.csv",
        [
            _stability_row(
                "N1ZR",
                "D4P8",
                tile_id,
                seed,
                lower_delta,
                higher_delta,
                stability_class,
            )
        ],
        TILE_SEED_STABILITY_FIELDNAMES,
    )
    return case_dir


def _write_phase34_fixture_inputs(tmp_path: Path) -> dict[str, object]:
    feature_rows = [
        _feature_row(
            "good_a",
            row_min=10,
            row_max=10,
            col_min=10,
            col_max=10,
            area=5.0,
            slope_mean=5.0,
            slope_max=8.0,
            farmland=1.0,
            low_slope_farmland=1.0,
            suitability=0.9,
        ),
        _feature_row(
            "good_b",
            row_min=11,
            row_max=11,
            col_min=11,
            col_max=11,
            area=4.0,
            slope_mean=7.0,
            slope_max=9.0,
            farmland=1.0,
            low_slope_farmland=1.0,
            suitability=0.8,
        ),
        _feature_row(
            "bad_a",
            row_min=20,
            row_max=20,
            col_min=20,
            col_max=20,
            area=1.0,
            slope_mean=25.0,
            slope_max=35.0,
            farmland=0.0,
            low_slope_farmland=0.0,
            suitability=0.1,
            built_up=1.0,
        ),
        _feature_row(
            "bad_b",
            row_min=21,
            row_max=21,
            col_min=21,
            col_max=21,
            area=1.0,
            slope_mean=24.0,
            slope_max=34.0,
            farmland=0.0,
            low_slope_farmland=0.0,
            suitability=0.2,
            water=1.0,
        ),
    ]
    features_csv = _write_csv(
        tmp_path / "block_geofm_features.csv",
        feature_rows,
        list(feature_rows[0].keys()),
    )
    tile_rows = [
        {
            "tile_id": "tile_positive",
            "tile_row": 1,
            "tile_col": 2,
            "n_blocks": 4,
            "min_grid_row": 8,
            "max_grid_row": 23,
            "min_grid_col": 8,
            "max_grid_col": 23,
            "block_ids": "good_a;good_b;bad_a;bad_b",
        },
        {
            "tile_id": "tile_failure",
            "tile_row": 5,
            "tile_col": 3,
            "n_blocks": 4,
            "min_grid_row": 8,
            "max_grid_row": 23,
            "min_grid_col": 8,
            "max_grid_col": 23,
            "block_ids": "good_a;good_b;bad_a;bad_b",
        },
    ]
    tile_index_csv = _write_csv(
        tmp_path / "phase13_tile_index.csv",
        tile_rows,
        list(tile_rows[0].keys()),
    )
    case_dirs = [
        _write_case_dir(
            tmp_path,
            "positive_case",
            tile_id="tile_positive",
            seed=0,
            variant_selected="good_a;good_b",
            comparator_selected="bad_a;bad_b",
            variant_reward=1.0,
            comparator_reward=0.2,
            lower_delta=-0.4,
            higher_delta=0.8,
            stability_class="flip_to_positive",
        ),
        _write_case_dir(
            tmp_path,
            "failure_case",
            tile_id="tile_failure",
            seed=1,
            variant_selected="bad_a;bad_b",
            comparator_selected="good_a;good_b",
            variant_reward=0.1,
            comparator_reward=0.9,
            lower_delta=-0.2,
            higher_delta=-0.8,
            stability_class="stable_negative",
        ),
    ]
    return {
        "case_dirs": case_dirs,
        "features_csv": features_csv,
        "tile_index_csv": tile_index_csv,
    }


def test_phase34_builds_case_map_rows_from_phase33_outputs(tmp_path):
    from paper11_geofm.phase34_case_map_diagnostics import (
        PHASE34_CASE_MAP_CLAIM_BOUNDARY,
        build_phase34_case_map_diagnostics,
    )

    paths = _write_phase34_fixture_inputs(tmp_path)
    analysis = build_phase34_case_map_diagnostics(
        phase33_output_dirs=paths["case_dirs"],
        phase2_features_csv=paths["features_csv"],
        tile_index_csv=paths["tile_index_csv"],
        variants=["N1ZR"],
        comparators=["D4P8"],
    )

    assert analysis["phase"] == "phase34_case_map_diagnostics"
    assert analysis["phase34_case_map_status"] == "case_map_diagnostics_ready"
    assert analysis["claim_boundary"] == PHASE34_CASE_MAP_CLAIM_BOUNDARY
    assert analysis["row_counts"]["case_rows"] == 2
    assert analysis["row_counts"]["case_map_block_rows"] == 8

    cases = {row["case_id"]: row for row in analysis["case_rows"]}
    positive = cases["tile_positive|0|N1ZR|D4P8"]
    assert positive["case_role"] == "phase33_positive_case"
    assert positive["stability_class"] == "flip_to_positive"
    assert positive["selected_block_jaccard"] == 0.0
    assert positive["variant_mean_base_planning_reward"] > positive[
        "comparator_mean_base_planning_reward"
    ]
    assert positive["spatial_pattern"] == "variant_selects_higher_base_reward_blocks"

    failure = cases["tile_failure|1|N1ZR|D4P8"]
    assert failure["case_role"] == "phase33_failure_case"
    assert failure["spatial_pattern"] == "variant_selects_lower_base_reward_blocks"

    positive_blocks = [
        row
        for row in analysis["case_map_block_rows"]
        if row["case_id"] == "tile_positive|0|N1ZR|D4P8"
    ]
    roles = {row["selection_role"] for row in positive_blocks}
    assert roles == {"variant_only", "comparator_only"}
    variant_steps = {
        row["block_id"]: row["variant_step"]
        for row in positive_blocks
        if row["selection_role"] == "variant_only"
    }
    assert variant_steps == {"good_a": 1, "good_b": 2}


def test_phase34_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase34_case_map_diagnostics import (
        build_phase34_case_map_diagnostics,
        write_phase34_case_map_diagnostics_artifacts,
    )

    paths = _write_phase34_fixture_inputs(tmp_path)
    analysis = build_phase34_case_map_diagnostics(
        phase33_output_dirs=paths["case_dirs"],
        phase2_features_csv=paths["features_csv"],
        tile_index_csv=paths["tile_index_csv"],
        variants=["N1ZR"],
        comparators=["D4P8"],
    )
    artifact_paths = write_phase34_case_map_diagnostics_artifacts(
        analysis,
        tmp_path / "outputs",
    )

    assert artifact_paths["case_summary_csv"].name == "phase34_case_map_cases.csv"
    assert artifact_paths["case_map_blocks_csv"].name == "phase34_case_map_blocks.csv"
    assert artifact_paths["diagnosis_json"].name == "phase34_case_map_diagnostics.json"
    assert artifact_paths["diagnosis_md"].name == "phase34_case_map_diagnostics.md"
    assert all(path.exists() for path in artifact_paths.values())

    saved = json.loads(
        artifact_paths["diagnosis_json"].read_text(encoding="utf-8")
    )
    assert saved["phase34_case_map_status"] == "case_map_diagnostics_ready"
    markdown = artifact_paths["diagnosis_md"].read_text(encoding="utf-8")
    assert "Phase 34 Case-Map Diagnostics" in markdown
    assert "variant_selects_higher_base_reward_blocks" in markdown


def test_phase34_cli_writes_outputs(tmp_path):
    paths = _write_phase34_fixture_inputs(tmp_path)
    script = (
        ROOT
        / "experiments"
        / "phase34_case_map_diagnostics"
        / "run_phase34_case_map_diagnostics.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--phase33-output-dirs",
            *(str(path) for path in paths["case_dirs"]),
            "--phase2-features-csv",
            str(paths["features_csv"]),
            "--tile-index-csv",
            str(paths["tile_index_csv"]),
            "--output-dir",
            str(tmp_path / "cli_outputs"),
            "--variants",
            "N1ZR",
            "--comparators",
            "D4P8",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "Phase 34 case-map status: case_map_diagnostics_ready"
        in result.stdout
    )
    assert (
        tmp_path
        / "cli_outputs"
        / "phase34_case_map_diagnostics.json"
    ).exists()
