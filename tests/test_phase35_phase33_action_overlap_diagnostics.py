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


def _summary_row(variant_id, tile_id, seed, reward, selected):
    from paper11_geofm.padded_heldout_policy import PHASE25_CLAIM_BOUNDARY

    return {
        "row_type": "trained_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": tile_id,
        "eval_tile_rank": 1,
        "seed": seed,
        "phase25_seed_rank": int(seed) + 1,
        "train_timesteps": 5120,
        "eval_max_steps": 8,
        "max_blocks": 4,
        "train_n_blocks": 4,
        "eval_n_blocks": 4,
        "n_features": 81,
        "observation_shape": 100,
        "action_space_n": 4,
        "episode_steps": len(selected.split(";")),
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


def _write_case_dir(tmp_path: Path) -> Path:
    from paper11_geofm.padded_heldout_policy import SUMMARY_FIELDNAMES
    from paper11_geofm.phase33_budget_robustness import TILE_SEED_STABILITY_FIELDNAMES

    case_dir = tmp_path / "phase33_matched"
    high_dir = case_dir / "phase30_high_budget"
    summary_rows = [
        _summary_row("N1ZR", "tile_alpha", 0, 1.2, "a;b;c"),
        _summary_row("D4P8", "tile_alpha", 0, 0.6, "b;c;d"),
        _summary_row("N1Z", "tile_beta", 1, 0.1, "a;b"),
        _summary_row("B1", "tile_beta", 1, 0.4, "a;c"),
    ]
    _write_csv(
        high_dir / "phase30_normalized_b1_summary.csv",
        summary_rows,
        SUMMARY_FIELDNAMES,
    )
    traces = {
        "trained_policy": {
            "N1ZR": {
                "tile_alpha": {
                    "0": [
                        {"step": 1, "selected_block_id": "c", "reward": 0.1},
                        {"step": 2, "selected_block_id": "b", "reward": 0.2},
                        {"step": 3, "selected_block_id": "a", "reward": 0.9},
                    ]
                }
            },
            "N1Z": {
                "tile_beta": {
                    "1": [
                        {"step": 1, "selected_block_id": "a", "reward": 0.2},
                        {"step": 2, "selected_block_id": "b", "reward": -0.1},
                    ]
                }
            },
            "B1": {
                "tile_beta": {
                    "1": [
                        {"step": 1, "selected_block_id": "a", "reward": 0.3},
                        {"step": 2, "selected_block_id": "c", "reward": 0.1},
                    ]
                }
            },
        }
    }
    (high_dir / "phase30_normalized_b1_traces.json").write_text(
        json.dumps(traces, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(
        case_dir / "phase33_tile_seed_stability.csv",
        [
            _stability_row("N1ZR", "D4P8", "tile_alpha", 0, -0.3, 0.6, "flip_to_positive"),
            _stability_row("N1Z", "B1", "tile_beta", 1, -0.2, -0.3, "stable_negative"),
        ],
        TILE_SEED_STABILITY_FIELDNAMES,
    )
    return case_dir


def test_phase35_builds_phase33_action_overlap_with_trace_and_summary_fallback(tmp_path):
    from paper11_geofm.phase35_phase33_action_overlap_diagnostics import (
        PHASE35_ACTION_OVERLAP_CLAIM_BOUNDARY,
        build_phase35_phase33_action_overlap_diagnostics,
    )

    case_dir = _write_case_dir(tmp_path)
    analysis = build_phase35_phase33_action_overlap_diagnostics(
        phase33_output_dirs=[case_dir],
        variants=["N1ZR", "N1Z"],
        comparators=["D4P8", "B1"],
    )

    assert analysis["phase"] == "phase35_phase33_action_overlap_diagnostics"
    assert analysis["phase35_action_overlap_status"] == "action_overlap_diagnostics_ready"
    assert analysis["claim_boundary"] == PHASE35_ACTION_OVERLAP_CLAIM_BOUNDARY
    assert analysis["row_counts"]["case_rows"] == 2
    assert analysis["row_counts"]["step_rows"] == 5

    cases = {row["case_id"]: row for row in analysis["case_rows"]}
    alpha = cases["tile_alpha|0|N1ZR|D4P8"]
    assert alpha["variant_step_source"] == "trace"
    assert alpha["comparator_step_source"] == "summary_selected_block_ids"
    assert alpha["selected_block_jaccard"] == 0.5
    assert alpha["shared_block_count"] == 2
    assert alpha["mean_abs_shared_step_displacement"] == 1.0
    assert alpha["first_step_reward_gap"] == ""
    assert alpha["summary_reward_gap"] == 0.6
    assert alpha["action_overlap_pattern"] == "partial_overlap_positive_gap"

    beta = cases["tile_beta|1|N1Z|B1"]
    assert beta["variant_step_source"] == "trace"
    assert beta["comparator_step_source"] == "trace"
    assert beta["selected_block_jaccard"] == 0.3333333333
    assert beta["same_step_match_count"] == 1
    assert beta["first_step_reward_gap"] == -0.1
    assert beta["trace_cumulative_reward_gap"] == -0.3
    assert beta["action_overlap_pattern"] == "partial_overlap_negative_gap"

    beta_steps = [
        row
        for row in analysis["step_rows"]
        if row["case_id"] == "tile_beta|1|N1Z|B1"
    ]
    assert beta_steps[0]["same_step_block_match"] is True
    assert beta_steps[0]["step_reward_gap"] == -0.1
    assert beta_steps[1]["cumulative_reward_gap"] == -0.3


def test_phase35_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase35_phase33_action_overlap_diagnostics import (
        build_phase35_phase33_action_overlap_diagnostics,
        write_phase35_phase33_action_overlap_diagnostics_artifacts,
    )

    case_dir = _write_case_dir(tmp_path)
    analysis = build_phase35_phase33_action_overlap_diagnostics(
        phase33_output_dirs=[case_dir],
        variants=["N1ZR", "N1Z"],
        comparators=["D4P8", "B1"],
    )
    artifact_paths = write_phase35_phase33_action_overlap_diagnostics_artifacts(
        analysis,
        tmp_path / "outputs",
    )

    assert artifact_paths["case_summary_csv"].name == "phase35_action_overlap_cases.csv"
    assert artifact_paths["step_alignment_csv"].name == "phase35_action_overlap_steps.csv"
    assert artifact_paths["diagnosis_json"].name == "phase35_action_overlap_diagnostics.json"
    assert artifact_paths["diagnosis_md"].name == "phase35_action_overlap_diagnostics.md"
    assert all(path.exists() for path in artifact_paths.values())

    saved = json.loads(
        artifact_paths["diagnosis_json"].read_text(encoding="utf-8")
    )
    assert saved["phase35_action_overlap_status"] == "action_overlap_diagnostics_ready"
    markdown = artifact_paths["diagnosis_md"].read_text(encoding="utf-8")
    assert "Phase 35 Phase 33 Action-Overlap Diagnostics" in markdown
    assert "partial_overlap_positive_gap" in markdown


def test_phase35_cli_writes_outputs(tmp_path):
    case_dir = _write_case_dir(tmp_path)
    script = (
        ROOT
        / "experiments"
        / "phase35_phase33_action_overlap_diagnostics"
        / "run_phase35_phase33_action_overlap_diagnostics.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--phase33-output-dirs",
            str(case_dir),
            "--output-dir",
            str(tmp_path / "cli_outputs"),
            "--variants",
            "N1ZR,N1Z",
            "--comparators",
            "D4P8,B1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "Phase 35 action-overlap status: action_overlap_diagnostics_ready"
        in result.stdout
    )
    assert (
        tmp_path
        / "cli_outputs"
        / "phase35_action_overlap_diagnostics.json"
    ).exists()
