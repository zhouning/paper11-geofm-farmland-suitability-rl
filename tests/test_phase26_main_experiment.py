import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_phase25_fixture(output_dir: Path, learned_delta_pattern: str = "supported") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "phase25_padded_heldout_policy_summary.csv"
    fieldnames = [
        "row_type",
        "variant_id",
        "train_tile_id",
        "eval_tile_id",
        "eval_tile_rank",
        "seed",
        "phase25_seed_rank",
        "train_timesteps",
        "eval_max_steps",
        "max_blocks",
        "train_n_blocks",
        "eval_n_blocks",
        "n_features",
        "observation_shape",
        "action_space_n",
        "episode_steps",
        "terminated",
        "truncated",
        "all_actions_valid",
        "invalid_action_count",
        "total_contract_reward",
        "selected_block_ids",
        "claim_boundary",
    ]
    rows = _phase25_rows(learned_delta_pattern)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    comparison = {
        "phase": "phase25_padded_heldout_policy_comparison",
        "train_tile_id": "tile_train",
        "train_tile_ids": ["tile_train"],
        "eval_tile_ids": ["tile_eval_a", "tile_eval_b"],
        "variants": ["B0", "B1"],
        "seeds": [0, 1],
        "seed_count": 2,
        "total_timesteps": 1024,
        "eval_max_steps": 8,
        "max_blocks": 10,
        "learned_policy": {"B1_minus_B0_mean_reward": 0.5},
        "remaining_evidence_gaps": ["suitability_reward_validation_before_B2_B3"],
    }
    (output_dir / "phase25_padded_heldout_policy_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    return output_dir


def _phase25_rows(pattern: str) -> list[dict[str, object]]:
    if pattern == "supported":
        learned = {
            ("tile_eval_a", 0): (1.0, 1.4),
            ("tile_eval_a", 1): (0.8, 1.1),
            ("tile_eval_b", 0): (0.4, 0.7),
            ("tile_eval_b", 1): (0.6, 0.6),
        }
    elif pattern == "mixed":
        learned = {
            ("tile_eval_a", 0): (1.0, 1.5),
            ("tile_eval_a", 1): (1.0, 1.4),
            ("tile_eval_b", 0): (1.0, 0.8),
            ("tile_eval_b", 1): (1.0, 0.9),
        }
    elif pattern == "not_supported":
        learned = {
            ("tile_eval_a", 0): (1.0, 0.8),
            ("tile_eval_a", 1): (1.0, 0.9),
            ("tile_eval_b", 0): (1.0, 1.0),
            ("tile_eval_b", 1): (1.0, 0.7),
        }
    else:
        learned = {}

    rows: list[dict[str, object]] = []
    for tile_rank, tile_id in enumerate(["tile_eval_a", "tile_eval_b"], start=1):
        for seed_rank, seed in enumerate([0, 1], start=1):
            b0, b1 = learned.get((tile_id, seed), (1.0, 1.0))
            rows.append(_phase25_row("trained_policy", "B0", tile_id, tile_rank, seed, seed_rank, b0))
            rows.append(_phase25_row("trained_policy", "B1", tile_id, tile_rank, seed, seed_rank, b1))
            rows.append(_phase25_row("first_valid", "B0", tile_id, tile_rank, seed, seed_rank, 0.2))
            rows.append(_phase25_row("first_valid", "B1", tile_id, tile_rank, seed, seed_rank, 0.2))
            rows.append(_phase25_row("seeded_random", "B0", tile_id, tile_rank, seed, seed_rank, 0.1))
            rows.append(_phase25_row("seeded_random", "B1", tile_id, tile_rank, seed, seed_rank, 0.15))
    return rows


def _phase25_row(row_type, variant_id, eval_tile_id, eval_tile_rank, seed, seed_rank, reward):
    return {
        "row_type": row_type,
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": eval_tile_id,
        "eval_tile_rank": eval_tile_rank,
        "seed": seed,
        "phase25_seed_rank": seed_rank,
        "train_timesteps": 1024,
        "eval_max_steps": 8,
        "max_blocks": 10,
        "train_n_blocks": 10,
        "eval_n_blocks": 5,
        "n_features": 17 if variant_id == "B0" else 81,
        "observation_shape": 190,
        "action_space_n": 10,
        "episode_steps": 4,
        "terminated": True,
        "truncated": False,
        "all_actions_valid": True,
        "invalid_action_count": 0,
        "total_contract_reward": reward,
        "selected_block_ids": "b1;b2",
        "claim_boundary": "phase25 fixture",
    }


def test_phase26_builds_main_empirical_analysis_from_phase25_outputs(tmp_path):
    from paper11_geofm.phase26_main_experiment import (
        PHASE26_CLAIM_BOUNDARY,
        build_phase26_main_empirical_analysis,
    )

    phase25_dir = _write_phase25_fixture(tmp_path / "phase25", "supported")
    analysis = build_phase26_main_empirical_analysis(phase25_dir)

    assert analysis["phase"] == "phase26_main_empirical_experiment"
    assert analysis["source_phase25"]["summary_csv"].endswith("phase25_padded_heldout_policy_summary.csv")
    assert analysis["variants"] == ["B0", "B1"]
    assert analysis["seeds"] == [0, 1]
    assert analysis["eval_tile_ids"] == ["tile_eval_a", "tile_eval_b"]
    assert analysis["train_timesteps"] == 1024
    assert analysis["eval_max_steps"] == 8
    assert analysis["learned_policy"]["B1_minus_B0_mean_reward"] == 0.25
    assert analysis["learned_policy"]["positive_tile_seed_count"] == 3
    assert analysis["learned_policy"]["total_tile_seed_count"] == 4
    assert analysis["phase26_claim_status"] == "pilot_supported"
    assert analysis["claim_boundary"] == PHASE26_CLAIM_BOUNDARY


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("supported", "pilot_supported"),
        ("mixed", "mixed"),
        ("not_supported", "not_supported"),
    ],
)
def test_phase26_claim_status_rules(tmp_path, pattern, expected):
    from paper11_geofm.phase26_main_experiment import build_phase26_main_empirical_analysis

    phase25_dir = _write_phase25_fixture(tmp_path / pattern, pattern)
    analysis = build_phase26_main_empirical_analysis(phase25_dir)

    assert analysis["phase26_claim_status"] == expected


def test_phase26_reports_insufficient_when_b1_rows_are_missing(tmp_path):
    from paper11_geofm.phase26_main_experiment import build_phase26_main_empirical_analysis

    phase25_dir = _write_phase25_fixture(tmp_path / "phase25", "supported")
    summary_path = phase25_dir / "phase25_padded_heldout_policy_summary.csv"
    rows = [row for row in csv.DictReader(summary_path.open("r", encoding="utf-8")) if row["variant_id"] != "B1"]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    analysis = build_phase26_main_empirical_analysis(phase25_dir)

    assert analysis["phase26_claim_status"] == "insufficient"


def test_phase26_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase26_main_experiment import (
        build_phase26_main_empirical_analysis,
        write_phase26_main_empirical_artifacts,
    )

    phase25_dir = _write_phase25_fixture(tmp_path / "phase25", "supported")
    analysis = build_phase26_main_empirical_analysis(phase25_dir)
    paths = write_phase26_main_empirical_artifacts(analysis, tmp_path / "outputs")

    assert paths["main_summary_csv"].name == "phase26_main_summary.csv"
    assert paths["tile_seed_delta_csv"].name == "phase26_tile_seed_delta_table.csv"
    assert paths["comparison_json"].name == "phase26_main_comparison.json"
    assert paths["claim_readiness_md"].name == "phase26_claim_readiness.md"
    delta_rows = list(csv.DictReader(paths["tile_seed_delta_csv"].open("r", encoding="utf-8")))
    assert delta_rows[0]["eval_tile_id"] == "tile_eval_a"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase26_claim_status"] == "pilot_supported"
    markdown = paths["claim_readiness_md"].read_text(encoding="utf-8")
    assert "pilot_supported" in markdown
    assert "suitability reward" in markdown


def test_phase26_cli_analyze_only_writes_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase26_main_experiment"
        / "run_phase26_main_experiment.py"
    )
    spec = importlib.util.spec_from_file_location("phase26_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    phase25_dir = _write_phase25_fixture(tmp_path / "phase25", "supported")
    exit_code = module.main(
        [
            "--mode",
            "analyze-only",
            "--phase25-output-dir",
            str(phase25_dir),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Phase 26 claim status: pilot_supported" in stdout
    assert "B1-B0 learned-policy mean reward delta: 0.25" in stdout
    assert "phase26_main_comparison.json" in stdout


def test_phase26_cli_run_and_analyze_requires_phase25_run_inputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase26_main_experiment"
        / "run_phase26_main_experiment.py"
    )
    spec = importlib.util.spec_from_file_location("phase26_runner_validation", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--mode",
            "run-and-analyze",
            "--phase25-output-dir",
            str(tmp_path / "phase25"),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert "run-and-analyze requires" in stderr
