import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _summary_row(variant_id, reward, tile_id="tile_eval_a", seed=0):
    return {
        "row_type": "trained_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": tile_id,
        "eval_tile_rank": 1 if tile_id == "tile_eval_a" else 2,
        "seed": seed,
        "phase25_seed_rank": seed + 1,
        "train_timesteps": 4096,
        "eval_max_steps": 8,
        "max_blocks": 4,
        "train_n_blocks": 4,
        "eval_n_blocks": 2,
        "n_features": 33 if variant_id == "D4P16" else 25 if variant_id == "D4P8" else 81,
        "observation_shape": 333,
        "action_space_n": 4,
        "episode_steps": 2,
        "terminated": True,
        "truncated": False,
        "all_actions_valid": True,
        "invalid_action_count": 0,
        "total_contract_reward": reward,
        "selected_block_ids": "b1;b3",
        "claim_boundary": "fixture",
    }


def _phase48_summary_rows(case="supported"):
    rewards_by_case = {
        "supported": {
            "B0": [1.0, 1.0, 1.0, 1.0],
            "B1": [0.7, 0.8, 0.9, 0.8],
            "D2": [0.8, 0.8, 0.8, 0.8],
            "D3": [0.9, 0.9, 0.9, 0.9],
            "D4P8": [1.2, 1.1, 1.3, 1.0],
            "D4P16": [1.4, 1.3, 1.2, 1.1],
        },
        "partial": {
            "B0": [1.0, 1.0, 1.0, 1.0],
            "B1": [0.7, 0.8, 0.9, 0.8],
            "D2": [1.3, 1.3, 1.3, 1.3],
            "D3": [1.2, 1.2, 1.2, 1.2],
            "D4P8": [1.1, 1.1, 1.1, 1.1],
            "D4P16": [1.15, 1.15, 1.15, 1.15],
        },
        "not_supported": {
            "B0": [1.0, 1.0, 1.0, 1.0],
            "B1": [0.9, 0.9, 0.9, 0.9],
            "D2": [0.8, 0.8, 0.8, 0.8],
            "D3": [0.8, 0.8, 0.8, 0.8],
            "D4P8": [0.7, 0.7, 0.7, 0.7],
            "D4P16": [0.75, 0.75, 0.75, 0.75],
        },
    }[case]
    rows = []
    pairs = [
        ("tile_eval_a", 0),
        ("tile_eval_a", 1),
        ("tile_eval_b", 0),
        ("tile_eval_b", 1),
    ]
    for pair_index, (tile_id, seed) in enumerate(pairs):
        for variant_id, values in rewards_by_case.items():
            rows.append(_summary_row(variant_id, values[pair_index], tile_id, seed))
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(_summary_row("B0", 1.0).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_phase48_analysis_supports_compressed_geofm_route_with_controls():
    from paper11_geofm.phase48_compressed_geofm_rescue import (
        PHASE48_CLAIM_BOUNDARY,
        build_phase48_compressed_geofm_rescue_analysis,
    )

    analysis = build_phase48_compressed_geofm_rescue_analysis(
        _phase48_summary_rows("supported"),
        metadata={
            "variants": ["B0", "B1", "D2", "D3", "D4P8", "D4P16"],
            "eval_tile_ids": ["tile_eval_a", "tile_eval_b"],
            "seeds": [0, 1],
            "train_timesteps": 4096,
            "eval_max_steps": 8,
        },
    )

    assert analysis["phase"] == "phase48_compressed_geofm_rescue_analysis"
    assert analysis["phase48_compressed_geofm_status"] == "compressed_geofm_route_supported"
    assert analysis["claim_boundary"] == PHASE48_CLAIM_BOUNDARY
    assert analysis["learned_policy"]["mean_reward_by_variant"]["D4P16"] == 1.25
    assert (
        analysis["learned_policy"]["compressed_deltas"]["D4P8_minus_B1"][
            "mean_reward_delta"
        ]
        == 0.35
    )
    assert (
        analysis["learned_policy"]["compressed_deltas"]["D4P16_minus_D3"][
            "positive_tile_seed_count"
        ]
        == 4
    )
    assert analysis["pooled_compressed_control_delta"]["positive_fraction"] == 0.96875
    assert "raw direct injection" in analysis["conclusion"]
    assert "suitability reward remains blocked" in analysis["conclusion"]


@pytest.mark.parametrize(
    ("case", "expected_status"),
    [
        ("partial", "compressed_geofm_route_partial"),
        ("not_supported", "compressed_geofm_route_not_supported"),
    ],
)
def test_phase48_status_rules_distinguish_partial_and_not_supported(case, expected_status):
    from paper11_geofm.phase48_compressed_geofm_rescue import (
        build_phase48_compressed_geofm_rescue_analysis,
    )

    analysis = build_phase48_compressed_geofm_rescue_analysis(
        _phase48_summary_rows(case),
        metadata={
            "variants": ["B0", "B1", "D2", "D3", "D4P8", "D4P16"],
            "eval_tile_ids": ["tile_eval_a", "tile_eval_b"],
            "seeds": [0, 1],
        },
    )

    assert analysis["phase48_compressed_geofm_status"] == expected_status


def test_phase48_reports_insufficient_for_missing_control_rows():
    from paper11_geofm.phase48_compressed_geofm_rescue import (
        build_phase48_compressed_geofm_rescue_analysis,
    )

    rows = [row for row in _phase48_summary_rows("supported") if row["variant_id"] != "D3"]

    analysis = build_phase48_compressed_geofm_rescue_analysis(
        rows,
        metadata={
            "variants": ["B0", "B1", "D2", "D3", "D4P8", "D4P16"],
            "eval_tile_ids": ["tile_eval_a", "tile_eval_b"],
            "seeds": [0, 1],
        },
    )

    assert analysis["phase48_compressed_geofm_status"] == "insufficient"
    missing_variants = {
        row["variant_id"] for row in analysis["coverage_issues"]["missing_variant_rows"]
    }
    assert missing_variants == {"D3"}


def test_phase48_writer_outputs_json_delta_and_markdown(tmp_path):
    from paper11_geofm.phase48_compressed_geofm_rescue import (
        build_phase48_compressed_geofm_rescue_analysis,
        write_phase48_compressed_geofm_rescue_artifacts,
    )

    rows = _phase48_summary_rows("supported")
    analysis = build_phase48_compressed_geofm_rescue_analysis(rows)

    paths = write_phase48_compressed_geofm_rescue_artifacts(
        {**analysis, "summaries": rows},
        tmp_path / "outputs",
    )

    assert paths["summary_csv"].name == "phase48_compressed_geofm_rescue_summary.csv"
    assert paths["comparison_json"].name == "phase48_compressed_geofm_rescue_comparison.json"
    assert paths["delta_csv"].name == "phase48_compressed_geofm_rescue_delta_table.csv"
    assert paths["readiness_md"].name == "phase48_compressed_geofm_rescue_readiness.md"
    assert all(path.exists() for path in paths.values())

    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase48_compressed_geofm_status"] == "compressed_geofm_route_supported"

    with paths["delta_csv"].open("r", encoding="utf-8", newline="") as handle:
        delta_rows = list(csv.DictReader(handle))
    assert any(
        row["compressed_variant_id"] == "D4P16"
        and row["comparator_variant_id"] == "D3"
        for row in delta_rows
    )

    markdown = paths["readiness_md"].read_text(encoding="utf-8")
    assert "Compressed GeoFM route" in markdown
    assert "does not enable suitability reward" in markdown
    assert "GeoFM is useless" not in markdown


def test_phase48_cli_analyze_only_writes_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase48_compressed_geofm_rescue"
        / "run_phase48_compressed_geofm_rescue.py"
    )
    spec = importlib.util.spec_from_file_location("phase48_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    summary_csv = _write_summary_csv(
        tmp_path / "summary.csv",
        _phase48_summary_rows("supported"),
    )

    exit_code = module.main(
        [
            "--existing-summary-csv",
            str(summary_csv),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "Phase 48 compressed GeoFM status: compressed_geofm_route_supported"
        in stdout
    )
    assert "phase48_compressed_geofm_rescue_comparison.json" in stdout


