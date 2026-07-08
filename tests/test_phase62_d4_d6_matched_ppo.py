import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _summary_row(variant_id, reward, tile_id="tile_a", seed=0):
    return {
        "row_type": "trained_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": tile_id,
        "eval_tile_rank": 1 if tile_id == "tile_a" else 2,
        "seed": seed,
        "phase25_seed_rank": seed + 1,
        "train_timesteps": 4096,
        "eval_max_steps": 8,
        "max_blocks": 4,
        "train_n_blocks": 4,
        "eval_n_blocks": 2,
        "n_features": 25,
        "observation_shape": 100,
        "action_space_n": 4,
        "episode_steps": 2,
        "terminated": True,
        "truncated": False,
        "all_actions_valid": True,
        "invalid_action_count": 0,
        "total_contract_reward": reward,
        "selected_block_ids": "b1;b2",
        "claim_boundary": "fixture",
    }


def _phase62_summary_rows(case="supported"):
    rewards = {
        "supported": {
            "D4P8": [1.25, 1.25, 1.25, 1.25],
            "D6R8": [1.0, 1.0, 1.0, 1.0],
            "D4P16": [1.5, 1.5, 1.5, 1.5],
            "D6R16": [1.25, 1.25, 1.25, 1.25],
        },
        "d6_advantage": {
            "D4P8": [0.75, 0.75, 0.75, 0.75],
            "D6R8": [1.0, 1.0, 1.0, 1.0],
            "D4P16": [1.0, 1.0, 1.0, 1.0],
            "D6R16": [1.25, 1.25, 1.25, 1.25],
        },
        "mixed": {
            "D4P8": [1.25, 1.25, 1.25, 1.25],
            "D6R8": [1.0, 1.0, 1.0, 1.0],
            "D4P16": [1.0, 1.0, 1.0, 1.0],
            "D6R16": [1.25, 1.25, 1.25, 1.25],
        },
    }[case]
    pairs = [("tile_a", 0), ("tile_a", 1), ("tile_b", 0), ("tile_b", 1)]
    rows = []
    for index, (tile_id, seed) in enumerate(pairs):
        for variant_id, values in rewards.items():
            rows.append(_summary_row(variant_id, values[index], tile_id, seed))
    return rows


def test_phase62_contract_routes_d4_and_d6_variants(tmp_path):
    from paper11_geofm.phase62_d4_d6_matched_ppo import (
        build_phase62_d4_d6_contract,
    )

    tile_index = _write_csv(
        tmp_path / "tiles.csv",
        [
            {"tile_id": "tile_train", "block_ids": "b1;b2;b3;b4"},
            {"tile_id": "tile_a", "block_ids": "b1;b2"},
        ],
    )
    contract = build_phase62_d4_d6_contract(
        phase8_output_dir=tmp_path / "phase8",
        phase61_output_dir=tmp_path / "phase61",
        tile_index_csv=tile_index,
        train_tile_id="tile_train",
        eval_tile_ids="tile_a",
        variants="D4P8,D4P16,D6R8,D6R16",
        seeds="0",
    )

    assert contract["variants"] == ["D4P8", "D4P16", "D6R8", "D6R16"]
    assert contract["variant_source_dirs"]["D4P8"].endswith("phase8")
    assert contract["variant_source_dirs"]["D6R16"].endswith("phase61")
    assert contract["eval_tile_ids"] == ["tile_a"]
    assert contract["seeds"] == [0]


def test_phase62_analysis_supports_d4_pca_advantage():
    from paper11_geofm.phase62_d4_d6_matched_ppo import (
        build_phase62_d4_d6_analysis,
    )

    analysis = build_phase62_d4_d6_analysis(
        _phase62_summary_rows("supported"),
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
        bootstrap_iterations=100,
        random_seed=62,
    )

    assert analysis["phase"] == "phase62_d4_d6_matched_ppo_analysis"
    assert analysis["phase62_d4_d6_status"] == "d4_pca_advantage_over_d6_supported"
    assert analysis["matched_deltas"]["D4P8_minus_D6R8"]["mean_delta"] == 0.25
    assert analysis["pooled_primary_delta"]["positive_count"] == 8
    assert analysis["cluster_summary"]["cluster_count"] == 4
    assert analysis["signed_rank_summary"]["positive_rank_sum"] == 10


def test_phase62_status_rules_distinguish_d6_advantage_and_mixed():
    from paper11_geofm.phase62_d4_d6_matched_ppo import (
        build_phase62_d4_d6_analysis,
    )

    d6_advantage = build_phase62_d4_d6_analysis(
        _phase62_summary_rows("d6_advantage"),
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
        bootstrap_iterations=100,
    )
    mixed = build_phase62_d4_d6_analysis(
        _phase62_summary_rows("mixed"),
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
        bootstrap_iterations=100,
    )

    assert d6_advantage["phase62_d4_d6_status"] == "d6_random_projection_advantage"
    assert mixed["phase62_d4_d6_status"] == "d4_d6_not_distinguishable"


def test_phase62_reports_insufficient_for_missing_primary_rows():
    from paper11_geofm.phase62_d4_d6_matched_ppo import (
        build_phase62_d4_d6_analysis,
    )

    rows = [
        row for row in _phase62_summary_rows("supported")
        if row["variant_id"] != "D6R16"
    ]
    analysis = build_phase62_d4_d6_analysis(
        rows,
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
    )

    assert analysis["phase62_d4_d6_status"] == "insufficient"
    missing = {
        row["variant_id"] for row in analysis["coverage_issues"]["missing_variant_rows"]
    }
    assert missing == {"D6R16"}


def test_phase62_writer_outputs_summary_traces_delta_cluster_json_and_markdown(tmp_path):
    from paper11_geofm.phase62_d4_d6_matched_ppo import (
        build_phase62_d4_d6_analysis,
        write_phase62_d4_d6_artifacts,
    )

    rows = _phase62_summary_rows("supported")
    analysis = build_phase62_d4_d6_analysis(
        rows,
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
        bootstrap_iterations=100,
    )
    paths = write_phase62_d4_d6_artifacts(
        {**analysis, "summaries": rows, "traces": {}},
        tmp_path / "outputs",
    )

    assert paths["summary_csv"].name == "phase62_d4_d6_matched_ppo_summary.csv"
    assert paths["traces_json"].name == "phase62_d4_d6_matched_ppo_traces.json"
    assert paths["delta_csv"].name == "phase62_d4_d6_delta_table.csv"
    assert paths["cluster_csv"].name == "phase62_d4_d6_cluster_summary.csv"
    assert paths["comparison_json"].name == "phase62_d4_d6_matched_ppo.json"
    assert paths["readiness_md"].name == "phase62_d4_d6_matched_ppo.md"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase62_d4_d6_status"] == "d4_pca_advantage_over_d6_supported"
    with paths["delta_csv"].open("r", encoding="utf-8", newline="") as handle:
        delta_rows = list(csv.DictReader(handle))
    assert any(
        row["d4_variant_id"] == "D4P16" and row["d6_variant_id"] == "D6R16"
        for row in delta_rows
    )
    markdown = paths["readiness_md"].read_text(encoding="utf-8")
    assert "D4/D6 matched PPO evaluation" in markdown
    assert "does not enable suitability reward" in markdown


def test_phase62_cli_analyze_only(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase62_d4_d6_matched_ppo"
        / "run_phase62_d4_d6_matched_ppo.py"
    )
    spec = importlib.util.spec_from_file_location("phase62_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    summary_csv = _write_csv(tmp_path / "summary.csv", _phase62_summary_rows("supported"))
    analyze_exit = module.main(
        [
            "--mode",
            "analyze-only",
            "--existing-summary-csv",
            str(summary_csv),
            "--output-dir",
            str(tmp_path / "analysis"),
            "--eval-tile-ids",
            "tile_a,tile_b",
            "--seeds",
            "0,1",
            "--bootstrap-iterations",
            "100",
        ]
    )

    stdout = capsys.readouterr().out
    assert analyze_exit == 0
    assert "Phase 62 D4/D6 status: d4_pca_advantage_over_d6_supported" in stdout
    assert "phase62_d4_d6_matched_ppo.json" in stdout


def test_phase62_cli_run_and_analyze_accepts_existing_summary_arg():
    runner_path = (
        ROOT
        / "experiments"
        / "phase62_d4_d6_matched_ppo"
        / "run_phase62_d4_d6_matched_ppo.py"
    )
    spec = importlib.util.spec_from_file_location("phase62_runner_args", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "run-and-analyze",
            "--phase8-output-dir",
            "phase8",
            "--phase61-output-dir",
            "phase61",
            "--tile-index-csv",
            "tiles.csv",
            "--existing-summary-csv",
            "existing.csv",
            "--variants",
            "D4P8,D6R8",
            "--output-dir",
            "outputs",
        ]
    )

    assert args.existing_summary_csv == Path("existing.csv")
    assert args.variants == "D4P8,D6R8"
