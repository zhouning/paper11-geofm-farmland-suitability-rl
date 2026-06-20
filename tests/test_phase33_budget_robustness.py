import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_phase30_comparison(
    path: Path,
    *,
    timesteps: int,
    deltas: dict[tuple[str, str, str, int], float],
    status: str = "normalized_b1_recovers_b0_gap",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    focal_deltas: dict[str, dict[str, object]] = {}
    tile_seed_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[float]] = {}
    positive_counts: dict[tuple[str, str], int] = {}
    for (variant_id, comparator_id, tile_id, seed), value in sorted(deltas.items()):
        key = (variant_id, comparator_id)
        grouped.setdefault(key, []).append(value)
        positive_counts[key] = positive_counts.get(key, 0) + int(value > 0.0)
        tile_seed_rows.append(
            {
                "variant_id": variant_id,
                "comparator_variant_id": comparator_id,
                "eval_tile_id": tile_id,
                "seed": seed,
                "variant_reward": round(1.0 + value, 10),
                "comparator_reward": 1.0,
                "variant_minus_comparator_reward": round(value, 10),
                "variant_improves_comparator": value > 0.0,
                "train_timesteps": timesteps,
                "eval_max_steps": 8,
                "claim_boundary": "fixture",
            }
        )
    for (variant_id, comparator_id), values in grouped.items():
        focal_deltas[f"{variant_id}_minus_{comparator_id}"] = {
            "mean_reward_delta": round(sum(values) / len(values), 10),
            "std_reward_delta": 0.0,
            "positive_tile_seed_count": positive_counts[(variant_id, comparator_id)],
            "total_tile_seed_count": len(values),
            "positive_fraction": round(
                positive_counts[(variant_id, comparator_id)] / len(values), 10
            ),
        }
    payload = {
        "phase": "phase30_normalized_b1_analysis",
        "train_timesteps": timesteps,
        "eval_max_steps": 8,
        "variants": ["B1", "N1Z", "N1ZR", "D4P8", "D4P16"],
        "eval_tile_ids": sorted({tile_id for _v, _c, tile_id, _s in deltas}),
        "seeds": sorted({seed for _v, _c, _tile, seed in deltas}),
        "phase30_normalized_b1_status": status,
        "learned_policy": {
            "mean_reward_by_variant": {},
            "focal_deltas": focal_deltas,
        },
        "delta_rows": tile_seed_rows,
        "claim_boundary": "fixture",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_phase33_builds_budget_transition_and_tile_seed_stability(tmp_path):
    from paper11_geofm.phase33_budget_robustness import (
        PHASE33_BUDGET_ROBUSTNESS_CLAIM_BOUNDARY,
        build_phase33_budget_robustness,
    )

    lower = _write_phase30_comparison(
        tmp_path / "lower" / "phase30_normalized_b1_comparison.json",
        timesteps=4096,
        deltas={
            ("N1Z", "D4P16", "tile_a", 0): -0.3,
            ("N1Z", "D4P16", "tile_a", 1): -0.1,
            ("N1ZR", "D4P16", "tile_a", 0): -0.5,
            ("N1ZR", "D4P16", "tile_a", 1): -0.2,
        },
    )
    higher = _write_phase30_comparison(
        tmp_path / "higher" / "phase30_normalized_b1_comparison.json",
        timesteps=8192,
        deltas={
            ("N1Z", "D4P16", "tile_a", 0): -0.1,
            ("N1Z", "D4P16", "tile_a", 1): 0.2,
            ("N1ZR", "D4P16", "tile_a", 0): -0.4,
            ("N1ZR", "D4P16", "tile_a", 1): -0.1,
        },
    )

    analysis = build_phase33_budget_robustness([lower, higher])

    assert analysis["phase"] == "phase33_budget_robustness"
    assert analysis["claim_boundary"] == PHASE33_BUDGET_ROBUSTNESS_CLAIM_BOUNDARY
    assert analysis["phase33_budget_status"] == "budget_improves_but_not_closed"
    assert analysis["budget_transition_rows"][1]["train_timesteps"] == 8192
    by_gap = {
        (
            row["variant_id"],
            row["comparator_variant_id"],
            row["budget_label"],
        ): row
        for row in analysis["focal_gap_transition_rows"]
    }
    assert by_gap[("N1Z", "D4P16", "8192_steps")]["mean_delta_change_from_previous"] == 0.25
    assert analysis["tile_seed_stability_counts"]["flip_to_positive"] == 1


def test_phase33_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase33_budget_robustness import (
        build_phase33_budget_robustness,
        write_phase33_budget_robustness_artifacts,
    )

    lower = _write_phase30_comparison(
        tmp_path / "lower" / "phase30_normalized_b1_comparison.json",
        timesteps=4096,
        deltas={("N1Z", "D4P16", "tile_a", 0): -0.3},
    )
    higher = _write_phase30_comparison(
        tmp_path / "higher" / "phase30_normalized_b1_comparison.json",
        timesteps=8192,
        deltas={("N1Z", "D4P16", "tile_a", 0): -0.1},
    )
    analysis = build_phase33_budget_robustness([lower, higher])
    paths = write_phase33_budget_robustness_artifacts(analysis, tmp_path / "outputs")

    assert paths["budget_transition_csv"].name == "phase33_budget_transition.csv"
    assert paths["focal_gap_transition_csv"].name == "phase33_focal_gap_transition.csv"
    assert paths["tile_seed_stability_csv"].name == "phase33_tile_seed_stability.csv"
    assert paths["summary_json"].name == "phase33_budget_robustness.json"
    assert paths["summary_md"].name == "phase33_budget_robustness.md"
    rows = list(csv.DictReader(paths["budget_transition_csv"].open("r", encoding="utf-8")))
    assert rows[0]["train_timesteps"] == "4096"
    saved = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert saved["phase33_budget_status"] == "budget_improves_but_not_closed"
    markdown = paths["summary_md"].read_text(encoding="utf-8")
    assert "Phase 33 Budget Robustness" in markdown


def test_phase33_writes_matched_baseline_subset(tmp_path):
    from paper11_geofm.phase33_budget_robustness import (
        build_phase33_budget_robustness,
        write_phase33_matched_baseline_comparison,
    )

    baseline = _write_phase30_comparison(
        tmp_path / "baseline" / "phase30_normalized_b1_comparison.json",
        timesteps=4096,
        deltas={
            ("N1Z", "D4P16", "tile_a", 0): -0.3,
            ("N1Z", "D4P16", "tile_b", 1): -0.1,
            ("N1ZR", "D4P16", "tile_a", 0): -0.5,
        },
    )
    higher = _write_phase30_comparison(
        tmp_path / "higher" / "phase30_normalized_b1_comparison.json",
        timesteps=8192,
        deltas={
            ("N1Z", "D4P16", "tile_a", 0): -0.1,
            ("N1ZR", "D4P16", "tile_a", 0): -0.4,
        },
    )

    subset_path = write_phase33_matched_baseline_comparison(
        baseline,
        higher,
        tmp_path / "phase33_matched_baseline_comparison.json",
    )
    subset = json.loads(subset_path.read_text(encoding="utf-8"))

    assert subset["phase33_subset_source"] == str(baseline)
    assert len(subset["delta_rows"]) == 2
    assert subset["learned_policy"]["focal_deltas"]["N1Z_minus_D4P16"]["total_tile_seed_count"] == 1
    analysis = build_phase33_budget_robustness([subset_path, higher])
    assert analysis["tile_seed_stability_counts"]["incomplete"] == 0


def test_phase33_cli_analyze_only_writes_outputs(tmp_path):
    runner_path = (
        ROOT
        / "experiments"
        / "phase33_budget_robustness"
        / "run_phase33_budget_robustness.py"
    )
    spec = importlib.util.spec_from_file_location("phase33_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    lower = _write_phase30_comparison(
        tmp_path / "lower" / "phase30_normalized_b1_comparison.json",
        timesteps=4096,
        deltas={("N1Z", "D4P16", "tile_a", 0): -0.3},
    )
    higher = _write_phase30_comparison(
        tmp_path / "higher" / "phase30_normalized_b1_comparison.json",
        timesteps=8192,
        deltas={("N1Z", "D4P16", "tile_a", 0): -0.1},
    )

    exit_code = module.main(
        [
            "--mode",
            "analyze-only",
            "--phase30-comparison-json",
            str(lower),
            "--phase30-comparison-json",
            str(higher),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "outputs" / "phase33_budget_robustness.json").exists()


def test_phase33_cli_rejects_single_input(tmp_path):
    runner_path = (
        ROOT
        / "experiments"
        / "phase33_budget_robustness"
        / "run_phase33_budget_robustness.py"
    )
    spec = importlib.util.spec_from_file_location("phase33_runner_err", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    lower = _write_phase30_comparison(
        tmp_path / "lower" / "phase30_normalized_b1_comparison.json",
        timesteps=4096,
        deltas={("N1Z", "D4P16", "tile_a", 0): -0.3},
    )
    exit_code = module.main(
        [
            "--mode",
            "analyze-only",
            "--phase30-comparison-json",
            str(lower),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    assert exit_code == 1


def test_phase33_cli_run_and_analyze_reuses_phase30_training(tmp_path, monkeypatch):
    runner_path = (
        ROOT
        / "experiments"
        / "phase33_budget_robustness"
        / "run_phase33_budget_robustness.py"
    )
    spec = importlib.util.spec_from_file_location("phase33_runner_run", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    baseline = _write_phase30_comparison(
        tmp_path / "baseline" / "phase30_normalized_b1_comparison.json",
        timesteps=4096,
        deltas={("N1Z", "D4P16", "tile_a", 0): -0.3},
    )
    baseline_control_summary = tmp_path / "phase28_control_summary.csv"
    baseline_control_summary.write_text("row_type,variant_id\n", encoding="utf-8")

    captured = {}

    def _fake_run_phase30_normalized_b1_ablation(**kwargs):
        captured.update(kwargs)
        comparison = {
            "phase": "phase30_normalized_b1_analysis",
            "train_timesteps": kwargs["total_timesteps"],
            "eval_max_steps": kwargs["eval_max_steps"],
            "variants": ["B1", "N1Z", "N1ZR", "D4P8", "D4P16"],
            "eval_tile_ids": ["tile_a"],
            "seeds": [0],
            "phase30_normalized_b1_status": "normalized_b1_recovers_b0_gap",
            "learned_policy": {
                "mean_reward_by_variant": {},
                "focal_deltas": {
                    "N1Z_minus_D4P16": {
                        "mean_reward_delta": -0.1,
                        "std_reward_delta": 0.0,
                        "positive_tile_seed_count": 0,
                        "total_tile_seed_count": 1,
                        "positive_fraction": 0.0,
                    }
                },
            },
            "delta_rows": [
                {
                    "variant_id": "N1Z",
                    "comparator_variant_id": "D4P16",
                    "eval_tile_id": "tile_a",
                    "seed": 0,
                    "variant_reward": 0.9,
                    "comparator_reward": 1.0,
                    "variant_minus_comparator_reward": -0.1,
                    "variant_improves_comparator": False,
                    "train_timesteps": kwargs["total_timesteps"],
                    "eval_max_steps": kwargs["eval_max_steps"],
                    "claim_boundary": "fixture",
                }
            ],
            "claim_boundary": "fixture",
        }
        return {
            "phase30_normalized_b1_status": "normalized_b1_recovers_b0_gap",
            "claim_boundary": "fixture",
            "summaries": [],
            "traces": {},
            "delta_rows": comparison["delta_rows"],
            "learned_policy": comparison["learned_policy"],
            "source_rows": [],
            "main_summary_rows": [],
            "variants": comparison["variants"],
            "eval_tile_ids": comparison["eval_tile_ids"],
            "seeds": comparison["seeds"],
            "train_timesteps": comparison["train_timesteps"],
            "eval_max_steps": comparison["eval_max_steps"],
        }

    def _fake_write_phase30_normalized_b1_artifacts(protocol, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = output_dir / "phase30_normalized_b1_comparison.json"
        comparison_payload = {
            "phase": "phase30_normalized_b1_analysis",
            "train_timesteps": protocol["train_timesteps"],
            "eval_max_steps": protocol["eval_max_steps"],
            "variants": protocol["variants"],
            "eval_tile_ids": protocol["eval_tile_ids"],
            "seeds": protocol["seeds"],
            "phase30_normalized_b1_status": protocol["phase30_normalized_b1_status"],
            "learned_policy": protocol["learned_policy"],
            "delta_rows": protocol["delta_rows"],
            "claim_boundary": protocol["claim_boundary"],
        }
        comparison_path.write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")
        for name in (
            "phase30_normalized_b1_summary.csv",
            "phase30_normalized_b1_traces.json",
            "phase30_normalized_b1_delta_table.csv",
            "phase30_normalized_b1_readiness.md",
        ):
            (output_dir / name).write_text("", encoding="utf-8")
        return {
            "summary_csv": output_dir / "phase30_normalized_b1_summary.csv",
            "traces_json": output_dir / "phase30_normalized_b1_traces.json",
            "comparison_json": comparison_path,
            "delta_csv": output_dir / "phase30_normalized_b1_delta_table.csv",
            "readiness_md": output_dir / "phase30_normalized_b1_readiness.md",
        }

    monkeypatch.setattr(module, "run_phase30_normalized_b1_ablation", _fake_run_phase30_normalized_b1_ablation)
    monkeypatch.setattr(module, "write_phase30_normalized_b1_artifacts", _fake_write_phase30_normalized_b1_artifacts)

    exit_code = module.main(
        [
            "--mode",
            "run-and-analyze",
            "--baseline-phase30-comparison-json",
            str(baseline),
            "--baseline-control-summary-csv",
            str(baseline_control_summary),
            "--phase2-output-dir",
            str(tmp_path / "phase2"),
            "--phase8-output-dir",
            str(tmp_path / "phase8"),
            "--tile-index-csv",
            str(tmp_path / "tile_index.csv"),
            "--variants",
            "B1,N1Z,N1ZR,D4P8,D4P16",
            "--total-timesteps",
            "8192",
            "--eval-max-steps",
            "8",
            "--seeds",
            "0",
            "--max-eval-tiles",
            "3",
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    assert exit_code == 0
    assert captured["total_timesteps"] == 8192
    assert captured["variants"] == ("B1", "N1Z", "N1ZR", "D4P8", "D4P16")
    assert captured["existing_control_summary_csv"] == baseline_control_summary
    assert (tmp_path / "outputs" / "phase33_matched_baseline_comparison.json").exists()
    assert (tmp_path / "outputs" / "phase33_budget_robustness.json").exists()
