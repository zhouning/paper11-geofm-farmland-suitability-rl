import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_phase26_comparison(
    path: Path,
    *,
    timesteps: int,
    deltas: dict[tuple[str, int], float],
    claim_status: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    positive_count = sum(1 for value in deltas.values() if value > 0)
    payload = {
        "phase": "phase26_main_empirical_experiment",
        "train_timesteps": timesteps,
        "eval_max_steps": 8,
        "eval_tile_ids": sorted({tile_id for tile_id, _seed in deltas}),
        "seeds": sorted({seed for _tile_id, seed in deltas}),
        "phase26_claim_status": claim_status,
        "learned_policy": {
            "B1_minus_B0_mean_reward": round(sum(deltas.values()) / len(deltas), 10),
            "positive_tile_seed_count": positive_count,
            "total_tile_seed_count": len(deltas),
            "positive_fraction": round(positive_count / len(deltas), 10),
        },
        "tile_seed_delta_rows": [
            {
                "eval_tile_id": tile_id,
                "seed": seed,
                "b0_reward": 0.0,
                "b1_reward": value,
                "b1_minus_b0_reward": value,
                "b1_improves_b0": value > 0,
                "train_timesteps": timesteps,
                "eval_max_steps": 8,
            }
            for (tile_id, seed), value in sorted(deltas.items())
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_phase27_builds_budget_transition_and_stability_classes(tmp_path):
    from paper11_geofm.phase27_stability_diagnosis import (
        build_phase27_stability_diagnosis,
    )

    lower = _write_phase26_comparison(
        tmp_path / "lower" / "phase26_main_comparison.json",
        timesteps=1024,
        claim_status="not_supported",
        deltas={
            ("tile_a", 0): 0.2,
            ("tile_a", 1): -1.0,
            ("tile_b", 0): -0.3,
            ("tile_b", 1): 0.4,
        },
    )
    higher = _write_phase26_comparison(
        tmp_path / "higher" / "phase26_main_comparison.json",
        timesteps=4096,
        claim_status="not_supported",
        deltas={
            ("tile_a", 0): 0.1,
            ("tile_a", 1): -0.1,
            ("tile_b", 0): 0.2,
            ("tile_b", 1): -0.5,
        },
    )

    analysis = build_phase27_stability_diagnosis([lower, higher])

    assert analysis["phase"] == "phase27_b0_b1_stability_diagnosis"
    assert analysis["phase27_diagnostic_status"] == "budget_not_explanatory"
    assert analysis["budget_transition_rows"][1]["mean_delta_change_from_previous"] == 0.1
    assert analysis["budget_transition_rows"][1]["positive_count_change_from_previous"] == 0
    assert analysis["stability_counts"] == {
        "stable_positive": 1,
        "stable_negative": 1,
        "flip_to_positive": 1,
        "flip_to_negative": 1,
        "incomplete": 0,
    }
    classes = {
        (row["eval_tile_id"], row["seed"]): row["stability_class"]
        for row in analysis["tile_seed_stability_rows"]
    }
    assert classes == {
        ("tile_a", 0): "stable_positive",
        ("tile_a", 1): "stable_negative",
        ("tile_b", 0): "flip_to_positive",
        ("tile_b", 1): "flip_to_negative",
    }


def test_phase27_reports_insufficient_for_unpaired_tile_seed_rows(tmp_path):
    from paper11_geofm.phase27_stability_diagnosis import (
        build_phase27_stability_diagnosis,
    )

    lower = _write_phase26_comparison(
        tmp_path / "lower" / "phase26_main_comparison.json",
        timesteps=1024,
        claim_status="not_supported",
        deltas={("tile_a", 0): -0.2, ("tile_b", 0): 0.1},
    )
    higher = _write_phase26_comparison(
        tmp_path / "higher" / "phase26_main_comparison.json",
        timesteps=4096,
        claim_status="not_supported",
        deltas={("tile_a", 0): -0.1},
    )

    analysis = build_phase27_stability_diagnosis([lower, higher])

    assert analysis["phase27_diagnostic_status"] == "insufficient"
    assert analysis["stability_counts"]["incomplete"] == 1
    assert analysis["tile_seed_stability_rows"][1]["stability_class"] == "incomplete"


def test_phase27_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase27_stability_diagnosis import (
        build_phase27_stability_diagnosis,
        write_phase27_stability_diagnosis_artifacts,
    )

    lower = _write_phase26_comparison(
        tmp_path / "lower" / "phase26_main_comparison.json",
        timesteps=1024,
        claim_status="not_supported",
        deltas={("tile_a", 0): -0.2, ("tile_b", 0): 0.1},
    )
    higher = _write_phase26_comparison(
        tmp_path / "higher" / "phase26_main_comparison.json",
        timesteps=4096,
        claim_status="not_supported",
        deltas={("tile_a", 0): -0.1, ("tile_b", 0): -0.3},
    )
    analysis = build_phase27_stability_diagnosis([lower, higher])
    paths = write_phase27_stability_diagnosis_artifacts(analysis, tmp_path / "outputs")

    assert paths["budget_transition_csv"].name == "phase27_budget_transition_table.csv"
    assert paths["tile_seed_stability_csv"].name == "phase27_tile_seed_stability.csv"
    assert paths["diagnostic_summary_json"].name == "phase27_diagnostic_summary.json"
    assert paths["diagnostic_readiness_md"].name == "phase27_diagnostic_readiness.md"
    transition_rows = list(
        csv.DictReader(paths["budget_transition_csv"].open("r", encoding="utf-8"))
    )
    assert transition_rows[0]["train_timesteps"] == "1024"
    summary = json.loads(paths["diagnostic_summary_json"].read_text(encoding="utf-8"))
    assert summary["phase27_diagnostic_status"] == "budget_not_explanatory"
    markdown = paths["diagnostic_readiness_md"].read_text(encoding="utf-8")
    assert "budget_not_explanatory" in markdown
    assert "GeoFM improves planning decisions" not in markdown


def test_phase27_cli_writes_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase27_stability_diagnosis"
        / "run_phase27_stability_diagnosis.py"
    )
    spec = importlib.util.spec_from_file_location("phase27_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    lower = _write_phase26_comparison(
        tmp_path / "lower" / "phase26_main_comparison.json",
        timesteps=1024,
        claim_status="not_supported",
        deltas={("tile_a", 0): -0.2, ("tile_b", 0): 0.1},
    )
    higher = _write_phase26_comparison(
        tmp_path / "higher" / "phase26_main_comparison.json",
        timesteps=4096,
        claim_status="not_supported",
        deltas={("tile_a", 0): -0.1, ("tile_b", 0): -0.3},
    )

    exit_code = module.main(
        [
            "--phase26-comparison-json",
            str(lower),
            "--phase26-comparison-json",
            str(higher),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Phase 27 diagnostic status: budget_not_explanatory" in stdout
    assert "phase27_diagnostic_summary.json" in stdout


def test_phase27_cli_reports_missing_input(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase27_stability_diagnosis"
        / "run_phase27_stability_diagnosis.py"
    )
    spec = importlib.util.spec_from_file_location("phase27_runner_error", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--phase26-comparison-json",
            str(tmp_path / "missing.json"),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert "Error:" in stderr
