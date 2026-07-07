import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _delta_row(
    compressed_variant_id,
    comparator_variant_id,
    delta,
    tile_id="tile_a",
    seed=0,
):
    return {
        "compressed_variant_id": compressed_variant_id,
        "comparator_variant_id": comparator_variant_id,
        "eval_tile_id": tile_id,
        "seed": seed,
        "compressed_reward": 1.0 + delta,
        "comparator_reward": 1.0,
        "compressed_minus_comparator_reward": delta,
        "compressed_improves_comparator": delta > 0.0,
        "train_timesteps": 4096,
        "eval_max_steps": 8,
        "claim_boundary": "fixture",
    }


def _phase49_delta_rows(case="robust"):
    values_by_case = {
        "robust": {
            ("D4P8", "B0"): [0.1, 0.2, 0.3, 0.4],
            ("D4P8", "B1"): [0.2, 0.2, 0.3, 0.4],
            ("D4P8", "D2"): [0.3, 0.2, 0.3, 0.4],
            ("D4P8", "D3"): [0.2, 0.2, 0.3, 0.4],
            ("D4P16", "B0"): [0.2, 0.3, 0.4, 0.5],
            ("D4P16", "B1"): [0.3, 0.3, 0.4, 0.5],
            ("D4P16", "D2"): [0.4, 0.3, 0.4, 0.5],
            ("D4P16", "D3"): [0.3, 0.3, 0.4, 0.5],
        },
        "fragile": {
            ("D4P8", "B0"): [0.1, 0.1, -0.6, -0.6],
            ("D4P8", "B1"): [0.2, 0.1, -0.5, -0.5],
            ("D4P8", "D2"): [0.3, 0.2, -0.4, -0.4],
            ("D4P8", "D3"): [0.2, 0.2, -0.5, -0.5],
            ("D4P16", "B0"): [0.2, 0.2, -0.5, -0.5],
            ("D4P16", "B1"): [0.3, 0.2, -0.4, -0.4],
            ("D4P16", "D2"): [0.4, 0.3, -0.3, -0.3],
            ("D4P16", "D3"): [0.3, 0.3, -0.4, -0.4],
        },
    }
    pairs = [("tile_a", 0), ("tile_a", 1), ("tile_b", 0), ("tile_b", 1)]
    rows = []
    for (compressed, comparator), deltas in values_by_case[case].items():
        for (tile_id, seed), delta in zip(pairs, deltas):
            rows.append(_delta_row(compressed, comparator, delta, tile_id, seed))
    return rows


def _write_delta_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(_delta_row("D4P8", "B0", 0.1).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_phase49_reports_robust_compressed_route():
    from paper11_geofm.phase49_compressed_route_robustness import (
        PHASE49_CLAIM_BOUNDARY,
        build_phase49_compressed_route_robustness,
    )

    analysis = build_phase49_compressed_route_robustness(
        _phase49_delta_rows("robust"),
        bootstrap_iterations=200,
        random_seed=7,
    )

    assert analysis["phase"] == "phase49_compressed_route_robustness"
    assert analysis["phase49_robustness_status"] == "compressed_route_statistically_robust"
    assert analysis["claim_boundary"] == PHASE49_CLAIM_BOUNDARY
    assert analysis["pooled_delta"]["positive_count"] == 32
    assert analysis["pooled_delta"]["total_count"] == 32
    assert analysis["pooled_delta"]["one_sided_sign_test_p"] < 0.001
    assert analysis["pooled_delta"]["bootstrap_ci95_low"] > 0.0
    assert analysis["leave_one_tile_summary"]["min_mean_delta"] > 0.0
    assert analysis["leave_one_seed_summary"]["min_mean_delta"] > 0.0
    assert analysis["per_comparison"]["D4P16_minus_D3"]["mean_delta"] == 0.375


def test_phase49_marks_route_fragile_when_leave_one_groups_turn_negative():
    from paper11_geofm.phase49_compressed_route_robustness import (
        build_phase49_compressed_route_robustness,
    )

    analysis = build_phase49_compressed_route_robustness(
        _phase49_delta_rows("fragile"),
        bootstrap_iterations=200,
        random_seed=7,
    )

    assert analysis["phase49_robustness_status"] == "compressed_route_fragile"
    assert analysis["leave_one_tile_summary"]["min_mean_delta"] < 0.0


def test_phase49_writer_outputs_json_csv_and_markdown(tmp_path):
    from paper11_geofm.phase49_compressed_route_robustness import (
        build_phase49_compressed_route_robustness,
        write_phase49_compressed_route_robustness_artifacts,
    )

    analysis = build_phase49_compressed_route_robustness(
        _phase49_delta_rows("robust"),
        bootstrap_iterations=200,
        random_seed=7,
    )
    paths = write_phase49_compressed_route_robustness_artifacts(
        analysis,
        tmp_path / "outputs",
    )

    assert paths["comparison_json"].name == "phase49_compressed_route_robustness.json"
    assert paths["per_comparison_csv"].name == "phase49_per_comparison_robustness.csv"
    assert paths["leave_one_csv"].name == "phase49_leave_one_sensitivity.csv"
    assert paths["readiness_md"].name == "phase49_compressed_route_robustness.md"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase49_robustness_status"] == "compressed_route_statistically_robust"
    markdown = paths["readiness_md"].read_text(encoding="utf-8")
    assert "Compressed route robustness" in markdown
    assert "does not enable suitability reward" in markdown


def test_phase49_cli_analyze_only_writes_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase49_compressed_route_robustness"
        / "run_phase49_compressed_route_robustness.py"
    )
    spec = importlib.util.spec_from_file_location("phase49_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    delta_csv = _write_delta_csv(tmp_path / "delta.csv", _phase49_delta_rows("robust"))
    exit_code = module.main(
        [
            "--phase48-delta-csv",
            str(delta_csv),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--bootstrap-iterations",
            "200",
            "--random-seed",
            "7",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "Phase 49 robustness status: compressed_route_statistically_robust"
        in stdout
    )
    assert "phase49_compressed_route_robustness.json" in stdout
