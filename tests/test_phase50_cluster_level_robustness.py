import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _delta_row(tile_id, seed, delta, compressed="D4P8", comparator="B0"):
    return {
        "compressed_variant_id": compressed,
        "comparator_variant_id": comparator,
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


def _cluster_rows(case="directional"):
    values = {
        "directional": {
            ("tile_a", 0): [0.5, 0.4],
            ("tile_a", 1): [0.2, 0.1],
            ("tile_b", 0): [0.3, -0.1],
            ("tile_b", 1): [-0.2, -0.1],
            ("tile_c", 0): [0.4, 0.2],
        },
        "strong": {
            ("tile_a", 0): [0.5, 0.4],
            ("tile_a", 1): [0.2, 0.1],
            ("tile_b", 0): [0.3, 0.1],
            ("tile_b", 1): [0.2, 0.1],
            ("tile_c", 0): [0.4, 0.2],
        },
    }[case]
    rows = []
    for (tile_id, seed), deltas in values.items():
        for idx, delta in enumerate(deltas):
            rows.append(
                _delta_row(
                    tile_id,
                    seed,
                    delta,
                    compressed="D4P8" if idx == 0 else "D4P16",
                    comparator="B0",
                )
            )
    return rows


def _write_delta_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_delta_row("tile", 0, 0.1).keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_phase50_reports_directional_cluster_support():
    from paper11_geofm.phase50_cluster_level_robustness import (
        PHASE50_CLAIM_BOUNDARY,
        build_phase50_cluster_level_robustness,
    )

    analysis = build_phase50_cluster_level_robustness(_cluster_rows("directional"))

    assert analysis["phase"] == "phase50_cluster_level_robustness"
    assert analysis["phase50_cluster_status"] == "cluster_directional_support"
    assert analysis["claim_boundary"] == PHASE50_CLAIM_BOUNDARY
    assert analysis["cluster_summary"]["cluster_count"] == 5
    assert analysis["cluster_summary"]["positive_cluster_count"] == 4
    assert analysis["cluster_summary"]["one_sided_sign_test_p"] == 0.1875
    assert analysis["cluster_summary"]["mean_cluster_delta"] == 0.17
    assert len(analysis["cluster_rows"]) == 5


def test_phase50_reports_cluster_statistical_support_when_all_clusters_positive():
    from paper11_geofm.phase50_cluster_level_robustness import (
        build_phase50_cluster_level_robustness,
    )

    analysis = build_phase50_cluster_level_robustness(_cluster_rows("strong"))

    assert analysis["phase50_cluster_status"] == "cluster_statistical_support"
    assert analysis["cluster_summary"]["positive_cluster_count"] == 5


def test_phase50_writer_outputs_json_csv_and_markdown(tmp_path):
    from paper11_geofm.phase50_cluster_level_robustness import (
        build_phase50_cluster_level_robustness,
        write_phase50_cluster_level_robustness_artifacts,
    )

    analysis = build_phase50_cluster_level_robustness(_cluster_rows("directional"))
    paths = write_phase50_cluster_level_robustness_artifacts(
        analysis,
        tmp_path / "outputs",
    )

    assert paths["comparison_json"].name == "phase50_cluster_level_robustness.json"
    assert paths["cluster_csv"].name == "phase50_cluster_delta_summary.csv"
    assert paths["readiness_md"].name == "phase50_cluster_level_robustness.md"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase50_cluster_status"] == "cluster_directional_support"
    assert "tile-seed cluster" in paths["readiness_md"].read_text(encoding="utf-8")


def test_phase50_cli_analyze_only_writes_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase50_cluster_level_robustness"
        / "run_phase50_cluster_level_robustness.py"
    )
    spec = importlib.util.spec_from_file_location("phase50_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    delta_csv = _write_delta_csv(tmp_path / "delta.csv", _cluster_rows("directional"))
    exit_code = module.main(
        [
            "--phase48-delta-csv",
            str(delta_csv),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Phase 50 cluster status: cluster_directional_support" in stdout
    assert "phase50_cluster_level_robustness.json" in stdout
