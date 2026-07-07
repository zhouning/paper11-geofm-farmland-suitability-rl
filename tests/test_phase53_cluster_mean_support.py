import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _cluster_row(tile_id, seed, mean_delta):
    return {
        "eval_tile_id": tile_id,
        "seed": seed,
        "cluster_delta_count": 8,
        "mean_cluster_delta": mean_delta,
        "cluster_positive": mean_delta > 0.0,
        "claim_boundary": "fixture",
    }


def _phase53_cluster_rows():
    values = [
        ("tile_r000_c004", 0, 0.1221986101),
        ("tile_r000_c004", 1, 0.3112516382),
        ("tile_r000_c004", 2, -0.2237474805),
        ("tile_r001_c004", 0, -0.0779840767),
        ("tile_r001_c004", 1, 0.3081394454),
        ("tile_r001_c004", 2, -0.2629167583),
        ("tile_r002_c003", 0, 0.6277219618),
        ("tile_r002_c003", 1, 0.1675840041),
        ("tile_r002_c003", 2, 0.3543095809),
        ("tile_r005_c003", 0, 1.4985375230),
        ("tile_r005_c003", 1, 1.4271455708),
        ("tile_r005_c003", 2, 0.3118752597),
        ("tile_r005_c004", 0, -0.1069505759),
        ("tile_r005_c004", 1, 0.0847331180),
        ("tile_r005_c004", 2, -0.1592460929),
    ]
    return [_cluster_row(*item) for item in values]


def _write_cluster_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_cluster_row("tile", 0, 0.1).keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_phase53_reports_cluster_mean_support():
    from paper11_geofm.phase53_cluster_mean_support import (
        PHASE53_CLAIM_BOUNDARY,
        build_phase53_cluster_mean_support,
    )

    analysis = build_phase53_cluster_mean_support(
        _phase53_cluster_rows(),
        bootstrap_iterations=2000,
        random_seed=53,
    )

    assert analysis["phase"] == "phase53_cluster_mean_support"
    assert analysis["phase53_cluster_mean_status"] == "cluster_mean_support"
    assert analysis["claim_boundary"] == PHASE53_CLAIM_BOUNDARY
    assert analysis["cluster_count"] == 15
    assert analysis["mean_cluster_delta"] == 0.2921767818
    assert analysis["exact_sign_flip_mean_p"] == 0.0196838379
    assert analysis["bootstrap_ci95_low"] > 0.0
    assert analysis["bootstrap_ci95_high"] > analysis["mean_cluster_delta"]
    assert analysis["influence_summary"]["min_leave_one_cluster_mean"] == 0.2060081575
    assert analysis["influence_summary"]["min_leave_one_tile_mean"] == 0.0954244478
    assert analysis["influence_summary"]["min_leave_one_seed_mean"] == 0.2083797951
    assert analysis["influence_summary"]["all_leave_one_means_positive"] is True


def test_phase53_writer_outputs_json_csv_and_markdown(tmp_path):
    from paper11_geofm.phase53_cluster_mean_support import (
        build_phase53_cluster_mean_support,
        write_phase53_cluster_mean_support_artifacts,
    )

    analysis = build_phase53_cluster_mean_support(
        _phase53_cluster_rows(),
        bootstrap_iterations=200,
        random_seed=53,
    )
    paths = write_phase53_cluster_mean_support_artifacts(
        analysis,
        tmp_path / "outputs",
    )

    assert paths["comparison_json"].name == "phase53_cluster_mean_support.json"
    assert paths["leave_one_csv"].name == "phase53_leave_one_influence.csv"
    assert paths["readiness_md"].name == "phase53_cluster_mean_support.md"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase53_cluster_mean_status"] == "cluster_mean_support"
    assert "sign-flip" in paths["readiness_md"].read_text(encoding="utf-8")


def test_phase53_cli_analyze_only_writes_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase53_cluster_mean_support"
        / "run_phase53_cluster_mean_support.py"
    )
    spec = importlib.util.spec_from_file_location("phase53_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    cluster_csv = _write_cluster_csv(tmp_path / "cluster.csv", _phase53_cluster_rows())
    exit_code = module.main(
        [
            "--phase50-cluster-csv",
            str(cluster_csv),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--bootstrap-iterations",
            "200",
            "--random-seed",
            "53",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Phase 53 cluster mean status: cluster_mean_support" in stdout
    assert "phase53_cluster_mean_support.json" in stdout
