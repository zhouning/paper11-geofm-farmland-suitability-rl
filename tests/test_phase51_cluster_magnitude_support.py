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


def _phase51_cluster_rows():
    values = [0.62, 0.16, 0.35, -0.10, 0.08, -0.15, 1.49, 1.42, 0.31]
    return [_cluster_row(f"tile_{idx // 3}", idx % 3, value) for idx, value in enumerate(values)]


def _write_cluster_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_cluster_row("tile", 0, 0.1).keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_phase51_reports_exact_signed_rank_support():
    from paper11_geofm.phase51_cluster_magnitude_support import (
        PHASE51_CLAIM_BOUNDARY,
        build_phase51_cluster_magnitude_support,
    )

    analysis = build_phase51_cluster_magnitude_support(_phase51_cluster_rows())

    assert analysis["phase"] == "phase51_cluster_magnitude_support"
    assert analysis["phase51_magnitude_status"] == "cluster_magnitude_support"
    assert analysis["claim_boundary"] == PHASE51_CLAIM_BOUNDARY
    assert analysis["cluster_count"] == 9
    assert analysis["positive_rank_sum"] == 40
    assert analysis["total_rank_sum"] == 45
    assert analysis["one_sided_signed_rank_p"] == 0.01953125


def test_phase51_writer_outputs_json_and_markdown(tmp_path):
    from paper11_geofm.phase51_cluster_magnitude_support import (
        build_phase51_cluster_magnitude_support,
        write_phase51_cluster_magnitude_support_artifacts,
    )

    analysis = build_phase51_cluster_magnitude_support(_phase51_cluster_rows())
    paths = write_phase51_cluster_magnitude_support_artifacts(
        analysis,
        tmp_path / "outputs",
    )

    assert paths["comparison_json"].name == "phase51_cluster_magnitude_support.json"
    assert paths["rank_csv"].name == "phase51_cluster_signed_rank.csv"
    assert paths["readiness_md"].name == "phase51_cluster_magnitude_support.md"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase51_magnitude_status"] == "cluster_magnitude_support"
    assert "signed-rank" in paths["readiness_md"].read_text(encoding="utf-8")


def test_phase51_cli_analyze_only_writes_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase51_cluster_magnitude_support"
        / "run_phase51_cluster_magnitude_support.py"
    )
    spec = importlib.util.spec_from_file_location("phase51_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    cluster_csv = _write_cluster_csv(tmp_path / "cluster.csv", _phase51_cluster_rows())
    exit_code = module.main(
        [
            "--phase50-cluster-csv",
            str(cluster_csv),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Phase 51 magnitude status: cluster_magnitude_support" in stdout
    assert "phase51_cluster_magnitude_support.json" in stdout
