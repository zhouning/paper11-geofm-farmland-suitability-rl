import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _cluster_values():
    return [
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


def _delta_rows():
    rows = []
    comparisons = [
        ("D4P8", "B0"),
        ("D4P8", "B1"),
        ("D4P8", "D2"),
        ("D4P8", "D3"),
        ("D4P16", "B0"),
        ("D4P16", "B1"),
        ("D4P16", "D2"),
        ("D4P16", "D3"),
    ]
    for tile_id, seed, mean_delta in _cluster_values():
        for compressed_variant, comparator in comparisons:
            rows.append(
                {
                    "compressed_variant_id": compressed_variant,
                    "comparator_variant_id": comparator,
                    "eval_tile_id": tile_id,
                    "seed": seed,
                    "compressed_reward": mean_delta,
                    "comparator_reward": 0.0,
                    "compressed_minus_comparator_reward": mean_delta,
                    "compressed_improves_comparator": mean_delta > 0.0,
                    "train_timesteps": 4096,
                    "eval_max_steps": 8,
                    "claim_boundary": "fixture",
                }
            )
    return rows


def _cluster_rows():
    return [
        {
            "eval_tile_id": tile_id,
            "seed": seed,
            "cluster_delta_count": 8,
            "mean_cluster_delta": mean_delta,
            "cluster_positive": mean_delta > 0.0,
            "claim_boundary": "fixture",
        }
        for tile_id, seed, mean_delta in _cluster_values()
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _fixture_paths(tmp_path: Path, mutate_cluster: bool = False) -> dict[str, Path]:
    from paper11_geofm.phase51_cluster_magnitude_support import (
        build_phase51_cluster_magnitude_support,
    )
    from paper11_geofm.phase53_cluster_mean_support import (
        build_phase53_cluster_mean_support,
    )

    cluster_rows = _cluster_rows()
    if mutate_cluster:
        cluster_rows[0] = dict(cluster_rows[0])
        cluster_rows[0]["mean_cluster_delta"] = 0.999999

    phase51 = build_phase51_cluster_magnitude_support(_cluster_rows())
    phase53 = build_phase53_cluster_mean_support(
        _cluster_rows(),
        bootstrap_iterations=2000,
        random_seed=53,
    )
    return {
        "delta_csv": _write_csv(tmp_path / "phase48_delta.csv", _delta_rows()),
        "cluster_csv": _write_csv(tmp_path / "phase50_cluster.csv", cluster_rows),
        "phase51_json": _write_json(tmp_path / "phase51.json", phase51),
        "phase53_json": _write_json(tmp_path / "phase53.json", phase53),
    }


def test_phase54_reports_artifact_lineage_consistent(tmp_path):
    from paper11_geofm.phase54_artifact_lineage_consistency import (
        PHASE54_CLAIM_BOUNDARY,
        build_phase54_artifact_lineage_consistency,
    )

    paths = _fixture_paths(tmp_path)
    analysis = build_phase54_artifact_lineage_consistency(**paths)

    assert analysis["phase"] == "phase54_artifact_lineage_consistency"
    assert analysis["phase54_lineage_status"] == "artifact_lineage_consistent"
    assert analysis["claim_boundary"] == PHASE54_CLAIM_BOUNDARY
    assert analysis["all_checks_passed"] is True
    assert analysis["recomputed_cluster_count"] == 15
    assert analysis["recomputed_mean_cluster_delta"] == 0.2921767818
    assert analysis["recomputed_phase51_signed_rank_p"] == 0.0206298828
    assert analysis["recomputed_phase53_sign_flip_p"] == 0.0196838379
    assert all(row["passed"] is True for row in analysis["check_rows"])


def test_phase54_reports_inconsistent_when_authoritative_cluster_differs(tmp_path):
    from paper11_geofm.phase54_artifact_lineage_consistency import (
        build_phase54_artifact_lineage_consistency,
    )

    paths = _fixture_paths(tmp_path, mutate_cluster=True)
    analysis = build_phase54_artifact_lineage_consistency(**paths)

    assert analysis["phase54_lineage_status"] == "artifact_lineage_inconsistent"
    assert analysis["all_checks_passed"] is False
    failed_checks = {row["check_name"] for row in analysis["check_rows"] if not row["passed"]}
    assert "phase50_cluster_rows_match_delta_recompute" in failed_checks


def test_phase54_cli_writes_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase54_artifact_lineage_consistency"
        / "run_phase54_artifact_lineage_consistency.py"
    )
    spec = importlib.util.spec_from_file_location("phase54_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    paths = _fixture_paths(tmp_path / "inputs")
    exit_code = module.main(
        [
            "--phase48-delta-csv",
            str(paths["delta_csv"]),
            "--phase50-cluster-csv",
            str(paths["cluster_csv"]),
            "--phase51-json",
            str(paths["phase51_json"]),
            "--phase53-json",
            str(paths["phase53_json"]),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Phase 54 artifact lineage status: artifact_lineage_consistent" in stdout
    assert (tmp_path / "outputs" / "phase54_artifact_lineage_consistency.json").exists()
    assert (tmp_path / "outputs" / "phase54_artifact_lineage_checks.csv").exists()
    assert "authoritative artifact chain" in (
        tmp_path / "outputs" / "phase54_artifact_lineage_consistency.md"
    ).read_text(encoding="utf-8")
