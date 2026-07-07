import csv
import importlib.util
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _feature_rows(prefix: str, values: list[list[float]]) -> list[dict[str, object]]:
    rows = []
    for index, row_values in enumerate(values, start=1):
        row = {"block_id": f"b{index}", "explicit_feature_00": 1.0}
        for dim, value in enumerate(row_values):
            row[f"{prefix}_{dim:02d}"] = value
        rows.append(row)
    return rows


def _fixture_feature_rows():
    raw = _feature_rows(
        "embedding_mean",
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
    )
    d4p8 = _feature_rows(
        "embedding_pca",
        [[1.0], [-1.0], [0.0], [0.0], [0.0], [0.0]],
    )
    d4p16 = _feature_rows(
        "embedding_pca",
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 0.0], [0.0, 0.0]],
    )
    return raw, d4p8, d4p16


def _delta_rows():
    rows = []
    for compressed in ("D4P8", "D4P16"):
        for tile in ("tile_a", "tile_b"):
            for seed in (0, 1):
                rows.append(
                    {
                        "compressed_variant_id": compressed,
                        "comparator_variant_id": "B1",
                        "eval_tile_id": tile,
                        "seed": seed,
                        "compressed_minus_comparator_reward": 0.2 if compressed == "D4P8" else 0.4,
                    }
                )
    return rows


def _tile_rows():
    return [
        {"tile_id": "tile_a", "block_ids": "b1;b2;b3"},
        {"tile_id": "tile_b", "block_ids": "b4;b5;b6"},
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_phase57_reports_compressed_geometry_and_status():
    from paper11_geofm.phase57_compressed_representation_mechanism import (
        PHASE57_CLAIM_BOUNDARY,
        build_phase57_compressed_representation_mechanism,
    )

    raw, d4p8, d4p16 = _fixture_feature_rows()
    analysis = build_phase57_compressed_representation_mechanism(
        raw,
        d4p8,
        d4p16,
        delta_rows_or_csv=_delta_rows(),
        tile_rows_or_csv=_tile_rows(),
    )

    assert analysis["phase"] == "phase57_compressed_representation_mechanism"
    assert analysis["phase57_mechanism_status"] == "compressed_geometry_consistent"
    assert analysis["claim_boundary"] == PHASE57_CLAIM_BOUNDARY
    assert analysis["row_alignment"]["common_block_count"] == 6
    geometry = {row["variant_id"]: row for row in analysis["geometry_rows"]}
    assert geometry["B1"]["feature_count"] == 3
    assert geometry["D4P8"]["feature_count"] == 1
    assert geometry["D4P16"]["feature_count"] == 2
    assert math.isclose(geometry["D4P8"]["raw_variance_retention"], 1.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(geometry["D4P16"]["raw_variance_retention"], 2.0 / 3.0, rel_tol=1e-9)
    assert geometry["B1"]["effective_rank"] > geometry["D4P16"]["effective_rank"]
    assert geometry["D4P16"]["effective_rank"] > geometry["D4P8"]["effective_rank"]
    gains = {row["compressed_variant_id"]: row for row in analysis["reward_gain_rows"]}
    assert gains["D4P8"]["mean_delta"] == 0.2
    assert gains["D4P16"]["mean_delta"] == 0.4
    assert len(analysis["tile_geometry_gain_rows"]) == 2


def test_phase57_writer_outputs_json_csv_and_markdown(tmp_path):
    from paper11_geofm.phase57_compressed_representation_mechanism import (
        build_phase57_compressed_representation_mechanism,
        write_phase57_compressed_representation_mechanism_artifacts,
    )

    raw, d4p8, d4p16 = _fixture_feature_rows()
    analysis = build_phase57_compressed_representation_mechanism(
        raw,
        d4p8,
        d4p16,
        delta_rows_or_csv=_delta_rows(),
        tile_rows_or_csv=_tile_rows(),
    )
    paths = write_phase57_compressed_representation_mechanism_artifacts(
        analysis,
        tmp_path / "outputs",
    )

    assert paths["comparison_json"].name == "phase57_compressed_representation_mechanism.json"
    assert paths["geometry_csv"].name == "phase57_representation_geometry.csv"
    assert paths["tile_geometry_gain_csv"].name == "phase57_tile_geometry_gain.csv"
    assert paths["readiness_md"].name == "phase57_compressed_representation_mechanism.md"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase57_mechanism_status"] == "compressed_geometry_consistent"
    assert "effective rank" in paths["readiness_md"].read_text(encoding="utf-8")


def test_phase57_cli_writes_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase57_compressed_representation_mechanism"
        / "run_phase57_compressed_representation_mechanism.py"
    )
    spec = importlib.util.spec_from_file_location("phase57_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    raw, d4p8, d4p16 = _fixture_feature_rows()
    exit_code = module.main(
        [
            "--b1-features-csv",
            str(_write_csv(tmp_path / "b1.csv", raw)),
            "--d4p8-features-csv",
            str(_write_csv(tmp_path / "d4p8.csv", d4p8)),
            "--d4p16-features-csv",
            str(_write_csv(tmp_path / "d4p16.csv", d4p16)),
            "--delta-csv",
            str(_write_csv(tmp_path / "delta.csv", _delta_rows())),
            "--tile-index-csv",
            str(_write_csv(tmp_path / "tiles.csv", _tile_rows())),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Phase 57 mechanism status: compressed_geometry_consistent" in stdout
    assert "phase57_compressed_representation_mechanism.json" in stdout