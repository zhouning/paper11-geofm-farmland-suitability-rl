import csv
import importlib.util
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _row(block_id, explicit_00, values, prefix="embedding_pca"):
    row = {
        "block_id": block_id,
        "explicit_feature_00": explicit_00,
    }
    for index, value in enumerate(values):
        row[f"{prefix}_{index:02d}"] = value
    return row


def _b0_rows():
    return [
        {"block_id": "b1", "explicit_feature_00": 1.0},
        {"block_id": "b2", "explicit_feature_00": 2.0},
        {"block_id": "b3", "explicit_feature_00": 3.0},
        {"block_id": "b4", "explicit_feature_00": 4.0},
    ]


def _d4p8_rows():
    return [
        _row("b1", 1.0, [0.0, 1.0]),
        _row("b2", 2.0, [2.0, 3.0]),
        _row("b3", 3.0, [4.0, 5.0]),
        _row("b4", 4.0, [6.0, 7.0]),
    ]


def _d4p16_rows():
    return [
        _row("b1", 1.0, [0.0, 10.0, 20.0]),
        _row("b2", 2.0, [1.0, 11.0, 21.0]),
        _row("b3", 3.0, [2.0, 12.0, 22.0]),
        _row("b4", 4.0, [3.0, 13.0, 23.0]),
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_phase59_builds_deterministic_matched_control_tables():
    from paper11_geofm.phase59_matched_dimension_controls import (
        PHASE59_CLAIM_BOUNDARY,
        build_phase59_matched_dimension_control_tables,
    )

    protocol = build_phase59_matched_dimension_control_tables(
        _b0_rows(),
        _d4p8_rows(),
        _d4p16_rows(),
        seed=59,
    )

    assert protocol["phase"] == "phase59_matched_dimension_control_features"
    assert protocol["claim_boundary"] == PHASE59_CLAIM_BOUNDARY
    assert protocol["variant_ids"] == ["D5R8", "D5S8", "D5R16", "D5S16"]
    assert set(protocol["variant_tables"]) == {"D5R8", "D5S8", "D5R16", "D5S16"}
    assert protocol["summary"]["D5R8"]["control_dimension"] == 2
    assert protocol["summary"]["D5R16"]["control_dimension"] == 3
    assert protocol["summary"]["D5S8"]["source_variant_id"] == "D4P8"
    assert protocol["summary"]["D5S16"]["source_variant_id"] == "D4P16"

    d5s8_values = [
        row["matched_control_00"] for row in protocol["variant_tables"]["D5S8"]
    ]
    d4p8_values = [row["embedding_pca_00"] for row in _d4p8_rows()]
    assert sorted(d5s8_values) == sorted(d4p8_values)
    assert d5s8_values != d4p8_values

    d5r8_values = [
        row["matched_control_00"] for row in protocol["variant_tables"]["D5R8"]
    ]
    assert len(d5r8_values) == 4
    assert not all(math.isclose(value, d4p8_values[0]) for value in d5r8_values)


def test_phase59_writes_control_tables_and_manifest(tmp_path):
    from paper11_geofm.phase59_matched_dimension_controls import (
        build_phase59_matched_dimension_control_tables,
        write_phase59_matched_dimension_control_tables,
    )

    protocol = build_phase59_matched_dimension_control_tables(
        _b0_rows(),
        _d4p8_rows(),
        _d4p16_rows(),
        seed=59,
    )
    paths = write_phase59_matched_dimension_control_tables(
        protocol,
        tmp_path / "controls",
    )

    assert paths["manifest"].name == "experiment_variants.json"
    assert paths["summary"].name == "phase59_matched_dimension_control_feature_summary.json"
    assert set(paths["variant_tables"]) == {"D5R8", "D5S8", "D5R16", "D5S16"}
    saved = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert saved["variants"]["D5R8"]["feature_table"] == "variant_D5R8_features.csv"
    with paths["variant_tables"]["D5S16"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["block_id"] == "b1"
    assert "matched_control_02" in rows[0]
