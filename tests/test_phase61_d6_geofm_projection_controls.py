import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _row(block_id, explicit_00, embedding_values=None, pca_values=None):
    row = {"block_id": block_id, "explicit_feature_00": explicit_00}
    if embedding_values is not None:
        for index, value in enumerate(embedding_values):
            row[f"embedding_mean_{index:02d}"] = value
    if pca_values is not None:
        for index, value in enumerate(pca_values):
            row[f"embedding_pca_{index:02d}"] = value
    return row


def _b0_rows():
    return [
        _row("b1", 1.0),
        _row("b2", 2.0),
        _row("b3", 3.0),
        _row("b4", 4.0),
        _row("b5", 5.0),
    ]


def _b1_rows():
    return [
        _row("b1", 1.0, [1.0, 0.0, 0.0]),
        _row("b2", 2.0, [0.0, 1.0, 0.0]),
        _row("b3", 3.0, [0.0, 0.0, 1.0]),
        _row("b4", 4.0, [1.0, 1.0, 0.0]),
        _row("b5", 5.0, [0.0, 1.0, 1.0]),
    ]


def _d4p2_rows():
    return [
        _row("b1", 1.0, pca_values=[0.0, 0.4]),
        _row("b2", 2.0, pca_values=[0.8, 0.1]),
        _row("b3", 3.0, pca_values=[-0.4, -0.3]),
        _row("b4", 4.0, pca_values=[0.5, 0.5]),
        _row("b5", 5.0, pca_values=[-0.9, -0.7]),
    ]


def _d4p3_rows():
    return [
        _row("b1", 1.0, pca_values=[0.0, 0.4, 0.2]),
        _row("b2", 2.0, pca_values=[0.8, 0.1, -0.1]),
        _row("b3", 3.0, pca_values=[-0.4, -0.3, 0.3]),
        _row("b4", 4.0, pca_values=[0.5, 0.5, -0.2]),
        _row("b5", 5.0, pca_values=[-0.9, -0.7, -0.2]),
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_phase61_builds_deterministic_d6_projection_controls():
    from paper11_geofm.phase61_d6_geofm_projection_controls import (
        build_phase61_d6_projection_controls,
    )

    protocol = build_phase61_d6_projection_controls(
        b0_rows_or_csv=_b0_rows(),
        b1_rows_or_csv=_b1_rows(),
        d4p8_rows_or_csv=_d4p2_rows(),
        d4p16_rows_or_csv=_d4p3_rows(),
        dimensions=(2, 3),
        seed=61,
    )

    assert protocol["phase"] == "phase61_d6_projection_control_features"
    assert protocol["phase61_d6_projection_status"] == "d6_projection_controls_ready_for_training"
    assert protocol["variant_ids"] == ["D6R2", "D6P2", "D6R3", "D6P3"]
    assert set(protocol["variant_tables"]) == {"D6R2", "D6P2", "D6R3", "D6P3"}
    assert protocol["summary"]["D6R2"]["projection_type"] == "random_orthonormal_raw_b1_projection"
    assert protocol["summary"]["D6P3"]["projection_type"] == "pca_raw_b1_projection"
    assert protocol["geometry_rows"][0]["row_count"] == 5
    assert all(
        row["explicit_feature_00"] == float(index + 1)
        for index, row in enumerate(protocol["variant_tables"]["D6P2"])
    )
    assert "projection_01" in protocol["variant_tables"]["D6R2"][0]
    assert "projection_02" in protocol["variant_tables"]["D6P3"][0]


def test_phase61_rejects_misaligned_block_ids():
    from paper11_geofm.phase61_d6_geofm_projection_controls import (
        build_phase61_d6_projection_controls,
    )

    b1_rows = _b1_rows()
    b1_rows[1] = {**b1_rows[1], "block_id": "different"}

    try:
        build_phase61_d6_projection_controls(
            _b0_rows(), b1_rows, _d4p2_rows(), _d4p3_rows(), dimensions=(2, 3)
        )
    except ValueError as exc:
        assert "aligned block IDs" in str(exc)
    else:
        raise AssertionError("expected row-alignment failure")


def test_phase61_status_blocks_zero_variance_projection():
    from paper11_geofm.phase61_d6_geofm_projection_controls import (
        build_phase61_d6_projection_controls,
    )

    b1_rows = [
        _row(row["block_id"], row["explicit_feature_00"], [1.0, 1.0, 1.0])
        for row in _b0_rows()
    ]
    analysis = build_phase61_d6_projection_controls(
        _b0_rows(), b1_rows, _d4p2_rows(), _d4p3_rows(), dimensions=(2, 3)
    )

    assert analysis["phase61_d6_projection_status"] == "d6_projection_controls_blocked"