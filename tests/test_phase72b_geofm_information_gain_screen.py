import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _protocol_payload() -> dict:
    return {
        "phase": "phase72b_geofm_information_gain_screen",
        "seed": 72,
        "terrain": {
            "source_id": "copernicus_dem_glo30",
            "collection": "COPERNICUS/DEM/GLO30",
            "band": "DEM",
            "scale_m": 500,
            "feature_names": [
                "elevation_mean",
                "elevation_std",
                "elevation_min",
                "elevation_max",
                "slope_mean",
                "slope_std",
                "slope_max",
                "local_relief",
            ],
        },
        "years": {
            "train": [2017, 2018, 2019, 2020, 2021],
            "validation": [2022],
            "test": [2023],
        },
        "controls": {
            "seeds": [72, 73, 74, 75, 76],
            "random_projection_dim": 320,
            "partition_local": True,
            "learned_transform_fit_scope": "training_rows_only",
            "reuse_phase8_d4_tables": False,
        },
        "spatial": {"block_size": 8, "folds": 5, "buffer_rings": 1},
        "bootstrap": {"iterations": 2000, "seed": 72},
        "models": {
            "logistic_c": [0.01, 0.1, 1.0, 10.0],
            "logistic_class_weight": ["none", "balanced"],
            "hgb_learning_rate": [0.03, 0.08],
            "hgb_max_leaf_nodes": [15, 31],
            "hgb_min_samples_leaf": [20, 50],
            "hgb_max_iter": 200,
            "hgb_l2_regularization": [0.0, 1.0],
        },
        "calibration": {
            "methods": ["none", "sigmoid", "isotonic"],
            "ece_bins": 10,
        },
        "budgets": [0.10, 0.20],
        "variants": [
            "explicit_static",
            "explicit_history",
            "geofm_current_only",
            "geofm_temporal_mean_only",
            "explicit_plus_geofm_current",
            "explicit_plus_geofm_temporal_full",
            "explicit_plus_temporal_order_shuffle",
            "explicit_plus_spatial_shuffle",
            "explicit_plus_random_projection",
        ],
        "gates": {
            "ap_vs_explicit": 0.015,
            "brier_vs_explicit": 0.005,
            "ece_vs_explicit": 0.010,
            "ap_vs_control": 0.005,
            "brier_vs_control": 0.002,
            "transfer_ap_gain": 0.005,
            "transfer_brier_gain": 0.002,
            "transfer_ap_harm": 0.005,
            "transfer_brier_harm": 0.002,
        },
    }


def _write_protocol(path: Path) -> Path:
    path.write_text(json.dumps(_protocol_payload()), encoding="utf-8")
    return path


def test_phase72b_protocol_loads_frozen_thresholds(tmp_path):
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol

    protocol = load_phase72b_protocol(
        _write_protocol(tmp_path / "protocol.json")
    )
    assert protocol.seed == 72
    assert protocol.terrain_features[-1] == "local_relief"
    assert protocol.train_years == (2017, 2018, 2019, 2020, 2021)
    assert protocol.gates["ap_vs_explicit"] == 0.015
    assert protocol.control_partition_local is True
    assert protocol.learned_transform_fit_scope == "training_rows_only"
    assert protocol.reuse_phase8_d4_tables is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda gates: gates.__setitem__("ap_vs_explicit", 0.0),
        lambda gates: gates.pop("brier_vs_control"),
        lambda gates: gates.__setitem__("undeclared_gate", 1.0),
    ],
    ids=("changed", "missing", "extra"),
)
def test_phase72b_protocol_rejects_mutated_frozen_gates(tmp_path, mutation):
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol

    payload = _protocol_payload()
    mutation(payload["gates"])
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="gate"):
        load_phase72b_protocol(path)


@pytest.mark.parametrize(
    ("mutation", "section"),
    [
        (lambda payload: payload.__setitem__("seed", 73), "seed"),
        (
            lambda payload: payload["years"].update(
                {
                    "train": [2017, 2018, 2019, 2020, 2022],
                    "validation": [2021],
                }
            ),
            "years",
        ),
        (lambda payload: payload.pop("years"), "top-level"),
        (
            lambda payload: payload["controls"].__setitem__(
                "random_projection_dim", 64
            ),
            "controls",
        ),
        (
            lambda payload: payload["terrain"].__setitem__(
                "undeclared_field", "changed"
            ),
            "terrain",
        ),
        (
            lambda payload: payload["spatial"].__setitem__("block_size", 4),
            "spatial",
        ),
        (
            lambda payload: payload["spatial"].__setitem__("folds", 4),
            "spatial",
        ),
        (
            lambda payload: payload["spatial"].__setitem__(
                "buffer_rings", 0
            ),
            "spatial",
        ),
        (
            lambda payload: payload["bootstrap"].__setitem__(
                "iterations", 100
            ),
            "bootstrap",
        ),
        (
            lambda payload: payload["bootstrap"].__setitem__("seed", 73),
            "bootstrap",
        ),
        (
            lambda payload: payload["models"].__setitem__(
                "logistic_c", [0.1]
            ),
            "models",
        ),
        (
            lambda payload: payload["models"].__setitem__(
                "undeclared_family", [1]
            ),
            "models",
        ),
        (
            lambda payload: payload["calibration"].__setitem__(
                "methods", ["none"]
            ),
            "calibration",
        ),
        (
            lambda payload: payload["calibration"].__setitem__(
                "ece_bins", 4
            ),
            "calibration",
        ),
        (
            lambda payload: payload.__setitem__("budgets", [0.20, 0.10]),
            "budgets",
        ),
        (
            lambda payload: payload.__setitem__(
                "variants", list(reversed(payload["variants"]))
            ),
            "variants",
        ),
        (
            lambda payload: payload["variants"].pop(),
            "variants",
        ),
        (
            lambda payload: payload.__setitem__("undeclared_section", {}),
            "top-level",
        ),
        (
            lambda payload: payload["gates"].__setitem__(
                "ap_vs_explicit", "0.015"
            ),
            "gate",
        ),
    ],
    ids=(
        "seed",
        "exact-year-roles",
        "missing-years-section",
        "random-projection-dimension",
        "extra-terrain-field",
        "spatial-block-size",
        "spatial-fold-count",
        "spatial-buffer-rings",
        "bootstrap-iterations",
        "bootstrap-seed",
        "model-grid",
        "extra-model-field",
        "calibration-methods",
        "calibration-bins",
        "budget-order",
        "variant-order",
        "missing-variant",
        "extra-top-level-field",
        "gate-number-type",
    ),
)
def test_phase72b_protocol_rejects_any_mutation_to_frozen_contract(
    tmp_path, mutation, section
):
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol

    payload = _protocol_payload()
    mutation(payload)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=section):
        load_phase72b_protocol(path)


def test_phase72b_protocol_rejects_nonlocal_control_partition(tmp_path):
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol

    payload = _protocol_payload()
    payload["controls"]["partition_local"] = False
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_phase72b_protocol(path)
    except ValueError as exc:
        assert "partition-local" in str(exc).lower()
    else:
        raise AssertionError("Expected nonlocal controls to be rejected")


def test_phase72b_protocol_rejects_nontraining_transform_fit(tmp_path):
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol

    payload = _protocol_payload()
    payload["controls"]["learned_transform_fit_scope"] = "all_rows"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_phase72b_protocol(path)
    except ValueError as exc:
        assert "training rows only" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected nontraining transform fitting to be rejected"
        )


def test_phase72b_protocol_rejects_phase8_d4_table_reuse(tmp_path):
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol

    payload = _protocol_payload()
    payload["controls"]["reuse_phase8_d4_tables"] = True
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_phase72b_protocol(path)
    except ValueError as exc:
        assert "must not reuse" in str(exc).lower()
    else:
        raise AssertionError("Expected Phase 8 D4 table reuse to be rejected")


def test_phase72b_protocol_rejects_mutated_copernicus_contract(tmp_path):
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol

    mutations = {
        "source_id": "other_dem",
        "collection": "OTHER/DEM",
        "band": "elevation",
        "scale_m": 250,
        "feature_names": list(
            reversed(_protocol_payload()["terrain"]["feature_names"])
        ),
    }
    for field, value in mutations.items():
        payload = _protocol_payload()
        payload["terrain"][field] = value
        path = tmp_path / f"protocol_{field}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            load_phase72b_protocol(path)
        except ValueError as exc:
            assert "terrain" in str(exc).lower()
        else:
            raise AssertionError(
                f"Expected frozen Copernicus contract refusal for {field}"
            )


def test_phase72b_protocol_rejects_missing_copernicus_contract_field(tmp_path):
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol

    payload = _protocol_payload()
    del payload["terrain"]["band"]
    path = tmp_path / "protocol_missing_band.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_phase72b_protocol(path)
    except ValueError as exc:
        assert "terrain" in str(exc).lower()
    else:
        raise AssertionError("Expected missing terrain field refusal")


def test_phase72b_protocol_rejects_missing_terrain_section(tmp_path):
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol

    payload = _protocol_payload()
    del payload["terrain"]
    path = tmp_path / "protocol_missing_terrain.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_phase72b_protocol(path)
    except ValueError as exc:
        assert "terrain" in str(exc).lower()
    else:
        raise AssertionError("Expected missing terrain section refusal")


def test_phase72b_protocol_rejects_null_terrain_feature_names(tmp_path):
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol

    payload = _protocol_payload()
    payload["terrain"]["feature_names"] = None
    path = tmp_path / "protocol_null_features.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_phase72b_protocol(path)
    except ValueError as exc:
        assert "terrain" in str(exc).lower()
    else:
        raise AssertionError("Expected null terrain feature names refusal")


def test_phase72b_canonical_json_hash_is_key_order_independent():
    from paper11_geofm.phase72b_protocol import canonical_json_sha256

    first = {"status": "frozen", "seed": 72}
    second = {"seed": 72, "status": "frozen"}
    assert canonical_json_sha256(first) == canonical_json_sha256(second)


def test_phase72b_hashed_json_rejects_modified_payload(tmp_path):
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )

    json_path, hash_path = write_hashed_json(
        tmp_path / "frozen.json", {"status": "frozen", "seed": 72}
    )
    assert load_hashed_json(json_path, hash_path)["status"] == "frozen"
    json_path.write_text(
        '{"status":"changed","seed":72}', encoding="utf-8"
    )
    try:
        load_hashed_json(json_path, hash_path)
    except ValueError as exc:
        assert "hash mismatch" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected a modified frozen payload to be rejected"
        )


def _phase72a_regions(path: Path) -> Path:
    payload = {
        "source": {
            "source_id": "esri",
            "collection": "esri",
            "label_role": "independent_annual_product_label",
            "independent_from_dltb_slope_reward_geofm": True,
            "crop_class_code": 5,
            "scale_m": 500,
        },
        "regions": [
            {
                "region_id": "alpha",
                "bbox": [100, 20, 101, 21],
                "years": [2017, 2018, 2019],
                "grid_shape": [2, 3],
                "embedding_dim": 2,
                "embedding_pattern": "alpha_emb_{year}.npy",
                "label_pattern": "alpha_lulc_{year}.npy",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _phase72a_two_regions(path: Path) -> Path:
    payload = json.loads(_phase72a_regions(path).read_text(encoding="utf-8"))
    payload["regions"].append(
        {
            "region_id": "beta",
            "bbox": [101, 21, 102, 22],
            "years": [2017, 2018, 2019],
            "grid_shape": [2, 3],
            "embedding_dim": 2,
            "embedding_pattern": "beta_emb_{year}.npy",
            "label_pattern": "beta_lulc_{year}.npy",
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase72b_terrain_fetch_and_audit_use_exact_grid(tmp_path):
    from paper11_geofm.phase72a_label_sources import (
        load_phase72a_region_contract,
    )
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol
    from paper11_geofm.phase72b_terrain import (
        audit_phase72b_terrain_assets,
        fetch_phase72b_terrain,
    )

    def extractor(*, bbox, shape, scale_m, collection, band):
        assert bbox == (100.0, 20.0, 101.0, 21.0)
        assert shape == (2, 3)
        assert scale_m == 500
        assert collection == "COPERNICUS/DEM/GLO30"
        assert band == "DEM"
        base = np.arange(6, dtype=np.float32).reshape(2, 3)
        return {
            "elevation_mean": base,
            "elevation_std": base + 1,
            "elevation_min": base - 1,
            "elevation_max": base + 2,
            "slope_mean": base + 3,
            "slope_std": base + 4,
            "slope_max": base + 5,
            "local_relief": np.full((2, 3), 3, np.float32),
        }

    protocol = load_phase72b_protocol(
        _write_protocol(tmp_path / "protocol.json")
    )
    regions = load_phase72a_region_contract(
        _phase72a_regions(tmp_path / "regions.json")
    )
    manifest = fetch_phase72b_terrain(
        protocol,
        regions,
        output_dir=tmp_path / "terrain",
        extractor=extractor,
    )
    audit = audit_phase72b_terrain_assets(
        protocol, regions, tmp_path / "terrain"
    )
    assert manifest["status"] == "complete"
    record = manifest["records"][0]
    assert record["source_id"] == "copernicus_dem_glo30"
    assert record["band"] == "DEM"
    assert record["scale_m"] == 500
    assert record["bbox"] == [100.0, 20.0, 101.0, 21.0]
    assert record["dtype"] == "float32"
    assert record["feature_derivations"]["local_relief"] == (
        "DEM:max-minus-min"
    )
    assert audit["status"] == "terrain_inputs_ready"
    assert audit["rows"][0]["shape"] == "2x3"
    assert len(audit["rows"][0]["sha256"]) == 64


def test_phase72b_terrain_fetch_rejects_unexpected_extractor_key(tmp_path):
    from paper11_geofm.phase72a_label_sources import (
        load_phase72a_region_contract,
    )
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol
    from paper11_geofm.phase72b_terrain import fetch_phase72b_terrain

    protocol = load_phase72b_protocol(
        _write_protocol(tmp_path / "protocol.json")
    )
    regions = load_phase72a_region_contract(
        _phase72a_regions(tmp_path / "regions.json")
    )

    def extractor(**_kwargs):
        arrays = {
            name: np.zeros((2, 3), dtype=np.float32)
            for name in protocol.terrain_features
        }
        arrays["fallback_slope_proxy"] = np.zeros((2, 3), dtype=np.float32)
        return arrays

    manifest = fetch_phase72b_terrain(
        protocol,
        regions,
        output_dir=tmp_path / "terrain",
        extractor=extractor,
    )
    assert manifest["status"] == "failed"
    assert not manifest["records"]
    assert "unexpected" in manifest["failures"][0]["reason"].lower()


def test_phase72b_terrain_audit_requires_fetch_manifest(tmp_path):
    from paper11_geofm.phase72a_label_sources import (
        load_phase72a_region_contract,
    )
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol
    from paper11_geofm.phase72b_terrain import audit_phase72b_terrain_assets

    protocol = load_phase72b_protocol(
        _write_protocol(tmp_path / "protocol.json")
    )
    regions = load_phase72a_region_contract(
        _phase72a_regions(tmp_path / "regions.json")
    )
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    np.savez_compressed(
        terrain / "alpha_terrain.npz",
        **{
            name: np.zeros((2, 3), dtype=np.float32)
            for name in protocol.terrain_features
        },
    )

    audit = audit_phase72b_terrain_assets(protocol, regions, terrain)
    assert audit["status"] == "phase72b_inputs_not_ready"
    assert "manifest" in " ".join(audit["errors"]).lower()


def test_phase72b_terrain_audit_binds_every_record_provenance_field(tmp_path):
    from paper11_geofm.phase72a_label_sources import (
        load_phase72a_region_contract,
    )
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol
    from paper11_geofm.phase72b_terrain import (
        audit_phase72b_terrain_assets,
        fetch_phase72b_terrain,
    )

    protocol = load_phase72b_protocol(
        _write_protocol(tmp_path / "protocol.json")
    )
    regions = load_phase72a_region_contract(
        _phase72a_regions(tmp_path / "regions.json")
    )
    terrain = tmp_path / "terrain"

    def extractor(**_kwargs):
        return {
            name: np.zeros((2, 3), dtype=np.float32)
            for name in protocol.terrain_features
        }

    fetch_phase72b_terrain(
        protocol, regions, output_dir=terrain, extractor=extractor
    )
    manifest_path = terrain / "phase72b_terrain_fetch_manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutations = {
        "source_id": "other_dem",
        "collection": "OTHER/DEM",
        "band": "elevation",
        "feature_derivations": {},
        "scale_m": 250,
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "path": "wrong.npz",
        "shape": "3x2",
        "dtype": "float64",
        "sha256": "0" * 64,
    }
    for field, value in mutations.items():
        payload = json.loads(json.dumps(original))
        payload["records"][0][field] = value
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        audit = audit_phase72b_terrain_assets(protocol, regions, terrain)
        assert audit["status"] == "phase72b_inputs_not_ready", field
    missing_payload = json.loads(json.dumps(original))
    del missing_payload["records"][0]["source_id"]
    manifest_path.write_text(
        json.dumps(missing_payload), encoding="utf-8"
    )
    missing_audit = audit_phase72b_terrain_assets(protocol, regions, terrain)
    assert missing_audit["status"] == "phase72b_inputs_not_ready"
    assert "missing" in " ".join(missing_audit["errors"]).lower()
    manifest_path.write_text(json.dumps(original), encoding="utf-8")


def test_phase72b_terrain_audit_rejects_duplicate_records(tmp_path):
    from paper11_geofm.phase72a_label_sources import (
        load_phase72a_region_contract,
    )
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol
    from paper11_geofm.phase72b_terrain import (
        audit_phase72b_terrain_assets,
        fetch_phase72b_terrain,
    )

    protocol = load_phase72b_protocol(
        _write_protocol(tmp_path / "protocol.json")
    )
    regions = load_phase72a_region_contract(
        _phase72a_regions(tmp_path / "regions.json")
    )
    terrain = tmp_path / "terrain"

    fetch_phase72b_terrain(
        protocol,
        regions,
        output_dir=terrain,
        extractor=lambda **_kwargs: {
            name: np.zeros((2, 3), dtype=np.float32)
            for name in protocol.terrain_features
        },
    )
    manifest_path = terrain / "phase72b_terrain_fetch_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["records"].append(dict(payload["records"][0]))
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_phase72b_terrain_assets(protocol, regions, terrain)
    assert audit["status"] == "phase72b_inputs_not_ready"
    assert "duplicate" in " ".join(audit["errors"]).lower()


def test_phase72b_terrain_audit_blocks_wrong_shape(tmp_path):
    from paper11_geofm.phase72a_label_sources import (
        load_phase72a_region_contract,
    )
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol
    from paper11_geofm.phase72b_terrain import audit_phase72b_terrain_assets

    protocol = load_phase72b_protocol(
        _write_protocol(tmp_path / "protocol.json")
    )
    regions = load_phase72a_region_contract(
        _phase72a_regions(tmp_path / "regions.json")
    )
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    np.savez(
        terrain / "alpha_terrain.npz",
        **{
            name: np.zeros((2, 2), np.float32)
            for name in protocol.terrain_features
        },
    )
    audit = audit_phase72b_terrain_assets(protocol, regions, terrain)
    assert audit["status"] == "phase72b_inputs_not_ready"
    errors = " ".join(audit["errors"]).lower()
    assert "manifest" in errors
    assert "shape" in errors


def test_phase72b_terrain_audit_blocks_unexpected_proxy_feature(tmp_path):
    from paper11_geofm.phase72a_label_sources import (
        load_phase72a_region_contract,
    )
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol
    from paper11_geofm.phase72b_terrain import audit_phase72b_terrain_assets

    protocol = load_phase72b_protocol(
        _write_protocol(tmp_path / "protocol.json")
    )
    regions = load_phase72a_region_contract(
        _phase72a_regions(tmp_path / "regions.json")
    )
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    arrays = {
        name: np.zeros((2, 3), np.float32)
        for name in protocol.terrain_features
    }
    arrays["fallback_slope_proxy"] = np.zeros((2, 3), np.float32)
    np.savez_compressed(terrain / "alpha_terrain.npz", **arrays)

    audit = audit_phase72b_terrain_assets(protocol, regions, terrain)
    assert audit["status"] == "phase72b_inputs_not_ready"
    errors = " ".join(audit["errors"]).lower()
    assert "manifest" in errors
    assert "unexpected" in errors


def test_phase72b_terrain_audit_rejects_fetch_manifest_hash_mismatch(
    tmp_path,
):
    from paper11_geofm.phase72a_label_sources import (
        load_phase72a_region_contract,
    )
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol
    from paper11_geofm.phase72b_terrain import (
        audit_phase72b_terrain_assets,
        fetch_phase72b_terrain,
    )

    protocol = load_phase72b_protocol(
        _write_protocol(tmp_path / "protocol.json")
    )
    regions = load_phase72a_region_contract(
        _phase72a_regions(tmp_path / "regions.json")
    )
    terrain = tmp_path / "terrain"

    def extractor(**_kwargs):
        return {
            name: np.zeros((2, 3), dtype=np.float32)
            for name in protocol.terrain_features
        }

    fetch_phase72b_terrain(
        protocol, regions, output_dir=terrain, extractor=extractor
    )
    np.savez_compressed(
        terrain / "alpha_terrain.npz",
        **{
            name: np.ones((2, 3), dtype=np.float32)
            for name in protocol.terrain_features
        },
    )

    audit = audit_phase72b_terrain_assets(protocol, regions, terrain)
    assert audit["status"] == "phase72b_inputs_not_ready"
    assert "hash mismatch" in " ".join(audit["errors"]).lower()


def test_phase72b_terrain_audit_rejects_stale_asset_after_partial_fetch(
    tmp_path,
):
    from paper11_geofm.phase72a_label_sources import (
        load_phase72a_region_contract,
    )
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol
    from paper11_geofm.phase72b_terrain import (
        audit_phase72b_terrain_assets,
        fetch_phase72b_terrain,
    )

    protocol = load_phase72b_protocol(
        _write_protocol(tmp_path / "protocol.json")
    )
    regions = load_phase72a_region_contract(
        _phase72a_two_regions(tmp_path / "regions.json")
    )
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    np.savez_compressed(
        terrain / "beta_terrain.npz",
        **{
            name: np.zeros((2, 3), dtype=np.float32)
            for name in protocol.terrain_features
        },
    )

    def extractor(*, bbox, **_kwargs):
        if bbox[0] == 101.0:
            raise RuntimeError("simulated Earth Engine failure")
        return {
            name: np.zeros((2, 3), dtype=np.float32)
            for name in protocol.terrain_features
        }

    manifest = fetch_phase72b_terrain(
        protocol, regions, output_dir=terrain, extractor=extractor
    )
    audit = audit_phase72b_terrain_assets(protocol, regions, terrain)

    assert manifest["status"] == "partial"
    assert audit["status"] == "phase72b_inputs_not_ready"
    assert "manifest" in " ".join(audit["errors"]).lower()


def test_phase72b_terrain_cli_requires_every_declared_region(tmp_path):
    from experiments.phase72b_geofm_information_gain_screen import (
        fetch_phase72b_terrain as terrain_cli,
    )

    region_path = _phase72a_two_regions(tmp_path / "regions.json")
    protocol_path = _write_protocol(tmp_path / "protocol.json")
    original_initialize = terrain_cli.initialize_earth_engine
    original_fetch = terrain_cli.fetch_phase72b_terrain
    terrain_cli.initialize_earth_engine = lambda **_kwargs: None
    terrain_cli.fetch_phase72b_terrain = lambda *_args, **_kwargs: {
        "status": "complete",
        "records": [{"region_id": "alpha"}],
        "failures": [],
    }
    try:
        exit_code = terrain_cli.main(
            [
                "--phase72a-region-config",
                str(region_path),
                "--phase72b-protocol",
                str(protocol_path),
                "--output-dir",
                str(tmp_path / "terrain"),
            ]
        )
    finally:
        terrain_cli.initialize_earth_engine = original_initialize
        terrain_cli.fetch_phase72b_terrain = original_fetch

    assert exit_code == 1


def _explicit_fixture():
    from paper11_geofm.phase72a_label_sources import Phase72ARegionSpec

    region = Phase72ARegionSpec(
        "alpha",
        (100.0, 20.0, 101.0, 21.0),
        (2017, 2018, 2019, 2020),
        (3, 3),
        2,
        "e{year}.npy",
        "l{year}.npy",
    )
    labels = {
        2017: np.array(
            [[1, 5, 1], [5, 7, 5], [1, 5, 1]], dtype=np.int16
        ),
        2018: np.array(
            [[5, 5, 5], [5, 5, 7], [1, 5, 1]], dtype=np.int16
        ),
        2019: np.full((3, 3), 7, dtype=np.int16),
        2020: np.full((3, 3), 1, dtype=np.int16),
    }
    feature_names = (
        "elevation_mean",
        "elevation_std",
        "elevation_min",
        "elevation_max",
        "slope_mean",
        "slope_std",
        "slope_max",
        "local_relief",
    )
    terrain = {
        name: np.full((3, 3), index, dtype=np.float32)
        for index, name in enumerate(feature_names)
    }
    return region, labels, terrain


def test_phase72b_explicit_features_use_only_history_through_origin():
    from paper11_geofm.phase72b_explicit_features import (
        build_phase72b_explicit_features,
    )

    region, labels, terrain = _explicit_fixture()
    rows = [
        {
            "sample_index": 0,
            "region_id": "alpha",
            "row": 1,
            "col": 1,
            "origin_year": 2018,
            "history_length": 2,
        }
    ]
    first = build_phase72b_explicit_features(
        rows,
        regions={"alpha": region},
        labels={"alpha": labels},
        terrain={"alpha": terrain},
        crop_class_code=5,
    )
    changed = {
        **labels,
        2019: np.zeros((3, 3), dtype=np.int16),
        2020: np.zeros((3, 3), dtype=np.int16),
    }
    second = build_phase72b_explicit_features(
        rows,
        regions={"alpha": region},
        labels={"alpha": changed},
        terrain={"alpha": terrain},
        crop_class_code=5,
    )
    assert np.array_equal(
        first["explicit_history"], second["explicit_history"]
    )
    registry = first["registry"]
    values = dict(
        zip(registry["explicit_history"], first["explicit_history"][0])
    )
    assert values["terrain_local_relief"] == 7.0
    assert values["cell_crop_transition_count"] == 1.0
    assert values["cell_history_count_lulc_07"] == 1.0
    assert np.isclose(values["neighbor3_current_crop_fraction"], 6 / 9)


def test_phase72b_explicit_neighborhoods_clip_at_grid_edges():
    from paper11_geofm.phase72b_explicit_features import (
        build_phase72b_explicit_features,
    )

    region, labels, terrain = _explicit_fixture()
    result = build_phase72b_explicit_features(
        [
            {
                "sample_index": 0,
                "region_id": "alpha",
                "row": 0,
                "col": 0,
                "origin_year": 2018,
                "history_length": 2,
            }
        ],
        regions={"alpha": region},
        labels={"alpha": labels},
        terrain={"alpha": terrain},
        crop_class_code=5,
    )
    values = dict(
        zip(
            result["registry"]["explicit_history"],
            result["explicit_history"][0],
        )
    )
    assert values["neighbor3_current_crop_fraction"] == 1.0
    assert np.isclose(values["neighbor5_current_crop_fraction"], 6 / 9)


@pytest.mark.parametrize(
    ("row", "col"),
    [(-1, 0), (0, -1), (3, 0), (0, 3)],
)
def test_phase72b_explicit_features_reject_out_of_grid_samples(
    row, col
):
    from paper11_geofm.phase72b_explicit_features import (
        build_phase72b_explicit_features,
    )

    region, labels, terrain = _explicit_fixture()
    with pytest.raises(ValueError, match="grid bounds"):
        build_phase72b_explicit_features(
            [
                {
                    "sample_index": 0,
                    "region_id": "alpha",
                    "row": row,
                    "col": col,
                    "origin_year": 2018,
                    "history_length": 2,
                }
            ],
            regions={"alpha": region},
            labels={"alpha": labels},
            terrain={"alpha": terrain},
            crop_class_code=5,
        )


def test_phase72b_explicit_features_reject_misaligned_lulc_grid():
    from paper11_geofm.phase72b_explicit_features import (
        build_phase72b_explicit_features,
    )

    region, labels, terrain = _explicit_fixture()
    labels = {**labels, 2017: np.zeros((2, 3), dtype=np.int16)}
    with pytest.raises(ValueError, match="LULC shape mismatch"):
        build_phase72b_explicit_features(
            [
                {
                    "sample_index": 0,
                    "region_id": "alpha",
                    "row": 1,
                    "col": 1,
                    "origin_year": 2018,
                    "history_length": 2,
                }
            ],
            regions={"alpha": region},
            labels={"alpha": labels},
            terrain={"alpha": terrain},
            crop_class_code=5,
        )


def test_phase72b_explicit_features_reject_invalid_origin_and_cohort():
    from paper11_geofm.phase72b_explicit_features import (
        build_phase72b_explicit_features,
    )

    region, labels, terrain = _explicit_fixture()
    base_row = {
        "sample_index": 0,
        "region_id": "alpha",
        "row": 1,
        "col": 1,
        "history_length": 2,
    }
    with pytest.raises(ValueError, match="origin year"):
        build_phase72b_explicit_features(
            [{**base_row, "origin_year": 2016, "history_length": 0}],
            regions={"alpha": region},
            labels={"alpha": labels},
            terrain={"alpha": terrain},
            crop_class_code=5,
        )

    changed = {**labels, 2018: labels[2018].copy()}
    changed[2018][1, 1] = 7
    with pytest.raises(ValueError, match="crop cohort"):
        build_phase72b_explicit_features(
            [{**base_row, "origin_year": 2018}],
            regions={"alpha": region},
            labels={"alpha": changed},
            terrain={"alpha": terrain},
            crop_class_code=5,
        )


def _geofm_control_fixture():
    histories = np.zeros((4, 4, 2), dtype=np.float32)
    masks = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=bool,
    )
    histories[0, :3] = [[1, 2], [2, 4], [4, 8]]
    histories[1, :3] = [[3, 1], [4, 2], [5, 4]]
    histories[2, :2] = [[1, 1], [2, 2]]
    histories[3, :2] = [[7, 7], [8, 8]]
    rows = [
        {"sample_index": 0, "region_id": "a", "origin_year": 2019},
        {"sample_index": 1, "region_id": "a", "origin_year": 2019},
        {"sample_index": 2, "region_id": "b", "origin_year": 2018},
        {"sample_index": 3, "region_id": "b", "origin_year": 2018},
    ]
    return histories, masks, rows


def test_phase72b_geofm_temporal_summary_and_controls_are_deterministic():
    from paper11_geofm.phase72b_geofm_features import (
        build_phase72b_control_features,
        build_phase72b_geofm_features,
    )

    histories, masks, rows = _geofm_control_fixture()
    partitions = [
        "pooled:train",
        "pooled:train",
        "pooled:validation",
        "pooled:validation",
    ]
    features = build_phase72b_geofm_features(histories, masks)
    assert features["geofm_current"][0].tolist() == [4.0, 8.0]
    assert features["geofm_temporal_full"].shape == (4, 10)
    first = build_phase72b_control_features(
        "spatial_shuffle",
        histories,
        masks,
        rows,
        partition_ids=partitions,
        seed=72,
        output_dim=10,
    )
    second = build_phase72b_control_features(
        "spatial_shuffle",
        histories,
        masks,
        rows,
        partition_ids=partitions,
        seed=72,
        output_dim=10,
    )
    assert np.array_equal(first["matrix"], second["matrix"])
    assert sorted(map(tuple, first["matrix"][:, :2])) == sorted(
        map(tuple, features["geofm_current"])
    )
    assert first["manifest"]["cross_partition_count"] == 0
    for target, source in enumerate(
        first["manifest"]["source_index_by_target"]
    ):
        assert partitions[target] == partitions[source]


def test_phase72b_temporal_shuffle_keeps_current_embedding():
    from paper11_geofm.phase72b_geofm_features import (
        build_phase72b_control_features,
        build_phase72b_geofm_features,
    )

    histories = np.array(
        [[[-1.0], [2.0], [8.0], [4.0]]], dtype=np.float32
    )
    masks = np.ones((1, 4), dtype=bool)
    control = build_phase72b_control_features(
        "temporal_order_shuffle",
        histories,
        masks,
        [{"sample_index": 0, "region_id": "a", "origin_year": 2020}],
        partition_ids=["pooled:test"],
        seed=72,
        output_dim=5,
    )
    original = build_phase72b_geofm_features(histories, masks)[
        "geofm_temporal_full"
    ]
    assert control["matrix"][0, 0] == 4.0
    assert not np.array_equal(control["matrix"], original)


def test_phase72b_spatial_shuffle_cannot_cross_split_partition():
    from paper11_geofm.phase72b_geofm_features import (
        build_phase72b_control_features,
    )

    histories = np.arange(16, dtype=np.float32).reshape(4, 2, 2)
    masks = np.ones((4, 2), dtype=bool)
    rows = [
        {"sample_index": index, "region_id": "a", "origin_year": 2018}
        for index in range(4)
    ]
    partitions = ["axis:train", "axis:train", "axis:test", "axis:test"]
    result = build_phase72b_control_features(
        "spatial_shuffle",
        histories,
        masks,
        rows,
        partition_ids=partitions,
        seed=72,
        output_dim=10,
    )
    sources = result["manifest"]["source_index_by_target"]
    assert all(
        partitions[target] == partitions[source]
        for target, source in enumerate(sources)
    )
    assert result["manifest"]["cross_partition_count"] == 0


def test_phase72b_controls_are_partition_materialization_invariant():
    from paper11_geofm.phase72b_geofm_features import (
        build_phase72b_control_features,
    )

    histories = np.arange(18, dtype=np.float32).reshape(6, 3, 1)
    masks = np.ones((6, 3), dtype=bool)
    rows = [
        {"sample_index": index, "region_id": "a", "origin_year": 2019}
        for index in range(6)
    ]
    partitions = ["axis:train"] * 3 + ["axis:test"] * 3
    for control_id in ("temporal_order_shuffle", "spatial_shuffle"):
        full = build_phase72b_control_features(
            control_id,
            histories,
            masks,
            rows,
            partition_ids=partitions,
            seed=72,
            output_dim=5,
        )
        test_only = build_phase72b_control_features(
            control_id,
            histories[3:],
            masks[3:],
            rows[3:],
            partition_ids=partitions[3:],
            seed=72,
            output_dim=5,
        )
        assert np.array_equal(full["matrix"][3:], test_only["matrix"])


def test_phase72b_fit_control_materialization_never_reads_test_rows(
    monkeypatch,
):
    import paper11_geofm.phase72b_models as models

    original = models.build_phase72b_control_features
    calls = []

    def guarded(*args, **kwargs):
        partitions = list(kwargs["partition_ids"])
        sample_indexes = [int(row["sample_index"]) for row in args[3]]
        calls.append((partitions, sample_indexes))
        assert len(set(partitions)) == 1
        assert not partitions[0].endswith(":test")
        assert not ({900, 901} & set(sample_indexes))
        return original(*args, **kwargs)

    monkeypatch.setattr(models, "build_phase72b_control_features", guarded)
    histories, masks, rows = _geofm_control_fixture()
    histories = np.concatenate(
        [histories, np.full((2, 4, 2), 900, dtype=np.float32)]
    )
    masks = np.concatenate([masks, np.ones((2, 4), dtype=bool)])
    rows = [
        *rows,
        {"sample_index": 900, "region_id": "test", "origin_year": 2023},
        {"sample_index": 901, "region_id": "test", "origin_year": 2023},
    ]
    matrices = {
        "explicit_history": np.ones((6, 1), dtype=np.float32),
        "embedding_history": histories,
        "history_mask": masks,
        "geofm_temporal_full": np.ones((6, 10), dtype=np.float32),
    }
    train_matrix, validation_matrix, manifest_rows = (
        models._fit_control_variant_matrices(
        "explicit_plus_spatial_shuffle",
        matrices,
        rows,
        train_indexes=[0, 1],
        validation_indexes=[2, 3],
        axis_id="pooled_temporal",
        seed=72,
        )
    )
    assert train_matrix.shape == validation_matrix.shape == (2, 11)
    assert np.isfinite(train_matrix).all()
    assert np.isfinite(validation_matrix).all()
    assert calls == [
        (["pooled_temporal:train"] * 2, [0, 1]),
        (["pooled_temporal:validation"] * 2, [2, 3]),
    ]
    assert [row["partition_id"] for row in manifest_rows] == [
        "pooled_temporal:train",
        "pooled_temporal:validation",
    ]
    assert all(row["axis_id"] == "pooled_temporal" for row in manifest_rows)
    assert all(row["control_id"] == "spatial_shuffle" for row in manifest_rows)
    assert all(row["seed"] == 72 for row in manifest_rows)
    assert all(len(row["index_sha256"]) == 64 for row in manifest_rows)
    assert all(len(row["matrix_sha256"]) == 64 for row in manifest_rows)
    assert all(row["cross_partition_count"] == 0 for row in manifest_rows)


def test_phase72b_random_projection_is_data_independent_and_orthonormal():
    from paper11_geofm.phase72b_geofm_features import (
        build_phase72b_random_projection,
    )

    first = build_phase72b_random_projection(
        input_dim=8, output_dim=3, seed=72
    )
    second = build_phase72b_random_projection(
        input_dim=8, output_dim=3, seed=72
    )
    assert np.array_equal(first, second)
    assert np.allclose(first.T @ first, np.eye(3), atol=1e-6)


def test_phase72b_random_projection_features_are_thread_count_invariant():
    from threadpoolctl import threadpool_limits

    from paper11_geofm.phase72b_geofm_features import (
        build_phase72b_control_features,
    )

    rng = np.random.default_rng(72)
    histories = rng.normal(size=(257, 8, 64)).astype(np.float32)
    masks = np.ones((257, 8), dtype=bool)
    rows = [
        {
            "sample_index": index,
            "region_id": "fixture",
            "origin_year": 2020,
        }
        for index in range(len(histories))
    ]
    partitions = ["axis:train"] * len(histories)

    with threadpool_limits(limits=1):
        single_thread = build_phase72b_control_features(
            "random_projection",
            histories,
            masks,
            rows,
            partition_ids=partitions,
            seed=72,
            output_dim=320,
        )["matrix"]
    with threadpool_limits(limits=4):
        multi_thread = build_phase72b_control_features(
            "random_projection",
            histories,
            masks,
            rows,
            partition_ids=partitions,
            seed=72,
            output_dim=320,
        )["matrix"]

    assert np.array_equal(single_thread, multi_thread)


def test_phase72b_controls_require_frozen_partition_contract():
    from paper11_geofm.phase72b_geofm_features import (
        build_phase72b_control_features,
    )

    histories = np.ones((1, 2, 2), dtype=np.float32)
    masks = np.ones((1, 2), dtype=bool)
    rows = [
        {"sample_index": 0, "region_id": "a", "origin_year": 2018}
    ]
    with pytest.raises(ValueError, match="partition"):
        build_phase72b_control_features(
            "spatial_shuffle",
            histories,
            masks,
            rows,
            seed=72,
            output_dim=10,
        )
    with pytest.raises(ValueError, match="partition"):
        build_phase72b_control_features(
            "spatial_shuffle",
            histories,
            masks,
            rows,
            partition_ids=[""],
            seed=72,
            output_dim=10,
        )
    with pytest.raises(ValueError, match="training rows only"):
        build_phase72b_control_features(
            "random_projection",
            histories,
            masks,
            rows,
            partition_ids=["axis:train"],
            seed=72,
            output_dim=2,
            learned_transform_fit_scope="all_rows",
        )


def test_phase72b_splits_lock_years_regions_and_spatial_buffers():
    from paper11_geofm.phase72b_splits import (
        audit_phase72b_splits,
        build_phase72b_split_registry,
    )

    rows = []
    for region in ("bishan", "dongxing"):
        for year in range(2017, 2024):
            for br, bc in ((0, 0), (0, 2), (2, 0), (2, 2), (4, 4)):
                rows.append(
                    {
                        "sample_index": len(rows),
                        "region_id": region,
                        "origin_year": year,
                        "spatial_block_id": (
                            f"{region}_br{br:03d}_bc{bc:03d}"
                        ),
                        "conversion_1y": (
                            br // 2 + bc // 2 + year
                        ) % 2,
                    }
                )
    registry = build_phase72b_split_registry(
        rows,
        train_years=(2017, 2018, 2019, 2020, 2021),
        validation_year=2022,
        test_year=2023,
        folds=5,
        buffer_rings=1,
    )
    pooled = registry["pooled_temporal"]
    assert {rows[index]["origin_year"] for index in pooled["train"]} == {
        2017,
        2018,
        2019,
        2020,
        2021,
    }
    transfer = registry["bishan_to_dongxing"]
    assert {rows[index]["region_id"] for index in transfer["train"]} == {
        "bishan"
    }
    assert {rows[index]["region_id"] for index in transfer["test"]} == {
        "dongxing"
    }
    spatial = registry["spatial_bishan_fold0"]
    assert not set(spatial["train_block_ids"]) & set(
        spatial["test_block_ids"]
    )
    assert not set(spatial["train_block_ids"]) & set(
        spatial["buffer_block_ids"]
    )
    assert pooled["region_summary"]["train"] == ["bishan", "dongxing"]
    assert pooled["year_summary"]["validation"] == [2022]
    audit = audit_phase72b_splits(
        rows,
        registry,
        train_years=(2017, 2018, 2019, 2020, 2021),
        validation_year=2022,
        test_year=2023,
        spatial_folds=5,
        control_partition_local=True,
        reuse_phase8_d4_tables=False,
    )
    assert audit["status"] == "leakage_audit_passed"

    unsafe = audit_phase72b_splits(
        rows,
        registry,
        train_years=(2017, 2018, 2019, 2020, 2021),
        validation_year=2022,
        test_year=2023,
        spatial_folds=5,
        control_partition_local=False,
        reuse_phase8_d4_tables=True,
    )
    assert unsafe["status"] == "phase72b_inputs_not_ready"
    assert "partition" in " ".join(unsafe["errors"]).lower()
    assert "phase 8" in " ".join(unsafe["errors"]).lower()

    wrong_transfer = json.loads(json.dumps(registry))
    wrong_transfer["bishan_to_dongxing"]["test"] = [
        next(
            index
            for index, row in enumerate(rows)
            if row["region_id"] == "bishan" and row["origin_year"] == 2023
        )
    ]
    wrong_transfer_audit = audit_phase72b_splits(
        rows,
        wrong_transfer,
        train_years=(2017, 2018, 2019, 2020, 2021),
        validation_year=2022,
        test_year=2023,
        spatial_folds=5,
        control_partition_local=True,
        reuse_phase8_d4_tables=False,
    )
    assert wrong_transfer_audit["status"] == "phase72b_inputs_not_ready"
    assert "transfer" in " ".join(
        wrong_transfer_audit["errors"]
    ).lower()

    invalid_index = json.loads(json.dumps(registry))
    invalid_index["pooled_temporal"]["train"].append(len(rows))
    invalid_index_audit = audit_phase72b_splits(
        rows,
        invalid_index,
        train_years=(2017, 2018, 2019, 2020, 2021),
        validation_year=2022,
        test_year=2023,
        spatial_folds=5,
        control_partition_local=True,
        reuse_phase8_d4_tables=False,
    )
    assert invalid_index_audit["status"] == "phase72b_inputs_not_ready"
    assert "index" in " ".join(invalid_index_audit["errors"]).lower()


def _phase72b_prepare_fixture(tmp_path: Path):
    from paper11_geofm.phase72a_temporal_label_package import (
        build_phase72a_temporal_label_package,
        write_phase72a_temporal_label_package_artifacts,
    )

    years = list(range(2017, 2025))
    regions_payload = {
        "source": {
            "source_id": "esri_global_lulc_10m_ts",
            "collection": "esri",
            "label_role": "independent_annual_product_label",
            "independent_from_dltb_slope_reward_geofm": True,
            "crop_class_code": 5,
            "scale_m": 500,
        },
        "regions": [],
    }
    embedding_dirs = {}
    label_dirs = {}
    terrain_dir = tmp_path / "terrain"
    terrain_dir.mkdir()
    terrain_arrays = {}
    validation_conversion = [5, 7, 5, 7, 5, 5, 7, 7]
    test_conversion = [5, 5, 5, 5, 5, 5, 5, 7]
    training_conversion = [5, 7, 5, 7, 5, 5, 5, 5]
    for region_index, region_id in enumerate(("bishan", "dongxing")):
        regions_payload["regions"].append(
            {
                "region_id": region_id,
                "bbox": [100 + region_index, 20, 101 + region_index, 21],
                "years": years,
                "grid_shape": [2, 3],
                "embedding_dim": 64,
                "embedding_pattern": f"{region_id}_emb_{{year}}.npy",
                "label_pattern": f"{region_id}_lulc_{{year}}.npy",
            }
        )
        embedding_dir = tmp_path / f"{region_id}_embeddings"
        label_dir = tmp_path / f"{region_id}_labels"
        embedding_dir.mkdir()
        label_dir.mkdir()
        embedding_dirs[region_id] = embedding_dir
        label_dirs[region_id] = label_dir
        for offset, year in enumerate(years):
            embedding = np.zeros((2, 3, 64), dtype=np.float32)
            embedding[..., 0] = year + region_index
            embedding[..., 1] = np.arange(6).reshape(2, 3)
            labels = np.full((2, 3), 7, dtype=np.int32)
            labels[0, 0] = 5
            labels[0, 1] = 5
            labels[0, 2] = validation_conversion[offset]
            labels[1, 0] = test_conversion[offset]
            labels[1, 1] = test_conversion[offset]
            labels[1, 2] = training_conversion[offset]
            np.save(
                embedding_dir / f"{region_id}_emb_{year}.npy", embedding
            )
            np.save(label_dir / f"{region_id}_lulc_{year}.npy", labels)
        terrain = {
            name: np.full((2, 3), index + region_index, np.float32)
            for index, name in enumerate(
                _protocol_payload()["terrain"]["feature_names"]
            )
        }
        terrain_arrays[region_id] = terrain
    region_config = tmp_path / "regions.json"
    region_config.write_text(
        json.dumps(regions_payload), encoding="utf-8"
    )
    fixture_protocol = _protocol_payload()
    phase72a = build_phase72a_temporal_label_package(
        region_config=region_config,
        embedding_dirs=embedding_dirs,
        label_dirs=label_dirs,
        manual_review_per_stratum=1,
        spatial_block_size=fixture_protocol["spatial"]["block_size"],
    )
    phase72a_dir = tmp_path / "phase72a"
    write_phase72a_temporal_label_package_artifacts(phase72a, phase72a_dir)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(fixture_protocol), encoding="utf-8"
    )
    from paper11_geofm.phase72a_label_sources import (
        load_phase72a_region_contract,
    )
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol
    from paper11_geofm.phase72b_terrain import fetch_phase72b_terrain

    terrain_protocol = load_phase72b_protocol(protocol_path)
    terrain_contract = load_phase72a_region_contract(region_config)

    def terrain_extractor(*, bbox, **_kwargs):
        region_id = next(
            region.region_id
            for region in terrain_contract.regions
            if region.bbox == tuple(bbox)
        )
        return terrain_arrays[region_id]

    terrain_manifest = fetch_phase72b_terrain(
        terrain_protocol,
        terrain_contract,
        output_dir=terrain_dir,
        extractor=terrain_extractor,
    )
    if terrain_manifest["status"] != "complete":
        raise AssertionError("Phase 72B terrain fixture fetch failed")
    return {
        "protocol_path": protocol_path,
        "region_config": region_config,
        "phase72a_dir": phase72a_dir,
        "embedding_dirs": embedding_dirs,
        "label_dirs": label_dirs,
        "terrain_dir": terrain_dir,
    }


def test_phase72b_prepare_separates_confirmation_targets_and_freezes_protocol(
    tmp_path,
):
    from paper11_geofm.phase72b_information_gain_screen import (
        prepare_phase72b_information_gain_screen,
        write_phase72b_prepared_artifacts,
    )
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )

    inputs = _phase72b_prepare_fixture(tmp_path)
    package = prepare_phase72b_information_gain_screen(
        protocol_path=inputs["protocol_path"],
        phase72a_region_config=inputs["region_config"],
        phase72a_package_dir=inputs["phase72a_dir"],
        embedding_dirs=inputs["embedding_dirs"],
        label_dirs=inputs["label_dirs"],
        terrain_dir=inputs["terrain_dir"],
    )
    paths = write_phase72b_prepared_artifacts(
        package, tmp_path / "prepared"
    )
    with np.load(paths["development_targets_npz"]) as development:
        assert set(development["origin_year"].tolist()) <= set(
            range(2017, 2023)
        )
    with np.load(paths["confirmation_targets_npz"]) as confirmation:
        assert set(confirmation["origin_year"].tolist()) == {2023}
    frozen = load_hashed_json(
        paths["protocol_json"], paths["protocol_hash"]
    )
    assert frozen["status"] == "phase72b_protocol_frozen"
    assert package["leakage_audit"]["status"] == "leakage_audit_passed"
    assert package["control_materialization_status"] == (
        "deferred_until_axis_partitions_frozen"
    )
    assert frozen["split_before_controls"] is True
    assert frozen["control_materialization_status"] == (
        "deferred_until_axis_partitions_frozen"
    )
    assert set(frozen["source_file_sha256"]) == {
        "tracked_protocol",
        "phase72a_region_config",
        "phase72a_package",
        "phase72a_sample_index",
        "phase72a_tensors",
    }
    assert all(
        len(value) == 64
        for value in frozen["source_file_sha256"].values()
    )
    with np.load(paths["feature_matrices_npz"]) as matrices:
        assert not any(
            "shuffle" in name or "random_projection" in name
            for name in matrices.files
        )
    feature_manifest = pd.read_csv(
        paths["feature_manifest_csv"], keep_default_na=False
    )
    assert set(feature_manifest["materialization_status"]) == {
        "prepared_base_matrix"
    }
    assert not feature_manifest["control_id"].any()
    assert not feature_manifest["partition_id"].any()
    split_registry = json.loads(
        paths["split_registry_json"].read_text(encoding="utf-8")
    )
    assert not any(
        key.endswith("test")
        for key in split_registry["pooled_temporal"]["class_counts"]
    )
    terrain_manifest = pd.read_csv(
        paths["terrain_manifest_csv"], keep_default_na=False
    )
    assert {
        "region_id",
        "source_id",
        "collection",
        "band",
        "feature_derivations_json",
        "scale_m",
        "bbox_json",
        "path",
        "shape",
        "dtype",
        "sha256",
    }.issubset(terrain_manifest.columns)
    assert frozen["terrain_manifest"][0]["source_id"] == (
        "copernicus_dem_glo30"
    )


def test_phase72b_prepare_rejects_tampered_phase72a_sample_csv(tmp_path):
    from paper11_geofm.phase72b_information_gain_screen import (
        prepare_phase72b_information_gain_screen,
    )

    inputs = _phase72b_prepare_fixture(tmp_path)
    sample_path = (
        inputs["phase72a_dir"] / "phase72a_temporal_sample_index.csv"
    )
    rows = pd.read_csv(sample_path, keep_default_na=False)
    rows.loc[0, "y_1y"] = 1 - int(rows.loc[0, "y_1y"])
    rows.to_csv(sample_path, index=False)
    try:
        prepare_phase72b_information_gain_screen(
            protocol_path=inputs["protocol_path"],
            phase72a_region_config=inputs["region_config"],
            phase72a_package_dir=inputs["phase72a_dir"],
            embedding_dirs=inputs["embedding_dirs"],
            label_dirs=inputs["label_dirs"],
            terrain_dir=inputs["terrain_dir"],
        )
    except ValueError as exc:
        assert "phase 72a derived sample mismatch" in str(exc).lower()
    else:
        raise AssertionError("Expected tampered Phase 72A CSV to be rejected")


def test_phase72b_prepare_rejects_tampered_spatial_block_id(tmp_path):
    from paper11_geofm.phase72b_information_gain_screen import (
        prepare_phase72b_information_gain_screen,
    )

    inputs = _phase72b_prepare_fixture(tmp_path)
    sample_path = (
        inputs["phase72a_dir"] / "phase72a_temporal_sample_index.csv"
    )
    rows = pd.read_csv(sample_path, keep_default_na=False)
    rows.loc[0, "spatial_block_id"] = "bishan_br999_bc999"
    rows.to_csv(sample_path, index=False)
    with pytest.raises(ValueError, match="spatial block"):
        prepare_phase72b_information_gain_screen(
            protocol_path=inputs["protocol_path"],
            phase72a_region_config=inputs["region_config"],
            phase72a_package_dir=inputs["phase72a_dir"],
            embedding_dirs=inputs["embedding_dirs"],
            label_dirs=inputs["label_dirs"],
            terrain_dir=inputs["terrain_dir"],
        )


def test_phase72b_prepare_rejects_tampered_phase72a_tensor_npz(tmp_path):
    from paper11_geofm.phase72b_information_gain_screen import (
        prepare_phase72b_information_gain_screen,
    )

    inputs = _phase72b_prepare_fixture(tmp_path)
    tensor_path = inputs["phase72a_dir"] / "phase72a_temporal_samples.npz"
    with np.load(tensor_path) as loaded:
        tensors = {name: loaded[name].copy() for name in loaded.files}
    tensors["embedding_history"][0, 0, 0] += 1.0
    np.savez_compressed(tensor_path, **tensors)
    try:
        prepare_phase72b_information_gain_screen(
            protocol_path=inputs["protocol_path"],
            phase72a_region_config=inputs["region_config"],
            phase72a_package_dir=inputs["phase72a_dir"],
            embedding_dirs=inputs["embedding_dirs"],
            label_dirs=inputs["label_dirs"],
            terrain_dir=inputs["terrain_dir"],
        )
    except ValueError as exc:
        assert "phase 72a derived tensor mismatch" in str(exc).lower()
    else:
        raise AssertionError("Expected tampered Phase 72A NPZ to be rejected")


def test_phase72b_prepare_accepts_equivalent_relocated_phase72a_assets(
    tmp_path,
):
    from paper11_geofm.phase72b_information_gain_screen import (
        prepare_phase72b_information_gain_screen,
    )

    inputs = _phase72b_prepare_fixture(tmp_path)
    relocated_embeddings = {}
    relocated_labels = {}
    for region_id in ("bishan", "dongxing"):
        relocated_embeddings[region_id] = tmp_path / f"moved_{region_id}_emb"
        relocated_labels[region_id] = tmp_path / f"moved_{region_id}_labels"
        shutil.copytree(
            inputs["embedding_dirs"][region_id],
            relocated_embeddings[region_id],
        )
        shutil.copytree(
            inputs["label_dirs"][region_id], relocated_labels[region_id]
        )
    package = prepare_phase72b_information_gain_screen(
        protocol_path=inputs["protocol_path"],
        phase72a_region_config=inputs["region_config"],
        phase72a_package_dir=inputs["phase72a_dir"],
        embedding_dirs=relocated_embeddings,
        label_dirs=relocated_labels,
        terrain_dir=inputs["terrain_dir"],
    )
    assert package["frozen_protocol"]["development_target_rows"] > 0


def test_phase72b_metrics_match_hand_computable_brier_and_ece():
    from paper11_geofm.phase72b_metrics import phase72b_metrics

    result = phase72b_metrics(
        np.array([0, 1, 1, 0]),
        np.array([0.1, 0.8, 0.6, 0.2]),
        threshold=0.5,
        budgets=(0.10, 0.20),
        ece_bins=2,
    )
    assert result["brier"] == 0.0625
    assert result["ece"] == 0.225
    assert result["balanced_accuracy"] == 1.0


def test_phase72b_block_bootstrap_uses_paired_blocks():
    from paper11_geofm.phase72b_metrics import paired_block_bootstrap

    rows = [
        {"region_id": "a", "spatial_block_id": "a0"},
        {"region_id": "a", "spatial_block_id": "a0"},
        {"region_id": "a", "spatial_block_id": "a1"},
        {"region_id": "a", "spatial_block_id": "a1"},
    ]
    y = np.array([0, 1, 0, 1])
    explicit = np.array([0.4, 0.6, 0.4, 0.6])
    geofm = np.array([0.1, 0.9, 0.2, 0.8])
    result = paired_block_bootstrap(
        y, explicit, geofm, rows, iterations=100, seed=72
    )
    assert result["ap_delta_mean"] >= 0
    assert result["brier_delta_mean"] > 0
    assert result["n_clusters"] == 2


def test_phase72b_metrics_and_bootstrap_reject_invalid_inputs():
    from paper11_geofm.phase72b_metrics import (
        expected_calibration_error,
        paired_block_bootstrap,
        phase72b_metrics,
    )

    with pytest.raises(ValueError, match="binary"):
        expected_calibration_error([0.2, 1.0], [0.1, 0.9], bins=2)
    with pytest.raises(ValueError, match="binary"):
        phase72b_metrics(
            [0.2, 1.0],
            [0.1, 0.9],
            threshold=0.5,
            budgets=(0.1,),
            ece_bins=2,
        )
    with pytest.raises(ValueError, match="probabil"):
        phase72b_metrics(
            [0, 1],
            [0.1, np.nan],
            threshold=0.5,
            budgets=(0.1,),
            ece_bins=2,
        )
    with pytest.raises(ValueError, match="threshold"):
        phase72b_metrics(
            [0, 1],
            [0.1, 0.9],
            threshold=1.5,
            budgets=(0.1,),
            ece_bins=2,
        )
    with pytest.raises(ValueError, match="budget"):
        phase72b_metrics(
            [0, 1],
            [0.1, 0.9],
            threshold=0.5,
            budgets=(2.0,),
            ece_bins=2,
        )
    with pytest.raises(ValueError, match="bin"):
        phase72b_metrics(
            [0, 1],
            [0.1, 0.9],
            threshold=0.5,
            budgets=(0.1,),
            ece_bins=0,
        )

    rows = [
        {"region_id": "", "spatial_block_id": "a0"},
        {"region_id": "a", "spatial_block_id": "a1"},
    ]
    with pytest.raises(ValueError, match="region"):
        paired_block_bootstrap(
            [0, 1], [0.4, 0.6], [0.1, 0.9], rows, iterations=10, seed=72
        )


def _gate_inputs():
    return {
        "pooled_delta": {
            "ap_delta": 0.020,
            "brier_delta": 0.006,
            "ece_delta": 0.002,
        },
        "pooled_bootstrap": {
            "ap_delta_ci_low": 0.001,
            "brier_delta_ci_low": -0.001,
        },
        "control_rows": [
            {
                "control_id": name,
                "ap_delta": 0.006,
                "brier_delta": 0.003,
                "ece_delta": 0.001,
            }
            for name in (
                "temporal_order_shuffle",
                "spatial_shuffle",
                "random_projection",
            )
        ],
        "transfer_rows": [
            {
                "axis_id": "bishan_to_dongxing",
                "ap_delta": 0.006,
                "brier_delta": 0.0,
                "ece_delta": 0.001,
            },
            {
                "axis_id": "dongxing_to_bishan",
                "ap_delta": 0.0,
                "brier_delta": 0.003,
                "ece_delta": 0.001,
            },
        ],
        "spatial_rows": [
            {
                "axis_id": f"spatial_{region_id}_fold{fold}",
                "region_id": region_id,
                "rows": 10,
                "ap_delta": 0.001,
                "brier_delta": 0.001,
                "ece_delta": 0.001,
            }
            for region_id in ("bishan", "dongxing")
            for fold in (0, 1)
        ],
        "gates": _protocol_payload()["gates"],
    }


def test_phase72b_gate_emits_all_frozen_statuses():
    from paper11_geofm.phase72b_metrics import build_phase72b_gate

    base = _gate_inputs()
    supported = build_phase72b_gate(**base, leakage_ok=True)
    assert supported["phase72b_status"] == "geofm_information_supported"

    mixed_inputs = {**base, "transfer_rows": [*base["transfer_rows"]]}
    mixed_inputs["transfer_rows"][-1] = {
        "axis_id": "dongxing_to_bishan",
        "ap_delta": -0.006,
        "brier_delta": -0.003,
        "ece_delta": -0.001,
    }
    mixed = build_phase72b_gate(**mixed_inputs, leakage_ok=True)
    assert mixed["phase72b_status"] == "geofm_information_mixed"

    unsupported_inputs = {
        **base,
        "pooled_delta": {
            "ap_delta": 0.001,
            "brier_delta": 0.001,
            "ece_delta": 0.001,
        },
    }
    unsupported = build_phase72b_gate(
        **unsupported_inputs, leakage_ok=True
    )
    assert (
        unsupported["phase72b_status"]
        == "geofm_information_not_supported"
    )

    blocked = build_phase72b_gate(**base, leakage_ok=False)
    assert blocked["phase72b_status"] == "phase72b_inputs_not_ready"


@pytest.mark.parametrize(
    "replacement",
    [
        {
            "control_rows": [
                {
                    "control_id": "temporal_order_shuffle",
                    "ap_delta": 0.006,
                    "brier_delta": 0.003,
                }
            ]
        },
        {
            "transfer_rows": [
                {
                    "axis_id": "bishan_to_dongxing",
                    "ap_delta": 0.006,
                    "brier_delta": 0.003,
                }
            ]
            * 2
        },
        {
            "spatial_rows": [
                {
                    "axis_id": "spatial_bishan_fold0",
                    "region_id": "bishan",
                    "rows": 10,
                    "ap_delta": 0.001,
                    "brier_delta": 0.001,
                    "ece_delta": 0.001,
                }
            ]
        },
        {
            "spatial_rows": [
                {
                    "axis_id": f"spatial_{region_id}_fold0",
                    "region_id": region_id,
                    "rows": 10,
                    "ap_delta": 0.001,
                    "brier_delta": 0.001,
                    "ece_delta": 0.001,
                }
                for region_id in ("bishan", "dongxing")
            ]
        },
        {
            "gates": {
                **_protocol_payload()["gates"],
                "ap_vs_explicit": 0.0,
            }
        },
    ],
    ids=(
        "missing-controls",
        "duplicate-transfer-axis",
        "missing-spatial-region",
        "one-fold-per-region",
        "mutated-gate-threshold",
    ),
)
def test_phase72b_gate_rejects_incomplete_evidence_identity(replacement):
    from paper11_geofm.phase72b_metrics import build_phase72b_gate

    inputs = {**_gate_inputs(), **replacement}
    result = build_phase72b_gate(**inputs, leakage_ok=True)
    assert result["phase72b_status"] == "phase72b_inputs_not_ready"
    assert result["checks"]["input_ready"] is False
    assert result["evidence"]["input_blockers"]


def test_phase72b_gate_rejects_noncanonical_spatial_axis_identity():
    from paper11_geofm.phase72b_metrics import build_phase72b_gate

    inputs = {
        **_gate_inputs(),
        "spatial_rows": [
            {
                "axis_id": f"spatial_{region_id}_fold{fold}",
                "region_id": region_id,
                "rows": 10,
                "ap_delta": 0.001,
                "brier_delta": 0.001,
                "ece_delta": 0.001,
            }
            for region_id in ("bishan", "dongxing")
            for fold in ("00", "1")
        ],
    }
    result = build_phase72b_gate(**inputs, leakage_ok=True)

    assert result["phase72b_status"] == "phase72b_inputs_not_ready"
    assert result["checks"]["input_ready"] is False
    assert {
        "non-canonical spatial fold identity: spatial_bishan_fold00",
        "non-canonical spatial fold identity: spatial_dongxing_fold00",
    } <= set(result["evidence"]["input_blockers"])


def test_phase72b_gate_audits_every_delta_and_preserves_pooled_direction():
    from paper11_geofm.phase72b_metrics import build_phase72b_gate

    base = _gate_inputs()
    supported = build_phase72b_gate(**base, leakage_ok=True)
    evidence = supported["evidence"]
    assert evidence["pooled"]["deltas"] == base["pooled_delta"]
    assert evidence["pooled"]["bootstrap"] == base["pooled_bootstrap"]
    assert len(evidence["controls"]) == 3
    assert len(evidence["transfers"]) == 2
    assert len(evidence["spatial_folds"]) == 4
    assert len(evidence["spatial_regions"]) == 2
    assert all(row["passed"] for row in evidence["controls"])
    assert all(row["passed"] for row in evidence["transfers"])
    assert all(row["passed"] for row in evidence["spatial_folds"])

    brier_only_spatial = {
        **base,
        "pooled_delta": {
            "ap_delta": 0.001,
            "brier_delta": 0.006,
            "ece_delta": 0.011,
        },
        "pooled_bootstrap": {
            "ap_delta_ci_low": -0.001,
            "brier_delta_ci_low": 0.001,
        },
        "spatial_rows": [
            {**row, "ap_delta": 0.100, "brier_delta": -0.100}
            for row in base["spatial_rows"]
        ],
    }
    mixed = build_phase72b_gate(**brier_only_spatial, leakage_ok=True)
    assert mixed["phase72b_status"] == "geofm_information_mixed"
    assert mixed["checks"]["spatial"] is False
    assert all(
        row["direction_checks"] == {"brier": False}
        for row in mixed["evidence"]["spatial_folds"]
    )


def test_phase72b_model_selection_returns_frozen_bundle():
    from paper11_geofm.phase72b_models import (
        fit_select_phase72b_model,
        predict_phase72b_bundle,
    )

    rng = np.random.default_rng(72)
    features = rng.normal(size=(120, 3))
    outcome = (features[:, 0] + 0.8 * features[:, 2] > 0).astype(int)
    protocol = _protocol_payload()
    protocol["models"] = {
        "logistic_c": [0.1, 1.0],
        "logistic_class_weight": ["none"],
        "hgb_learning_rate": [0.08],
        "hgb_max_leaf_nodes": [15],
        "hgb_min_samples_leaf": [5],
        "hgb_max_iter": 30,
        "hgb_l2_regularization": [0.0],
    }
    bundle, rows = fit_select_phase72b_model(
        features[:80],
        outcome[:80],
        features[80:],
        outcome[80:],
        variant_id="fixture",
        axis_id="pooled_temporal",
        protocol=protocol,
    )
    probability = predict_phase72b_bundle(bundle, features[80:])
    assert bundle["variant_id"] == "fixture"
    assert bundle["calibration_method"] in {
        "none",
        "sigmoid",
        "isotonic",
    }
    assert len(rows) > 1
    assert np.isfinite(probability).all()


@pytest.mark.parametrize(
    ("train_x", "train_y", "validation_x", "validation_y"),
    [
        (np.ones((4, 2)), [0, 1, 0.5, 1], np.ones((4, 2)), [0, 1, 0, 1]),
        (np.ones((4, 2)), [0, 2, 0, 2], np.ones((4, 2)), [0, 1, 0, 1]),
        (
            np.ones((4, 2)),
            np.asarray(["0", "1", "0", "1"]),
            np.ones((4, 2)),
            [0, 1, 0, 1],
        ),
        (
            np.asarray([["1", "2"]] * 4),
            [0, 1, 0, 1],
            np.ones((4, 2)),
            [0, 1, 0, 1],
        ),
        (np.ones((4, 2)), [0, 1, 0, 1], np.ones((4, 2)), [0, 0, 0, 0]),
        (
            np.asarray([[0.0], [1.0], [np.nan], [3.0]]),
            [0, 1, 0, 1],
            np.ones((4, 1)),
            [0, 1, 0, 1],
        ),
        (np.ones((3, 2)), [0, 1, 0, 1], np.ones((4, 2)), [0, 1, 0, 1]),
        (np.ones(4), [0, 1, 0, 1], np.ones((4, 1)), [0, 1, 0, 1]),
    ],
    ids=(
        "fractional-train-label",
        "nonbinary-train-label",
        "string-train-label",
        "string-train-feature",
        "single-class-validation",
        "nonfinite-feature",
        "row-mismatch",
        "one-dimensional-feature",
    ),
)
def test_phase72b_model_fit_rejects_invalid_inputs(
    train_x, train_y, validation_x, validation_y
):
    from paper11_geofm.phase72b_models import fit_select_phase72b_model

    with pytest.raises(ValueError, match="Phase 72B model"):
        fit_select_phase72b_model(
            train_x,
            train_y,
            validation_x,
            validation_y,
            variant_id="fixture",
            axis_id="pooled_temporal",
            protocol=_protocol_payload(),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda arrays: arrays["sample_index"].__setitem__(
            1, arrays["sample_index"][0]
        ),
        lambda arrays: arrays["origin_year"].__setitem__(0, 2023),
        lambda arrays: arrays.__setitem__(
            "sample_index", np.asarray([0.5, 1.0])
        ),
        lambda arrays: arrays.__setitem__(
            "origin_year", np.asarray([2017.5, 2022.0])
        ),
        lambda arrays: arrays["conversion_1y"].__setitem__(0, 2),
        lambda arrays: arrays.__setitem__(
            "conversion_1y", np.asarray([0.5, 1.0])
        ),
        lambda arrays: arrays.__setitem__("undeclared", np.asarray([1, 2])),
    ],
    ids=(
        "duplicate-index",
        "wrong-year",
        "fractional-index",
        "fractional-year",
        "nonbinary-label",
        "fractional-label",
        "extra-array",
    ),
)
def test_phase72b_development_targets_require_exact_semantics(
    tmp_path, mutation
):
    from paper11_geofm.phase72b_models import _development_outcome

    feature_rows = [
        {"sample_index": 0, "origin_year": 2017},
        {"sample_index": 1, "origin_year": 2022},
        {"sample_index": 2, "origin_year": 2023},
    ]
    arrays = {
        "sample_index": np.asarray([0, 1], np.int32),
        "origin_year": np.asarray([2017, 2022], np.int16),
        "conversion_1y": np.asarray([0, 1], np.int8),
    }
    mutation(arrays)
    path = tmp_path / "development.npz"
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="development target"):
        _development_outcome(
            path,
            feature_rows=feature_rows,
            development_years={2017, 2018, 2019, 2020, 2021, 2022},
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda arrays: arrays["sample_index"].__setitem__(
            1, arrays["sample_index"][0]
        ),
        lambda arrays: arrays["origin_year"].__setitem__(0, 2022),
        lambda arrays: arrays.__setitem__(
            "sample_index", np.asarray([1.5, 2.0])
        ),
        lambda arrays: arrays.__setitem__(
            "origin_year", np.asarray([2023.5, 2023.0])
        ),
        lambda arrays: arrays["conversion_1y"].__setitem__(0, 2),
        lambda arrays: arrays.__setitem__(
            "conversion_1y", np.asarray([0.5, 1.0])
        ),
        lambda arrays: arrays.__setitem__("undeclared", np.asarray([1, 2])),
    ],
    ids=(
        "duplicate-index",
        "wrong-year",
        "fractional-index",
        "fractional-year",
        "nonbinary-label",
        "fractional-label",
        "extra-array",
    ),
)
def test_phase72b_confirmation_targets_require_exact_semantics(
    tmp_path, mutation
):
    from paper11_geofm.phase72b_information_gain_screen import (
        _confirmation_outcome,
    )

    feature_rows = [
        {"sample_index": 0, "origin_year": 2022},
        {"sample_index": 1, "origin_year": 2023},
        {"sample_index": 2, "origin_year": 2023},
    ]
    arrays = {
        "sample_index": np.asarray([1, 2], np.int32),
        "origin_year": np.asarray([2023, 2023], np.int16),
        "conversion_1y": np.asarray([0, 1], np.int8),
    }
    mutation(arrays)
    path = tmp_path / "confirmation.npz"
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="confirmation target"):
        _confirmation_outcome(
            path,
            feature_rows=feature_rows,
            confirmation_years={2023},
        )


def test_phase72b_frozen_model_grid_expands_to_all_candidates():
    from paper11_geofm.phase72b_models import _candidate_configs

    candidates = _candidate_configs(_protocol_payload())

    assert len(candidates) == 24
    assert sum(row["model_family"] == "logistic" for row in candidates) == 8
    assert sum(row["model_family"] == "hgb" for row in candidates) == 16


def test_phase72b_model_fit_limits_native_threads(monkeypatch):
    import paper11_geofm.phase72b_models as models

    entered = []

    class _ThreadLimit:
        def __enter__(self):
            entered.append(True)

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(
        models, "threadpool_limits", lambda *, limits: _ThreadLimit()
    )
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    outcome = np.asarray([0, 0, 1, 1])
    models._fit_estimator(
        features,
        outcome,
        {"model_family": "logistic", "C": 1.0, "class_weight": None},
        seed=72,
    )
    assert entered == [True]


def test_phase72b_model_search_uses_bounded_thread_parallelism(monkeypatch):
    import paper11_geofm.phase72b_models as models

    captured = {}

    class _ImmediateParallel:
        def __init__(self, *, n_jobs, prefer):
            captured.update({"n_jobs": n_jobs, "prefer": prefer})

        def __call__(self, jobs):
            return [function(*args, **kwargs) for function, args, kwargs in jobs]

    monkeypatch.setattr(models, "Parallel", _ImmediateParallel)
    monkeypatch.setattr(
        models,
        "delayed",
        lambda function: (
            lambda *args, **kwargs: (function, args, kwargs)
        ),
    )
    protocol = _protocol_payload()
    protocol["models"] = {
        "logistic_c": [0.1],
        "logistic_class_weight": ["none"],
        "hgb_learning_rate": [],
        "hgb_max_leaf_nodes": [],
        "hgb_min_samples_leaf": [],
        "hgb_max_iter": 20,
        "hgb_l2_regularization": [],
    }
    protocol["calibration"] = {"methods": ["none"], "ece_bins": 2}
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    outcome = np.asarray([0, 1, 0, 1])
    models.fit_select_phase72b_model(
        features,
        outcome,
        features,
        outcome,
        variant_id="fixture",
        axis_id="pooled_temporal",
        protocol=protocol,
    )
    assert captured == {"n_jobs": 4, "prefer": "threads"}


def _install_phase72b_fast_pipeline(monkeypatch):
    import paper11_geofm.phase72b_geofm_features as geofm_features
    import paper11_geofm.phase72b_information_gain_screen as screen
    import paper11_geofm.phase72b_models as models

    monkeypatch.setattr(
        models,
        "_candidate_configs",
        lambda _protocol: [
            {
                "model_family": "logistic",
                "C": 0.1,
                "class_weight": None,
            }
        ],
    )

    def fast_projection(*, input_dim, output_dim, seed):
        del seed
        projection = np.zeros((int(input_dim), int(output_dim)), np.float32)
        projection[: int(output_dim), :] = np.eye(
            int(output_dim), dtype=np.float32
        )
        return projection

    monkeypatch.setattr(
        geofm_features, "build_phase72b_random_projection", fast_projection
    )
    full_bootstrap = screen.paired_block_bootstrap

    def fast_bootstrap(*args, iterations, **kwargs):
        del iterations
        return full_bootstrap(*args, iterations=100, **kwargs)

    monkeypatch.setattr(screen, "paired_block_bootstrap", fast_bootstrap)


def test_phase72b_fit_freeze_writes_hashed_bundles(tmp_path, monkeypatch):
    from paper11_geofm.phase72b_information_gain_screen import (
        prepare_phase72b_information_gain_screen,
        write_phase72b_prepared_artifacts,
    )
    from paper11_geofm.phase72b_models import (
        fit_freeze_phase72b_models,
        load_phase72b_model_bundle,
    )
    from paper11_geofm.phase72b_protocol import load_hashed_json

    _install_phase72b_fast_pipeline(monkeypatch)
    inputs = _phase72b_prepare_fixture(tmp_path)
    prepared_package = prepare_phase72b_information_gain_screen(
        protocol_path=inputs["protocol_path"],
        phase72a_region_config=inputs["region_config"],
        phase72a_package_dir=inputs["phase72a_dir"],
        embedding_dirs=inputs["embedding_dirs"],
        label_dirs=inputs["label_dirs"],
        terrain_dir=inputs["terrain_dir"],
    )
    prepared_dir = tmp_path / "prepared"
    write_phase72b_prepared_artifacts(prepared_package, prepared_dir)
    frozen_dir = tmp_path / "frozen"
    target_path = prepared_dir / "phase72b_development_targets.npz"
    original_target_bytes = target_path.read_bytes()
    with np.load(target_path) as loaded:
        original_targets = {
            name: loaded[name].copy() for name in loaded.files
        }
    changed_targets = {
        name: value.copy() for name, value in original_targets.items()
    }
    changed_targets["conversion_1y"][0] = (
        1 - changed_targets["conversion_1y"][0]
    )
    np.savez_compressed(target_path, **changed_targets)
    try:
        fit_freeze_phase72b_models(
            prepared_dir=prepared_dir, output_dir=frozen_dir
        )
    except ValueError as exc:
        assert "prepared artifact hash mismatch" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected modified development targets to be rejected"
        )
    target_path.write_bytes(original_target_bytes)
    selected, paths = fit_freeze_phase72b_models(
        prepared_dir=prepared_dir, output_dir=frozen_dir
    )
    assert selected["status"] == "phase72b_models_frozen"
    assert set(selected["axes"]) >= {
        "pooled_temporal",
        "bishan_to_dongxing",
        "dongxing_to_bishan",
    }
    loaded = load_hashed_json(
        paths["selected_models_json"], paths["selected_models_hash"]
    )
    assert loaded["status"] == "phase72b_models_frozen"
    assert len(loaded["fit_control_manifest_sha256"]) == 64
    assert loaded["fit_implementation_id"]
    assert paths["fit_control_manifest_csv"].exists()
    fit_control_rows = pd.read_csv(
        paths["fit_control_manifest_csv"], keep_default_na=False
    )
    assert set(fit_control_rows.columns) == {
        "axis_id",
        "partition_id",
        "control_id",
        "seed",
        "index_sha256",
        "matrix_sha256",
        "cross_partition_count",
    }
    assert len(fit_control_rows) > 0
    assert (fit_control_rows["cross_partition_count"] == 0).all()
    assert (
        loaded["fit_control_manifest_sha256"]
        == paths["fit_control_manifest_sha256"]
    )
    assert all(
        len(record["bundle_sha256"]) == 64
        for record in selected["bundle_records"]
    )
    import paper11_geofm.phase72b_models as models

    def _unexpected_fit(*args, **kwargs):
        raise AssertionError("Completed Phase 72B bundles should be resumed")

    monkeypatch.setattr(models, "fit_select_phase72b_model", _unexpected_fit)
    monkeypatch.setattr(models, "fit_fixed_phase72b_model", _unexpected_fit)
    resumed, resumed_paths = fit_freeze_phase72b_models(
        prepared_dir=prepared_dir, output_dir=frozen_dir
    )
    assert resumed == selected
    assert (
        resumed_paths["selected_models_hash"].read_text(encoding="ascii")
        == paths["selected_models_hash"].read_text(encoding="ascii")
    )
    record = selected["bundle_records"][0]
    bundle_path = frozen_dir / record["bundle_path"]
    loaded_bundle = load_phase72b_model_bundle(
        bundle_path, record["bundle_sha256"]
    )
    assert loaded_bundle["fit_implementation_id"] == loaded[
        "fit_implementation_id"
    ]
    original = bundle_path.read_bytes()
    bundle_path.write_bytes(original + b"changed")
    try:
        load_phase72b_model_bundle(bundle_path, record["bundle_sha256"])
    except ValueError as exc:
        assert "hash" in str(exc).lower()
    else:
        raise AssertionError("Expected a modified model bundle to be rejected")


def test_phase72b_resume_rejects_record_bundle_semantic_mismatch(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_models import fit_freeze_phase72b_models
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    progress_path = frozen_dir / "phase72b_fit_progress.json"
    progress = load_hashed_json(progress_path)
    first_entry = next(iter(progress["entries"].values()))
    first_entry["record"]["candidate_id"] = "resigned-mismatch"
    write_hashed_json(progress_path, progress)

    with pytest.raises(ValueError, match="bundle semantics"):
        fit_freeze_phase72b_models(
            prepared_dir=prepared_dir, output_dir=frozen_dir
        )


def test_phase72b_bundle_semantics_bind_control_seed(tmp_path, monkeypatch):
    from paper11_geofm.phase72b_models import (
        load_phase72b_model_bundle,
        validate_phase72b_bundle_record_semantics,
    )

    _, _, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    selected = json.loads(
        (frozen_dir / "phase72b_selected_models.json").read_text(
            encoding="utf-8"
        )
    )
    record = next(
        row for row in selected["bundle_records"] if row["control_seed"] != ""
    )
    bundle = load_phase72b_model_bundle(
        frozen_dir / record["bundle_path"], record["bundle_sha256"]
    )
    changed_record = {**record, "control_seed": int(record["control_seed"]) + 1}

    with pytest.raises(ValueError, match="bundle semantics"):
        validate_phase72b_bundle_record_semantics(changed_record, bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("validation_metrics", None),
        ("estimator_params", None),
        ("f1_threshold", None),
        ("budget_thresholds", None),
    ],
)
def test_phase72b_bundle_semantics_normalizes_malformed_fields(
    tmp_path, monkeypatch, field, value
):
    from paper11_geofm.phase72b_models import (
        load_phase72b_model_bundle,
        validate_phase72b_bundle_record_semantics,
    )

    _, _, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    selected = json.loads(
        (frozen_dir / "phase72b_selected_models.json").read_text(
            encoding="utf-8"
        )
    )
    record = selected["bundle_records"][0]
    bundle = load_phase72b_model_bundle(
        frozen_dir / record["bundle_path"], record["bundle_sha256"]
    )
    bundle[field] = value

    with pytest.raises(ValueError, match="bundle semantics"):
        validate_phase72b_bundle_record_semantics(record, bundle)


def test_phase72b_bundle_semantics_rejects_resigned_invalid_metric(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_models import (
        load_phase72b_model_bundle,
        validate_phase72b_bundle_record_semantics,
    )

    _, _, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    selected = json.loads(
        (frozen_dir / "phase72b_selected_models.json").read_text(
            encoding="utf-8"
        )
    )
    record = selected["bundle_records"][0]
    bundle = load_phase72b_model_bundle(
        frozen_dir / record["bundle_path"], record["bundle_sha256"]
    )
    bundle["validation_metrics"]["average_precision"] = 2.0
    record["validation_average_precision"] = 2.0

    with pytest.raises(ValueError, match="bundle semantics"):
        validate_phase72b_bundle_record_semantics(record, bundle)


def test_phase72b_confirmation_rejects_resigned_invalid_secondary_metric(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )
    from paper11_geofm.phase72b_terrain import _file_sha256

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    selected_path = frozen_dir / "phase72b_selected_models.json"
    selected = load_hashed_json(selected_path)
    record = selected["bundle_records"][0]
    bundle_path = frozen_dir / record["bundle_path"]
    bundle = joblib.load(bundle_path)
    bundle["validation_metrics"]["roc_auc"] = float("inf")
    joblib.dump(bundle, bundle_path)
    record["bundle_sha256"] = _file_sha256(bundle_path)
    _, selected_hash_path = write_hashed_json(selected_path, selected)

    progress_path = frozen_dir / "phase72b_fit_progress.json"
    progress = load_hashed_json(progress_path)
    filename = Path(record["bundle_path"]).name
    progress["entries"][filename]["record"] = dict(record)
    progress["selected_models_sha256"] = selected_hash_path.read_text(
        encoding="ascii"
    ).strip()
    write_hashed_json(progress_path, progress)

    with pytest.raises(ValueError, match="bundle semantics"):
        confirm_phase72b_information_gain_screen(
            prepared_dir=prepared_dir, frozen_dir=frozen_dir
        )


def test_phase72b_confirmation_checks_feature_count_before_target_open(
    tmp_path, monkeypatch
):
    import paper11_geofm.phase72b_information_gain_screen as screen
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )
    from paper11_geofm.phase72b_terrain import _file_sha256

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    selected_path = frozen_dir / "phase72b_selected_models.json"
    selected = load_hashed_json(selected_path)
    record = selected["bundle_records"][0]
    bundle_path = frozen_dir / record["bundle_path"]
    bundle = joblib.load(bundle_path)
    bundle["feature_count"] = int(bundle["feature_count"]) + 1
    joblib.dump(bundle, bundle_path)
    record["bundle_sha256"] = _file_sha256(bundle_path)
    _, selected_hash_path = write_hashed_json(selected_path, selected)

    progress_path = frozen_dir / "phase72b_fit_progress.json"
    progress = load_hashed_json(progress_path)
    filename = Path(record["bundle_path"]).name
    progress["entries"][filename]["record"] = dict(record)
    progress["selected_models_sha256"] = selected_hash_path.read_text(
        encoding="ascii"
    ).strip()
    write_hashed_json(progress_path, progress)

    def unexpected_target_open(*args, **kwargs):
        raise AssertionError("confirmation target opened before bundle audit")

    monkeypatch.setattr(screen, "_confirmation_outcome", unexpected_target_open)
    with pytest.raises(ValueError, match="feature count mismatch"):
        screen.confirm_phase72b_information_gain_screen(
            prepared_dir=prepared_dir, frozen_dir=frozen_dir
        )


def _prepare_and_freeze(tmp_path: Path, monkeypatch):
    from paper11_geofm.phase72b_information_gain_screen import (
        prepare_phase72b_information_gain_screen,
        write_phase72b_prepared_artifacts,
    )
    from paper11_geofm.phase72b_models import fit_freeze_phase72b_models

    _install_phase72b_fast_pipeline(monkeypatch)
    inputs = _phase72b_prepare_fixture(tmp_path)
    package = prepare_phase72b_information_gain_screen(
        protocol_path=inputs["protocol_path"],
        phase72a_region_config=inputs["region_config"],
        phase72a_package_dir=inputs["phase72a_dir"],
        embedding_dirs=inputs["embedding_dirs"],
        label_dirs=inputs["label_dirs"],
        terrain_dir=inputs["terrain_dir"],
    )
    prepared_dir = tmp_path / "prepared"
    write_phase72b_prepared_artifacts(package, prepared_dir)
    frozen_dir = tmp_path / "frozen"
    fit_freeze_phase72b_models(
        prepared_dir=prepared_dir, output_dir=frozen_dir
    )
    return inputs, prepared_dir, frozen_dir


def _resign_selected_and_fit_progress(
    frozen_dir: Path, selected: dict
) -> None:
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )

    selected_path = frozen_dir / "phase72b_selected_models.json"
    _, selected_hash_path = write_hashed_json(selected_path, selected)
    progress_path = frozen_dir / "phase72b_fit_progress.json"
    progress = load_hashed_json(progress_path)
    progress["selected_models_sha256"] = selected_hash_path.read_text(
        encoding="ascii"
    ).strip()
    write_hashed_json(progress_path, progress)


def _prepare_only(tmp_path: Path):
    from paper11_geofm.phase72b_information_gain_screen import (
        prepare_phase72b_information_gain_screen,
        write_phase72b_prepared_artifacts,
    )

    inputs = _phase72b_prepare_fixture(tmp_path)
    package = prepare_phase72b_information_gain_screen(
        protocol_path=inputs["protocol_path"],
        phase72a_region_config=inputs["region_config"],
        phase72a_package_dir=inputs["phase72a_dir"],
        embedding_dirs=inputs["embedding_dirs"],
        label_dirs=inputs["label_dirs"],
        terrain_dir=inputs["terrain_dir"],
    )
    prepared_dir = tmp_path / "prepared"
    write_phase72b_prepared_artifacts(package, prepared_dir)
    return inputs, prepared_dir


def test_phase72b_prepared_loader_rejects_downgraded_terrain_manifest(
    tmp_path,
):
    from paper11_geofm.phase72b_prepared import (
        load_verified_phase72b_prepared,
    )
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )
    from paper11_geofm.phase72b_terrain import _file_sha256

    _, prepared_dir = _prepare_only(tmp_path)
    terrain_path = prepared_dir / "phase72b_terrain_manifest.csv"
    terrain_rows = pd.read_csv(terrain_path, keep_default_na=False)
    terrain_rows[["region_id", "path", "shape", "sha256"]].to_csv(
        terrain_path, index=False
    )
    manifest_path = prepared_dir / "phase72b_prepared_artifacts.json"
    manifest = load_hashed_json(manifest_path)
    for record in manifest["artifacts"]:
        if record["name"] == terrain_path.name:
            record["sha256"] = _file_sha256(terrain_path)
    write_hashed_json(manifest_path, manifest)

    try:
        load_verified_phase72b_prepared(prepared_dir)
    except ValueError as exc:
        assert "terrain manifest" in str(exc).lower()
    else:
        raise AssertionError("Expected terrain provenance downgrade rejection")


def test_phase72b_prepared_loader_rejects_prepare_time_control_matrix(
    tmp_path,
):
    from paper11_geofm.phase72b_prepared import (
        load_verified_phase72b_prepared,
    )
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )
    from paper11_geofm.phase72b_terrain import _file_sha256

    _, prepared_dir = _prepare_only(tmp_path)
    feature_path = prepared_dir / "phase72b_feature_manifest.csv"
    feature_rows = pd.read_csv(feature_path, keep_default_na=False)
    feature_rows.loc[0, "control_id"] = "spatial_shuffle"
    feature_rows.loc[0, "partition_id"] = "pooled_temporal:train"
    feature_rows.to_csv(feature_path, index=False)
    manifest_path = prepared_dir / "phase72b_prepared_artifacts.json"
    manifest = load_hashed_json(manifest_path)
    for record in manifest["artifacts"]:
        if record["name"] == feature_path.name:
            record["sha256"] = _file_sha256(feature_path)
    write_hashed_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="feature manifest"):
        load_verified_phase72b_prepared(prepared_dir)


def test_phase72b_fit_freeze_rejects_tampered_prepared_matrix(tmp_path):
    from paper11_geofm.phase72b_models import fit_freeze_phase72b_models

    _, prepared_dir = _prepare_only(tmp_path)
    matrix_path = prepared_dir / "phase72b_feature_matrices.npz"
    with np.load(matrix_path) as loaded:
        matrices = {name: loaded[name].copy() for name in loaded.files}
    matrices["explicit_static"][0, 0] += 1.0
    np.savez_compressed(matrix_path, **matrices)
    try:
        fit_freeze_phase72b_models(
            prepared_dir=prepared_dir, output_dir=tmp_path / "frozen"
        )
    except ValueError as exc:
        assert "prepared artifact hash mismatch" in str(exc).lower()
    else:
        raise AssertionError("Expected tampered prepared matrix rejection")


def test_phase72b_fit_freeze_rejects_repacked_development_target(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_models import fit_freeze_phase72b_models

    _install_phase72b_fast_pipeline(monkeypatch)
    _, prepared_dir = _prepare_only(tmp_path)
    target_path = prepared_dir / "phase72b_development_targets.npz"
    target_path.write_bytes(target_path.read_bytes() + b"repacked")

    with pytest.raises(ValueError, match="prepared artifact hash mismatch"):
        fit_freeze_phase72b_models(
            prepared_dir=prepared_dir, output_dir=tmp_path / "frozen"
        )


def test_phase72b_fit_freeze_rejects_tampered_split_registry(tmp_path):
    from paper11_geofm.phase72b_models import fit_freeze_phase72b_models

    _, prepared_dir = _prepare_only(tmp_path)
    split_path = prepared_dir / "phase72b_split_registry.json"
    split_registry = json.loads(split_path.read_text(encoding="utf-8"))
    split_registry["pooled_temporal"]["train"] = split_registry[
        "pooled_temporal"
    ]["train"][1:]
    split_path.write_text(json.dumps(split_registry), encoding="utf-8")
    try:
        fit_freeze_phase72b_models(
            prepared_dir=prepared_dir, output_dir=tmp_path / "frozen"
        )
    except ValueError as exc:
        assert "prepared artifact hash mismatch" in str(exc).lower()
    else:
        raise AssertionError("Expected tampered split registry rejection")


def test_phase72b_fit_and_confirm_reject_resigned_protocol_mutation(tmp_path):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_models import fit_freeze_phase72b_models
    from paper11_geofm.phase72b_prepared import (
        load_verified_phase72b_prepared,
    )
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )
    from paper11_geofm.phase72b_terrain import _file_sha256

    _, prepared_dir = _prepare_only(tmp_path)
    protocol_path = prepared_dir / "phase72b_frozen_protocol.json"
    protocol_hash_path = prepared_dir / "phase72b_frozen_protocol.sha256"
    frozen_protocol = load_hashed_json(protocol_path, protocol_hash_path)
    frozen_protocol["tracked_protocol"]["bootstrap"]["iterations"] = 100
    write_hashed_json(protocol_path, frozen_protocol)

    manifest_path = prepared_dir / "phase72b_prepared_artifacts.json"
    manifest = load_hashed_json(manifest_path)
    manifest["frozen_protocol_sha256"] = protocol_hash_path.read_text(
        encoding="ascii"
    ).strip()
    for record in manifest["artifacts"]:
        if record["name"] in {
            protocol_path.name,
            protocol_hash_path.name,
        }:
            record["sha256"] = _file_sha256(
                prepared_dir / record["name"]
            )
    write_hashed_json(manifest_path, manifest)

    for operation in (
        lambda: load_verified_phase72b_prepared(prepared_dir),
        lambda: fit_freeze_phase72b_models(
            prepared_dir=prepared_dir, output_dir=tmp_path / "frozen"
        ),
        lambda: confirm_phase72b_information_gain_screen(
            prepared_dir=prepared_dir, frozen_dir=tmp_path / "missing_frozen"
        ),
    ):
        with pytest.raises(ValueError, match="bootstrap"):
            operation()


def test_phase72b_confirmation_rejects_tampered_feature_rows(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    row_path = prepared_dir / "phase72b_feature_rows.csv"
    rows = pd.read_csv(row_path, keep_default_na=False)
    rows.loc[0, "spatial_block_id"] = "bishan_br999_bc999"
    rows.to_csv(row_path, index=False)
    try:
        confirm_phase72b_information_gain_screen(
            prepared_dir=prepared_dir, frozen_dir=frozen_dir
        )
    except ValueError as exc:
        assert "prepared artifact hash mismatch" in str(exc).lower()
    else:
        raise AssertionError("Expected tampered feature-row rejection")


def test_phase72b_confirmation_rejects_repacked_confirmation_target(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    target_path = prepared_dir / "phase72b_confirmation_targets.npz"
    target_path.write_bytes(target_path.read_bytes() + b"repacked")

    with pytest.raises(ValueError, match="prepared artifact hash mismatch"):
        confirm_phase72b_information_gain_screen(
            prepared_dir=prepared_dir, frozen_dir=frozen_dir
        )


def test_phase72b_confirmation_rejects_repacked_development_target(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    target_path = prepared_dir / "phase72b_development_targets.npz"
    target_path.write_bytes(target_path.read_bytes() + b"repacked")

    with pytest.raises(ValueError, match="prepared artifact hash mismatch"):
        confirm_phase72b_information_gain_screen(
            prepared_dir=prepared_dir, frozen_dir=frozen_dir
        )


def test_phase72b_confirmation_rejects_tampered_leakage_audit(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    audit_path = prepared_dir / "phase72b_leakage_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["invalid_spatial_axes"] = ["spatial_bishan_fold0"]
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    try:
        confirm_phase72b_information_gain_screen(
            prepared_dir=prepared_dir, frozen_dir=frozen_dir
        )
    except ValueError as exc:
        assert "prepared artifact hash mismatch" in str(exc).lower()
    else:
        raise AssertionError("Expected tampered leakage-audit rejection")


def test_phase72b_confirmation_audits_fit_control_manifest_before_targets(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    manifest_path = frozen_dir / "phase72b_fit_control_manifest.csv"
    rows = pd.read_csv(manifest_path, keep_default_na=False)
    rows.loc[0, "partition_id"] = "pooled_temporal:test"
    rows.to_csv(manifest_path, index=False)
    target_path = prepared_dir / "phase72b_confirmation_targets.npz"
    with np.load(target_path) as loaded:
        targets = {name: loaded[name].copy() for name in loaded.files}
    targets["conversion_1y"][0] = 1 - targets["conversion_1y"][0]
    np.savez_compressed(target_path, **targets)

    result = confirm_phase72b_information_gain_screen(
        prepared_dir=prepared_dir, frozen_dir=frozen_dir
    )

    assert result["phase72b_status"] == "phase72b_inputs_not_ready"
    assert any("fit control manifest" in row for row in result["blockers"])


def test_phase72b_confirmation_recomputes_fit_control_matrix_hash(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )
    from paper11_geofm.phase72b_terrain import _file_sha256

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    manifest_path = frozen_dir / "phase72b_fit_control_manifest.csv"
    rows = pd.read_csv(manifest_path, keep_default_na=False)
    rows.loc[0, "matrix_sha256"] = "0" * 64
    rows.to_csv(manifest_path, index=False)
    selected_path = frozen_dir / "phase72b_selected_models.json"
    selected = load_hashed_json(selected_path)
    selected["fit_control_manifest_sha256"] = _file_sha256(manifest_path)
    _resign_selected_and_fit_progress(frozen_dir, selected)

    result = confirm_phase72b_information_gain_screen(
        prepared_dir=prepared_dir, frozen_dir=frozen_dir
    )

    assert result["phase72b_status"] == "phase72b_inputs_not_ready"
    assert any(
        "fit control manifest matrix mismatch" in blocker
        for blocker in result["blockers"]
    )


def test_phase72b_confirmation_blocks_missing_fit_control_manifest(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    (frozen_dir / "phase72b_fit_control_manifest.csv").unlink()

    result = confirm_phase72b_information_gain_screen(
        prepared_dir=prepared_dir, frozen_dir=frozen_dir
    )

    assert result["phase72b_status"] == "phase72b_inputs_not_ready"
    assert "fit control manifest is missing" in result["blockers"]


def test_phase72b_confirmation_rejects_resigned_bundle_partition_identity(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )
    from paper11_geofm.phase72b_terrain import _file_sha256

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    selected_path = frozen_dir / "phase72b_selected_models.json"
    selected = load_hashed_json(selected_path)
    record = selected["bundle_records"][0]
    bundle_path = frozen_dir / record["bundle_path"]
    bundle = joblib.load(bundle_path)
    bundle["train_index_sha256"] = "0" * 64
    joblib.dump(bundle, bundle_path)
    record["bundle_sha256"] = _file_sha256(bundle_path)
    _, selected_hash_path = write_hashed_json(selected_path, selected)
    progress_path = frozen_dir / "phase72b_fit_progress.json"
    progress = load_hashed_json(progress_path)
    filename = Path(record["bundle_path"]).name
    progress["entries"][filename]["record"] = dict(record)
    progress["selected_models_sha256"] = selected_hash_path.read_text(
        encoding="ascii"
    ).strip()
    write_hashed_json(progress_path, progress)
    target_path = prepared_dir / "phase72b_confirmation_targets.npz"
    with np.load(target_path) as loaded:
        targets = {name: loaded[name].copy() for name in loaded.files}
    targets["conversion_1y"][0] = 1 - targets["conversion_1y"][0]
    np.savez_compressed(target_path, **targets)

    with pytest.raises(ValueError, match="partition identity"):
        confirm_phase72b_information_gain_screen(
            prepared_dir=prepared_dir, frozen_dir=frozen_dir
        )


def test_phase72b_confirmation_binds_selected_hash_to_fit_progress(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    selected_path = frozen_dir / "phase72b_selected_models.json"
    selected = load_hashed_json(selected_path)
    selected["resigned_after_fit"] = True
    write_hashed_json(selected_path, selected)

    with pytest.raises(ValueError, match="fit progress selected-model hash"):
        confirm_phase72b_information_gain_screen(
            prepared_dir=prepared_dir, frozen_dir=frozen_dir
        )


def test_phase72b_confirmation_rejects_resigned_empty_fit_progress_entries(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    progress_path = frozen_dir / "phase72b_fit_progress.json"
    progress = load_hashed_json(progress_path)
    progress["entries"] = {}
    write_hashed_json(progress_path, progress)

    with pytest.raises(ValueError, match="fit progress entries mismatch"):
        confirm_phase72b_information_gain_screen(
            prepared_dir=prepared_dir, frozen_dir=frozen_dir
        )


def test_phase72b_confirmation_rejects_selected_record_not_in_fit_progress(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_protocol import load_hashed_json
    from paper11_geofm.phase72b_terrain import _file_sha256

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    selected_path = frozen_dir / "phase72b_selected_models.json"
    selected = load_hashed_json(selected_path)
    record = selected["bundle_records"][0]
    bundle_path = frozen_dir / record["bundle_path"]
    bundle = joblib.load(bundle_path)
    changed_ap = min(
        1.0,
        float(bundle["validation_metrics"]["average_precision"]) + 0.01,
    )
    bundle["validation_metrics"]["average_precision"] = changed_ap
    joblib.dump(bundle, bundle_path)
    record["validation_average_precision"] = changed_ap
    record["bundle_sha256"] = _file_sha256(bundle_path)
    _resign_selected_and_fit_progress(frozen_dir, selected)

    with pytest.raises(ValueError, match="fit progress entries mismatch"):
        confirm_phase72b_information_gain_screen(
            prepared_dir=prepared_dir, frozen_dir=frozen_dir
        )


def test_phase72b_confirmation_rejects_invalid_fit_progress_validation_row(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    progress_path = frozen_dir / "phase72b_fit_progress.json"
    progress = load_hashed_json(progress_path)
    first_entry = next(iter(progress["entries"].values()))
    first_entry["validation_rows"][0]["roc_auc"] = float("inf")
    write_hashed_json(progress_path, progress)

    with pytest.raises(ValueError, match="fit progress entries mismatch"):
        confirm_phase72b_information_gain_screen(
            prepared_dir=prepared_dir, frozen_dir=frozen_dir
        )


def test_phase72b_confirmation_rejects_duplicate_fit_progress_history(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    progress_path = frozen_dir / "phase72b_fit_progress.json"
    progress = load_hashed_json(progress_path)
    first_entry = next(iter(progress["entries"].values()))
    record = first_entry["record"]
    selected_row = next(
        row
        for row in first_entry["validation_rows"]
        if row["candidate_id"] == record["candidate_id"]
        and row["calibration_method"] == record["calibration_method"]
    )
    row_count = len(first_entry["validation_rows"])
    first_entry["validation_rows"] = [
        dict(selected_row) for _ in range(row_count)
    ]
    write_hashed_json(progress_path, progress)

    with pytest.raises(ValueError, match="fit progress entries mismatch"):
        confirm_phase72b_information_gain_screen(
            prepared_dir=prepared_dir, frozen_dir=frozen_dir
        )


def test_phase72b_confirmation_binds_all_bundle_metrics_to_fit_progress(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )
    from paper11_geofm.phase72b_terrain import _file_sha256

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    selected_path = frozen_dir / "phase72b_selected_models.json"
    selected = load_hashed_json(selected_path)
    record = selected["bundle_records"][0]
    bundle_path = frozen_dir / record["bundle_path"]
    bundle = joblib.load(bundle_path)
    original_auc = float(bundle["validation_metrics"]["roc_auc"])
    bundle["validation_metrics"]["roc_auc"] = (
        0.0 if original_auc > 0.5 else 1.0
    )
    joblib.dump(bundle, bundle_path)
    record["bundle_sha256"] = _file_sha256(bundle_path)
    _, selected_hash_path = write_hashed_json(selected_path, selected)

    progress_path = frozen_dir / "phase72b_fit_progress.json"
    progress = load_hashed_json(progress_path)
    filename = Path(record["bundle_path"]).name
    progress["entries"][filename]["record"] = dict(record)
    progress["selected_models_sha256"] = selected_hash_path.read_text(
        encoding="ascii"
    ).strip()
    write_hashed_json(progress_path, progress)

    with pytest.raises(ValueError, match="fit progress entries mismatch"):
        confirm_phase72b_information_gain_screen(
            prepared_dir=prepared_dir, frozen_dir=frozen_dir
        )


def test_phase72b_confirmation_rejects_resigned_weaker_control_seed(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    selected_path = frozen_dir / "phase72b_selected_models.json"
    selected = load_hashed_json(selected_path)
    variant_id = "explicit_plus_temporal_order_shuffle"
    original_seed = int(
        selected["selected_control_seeds"]["pooled_temporal"][variant_id]
    )
    replacement_seed = next(
        seed for seed in _protocol_payload()["controls"]["seeds"]
        if int(seed) != original_seed
    )
    selected["selected_control_seeds"]["pooled_temporal"][variant_id] = int(
        replacement_seed
    )
    _resign_selected_and_fit_progress(frozen_dir, selected)

    result = confirm_phase72b_information_gain_screen(
        prepared_dir=prepared_dir, frozen_dir=frozen_dir
    )

    assert result["phase72b_status"] == "phase72b_inputs_not_ready"
    assert any(
        "selected control seed mismatch" in blocker
        for blocker in result["blockers"]
    )


@pytest.mark.parametrize(
    ("axis_id", "mutation"),
    [
        ("pooled_temporal", lambda seed: float(seed) + 0.9),
        ("bishan_to_dongxing", lambda seed: "bad"),
    ],
    ids=("fractional-pooled", "nonnumeric-transfer"),
)
def test_phase72b_confirmation_rejects_noninteger_selected_control_seed(
    tmp_path, monkeypatch, axis_id, mutation
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_protocol import load_hashed_json

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    selected_path = frozen_dir / "phase72b_selected_models.json"
    selected = load_hashed_json(selected_path)
    variant_id = "explicit_plus_temporal_order_shuffle"
    original_seed = selected["selected_control_seeds"][axis_id][variant_id]
    selected["selected_control_seeds"][axis_id][variant_id] = mutation(
        original_seed
    )
    _resign_selected_and_fit_progress(frozen_dir, selected)

    result = confirm_phase72b_information_gain_screen(
        prepared_dir=prepared_dir, frozen_dir=frozen_dir
    )

    assert result["phase72b_status"] == "phase72b_inputs_not_ready"
    assert any(
        "selected control seed invalid" in blocker
        for blocker in result["blockers"]
    )


def test_phase72b_confirmation_rejects_resigned_axis_bundle_paths(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    selected_path = frozen_dir / "phase72b_selected_models.json"
    selected = load_hashed_json(selected_path)
    selected["axes"]["pooled_temporal"] = []
    _resign_selected_and_fit_progress(frozen_dir, selected)

    result = confirm_phase72b_information_gain_screen(
        prepared_dir=prepared_dir, frozen_dir=frozen_dir
    )

    assert result["phase72b_status"] == "phase72b_inputs_not_ready"
    assert "selected axis bundle paths mismatch" in result["blockers"]


def test_phase72b_confirmation_rejects_coordinated_missing_core_bundle(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    selected_path = frozen_dir / "phase72b_selected_models.json"
    selected = load_hashed_json(selected_path)
    removed = next(
        record
        for record in selected["bundle_records"]
        if record["axis_id"] == "pooled_temporal"
        and record["variant_id"] == "geofm_current_only"
    )
    selected["bundle_records"].remove(removed)
    selected["axes"]["pooled_temporal"].remove(removed["bundle_path"])
    progress_path = frozen_dir / "phase72b_fit_progress.json"
    progress = load_hashed_json(progress_path)
    removed_entry = progress["entries"].pop(
        Path(removed["bundle_path"]).name
    )
    selected["validation_metric_rows"] -= len(
        removed_entry["validation_rows"]
    )
    _, selected_hash_path = write_hashed_json(selected_path, selected)
    progress["selected_models_sha256"] = selected_hash_path.read_text(
        encoding="ascii"
    ).strip()
    write_hashed_json(progress_path, progress)

    result = confirm_phase72b_information_gain_screen(
        prepared_dir=prepared_dir, frozen_dir=frozen_dir
    )

    assert result["phase72b_status"] == "phase72b_inputs_not_ready"
    assert "selected bundle set mismatch" in result["blockers"]


def test_phase72b_confirmation_rejects_coordinated_noncanonical_bundle_path(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    selected_path = frozen_dir / "phase72b_selected_models.json"
    selected = load_hashed_json(selected_path)
    record = selected["bundle_records"][0]
    original_relative = Path(record["bundle_path"])
    renamed_relative = original_relative.with_name(
        f"renamed_{original_relative.name}"
    )
    shutil.copy2(
        frozen_dir / original_relative,
        frozen_dir / renamed_relative,
    )
    axis_paths = selected["axes"][record["axis_id"]]
    axis_paths[axis_paths.index(record["bundle_path"])] = str(
        renamed_relative
    )
    record["bundle_path"] = str(renamed_relative)
    _, selected_hash_path = write_hashed_json(selected_path, selected)

    progress_path = frozen_dir / "phase72b_fit_progress.json"
    progress = load_hashed_json(progress_path)
    entry = progress["entries"].pop(original_relative.name)
    entry["record"] = dict(record)
    progress["entries"][renamed_relative.name] = entry
    progress["selected_models_sha256"] = selected_hash_path.read_text(
        encoding="ascii"
    ).strip()
    write_hashed_json(progress_path, progress)

    result = confirm_phase72b_information_gain_screen(
        prepared_dir=prepared_dir, frozen_dir=frozen_dir
    )

    assert result["phase72b_status"] == "phase72b_inputs_not_ready"
    assert any(
        "bundle path mismatch" in blocker for blocker in result["blockers"]
    )


def test_phase72b_confirmation_controls_only_read_axis_test_rows(
    tmp_path, monkeypatch
):
    import paper11_geofm.phase72b_information_gain_screen as screen

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    split_registry = json.loads(
        (prepared_dir / "phase72b_split_registry.json").read_text(
            encoding="utf-8"
        )
    )
    expected_indexes = {
        f"{axis_id}:{partition_name}": [
            int(value) for value in axis[partition_name]
        ]
        for axis_id, axis in split_registry.items()
        for partition_name in ("train", "validation", "test")
    }
    original = screen.build_phase72b_control_features
    calls = []

    def guarded(*args, **kwargs):
        partitions = list(kwargs["partition_ids"])
        sample_indexes = [int(row["sample_index"]) for row in args[3]]
        assert len(set(partitions)) == 1
        partition_id = partitions[0]
        assert sample_indexes == expected_indexes[partition_id]
        calls.append((partition_id, sample_indexes))
        return original(*args, **kwargs)

    monkeypatch.setattr(screen, "build_phase72b_control_features", guarded)
    screen.confirm_phase72b_information_gain_screen(
        prepared_dir=prepared_dir, frozen_dir=frozen_dir
    )
    selected = json.loads(
        (frozen_dir / "phase72b_selected_models.json").read_text(
            encoding="utf-8"
        )
    )
    control_variants = {
        "explicit_plus_temporal_order_shuffle",
        "explicit_plus_spatial_shuffle",
        "explicit_plus_random_projection",
    }
    expected_test_partitions = sorted(
        f"{record['axis_id']}:test"
        for record in selected["bundle_records"]
        if record["variant_id"] in control_variants
        and split_registry[record["axis_id"]]["test"]
    )
    actual_test_partitions = sorted(
        partition_id
        for partition_id, _ in calls
        if partition_id.endswith(":test")
    )
    assert actual_test_partitions == expected_test_partitions


def test_phase72b_fit_and_confirmation_bind_prepared_manifest(
    tmp_path, monkeypatch
):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_models import fit_freeze_phase72b_models
    from paper11_geofm.phase72b_protocol import (
        load_hashed_json,
        write_hashed_json,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    prepared_hash = (
        prepared_dir / "phase72b_prepared_artifacts.sha256"
    ).read_text(encoding="ascii").strip()
    progress = load_hashed_json(
        frozen_dir / "phase72b_fit_progress.json",
        frozen_dir / "phase72b_fit_progress.sha256",
    )
    selected = load_hashed_json(
        frozen_dir / "phase72b_selected_models.json",
        frozen_dir / "phase72b_selected_models.sha256",
    )
    assert progress["prepared_artifacts_sha256"] == prepared_hash
    assert selected["prepared_artifacts_sha256"] == prepared_hash

    manifest_path = prepared_dir / "phase72b_prepared_artifacts.json"
    manifest = load_hashed_json(manifest_path)
    manifest["integrity_revision"] = "changed"
    write_hashed_json(manifest_path, manifest)
    for operation in (
        lambda: fit_freeze_phase72b_models(
            prepared_dir=prepared_dir, output_dir=frozen_dir
        ),
        lambda: confirm_phase72b_information_gain_screen(
            prepared_dir=prepared_dir, frozen_dir=frozen_dir
        ),
    ):
        try:
            operation()
        except ValueError as exc:
            assert "prepared artifact manifest hash mismatch" in str(exc).lower()
        else:
            raise AssertionError("Expected prepared-manifest binding rejection")


def test_phase72b_confirmation_writes_stable_outputs(tmp_path, monkeypatch):
    from paper11_geofm.phase72b_information_gain_screen import (
        _array_sha256,
        confirm_phase72b_information_gain_screen,
        write_phase72b_confirmation_artifacts,
    )
    from paper11_geofm.phase72b_terrain import _file_sha256

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path, monkeypatch)
    result = confirm_phase72b_information_gain_screen(
        prepared_dir=prepared_dir, frozen_dir=frozen_dir
    )
    paths = write_phase72b_confirmation_artifacts(
        result, tmp_path / "confirm"
    )
    assert set(paths) == {
        "metrics_csv",
        "predictions_csv",
        "calibration_csv",
        "bootstrap_csv",
        "control_csv",
        "confirmation_control_csv",
        "transfer_csv",
        "screen_json",
        "screen_md",
        "receipt_json",
        "receipt_hash",
    }
    confirmation_control = pd.read_csv(
        paths["confirmation_control_csv"], keep_default_na=False
    )
    assert set(confirmation_control.columns) == {
        "axis_id",
        "partition_id",
        "control_id",
        "seed",
        "index_sha256",
        "matrix_sha256",
        "cross_partition_count",
    }
    assert len(confirmation_control) == len(
        result["confirmation_control_rows"]
    )
    assert len(confirmation_control) > 0
    assert (confirmation_control["cross_partition_count"] == 0).all()
    control_ids = {
        "explicit_plus_temporal_order_shuffle": "temporal_order_shuffle",
        "explicit_plus_spatial_shuffle": "spatial_shuffle",
        "explicit_plus_random_projection": "random_projection",
    }
    actual_keys = {
        (row.axis_id, row.control_id, int(row.seed))
        for row in confirmation_control.itertuples(index=False)
    }
    expected_keys = {
        (
            str(row["axis_id"]),
            control_ids[str(row["variant_id"])],
            int(row["control_seed"]),
        )
        for row in result["metrics_rows"]
        if str(row["variant_id"]) in control_ids
    }
    assert len(confirmation_control) == len(actual_keys)
    assert actual_keys == expected_keys
    split_registry = json.loads(
        (prepared_dir / "phase72b_split_registry.json").read_text(
            encoding="utf-8"
        )
    )
    for row in confirmation_control.itertuples(index=False):
        assert row.partition_id == f"{row.axis_id}:test"
        indexes = np.asarray(split_registry[row.axis_id]["test"], np.int64)
        assert row.index_sha256 == _array_sha256(indexes)
        assert len(row.matrix_sha256) == 64
        assert set(row.matrix_sha256.lower()) <= set("0123456789abcdef")
    receipt = json.loads(paths["receipt_json"].read_text(encoding="utf-8"))
    assert "phase72b_confirmation_control_manifest.csv" in {
        row["name"] for row in receipt["artifacts"]
    }
    for artifact in receipt["artifacts"]:
        assert artifact["sha256"] == _file_sha256(
            paths["receipt_json"].parent / artifact["name"]
        )
    assert result["phase72b_status"] in {
        "phase72b_inputs_not_ready",
        "geofm_information_not_supported",
        "geofm_information_mixed",
        "geofm_information_supported",
    }
    try:
        write_phase72b_confirmation_artifacts(result, tmp_path / "confirm")
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected confirmation output reuse rejection")
    target_path = prepared_dir / "phase72b_confirmation_targets.npz"
    with np.load(target_path) as loaded:
        changed = {name: loaded[name].copy() for name in loaded.files}
    changed["conversion_1y"][0] = 1 - changed["conversion_1y"][0]
    np.savez_compressed(target_path, **changed)
    try:
        confirm_phase72b_information_gain_screen(
            prepared_dir=prepared_dir, frozen_dir=frozen_dir
        )
    except ValueError as exc:
        assert "prepared artifact hash mismatch" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected modified confirmation targets to be rejected"
        )


def test_phase72b_runner_executes_modes_and_rejects_changed_manifest(
    tmp_path, monkeypatch
):
    inputs = _phase72b_prepare_fixture(tmp_path)
    script = (
        ROOT
        / "experiments"
        / "phase72b_geofm_information_gain_screen"
        / "run_phase72b_information_gain_screen.py"
    )
    prepared_dir = tmp_path / "cli_prepared"
    frozen_dir = tmp_path / "cli_frozen"
    confirm_dir = tmp_path / "cli_confirm"
    prepare_command = [
        sys.executable,
        str(script),
        "--mode",
        "prepare",
        "--protocol",
        str(inputs["protocol_path"]),
        "--phase72a-region-config",
        str(inputs["region_config"]),
        "--phase72a-package-dir",
        str(inputs["phase72a_dir"]),
        "--terrain-dir",
        str(inputs["terrain_dir"]),
        "--output-dir",
        str(prepared_dir),
    ]
    for region_id in ("bishan", "dongxing"):
        prepare_command.extend(
            [
                "--embedding-dir",
                f"{region_id}={inputs['embedding_dirs'][region_id]}",
                "--label-dir",
                f"{region_id}={inputs['label_dirs'][region_id]}",
            ]
        )
    prepare = subprocess.run(
        prepare_command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert prepare.returncode == 0, prepare.stderr
    _install_phase72b_fast_pipeline(monkeypatch)
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
    )
    from paper11_geofm.phase72b_models import fit_freeze_phase72b_models

    fit_freeze_phase72b_models(
        prepared_dir=prepared_dir, output_dir=frozen_dir
    )
    result = confirm_phase72b_information_gain_screen(
        prepared_dir=prepared_dir, frozen_dir=frozen_dir
    )
    assert result["phase72b_status"] == "phase72b_inputs_not_ready"

    spec = importlib.util.spec_from_file_location("phase72b_runner", script)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)
    monkeypatch.setattr(
        runner, "confirm_phase72b_information_gain_screen", lambda **_: result
    )
    monkeypatch.setattr(
        runner, "write_phase72b_confirmation_artifacts", lambda *_: {}
    )
    assert runner.main(
        [
            "--mode",
            "confirm",
            "--prepared-dir",
            str(prepared_dir),
            "--frozen-dir",
            str(frozen_dir),
            "--output-dir",
            str(confirm_dir),
        ]
    ) == 1
    selected_path = frozen_dir / "phase72b_selected_models.json"
    changed = json.loads(selected_path.read_text(encoding="utf-8"))
    changed["changed_after_freeze"] = True
    selected_path.write_text(json.dumps(changed), encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "confirm",
            "--prepared-dir",
            str(prepared_dir),
            "--frozen-dir",
            str(frozen_dir),
            "--output-dir",
            str(tmp_path / "rejected"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 1
    assert "hash mismatch" in rejected.stderr.lower()


def test_phase72b_spatial_coverage_requires_every_expected_axis():
    from paper11_geofm.phase72b_information_gain_screen import (
        _audit_phase72b_spatial_confirmation_coverage,
    )

    groups = {
        ("spatial_bishan_fold0", "explicit_history", None): {
            "outcome": np.asarray([0, 1])
        },
        (
            "spatial_bishan_fold0",
            "explicit_plus_geofm_temporal_full",
            None,
        ): {"outcome": np.asarray([0, 1])},
        ("spatial_bishan_fold1", "explicit_history", None): {
            "outcome": np.asarray([1, 1])
        },
        (
            "spatial_bishan_fold1",
            "explicit_plus_geofm_temporal_full",
            None,
        ): {"outcome": np.asarray([1, 1])},
    }
    valid_axes, blockers = _audit_phase72b_spatial_confirmation_coverage(
        ["spatial_bishan_fold0", "spatial_bishan_fold1"], groups
    )
    assert valid_axes == ["spatial_bishan_fold0"]
    assert blockers == [
        "incomplete spatial confirmation coverage: spatial_bishan_fold1"
    ]
