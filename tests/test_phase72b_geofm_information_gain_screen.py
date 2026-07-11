import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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
        assert "hash" in str(exc).lower()
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
    assert audit["status"] == "terrain_inputs_ready"
    assert audit["rows"][0]["shape"] == "2x3"
    assert len(audit["rows"][0]["sha256"]) == 64


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
    assert "shape" in " ".join(audit["errors"]).lower()


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
        {"region_id": "a", "origin_year": 2019},
        {"region_id": "a", "origin_year": 2019},
        {"region_id": "b", "origin_year": 2018},
        {"region_id": "b", "origin_year": 2018},
    ]
    return histories, masks, rows


def test_phase72b_geofm_temporal_summary_and_controls_are_deterministic():
    from paper11_geofm.phase72b_geofm_features import (
        build_phase72b_control_features,
        build_phase72b_geofm_features,
    )

    histories, masks, rows = _geofm_control_fixture()
    features = build_phase72b_geofm_features(histories, masks)
    assert features["geofm_current"][0].tolist() == [4.0, 8.0]
    assert features["geofm_temporal_full"].shape == (4, 10)
    first = build_phase72b_control_features(
        "spatial_shuffle",
        histories,
        masks,
        rows,
        seed=72,
        output_dim=10,
    )
    second = build_phase72b_control_features(
        "spatial_shuffle",
        histories,
        masks,
        rows,
        seed=72,
        output_dim=10,
    )
    assert np.array_equal(first, second)
    assert sorted(map(tuple, first[:, :2])) == sorted(
        map(tuple, features["geofm_current"])
    )


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
        [{"region_id": "a", "origin_year": 2020}],
        seed=72,
        output_dim=5,
    )
    original = build_phase72b_geofm_features(histories, masks)[
        "geofm_temporal_full"
    ]
    assert control[0, 0] == 4.0
    assert not np.array_equal(control, original)


def test_phase72b_random_projection_is_seeded_and_same_dimension():
    from paper11_geofm.phase72b_geofm_features import (
        build_phase72b_control_features,
    )

    histories, masks, rows = _geofm_control_fixture()
    first = build_phase72b_control_features(
        "random_projection",
        histories,
        masks,
        rows,
        seed=72,
        output_dim=5,
    )
    second = build_phase72b_control_features(
        "random_projection",
        histories,
        masks,
        rows,
        seed=72,
        output_dim=5,
    )
    third = build_phase72b_control_features(
        "random_projection",
        histories,
        masks,
        rows,
        seed=73,
        output_dim=5,
    )
    assert first.shape == (4, 5)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_phase72b_splits_lock_years_regions_and_spatial_buffers():
    from paper11_geofm.phase72b_splits import build_phase72b_split_registry

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
                        "conversion_1y": (br + bc + year) % 2,
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
                "embedding_dim": 2,
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
            embedding = np.zeros((2, 3, 2), dtype=np.float32)
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
        np.savez_compressed(
            terrain_dir / f"{region_id}_terrain.npz", **terrain
        )
    region_config = tmp_path / "regions.json"
    region_config.write_text(
        json.dumps(regions_payload), encoding="utf-8"
    )
    phase72a = build_phase72a_temporal_label_package(
        region_config=region_config,
        embedding_dirs=embedding_dirs,
        label_dirs=label_dirs,
        manual_review_per_stratum=1,
        spatial_block_size=1,
    )
    phase72a_dir = tmp_path / "phase72a"
    write_phase72a_temporal_label_package_artifacts(phase72a, phase72a_dir)
    fixture_protocol = _protocol_payload()
    fixture_protocol["models"] = {
        "logistic_c": [0.1],
        "logistic_class_weight": ["none"],
        "hgb_learning_rate": [0.08],
        "hgb_max_leaf_nodes": [15],
        "hgb_min_samples_leaf": [2],
        "hgb_max_iter": 20,
        "hgb_l2_regularization": [0.0],
    }
    fixture_protocol["calibration"] = {
        "methods": ["none", "sigmoid"],
        "ece_bins": 4,
    }
    fixture_protocol["controls"]["random_projection_dim"] = 10
    fixture_protocol["bootstrap"] = {"iterations": 100, "seed": 72}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(fixture_protocol), encoding="utf-8"
    )
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
    from paper11_geofm.phase72b_protocol import load_hashed_json

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
            {"control_id": name, "ap_delta": 0.006, "brier_delta": 0.003}
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
            },
            {
                "axis_id": "dongxing_to_bishan",
                "ap_delta": 0.0,
                "brier_delta": 0.003,
            },
        ],
        "spatial_rows": [
            {"region_id": "bishan", "ap_delta": 0.001, "brier_delta": 0.0},
            {"region_id": "dongxing", "ap_delta": 0.0, "brier_delta": 0.001},
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
        assert "development target hash mismatch" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected modified development targets to be rejected"
        )
    np.savez_compressed(target_path, **original_targets)
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
    load_phase72b_model_bundle(bundle_path, record["bundle_sha256"])
    original = bundle_path.read_bytes()
    bundle_path.write_bytes(original + b"changed")
    try:
        load_phase72b_model_bundle(bundle_path, record["bundle_sha256"])
    except ValueError as exc:
        assert "hash" in str(exc).lower()
    else:
        raise AssertionError("Expected a modified model bundle to be rejected")


def _prepare_and_freeze(tmp_path: Path):
    from paper11_geofm.phase72b_information_gain_screen import (
        prepare_phase72b_information_gain_screen,
        write_phase72b_prepared_artifacts,
    )
    from paper11_geofm.phase72b_models import fit_freeze_phase72b_models

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


def test_phase72b_confirmation_writes_stable_outputs(tmp_path):
    from paper11_geofm.phase72b_information_gain_screen import (
        confirm_phase72b_information_gain_screen,
        write_phase72b_confirmation_artifacts,
    )

    _, prepared_dir, frozen_dir = _prepare_and_freeze(tmp_path)
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
        "transfer_csv",
        "screen_json",
        "screen_md",
    }
    assert result["phase72b_status"] in {
        "phase72b_inputs_not_ready",
        "geofm_information_not_supported",
        "geofm_information_mixed",
        "geofm_information_supported",
    }
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
        assert "confirmation target hash mismatch" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected modified confirmation targets to be rejected"
        )


def test_phase72b_runner_executes_modes_and_rejects_changed_manifest(tmp_path):
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
    frozen = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "fit-freeze",
            "--prepared-dir",
            str(prepared_dir),
            "--output-dir",
            str(frozen_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert frozen.returncode == 0, frozen.stderr
    confirmed = subprocess.run(
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
            str(confirm_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert confirmed.returncode == 0, confirmed.stderr
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
