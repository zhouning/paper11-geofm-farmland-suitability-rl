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
