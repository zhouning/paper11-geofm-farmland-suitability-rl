from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping


PHASE72B_CLAIM_BOUNDARY = (
    "Phase 72B is a leakage-free low-cost information-gain screen using "
    "independent annual product labels. It does not implement GeoFM-STaR, "
    "alter planning rewards, run planning, or revise the formal manuscript."
)

PHASE72B_TERRAIN_SOURCE_ID = "copernicus_dem_glo30"
PHASE72B_TERRAIN_COLLECTION = "COPERNICUS/DEM/GLO30"
PHASE72B_TERRAIN_BAND = "DEM"
PHASE72B_TERRAIN_SCALE_M = 500
PHASE72B_TERRAIN_FEATURES = (
    "elevation_mean",
    "elevation_std",
    "elevation_min",
    "elevation_max",
    "slope_mean",
    "slope_std",
    "slope_max",
    "local_relief",
)
PHASE72B_TERRAIN = {
    "source_id": PHASE72B_TERRAIN_SOURCE_ID,
    "collection": PHASE72B_TERRAIN_COLLECTION,
    "band": PHASE72B_TERRAIN_BAND,
    "scale_m": PHASE72B_TERRAIN_SCALE_M,
    "feature_names": list(PHASE72B_TERRAIN_FEATURES),
}
PHASE72B_GATES = {
    "ap_vs_explicit": 0.015,
    "brier_vs_explicit": 0.005,
    "ece_vs_explicit": 0.010,
    "ap_vs_control": 0.005,
    "brier_vs_control": 0.002,
    "transfer_ap_gain": 0.005,
    "transfer_brier_gain": 0.002,
    "transfer_ap_harm": 0.005,
    "transfer_brier_harm": 0.002,
}
PHASE72B_SEED = 72
PHASE72B_YEARS = {
    "train": [2017, 2018, 2019, 2020, 2021],
    "validation": [2022],
    "test": [2023],
}
PHASE72B_CONTROLS = {
    "seeds": [72, 73, 74, 75, 76],
    "random_projection_dim": 320,
    "partition_local": True,
    "learned_transform_fit_scope": "training_rows_only",
    "reuse_phase8_d4_tables": False,
}
PHASE72B_SPATIAL = {"block_size": 8, "folds": 5, "buffer_rings": 1}
PHASE72B_BOOTSTRAP = {"iterations": 2000, "seed": 72}
PHASE72B_MODELS = {
    "logistic_c": [0.01, 0.1, 1.0, 10.0],
    "logistic_class_weight": ["none", "balanced"],
    "hgb_learning_rate": [0.03, 0.08],
    "hgb_max_leaf_nodes": [15, 31],
    "hgb_min_samples_leaf": [20, 50],
    "hgb_max_iter": 200,
    "hgb_l2_regularization": [0.0, 1.0],
}
PHASE72B_CALIBRATION = {
    "methods": ["none", "sigmoid", "isotonic"],
    "ece_bins": 10,
}
PHASE72B_BUDGETS = [0.10, 0.20]
PHASE72B_VARIANTS = [
    "explicit_static",
    "explicit_history",
    "geofm_current_only",
    "geofm_temporal_mean_only",
    "explicit_plus_geofm_current",
    "explicit_plus_geofm_temporal_full",
    "explicit_plus_temporal_order_shuffle",
    "explicit_plus_spatial_shuffle",
    "explicit_plus_random_projection",
]
PHASE72B_TOP_LEVEL_FIELDS = {
    "phase",
    "seed",
    "terrain",
    "years",
    "controls",
    "spatial",
    "bootstrap",
    "models",
    "calibration",
    "budgets",
    "variants",
    "gates",
}


@dataclass(frozen=True)
class Phase72BProtocol:
    seed: int
    terrain_source_id: str
    terrain_collection: str
    terrain_band: str
    terrain_scale_m: int
    terrain_features: tuple[str, ...]
    train_years: tuple[int, ...]
    validation_years: tuple[int, ...]
    test_years: tuple[int, ...]
    control_seeds: tuple[int, ...]
    random_projection_dim: int
    control_partition_local: bool
    learned_transform_fit_scope: str
    reuse_phase8_d4_tables: bool
    spatial_block_size: int
    spatial_folds: int
    buffer_rings: int
    bootstrap_iterations: int
    bootstrap_seed: int
    gates: dict[str, float]
    raw: dict[str, object]


def canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_hashed_json(
    path: Path | str, payload: Mapping[str, object]
) -> tuple[Path, Path]:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = json.loads(json.dumps(dict(payload)))
    json_path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8"
    )
    hash_path = json_path.with_suffix(".sha256")
    hash_path.write_text(
        canonical_json_sha256(normalized) + "\n", encoding="ascii"
    )
    return json_path, hash_path


def load_hashed_json(
    json_path: Path | str, hash_path: Path | str | None = None
) -> dict[str, object]:
    source = Path(json_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    digest_path = (
        Path(hash_path) if hash_path is not None else source.with_suffix(".sha256")
    )
    expected = digest_path.read_text(encoding="ascii").strip().lower()
    actual = canonical_json_sha256(payload)
    if expected != actual:
        raise ValueError(
            "Phase 72B frozen JSON hash mismatch: "
            f"expected {expected}, got {actual}"
        )
    return payload


def _require_frozen_value(
    actual: object, expected: object, *, section: str
) -> None:
    if canonical_json_sha256({"value": actual}) != canonical_json_sha256(
        {"value": expected}
    ):
        raise ValueError(f"Phase 72B frozen {section} contract mismatch")


def validate_phase72b_protocol_payload(
    payload: Mapping[str, object],
) -> Phase72BProtocol:
    payload = dict(payload)
    if payload.get("phase") != "phase72b_geofm_information_gain_screen":
        raise ValueError("Invalid Phase 72B protocol phase")
    _require_frozen_value(payload.get("seed"), PHASE72B_SEED, section="seed")

    raw_terrain = payload.get("terrain")
    if not isinstance(raw_terrain, dict):
        raise ValueError("Phase 72B frozen terrain contract is missing")
    if set(payload) != PHASE72B_TOP_LEVEL_FIELDS:
        raise ValueError("Phase 72B frozen top-level contract mismatch")
    terrain = dict(raw_terrain)
    years = dict(payload.get("years", {}))
    controls = dict(payload.get("controls", {}))
    spatial = dict(payload.get("spatial", {}))
    bootstrap = dict(payload.get("bootstrap", {}))
    _require_frozen_value(
        payload.get("gates"), PHASE72B_GATES, section="gate thresholds"
    )
    gates = {
        str(key): float(value)
        for key, value in dict(payload["gates"]).items()
    }
    if gates != PHASE72B_GATES:
        raise ValueError("Phase 72B frozen gate thresholds mismatch")
    expected_terrain = {
        "source_id": PHASE72B_TERRAIN_SOURCE_ID,
        "collection": PHASE72B_TERRAIN_COLLECTION,
        "band": PHASE72B_TERRAIN_BAND,
        "scale_m": PHASE72B_TERRAIN_SCALE_M,
    }
    for field, expected in expected_terrain.items():
        if field not in terrain:
            raise ValueError(
                f"Phase 72B frozen terrain contract missing field: {field}"
            )
        if terrain[field] != expected:
            raise ValueError(
                f"Phase 72B frozen terrain contract mismatch for {field}: "
                f"expected {expected}, got {terrain[field]}"
            )
    terrain_feature_names = terrain.get("feature_names")
    if not isinstance(terrain_feature_names, list):
        raise ValueError(
            "Phase 72B frozen terrain contract missing field: feature_names"
        )
    if (
        tuple(str(value) for value in terrain_feature_names)
        != PHASE72B_TERRAIN_FEATURES
    ):
        raise ValueError(
            "Phase 72B frozen terrain contract mismatch for feature_names"
        )
    _require_frozen_value(terrain, PHASE72B_TERRAIN, section="terrain")
    partition = [
        int(value)
        for name in ("train", "validation", "test")
        for value in years[name]
    ]
    if sorted(partition) != list(range(2017, 2024)):
        raise ValueError("Phase 72B years must partition origins 2017-2023")
    if tuple(int(value) for value in controls["seeds"]) != (
        72,
        73,
        74,
        75,
        76,
    ):
        raise ValueError("Phase 72B control seeds are frozen")
    if controls.get("partition_local") is not True:
        raise ValueError("Phase 72B controls must be partition-local")
    if controls.get("learned_transform_fit_scope") != "training_rows_only":
        raise ValueError(
            "Phase 72B learned transforms must fit training rows only"
        )
    if controls.get("reuse_phase8_d4_tables") is not False:
        raise ValueError(
            "Phase 72B must not reuse transductive Phase 8 D4 tables"
        )
    _require_frozen_value(years, PHASE72B_YEARS, section="years")
    _require_frozen_value(controls, PHASE72B_CONTROLS, section="controls")
    _require_frozen_value(spatial, PHASE72B_SPATIAL, section="spatial")
    _require_frozen_value(
        bootstrap, PHASE72B_BOOTSTRAP, section="bootstrap"
    )
    _require_frozen_value(
        payload.get("models"), PHASE72B_MODELS, section="models"
    )
    _require_frozen_value(
        payload.get("calibration"),
        PHASE72B_CALIBRATION,
        section="calibration",
    )
    _require_frozen_value(
        payload.get("budgets"), PHASE72B_BUDGETS, section="budgets"
    )
    _require_frozen_value(
        payload.get("variants"), PHASE72B_VARIANTS, section="variants"
    )

    return Phase72BProtocol(
        seed=int(payload["seed"]),
        terrain_source_id=str(terrain["source_id"]),
        terrain_collection=str(terrain["collection"]),
        terrain_band=str(terrain["band"]),
        terrain_scale_m=int(terrain["scale_m"]),
        terrain_features=tuple(str(value) for value in terrain["feature_names"]),
        train_years=tuple(int(value) for value in years["train"]),
        validation_years=tuple(
            int(value) for value in years["validation"]
        ),
        test_years=tuple(int(value) for value in years["test"]),
        control_seeds=tuple(int(value) for value in controls["seeds"]),
        random_projection_dim=int(controls["random_projection_dim"]),
        control_partition_local=bool(controls["partition_local"]),
        learned_transform_fit_scope=str(
            controls["learned_transform_fit_scope"]
        ),
        reuse_phase8_d4_tables=bool(controls["reuse_phase8_d4_tables"]),
        spatial_block_size=int(spatial["block_size"]),
        spatial_folds=int(spatial["folds"]),
        buffer_rings=int(spatial["buffer_rings"]),
        bootstrap_iterations=int(bootstrap["iterations"]),
        bootstrap_seed=int(bootstrap["seed"]),
        gates=gates,
        raw=payload,
    )


def load_phase72b_protocol(path: Path | str) -> Phase72BProtocol:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_phase72b_protocol_payload(payload)
