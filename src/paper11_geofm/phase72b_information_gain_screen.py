from __future__ import annotations

from collections.abc import Mapping
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .phase72a_label_sources import (
    audit_phase72a_region_assets,
    load_phase72a_region_contract,
)
from .phase72b_explicit_features import build_phase72b_explicit_features
from .phase72b_geofm_features import build_phase72b_geofm_features
from .phase72b_protocol import (
    PHASE72B_CLAIM_BOUNDARY,
    canonical_json_sha256,
    load_phase72b_protocol,
    write_hashed_json,
)
from .phase72b_splits import (
    audit_phase72b_splits,
    build_phase72b_split_registry,
)
from .phase72b_terrain import audit_phase72b_terrain_assets, _file_sha256


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _load_phase72a_inputs(
    package_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, np.ndarray]]:
    package_path = package_dir / "phase72a_temporal_label_package.json"
    sample_path = package_dir / "phase72a_temporal_sample_index.csv"
    tensor_path = package_dir / "phase72a_temporal_samples.npz"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    if package.get("phase72a_status") != "phase72a_label_inputs_ready":
        raise ValueError("Phase 72A package is not ready")
    rows = pd.read_csv(sample_path, keep_default_na=False).to_dict(
        orient="records"
    )
    tensors = {}
    with np.load(tensor_path) as loaded:
        for name in loaded.files:
            tensors[name] = loaded[name]
    return package, rows, tensors


def _geofm_registry(embedding_dim: int) -> dict[str, list[str]]:
    current = [f"geofm_current_{index:03d}" for index in range(embedding_dim)]
    mean = [f"geofm_temporal_mean_{index:03d}" for index in range(embedding_dim)]
    full = []
    for prefix in ("current", "mean", "std", "delta", "trend"):
        full.extend(
            [f"geofm_{prefix}_{index:03d}" for index in range(embedding_dim)]
        )
    return {
        "geofm_current": current,
        "geofm_temporal_mean": mean,
        "geofm_temporal_full": full,
    }


def prepare_phase72b_information_gain_screen(
    *,
    protocol_path: Path | str,
    phase72a_region_config: Path | str,
    phase72a_package_dir: Path | str,
    embedding_dirs: Mapping[str, Path | str],
    label_dirs: Mapping[str, Path | str],
    terrain_dir: Path | str,
) -> dict[str, object]:
    protocol = load_phase72b_protocol(protocol_path)
    contract = load_phase72a_region_contract(phase72a_region_config)
    phase72a_package, sample_rows, tensors = _load_phase72a_inputs(
        Path(phase72a_package_dir)
    )
    if [int(row["sample_index"]) for row in sample_rows] != list(
        range(len(sample_rows))
    ):
        raise ValueError("Phase 72B sample indexes must be contiguous")

    manifest_rows = []
    asset_errors = []
    region_specs = {region.region_id: region for region in contract.regions}
    for region in contract.regions:
        if region.region_id not in embedding_dirs or region.region_id not in label_dirs:
            asset_errors.append(f"missing Phase 72B path mapping: {region.region_id}")
            continue
        audit = audit_phase72a_region_assets(
            contract,
            region,
            embedding_dir=embedding_dirs[region.region_id],
            label_dir=label_dirs[region.region_id],
        )
        manifest_rows.extend(audit["file_manifest_rows"])
        asset_errors.extend(audit["errors"])
    terrain_audit = audit_phase72b_terrain_assets(
        protocol, contract, terrain_dir
    )
    asset_errors.extend(terrain_audit["errors"])
    if asset_errors:
        raise ValueError("Phase 72B inputs not ready: " + " | ".join(asset_errors))

    labels = {}
    terrain = {}
    for region in contract.regions:
        labels[region.region_id] = {
            year: np.load(
                Path(label_dirs[region.region_id])
                / region.label_pattern.format(year=year)
            )
            for year in region.years
        }
        terrain_path = Path(terrain_dir) / f"{region.region_id}_terrain.npz"
        with np.load(terrain_path) as loaded:
            terrain[region.region_id] = {
                name: loaded[name] for name in protocol.terrain_features
            }

    explicit = build_phase72b_explicit_features(
        sample_rows,
        regions=region_specs,
        labels=labels,
        terrain=terrain,
        crop_class_code=contract.crop_class_code,
    )
    geofm = build_phase72b_geofm_features(
        tensors["embedding_history"], tensors["history_mask"]
    )
    matrices = {
        "explicit_static": explicit["explicit_static"],
        "explicit_history": explicit["explicit_history"],
        **geofm,
        "embedding_history": tensors["embedding_history"].astype(np.float32),
        "history_mask": tensors["history_mask"].astype(bool),
    }
    registry = {
        **explicit["registry"],
        **_geofm_registry(contract.regions[0].embedding_dim),
    }
    split_rows = []
    for row in sample_rows:
        adjusted = dict(row)
        adjusted["conversion_1y"] = 1 - int(row["y_1y"])
        split_rows.append(adjusted)
    split_registry = build_phase72b_split_registry(
        split_rows,
        train_years=protocol.train_years,
        validation_year=protocol.validation_years[0],
        test_year=protocol.test_years[0],
        folds=protocol.spatial_folds,
        buffer_rings=protocol.buffer_rings,
    )
    leakage_audit = audit_phase72b_splits(
        split_rows,
        split_registry,
        train_years=protocol.train_years,
        validation_year=protocol.validation_years[0],
        test_year=protocol.test_years[0],
    )
    if leakage_audit["status"] != "leakage_audit_passed":
        raise ValueError(
            "Phase 72B leakage audit failed: "
            + " | ".join(leakage_audit["errors"])
        )

    origins = np.asarray(
        [int(row["origin_year"]) for row in split_rows], dtype=np.int16
    )
    sample_indexes = np.asarray(
        [int(row["sample_index"]) for row in split_rows], dtype=np.int32
    )
    conversion = np.asarray(
        [int(row["conversion_1y"]) for row in split_rows], dtype=np.int8
    )
    development_mask = origins <= protocol.validation_years[0]
    confirmation_mask = origins == protocol.test_years[0]
    development_targets = {
        "sample_index": sample_indexes[development_mask],
        "origin_year": origins[development_mask],
        "conversion_1y": conversion[development_mask],
    }
    confirmation_targets = {
        "sample_index": sample_indexes[confirmation_mask],
        "origin_year": origins[confirmation_mask],
        "conversion_1y": conversion[confirmation_mask],
    }

    feature_manifest_rows = [
        {
            "matrix_id": name,
            "shape": "x".join(map(str, value.shape)),
            "dtype": str(value.dtype),
            "sha256": _array_sha256(value),
        }
        for name, value in matrices.items()
    ]
    feature_rows = [
        {
            key: row[key]
            for key in (
                "sample_index",
                "region_id",
                "unit_id",
                "row",
                "col",
                "spatial_block_id",
                "origin_year",
            )
        }
        for row in split_rows
    ]
    frozen_protocol = {
        "status": "phase72b_protocol_frozen",
        "tracked_protocol": protocol.raw,
        "tracked_protocol_sha256": canonical_json_sha256(protocol.raw),
        "phase72a_package_sha256": _file_sha256(
            Path(phase72a_package_dir)
            / "phase72a_temporal_label_package.json"
        ),
        "terrain_manifest": terrain_audit["rows"],
        "annual_asset_manifest_rows": manifest_rows,
        "feature_manifest_rows": feature_manifest_rows,
        "feature_registry_sha256": canonical_json_sha256(registry),
        "split_registry_sha256": canonical_json_sha256(split_registry),
        "leakage_status": leakage_audit["status"],
        "development_target_rows": int(development_mask.sum()),
        "confirmation_target_rows": int(confirmation_mask.sum()),
        "claim_boundary": PHASE72B_CLAIM_BOUNDARY,
    }
    return {
        "protocol": protocol,
        "phase72a_package": phase72a_package,
        "terrain_manifest_rows": terrain_audit["rows"],
        "feature_manifest_rows": feature_manifest_rows,
        "feature_registry": registry,
        "feature_rows": feature_rows,
        "matrices": matrices,
        "development_targets": development_targets,
        "confirmation_targets": confirmation_targets,
        "split_registry": split_registry,
        "leakage_audit": leakage_audit,
        "frozen_protocol": frozen_protocol,
        "claim_boundary": PHASE72B_CLAIM_BOUNDARY,
    }


def _write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: list[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_phase72b_prepared_artifacts(
    package: Mapping[str, object], output_dir: Path | str
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "terrain_manifest_csv": output / "phase72b_terrain_manifest.csv",
        "feature_manifest_csv": output / "phase72b_feature_manifest.csv",
        "feature_registry_json": output / "phase72b_feature_registry.json",
        "feature_rows_csv": output / "phase72b_feature_rows.csv",
        "feature_matrices_npz": output / "phase72b_feature_matrices.npz",
        "development_targets_npz": output
        / "phase72b_development_targets.npz",
        "confirmation_targets_npz": output
        / "phase72b_confirmation_targets.npz",
        "split_registry_json": output / "phase72b_split_registry.json",
        "row_alignment_csv": output / "phase72b_row_alignment_audit.csv",
        "leakage_audit_json": output / "phase72b_leakage_audit.json",
        "protocol_json": output / "phase72b_frozen_protocol.json",
        "protocol_hash": output / "phase72b_frozen_protocol.sha256",
    }
    _write_csv(
        paths["terrain_manifest_csv"],
        list(package["terrain_manifest_rows"]),
        ["region_id", "path", "shape", "sha256"],
    )
    _write_csv(
        paths["feature_manifest_csv"],
        list(package["feature_manifest_rows"]),
        ["matrix_id", "shape", "dtype", "sha256"],
    )
    paths["feature_registry_json"].write_text(
        json.dumps(package["feature_registry"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    feature_rows = list(package["feature_rows"])
    _write_csv(
        paths["feature_rows_csv"],
        feature_rows,
        [
            "sample_index",
            "region_id",
            "unit_id",
            "row",
            "col",
            "spatial_block_id",
            "origin_year",
        ],
    )
    np.savez_compressed(paths["feature_matrices_npz"], **package["matrices"])
    np.savez_compressed(
        paths["development_targets_npz"], **package["development_targets"]
    )
    np.savez_compressed(
        paths["confirmation_targets_npz"], **package["confirmation_targets"]
    )
    paths["split_registry_json"].write_text(
        json.dumps(package["split_registry"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(
        paths["row_alignment_csv"],
        [
            {
                "sample_rows": len(feature_rows),
                "first_sample_index": 0 if feature_rows else "",
                "last_sample_index": len(feature_rows) - 1 if feature_rows else "",
                "status": "row_alignment_passed",
            }
        ],
        [
            "sample_rows",
            "first_sample_index",
            "last_sample_index",
            "status",
        ],
    )
    paths["leakage_audit_json"].write_text(
        json.dumps(package["leakage_audit"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    protocol_json, protocol_hash = write_hashed_json(
        paths["protocol_json"], package["frozen_protocol"]
    )
    paths["protocol_json"] = protocol_json
    paths["protocol_hash"] = protocol_hash
    return paths
