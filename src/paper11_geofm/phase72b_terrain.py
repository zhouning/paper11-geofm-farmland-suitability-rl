from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .phase72a_label_sources import Phase72ARegionContract
from .phase72b_protocol import (
    PHASE72B_CLAIM_BOUNDARY,
    Phase72BProtocol,
)


def _feature_derivations(band: str) -> dict[str, str]:
    return {
        "elevation_mean": f"{band}:mean",
        "elevation_std": f"{band}:stdDev",
        "elevation_min": f"{band}:min",
        "elevation_max": f"{band}:max",
        "slope_mean": f"ee.Terrain.slope({band}):mean",
        "slope_std": f"ee.Terrain.slope({band}):stdDev",
        "slope_max": f"ee.Terrain.slope({band}):max",
        "local_relief": f"{band}:max-minus-min",
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_phase72b_terrain(
    protocol: Phase72BProtocol,
    regions: Phase72ARegionContract,
    *,
    output_dir: Path | str,
    extractor: Callable[..., dict[str, np.ndarray]],
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []

    for region in regions.regions:
        path = output / f"{region.region_id}_terrain.npz"
        try:
            arrays = extractor(
                bbox=region.bbox,
                shape=region.grid_shape,
                scale_m=protocol.terrain_scale_m,
                collection=protocol.terrain_collection,
                band=protocol.terrain_band,
            )
            missing = [
                name for name in protocol.terrain_features if name not in arrays
            ]
            if missing:
                raise ValueError(f"missing terrain features: {missing}")
            unexpected = sorted(
                set(arrays) - set(protocol.terrain_features)
            )
            if unexpected:
                raise ValueError(
                    f"unexpected terrain features: {unexpected}"
                )
            normalized = {}
            for name in protocol.terrain_features:
                value = np.asarray(arrays[name], dtype=np.float32)
                if tuple(value.shape) != region.grid_shape:
                    raise ValueError(
                        "terrain shape mismatch for "
                        f"{region.region_id} {name}: expected "
                        f"{region.grid_shape}, got {tuple(value.shape)}"
                    )
                if not np.isfinite(value).all():
                    raise ValueError(
                        f"non-finite terrain values: {region.region_id} {name}"
                    )
                normalized[name] = value
            np.savez_compressed(path, **normalized)
            records.append(
                {
                    "region_id": region.region_id,
                    "source_id": protocol.terrain_source_id,
                    "collection": protocol.terrain_collection,
                    "band": protocol.terrain_band,
                    "feature_derivations": _feature_derivations(
                        protocol.terrain_band
                    ),
                    "scale_m": protocol.terrain_scale_m,
                    "bbox": list(region.bbox),
                    "path": path.name,
                    "shape": "x".join(map(str, region.grid_shape)),
                    "dtype": "float32",
                    "sha256": _file_sha256(path),
                }
            )
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(
                {"region_id": region.region_id, "reason": str(exc)}
            )

    declared_region_ids = [region.region_id for region in regions.regions]
    fetched_region_ids = [str(record["region_id"]) for record in records]
    all_regions_complete = (
        len(fetched_region_ids) == len(declared_region_ids)
        and set(fetched_region_ids) == set(declared_region_ids)
        and not failures
    )
    manifest = {
        "status": (
            "complete"
            if all_regions_complete
            else "partial"
            if records
            else "failed"
        ),
        "declared_region_ids": declared_region_ids,
        "source_id": protocol.terrain_source_id,
        "collection": protocol.terrain_collection,
        "band": protocol.terrain_band,
        "scale_m": protocol.terrain_scale_m,
        "records": records,
        "failures": failures,
        "claim_boundary": PHASE72B_CLAIM_BOUNDARY,
    }
    (output / "phase72b_terrain_fetch_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _audit_fetch_manifest(
    protocol: Phase72BProtocol,
    regions: Phase72ARegionContract,
    root: Path,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    manifest_path = root / "phase72b_terrain_fetch_manifest.json"
    if not manifest_path.exists():
        return {}, [f"missing terrain fetch manifest: {manifest_path}"]

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("manifest root must be an object")
    except (OSError, ValueError) as exc:
        return {}, [f"unreadable terrain fetch manifest: {exc}"]

    errors = []
    expected_region_ids = [region.region_id for region in regions.regions]
    region_specs = {region.region_id: region for region in regions.regions}
    if payload.get("status") != "complete":
        errors.append(
            "terrain fetch manifest is not complete: "
            f"{payload.get('status', 'missing')}"
        )
    expected_contract = {
        "source_id": protocol.terrain_source_id,
        "collection": protocol.terrain_collection,
        "band": protocol.terrain_band,
        "scale_m": protocol.terrain_scale_m,
    }
    for field, expected in expected_contract.items():
        if payload.get(field) != expected:
            errors.append(
                f"terrain fetch manifest {field} mismatch: "
                f"expected {expected}, got {payload.get(field)}"
            )

    declared_ids = payload.get("declared_region_ids")
    if not isinstance(declared_ids, list) or [
        str(value) for value in declared_ids
    ] != expected_region_ids:
        errors.append("terrain fetch manifest declared regions mismatch")

    records = payload.get("records")
    if not isinstance(records, list):
        return {}, [*errors, "terrain fetch manifest records must be a list"]
    record_by_region = {}
    observed_region_ids = []
    for raw_record in records:
        if not isinstance(raw_record, dict):
            errors.append("terrain fetch manifest record must be an object")
            continue
        region_id = str(raw_record.get("region_id", ""))
        observed_region_ids.append(region_id)
        if not region_id or region_id in record_by_region:
            errors.append(
                f"terrain fetch manifest duplicate or blank region: {region_id}"
            )
            continue
        record_by_region[region_id] = dict(raw_record)
        if region_id not in region_specs:
            errors.append(
                f"terrain fetch manifest unexpected region: {region_id}"
            )
            continue
        region = region_specs[region_id]
        expected_record = {
            "source_id": protocol.terrain_source_id,
            "collection": protocol.terrain_collection,
            "band": protocol.terrain_band,
            "feature_derivations": _feature_derivations(
                protocol.terrain_band
            ),
            "scale_m": protocol.terrain_scale_m,
            "bbox": list(region.bbox),
            "path": f"{region.region_id}_terrain.npz",
            "shape": "x".join(map(str, region.grid_shape)),
            "dtype": "float32",
        }
        for field, expected in expected_record.items():
            if field not in raw_record:
                errors.append(
                    f"terrain fetch manifest record missing {region_id} "
                    f"{field}"
                )
            elif raw_record[field] != expected:
                errors.append(
                    f"terrain fetch manifest record {field} mismatch "
                    f"{region_id}: expected {expected}, "
                    f"got {raw_record[field]}"
                )
        expected_hash = str(raw_record.get("sha256", "")).lower()
        if len(expected_hash) != 64 or any(
            value not in "0123456789abcdef" for value in expected_hash
        ):
            errors.append(
                f"terrain fetch manifest record sha256 invalid: {region_id}"
            )
    if observed_region_ids != expected_region_ids:
        errors.append("terrain fetch manifest record regions mismatch")
    if payload.get("failures"):
        errors.append("terrain fetch manifest contains failures")
    return record_by_region, errors


def audit_phase72b_terrain_assets(
    protocol: Phase72BProtocol,
    regions: Phase72ARegionContract,
    terrain_dir: Path | str,
) -> dict[str, object]:
    root = Path(terrain_dir)
    rows = []
    manifest_records, errors = _audit_fetch_manifest(
        protocol, regions, root
    )

    for region in regions.regions:
        path = root / f"{region.region_id}_terrain.npz"
        if not path.exists():
            errors.append(
                f"missing terrain file for {region.region_id}: {path}"
            )
            continue
        try:
            with np.load(path) as package:
                unexpected = sorted(
                    set(package.files) - set(protocol.terrain_features)
                )
                if unexpected:
                    errors.append(
                        f"unexpected terrain features {region.region_id}: "
                        f"{unexpected}"
                    )
                for name in protocol.terrain_features:
                    if name not in package:
                        errors.append(
                            f"missing terrain feature {region.region_id} {name}"
                        )
                        continue
                    value = package[name]
                    if tuple(value.shape) != region.grid_shape:
                        errors.append(
                            f"terrain shape mismatch {region.region_id} {name}: "
                            f"expected {region.grid_shape}, got {tuple(value.shape)}"
                        )
                    if not np.isfinite(value).all():
                        errors.append(
                            f"non-finite terrain values {region.region_id} {name}"
                        )
                    if value.dtype != np.dtype("float32"):
                        errors.append(
                            f"terrain dtype mismatch {region.region_id} {name}: "
                            f"expected float32, got {value.dtype}"
                        )
            actual_hash = _file_sha256(path)
            manifest_record = manifest_records.get(region.region_id)
            expected_hash = (
                str(manifest_record.get("sha256", "")).lower()
                if manifest_record is not None
                else None
            )
            if expected_hash is not None and expected_hash != actual_hash:
                errors.append(
                    f"terrain fetch manifest hash mismatch "
                    f"{region.region_id}: expected {expected_hash}, "
                    f"got {actual_hash}"
                )
            if manifest_record is not None:
                feature_derivations = _feature_derivations(
                    protocol.terrain_band
                )
                bbox = list(region.bbox)
                rows.append(
                    {
                        "region_id": region.region_id,
                        "source_id": protocol.terrain_source_id,
                        "collection": protocol.terrain_collection,
                        "band": protocol.terrain_band,
                        "feature_derivations": feature_derivations,
                        "feature_derivations_json": json.dumps(
                            feature_derivations,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "scale_m": protocol.terrain_scale_m,
                        "bbox": bbox,
                        "bbox_json": json.dumps(
                            bbox, separators=(",", ":")
                        ),
                        "path": f"{region.region_id}_terrain.npz",
                        "shape": "x".join(map(str, region.grid_shape)),
                        "dtype": "float32",
                        "sha256": actual_hash,
                    }
                )
        except (OSError, ValueError) as exc:
            errors.append(
                f"unreadable terrain package {region.region_id}: {exc}"
            )

    return {
        "status": (
            "terrain_inputs_ready"
            if not errors and len(rows) == len(regions.regions)
            else "phase72b_inputs_not_ready"
        ),
        "rows": rows,
        "errors": errors,
        "claim_boundary": PHASE72B_CLAIM_BOUNDARY,
    }
