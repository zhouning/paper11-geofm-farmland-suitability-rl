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
                    "path": str(path),
                    "shape": "x".join(map(str, region.grid_shape)),
                    "sha256": _file_sha256(path),
                }
            )
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(
                {"region_id": region.region_id, "reason": str(exc)}
            )

    manifest = {
        "status": (
            "complete"
            if records and not failures
            else "partial"
            if records
            else "failed"
        ),
        "source_id": protocol.terrain_source_id,
        "collection": protocol.terrain_collection,
        "scale_m": protocol.terrain_scale_m,
        "records": records,
        "failures": failures,
        "claim_boundary": PHASE72B_CLAIM_BOUNDARY,
    }
    (output / "phase72b_terrain_fetch_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def audit_phase72b_terrain_assets(
    protocol: Phase72BProtocol,
    regions: Phase72ARegionContract,
    terrain_dir: Path | str,
) -> dict[str, object]:
    root = Path(terrain_dir)
    rows = []
    errors = []

    for region in regions.regions:
        path = root / f"{region.region_id}_terrain.npz"
        if not path.exists():
            errors.append(
                f"missing terrain file for {region.region_id}: {path}"
            )
            continue
        try:
            with np.load(path) as package:
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
            rows.append(
                {
                    "region_id": region.region_id,
                    "path": str(path),
                    "shape": "x".join(map(str, region.grid_shape)),
                    "sha256": _file_sha256(path),
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
