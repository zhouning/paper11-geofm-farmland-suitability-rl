from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


PHASE72A_CLAIM_BOUNDARY = (
    "Phase 72A validates and aligns independent annual product labels with "
    "temporally truncated AlphaEarth histories. It does not train a prediction "
    "model, alter rewards, run planning, prove GeoFM value, or revise the "
    "formal manuscript."
)


@dataclass(frozen=True)
class Phase72ARegionSpec:
    region_id: str
    bbox: tuple[float, float, float, float]
    years: tuple[int, ...]
    grid_shape: tuple[int, int]
    embedding_dim: int
    embedding_pattern: str
    label_pattern: str


@dataclass(frozen=True)
class Phase72ARegionContract:
    source_id: str
    collection: str
    label_role: str
    crop_class_code: int
    scale_m: int
    regions: tuple[Phase72ARegionSpec, ...]


def load_phase72a_region_contract(
    path: Path | str,
) -> Phase72ARegionContract:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    source = payload.get("source", {})
    if source.get("independent_from_dltb_slope_reward_geofm") is not True:
        raise ValueError("Phase 72A label source must be independent")

    regions = []
    seen = set()
    for raw in payload.get("regions", []):
        region_id = str(raw["region_id"]).strip().lower()
        if not region_id or region_id in seen:
            raise ValueError(
                "Phase 72A region_id must be nonblank and unique: "
                f"{region_id}"
            )
        seen.add(region_id)
        years = tuple(sorted({int(year) for year in raw["years"]}))
        if len(years) < 3:
            raise ValueError(
                f"Phase 72A region requires at least three years: {region_id}"
            )
        regions.append(
            Phase72ARegionSpec(
                region_id=region_id,
                bbox=tuple(float(value) for value in raw["bbox"]),
                years=years,
                grid_shape=tuple(int(value) for value in raw["grid_shape"]),
                embedding_dim=int(raw["embedding_dim"]),
                embedding_pattern=str(raw["embedding_pattern"]),
                label_pattern=str(raw["label_pattern"]),
            )
        )
    if not regions:
        raise ValueError("Phase 72A region contract has no regions")

    return Phase72ARegionContract(
        source_id=str(source["source_id"]),
        collection=str(source["collection"]),
        label_role=str(source["label_role"]),
        crop_class_code=int(source["crop_class_code"]),
        scale_m=int(source["scale_m"]),
        regions=tuple(regions),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_phase72a_region_assets(
    contract: Phase72ARegionContract,
    region: Phase72ARegionSpec,
    *,
    embedding_dir: Path | str,
    label_dir: Path | str,
) -> dict[str, object]:
    embedding_dir = Path(embedding_dir)
    label_dir = Path(label_dir)
    errors: list[str] = []
    rows: list[dict[str, object]] = []
    years_ready: list[int] = []

    for year in region.years:
        year_ok = True
        assets = {
            "embedding": embedding_dir
            / region.embedding_pattern.format(year=year),
            "label": label_dir / region.label_pattern.format(year=year),
        }
        for asset_type, path in assets.items():
            if not path.exists():
                errors.append(
                    f"missing {asset_type} for {region.region_id} {year}: "
                    f"{path}"
                )
                year_ok = False
                continue

            array = np.load(path, mmap_mode="r")
            expected_shape = (
                (*region.grid_shape, region.embedding_dim)
                if asset_type == "embedding"
                else region.grid_shape
            )
            if tuple(array.shape) != tuple(expected_shape):
                errors.append(
                    f"{asset_type} shape mismatch for {region.region_id} "
                    f"{year}: expected {expected_shape}, got "
                    f"{tuple(array.shape)}"
                )
                year_ok = False
            rows.append(
                {
                    "region_id": region.region_id,
                    "year": int(year),
                    "asset_type": asset_type,
                    "source_id": (
                        contract.source_id
                        if asset_type == "label"
                        else "alphaearth_annual"
                    ),
                    "path": str(path),
                    "shape": "x".join(str(value) for value in array.shape),
                    "dtype": str(array.dtype),
                    "sha256": _sha256(path),
                    "independent_label": asset_type == "label",
                }
            )
        if year_ok:
            years_ready.append(int(year))

    return {
        "region_id": region.region_id,
        "status": (
            "region_label_inputs_ready"
            if not errors
            else "label_inputs_not_ready"
        ),
        "years_ready": years_ready,
        "errors": errors,
        "file_manifest_rows": rows,
        "claim_boundary": PHASE72A_CLAIM_BOUNDARY,
    }
