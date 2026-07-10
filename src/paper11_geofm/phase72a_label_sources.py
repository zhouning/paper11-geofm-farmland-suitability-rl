from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


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
