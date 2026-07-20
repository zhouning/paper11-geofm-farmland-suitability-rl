from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase72a_label_sources import (  # noqa: E402
    load_phase72a_region_contract,
)
from paper11_geofm.phase72b_protocol import (  # noqa: E402
    load_phase72b_protocol,
)
from paper11_geofm.phase72b_terrain import (  # noqa: E402
    fetch_phase72b_terrain,
)


def initialize_earth_engine(
    *, project: str | None = None, authenticate: bool = False
) -> None:
    import ee

    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception as exc:
        if not authenticate:
            raise RuntimeError(
                "Google Earth Engine is not initialized; authenticate first "
                "or use --authenticate"
            ) from exc
        ee.Authenticate()
        ee.Initialize(project=project) if project else ee.Initialize()


def _default_extractor(*, bbox, shape, scale_m, collection, band):
    import ee

    region = ee.Geometry.Rectangle(list(bbox))
    source = (
        ee.ImageCollection(collection)
        .filterBounds(region)
        .select([band])
    )
    source_projection = ee.Image(source.first()).select([band]).projection()
    elevation = (
        source.mosaic()
        .setDefaultProjection(source_projection)
        .clip(region)
    )
    slope = ee.Terrain.slope(elevation)
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.minMax(), sharedInputs=True)
    )
    target_projection = ee.Projection("EPSG:4326").atScale(int(scale_m))
    elevation_stats = (
        elevation.reduceResolution(reducer=reducer, maxPixels=4096)
        .reproject(target_projection)
    )
    slope_stats = (
        slope.reduceResolution(reducer=reducer, maxPixels=4096)
        .reproject(target_projection)
    )
    image = (
        ee.Image.cat(
            [
                elevation_stats.select([f"{band}_mean"]).rename(
                    "elevation_mean"
                ),
                elevation_stats.select([f"{band}_stdDev"]).rename(
                    "elevation_std"
                ),
                elevation_stats.select([f"{band}_min"]).rename(
                    "elevation_min"
                ),
                elevation_stats.select([f"{band}_max"]).rename(
                    "elevation_max"
                ),
                slope_stats.select(["slope_mean"]).rename("slope_mean"),
                slope_stats.select(["slope_stdDev"]).rename("slope_std"),
                slope_stats.select(["slope_max"]).rename("slope_max"),
                elevation_stats.select([f"{band}_max"])
                .subtract(elevation_stats.select([f"{band}_min"]))
                .rename("local_relief"),
            ]
        )
        .setDefaultProjection(target_projection)
    )
    properties = image.sampleRectangle(
        region=region, defaultValue=0
    ).getInfo().get("properties", {})
    expected_names = (
        "elevation_mean",
        "elevation_std",
        "elevation_min",
        "elevation_max",
        "slope_mean",
        "slope_std",
        "slope_max",
        "local_relief",
    )
    arrays = {
        name: np.asarray(properties[name], dtype=np.float32)
        for name in expected_names
        if name in properties
    }
    observed_shapes = {name: value.shape for name, value in arrays.items()}
    if set(arrays) != set(expected_names):
        raise RuntimeError(
            f"Earth Engine terrain returned missing bands: {observed_shapes}"
        )
    if any(tuple(value.shape) != tuple(shape) for value in arrays.values()):
        raise ValueError(
            "Earth Engine terrain shape mismatch: "
            f"expected {shape}, got {observed_shapes}"
        )
    return arrays


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Paper11 Phase 72B Copernicus terrain"
    )
    parser.add_argument("--phase72a-region-config", type=Path, required=True)
    parser.add_argument("--phase72b-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project")
    parser.add_argument("--authenticate", action="store_true")
    args = parser.parse_args(argv)

    try:
        initialize_earth_engine(
            project=args.project, authenticate=args.authenticate
        )
        protocol = load_phase72b_protocol(args.phase72b_protocol)
        regions = load_phase72a_region_contract(args.phase72a_region_config)
        manifest = fetch_phase72b_terrain(
            protocol,
            regions,
            output_dir=args.output_dir,
            extractor=_default_extractor,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Phase 72B terrain fetch: "
        f"{manifest['status']}, {len(manifest['records'])} record(s), "
        f"{len(manifest['failures'])} failure(s)"
    )
    print(
        "Manifest: "
        f"{args.output_dir / 'phase72b_terrain_fetch_manifest.json'}"
    )
    declared_region_ids = {region.region_id for region in regions.regions}
    fetched_region_ids = {
        str(record.get("region_id", ""))
        for record in manifest.get("records", [])
        if isinstance(record, dict)
    }
    complete = (
        manifest.get("status") == "complete"
        and fetched_region_ids == declared_region_ids
        and not manifest.get("failures")
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
