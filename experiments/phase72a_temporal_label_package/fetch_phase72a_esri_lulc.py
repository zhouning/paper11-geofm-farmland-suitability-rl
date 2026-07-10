from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase72a_label_sources import (  # noqa: E402
    _sha256,
    load_phase72a_region_contract,
)


def _parse_years(raw: str) -> tuple[int, ...]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        elif part:
            values.append(int(part))
    return tuple(sorted(set(values)))


def _default_extractor(*, bbox, year, scale, collection):
    import ee

    region = ee.Geometry.Rectangle(list(bbox))
    image = (
        ee.ImageCollection(collection)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(region)
        .select(["b1"])
        .mosaic()
        .clip(region)
        .setDefaultProjection(
            ee.Projection("EPSG:4326").atScale(int(scale))
        )
    )
    result = image.sampleRectangle(
        region=region, defaultValue=0
    ).getInfo()
    values = result.get("properties", {}).get("b1")
    if values is None:
        raise RuntimeError(f"ESRI LULC returned no b1 values for {year}")
    return np.asarray(values, dtype=np.int32)


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


def fetch_phase72a_labels(
    *,
    region_config: Path | str,
    output_dir: Path | str,
    regions: tuple[str, ...],
    years: tuple[int, ...],
    extractor=None,
    overwrite: bool = False,
) -> dict[str, object]:
    contract = load_phase72a_region_contract(region_config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    requested_regions = set(regions)
    requested_years = set(int(year) for year in years)
    extractor = extractor or _default_extractor
    records = []
    failures = []

    for region in contract.regions:
        if region.region_id not in requested_regions:
            continue
        for year in region.years:
            if year not in requested_years:
                continue
            path = output / region.label_pattern.format(year=year)
            try:
                if path.exists() and not overwrite:
                    array = np.load(path)
                    status = "cached"
                else:
                    array = np.asarray(
                        extractor(
                            bbox=region.bbox,
                            year=year,
                            scale=contract.scale_m,
                            collection=contract.collection,
                        ),
                        dtype=np.int32,
                    )
                    if tuple(array.shape) != region.grid_shape:
                        raise ValueError(
                            "label shape mismatch: expected "
                            f"{region.grid_shape}, got {tuple(array.shape)}"
                        )
                    np.save(path, array)
                    status = "fetched"
                if tuple(array.shape) != region.grid_shape:
                    raise ValueError(
                        "cached label shape mismatch: expected "
                        f"{region.grid_shape}, got {tuple(array.shape)}"
                    )
                records.append(
                    {
                        "region_id": region.region_id,
                        "year": year,
                        "bbox": list(region.bbox),
                        "scale_m": contract.scale_m,
                        "collection": contract.collection,
                        "path": str(path),
                        "shape": list(array.shape),
                        "sha256": _sha256(path),
                        "status": status,
                    }
                )
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append(
                    {
                        "region_id": region.region_id,
                        "year": year,
                        "reason": str(exc),
                    }
                )

    manifest = {
        "status": (
            "complete"
            if records and not failures
            else "partial"
            if records
            else "failed"
        ),
        "source_id": contract.source_id,
        "collection": contract.collection,
        "n_records": len(records),
        "n_failures": len(failures),
        "records": records,
        "failures": failures,
    }
    (output / "phase72a_lulc_fetch_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Paper11 Phase 72A ESRI annual LULC labels"
    )
    parser.add_argument("--region-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--regions", required=True)
    parser.add_argument("--years", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--project")
    parser.add_argument("--authenticate", action="store_true")
    args = parser.parse_args(argv)

    try:
        initialize_earth_engine(
            project=args.project, authenticate=args.authenticate
        )
        regions = tuple(
            sorted(
                {
                    value.strip().lower()
                    for value in args.regions.split(",")
                    if value.strip()
                }
            )
        )
        manifest = fetch_phase72a_labels(
            region_config=args.region_config,
            output_dir=args.output_dir,
            regions=regions,
            years=_parse_years(args.years),
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    manifest_path = args.output_dir / "phase72a_lulc_fetch_manifest.json"
    print(
        "Phase 72A ESRI LULC fetch: "
        f"{manifest['status']}, {manifest['n_records']} record(s), "
        f"{manifest['n_failures']} failure(s)"
    )
    print(f"Manifest: {manifest_path}")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
