from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


PHASE11_CLAIM_BOUNDARY = (
    "Phase 11 builds real Bishan DLTB-derived Phase 2 inputs; "
    "centroid-to-grid assignment is an alignment adapter, not final "
    "parcel-accurate GeoFM evidence, and this phase does not train or "
    "evaluate a DRL policy."
)
EXPLICIT_FEATURE_COLUMNS = [f"explicit_feature_{idx:02d}" for idx in range(17)]
LABEL_COLUMNS = [
    "current_farmland_label",
    "low_slope_farmland_label",
    "farmland_or_orchard_label",
]


def build_bishan_dltb_phase2_inputs(
    dltb_path: Path | str,
    metadata_path: Path | str,
    max_blocks: int | None = None,
) -> dict[str, object]:
    dltb_source = Path(dltb_path)
    metadata_source = Path(metadata_path)
    metadata = _read_metadata(metadata_source)
    bbox = tuple(float(value) for value in metadata["bbox"])
    grid_shape = tuple(int(value) for value in metadata["grid_shape"])
    gdf = _load_dltb(dltb_source, bbox)

    rows_read = len(gdf)
    gdf = _prepare_dltb_records(gdf, bbox, grid_shape)
    if max_blocks is not None:
        if int(max_blocks) <= 0:
            raise ValueError("max_blocks must be positive when provided")
        gdf = gdf.head(int(max_blocks)).copy()

    mapping_rows: list[dict[str, object]] = []
    attribute_rows: list[dict[str, object]] = []
    for record in gdf.to_dict(orient="records"):
        block_id = str(record["block_id"])
        mapping_rows.append(
            {
                "block_id": block_id,
                "row": int(record["row"]),
                "col": int(record["col"]),
                "weight": 1.0,
            }
        )
        attribute_rows.append(_build_attribute_row(record))

    summary = _build_summary(
        dltb_source,
        metadata_source,
        metadata,
        gdf,
        rows_read,
        mapping_rows,
        attribute_rows,
    )
    return {
        "mapping_rows": mapping_rows,
        "attribute_rows": attribute_rows,
        "summary": summary,
    }


def write_bishan_dltb_phase2_inputs(
    payload: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    mapping_rows = payload.get("mapping_rows")
    attribute_rows = payload.get("attribute_rows")
    summary = payload.get("summary")
    if not isinstance(mapping_rows, list):
        raise ValueError("Phase 11 payload is missing mapping_rows")
    if not isinstance(attribute_rows, list):
        raise ValueError("Phase 11 payload is missing attribute_rows")
    if not isinstance(summary, Mapping):
        raise ValueError("Phase 11 payload is missing summary")

    mapping_path = output_path / "block_pixel_mapping.csv"
    attributes_path = output_path / "block_attributes.csv"
    summary_path = output_path / "phase11_bishan_dltb_adapter_summary.json"
    _write_csv(mapping_path, mapping_rows, ["block_id", "row", "col", "weight"])
    _write_csv(attributes_path, attribute_rows, _attribute_fieldnames(attribute_rows))
    summary_path.write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "mapping_csv": mapping_path,
        "attributes_csv": attributes_path,
        "summary": summary_path,
    }


def _read_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Bishan AlphaEarth metadata: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    bbox = metadata.get("bbox")
    grid_shape = metadata.get("grid_shape")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("Bishan metadata must contain bbox with four values")
    if not isinstance(grid_shape, list) or len(grid_shape) != 2:
        raise ValueError("Bishan metadata must contain grid_shape with two values")
    return metadata


def _load_dltb(path: Path, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Bishan DLTB source: {path}")
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "Phase 11 requires geopandas to read DLTB geospatial data"
        ) from exc

    required_columns = [
        "BSM",
        "DLBM",
        "DLMC",
        "TBMJ",
        "category",
        "slope_mean",
        "slope_max",
        "slope_pixel_count",
    ]
    gdf = gpd.read_file(path, bbox=bbox)
    missing = [column for column in required_columns if column not in gdf.columns]
    if missing:
        raise ValueError(f"Bishan DLTB source is missing columns: {missing}")
    if gdf.empty:
        raise ValueError("No DLTB polygons intersect the Bishan AlphaEarth bbox")
    return gdf


def _prepare_dltb_records(
    gdf: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    grid_shape: tuple[int, int],
) -> pd.DataFrame:
    min_lon, min_lat, max_lon, max_lat = bbox
    prepared = gdf.copy()
    centroids = [geometry.centroid for geometry in prepared.geometry]
    prepared["centroid_x"] = [point.x for point in centroids]
    prepared["centroid_y"] = [point.y for point in centroids]
    inside = (
        (prepared["centroid_x"] >= min_lon)
        & (prepared["centroid_x"] <= max_lon)
        & (prepared["centroid_y"] >= min_lat)
        & (prepared["centroid_y"] <= max_lat)
    )
    prepared = prepared.loc[inside].copy()
    if prepared.empty:
        raise ValueError("No DLTB centroids fall inside the Bishan AlphaEarth bbox")

    rows_cols = [
        _assign_row_col(row.centroid_x, row.centroid_y, bbox, grid_shape)
        for row in prepared.itertuples()
    ]
    prepared["row"] = [row for row, _ in rows_cols]
    prepared["col"] = [col for _, col in rows_cols]
    prepared["block_id"] = [
        f"dltb_{_normalize_bsm(value)}" for value in prepared["BSM"].tolist()
    ]
    prepared = prepared.sort_values("block_id", kind="stable").reset_index(drop=True)
    return prepared


def _assign_row_col(
    centroid_x: float,
    centroid_y: float,
    bbox: tuple[float, float, float, float],
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    min_lon, min_lat, max_lon, max_lat = bbox
    n_rows, n_cols = grid_shape
    col = math.floor(((centroid_x - min_lon) / (max_lon - min_lon)) * n_cols)
    row = math.floor(((max_lat - centroid_y) / (max_lat - min_lat)) * n_rows)
    return _clip(row, 0, n_rows - 1), _clip(col, 0, n_cols - 1)


def _build_attribute_row(record: Mapping[str, object]) -> dict[str, object]:
    dlbm = str(record.get("DLBM", "")).strip()
    category = str(record.get("category", "")).strip()
    area_m2 = _float_or_zero(record.get("TBMJ"))
    slope_mean = _float_or_zero(record.get("slope_mean"))
    slope_max = _float_or_zero(record.get("slope_max"))
    slope_pixel_count = _float_or_zero(record.get("slope_pixel_count"))

    farmland = _is_farmland(dlbm, category)
    orchard = _is_orchard(dlbm, category)
    low_slope = slope_mean <= 6.0
    moderate_slope = 6.0 < slope_mean <= 15.0
    high_slope = slope_mean > 15.0
    farmland_or_orchard = farmland or orchard

    row: dict[str, object] = {
        "block_id": str(record["block_id"]),
        "explicit_feature_00": area_m2 / 10000.0,
        "explicit_feature_01": slope_mean,
        "explicit_feature_02": slope_max,
        "explicit_feature_03": slope_pixel_count,
        "explicit_feature_04": float(farmland),
        "explicit_feature_05": float(dlbm.startswith("011")),
        "explicit_feature_06": float(dlbm.startswith("013")),
        "explicit_feature_07": float(orchard),
        "explicit_feature_08": float(_is_forest(dlbm, category)),
        "explicit_feature_09": float(_is_built_up(dlbm)),
        "explicit_feature_10": float(_is_water(dlbm)),
        "explicit_feature_11": float(dlbm.startswith("122")),
        "explicit_feature_12": float(_is_grass_or_bare(dlbm)),
        "explicit_feature_13": float(low_slope),
        "explicit_feature_14": float(moderate_slope),
        "explicit_feature_15": float(high_slope),
        "explicit_feature_16": float(farmland_or_orchard and low_slope),
        "current_farmland_label": int(farmland),
        "low_slope_farmland_label": int(farmland and low_slope),
        "farmland_or_orchard_label": int(farmland_or_orchard),
        "split": _split_for_block(str(record["block_id"])),
        "source_bsm": _normalize_bsm(record.get("BSM")),
        "source_dlbm": dlbm,
        "source_dlmc": str(record.get("DLMC", "")).strip(),
        "source_category": category,
        "area_m2": area_m2,
        "slope_mean": slope_mean,
        "slope_max": slope_max,
        "slope_pixel_count": int(slope_pixel_count),
    }
    return row


def _build_summary(
    dltb_path: Path,
    metadata_path: Path,
    metadata: Mapping[str, object],
    gdf: pd.DataFrame,
    rows_read: int,
    mapping_rows: list[dict[str, object]],
    attribute_rows: list[dict[str, object]],
) -> dict[str, object]:
    label_positive_counts = {
        label: int(sum(int(row[label]) for row in attribute_rows))
        for label in LABEL_COLUMNS
    }
    slope_values = pd.Series([row["slope_mean"] for row in attribute_rows], dtype=float)
    return {
        "phase": "phase11_bishan_dltb_real_adapter",
        "dltb_path": str(dltb_path),
        "metadata_path": str(metadata_path),
        "bbox": list(metadata["bbox"]),
        "grid_shape": list(metadata["grid_shape"]),
        "rows_read_in_bbox": int(rows_read),
        "rows_exported": len(mapping_rows),
        "category_counts": _value_counts(gdf["category"]),
        "label_positive_counts": label_positive_counts,
        "slope_summary": {
            "min": round(float(slope_values.min()), 10),
            "max": round(float(slope_values.max()), 10),
            "mean": round(float(slope_values.mean()), 10),
            "median": round(float(slope_values.median()), 10),
        },
        "mapping_csv": "block_pixel_mapping.csv",
        "attributes_csv": "block_attributes.csv",
        "claim_boundary": PHASE11_CLAIM_BOUNDARY,
    }


def _is_farmland(dlbm: str, category: str) -> bool:
    return category.lower() == "farmland" or dlbm.startswith(("011", "012", "013"))


def _is_orchard(dlbm: str, category: str) -> bool:
    return category.lower() == "orchard" or dlbm.startswith(("021", "022", "023"))


def _is_forest(dlbm: str, category: str) -> bool:
    return category.lower() == "forest" or dlbm.startswith("03")


def _is_built_up(dlbm: str) -> bool:
    return dlbm.startswith(("20",))


def _is_water(dlbm: str) -> bool:
    return dlbm.startswith(("111", "112", "113", "114", "117"))


def _is_grass_or_bare(dlbm: str) -> bool:
    return dlbm.startswith(("04", "127"))


def _split_for_block(block_id: str) -> str:
    bucket = int(hashlib.md5(block_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 7:
        return "train"
    if bucket < 8:
        return "validation"
    return "test"


def _normalize_bsm(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().items()}


def _attribute_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    preferred = [
        "block_id",
        *EXPLICIT_FEATURE_COLUMNS,
        *LABEL_COLUMNS,
        "split",
        "source_bsm",
        "source_dlbm",
        "source_dlmc",
        "source_category",
        "area_m2",
        "slope_mean",
        "slope_max",
        "slope_pixel_count",
    ]
    extras = sorted({key for row in rows for key in row if key not in preferred})
    return preferred + extras


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _clip(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))
