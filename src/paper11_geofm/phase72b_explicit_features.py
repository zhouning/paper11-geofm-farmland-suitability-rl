from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .phase72a_label_sources import Phase72ARegionSpec


ESRI_CLASS_CODES = (1, 2, 4, 5, 7, 8, 9, 10, 11)
TERRAIN_FEATURES = (
    "elevation_mean",
    "elevation_std",
    "elevation_min",
    "elevation_max",
    "slope_mean",
    "slope_std",
    "slope_max",
    "local_relief",
)


def _window(
    array: np.ndarray, row: int, col: int, radius: int
) -> np.ndarray:
    row_start = max(0, int(row) - int(radius))
    row_stop = min(array.shape[0], int(row) + int(radius) + 1)
    col_start = max(0, int(col) - int(radius))
    col_stop = min(array.shape[1], int(col) + int(radius) + 1)
    return np.asarray(array[row_start:row_stop, col_start:col_stop])


def _linear_trend(values: Sequence[float]) -> float:
    data = np.asarray(values, dtype=np.float64)
    if data.size <= 1:
        return 0.0
    x = np.arange(data.size, dtype=np.float64)
    centered = x - float(x.mean())
    denominator = float(np.sum(centered**2))
    if denominator <= 0:
        return 0.0
    return float(np.sum(centered * data) / denominator)


def _class_count_names() -> list[str]:
    return [
        *[
            f"cell_history_count_lulc_{code:02d}"
            for code in ESRI_CLASS_CODES
        ],
        "cell_history_count_lulc_unknown",
    ]


def _neighbor_names(prefix: str) -> list[str]:
    return [
        *[
            f"{prefix}_current_fraction_lulc_{code:02d}"
            for code in ESRI_CLASS_CODES
        ],
        f"{prefix}_current_fraction_lulc_unknown",
        f"{prefix}_current_crop_fraction",
    ]


def build_phase72b_explicit_features(
    sample_rows: Sequence[Mapping[str, object]],
    *,
    regions: Mapping[str, Phase72ARegionSpec],
    labels: Mapping[str, Mapping[int, np.ndarray]],
    terrain: Mapping[str, Mapping[str, np.ndarray]],
    crop_class_code: int = 5,
) -> dict[str, object]:
    if not sample_rows:
        raise ValueError("Phase 72B explicit features require sample rows")
    indexes = [int(row["sample_index"]) for row in sample_rows]
    if indexes != list(range(len(sample_rows))):
        raise ValueError("Phase 72B sample indexes must be contiguous")

    static_names = [
        *[f"terrain_{name}" for name in TERRAIN_FEATURES],
        "cell_longitude",
        "cell_latitude",
        "cell_row_normalized",
        "cell_col_normalized",
        "region_index",
        "origin_year",
        "history_length",
    ]
    history_names = [
        *static_names,
        "previous_lulc_class",
        "previous_crop_flag",
        "cell_historical_crop_fraction",
        "cell_crop_transition_count",
        "cell_years_since_last_non_crop",
        *_class_count_names(),
        *_neighbor_names("neighbor3"),
        *_neighbor_names("neighbor5"),
        "neighbor3_historical_crop_mean",
        "neighbor3_historical_crop_trend",
        "neighbor5_historical_crop_mean",
        "neighbor5_historical_crop_trend",
    ]
    region_order = {
        name: index for index, name in enumerate(sorted(regions))
    }
    static_rows = []
    history_rows = []
    label_arrays: dict[str, dict[int, np.ndarray]] = {}
    terrain_arrays: dict[str, dict[str, np.ndarray]] = {}

    for row in sample_rows:
        region_id = str(row["region_id"])
        if region_id not in regions:
            raise ValueError(f"Unknown Phase 72B region: {region_id}")
        spec = regions[region_id]
        grid_row = int(row["row"])
        grid_col = int(row["col"])
        if not (
            0 <= grid_row < spec.grid_shape[0]
            and 0 <= grid_col < spec.grid_shape[1]
        ):
            raise ValueError(
                "Phase 72B sample is outside grid bounds: "
                f"{region_id} ({grid_row}, {grid_col})"
            )
        origin = int(row["origin_year"])
        if origin not in spec.years:
            raise ValueError(
                f"Phase 72B origin year is outside the region contract: "
                f"{region_id} {origin}"
            )
        years = [year for year in spec.years if year <= origin]
        if len(years) != int(row["history_length"]):
            raise ValueError(
                f"Phase 72B history length mismatch at sample {row['sample_index']}"
            )
        if region_id not in labels:
            raise ValueError(f"Missing Phase 72B LULC region: {region_id}")
        region_labels = label_arrays.setdefault(region_id, {})
        missing_years = [
            year for year in years if year not in labels[region_id]
        ]
        if missing_years:
            raise ValueError(
                "Missing Phase 72B LULC history for "
                f"{region_id}: {missing_years}"
            )
        for year in years:
            if year in region_labels:
                continue
            label_array = np.asarray(labels[region_id][year])
            if tuple(label_array.shape) != spec.grid_shape:
                raise ValueError(
                    "Phase 72B LULC shape mismatch: "
                    f"{region_id} {year}; expected {spec.grid_shape}, "
                    f"got {tuple(label_array.shape)}"
                )
            if not np.isfinite(label_array).all():
                raise ValueError(
                    f"Phase 72B LULC values must be finite: "
                    f"{region_id} {year}"
                )
            region_labels[year] = label_array
        cell_history = [
            int(region_labels[year][grid_row, grid_col])
            for year in years
        ]
        if cell_history[-1] != int(crop_class_code):
            raise ValueError(
                "Phase 72B sample is outside the origin-year crop cohort: "
                f"{region_id} {origin} ({grid_row}, {grid_col})"
            )

        min_lon, min_lat, max_lon, max_lat = spec.bbox
        lon = min_lon + (grid_col + 0.5) / spec.grid_shape[1] * (
            max_lon - min_lon
        )
        lat = max_lat - (grid_row + 0.5) / spec.grid_shape[0] * (
            max_lat - min_lat
        )
        if region_id not in terrain_arrays:
            if region_id not in terrain:
                raise ValueError(
                    f"Missing Phase 72B terrain region: {region_id}"
                )
            validated_terrain = {}
            for name in TERRAIN_FEATURES:
                if name not in terrain[region_id]:
                    raise ValueError(
                        "Missing Phase 72B terrain feature: "
                        f"{region_id} {name}"
                    )
                array = np.asarray(terrain[region_id][name])
                if tuple(array.shape) != spec.grid_shape:
                    raise ValueError(
                        "Phase 72B terrain shape mismatch: "
                        f"{region_id} {name}"
                    )
                validated_terrain[name] = array
            terrain_arrays[region_id] = validated_terrain
        terrain_values = [
            float(terrain_arrays[region_id][name][grid_row, grid_col])
            for name in TERRAIN_FEATURES
        ]
        static = [
            *terrain_values,
            float(lon),
            float(lat),
            grid_row / max(1, spec.grid_shape[0] - 1),
            grid_col / max(1, spec.grid_shape[1] - 1),
            float(region_order[region_id]),
            float(origin),
            float(len(years)),
        ]

        previous = cell_history[-2] if len(cell_history) >= 2 else -1
        transitions = sum(
            int(first != second)
            for first, second in zip(cell_history[:-1], cell_history[1:])
        )
        non_crop_indexes = [
            index
            for index, value in enumerate(cell_history)
            if value != int(crop_class_code)
        ]
        since_non_crop = (
            len(cell_history) - 1 - non_crop_indexes[-1]
            if non_crop_indexes
            else len(cell_history)
        )
        counts = [
            float(sum(value == code for value in cell_history))
            for code in ESRI_CLASS_CODES
        ]
        counts.append(
            float(
                sum(value not in ESRI_CLASS_CODES for value in cell_history)
            )
        )

        current = region_labels[origin]
        neighbor_values = []
        historical_neighbor_values = []
        for radius in (1, 2):
            current_window = _window(current, grid_row, grid_col, radius)
            neighbor_values.extend(
                [
                    float(np.mean(current_window == code))
                    for code in ESRI_CLASS_CODES
                ]
            )
            neighbor_values.append(
                float(np.mean(~np.isin(current_window, ESRI_CLASS_CODES)))
            )
            neighbor_values.append(
                float(np.mean(current_window == int(crop_class_code)))
            )
            annual_crop_fractions = [
                float(
                    np.mean(
                        _window(
                            region_labels[year],
                            grid_row,
                            grid_col,
                            radius,
                        )
                        == int(crop_class_code)
                    )
                )
                for year in years
            ]
            historical_neighbor_values.extend(
                [
                    float(np.mean(annual_crop_fractions)),
                    _linear_trend(annual_crop_fractions),
                ]
            )

        history = [
            *static,
            float(previous),
            float(previous == int(crop_class_code)),
            float(
                np.mean(
                    np.asarray(cell_history, dtype=np.int16)
                    == int(crop_class_code)
                )
            ),
            float(transitions),
            float(since_non_crop),
            *counts,
            *neighbor_values,
            *historical_neighbor_values,
        ]
        static_rows.append(static)
        history_rows.append(history)

    static_matrix = np.asarray(static_rows, dtype=np.float32)
    history_matrix = np.asarray(history_rows, dtype=np.float32)
    if static_matrix.shape[1] != len(static_names):
        raise ValueError("Phase 72B static feature registry mismatch")
    if history_matrix.shape[1] != len(history_names):
        raise ValueError("Phase 72B history feature registry mismatch")
    if not np.isfinite(static_matrix).all() or not np.isfinite(
        history_matrix
    ).all():
        raise ValueError("Phase 72B explicit features must be finite")
    return {
        "explicit_static": static_matrix,
        "explicit_history": history_matrix,
        "registry": {
            "explicit_static": static_names,
            "explicit_history": history_names,
        },
    }
