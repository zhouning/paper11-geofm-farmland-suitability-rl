from __future__ import annotations

from typing import Mapping

import numpy as np

from .phase72a_label_sources import (
    PHASE72A_CLAIM_BOUNDARY,
    Phase72ARegionSpec,
)


def build_phase72a_temporal_samples(
    region: Phase72ARegionSpec,
    *,
    embeddings: Mapping[int, np.ndarray],
    labels: Mapping[int, np.ndarray],
    crop_class_code: int,
    source_id: str,
    source_role: str,
    max_history_years: int,
    spatial_block_size: int,
) -> dict[str, object]:
    if max_history_years < len(region.years):
        raise ValueError("max_history_years must cover the contract history")

    rows = []
    histories = []
    masks = []
    y1_values = []
    y2_values = []
    continuous_values = []
    row_values = []
    col_values = []
    origin_values = []
    years = list(region.years)

    for origin_offset, origin_year in enumerate(years[:-1]):
        for grid_row, grid_col in np.argwhere(
            np.asarray(labels[origin_year]) == int(crop_class_code)
        ):
            history_years = years[: origin_offset + 1]
            history = np.zeros(
                (max_history_years, region.embedding_dim),
                dtype=np.float32,
            )
            mask = np.zeros(max_history_years, dtype=bool)
            for history_offset, year in enumerate(history_years):
                history[history_offset] = embeddings[year][
                    grid_row, grid_col
                ]
                mask[history_offset] = True

            y1 = int(
                labels[origin_year + 1][grid_row, grid_col]
                == crop_class_code
            )
            has_2y = origin_offset + 2 < len(years)
            y2 = (
                int(
                    labels[years[origin_offset + 2]][grid_row, grid_col]
                    == crop_class_code
                )
                if has_2y
                else -1
            )
            continuous = int(y1 == 1 and y2 == 1) if has_2y else -1
            sample_index = len(rows)
            rows.append(
                {
                    "sample_index": sample_index,
                    "region_id": region.region_id,
                    "unit_id": (
                        f"r{int(grid_row):04d}_c{int(grid_col):04d}"
                    ),
                    "row": int(grid_row),
                    "col": int(grid_col),
                    "spatial_block_id": (
                        f"{region.region_id}_"
                        f"br{int(grid_row) // spatial_block_size:03d}_"
                        f"bc{int(grid_col) // spatial_block_size:03d}"
                    ),
                    "origin_year": int(origin_year),
                    "history_start_year": int(history_years[0]),
                    "history_end_year": int(origin_year),
                    "history_length": len(history_years),
                    "current_lulc_class": int(crop_class_code),
                    "target_year_1y": int(origin_year + 1),
                    "y_1y": y1,
                    "target_year_2y": (
                        int(years[origin_offset + 2]) if has_2y else ""
                    ),
                    "y_2y": y2 if has_2y else "",
                    "y_continuous_2y": continuous if has_2y else "",
                    "label_source_id": source_id,
                    "label_source_role": source_role,
                    "label_confidence": "product_label",
                    "claim_boundary": PHASE72A_CLAIM_BOUNDARY,
                }
            )
            histories.append(history)
            masks.append(mask)
            y1_values.append(y1)
            y2_values.append(y2)
            continuous_values.append(continuous)
            row_values.append(int(grid_row))
            col_values.append(int(grid_col))
            origin_values.append(int(origin_year))

    if not rows:
        raise ValueError(
            f"Phase 72A region has no farmland samples: {region.region_id}"
        )

    tensors = {
        "embedding_history": np.stack(histories).astype(np.float32),
        "history_mask": np.stack(masks).astype(bool),
        "origin_year": np.asarray(origin_values, dtype=np.int16),
        "current_lulc_class": np.full(
            len(rows), crop_class_code, dtype=np.int16
        ),
        "y_1y": np.asarray(y1_values, dtype=np.int8),
        "y_2y": np.asarray(y2_values, dtype=np.int8),
        "y_continuous_2y": np.asarray(
            continuous_values, dtype=np.int8
        ),
        "row": np.asarray(row_values, dtype=np.int16),
        "col": np.asarray(col_values, dtype=np.int16),
    }
    return {"sample_rows": rows, "tensors": tensors}
