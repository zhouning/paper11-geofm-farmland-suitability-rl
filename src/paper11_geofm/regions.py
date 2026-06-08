from __future__ import annotations

import numpy as np


def make_grid_region_labels(
    grid_shape: tuple[int, int],
    n_row_bins: int = 5,
    n_col_bins: int = 5,
) -> np.ndarray:
    """Create deterministic row-major grid region labels."""
    rows, cols = grid_shape
    if rows <= 0 or cols <= 0:
        raise ValueError(f"grid_shape must be positive, got {grid_shape}")
    if n_row_bins <= 0 or n_col_bins <= 0:
        raise ValueError("n_row_bins and n_col_bins must be positive")
    if n_row_bins > rows or n_col_bins > cols:
        raise ValueError("region bins cannot exceed grid dimensions")

    row_edges = np.linspace(0, rows, n_row_bins + 1, dtype=int)
    col_edges = np.linspace(0, cols, n_col_bins + 1, dtype=int)
    labels = np.empty((rows, cols), dtype=np.int32)

    region_id = 0
    for row_idx in range(n_row_bins):
        r0, r1 = int(row_edges[row_idx]), int(row_edges[row_idx + 1])
        for col_idx in range(n_col_bins):
            c0, c1 = int(col_edges[col_idx]), int(col_edges[col_idx + 1])
            labels[r0:r1, c0:c1] = region_id
            region_id += 1

    return labels


def iter_region_bounds(labels: np.ndarray) -> list[dict[str, int]]:
    """Return bounds and pixel counts for each contiguous region ID."""
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D, got shape {labels.shape}")

    bounds = []
    for region_id in sorted(int(value) for value in np.unique(labels)):
        rows, cols = np.where(labels == region_id)
        if rows.size == 0:
            continue
        bounds.append(
            {
                "region_id": region_id,
                "pixel_count": int(rows.size),
                "row_min": int(rows.min()),
                "row_max": int(rows.max()),
                "col_min": int(cols.min()),
                "col_max": int(cols.max()),
            }
        )
    return bounds
