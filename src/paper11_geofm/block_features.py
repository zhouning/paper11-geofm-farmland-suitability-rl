from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from .block_mapping import BlockPixel


def compute_block_geofm_features(
    base_embedding: np.ndarray,
    mapping: Sequence[BlockPixel],
    annual_embeddings: Mapping[int, np.ndarray] | None = None,
) -> list[dict[str, float | int | str]]:
    """Aggregate embedding-grid pixels into block-level GeoFM feature rows."""
    _validate_embedding_grid("base_embedding", base_embedding)
    if annual_embeddings is not None:
        for year, embedding in annual_embeddings.items():
            _validate_embedding_grid(f"annual_embeddings[{year}]", embedding)
            if embedding.shape != base_embedding.shape:
                raise ValueError(
                    f"annual_embeddings[{year}] shape {embedding.shape} "
                    f"must match base embedding shape {base_embedding.shape}"
                )

    grouped: dict[str, list[BlockPixel]] = defaultdict(list)
    for entry in mapping:
        grouped[str(entry["block_id"])].append(entry)

    rows: list[dict[str, float | int | str]] = []
    for block_id in sorted(grouped):
        entries = grouped[block_id]
        pixel_rows = np.array([int(entry["row"]) for entry in entries], dtype=np.int64)
        pixel_cols = np.array([int(entry["col"]) for entry in entries], dtype=np.int64)
        weights = np.array(
            [float(entry["weight"]) for entry in entries], dtype=np.float64
        )
        pixels = np.asarray(base_embedding[pixel_rows, pixel_cols], dtype=np.float64)
        weight_sum = float(weights.sum())
        mean_embedding = np.average(pixels, axis=0, weights=weights)
        centered = pixels - mean_embedding
        weighted_variance = np.average(centered * centered, axis=0, weights=weights)

        row: dict[str, float | int | str] = {
            "block_id": block_id,
            "pixel_count": int(len(entries)),
            "pixel_weight_sum": weight_sum,
            "row_min": int(pixel_rows.min()),
            "row_max": int(pixel_rows.max()),
            "col_min": int(pixel_cols.min()),
            "col_max": int(pixel_cols.max()),
            "embedding_std_mean": float(np.sqrt(weighted_variance).mean()),
            "temporal_stability": _compute_temporal_stability(
                entries, annual_embeddings
            ),
        }
        for dim, value in enumerate(mean_embedding):
            row[f"embedding_mean_{dim:02d}"] = float(value)
        rows.append(row)

    return rows


def attach_optional_block_attributes(
    rows: Sequence[Mapping[str, object]],
    attributes: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    """Join optional explicit planning features and labels by block_id."""
    if not attributes:
        return [dict(row) for row in rows]

    by_block = {str(row["block_id"]): dict(row) for row in attributes}
    joined: list[dict[str, object]] = []
    for row in rows:
        block_id = str(row["block_id"])
        output = dict(row)
        extra = by_block.get(block_id, {})
        for key, value in extra.items():
            if key != "block_id":
                output[key] = value
        joined.append(output)
    return joined


def _validate_embedding_grid(name: str, embedding: np.ndarray) -> None:
    if embedding.ndim != 3 or embedding.shape[-1] != 64:
        raise ValueError(f"{name} must have shape [rows, cols, 64], got {embedding.shape}")


def _compute_temporal_stability(
    entries: Sequence[BlockPixel],
    annual_embeddings: Mapping[int, np.ndarray] | None,
) -> float:
    if not annual_embeddings:
        return 1.0

    pixel_rows = np.array([int(entry["row"]) for entry in entries], dtype=np.int64)
    pixel_cols = np.array([int(entry["col"]) for entry in entries], dtype=np.int64)
    weights = np.array([float(entry["weight"]) for entry in entries], dtype=np.float64)
    year_means = []
    for year in sorted(annual_embeddings):
        pixels = np.asarray(
            annual_embeddings[year][pixel_rows, pixel_cols], dtype=np.float64
        )
        year_means.append(np.average(pixels, axis=0, weights=weights))

    temporal_variation = float(np.vstack(year_means).std(axis=0).mean())
    return float(1.0 / (1.0 + temporal_variation))
