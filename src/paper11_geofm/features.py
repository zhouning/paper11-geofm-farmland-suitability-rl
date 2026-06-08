from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .regions import iter_region_bounds


def _validate_embedding_grid(name: str, embedding: np.ndarray) -> None:
    if embedding.ndim != 3 or embedding.shape[-1] != 64:
        raise ValueError(f"{name} must have shape [rows, cols, 64], got {embedding.shape}")


def compute_region_features(
    base_embedding: np.ndarray,
    labels: np.ndarray,
    annual_embeddings: Mapping[int, np.ndarray] | None = None,
) -> list[dict[str, float | int]]:
    """Aggregate pixel embeddings into region-level feature rows."""
    _validate_embedding_grid("base_embedding", base_embedding)
    if labels.shape != base_embedding.shape[:2]:
        raise ValueError(
            f"labels shape {labels.shape} must match embedding grid {base_embedding.shape[:2]}"
        )

    if annual_embeddings is not None:
        for year, embedding in annual_embeddings.items():
            _validate_embedding_grid(f"annual_embeddings[{year}]", embedding)
            if embedding.shape != base_embedding.shape:
                raise ValueError(
                    f"annual_embeddings[{year}] shape {embedding.shape} "
                    f"must match base embedding shape {base_embedding.shape}"
                )

    rows: list[dict[str, float | int]] = []
    for bounds in iter_region_bounds(labels):
        region_id = bounds["region_id"]
        mask = labels == region_id
        pixels = np.asarray(base_embedding[mask], dtype=np.float64)
        mean_embedding = pixels.mean(axis=0)
        std_embedding = pixels.std(axis=0)

        row: dict[str, float | int] = dict(bounds)
        for dim, value in enumerate(mean_embedding):
            row[f"embedding_mean_{dim:02d}"] = float(value)
        row["embedding_std_mean"] = float(std_embedding.mean())
        row["temporal_stability"] = _compute_temporal_stability(mask, annual_embeddings)
        rows.append(row)

    return rows


def _compute_temporal_stability(
    mask: np.ndarray,
    annual_embeddings: Mapping[int, np.ndarray] | None,
) -> float:
    if not annual_embeddings:
        return 1.0

    region_year_means = []
    for year in sorted(annual_embeddings):
        pixels = np.asarray(annual_embeddings[year][mask], dtype=np.float64)
        region_year_means.append(pixels.mean(axis=0))

    year_means = np.vstack(region_year_means)
    temporal_variation = float(year_means.std(axis=0).mean())
    return float(1.0 / (1.0 + temporal_variation))
