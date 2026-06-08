from __future__ import annotations

import numpy as np


EMBEDDING_COLUMNS = [f"embedding_mean_{idx:02d}" for idx in range(64)]


def add_suitability_proxy(
    rows: list[dict[str, float | int]]
) -> list[dict[str, float | int]]:
    """Add a bounded latent remote-sensing suitability proxy to each row."""
    if not rows:
        return []

    embeddings = np.array(
        [[float(row[column]) for column in EMBEDDING_COLUMNS] for row in rows],
        dtype=np.float64,
    )
    centroid = embeddings.mean(axis=0)
    similarities = _cosine_similarity_to_centroid(embeddings, centroid)
    dispersion = np.array(
        [float(row.get("embedding_std_mean", 0.0)) for row in rows], dtype=np.float64
    )
    stability = np.array(
        [float(row.get("temporal_stability", 1.0)) for row in rows], dtype=np.float64
    )

    combined = (
        0.60 * _minmax(similarities)
        + 0.25 * _minmax(stability)
        + 0.15 * _minmax(1.0 / (1.0 + np.maximum(dispersion, 0.0)))
    )
    scores = _minmax(combined)

    scored_rows: list[dict[str, float | int]] = []
    for row, score in zip(rows, scores, strict=True):
        scored = dict(row)
        scored["suitability_proxy"] = float(np.clip(score, 0.0, 1.0))
        scored_rows.append(scored)
    return scored_rows


def _cosine_similarity_to_centroid(
    embeddings: np.ndarray, centroid: np.ndarray
) -> np.ndarray:
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    centroid_norm = float(np.linalg.norm(centroid))
    denom = embedding_norms * centroid_norm
    similarities = np.zeros(embeddings.shape[0], dtype=np.float64)
    valid = denom > 0
    similarities[valid] = embeddings[valid] @ centroid / denom[valid]
    return similarities


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if np.isclose(max_value, min_value):
        return np.full(values.shape, 0.5, dtype=np.float64)
    return (values - min_value) / (max_value - min_value)
