from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _vector_trend(observed: np.ndarray) -> np.ndarray:
    values = np.asarray(observed, dtype=np.float64)
    if len(values) <= 1:
        return np.zeros(values.shape[1], dtype=np.float32)
    x = np.arange(len(values), dtype=np.float64)
    centered = x - float(x.mean())
    denominator = float(np.sum(centered**2))
    if denominator <= 0:
        return np.zeros(values.shape[1], dtype=np.float32)
    return np.asarray(
        np.sum(centered[:, None] * values, axis=0) / denominator,
        dtype=np.float32,
    )


def build_phase72b_geofm_features(
    embedding_history: np.ndarray, history_mask: np.ndarray
) -> dict[str, np.ndarray]:
    history = np.asarray(embedding_history, dtype=np.float32)
    mask = np.asarray(history_mask, dtype=bool)
    if history.ndim != 3 or mask.shape != history.shape[:2]:
        raise ValueError("Invalid Phase 72B embedding history")

    current = []
    means = []
    stds = []
    deltas = []
    trends = []
    for values, valid in zip(history, mask):
        observed = values[valid]
        if len(observed) == 0:
            raise ValueError("Phase 72B history cannot be empty")
        current.append(observed[-1])
        means.append(observed.mean(axis=0))
        stds.append(observed.std(axis=0))
        deltas.append(observed[-1] - observed[0])
        trends.append(_vector_trend(observed))

    current_matrix = np.asarray(current, dtype=np.float32)
    mean_matrix = np.asarray(means, dtype=np.float32)
    full = np.concatenate(
        [
            current_matrix,
            mean_matrix,
            np.asarray(stds, dtype=np.float32),
            np.asarray(deltas, dtype=np.float32),
            np.asarray(trends, dtype=np.float32),
        ],
        axis=1,
    )
    if not np.isfinite(full).all():
        raise ValueError("Phase 72B GeoFM features must be finite")
    return {
        "geofm_current": current_matrix,
        "geofm_temporal_mean": mean_matrix,
        "geofm_temporal_full": np.asarray(full, dtype=np.float32),
    }


def build_phase72b_control_features(
    control_id: str,
    embedding_history: np.ndarray,
    history_mask: np.ndarray,
    sample_rows: Sequence[Mapping[str, object]],
    *,
    seed: int,
    output_dim: int,
) -> np.ndarray:
    history = np.asarray(embedding_history, dtype=np.float32).copy()
    mask = np.asarray(history_mask, dtype=bool)
    if history.ndim != 3 or mask.shape != history.shape[:2]:
        raise ValueError("Invalid Phase 72B control history")
    if len(sample_rows) != len(history):
        raise ValueError("Phase 72B control rows must align with history")
    rng = np.random.default_rng(int(seed))

    if control_id == "temporal_order_shuffle":
        for row_index in range(len(history)):
            valid_indexes = np.flatnonzero(mask[row_index])
            earlier_positions = valid_indexes[:-1]
            if len(earlier_positions) > 1:
                original = history[row_index, earlier_positions].copy()
                history[row_index, earlier_positions] = original[
                    rng.permutation(len(earlier_positions))
                ]
        result = build_phase72b_geofm_features(history, mask)[
            "geofm_temporal_full"
        ]
    elif control_id == "spatial_shuffle":
        groups: dict[tuple[str, int], list[int]] = {}
        for index, row in enumerate(sample_rows):
            key = (str(row["region_id"]), int(row["origin_year"]))
            groups.setdefault(key, []).append(index)
        shuffled = history.copy()
        for indexes in groups.values():
            source = np.asarray(indexes, dtype=np.int64)
            group_masks = mask[source]
            if not np.all(group_masks == group_masks[0]):
                raise ValueError(
                    "Phase 72B spatial shuffle stratum has unequal masks"
                )
            shuffled[source] = history[rng.permutation(source)]
        result = build_phase72b_geofm_features(shuffled, mask)[
            "geofm_temporal_full"
        ]
    elif control_id == "random_projection":
        flattened = (history * mask[..., None]).reshape(len(history), -1)
        if int(output_dim) <= 0 or int(output_dim) > flattened.shape[1]:
            raise ValueError(
                "Phase 72B random projection dimension must be positive and "
                "no larger than flattened history"
            )
        matrix = rng.normal(
            size=(flattened.shape[1], int(output_dim))
        )
        q, _ = np.linalg.qr(matrix, mode="reduced")
        result = np.asarray(
            flattened @ q[:, : int(output_dim)], dtype=np.float32
        )
    else:
        raise ValueError(f"Unknown Phase 72B control: {control_id}")

    if result.shape[1] != int(output_dim):
        raise ValueError(
            f"Phase 72B control dimension mismatch: {result.shape[1]}"
        )
    if not np.isfinite(result).all():
        raise ValueError("Phase 72B control features must be finite")
    return np.asarray(result, dtype=np.float32)
