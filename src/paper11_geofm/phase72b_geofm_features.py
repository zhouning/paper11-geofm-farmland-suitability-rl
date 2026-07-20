from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json

import numpy as np


def _derived_seed(seed: int, *parts: object) -> int:
    payload = json.dumps(
        [int(seed), *[str(value) for value in parts]],
        separators=(",", ":"),
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


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
    if (
        history.ndim != 3
        or len(history) == 0
        or mask.shape != history.shape[:2]
    ):
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


def build_phase72b_random_projection(
    *, input_dim: int, output_dim: int, seed: int
) -> np.ndarray:
    input_size = int(input_dim)
    output_size = int(output_dim)
    if input_size <= 0 or output_size <= 0 or output_size > input_size:
        raise ValueError(
            "Phase 72B random projection requires "
            "0 < output_dim <= input_dim"
        )
    rng = np.random.default_rng(int(seed))
    matrix = rng.normal(size=(input_size, output_size))
    q, r = np.linalg.qr(matrix, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return np.asarray(q[:, :output_size] * signs, dtype=np.float32)


def build_phase72b_control_features(
    control_id: str,
    embedding_history: np.ndarray,
    history_mask: np.ndarray,
    sample_rows: Sequence[Mapping[str, object]],
    *,
    partition_ids: Sequence[str] | None = None,
    seed: int,
    output_dim: int,
    learned_transform_fit_scope: str = "training_rows_only",
) -> dict[str, object]:
    history = np.asarray(embedding_history, dtype=np.float32).copy()
    mask = np.asarray(history_mask, dtype=bool)
    if history.ndim != 3 or mask.shape != history.shape[:2]:
        raise ValueError("Invalid Phase 72B control history")
    if partition_ids is None:
        raise ValueError("Phase 72B controls require partition IDs")
    if len(sample_rows) != len(history) or len(partition_ids) != len(history):
        raise ValueError(
            "Phase 72B controls require aligned histories, rows, and "
            "partitions"
        )
    partitions = np.asarray(
        ["" if value is None else str(value) for value in partition_ids],
        dtype=object,
    )
    if any(not value.strip() for value in partitions.tolist()):
        raise ValueError(
            "Phase 72B controls require nonblank partition IDs"
        )
    if learned_transform_fit_scope != "training_rows_only":
        raise ValueError(
            "Phase 72B learned transforms must fit training rows only"
        )
    try:
        sample_ids = [int(row["sample_index"]) for row in sample_rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Phase 72B controls require integer sample indexes"
        ) from exc
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Phase 72B control sample indexes must be unique")

    control_name = str(control_id)
    expected_temporal_dim = int(history.shape[2]) * 5
    if (
        control_name in {"temporal_order_shuffle", "spatial_shuffle"}
        and int(output_dim) != expected_temporal_dim
    ):
        raise ValueError(
            "Phase 72B shuffled control dimension must match the full "
            f"temporal dimension: {expected_temporal_dim}"
        )
    source_by_target = list(range(len(history)))

    if control_name == "temporal_order_shuffle":
        for row_index in range(len(history)):
            valid_indexes = np.flatnonzero(mask[row_index])
            earlier_positions = valid_indexes[:-1]
            if len(earlier_positions) > 1:
                original = history[row_index, earlier_positions].copy()
                row_rng = np.random.default_rng(
                    _derived_seed(
                        seed,
                        control_name,
                        partitions[row_index],
                        sample_ids[row_index],
                    )
                )
                permutation = row_rng.permutation(len(earlier_positions))
                if np.array_equal(
                    permutation, np.arange(len(earlier_positions))
                ):
                    permutation = np.roll(permutation, 1)
                history[row_index, earlier_positions] = original[permutation]
        result = build_phase72b_geofm_features(history, mask)[
            "geofm_temporal_full"
        ]
    elif control_name == "spatial_shuffle":
        groups: dict[tuple[str, str, int], list[int]] = {}
        for index, row in enumerate(sample_rows):
            key = (
                str(partitions[index]),
                str(row["region_id"]),
                int(row["origin_year"]),
            )
            groups.setdefault(key, []).append(index)
        shuffled = history.copy()
        shuffled_mask = mask.copy()
        for group_key, indexes in groups.items():
            source = np.asarray(indexes, dtype=np.int64)
            group_rng = np.random.default_rng(
                _derived_seed(seed, control_name, *group_key)
            )
            permuted = group_rng.permutation(source)
            if len(source) > 1 and np.array_equal(source, permuted):
                permuted = np.roll(permuted, 1)
            shuffled[source] = history[permuted]
            shuffled_mask[source] = mask[permuted]
            for target, source_index in zip(
                source.tolist(), permuted.tolist()
            ):
                source_by_target[target] = source_index
        result = build_phase72b_geofm_features(shuffled, shuffled_mask)[
            "geofm_temporal_full"
        ]
    elif control_name == "random_projection":
        flattened = np.where(mask[..., None], history, 0.0).reshape(
            len(history), -1
        )
        projection = build_phase72b_random_projection(
            input_dim=flattened.shape[1],
            output_dim=int(output_dim),
            seed=int(seed),
        )
        result = np.asarray(
            flattened @ projection, dtype=np.float32
        )
    else:
        raise ValueError(f"Unknown Phase 72B control: {control_name}")

    if result.shape[1] != int(output_dim):
        raise ValueError(
            f"Phase 72B control dimension mismatch: {result.shape[1]}"
        )
    if not np.isfinite(result).all():
        raise ValueError("Phase 72B control features must be finite")
    cross_partition_count = sum(
        partitions[target] != partitions[source]
        for target, source in enumerate(source_by_target)
    )
    if cross_partition_count:
        raise ValueError("Phase 72B control crossed a split partition")
    return {
        "matrix": np.asarray(result, dtype=np.float32),
        "manifest": {
            "control_id": control_name,
            "seed": int(seed),
            "partition_ids": sorted(set(partitions.tolist())),
            "data_dependent": control_name
            in {"temporal_order_shuffle", "spatial_shuffle"},
            "learned_transform_fit_scope": learned_transform_fit_scope,
            "source_index_by_target": source_by_target,
            "cross_partition_count": int(cross_partition_count),
        },
    }
