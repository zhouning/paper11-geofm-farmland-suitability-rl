from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import math
from os import PathLike
from pathlib import Path
import statistics

import numpy as np

from .block_schema import EXPLICIT_FEATURE_COLUMNS


PHASE61_CLAIM_BOUNDARY = (
    "Phase 61 is a read-only D6 GeoFM projection-control preparation and "
    "geometry audit. It builds GeoFM-derived same-dimension projection controls "
    "for later matched training; it does not train PPO policies, does not "
    "compare learned rewards, does not enable suitability reward, does not test "
    "B2/B3, does not test transfer, and does not validate independent agronomic "
    "suitability."
)

PHASE61_GEOMETRY_FIELDNAMES = [
    "variant_id",
    "projection_type",
    "row_count",
    "projection_dimension",
    "total_centered_variance",
    "raw_variance_retention",
    "effective_rank",
    "positive_variance_column_count",
    "d4_reference_variant_id",
    "d4_mean_abs_column_correlation",
    "claim_boundary",
]

PHASE61_SIMILARITY_FIELDNAMES = [
    "variant_id",
    "reference_variant_id",
    "mean_abs_column_correlation",
    "claim_boundary",
]


def build_phase61_d6_projection_controls(
    b0_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    b1_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    d4p8_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    d4p16_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    dimensions: Sequence[int] | str = (8, 16),
    seed: int = 61,
) -> dict[str, object]:
    b0_rows = _load_rows(b0_rows_or_csv, "B0")
    b1_rows = _load_rows(b1_rows_or_csv, "B1")
    d4p8_rows = _load_rows(d4p8_rows_or_csv, "D4P8")
    d4p16_rows = _load_rows(d4p16_rows_or_csv, "D4P16")
    _require_aligned_block_ids(b0_rows, b1_rows, "B1")
    _require_aligned_block_ids(b0_rows, d4p8_rows, "D4P8")
    _require_aligned_block_ids(b0_rows, d4p16_rows, "D4P16")

    normalized_dimensions = _normalize_dimensions(dimensions)
    explicit_columns = _available_explicit_columns(b0_rows)
    explicit_matrix = _matrix_for_columns(b0_rows, explicit_columns)
    b1_columns = _numeric_columns(b1_rows, "embedding_mean_", "B1")
    raw_matrix = _matrix_for_columns(b1_rows, b1_columns)
    centered_raw = _centered(raw_matrix)
    raw_variance = _total_centered_variance(raw_matrix)
    rng = np.random.default_rng(int(seed))

    d4_matrices = {
        8: _matrix_for_prefix_if_available(d4p8_rows, "embedding_pca_"),
        16: _matrix_for_prefix_if_available(d4p16_rows, "embedding_pca_"),
    }
    d4_variant_ids = {8: "D4P8", 16: "D4P16"}
    if len(normalized_dimensions) >= 1:
        d4_matrices.setdefault(normalized_dimensions[0], d4_matrices[8])
        d4_variant_ids.setdefault(normalized_dimensions[0], "D4P8")
    if len(normalized_dimensions) >= 2:
        d4_matrices.setdefault(normalized_dimensions[1], d4_matrices[16])
        d4_variant_ids.setdefault(normalized_dimensions[1], "D4P16")

    variant_ids: list[str] = []
    variant_tables: dict[str, list[dict[str, object]]] = {}
    summary: dict[str, dict[str, object]] = {}
    geometry_rows: list[dict[str, object]] = []
    similarity_rows: list[dict[str, object]] = []

    for dimension in normalized_dimensions:
        projections = {
            f"D6R{dimension}": (
                "random_orthonormal_raw_b1_projection",
                _random_orthonormal_projection(centered_raw, dimension, rng),
            ),
            f"D6P{dimension}": (
                "pca_raw_b1_projection",
                _pca_projection(centered_raw, dimension),
            ),
        }
        for variant_id, (projection_type, projection_matrix) in projections.items():
            variant_ids.append(variant_id)
            table = _build_variant_rows(
                b0_rows,
                explicit_columns,
                explicit_matrix,
                projection_matrix,
            )
            reference_variant_id = str(d4_variant_ids.get(dimension, ""))
            reference_matrix = d4_matrices.get(dimension)
            similarity = _mean_abs_column_correlation(
                projection_matrix,
                reference_matrix,
            )
            variant_tables[variant_id] = table
            projection_columns = _projection_columns(dimension)
            summary[variant_id] = {
                "row_count": len(table),
                "projection_dimension": int(dimension),
                "projection_type": projection_type,
                "feature_table": f"variant_{variant_id}_features.csv",
                "required_columns": list(explicit_columns) + projection_columns,
                "d4_reference_variant_id": reference_variant_id,
                "d4_mean_abs_column_correlation": _round_optional_float(similarity),
            }
            geometry = _geometry_row(
                variant_id,
                projection_type,
                projection_matrix,
                raw_variance,
                reference_variant_id,
                similarity,
            )
            geometry_rows.append(geometry)
            similarity_rows.append(
                {
                    "variant_id": variant_id,
                    "reference_variant_id": reference_variant_id,
                    "mean_abs_column_correlation": _round_optional_float(similarity),
                    "claim_boundary": PHASE61_CLAIM_BOUNDARY,
                }
            )

    manifest = _build_manifest(summary)
    status = _phase61_status(geometry_rows, summary, row_count=len(b0_rows))
    return {
        "phase": "phase61_d6_projection_control_features",
        "phase61_d6_projection_status": status,
        "seed": int(seed),
        "variant_ids": variant_ids,
        "dimensions": normalized_dimensions,
        "explicit_columns": list(explicit_columns),
        "raw_embedding_columns": b1_columns,
        "row_alignment": {
            "row_count": len(b0_rows),
            "all_rows_align": True,
        },
        "summary": summary,
        "manifest": manifest,
        "variant_tables": variant_tables,
        "geometry_rows": geometry_rows,
        "similarity_rows": similarity_rows,
        "conclusion": _phase61_conclusion(status),
        "claim_boundary": PHASE61_CLAIM_BOUNDARY,
    }


def _load_rows(
    rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    variant_id: str,
) -> list[dict[str, object]]:
    if isinstance(rows_or_csv, (str, PathLike)):
        path = Path(rows_or_csv)
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 61 {variant_id} CSV: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return [dict(row) for row in rows_or_csv]


def _require_aligned_block_ids(
    reference_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    variant_id: str,
) -> None:
    reference = [str(row.get("block_id", "")) for row in reference_rows]
    candidate = [str(row.get("block_id", "")) for row in candidate_rows]
    if reference != candidate:
        raise ValueError(f"Phase 61 requires aligned block IDs for {variant_id}")


def _normalize_dimensions(dimensions: Sequence[int] | str) -> list[int]:
    if isinstance(dimensions, str):
        values = [int(part.strip()) for part in dimensions.split(",") if part.strip()]
    else:
        values = [int(value) for value in dimensions]
    if not values:
        raise ValueError("Phase 61 requires at least one projection dimension")
    if any(value <= 0 for value in values):
        raise ValueError("Phase 61 projection dimensions must be positive")
    if len(set(values)) != len(values):
        raise ValueError("Phase 61 projection dimensions must be unique")
    return values


def _available_explicit_columns(rows: Sequence[Mapping[str, object]]) -> list[str]:
    if not rows:
        raise ValueError("Phase 61 requires feature rows")
    present = set(rows[0].keys())
    columns = [column for column in EXPLICIT_FEATURE_COLUMNS if column in present]
    if not columns:
        columns = sorted(
            column for column in rows[0] if str(column).startswith("explicit_feature_")
        )
    if not columns:
        raise ValueError("Phase 61 requires explicit planning feature columns")
    return columns


def _numeric_columns(
    rows: Sequence[Mapping[str, object]],
    prefix: str,
    variant_id: str,
) -> list[str]:
    if not rows:
        raise ValueError(f"Phase 61 requires rows for {variant_id}")
    columns = sorted(column for column in rows[0] if str(column).startswith(prefix))
    if not columns:
        raise ValueError(f"Phase 61 requires {prefix} columns for {variant_id}")
    return columns


def _matrix_for_prefix_if_available(
    rows: Sequence[Mapping[str, object]],
    prefix: str,
) -> np.ndarray | None:
    columns = sorted(column for column in rows[0] if str(column).startswith(prefix)) if rows else []
    if not columns:
        return None
    return _matrix_for_columns(rows, columns)


def _matrix_for_columns(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [[float(row[column]) for column in columns] for row in rows],
        dtype=float,
    )


def _centered(matrix: np.ndarray) -> np.ndarray:
    return matrix - np.mean(matrix, axis=0, keepdims=True)


def _random_orthonormal_projection(
    centered: np.ndarray,
    dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    random_matrix = rng.standard_normal(size=(centered.shape[1], int(dimension)))
    q, _ = np.linalg.qr(random_matrix)
    projected = centered @ q[:, : min(int(dimension), q.shape[1])]
    return _pad_projection(projected, centered.shape[0], int(dimension))


def _pca_projection(centered: np.ndarray, dimension: int) -> np.ndarray:
    if centered.size == 0:
        return np.empty((centered.shape[0], int(dimension)))
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: min(int(dimension), vt.shape[0])].T
    projected = centered @ components if components.size else np.empty((centered.shape[0], 0))
    return _pad_projection(projected, centered.shape[0], int(dimension))


def _pad_projection(
    projected: np.ndarray,
    row_count: int,
    dimension: int,
) -> np.ndarray:
    if projected.shape[1] >= dimension:
        return projected[:, :dimension]
    padding = np.zeros((row_count, dimension - projected.shape[1]), dtype=float)
    return np.hstack([projected, padding])


def _build_variant_rows(
    source_rows: Sequence[Mapping[str, object]],
    explicit_columns: Sequence[str],
    explicit_matrix: np.ndarray,
    projection_matrix: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_index, source_row in enumerate(source_rows):
        row: dict[str, object] = {"block_id": str(source_row["block_id"])}
        for column_index, column in enumerate(explicit_columns):
            row[column] = float(explicit_matrix[row_index, column_index])
        for column_index in range(projection_matrix.shape[1]):
            row[f"projection_{column_index:02d}"] = float(
                projection_matrix[row_index, column_index]
            )
        rows.append(row)
    return rows


def _projection_columns(dimension: int) -> list[str]:
    return [f"projection_{index:02d}" for index in range(int(dimension))]


def _build_manifest(summary: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    variants: dict[str, dict[str, object]] = {}
    for variant_id, row in summary.items():
        variants[variant_id] = {
            "description": f"Explicit planning features plus {variant_id} GeoFM projection controls.",
            "state_groups": [
                "explicit_planning_features",
                f"phase61_{variant_id.lower()}_geofm_projection",
            ],
            "reward": "base_planning_reward",
            "required_columns": list(row["required_columns"]),
            "ready": True,
            "missing": [],
            "feature_table": str(row["feature_table"]),
            "row_count": int(row["row_count"]),
        }
    return {"claim_boundary": PHASE61_CLAIM_BOUNDARY, "variants": variants}


def _geometry_row(
    variant_id: str,
    projection_type: str,
    matrix: np.ndarray,
    raw_variance: float,
    reference_variant_id: str,
    similarity: float | None,
) -> dict[str, object]:
    total_variance = _total_centered_variance(matrix)
    eigenvalues = _covariance_eigenvalues(matrix)
    variances = _column_variances(matrix)
    return {
        "variant_id": variant_id,
        "projection_type": projection_type,
        "row_count": int(matrix.shape[0]),
        "projection_dimension": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "total_centered_variance": _round_float(total_variance),
        "raw_variance_retention": _round_optional_float(
            total_variance / raw_variance if raw_variance > 0.0 else None
        ),
        "effective_rank": _round_float(_effective_rank(eigenvalues)),
        "positive_variance_column_count": sum(1 for value in variances if value > 1e-12),
        "d4_reference_variant_id": reference_variant_id,
        "d4_mean_abs_column_correlation": _round_optional_float(similarity),
        "claim_boundary": PHASE61_CLAIM_BOUNDARY,
    }


def _phase61_status(
    geometry_rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, Mapping[str, object]],
    row_count: int,
) -> str:
    if not geometry_rows or len(geometry_rows) != len(summary):
        return "d6_projection_controls_blocked"
    basic_valid = all(
        int(row.get("row_count", -1)) == int(row_count)
        and int(row.get("projection_dimension", 0)) > 0
        and float(row.get("total_centered_variance", 0.0)) > 0.0
        and int(row.get("positive_variance_column_count", 0)) > 0
        for row in geometry_rows
    )
    if not basic_valid:
        return "d6_projection_controls_blocked"

    return "d6_projection_controls_ready_for_training"


def _phase61_conclusion(status: str) -> str:
    if status == "d6_projection_controls_ready_for_training":
        return (
            "Phase 61 conclusion: D6 GeoFM projection controls are valid feature "
            "tables for later matched training, subject to the stated claim boundary."
        )
    if status == "d6_projection_controls_partial":
        return (
            "Phase 61 conclusion: D6 feature tables are generated, but one or more "
            "projection similarity diagnostics is weak."
        )
    return (
        "Phase 61 conclusion: D6 projection controls are blocked by row lineage, "
        "dimension, or variance checks."
    )


def _total_centered_variance(matrix: np.ndarray) -> float:
    if matrix.size == 0 or matrix.shape[0] == 0:
        return 0.0
    centered = _centered(matrix)
    return float(np.sum(np.mean(centered * centered, axis=0)))


def _column_variances(matrix: np.ndarray) -> list[float]:
    if matrix.size == 0 or matrix.shape[0] == 0:
        return []
    centered = _centered(matrix)
    return [float(value) for value in np.mean(centered * centered, axis=0)]


def _covariance_eigenvalues(matrix: np.ndarray) -> list[float]:
    if matrix.size == 0 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return []
    centered = _centered(matrix)
    covariance = (centered.T @ centered) / float(matrix.shape[0])
    values = np.linalg.eigvalsh(covariance)
    return [float(max(value, 0.0)) for value in sorted(values, reverse=True)]


def _effective_rank(eigenvalues: Sequence[float]) -> float:
    positive = [float(value) for value in eigenvalues if float(value) > 1e-12]
    total = sum(positive)
    if total <= 0.0:
        return 0.0
    probabilities = [value / total for value in positive]
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
    return float(math.exp(entropy))


def _mean_abs_column_correlation(
    matrix: np.ndarray,
    reference: np.ndarray | None,
) -> float | None:
    if reference is None or matrix.size == 0 or reference.size == 0:
        return None
    count = min(matrix.shape[1], reference.shape[1])
    if count <= 0:
        return None
    values: list[float] = []
    for index in range(count):
        correlation = _pearson(matrix[:, index], reference[:, index])
        if correlation is not None:
            values.append(abs(correlation))
    if not values:
        return None
    return sum(values) / len(values)


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = math.sqrt(
        float(np.sum(left_centered * left_centered))
        * float(np.sum(right_centered * right_centered))
    )
    if denominator <= 0.0:
        return None
    return float(np.sum(left_centered * right_centered) / denominator)


def _round_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return _round_float(value)


def _round_float(value: float) -> float:
    return round(float(value), 10)


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value