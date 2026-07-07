from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import math
from os import PathLike
from pathlib import Path
import statistics

import numpy as np


PHASE57_CLAIM_BOUNDARY = (
    "Phase 57 is a read-only compressed-representation mechanism audit over "
    "existing Bishan feature, tile, and expanded-replication delta artifacts. "
    "It evaluates whether PCA-compressed GeoFM state routes have lower "
    "effective rank and retain nonzero raw embedding variance while the "
    "expanded compressed-route reward deltas remain positive; it does not "
    "prove PCA optimality, does not retrain RL policies, does not enable "
    "B2/B3 suitability reward, does not test transfer, and does not validate "
    "independent agronomic suitability."
)

GEOMETRY_FIELDNAMES = [
    "variant_id",
    "row_count",
    "feature_count",
    "total_centered_variance",
    "raw_variance_retention",
    "effective_rank",
    "participation_ratio",
    "positive_eigenvalue_count",
    "condition_number",
    "min_feature_std",
    "max_feature_std",
    "feature_std_spread",
    "claim_boundary",
]

REWARD_GAIN_FIELDNAMES = [
    "compressed_variant_id",
    "row_count",
    "mean_delta",
    "std_delta",
    "positive_count",
    "total_count",
    "positive_fraction",
    "comparator_ids",
    "claim_boundary",
]

TILE_GEOMETRY_GAIN_FIELDNAMES = [
    "eval_tile_id",
    "tile_block_count",
    "matched_block_count",
    "raw_total_centered_variance",
    "d4p8_total_centered_variance",
    "d4p16_total_centered_variance",
    "d4p8_raw_variance_retention",
    "d4p16_raw_variance_retention",
    "d4p8_effective_rank",
    "d4p16_effective_rank",
    "d4p8_mean_delta",
    "d4p16_mean_delta",
    "pooled_mean_delta",
    "pooled_positive_fraction",
    "claim_boundary",
]


def build_phase57_compressed_representation_mechanism(
    b1_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    d4p8_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    d4p16_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    delta_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    tile_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
) -> dict[str, object]:
    b1_rows = _load_mapping_rows(b1_rows_or_csv, "B1 feature rows")
    d4p8_rows = _load_mapping_rows(d4p8_rows_or_csv, "D4P8 feature rows")
    d4p16_rows = _load_mapping_rows(d4p16_rows_or_csv, "D4P16 feature rows")
    delta_rows = _load_mapping_rows(delta_rows_or_csv, "delta rows")
    tile_rows = _load_mapping_rows(tile_rows_or_csv, "tile rows")

    b1_index = _index_by_block_id(b1_rows, "B1")
    d4p8_index = _index_by_block_id(d4p8_rows, "D4P8")
    d4p16_index = _index_by_block_id(d4p16_rows, "D4P16")
    common_blocks = [
        block_id
        for block_id in b1_index
        if block_id in d4p8_index and block_id in d4p16_index
    ]
    row_alignment = _row_alignment(b1_index, d4p8_index, d4p16_index, common_blocks)

    b1_columns = _numeric_columns(b1_rows, "embedding_mean_")
    d4p8_columns = _numeric_columns(d4p8_rows, "embedding_pca_")
    d4p16_columns = _numeric_columns(d4p16_rows, "embedding_pca_")
    b1_matrix = _matrix_for_blocks(b1_index, common_blocks, b1_columns)
    d4p8_matrix = _matrix_for_blocks(d4p8_index, common_blocks, d4p8_columns)
    d4p16_matrix = _matrix_for_blocks(d4p16_index, common_blocks, d4p16_columns)

    raw_variance = _total_centered_variance(b1_matrix)
    geometry_rows = [
        _geometry_row("B1", b1_matrix, raw_variance, raw_variance),
        _geometry_row("D4P8", d4p8_matrix, raw_variance, raw_variance),
        _geometry_row("D4P16", d4p16_matrix, raw_variance, raw_variance),
    ]
    reward_gain_rows = _reward_gain_rows(delta_rows)
    tile_geometry_gain_rows = _tile_geometry_gain_rows(
        tile_rows,
        delta_rows,
        b1_index,
        d4p8_index,
        d4p16_index,
        b1_columns,
        d4p8_columns,
        d4p16_columns,
    )
    associations = _tile_associations(tile_geometry_gain_rows)
    status = _phase57_status(row_alignment, geometry_rows, reward_gain_rows)

    return {
        "phase": "phase57_compressed_representation_mechanism",
        "phase57_mechanism_status": status,
        "row_alignment": row_alignment,
        "geometry_rows": geometry_rows,
        "reward_gain_rows": reward_gain_rows,
        "tile_geometry_gain_rows": tile_geometry_gain_rows,
        "tile_geometry_gain_associations": associations,
        "conclusion": _phase57_conclusion(status),
        "claim_boundary": PHASE57_CLAIM_BOUNDARY,
    }


def write_phase57_compressed_representation_mechanism_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_path = output_path / "phase57_compressed_representation_mechanism.json"
    geometry_path = output_path / "phase57_representation_geometry.csv"
    reward_gain_path = output_path / "phase57_reward_gain_summary.csv"
    tile_path = output_path / "phase57_tile_geometry_gain.csv"
    readiness_path = output_path / "phase57_compressed_representation_mechanism.md"

    comparison_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv_mapping_rows(
        geometry_path,
        GEOMETRY_FIELDNAMES,
        analysis.get("geometry_rows"),
        "geometry_rows",
    )
    _write_csv_mapping_rows(
        reward_gain_path,
        REWARD_GAIN_FIELDNAMES,
        analysis.get("reward_gain_rows"),
        "reward_gain_rows",
    )
    _write_csv_mapping_rows(
        tile_path,
        TILE_GEOMETRY_GAIN_FIELDNAMES,
        analysis.get("tile_geometry_gain_rows"),
        "tile_geometry_gain_rows",
    )
    readiness_path.write_text(_readiness_markdown(analysis), encoding="utf-8")
    return {
        "comparison_json": comparison_path,
        "geometry_csv": geometry_path,
        "reward_gain_csv": reward_gain_path,
        "tile_geometry_gain_csv": tile_path,
        "readiness_md": readiness_path,
    }


def _load_mapping_rows(
    rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    label: str,
) -> list[dict[str, object]]:
    if isinstance(rows_or_csv, (str, PathLike)):
        path = Path(rows_or_csv)
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} CSV: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    rows = []
    for row in rows_or_csv:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} must contain mapping rows")
        rows.append(dict(row))
    return rows


def _index_by_block_id(
    rows: Sequence[Mapping[str, object]],
    variant_id: str,
) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for row in rows:
        block_id = str(row.get("block_id", "")).strip()
        if not block_id:
            raise ValueError(f"{variant_id} feature row is missing block_id")
        if block_id in indexed:
            raise ValueError(f"{variant_id} feature table has duplicate block_id: {block_id}")
        indexed[block_id] = dict(row)
    return indexed


def _row_alignment(
    b1_index: Mapping[str, object],
    d4p8_index: Mapping[str, object],
    d4p16_index: Mapping[str, object],
    common_blocks: Sequence[str],
) -> dict[str, object]:
    b1_blocks = set(b1_index)
    d4p8_blocks = set(d4p8_index)
    d4p16_blocks = set(d4p16_index)
    common = set(common_blocks)
    return {
        "b1_row_count": len(b1_index),
        "d4p8_row_count": len(d4p8_index),
        "d4p16_row_count": len(d4p16_index),
        "common_block_count": len(common_blocks),
        "all_rows_align": len(common) == len(b1_blocks) == len(d4p8_blocks) == len(d4p16_blocks),
        "missing_from_b1": sorted((d4p8_blocks | d4p16_blocks) - b1_blocks),
        "missing_from_d4p8": sorted((b1_blocks | d4p16_blocks) - d4p8_blocks),
        "missing_from_d4p16": sorted((b1_blocks | d4p8_blocks) - d4p16_blocks),
    }


def _numeric_columns(rows: Sequence[Mapping[str, object]], prefix: str) -> list[str]:
    if not rows:
        raise ValueError(f"No rows available for numeric prefix: {prefix}")
    columns = [
        key
        for key in rows[0]
        if str(key).startswith(prefix)
    ]
    if not columns:
        raise ValueError(f"No numeric columns found with prefix: {prefix}")
    return sorted(columns)


def _matrix_for_blocks(
    indexed_rows: Mapping[str, Mapping[str, object]],
    block_ids: Sequence[str],
    columns: Sequence[str],
) -> np.ndarray:
    values: list[list[float]] = []
    for block_id in block_ids:
        row = indexed_rows[block_id]
        values.append([_float_value(row, column) for column in columns])
    if not values:
        return np.empty((0, len(columns)), dtype=float)
    return np.asarray(values, dtype=float)


def _geometry_row(
    variant_id: str,
    matrix: np.ndarray,
    raw_variance: float,
    fallback_raw_variance: float,
) -> dict[str, object]:
    total_variance = _total_centered_variance(matrix)
    eigenvalues = _covariance_eigenvalues(matrix)
    positive_eigenvalues = [value for value in eigenvalues if value > 1e-12]
    feature_stds = _feature_stds(matrix)
    retention = (
        total_variance / raw_variance
        if raw_variance > 0.0
        else None
    )
    if variant_id == "B1" and fallback_raw_variance > 0.0:
        retention = 1.0
    return {
        "variant_id": variant_id,
        "row_count": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "total_centered_variance": _round_float(total_variance),
        "raw_variance_retention": _round_optional_float(retention),
        "effective_rank": _round_float(_effective_rank(eigenvalues)),
        "participation_ratio": _round_float(_participation_ratio(eigenvalues)),
        "positive_eigenvalue_count": len(positive_eigenvalues),
        "condition_number": _round_optional_float(_condition_number(positive_eigenvalues)),
        "min_feature_std": _round_optional_float(min(feature_stds) if feature_stds else None),
        "max_feature_std": _round_optional_float(max(feature_stds) if feature_stds else None),
        "feature_std_spread": _round_optional_float(_feature_std_spread(feature_stds)),
        "claim_boundary": PHASE57_CLAIM_BOUNDARY,
    }


def _total_centered_variance(matrix: np.ndarray) -> float:
    if matrix.size == 0 or matrix.shape[0] == 0:
        return 0.0
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    return float(np.sum(np.mean(centered * centered, axis=0)))


def _covariance_eigenvalues(matrix: np.ndarray) -> list[float]:
    if matrix.size == 0 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return []
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    covariance = (centered.T @ centered) / float(matrix.shape[0])
    values = np.linalg.eigvalsh(covariance)
    return [float(max(value, 0.0)) for value in sorted(values, reverse=True)]


def _feature_stds(matrix: np.ndarray) -> list[float]:
    if matrix.size == 0 or matrix.shape[0] == 0:
        return []
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    variances = np.mean(centered * centered, axis=0)
    return [float(math.sqrt(max(value, 0.0))) for value in variances]


def _effective_rank(eigenvalues: Sequence[float]) -> float:
    positive = [float(value) for value in eigenvalues if float(value) > 1e-12]
    total = sum(positive)
    if total <= 0.0:
        return 0.0
    probabilities = [value / total for value in positive]
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
    return float(math.exp(entropy))


def _participation_ratio(eigenvalues: Sequence[float]) -> float:
    positive = [float(value) for value in eigenvalues if float(value) > 1e-12]
    numerator = sum(positive) ** 2
    denominator = sum(value * value for value in positive)
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _condition_number(positive_eigenvalues: Sequence[float]) -> float | None:
    if not positive_eigenvalues:
        return None
    low = min(positive_eigenvalues)
    if low <= 0.0:
        return None
    return max(positive_eigenvalues) / low


def _feature_std_spread(feature_stds: Sequence[float]) -> float | None:
    positive = [float(value) for value in feature_stds if float(value) > 1e-12]
    if not positive:
        return None
    return max(positive) / min(positive)


def _reward_gain_rows(delta_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in delta_rows:
        variant_id = str(row.get("compressed_variant_id", "")).strip()
        if not variant_id:
            continue
        grouped.setdefault(variant_id, []).append(row)
    rows: list[dict[str, object]] = []
    for variant_id in sorted(grouped):
        group_rows = grouped[variant_id]
        deltas = [_float_value(row, "compressed_minus_comparator_reward") for row in group_rows]
        comparators = sorted({str(row.get("comparator_variant_id", "")) for row in group_rows})
        positive_count = sum(1 for value in deltas if value > 0.0)
        total_count = len(deltas)
        rows.append(
            {
                "compressed_variant_id": variant_id,
                "row_count": total_count,
                "mean_delta": _mean_or_none(deltas),
                "std_delta": _std_or_none(deltas),
                "positive_count": positive_count,
                "total_count": total_count,
                "positive_fraction": _round_float(positive_count / total_count)
                if total_count
                else None,
                "comparator_ids": ";".join(comparators),
                "claim_boundary": PHASE57_CLAIM_BOUNDARY,
            }
        )
    return rows


def _tile_geometry_gain_rows(
    tile_rows: Sequence[Mapping[str, object]],
    delta_rows: Sequence[Mapping[str, object]],
    b1_index: Mapping[str, Mapping[str, object]],
    d4p8_index: Mapping[str, Mapping[str, object]],
    d4p16_index: Mapping[str, Mapping[str, object]],
    b1_columns: Sequence[str],
    d4p8_columns: Sequence[str],
    d4p16_columns: Sequence[str],
) -> list[dict[str, object]]:
    delta_by_tile = _delta_rows_by_tile(delta_rows)
    common_blocks = set(b1_index) & set(d4p8_index) & set(d4p16_index)
    result: list[dict[str, object]] = []
    for tile in tile_rows:
        tile_id = str(tile.get("tile_id", "")).strip()
        if not tile_id or tile_id not in delta_by_tile:
            continue
        tile_blocks = _parse_block_ids(tile.get("block_ids", ""))
        matched_blocks = [block_id for block_id in tile_blocks if block_id in common_blocks]
        if not matched_blocks:
            continue
        b1_matrix = _matrix_for_blocks(b1_index, matched_blocks, b1_columns)
        d4p8_matrix = _matrix_for_blocks(d4p8_index, matched_blocks, d4p8_columns)
        d4p16_matrix = _matrix_for_blocks(d4p16_index, matched_blocks, d4p16_columns)
        raw_variance = _total_centered_variance(b1_matrix)
        d4p8_variance = _total_centered_variance(d4p8_matrix)
        d4p16_variance = _total_centered_variance(d4p16_matrix)
        tile_deltas = delta_by_tile[tile_id]
        pooled_values = [
            _float_value(row, "compressed_minus_comparator_reward")
            for row in tile_deltas
        ]
        pooled_positive = sum(1 for value in pooled_values if value > 0.0)
        result.append(
            {
                "eval_tile_id": tile_id,
                "tile_block_count": len(tile_blocks),
                "matched_block_count": len(matched_blocks),
                "raw_total_centered_variance": _round_float(raw_variance),
                "d4p8_total_centered_variance": _round_float(d4p8_variance),
                "d4p16_total_centered_variance": _round_float(d4p16_variance),
                "d4p8_raw_variance_retention": _round_optional_float(
                    d4p8_variance / raw_variance if raw_variance > 0.0 else None
                ),
                "d4p16_raw_variance_retention": _round_optional_float(
                    d4p16_variance / raw_variance if raw_variance > 0.0 else None
                ),
                "d4p8_effective_rank": _round_float(_effective_rank(_covariance_eigenvalues(d4p8_matrix))),
                "d4p16_effective_rank": _round_float(_effective_rank(_covariance_eigenvalues(d4p16_matrix))),
                "d4p8_mean_delta": _mean_delta_for_variant(tile_deltas, "D4P8"),
                "d4p16_mean_delta": _mean_delta_for_variant(tile_deltas, "D4P16"),
                "pooled_mean_delta": _mean_or_none(pooled_values),
                "pooled_positive_fraction": _round_float(pooled_positive / len(pooled_values))
                if pooled_values
                else None,
                "claim_boundary": PHASE57_CLAIM_BOUNDARY,
            }
        )
    return result


def _delta_rows_by_tile(
    delta_rows: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in delta_rows:
        tile_id = str(row.get("eval_tile_id", "")).strip()
        if not tile_id:
            continue
        grouped.setdefault(tile_id, []).append(row)
    return grouped


def _parse_block_ids(value: object) -> list[str]:
    return [part.strip() for part in str(value).split(";") if part.strip()]


def _mean_delta_for_variant(
    delta_rows: Sequence[Mapping[str, object]],
    variant_id: str,
) -> float | None:
    values = [
        _float_value(row, "compressed_minus_comparator_reward")
        for row in delta_rows
        if str(row.get("compressed_variant_id", "")) == variant_id
    ]
    return _mean_or_none(values)


def _tile_associations(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "d4p8_retention_vs_gain_pearson": _pearson_from_rows(
            rows,
            "d4p8_raw_variance_retention",
            "d4p8_mean_delta",
        ),
        "d4p16_retention_vs_gain_pearson": _pearson_from_rows(
            rows,
            "d4p16_raw_variance_retention",
            "d4p16_mean_delta",
        ),
        "pooled_retention_vs_gain_pearson": _pooled_retention_gain_correlation(rows),
        "tile_count": len(rows),
        "claim_boundary": PHASE57_CLAIM_BOUNDARY,
    }


def _pearson_from_rows(
    rows: Sequence[Mapping[str, object]],
    x_field: str,
    y_field: str,
) -> float | None:
    pairs = [
        (float(row[x_field]), float(row[y_field]))
        for row in rows
        if row.get(x_field) is not None and row.get(y_field) is not None
    ]
    return _pearson(pairs)


def _pooled_retention_gain_correlation(
    rows: Sequence[Mapping[str, object]],
) -> float | None:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        retentions = [
            row.get("d4p8_raw_variance_retention"),
            row.get("d4p16_raw_variance_retention"),
        ]
        gains = [row.get("d4p8_mean_delta"), row.get("d4p16_mean_delta")]
        for retention, gain in zip(retentions, gains):
            if retention is not None and gain is not None:
                pairs.append((float(retention), float(gain)))
    return _pearson(pairs)


def _pearson(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denom <= 0.0:
        return None
    return _round_float(sum(x * y for x, y in zip(dx, dy)) / denom)


def _phase57_status(
    row_alignment: Mapping[str, object],
    geometry_rows: Sequence[Mapping[str, object]],
    reward_gain_rows: Sequence[Mapping[str, object]],
) -> str:
    geometry_by_variant = {str(row.get("variant_id")): row for row in geometry_rows}
    gains_by_variant = {
        str(row.get("compressed_variant_id")): row for row in reward_gain_rows
    }
    required_geometry = all(variant in geometry_by_variant for variant in ("B1", "D4P8", "D4P16"))
    required_gains = all(variant in gains_by_variant for variant in ("D4P8", "D4P16"))
    if not bool(row_alignment.get("all_rows_align")) or not required_geometry or not required_gains:
        return "insufficient"
    raw_rank = _optional_float(geometry_by_variant["B1"].get("effective_rank"))
    compressed_ranks_lower = all(
        _optional_float(geometry_by_variant[variant].get("effective_rank")) is not None
        and raw_rank is not None
        and _optional_float(geometry_by_variant[variant].get("effective_rank")) < raw_rank
        for variant in ("D4P8", "D4P16")
    )
    compressed_retains_variance = all(
        _optional_float(geometry_by_variant[variant].get("raw_variance_retention")) is not None
        and _optional_float(geometry_by_variant[variant].get("raw_variance_retention")) > 0.0
        for variant in ("D4P8", "D4P16")
    )
    compressed_gains_positive = all(
        _optional_float(gains_by_variant[variant].get("mean_delta")) is not None
        and _optional_float(gains_by_variant[variant].get("mean_delta")) > 0.0
        for variant in ("D4P8", "D4P16")
    )
    if compressed_ranks_lower and compressed_retains_variance and compressed_gains_positive:
        return "compressed_geometry_consistent"
    return "compressed_geometry_partial"


def _phase57_conclusion(status: str) -> str:
    if status == "compressed_geometry_consistent":
        return (
            "Phase 57 conclusion: the compressed-route mechanism interpretation "
            "is geometrically consistent with the current artifacts. D4P8/D4P16 "
            "retain nonzero raw GeoFM variance, have lower effective rank than "
            "raw B1, and keep positive expanded-replication reward gains."
        )
    if status == "compressed_geometry_partial":
        return (
            "Phase 57 conclusion: the compressed-route mechanism audit is only "
            "partial. At least one geometry or reward-gain condition did not "
            "meet the pre-specified consistency rule."
        )
    return (
        "Phase 57 conclusion: insufficient aligned feature or reward-gain "
        "coverage for a compressed-representation mechanism decision."
    )


def _readiness_markdown(analysis: Mapping[str, object]) -> str:
    row_alignment = analysis.get("row_alignment")
    if not isinstance(row_alignment, Mapping):
        row_alignment = {}
    geometry_rows = analysis.get("geometry_rows")
    if not isinstance(geometry_rows, list):
        geometry_rows = []
    reward_gain_rows = analysis.get("reward_gain_rows")
    if not isinstance(reward_gain_rows, list):
        reward_gain_rows = []
    associations = analysis.get("tile_geometry_gain_associations")
    if not isinstance(associations, Mapping):
        associations = {}

    lines = [
        "# Phase 57 Compressed Representation Mechanism",
        "",
        f"Status: {analysis.get('phase57_mechanism_status', '')}",
        "",
        "Mechanism conclusion:",
        str(analysis.get("conclusion", "")),
        "",
        "Row alignment:",
        "- "
        f"common blocks={row_alignment.get('common_block_count')}, "
        f"all rows align={row_alignment.get('all_rows_align')}",
        "",
        "Representation geometry:",
    ]
    for row in geometry_rows:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "- "
            f"{row.get('variant_id')}: features={row.get('feature_count')}, "
            f"variance={row.get('total_centered_variance')}, "
            f"retention={row.get('raw_variance_retention')}, "
            f"effective rank={row.get('effective_rank')}, "
            f"condition={row.get('condition_number')}"
        )
    lines.extend(["", "Expanded-replication reward gains:"])
    for row in reward_gain_rows:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "- "
            f"{row.get('compressed_variant_id')}: mean delta={row.get('mean_delta')}, "
            f"positive={row.get('positive_count')} / {row.get('total_count')}"
        )
    lines.extend(
        [
            "",
            "Tile-level diagnostic associations:",
            "- "
            f"D4P8 retention/gain Pearson={associations.get('d4p8_retention_vs_gain_pearson')}",
            "- "
            f"D4P16 retention/gain Pearson={associations.get('d4p16_retention_vs_gain_pearson')}",
            "- "
            f"Pooled retention/gain Pearson={associations.get('pooled_retention_vs_gain_pearson')}",
            "",
            "Claim boundary:",
            str(analysis.get("claim_boundary", PHASE57_CLAIM_BOUNDARY)),
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    row_key: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 57 analysis is missing {row_key}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 57 {row_key} rows must be objects")
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _float_value(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing numeric field: {field}")
    return float(value)


def _optional_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return _round_float(sum(float(value) for value in values) / len(values))


def _std_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return (
        _round_float(statistics.pstdev(float(value) for value in values))
        if len(values) > 1
        else 0.0
    )


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


def _round_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return _round_float(value)


def _round_float(value: float) -> float:
    return round(float(value), 10)
