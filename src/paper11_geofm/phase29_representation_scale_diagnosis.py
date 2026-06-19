from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from .block_schema import EMBEDDING_COLUMNS


PHASE29_REPRESENTATION_SCALE_CLAIM_BOUNDARY = (
    "Phase 29 representation-scale diagnosis is a read-only follow-up over "
    "existing Phase 2, Phase 8, and Phase 13 artifacts; it does not run new "
    "policy training, does not alter rewards, does not test B2/B3, does not "
    "prove that PCA is intrinsically superior, and does not prove that "
    "normalization would improve PPO performance."
)

PHASE29_VARIANT_SCALE_FIELDNAMES = [
    "variant_id",
    "n_blocks",
    "n_dimensions",
    "mean_abs_value",
    "mean_column_std",
    "mean_row_l2_norm",
    "std_row_l2_norm",
    "max_row_l2_norm",
    "claim_boundary",
]

PHASE29_TILE_SCALE_FIELDNAMES = [
    "tile_id",
    "n_blocks",
    "matched_block_count",
    "mean_b1_mean_abs_value",
    "mean_b1_row_l2_norm",
    "std_b1_row_l2_norm",
    "claim_boundary",
]

PHASE29_NORMALIZATION_PROFILE_FIELDNAMES = [
    "profile_id",
    "n_blocks",
    "n_dimensions",
    "mean_abs_value",
    "mean_column_std",
    "mean_row_l2_norm",
    "std_row_l2_norm",
    "min_row_l2_norm",
    "max_row_l2_norm",
    "claim_boundary",
]


def build_phase29_representation_scale_diagnosis(
    phase2_b1_features_csv: Path | str,
    d4p8_features_csv: Path | str,
    d4p16_features_csv: Path | str,
    tile_index_csv: Path | str,
    phase28_summary_csv: Path | str | None = None,
    rank_threshold: float = 1e-12,
) -> dict[str, object]:
    b1_features = _read_feature_table(Path(phase2_b1_features_csv))
    d4p8_features = _read_feature_table(Path(d4p8_features_csv))
    d4p16_features = _read_feature_table(Path(d4p16_features_csv))
    tile_rows = _read_csv_rows(Path(tile_index_csv), "Phase 13 tile index CSV")

    b1_matrix = _matrix_for_columns(b1_features, EMBEDDING_COLUMNS)
    d4p8_matrix = _matrix_for_columns(d4p8_features, _pca_columns(8))
    d4p16_matrix = _matrix_for_columns(d4p16_features, _pca_columns(16))

    variant_scale_rows = [
        _matrix_summary_row("B1", b1_matrix),
        _matrix_summary_row("D4P8", d4p8_matrix),
        _matrix_summary_row("D4P16", d4p16_matrix),
    ]
    tile_scale_rows = _tile_scale_rows(tile_rows, b1_features)
    normalization_rows = _b1_normalization_profile_rows(b1_matrix)
    pca_diagnostics = _pca_diagnostics(
        b1_matrix,
        d4p8_matrix,
        d4p16_matrix,
        rank_threshold=float(rank_threshold),
    )
    phase28_metadata = _phase28_metadata(Path(phase28_summary_csv)) if phase28_summary_csv else {}
    status = _phase29_status(
        variant_scale_rows,
        normalization_rows,
        pca_diagnostics,
    )

    return {
        "phase": "phase29_representation_scale_diagnosis",
        "phase29_representation_scale_status": status,
        "source_paths": {
            "phase2_b1_features_csv": str(Path(phase2_b1_features_csv)),
            "d4p8_features_csv": str(Path(d4p8_features_csv)),
            "d4p16_features_csv": str(Path(d4p16_features_csv)),
            "tile_index_csv": str(Path(tile_index_csv)),
            "phase28_summary_csv": str(Path(phase28_summary_csv))
            if phase28_summary_csv
            else None,
        },
        "row_counts": {
            "b1_feature_rows": len(b1_features),
            "d4p8_feature_rows": len(d4p8_features),
            "d4p16_feature_rows": len(d4p16_features),
            "tile_rows": len(tile_rows),
        },
        "phase28_metadata": phase28_metadata,
        "variant_scale_rows": variant_scale_rows,
        "tile_scale_rows": tile_scale_rows,
        "b1_normalization_profile_rows": normalization_rows,
        "pca_diagnostics": pca_diagnostics,
        "interpretation": _phase29_interpretation(status),
        "claim_boundary": PHASE29_REPRESENTATION_SCALE_CLAIM_BOUNDARY,
    }


def write_phase29_representation_scale_diagnosis_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    variant_scale_path = output_path / "phase29_variant_scale_summary.csv"
    tile_scale_path = output_path / "phase29_tile_scale_summary.csv"
    normalization_profiles_path = output_path / "phase29_b1_normalization_profiles.csv"
    diagnosis_json_path = output_path / "phase29_representation_scale_diagnosis.json"
    diagnosis_md_path = output_path / "phase29_representation_scale_diagnosis.md"

    _write_csv_mapping_rows(
        variant_scale_path,
        PHASE29_VARIANT_SCALE_FIELDNAMES,
        analysis.get("variant_scale_rows"),
        "variant_scale_rows",
    )
    _write_csv_mapping_rows(
        tile_scale_path,
        PHASE29_TILE_SCALE_FIELDNAMES,
        analysis.get("tile_scale_rows"),
        "tile_scale_rows",
    )
    _write_csv_mapping_rows(
        normalization_profiles_path,
        PHASE29_NORMALIZATION_PROFILE_FIELDNAMES,
        analysis.get("b1_normalization_profile_rows"),
        "b1_normalization_profile_rows",
    )
    diagnosis_json_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    diagnosis_md_path.write_text(
        _phase29_markdown(analysis),
        encoding="utf-8",
    )

    return {
        "variant_scale_csv": variant_scale_path,
        "tile_scale_csv": tile_scale_path,
        "normalization_profiles_csv": normalization_profiles_path,
        "diagnosis_json": diagnosis_json_path,
        "diagnosis_md": diagnosis_md_path,
    }


def _read_csv_rows(path: Path, label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_feature_table(path: Path) -> dict[str, dict[str, object]]:
    rows = _read_csv_rows(path, "feature table")
    table: dict[str, dict[str, object]] = {}
    for row_number, row in enumerate(rows, start=2):
        block_id = str(row.get("block_id", "")).strip()
        if not block_id:
            raise ValueError(f"Feature table is missing block_id at {path}:{row_number}")
        if block_id in table:
            raise ValueError(f"Duplicate block_id in feature table {path}: {block_id}")
        table[block_id] = dict(row)
    if not table:
        raise ValueError(f"Feature table has no rows: {path}")
    return table


def _matrix_for_columns(
    feature_table: Mapping[str, Mapping[str, object]],
    columns: Sequence[str],
) -> np.ndarray:
    rows = []
    for block_id in sorted(feature_table):
        row = feature_table[block_id]
        missing = [
            column
            for column in columns
            if column not in row or str(row[column]).strip() == ""
        ]
        if missing:
            raise ValueError(
                f"Feature table row {block_id} is missing columns: {missing}"
            )
        rows.append([float(row[column]) for column in columns])
    return np.asarray(rows, dtype=float)


def _matrix_summary_row(variant_id: str, matrix: np.ndarray) -> dict[str, object]:
    row_norms = _row_l2_norms(matrix)
    return {
        "variant_id": variant_id,
        "n_blocks": int(matrix.shape[0]),
        "n_dimensions": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "mean_abs_value": _round_float(float(np.mean(np.abs(matrix)))) if matrix.size else 0.0,
        "mean_column_std": _mean_column_std(matrix),
        "mean_row_l2_norm": _mean(row_norms),
        "std_row_l2_norm": _std(row_norms),
        "max_row_l2_norm": _round_float(float(np.max(row_norms))) if row_norms.size else 0.0,
        "claim_boundary": PHASE29_REPRESENTATION_SCALE_CLAIM_BOUNDARY,
    }


def _tile_scale_rows(
    tile_rows: Sequence[Mapping[str, object]],
    b1_features: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for tile_row in tile_rows:
        tile_id = str(tile_row.get("tile_id", "")).strip()
        if not tile_id:
            continue
        block_ids = _split_block_ids(tile_row.get("block_ids", ""))
        if not block_ids:
            continue
        missing = [block_id for block_id in block_ids if block_id not in b1_features]
        if missing:
            raise ValueError(
                f"Tile {tile_id} references block IDs missing from B1 feature table: {missing[:5]}"
            )
        matrix = np.asarray(
            [
                [float(b1_features[block_id][column]) for column in EMBEDDING_COLUMNS]
                for block_id in block_ids
            ],
            dtype=float,
        )
        row_norms = _row_l2_norms(matrix)
        output_rows.append(
            {
                "tile_id": tile_id,
                "n_blocks": int(tile_row.get("n_blocks") or len(block_ids)),
                "matched_block_count": len(block_ids),
                "mean_b1_mean_abs_value": _round_float(float(np.mean(np.abs(matrix))))
                if matrix.size
                else 0.0,
                "mean_b1_row_l2_norm": _mean(row_norms),
                "std_b1_row_l2_norm": _std(row_norms),
                "claim_boundary": PHASE29_REPRESENTATION_SCALE_CLAIM_BOUNDARY,
            }
        )
    return output_rows


def _b1_normalization_profile_rows(matrix: np.ndarray) -> list[dict[str, object]]:
    profiles = {
        "raw": matrix,
        "column_zscore": _column_std_scale(matrix),
        "row_l2": _row_l2_normalize(matrix),
        "column_zscore_row_l2": _row_l2_normalize(_column_std_scale(matrix)),
    }
    rows: list[dict[str, object]] = []
    for profile_id, profile_matrix in profiles.items():
        row_norms = _row_l2_norms(profile_matrix)
        rows.append(
            {
                "profile_id": profile_id,
                "n_blocks": int(profile_matrix.shape[0]),
                "n_dimensions": int(profile_matrix.shape[1])
                if profile_matrix.ndim == 2
                else 0,
                "mean_abs_value": _round_float(float(np.mean(np.abs(profile_matrix))))
                if profile_matrix.size
                else 0.0,
                "mean_column_std": _mean_column_std(profile_matrix),
                "mean_row_l2_norm": _mean(row_norms),
                "std_row_l2_norm": _std(row_norms),
                "min_row_l2_norm": _round_float(float(np.min(row_norms)))
                if row_norms.size
                else 0.0,
                "max_row_l2_norm": _round_float(float(np.max(row_norms)))
                if row_norms.size
                else 0.0,
                "claim_boundary": PHASE29_REPRESENTATION_SCALE_CLAIM_BOUNDARY,
            }
        )
    return rows


def _column_std_scale(matrix: np.ndarray) -> np.ndarray:
    if not matrix.size:
        return np.zeros_like(matrix, dtype=float)
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    safe_stds = np.where(stds > 0.0, stds, 1.0)
    return (matrix - means) / safe_stds


def _row_l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if not matrix.size:
        return np.zeros_like(matrix, dtype=float)
    row_norms = _row_l2_norms(matrix)
    safe_norms = np.where(row_norms > 0.0, row_norms, 1.0)
    return matrix / safe_norms[:, None]


def _pca_diagnostics(
    b1_matrix: np.ndarray,
    d4p8_matrix: np.ndarray,
    d4p16_matrix: np.ndarray,
    rank_threshold: float,
) -> dict[str, object]:
    centered = b1_matrix - b1_matrix.mean(axis=0)
    if centered.size:
        _u, singular_values, _vt = np.linalg.svd(centered, full_matrices=False)
    else:
        singular_values = np.asarray([], dtype=float)
    variance = singular_values**2
    total_variance = float(variance.sum())
    effective_rank = (
        float(total_variance**2 / np.sum(variance**2))
        if variance.size and float(np.sum(variance**2)) > 0.0
        else 0.0
    )
    return {
        "raw_embedding_rank_threshold": float(rank_threshold),
        "raw_embedding_numerical_rank": int(np.sum(singular_values > rank_threshold)),
        "raw_embedding_effective_rank": _round_float(effective_rank),
        "top8_pca_variance_ratio": _variance_ratio(variance, total_variance, 8),
        "top16_pca_variance_ratio": _variance_ratio(variance, total_variance, 16),
        "mean_raw_embedding_std": _mean_column_std(b1_matrix),
        "mean_d4p8_component_std": _mean_column_std(d4p8_matrix),
        "mean_d4p16_component_std": _mean_column_std(d4p16_matrix),
    }


def _phase28_metadata(path: Path) -> dict[str, object]:
    rows = _read_csv_rows(path, "Phase 28 summary CSV")
    trained_rows = [
        row for row in rows if str(row.get("row_type", "")) == "trained_policy"
    ]
    return {
        "summary_rows": len(rows),
        "trained_policy_rows": len(trained_rows),
        "eval_tile_ids": _unique_strings(trained_rows, "eval_tile_id"),
        "seeds": _unique_ints(trained_rows, "seed"),
        "train_timesteps": _first_int_or_none(trained_rows, "train_timesteps"),
        "eval_max_steps": _first_int_or_none(trained_rows, "eval_max_steps"),
    }


def _phase29_status(
    variant_rows: Sequence[Mapping[str, object]],
    normalization_rows: Sequence[Mapping[str, object]],
    pca_diagnostics: Mapping[str, object],
) -> str:
    variants = {str(row.get("variant_id", "")): row for row in variant_rows}
    profiles = {str(row.get("profile_id", "")): row for row in normalization_rows}
    b1 = variants.get("B1")
    d4p8 = variants.get("D4P8")
    d4p16 = variants.get("D4P16")
    raw_profile = profiles.get("raw")
    row_l2_profile = profiles.get("row_l2")
    if not all(isinstance(row, Mapping) for row in (b1, d4p8, d4p16, raw_profile, row_l2_profile)):
        return "insufficient"
    if (
        float(pca_diagnostics.get("top8_pca_variance_ratio", 0.0)) >= 0.8
        and float(pca_diagnostics.get("top16_pca_variance_ratio", 0.0)) >= 0.9
        and float(d4p8.get("mean_column_std", 0.0)) > float(b1.get("mean_column_std", 0.0))
        and float(d4p16.get("mean_column_std", 0.0)) > float(b1.get("mean_column_std", 0.0))
        and float(row_l2_profile.get("std_row_l2_norm", 0.0))
        <= float(raw_profile.get("std_row_l2_norm", 0.0))
    ):
        return "raw_b1_scale_may_affect_optimization"
    return "representation_scale_descriptive"


def _phase29_interpretation(status: str) -> str:
    if status == "raw_b1_scale_may_affect_optimization":
        return (
            "The current read-only evidence suggests that raw B1 may present a "
            "harder optimization surface than the compressed controls because "
            "observed variance is concentrated in a low-dimensional subspace and "
            "the raw per-dimension scale is smaller. This is a descriptive "
            "optimization hypothesis, not proof that normalization or PCA would "
            "improve PPO."
        )
    if status == "insufficient":
        return "Required B1, D4P8, D4P16, or normalization rows are missing."
    return "The representation-scale diagnosis is descriptive under the current inputs."


def _phase29_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Phase 29 Representation-Scale Diagnosis",
        "",
        f"Status: {analysis.get('phase29_representation_scale_status', '')}",
        "",
        "Variant-scale summary:",
    ]
    variant_rows = analysis.get("variant_scale_rows")
    if isinstance(variant_rows, list):
        for row in variant_rows:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "- "
                f"{row.get('variant_id')}: "
                f"mean row L2 {row.get('mean_row_l2_norm')}, "
                f"row-L2 std {row.get('std_row_l2_norm')}, "
                f"mean column std {row.get('mean_column_std')}"
            )
    lines.extend(["", "Tile-level B1 scale summary:"])
    tile_rows = analysis.get("tile_scale_rows")
    if isinstance(tile_rows, list):
        for row in tile_rows:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "- "
                f"{row.get('tile_id')}: "
                f"mean row L2 {row.get('mean_b1_row_l2_norm')}, "
                f"row-L2 std {row.get('std_b1_row_l2_norm')}"
            )
    lines.extend(["", "B1 normalization profiles:"])
    normalization_rows = analysis.get("b1_normalization_profile_rows")
    if isinstance(normalization_rows, list):
        for row in normalization_rows:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "- "
                f"{row.get('profile_id')}: "
                f"mean row L2 {row.get('mean_row_l2_norm')}, "
                f"row-L2 std {row.get('std_row_l2_norm')}, "
                f"mean column std {row.get('mean_column_std')}"
            )
    pca = analysis.get("pca_diagnostics")
    if not isinstance(pca, Mapping):
        pca = {}
    lines.extend(
        [
            "",
            "PCA diagnostics:",
            f"- raw embedding numerical rank: {pca.get('raw_embedding_numerical_rank')}",
            f"- raw embedding effective rank: {pca.get('raw_embedding_effective_rank')}",
            f"- top-8 variance ratio: {pca.get('top8_pca_variance_ratio')}",
            f"- top-16 variance ratio: {pca.get('top16_pca_variance_ratio')}",
            f"- mean raw embedding std: {pca.get('mean_raw_embedding_std')}",
            f"- mean D4P8 component std: {pca.get('mean_d4p8_component_std')}",
            f"- mean D4P16 component std: {pca.get('mean_d4p16_component_std')}",
            "",
            "Interpretation:",
            str(analysis.get("interpretation", "")),
            "",
            "Boundary:",
            str(
                analysis.get(
                    "claim_boundary",
                    PHASE29_REPRESENTATION_SCALE_CLAIM_BOUNDARY,
                )
            ),
            "",
            "This diagnosis does not prove that normalization would improve PPO performance.",
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
        raise ValueError(f"Phase 29 representation-scale analysis is missing {row_key}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 29 {row_key} rows must be objects")
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    return value


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


def _split_block_ids(value: object) -> list[str]:
    if isinstance(value, list):
        parts = [str(item) for item in value]
    else:
        parts = str(value).split(";")
    return [part.strip() for part in parts if part.strip()]


def _row_l2_norms(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or not matrix.size:
        return np.asarray([], dtype=float)
    return np.linalg.norm(matrix, axis=1)


def _mean_column_std(matrix: np.ndarray) -> float:
    if not matrix.size:
        return 0.0
    return _round_float(float(np.mean(np.std(matrix, axis=0))))


def _variance_ratio(
    variance: np.ndarray,
    total_variance: float,
    count: int,
) -> float:
    if total_variance <= 0.0:
        return 0.0
    return _round_float(float(variance[: int(count)].sum()) / total_variance)


def _pca_columns(count: int) -> list[str]:
    return [f"embedding_pca_{index:02d}" for index in range(int(count))]


def _unique_strings(rows: Sequence[Mapping[str, object]], field: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        text = str(value)
        if text not in seen:
            values.append(text)
            seen.add(text)
    return values


def _unique_ints(rows: Sequence[Mapping[str, object]], field: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for row in rows:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        number = int(value)
        if number not in seen:
            values.append(number)
            seen.add(number)
    return values


def _first_int_or_none(
    rows: Sequence[Mapping[str, object]],
    field: str,
) -> int | None:
    for row in rows:
        value = row.get(field)
        if value is not None and str(value).strip() != "":
            return int(value)
    return None


def _mean(values: np.ndarray | Sequence[float]) -> float:
    if len(values) == 0:
        return 0.0
    return _round_float(statistics.fmean(float(value) for value in values))


def _std(values: np.ndarray | Sequence[float]) -> float:
    if len(values) == 0:
        return 0.0
    return _round_float(float(np.std(np.asarray(list(values), dtype=float))))


def _round_float(value: float) -> float:
    return round(float(value), 10)
