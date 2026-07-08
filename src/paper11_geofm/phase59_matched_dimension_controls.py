from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from os import PathLike
from pathlib import Path

import numpy as np

from .block_schema import EXPLICIT_FEATURE_COLUMNS


PHASE59_CLAIM_BOUNDARY = (
    "Phase 59 is a matched-dimension control audit over compressed GeoFM "
    "base-reward held-out Bishan policy runs. It tests D4P8/D4P16 against "
    "8- and 16-dimensional random or shuffled controls; it does not enable "
    "suitability reward, does not test B2/B3, does not test cross-region "
    "transfer, and does not validate independent agronomic suitability."
)

PHASE59_COMPRESSED_VARIANTS = ("D4P8", "D4P16")
PHASE59_MATCHED_CONTROL_VARIANTS = ("D5R8", "D5S8", "D5R16", "D5S16")
PHASE59_REQUIRED_VARIANTS = (
    "D4P8",
    "D4P16",
    "D5R8",
    "D5S8",
    "D5R16",
    "D5S16",
)
PHASE59_MATCHED_COMPARISONS = (
    ("D4P8", "D5R8"),
    ("D4P8", "D5S8"),
    ("D4P16", "D5R16"),
    ("D4P16", "D5S16"),
)


def build_phase59_matched_dimension_control_tables(
    b0_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    d4p8_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    d4p16_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    seed: int = 59,
) -> dict[str, object]:
    b0_rows = _load_rows(b0_rows_or_csv, "B0")
    d4p8_rows = _load_rows(d4p8_rows_or_csv, "D4P8")
    d4p16_rows = _load_rows(d4p16_rows_or_csv, "D4P16")
    _require_aligned_block_ids(b0_rows, d4p8_rows, "D4P8")
    _require_aligned_block_ids(b0_rows, d4p16_rows, "D4P16")

    explicit_columns = _available_explicit_columns(b0_rows)
    explicit_matrix = _matrix_for_columns(b0_rows, explicit_columns)
    d4p8_matrix = _matrix_for_prefix(d4p8_rows, "embedding_pca_", "D4P8")
    d4p16_matrix = _matrix_for_prefix(d4p16_rows, "embedding_pca_", "D4P16")

    rng = np.random.default_rng(int(seed))
    tables = {
        "D5R8": _build_rows(
            b0_rows,
            explicit_columns,
            explicit_matrix,
            _matched_control_matrix(d4p8_matrix, rng),
        ),
        "D5S8": _build_rows(
            b0_rows,
            explicit_columns,
            explicit_matrix,
            _shuffled_matrix(d4p8_matrix, rng),
        ),
        "D5R16": _build_rows(
            b0_rows,
            explicit_columns,
            explicit_matrix,
            _matched_control_matrix(d4p16_matrix, rng),
        ),
        "D5S16": _build_rows(
            b0_rows,
            explicit_columns,
            explicit_matrix,
            _shuffled_matrix(d4p16_matrix, rng),
        ),
    }
    manifest = _build_manifest(tables, explicit_columns)
    summary = _build_feature_summary(tables, d4p8_matrix, d4p16_matrix)
    return {
        "phase": "phase59_matched_dimension_control_features",
        "seed": int(seed),
        "variant_ids": list(PHASE59_MATCHED_CONTROL_VARIANTS),
        "summary": summary,
        "manifest": manifest,
        "variant_tables": tables,
        "claim_boundary": PHASE59_CLAIM_BOUNDARY,
    }


def _load_rows(
    rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    variant_id: str,
) -> list[dict[str, object]]:
    if isinstance(rows_or_csv, (str, PathLike)):
        path = Path(rows_or_csv)
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 59 {variant_id} CSV: {path}")
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
        raise ValueError(f"Phase 59 requires aligned block IDs for {variant_id}")


def _available_explicit_columns(rows: Sequence[Mapping[str, object]]) -> list[str]:
    if not rows:
        raise ValueError("Phase 59 requires feature rows")
    present = set(rows[0].keys())
    columns = [column for column in EXPLICIT_FEATURE_COLUMNS if column in present]
    if not columns:
        raise ValueError("Phase 59 requires explicit planning feature columns")
    return columns


def _matrix_for_columns(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [[float(row[column]) for column in columns] for row in rows],
        dtype=float,
    )


def _matrix_for_prefix(
    rows: Sequence[Mapping[str, object]],
    prefix: str,
    variant_id: str,
) -> np.ndarray:
    if not rows:
        raise ValueError(f"Phase 59 requires embedding_pca columns for {variant_id}")
    columns = sorted(column for column in rows[0] if str(column).startswith(prefix))
    if not columns:
        raise ValueError(f"Phase 59 requires embedding_pca columns for {variant_id}")
    return _matrix_for_columns(rows, columns)


def _matched_control_matrix(source_matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    random_values = rng.standard_normal(size=source_matrix.shape)
    means = source_matrix.mean(axis=0)
    stds = source_matrix.std(axis=0)
    safe_stds = np.where(stds > 0.0, stds, 1.0)
    return random_values * safe_stds + means


def _shuffled_matrix(source_matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permutation = rng.permutation(source_matrix.shape[0])
    if source_matrix.shape[0] > 1 and np.array_equal(
        permutation,
        np.arange(source_matrix.shape[0]),
    ):
        permutation = np.roll(permutation, 1)
    return source_matrix[permutation]


def _build_rows(
    source_rows: Sequence[Mapping[str, object]],
    explicit_columns: Sequence[str],
    explicit_matrix: np.ndarray,
    control_matrix: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_index, source_row in enumerate(source_rows):
        row: dict[str, object] = {"block_id": str(source_row["block_id"])}
        for column_index, column in enumerate(explicit_columns):
            row[column] = float(explicit_matrix[row_index, column_index])
        for column_index in range(control_matrix.shape[1]):
            row[f"matched_control_{column_index:02d}"] = float(
                control_matrix[row_index, column_index]
            )
        rows.append(row)
    return rows


def _build_manifest(
    tables: Mapping[str, list[dict[str, object]]],
    explicit_columns: Sequence[str],
) -> dict[str, object]:
    variants: dict[str, dict[str, object]] = {}
    for variant_id in PHASE59_MATCHED_CONTROL_VARIANTS:
        rows = tables[variant_id]
        control_columns = sorted(
            column
            for column in rows[0]
            if str(column).startswith("matched_control_")
        )
        variants[variant_id] = {
            "description": f"Explicit planning features plus {variant_id} matched-dimension control features.",
            "state_groups": [
                "explicit_planning_features",
                f"phase59_{variant_id.lower()}_matched_control",
            ],
            "reward": "base_planning_reward",
            "required_columns": list(explicit_columns) + control_columns,
            "ready": True,
            "missing": [],
            "feature_table": f"variant_{variant_id}_features.csv",
            "row_count": len(rows),
        }
    return {"claim_boundary": PHASE59_CLAIM_BOUNDARY, "variants": variants}


def _build_feature_summary(
    tables: Mapping[str, list[dict[str, object]]],
    d4p8_matrix: np.ndarray,
    d4p16_matrix: np.ndarray,
) -> dict[str, dict[str, object]]:
    source_by_variant = {
        "D5R8": ("D4P8", d4p8_matrix.shape[1], "random_matched_moments"),
        "D5S8": ("D4P8", d4p8_matrix.shape[1], "shuffled_pca_scores"),
        "D5R16": ("D4P16", d4p16_matrix.shape[1], "random_matched_moments"),
        "D5S16": ("D4P16", d4p16_matrix.shape[1], "shuffled_pca_scores"),
    }
    summary: dict[str, dict[str, object]] = {}
    for variant_id, (source_variant_id, dimension, control_type) in source_by_variant.items():
        summary[variant_id] = {
            "row_count": len(tables[variant_id]),
            "control_dimension": int(dimension),
            "source_variant_id": source_variant_id,
            "control_type": control_type,
            "feature_table": f"variant_{variant_id}_features.csv",
        }
    return summary


def write_phase59_matched_dimension_control_tables(
    protocol: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path | dict[str, Path]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = protocol["manifest"]
    summary = protocol["summary"]
    variant_tables = protocol["variant_tables"]
    if not isinstance(manifest, Mapping):
        raise ValueError("Phase 59 protocol is missing a manifest")
    if not isinstance(summary, Mapping):
        raise ValueError("Phase 59 protocol is missing a summary")
    if not isinstance(variant_tables, Mapping):
        raise ValueError("Phase 59 protocol is missing variant tables")

    variants = manifest.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError("Phase 59 manifest is missing variants")

    table_paths: dict[str, Path] = {}
    for variant_id, rows in variant_tables.items():
        variant = variants[str(variant_id)]
        if not isinstance(variant, Mapping):
            raise ValueError(f"Phase 59 variant metadata must be an object: {variant_id}")
        feature_table = variant.get("feature_table")
        required_columns = variant.get("required_columns")
        if not feature_table:
            raise ValueError(f"Phase 59 variant has no feature table: {variant_id}")
        if not isinstance(required_columns, list):
            raise ValueError(f"Phase 59 variant has no required columns: {variant_id}")
        path = output_path / str(feature_table)
        _write_variant_csv(path, rows, [str(column) for column in required_columns])
        table_paths[str(variant_id)] = path

    manifest_path = output_path / "experiment_variants.json"
    summary_path = output_path / "phase59_matched_dimension_control_feature_summary.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "manifest": manifest_path,
        "summary": summary_path,
        "variant_tables": table_paths,
    }


def _write_variant_csv(
    path: Path,
    rows: object,
    required_columns: Sequence[str],
) -> None:
    if not isinstance(rows, list):
        raise ValueError("Phase 59 variant table rows must be a list")
    fieldnames = ["block_id", *required_columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Phase 59 variant table rows must be objects")
            writer.writerow({field: row.get(field, "") for field in fieldnames})
