from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .block_schema import EMBEDDING_COLUMNS, EXPLICIT_FEATURE_COLUMNS
from .drl_inputs import VariantInput, load_variant_input


PHASE8_CLAIM_BOUNDARY = (
    "Phase 8 builds diagnostic ablation-control feature tables; it does not "
    "train, tune, evaluate, compare, or report a useful DRL policy."
)

def build_phase8_ablation_controls(
    phase2_output_dir: Path | str,
    seed: int = 0,
    pca_dimensions: Sequence[int] = (8, 16),
) -> dict[str, object]:
    normalized_pca_dimensions = _normalize_pca_dimensions(pca_dimensions)
    source_b0 = load_variant_input(phase2_output_dir, "B0")
    source_b1 = load_variant_input(phase2_output_dir, "B1")
    if source_b0.block_ids != source_b1.block_ids:
        raise ValueError("Phase 8 requires aligned B0 and B1 block IDs")

    explicit_matrix = source_b0.state_matrix.astype(float, copy=True)
    embedding_matrix = _matrix_for_columns(source_b1, EMBEDDING_COLUMNS)

    d2_matrix = _random_embedding_control(embedding_matrix, seed)
    d3_matrix, shuffle_permutation = _shuffled_embedding_control(
        embedding_matrix,
        seed,
    )
    pca_matrices = {
        dimension: _pca_embedding_control(embedding_matrix, dimension)
        for dimension in normalized_pca_dimensions
    }

    variant_specs = _variant_specs(normalized_pca_dimensions)
    control_matrices = {
        "D2": d2_matrix,
        "D3": d3_matrix,
        **{
            f"D4P{dimension}": matrix
            for dimension, matrix in pca_matrices.items()
        },
    }
    variant_tables = {
        variant_id: _build_rows(
            source_b0.block_ids,
            explicit_matrix,
            spec["control_columns"],
            control_matrices[variant_id],
        )
        for variant_id, spec in variant_specs.items()
    }
    manifest = _build_manifest(variant_specs, variant_tables)
    summary = _build_summary(
        variant_specs,
        variant_tables,
        shuffle_permutation,
        normalized_pca_dimensions,
    )

    return {
        "phase": "phase8_ablation_control_features",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "seed": int(seed),
        "variant_ids": list(variant_specs.keys()),
        "source_variants": {
            "B0": _source_summary(source_b0),
            "B1": _source_summary(source_b1),
        },
        "summary": summary,
        "manifest": manifest,
        "variant_tables": variant_tables,
        "claim_boundary": PHASE8_CLAIM_BOUNDARY,
    }


def write_phase8_ablation_artifacts(
    protocol: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path | dict[str, Path]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = protocol.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("Phase 8 protocol is missing a manifest object")
    variants = manifest.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError("Phase 8 manifest is missing a variants object")
    variant_tables = protocol.get("variant_tables")
    if not isinstance(variant_tables, Mapping):
        raise ValueError("Phase 8 protocol is missing variant tables")

    table_paths: dict[str, Path] = {}
    for variant_id, variant in variants.items():
        if not isinstance(variant, Mapping):
            raise ValueError(f"Phase 8 variant metadata must be an object: {variant_id}")
        rows = variant_tables.get(variant_id)
        if not isinstance(rows, list):
            raise ValueError(f"Phase 8 protocol is missing rows for {variant_id}")
        feature_table = variant.get("feature_table")
        if not feature_table:
            raise ValueError(f"Phase 8 variant has no feature_table: {variant_id}")
        required_columns = variant.get("required_columns")
        if not isinstance(required_columns, list):
            raise ValueError(
                f"Phase 8 variant has no required_columns list: {variant_id}"
            )
        table_path = output_path / str(feature_table)
        _write_variant_csv(table_path, rows, [str(item) for item in required_columns])
        table_paths[str(variant_id)] = table_path

    manifest_path = output_path / "experiment_variants.json"
    summary_path = output_path / "phase8_ablation_control_summary.json"
    manifest_path.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            _summary_payload(protocol, table_paths),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "manifest": manifest_path,
        "summary": summary_path,
        "variant_tables": table_paths,
    }


def _normalize_pca_dimensions(pca_dimensions: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(dimension) for dimension in pca_dimensions)
    if not normalized:
        raise ValueError("At least one Phase 8 PCA dimension must be requested")
    invalid = [dimension for dimension in normalized if dimension <= 0]
    if invalid:
        raise ValueError(f"Phase 8 PCA dimensions must be positive: {invalid}")
    return normalized


def _matrix_for_columns(
    variant_input: VariantInput,
    columns: Sequence[str],
) -> np.ndarray:
    indexes = [variant_input.feature_columns.index(column) for column in columns]
    return variant_input.state_matrix[:, indexes].astype(float, copy=True)


def _random_embedding_control(embedding_matrix: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    random_values = rng.standard_normal(size=embedding_matrix.shape)
    means = embedding_matrix.mean(axis=0)
    stds = embedding_matrix.std(axis=0)
    safe_stds = np.where(stds > 0.0, stds, 1.0)
    return random_values * safe_stds + means


def _shuffled_embedding_control(
    embedding_matrix: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(embedding_matrix.shape[0])
    if embedding_matrix.shape[0] > 1 and np.array_equal(
        permutation,
        np.arange(embedding_matrix.shape[0]),
    ):
        permutation = np.roll(permutation, 1)
    return embedding_matrix[permutation], [int(index) for index in permutation]


def _pca_embedding_control(
    embedding_matrix: np.ndarray,
    dimension: int,
) -> np.ndarray:
    centered = embedding_matrix - embedding_matrix.mean(axis=0)
    if centered.size == 0:
        return np.zeros((embedding_matrix.shape[0], dimension), dtype=float)

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    effective_rank = int(np.sum(singular_values > 1e-12))
    usable_components = min(effective_rank, int(dimension))
    if usable_components:
        scores = centered @ vt[:usable_components].T
    else:
        scores = np.zeros((embedding_matrix.shape[0], 0), dtype=float)
    if usable_components < dimension:
        padding = np.zeros(
            (embedding_matrix.shape[0], int(dimension) - usable_components),
            dtype=float,
        )
        scores = np.hstack([scores, padding])
    return scores.astype(float, copy=False)


def _variant_specs(pca_dimensions: Sequence[int]) -> dict[str, dict[str, object]]:
    specs: dict[str, dict[str, object]] = {
        "D2": {
            "description": "Explicit planning features plus deterministic random 64d control.",
            "state_groups": [
                "explicit_planning_features",
                "random_embedding_control",
            ],
            "reward": "base_planning_reward",
            "control_columns": list(EMBEDDING_COLUMNS),
            "feature_table": "variant_D2_features.csv",
        },
        "D3": {
            "description": "Explicit planning features plus block-shuffled GeoFM embeddings.",
            "state_groups": [
                "explicit_planning_features",
                "shuffled_geofm_embedding",
            ],
            "reward": "base_planning_reward",
            "control_columns": list(EMBEDDING_COLUMNS),
            "feature_table": "variant_D3_features.csv",
        },
    }
    for dimension in pca_dimensions:
        variant_id = f"D4P{int(dimension)}"
        specs[variant_id] = {
            "description": (
                "Explicit planning features plus deterministic PCA-compressed "
                f"GeoFM embeddings ({int(dimension)} components)."
            ),
            "state_groups": [
                "explicit_planning_features",
                f"pca_geofm_embedding_{int(dimension)}",
            ],
            "reward": "base_planning_reward",
            "control_columns": [
                f"embedding_pca_{index:02d}" for index in range(int(dimension))
            ],
            "feature_table": f"variant_{variant_id}_features.csv",
        }
    return specs


def _build_rows(
    block_ids: Sequence[str],
    explicit_matrix: np.ndarray,
    control_columns: Sequence[str],
    control_matrix: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for row_index, block_id in enumerate(block_ids):
        row: dict[str, object] = {"block_id": str(block_id)}
        for column_index, column in enumerate(EXPLICIT_FEATURE_COLUMNS):
            row[column] = float(explicit_matrix[row_index, column_index])
        for column_index, column in enumerate(control_columns):
            row[column] = float(control_matrix[row_index, column_index])
        rows.append(row)
    return rows


def _build_manifest(
    variant_specs: dict[str, dict[str, object]],
    variant_tables: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    variants = {}
    for variant_id, spec in variant_specs.items():
        variants[variant_id] = {
            "description": spec["description"],
            "state_groups": list(spec["state_groups"]),
            "reward": spec["reward"],
            "required_columns": list(EXPLICIT_FEATURE_COLUMNS)
            + list(spec["control_columns"]),
            "ready": True,
            "missing": [],
            "feature_table": spec["feature_table"],
            "row_count": len(variant_tables[variant_id]),
        }
    return {"claim_boundary": PHASE8_CLAIM_BOUNDARY, "variants": variants}


def _build_summary(
    variant_specs: dict[str, dict[str, object]],
    variant_tables: dict[str, list[dict[str, object]]],
    shuffle_permutation: list[int],
    pca_dimensions: Sequence[int],
) -> dict[str, dict[str, object]]:
    summary = {}
    for variant_id, spec in variant_specs.items():
        row_count = len(variant_tables[variant_id])
        control_columns = list(spec["control_columns"])
        summary[variant_id] = {
            "row_count": row_count,
            "n_features": len(EXPLICIT_FEATURE_COLUMNS) + len(control_columns),
            "feature_table": spec["feature_table"],
            "state_groups": list(spec["state_groups"]),
            "reward": spec["reward"],
        }
    summary["D3"]["shuffle_permutation"] = list(shuffle_permutation)
    for dimension in pca_dimensions:
        variant_id = f"D4P{int(dimension)}"
        summary[variant_id]["pca_dimension_requested"] = int(dimension)
        summary[variant_id]["pca_dimension_emitted"] = int(dimension)
    return summary


def _source_summary(variant_input: VariantInput) -> dict[str, object]:
    return {
        "variant_id": variant_input.variant_id,
        "n_blocks": len(variant_input.block_ids),
        "n_features": len(variant_input.feature_columns),
        "source_table": str(variant_input.source_table),
    }


def _write_variant_csv(
    path: Path,
    rows: list[dict[str, object]],
    required_columns: Sequence[str],
) -> None:
    fieldnames = ["block_id", *required_columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _summary_payload(
    protocol: Mapping[str, object],
    table_paths: Mapping[str, Path],
) -> dict[str, object]:
    payload = {
        key: value
        for key, value in protocol.items()
        if key not in {"variant_tables"}
    }
    payload["artifacts"] = {
        "manifest": "experiment_variants.json",
        "summary": "phase8_ablation_control_summary.json",
        "variant_tables": {
            variant_id: path.name for variant_id, path in table_paths.items()
        },
    }
    return payload
