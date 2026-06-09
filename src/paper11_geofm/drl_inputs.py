from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VariantInput:
    variant_id: str
    block_ids: tuple[str, ...]
    feature_columns: tuple[str, ...]
    state_matrix: np.ndarray
    reward_mode: str
    state_groups: tuple[str, ...]
    source_table: Path


def load_variant_input(phase2_output_dir: Path | str, variant_id: str) -> VariantInput:
    output_dir = Path(phase2_output_dir)
    manifest_path = output_dir / "experiment_variants.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Phase 2 variant manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants = manifest.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("Phase 2 variant manifest is missing a variants object")

    normalized_variant_id = variant_id.upper()
    variant = variants.get(normalized_variant_id)
    if not isinstance(variant, dict):
        raise ValueError(f"Unknown Phase 2 variant: {normalized_variant_id}")
    if not variant.get("ready"):
        missing = variant.get("missing", [])
        raise ValueError(
            f"Phase 2 variant {normalized_variant_id} is not ready: {missing}"
        )

    feature_table = variant.get("feature_table")
    if not feature_table:
        raise ValueError(
            f"Phase 2 variant {normalized_variant_id} has no feature_table"
        )
    source_table = output_dir / str(feature_table)
    if not source_table.exists():
        raise FileNotFoundError(
            f"Missing feature table for variant {normalized_variant_id}: "
            f"{source_table}"
        )

    required_columns = tuple(str(column) for column in variant["required_columns"])
    block_ids, matrix = _read_variant_csv(
        source_table,
        normalized_variant_id,
        required_columns,
    )

    return VariantInput(
        variant_id=normalized_variant_id,
        block_ids=block_ids,
        feature_columns=required_columns,
        state_matrix=matrix,
        reward_mode=str(variant.get("reward", "")),
        state_groups=tuple(str(group) for group in variant.get("state_groups", [])),
        source_table=source_table,
    )


def _read_variant_csv(
    path: Path,
    variant_id: str,
    required_columns: tuple[str, ...],
) -> tuple[tuple[str, ...], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = [
            column
            for column in ("block_id", *required_columns)
            if column not in fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"Feature table for variant {variant_id} is missing columns: "
                f"{missing_columns}"
            )

        block_ids: list[str] = []
        rows: list[list[float]] = []
        seen_block_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            block_id = str(row.get("block_id", "")).strip()
            if not block_id:
                raise ValueError(f"Missing block_id at {path}:{row_number}")
            if block_id in seen_block_ids:
                raise ValueError(f"Duplicate block_id in {path}: {block_id}")
            seen_block_ids.add(block_id)
            block_ids.append(block_id)
            rows.append(
                [
                    _parse_required_float(
                        row.get(column),
                        path,
                        row_number,
                        column,
                    )
                    for column in required_columns
                ]
            )

    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(0, len(required_columns))
    return tuple(block_ids), matrix


def _parse_required_float(
    value: Any,
    path: Path,
    row_number: int,
    column: str,
) -> float:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing numeric value at {path}:{row_number} column {column}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric value at {path}:{row_number} column {column}: {value!r}"
        ) from exc
