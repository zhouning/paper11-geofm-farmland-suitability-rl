from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import statistics
from os import PathLike
from pathlib import Path

import numpy as np

from .block_schema import EMBEDDING_COLUMNS, EXPLICIT_FEATURE_COLUMNS


PHASE28_COMPRESSION_CLAIM_BOUNDARY = (
    "Phase 28 compression diagnosis is a read-only follow-up over existing "
    "Phase 28 representation-control artifacts; it does not run new training, "
    "does not alter the reward, does not enable suitability reward, does not "
    "test B2/B3, and does not support final submission-level planning-performance "
    "claims."
)

PHASE28_COMPRESSION_OVERLAP_FIELDNAMES = [
    "comparator_variant_id",
    "tile_seed_pair_count",
    "mean_jaccard_overlap",
    "mean_shared_selected_blocks",
    "mean_b1_minus_comparator_reward",
    "b1_improves_comparator_count",
    "b1_selected_block_count_mean",
    "comparator_selected_block_count_mean",
    "claim_boundary",
]

PHASE28_COMPRESSION_REWARD_COMPONENT_FIELDNAMES = [
    "variant_id",
    "selected_block_count",
    "low_slope_farmland_or_orchard",
    "current_farmland_or_orchard",
    "low_slope",
    "area_score",
    "mean_slope_penalty",
    "max_slope_penalty",
    "built_up_penalty",
    "water_penalty",
    "base_planning_reward",
    "claim_boundary",
]

PHASE28_COMPRESSION_VARIANTS = ("B1", "D4P8", "D4P16")
PHASE28_COMPRESSION_COMPARATORS = ("B0", "D2", "D3", "D4P8", "D4P16")


def build_phase28_compression_diagnosis(
    summary_csv: Path | str,
    phase2_b1_features_csv: Path | str,
    d4p8_features_csv: Path | str,
    d4p16_features_csv: Path | str,
    rank_threshold: float = 1e-12,
) -> dict[str, object]:
    summary_rows = _read_csv_rows(Path(summary_csv), "Phase 28 summary CSV")
    trained_rows = [
        row
        for row in summary_rows
        if str(row.get("row_type", "")) == "trained_policy"
    ]
    if not trained_rows:
        raise ValueError("Phase 28 compression diagnosis requires trained_policy rows")

    b1_features = _read_feature_table(Path(phase2_b1_features_csv))
    d4p8_features = _read_feature_table(Path(d4p8_features_csv))
    d4p16_features = _read_feature_table(Path(d4p16_features_csv))
    feature_tables = {
        "B1": b1_features,
        "D4P8": d4p8_features,
        "D4P16": d4p16_features,
    }

    overlap_rows = _selection_overlap_rows(trained_rows)
    reward_component_rows = _reward_component_rows(trained_rows, feature_tables)
    pca_diagnostics = _pca_diagnostics(
        b1_features,
        d4p8_features,
        d4p16_features,
        rank_threshold=float(rank_threshold),
    )
    status = _phase28_compression_status(overlap_rows)

    return {
        "phase": "phase28_compression_diagnosis",
        "phase28_compression_diagnostic_status": status,
        "source_paths": {
            "summary_csv": str(Path(summary_csv)),
            "phase2_b1_features_csv": str(Path(phase2_b1_features_csv)),
            "d4p8_features_csv": str(Path(d4p8_features_csv)),
            "d4p16_features_csv": str(Path(d4p16_features_csv)),
        },
        "row_counts": {
            "summary_rows": len(summary_rows),
            "trained_policy_rows": len(trained_rows),
            "b1_feature_rows": len(b1_features),
            "d4p8_feature_rows": len(d4p8_features),
            "d4p16_feature_rows": len(d4p16_features),
        },
        "eval_tile_ids": _unique_strings(trained_rows, "eval_tile_id"),
        "seeds": _unique_ints(trained_rows, "seed"),
        "train_timesteps": _first_int_or_none(trained_rows, "train_timesteps"),
        "eval_max_steps": _first_int_or_none(trained_rows, "eval_max_steps"),
        "selection_overlap_rows": overlap_rows,
        "reward_component_rows": reward_component_rows,
        "pca_diagnostics": pca_diagnostics,
        "interpretation": _phase28_compression_interpretation(status),
        "claim_boundary": PHASE28_COMPRESSION_CLAIM_BOUNDARY,
    }


def write_phase28_compression_diagnosis_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    overlap_path = output_path / "phase28_compression_overlap.csv"
    reward_components_path = output_path / "phase28_compression_reward_components.csv"
    diagnosis_json_path = output_path / "phase28_compression_diagnosis.json"
    diagnosis_md_path = output_path / "phase28_compression_diagnosis.md"

    _write_csv_mapping_rows(
        overlap_path,
        PHASE28_COMPRESSION_OVERLAP_FIELDNAMES,
        analysis.get("selection_overlap_rows"),
        "selection_overlap_rows",
    )
    _write_csv_mapping_rows(
        reward_components_path,
        PHASE28_COMPRESSION_REWARD_COMPONENT_FIELDNAMES,
        analysis.get("reward_component_rows"),
        "reward_component_rows",
    )
    diagnosis_json_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    diagnosis_md_path.write_text(
        _phase28_compression_markdown(analysis),
        encoding="utf-8",
    )

    return {
        "overlap_csv": overlap_path,
        "reward_components_csv": reward_components_path,
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


def _selection_overlap_rows(
    trained_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    indexed = _summary_index(trained_rows)
    rows: list[dict[str, object]] = []
    for comparator in PHASE28_COMPRESSION_COMPARATORS:
        jaccards: list[float] = []
        shared_counts: list[float] = []
        deltas: list[float] = []
        b1_counts: list[float] = []
        comparator_counts: list[float] = []
        positive_count = 0
        for key in sorted(indexed, key=lambda item: (item[0], item[1])):
            variants = indexed[key]
            b1_row = variants.get("B1")
            comparator_row = variants.get(comparator)
            if b1_row is None or comparator_row is None:
                continue
            b1_selected = _selected_block_set(b1_row)
            comparator_selected = _selected_block_set(comparator_row)
            union = b1_selected | comparator_selected
            shared = b1_selected & comparator_selected
            jaccard = len(shared) / len(union) if union else 1.0
            delta = _float_value(b1_row, "total_contract_reward") - _float_value(
                comparator_row,
                "total_contract_reward",
            )
            if delta > 0.0:
                positive_count += 1
            jaccards.append(jaccard)
            shared_counts.append(float(len(shared)))
            deltas.append(delta)
            b1_counts.append(float(len(b1_selected)))
            comparator_counts.append(float(len(comparator_selected)))
        if not jaccards:
            continue
        rows.append(
            {
                "comparator_variant_id": comparator,
                "tile_seed_pair_count": len(jaccards),
                "mean_jaccard_overlap": _mean(jaccards),
                "mean_shared_selected_blocks": _mean(shared_counts),
                "mean_b1_minus_comparator_reward": _mean(deltas),
                "b1_improves_comparator_count": positive_count,
                "b1_selected_block_count_mean": _mean(b1_counts),
                "comparator_selected_block_count_mean": _mean(comparator_counts),
                "claim_boundary": PHASE28_COMPRESSION_CLAIM_BOUNDARY,
            }
        )
    return rows


def _summary_index(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int], dict[str, Mapping[str, object]]]:
    indexed: dict[tuple[str, int], dict[str, Mapping[str, object]]] = {}
    for row in rows:
        variant_id = str(row.get("variant_id", ""))
        if not variant_id:
            continue
        key = (str(row.get("eval_tile_id", "")), _int_value(row, "seed"))
        by_variant = indexed.setdefault(key, {})
        if variant_id in by_variant:
            raise ValueError(
                "Phase 28 compression diagnosis requires unique trained rows "
                f"for {key[0]} seed {key[1]} variant {variant_id}"
            )
        by_variant[variant_id] = row
    return indexed


def _selected_block_set(row: Mapping[str, object]) -> set[str]:
    value = row.get("selected_block_ids", "")
    if isinstance(value, list):
        parts = [str(item) for item in value]
    else:
        parts = str(value).split(";")
    return {part.strip() for part in parts if part.strip()}


def _reward_component_rows(
    trained_rows: Sequence[Mapping[str, object]],
    feature_tables: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant_id in PHASE28_COMPRESSION_VARIANTS:
        table = feature_tables.get(variant_id)
        if not isinstance(table, Mapping):
            raise ValueError(f"Missing feature table for {variant_id}")
        selected = [
            block_id
            for row in trained_rows
            if str(row.get("variant_id", "")) == variant_id
            for block_id in _selected_block_ids(row)
        ]
        if not selected:
            continue
        component_values = {
            component: []
            for component in PHASE28_COMPRESSION_REWARD_COMPONENT_FIELDNAMES
            if component
            not in {"variant_id", "selected_block_count", "claim_boundary"}
        }
        for block_id in selected:
            feature_row = table.get(block_id)
            if not isinstance(feature_row, Mapping):
                raise ValueError(
                    f"Selected block {block_id} is missing from {variant_id} feature table"
                )
            components = _reward_components(feature_row)
            for component, value in components.items():
                component_values[component].append(value)
        output_row: dict[str, object] = {
            "variant_id": variant_id,
            "selected_block_count": len(selected),
            "claim_boundary": PHASE28_COMPRESSION_CLAIM_BOUNDARY,
        }
        for component, values in component_values.items():
            output_row[component] = _mean(values)
        rows.append(output_row)
    return rows


def _selected_block_ids(row: Mapping[str, object]) -> list[str]:
    value = row.get("selected_block_ids", "")
    if isinstance(value, list):
        parts = [str(item) for item in value]
    else:
        parts = str(value).split(";")
    return [part.strip() for part in parts if part.strip()]


def _reward_components(row: Mapping[str, object]) -> dict[str, float]:
    missing = [
        column
        for column in EXPLICIT_FEATURE_COLUMNS
        if column not in row or str(row[column]).strip() == ""
    ]
    if missing:
        raise ValueError(
            "Phase 28 compression diagnosis requires explicit features: "
            f"{', '.join(missing)}"
        )
    low_slope_farmland_or_orchard = 0.35 * _clip01(
        _float_value(row, "explicit_feature_16")
    )
    current_farmland_or_orchard = 0.20 * max(
        _clip01(_float_value(row, "explicit_feature_04")),
        _clip01(_float_value(row, "explicit_feature_07")),
    )
    low_slope = 0.10 * _clip01(_float_value(row, "explicit_feature_13"))
    area_score = 0.10 * _clip01(_float_value(row, "explicit_feature_00") / 5.0)
    mean_slope_penalty = -0.15 * _clip01(
        _float_value(row, "explicit_feature_01") / 25.0
    )
    max_slope_penalty = -0.05 * _clip01(
        _float_value(row, "explicit_feature_02") / 35.0
    )
    built_up_penalty = -0.10 * _clip01(_float_value(row, "explicit_feature_09"))
    water_penalty = -0.10 * _clip01(_float_value(row, "explicit_feature_10"))
    base_planning_reward = (
        low_slope_farmland_or_orchard
        + current_farmland_or_orchard
        + low_slope
        + area_score
        + mean_slope_penalty
        + max_slope_penalty
        + built_up_penalty
        + water_penalty
    )
    return {
        "low_slope_farmland_or_orchard": low_slope_farmland_or_orchard,
        "current_farmland_or_orchard": current_farmland_or_orchard,
        "low_slope": low_slope,
        "area_score": area_score,
        "mean_slope_penalty": mean_slope_penalty,
        "max_slope_penalty": max_slope_penalty,
        "built_up_penalty": built_up_penalty,
        "water_penalty": water_penalty,
        "base_planning_reward": base_planning_reward,
    }


def _pca_diagnostics(
    b1_features: Mapping[str, Mapping[str, object]],
    d4p8_features: Mapping[str, Mapping[str, object]],
    d4p16_features: Mapping[str, Mapping[str, object]],
    rank_threshold: float,
) -> dict[str, object]:
    raw_embedding_matrix = _matrix_for_columns(b1_features, EMBEDDING_COLUMNS)
    centered = raw_embedding_matrix - raw_embedding_matrix.mean(axis=0)
    if centered.size:
        _u, singular_values, _vt = np.linalg.svd(centered, full_matrices=False)
    else:
        singular_values = np.asarray([], dtype=float)
    variance = singular_values**2
    total_variance = float(variance.sum())
    d4p8_columns = _pca_columns(8)
    d4p16_columns = _pca_columns(16)
    d4p8_matrix = _matrix_for_columns(d4p8_features, d4p8_columns)
    d4p16_matrix = _matrix_for_columns(d4p16_features, d4p16_columns)
    return {
        "raw_embedding_rank_threshold": float(rank_threshold),
        "raw_embedding_numerical_rank": int(np.sum(singular_values > rank_threshold)),
        "top8_pca_variance_ratio": _variance_ratio(variance, total_variance, 8),
        "top16_pca_variance_ratio": _variance_ratio(variance, total_variance, 16),
        "mean_raw_embedding_std": _mean_column_std(raw_embedding_matrix),
        "mean_d4p8_component_std": _mean_column_std(d4p8_matrix),
        "mean_d4p16_component_std": _mean_column_std(d4p16_matrix),
    }


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


def _pca_columns(count: int) -> list[str]:
    return [f"embedding_pca_{index:02d}" for index in range(int(count))]


def _variance_ratio(
    variance: np.ndarray,
    total_variance: float,
    count: int,
) -> float:
    if total_variance <= 0.0:
        return 0.0
    return _round_float(float(variance[: int(count)].sum()) / total_variance)


def _mean_column_std(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return _round_float(float(np.mean(np.std(matrix, axis=0))))


def _phase28_compression_status(
    overlap_rows: Sequence[Mapping[str, object]],
) -> str:
    by_comparator = {
        str(row.get("comparator_variant_id", "")): row for row in overlap_rows
    }
    d4_rows = [
        by_comparator.get("D4P8"),
        by_comparator.get("D4P16"),
    ]
    if not all(isinstance(row, Mapping) for row in d4_rows):
        return "insufficient"
    if all(
        float(row.get("mean_b1_minus_comparator_reward", 0.0)) < 0.0
        and float(row.get("mean_jaccard_overlap", 1.0)) <= 0.1
        for row in d4_rows
        if isinstance(row, Mapping)
    ):
        return "compressed_controls_select_distinct_higher_reward_blocks"
    if all(
        float(row.get("mean_b1_minus_comparator_reward", 0.0)) < 0.0
        for row in d4_rows
        if isinstance(row, Mapping)
    ):
        return "compressed_controls_exceed_raw"
    return "compression_diagnosis_descriptive"


def _phase28_compression_interpretation(status: str) -> str:
    if status == "compressed_controls_select_distinct_higher_reward_blocks":
        return (
            "D4P8 and D4P16 exceed raw B1 while selecting nearly disjoint "
            "block sets under the current Phase 28 protocol. This supports a "
            "read-only diagnostic association with selected-block composition, "
            "not a causal claim that PCA is superior."
        )
    if status == "compressed_controls_exceed_raw":
        return (
            "D4P8 and D4P16 exceed raw B1, but selection overlap is not low "
            "enough to describe the advantage as a distinct-block pattern."
        )
    if status == "insufficient":
        return "Required D4P8/D4P16 trained-policy rows are missing."
    return "The compression diagnosis is descriptive under the current inputs."


def _phase28_compression_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Phase 28 Compression Diagnosis",
        "",
        f"Status: {analysis.get('phase28_compression_diagnostic_status', '')}",
        "",
        "Selection overlap:",
    ]
    overlap_rows = analysis.get("selection_overlap_rows")
    if isinstance(overlap_rows, list):
        for row in overlap_rows:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "- "
                f"B1 versus {row.get('comparator_variant_id')}: "
                f"mean Jaccard {row.get('mean_jaccard_overlap')}, "
                f"shared blocks {row.get('mean_shared_selected_blocks')}, "
                f"B1 minus comparator reward "
                f"{row.get('mean_b1_minus_comparator_reward')}"
            )
    lines.extend(["", "Reward components:"])
    component_rows = analysis.get("reward_component_rows")
    if isinstance(component_rows, list):
        for row in component_rows:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "- "
                f"{row.get('variant_id')}: "
                f"low-slope farmland/orchard "
                f"{row.get('low_slope_farmland_or_orchard')}, "
                f"current farmland/orchard "
                f"{row.get('current_farmland_or_orchard')}, "
                f"base reward {row.get('base_planning_reward')}"
            )
    pca = analysis.get("pca_diagnostics")
    if not isinstance(pca, Mapping):
        pca = {}
    lines.extend(
        [
            "",
            "Compression-scale diagnostics:",
            f"- raw embedding rank: {pca.get('raw_embedding_numerical_rank')}",
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
            str(analysis.get("claim_boundary", PHASE28_COMPRESSION_CLAIM_BOUNDARY)),
            "",
            (
                "This diagnosis does not prove that PCA is intrinsically superior "
                "to raw GeoFM embeddings."
            ),
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
        raise ValueError(f"Phase 28 compression analysis is missing {row_key}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 28 compression {row_key} rows must be objects")
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


def _float_value(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing numeric field: {field}")
    return float(value)


def _int_value(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing integer field: {field}")
    return int(value)


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return _round_float(statistics.fmean(values))


def _round_float(value: float) -> float:
    return round(float(value), 10)
