from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import math
import random
import statistics
from itertools import product
from os import PathLike
from pathlib import Path

import numpy as np

from .block_schema import EXPLICIT_FEATURE_COLUMNS
from .padded_heldout_policy import SUMMARY_FIELDNAMES


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




PHASE59_DELTA_FIELDNAMES = [
    "compressed_variant_id",
    "matched_control_variant_id",
    "eval_tile_id",
    "seed",
    "compressed_reward",
    "matched_control_reward",
    "compressed_minus_matched_control_reward",
    "compressed_improves_matched_control",
    "train_timesteps",
    "eval_max_steps",
    "claim_boundary",
]

PHASE59_CLUSTER_FIELDNAMES = [
    "eval_tile_id",
    "seed",
    "cluster_delta_count",
    "mean_cluster_delta",
    "cluster_positive",
    "claim_boundary",
]


def build_phase59_matched_dimension_control_analysis(
    summary_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    metadata: Mapping[str, object] | None = None,
    bootstrap_iterations: int = 5000,
    random_seed: int = 59,
    pooled_positive_threshold: float = 0.5,
) -> dict[str, object]:
    if int(bootstrap_iterations) <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    rows = _load_summary_rows(summary_rows_or_csv)
    trained_rows = [
        row for row in rows if str(row.get("row_type", "")) == "trained_policy"
    ]
    metadata_map = {} if metadata is None else dict(metadata)
    eval_tile_ids = _metadata_string_list(
        metadata_map,
        "eval_tile_ids",
        fallback=_unique_strings(trained_rows, "eval_tile_id"),
    )
    seeds = _metadata_int_list(
        metadata_map,
        "seeds",
        fallback=_unique_ints(trained_rows, "seed"),
    )
    coverage_issues = _coverage_issues(
        trained_rows,
        variants=PHASE59_REQUIRED_VARIANTS,
        eval_tile_ids=eval_tile_ids,
        seeds=seeds,
    )
    delta_rows = _matched_delta_rows(trained_rows, eval_tile_ids, seeds)
    learned_policy = _phase59_policy_summary(trained_rows, delta_rows)
    pooled = _delta_summary(
        [
            float(row["compressed_minus_matched_control_reward"])
            for row in delta_rows
        ],
        bootstrap_iterations=int(bootstrap_iterations),
        random_seed=int(random_seed),
    )
    cluster_rows = _cluster_rows(delta_rows)
    cluster_summary = _cluster_summary(cluster_rows)
    signed_rank_summary = _signed_rank_summary(cluster_rows)
    status = _phase59_status(
        learned_policy["matched_deltas"],
        pooled,
        coverage_issues,
        pooled_positive_threshold=float(pooled_positive_threshold),
    )
    return {
        "phase": "phase59_matched_dimension_control_analysis",
        "variants": list(PHASE59_REQUIRED_VARIANTS),
        "matched_comparisons": [
            {"compressed_variant_id": left, "matched_control_variant_id": right}
            for left, right in PHASE59_MATCHED_COMPARISONS
        ],
        "eval_tile_ids": eval_tile_ids,
        "seeds": seeds,
        "source_rows": rows,
        "main_summary_rows": _main_summary_rows(rows),
        "delta_rows": delta_rows,
        "learned_policy": learned_policy,
        "pooled_matched_control_delta": pooled,
        "cluster_rows": cluster_rows,
        "cluster_summary": cluster_summary,
        "signed_rank_summary": signed_rank_summary,
        "coverage_issues": coverage_issues,
        "phase59_matched_dimension_status": status,
        "conclusion": _phase59_conclusion(status),
        "claim_boundary": PHASE59_CLAIM_BOUNDARY,
    }


def _load_summary_rows(
    summary_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(summary_rows_or_csv, (str, PathLike)):
        path = Path(summary_rows_or_csv)
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 59 summary CSV: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return [dict(row) for row in summary_rows_or_csv]


def _metadata_string_list(
    metadata: Mapping[str, object],
    key: str,
    fallback: list[str],
) -> list[str]:
    value = metadata.get(key)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item).strip()]
    return fallback


def _metadata_int_list(
    metadata: Mapping[str, object],
    key: str,
    fallback: list[int],
) -> list[int]:
    value = metadata.get(key)
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, Sequence):
        return [int(item) for item in value if str(item).strip()]
    return fallback


def _unique_strings(rows: list[dict[str, object]], field: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        text = str(value)
        if text not in seen:
            seen.add(text)
            values.append(text)
    return values


def _unique_ints(rows: list[dict[str, object]], field: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for row in rows:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        number = int(value)
        if number not in seen:
            seen.add(number)
            values.append(number)
    return values


def _coverage_issues(
    rows: list[dict[str, object]],
    variants: Sequence[str],
    eval_tile_ids: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, object]:
    expected_keys = {
        (str(eval_tile_id), int(seed), str(variant_id))
        for eval_tile_id in eval_tile_ids
        for seed in seeds
        for variant_id in variants
    }
    expected_variants = {str(item) for item in variants}
    expected_tiles = {str(item) for item in eval_tile_ids}
    expected_seeds = {int(item) for item in seeds}
    observed_keys: set[tuple[str, int, str]] = set()
    duplicate_keys: set[tuple[str, int, str]] = set()
    unexpected_keys: set[tuple[str, int, str]] = set()
    for row in rows:
        key = (
            str(row.get("eval_tile_id", "")),
            _int_value(row, "seed"),
            str(row.get("variant_id", "")),
        )
        if key in observed_keys:
            duplicate_keys.add(key)
        observed_keys.add(key)
        if (
            key[2] not in expected_variants
            or key[0] not in expected_tiles
            or key[1] not in expected_seeds
        ):
            unexpected_keys.add(key)
    comparable_observed = {
        key
        for key in observed_keys
        if key[2] in expected_variants
        and key[0] in expected_tiles
        and key[1] in expected_seeds
    }
    return {
        "missing_variant_rows": _variant_key_dicts(expected_keys - comparable_observed),
        "unexpected_variant_rows": _variant_key_dicts(unexpected_keys),
        "duplicate_variant_rows": _variant_key_dicts(duplicate_keys),
    }


def _matched_delta_rows(
    rows: list[dict[str, object]],
    eval_tile_ids: Sequence[str],
    seeds: Sequence[int],
) -> list[dict[str, object]]:
    indexed: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (str(row.get("eval_tile_id", "")), _int_value(row, "seed"))
        indexed.setdefault(key, {})[str(row.get("variant_id", ""))] = row

    delta_rows: list[dict[str, object]] = []
    for eval_tile_id in eval_tile_ids:
        for seed in seeds:
            by_variant = indexed.get((str(eval_tile_id), int(seed)), {})
            for compressed_variant, matched_control_variant in PHASE59_MATCHED_COMPARISONS:
                compressed_row = by_variant.get(compressed_variant)
                matched_row = by_variant.get(matched_control_variant)
                if compressed_row is None or matched_row is None:
                    continue
                compressed_reward = _float_value(compressed_row, "total_contract_reward")
                matched_reward = _float_value(matched_row, "total_contract_reward")
                delta = _round_float(compressed_reward - matched_reward)
                delta_rows.append(
                    {
                        "compressed_variant_id": compressed_variant,
                        "matched_control_variant_id": matched_control_variant,
                        "eval_tile_id": str(eval_tile_id),
                        "seed": int(seed),
                        "compressed_reward": _round_float(compressed_reward),
                        "matched_control_reward": _round_float(matched_reward),
                        "compressed_minus_matched_control_reward": delta,
                        "compressed_improves_matched_control": delta > 0.0,
                        "train_timesteps": _optional_int(
                            compressed_row,
                            "train_timesteps",
                            fallback=_optional_int(matched_row, "train_timesteps"),
                        ),
                        "eval_max_steps": _optional_int(
                            compressed_row,
                            "eval_max_steps",
                            fallback=_optional_int(matched_row, "eval_max_steps"),
                        ),
                        "claim_boundary": PHASE59_CLAIM_BOUNDARY,
                    }
                )
    return delta_rows


def _phase59_policy_summary(
    rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
) -> dict[str, object]:
    mean_reward_by_variant: dict[str, float] = {}
    for variant_id in PHASE59_REQUIRED_VARIANTS:
        values = [
            _float_value(row, "total_contract_reward")
            for row in rows
            if str(row.get("variant_id", "")) == str(variant_id)
        ]
        if values:
            mean_reward_by_variant[str(variant_id)] = _mean_or_none(values)
    return {
        "mean_reward_by_variant": mean_reward_by_variant,
        "matched_deltas": _matched_delta_summaries(delta_rows),
    }


def _matched_delta_summaries(
    delta_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in delta_rows:
        key = (
            str(row["compressed_variant_id"]),
            str(row["matched_control_variant_id"]),
        )
        grouped.setdefault(key, []).append(
            _float_value(row, "compressed_minus_matched_control_reward")
        )
    summaries: dict[str, dict[str, object]] = {}
    for (compressed, matched), values in sorted(grouped.items()):
        summaries[f"{compressed}_minus_{matched}"] = {
            "compressed_variant_id": compressed,
            "matched_control_variant_id": matched,
            **_simple_delta_summary(values),
        }
    return summaries


def _delta_summary(
    values: Sequence[float],
    bootstrap_iterations: int,
    random_seed: int,
) -> dict[str, object]:
    deltas = [float(value) for value in values]
    positive_count = sum(1 for value in deltas if value > 0.0)
    total_count = len(deltas)
    low, high = _bootstrap_mean_ci(
        deltas,
        iterations=int(bootstrap_iterations),
        random_seed=int(random_seed),
    )
    return {
        "mean_delta": _mean_or_none(deltas),
        "std_delta": _std_or_none(deltas),
        "positive_count": positive_count,
        "total_count": total_count,
        "positive_fraction": _round_float(positive_count / total_count)
        if total_count
        else None,
        "one_sided_sign_test_p": _one_sided_sign_test_p(positive_count, total_count),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
    }


def _simple_delta_summary(values: Sequence[float]) -> dict[str, object]:
    deltas = [float(value) for value in values]
    positive_count = sum(1 for value in deltas if value > 0.0)
    total_count = len(deltas)
    return {
        "mean_delta": _mean_or_none(deltas),
        "std_delta": _std_or_none(deltas),
        "positive_count": positive_count,
        "total_count": total_count,
        "positive_fraction": _round_float(positive_count / total_count)
        if total_count
        else None,
    }


def _bootstrap_mean_ci(
    values: Sequence[float],
    iterations: int,
    random_seed: int,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(int(random_seed))
    samples = []
    n = len(values)
    for _ in range(int(iterations)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(sample) / n)
    samples.sort()
    low_index = max(0, int(math.floor(0.025 * (len(samples) - 1))))
    high_index = min(len(samples) - 1, int(math.ceil(0.975 * (len(samples) - 1))))
    return _round_float(samples[low_index]), _round_float(samples[high_index])


def _cluster_rows(delta_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in delta_rows:
        key = (str(row.get("eval_tile_id", "")), _int_value(row, "seed"))
        grouped.setdefault(key, []).append(
            _float_value(row, "compressed_minus_matched_control_reward")
        )
    rows = []
    for eval_tile_id, seed in sorted(grouped):
        values = grouped[(eval_tile_id, seed)]
        mean_delta = _round_float(sum(values) / len(values))
        rows.append(
            {
                "eval_tile_id": eval_tile_id,
                "seed": seed,
                "cluster_delta_count": len(values),
                "mean_cluster_delta": mean_delta,
                "cluster_positive": mean_delta > 0.0,
                "claim_boundary": PHASE59_CLAIM_BOUNDARY,
            }
        )
    return rows


def _cluster_summary(cluster_rows: list[dict[str, object]]) -> dict[str, object]:
    values = [float(row["mean_cluster_delta"]) for row in cluster_rows]
    positive_count = sum(1 for value in values if value > 0.0)
    total_count = len(values)
    return {
        "cluster_count": total_count,
        "mean_cluster_delta": _round_float(sum(values) / total_count)
        if total_count
        else None,
        "positive_cluster_count": positive_count,
        "positive_cluster_fraction": _round_float(positive_count / total_count)
        if total_count
        else None,
        "one_sided_sign_test_p": _one_sided_sign_test_p(positive_count, total_count),
    }


def _signed_rank_summary(cluster_rows: list[dict[str, object]]) -> dict[str, object]:
    nonzero = [
        (abs(float(row["mean_cluster_delta"])), index, row)
        for index, row in enumerate(cluster_rows)
        if float(row["mean_cluster_delta"]) != 0.0
    ]
    ranks_by_index = {}
    for rank, (_, index, _) in enumerate(sorted(nonzero), start=1):
        ranks_by_index[index] = rank
    positive_rank_sum = 0.0
    total_rank_sum = 0.0
    ranks = []
    for index, row in enumerate(cluster_rows):
        rank = ranks_by_index.get(index)
        if rank is None:
            continue
        ranks.append(float(rank))
        total_rank_sum += float(rank)
        if float(row["mean_cluster_delta"]) > 0.0:
            positive_rank_sum += float(rank)
    p_value = _exact_signed_rank_p(ranks, positive_rank_sum) if ranks else None
    return {
        "cluster_count": len(ranks),
        "positive_rank_sum": _int_if_whole(positive_rank_sum),
        "total_rank_sum": _int_if_whole(total_rank_sum),
        "one_sided_signed_rank_p": p_value,
    }


def _phase59_status(
    matched_deltas: Mapping[str, object],
    pooled: Mapping[str, object],
    coverage_issues: Mapping[str, object],
    pooled_positive_threshold: float,
) -> str:
    if _has_coverage_issues(coverage_issues):
        return "insufficient"
    full_support = all(
        isinstance(matched_deltas.get(f"{compressed}_minus_{matched}"), Mapping)
        and float(matched_deltas[f"{compressed}_minus_{matched}"].get("mean_delta") or 0.0) > 0.0
        for compressed, matched in PHASE59_MATCHED_COMPARISONS
    )
    pooled_support = (
        float(pooled.get("mean_delta") or 0.0) > 0.0
        and float(pooled.get("positive_fraction") or 0.0) >= float(pooled_positive_threshold)
    )
    if full_support and pooled_support:
        return "matched_dimension_geofm_supported"
    for compressed in PHASE59_COMPRESSED_VARIANTS:
        controls = [
            matched
            for left, matched in PHASE59_MATCHED_COMPARISONS
            if left == compressed
        ]
        if controls and all(
            isinstance(matched_deltas.get(f"{compressed}_minus_{control}"), Mapping)
            and float(matched_deltas[f"{compressed}_minus_{control}"].get("mean_delta") or 0.0) > 0.0
            for control in controls
        ):
            return "matched_dimension_geofm_partial"
    return "matched_dimension_geofm_not_supported"


def _phase59_conclusion(status: str) -> str:
    if status == "matched_dimension_geofm_supported":
        return (
            "Phase 59 conclusion: D4P8/D4P16 outperform same-dimension random "
            "and shuffled controls, strengthening the compressed GeoFM route."
        )
    if status == "matched_dimension_geofm_partial":
        return (
            "Phase 59 conclusion: matched-dimension evidence is partial; only "
            "one compressed route clears both same-dimension controls."
        )
    if status == "matched_dimension_geofm_not_supported":
        return (
            "Phase 59 conclusion: compressed GeoFM does not outperform matched "
            "low-dimensional controls under this protocol."
        )
    return (
        "Phase 59 conclusion: insufficient comparable rows for a matched-dimension "
        "control decision."
    )


def _main_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    order: list[tuple[str, str, str]] = []
    for row in rows:
        key = (
            str(row.get("row_type", "")),
            str(row.get("variant_id", "")),
            str(row.get("eval_tile_id", "")),
        )
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(row)
    summary_rows = []
    for row_type, variant_id, eval_tile_id in order:
        group_rows = groups[(row_type, variant_id, eval_tile_id)]
        rewards = [_float_value(row, "total_contract_reward") for row in group_rows]
        seeds = {_int_value(row, "seed") for row in group_rows}
        summary_rows.append(
            {
                "row_type": row_type,
                "variant_id": variant_id,
                "eval_tile_id": eval_tile_id,
                "seed_count": len(seeds),
                "mean_total_contract_reward": _mean_or_none(rewards),
                "std_total_contract_reward": _std_or_none(rewards),
                "claim_boundary": PHASE59_CLAIM_BOUNDARY,
            }
        )
    return summary_rows


def _has_coverage_issues(coverage_issues: Mapping[str, object]) -> bool:
    return any(
        bool(coverage_issues.get(key))
        for key in (
            "missing_variant_rows",
            "unexpected_variant_rows",
            "duplicate_variant_rows",
        )
    )


def _variant_key_dicts(keys: set[tuple[str, int, str]]) -> list[dict[str, object]]:
    return [
        {"eval_tile_id": eval_tile_id, "seed": seed, "variant_id": variant_id}
        for eval_tile_id, seed, variant_id in sorted(
            keys,
            key=lambda item: (item[0], item[1], item[2]),
        )
    ]


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


def _optional_int(
    row: Mapping[str, object],
    field: str,
    fallback: int | None = None,
) -> int | None:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        return fallback
    return int(value)


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


def _one_sided_sign_test_p(positive_count: int, total_count: int) -> float | None:
    if total_count <= 0:
        return None
    tail = sum(math.comb(total_count, k) for k in range(positive_count, total_count + 1))
    return _round_float(tail / (2**total_count))


def _exact_signed_rank_p(ranks: Sequence[float], observed_positive_rank_sum: float) -> float:
    total = 0
    at_least_observed = 0
    for signs in product((0, 1), repeat=len(ranks)):
        rank_sum = sum(rank for rank, sign in zip(ranks, signs) if sign)
        total += 1
        if rank_sum >= observed_positive_rank_sum:
            at_least_observed += 1
    return _round_float(at_least_observed / total)


def _int_if_whole(value: float) -> int | float:
    return int(value) if float(value).is_integer() else _round_float(value)


def _round_float(value: float) -> float:
    return round(float(value), 10)



def write_phase59_matched_dimension_control_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "phase59_matched_dimension_control_summary.csv"
    delta_path = output_path / "phase59_matched_dimension_delta_table.csv"
    comparison_path = output_path / "phase59_matched_dimension_controls.json"
    readiness_path = output_path / "phase59_matched_dimension_controls.md"

    _write_summary_csv(
        summary_path,
        analysis.get("summaries", analysis.get("source_rows", [])),
    )
    _write_csv_mapping_rows(
        delta_path,
        PHASE59_DELTA_FIELDNAMES,
        analysis.get("delta_rows"),
        "delta_rows",
    )
    comparison = {
        key: value
        for key, value in dict(analysis).items()
        if key not in {"source_rows", "summaries", "traces"}
    }
    comparison_path.write_text(
        json.dumps(_json_ready(comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    readiness_path.write_text(_phase59_readiness_markdown(analysis), encoding="utf-8")
    return {
        "summary_csv": summary_path,
        "delta_csv": delta_path,
        "comparison_json": comparison_path,
        "readiness_md": readiness_path,
    }


def _phase59_readiness_markdown(analysis: Mapping[str, object]) -> str:
    learned = analysis.get("learned_policy")
    if not isinstance(learned, Mapping):
        learned = {}
    matched_deltas = learned.get("matched_deltas")
    if not isinstance(matched_deltas, Mapping):
        matched_deltas = {}
    pooled = analysis.get("pooled_matched_control_delta")
    if not isinstance(pooled, Mapping):
        pooled = {}
    cluster = analysis.get("cluster_summary")
    if not isinstance(cluster, Mapping):
        cluster = {}
    signed_rank = analysis.get("signed_rank_summary")
    if not isinstance(signed_rank, Mapping):
        signed_rank = {}

    lines = [
        "# Phase 59 Matched-Dimension Controls",
        "",
        f"Status: {analysis.get('phase59_matched_dimension_status', '')}",
        "",
        "Matched-dimension control audit conclusion:",
        str(analysis.get("conclusion", "")),
        "",
        "Matched comparison deltas:",
    ]
    for key in sorted(matched_deltas):
        value = matched_deltas[key]
        if not isinstance(value, Mapping):
            continue
        lines.append(
            "- "
            f"{key}: mean={value.get('mean_delta')}, "
            f"positive={value.get('positive_count')} / {value.get('total_count')}"
        )
    lines.extend(
        [
            "",
            "Pooled matched-control delta:",
            "- "
            f"mean={pooled.get('mean_delta')}, "
            f"positive={pooled.get('positive_count')} / {pooled.get('total_count')}, "
            f"bootstrap CI95=[{pooled.get('bootstrap_ci95_low')}, "
            f"{pooled.get('bootstrap_ci95_high')}]",
            "",
            "Cluster summary:",
            "- "
            f"mean={cluster.get('mean_cluster_delta')}, "
            f"positive={cluster.get('positive_cluster_count')} / "
            f"{cluster.get('cluster_count')}, "
            f"sign-test p={cluster.get('one_sided_sign_test_p')}",
            "",
            "Signed-rank summary:",
            "- "
            f"positive rank sum={signed_rank.get('positive_rank_sum')}, "
            f"total rank sum={signed_rank.get('total_rank_sum')}, "
            f"p={signed_rank.get('one_sided_signed_rank_p')}",
            "",
            "Claim boundary:",
            str(analysis.get("claim_boundary", PHASE59_CLAIM_BOUNDARY)),
            "",
        ]
    )
    return "\n".join(lines)


def _write_summary_csv(path: Path, rows: object) -> None:
    if not isinstance(rows, list):
        raise ValueError("Phase 59 analysis is missing summary rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Phase 59 summary rows must be objects")
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDNAMES})


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    row_key: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 59 analysis is missing {row_key}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 59 {row_key} rows must be objects")
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
