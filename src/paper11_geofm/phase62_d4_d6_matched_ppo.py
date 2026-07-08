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

from .padded_heldout_policy import (
    SUMMARY_FIELDNAMES,
    Phase25PaddedTileEnv,
    _dependency_metadata,
    _evaluate_baseline_policy,
    _evaluate_trained_policy,
    _normalize_seeds,
    _select_train_eval_tiles,
    _store_trace,
)
from .phase28_representation_controls import _train_maskable_ppo_model
from .tiled_inputs import load_tiled_variant_input


PHASE62_CLAIM_BOUNDARY = (
    "Phase 62 is a base-reward learned-policy comparison between D4 PCA "
    "compressed states and D6 raw-B1 random projection controls. It does not "
    "enable suitability reward, does not test B2/B3, does not test transfer, "
    "does not prove independent agronomic suitability, and does not by itself "
    "justify final submission-level performance claims."
)

PHASE62_PRIMARY_VARIANTS = ("D4P8", "D4P16", "D6R8", "D6R16")
PHASE62_OPTIONAL_VARIANTS = ("D6P8", "D6P16")
PHASE62_ALLOWED_VARIANTS = PHASE62_PRIMARY_VARIANTS + PHASE62_OPTIONAL_VARIANTS
PHASE62_PRIMARY_COMPARISONS = (("D4P8", "D6R8"), ("D4P16", "D6R16"))
PHASE62_OPTIONAL_COMPARISONS = (("D4P8", "D6P8"), ("D4P16", "D6P16"))
PHASE62_ALL_COMPARISONS = PHASE62_PRIMARY_COMPARISONS + PHASE62_OPTIONAL_COMPARISONS


PHASE62_DELTA_FIELDNAMES = [
    "d4_variant_id",
    "d6_variant_id",
    "comparison_role",
    "eval_tile_id",
    "seed",
    "d4_reward",
    "d6_reward",
    "d4_minus_d6_reward",
    "d4_improves_d6",
    "train_timesteps",
    "eval_max_steps",
    "claim_boundary",
]

PHASE62_CLUSTER_FIELDNAMES = [
    "eval_tile_id",
    "seed",
    "cluster_delta_count",
    "mean_cluster_delta",
    "cluster_positive",
    "claim_boundary",
]


def build_phase62_d4_d6_contract(
    phase8_output_dir: Path | str,
    phase61_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] | str = PHASE62_PRIMARY_VARIANTS,
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 5,
    total_timesteps: int = 4096,
    eval_max_steps: int = 8,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
) -> dict[str, object]:
    if int(total_timesteps) <= 0:
        raise ValueError("total_timesteps must be positive")
    if int(eval_max_steps) <= 0:
        raise ValueError("eval_max_steps must be positive")

    normalized_seeds = _normalize_seeds(seeds)
    selected = _select_train_eval_tiles(
        Path(tile_index_csv),
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
    )
    eval_ids = list(selected["eval_tile_ids"])
    selected_counts = dict(selected["selected_tile_block_counts"])
    max_blocks = max(int(selected_counts[tile_id]) for tile_id in selected_counts)
    train_id = str(selected["train_tile_id"])
    normalized_variants = _normalize_phase62_variants(variants)
    return {
        "phase": "phase62_d4_d6_matched_ppo_evaluation",
        "phase8_output_dir": str(Path(phase8_output_dir)),
        "phase61_output_dir": str(Path(phase61_output_dir)),
        "tile_index_csv": str(Path(tile_index_csv)),
        "variants": normalized_variants,
        "variant_source_dirs": _phase62_variant_source_dirs(
            normalized_variants,
            phase8_output_dir=phase8_output_dir,
            phase61_output_dir=phase61_output_dir,
        ),
        "train_tile_id": train_id,
        "train_tile_ids": [train_id],
        "eval_tile_ids": eval_ids,
        "eval_tile_count": len(eval_ids),
        "eval_tile_ranks": {
            str(tile_id): rank for rank, tile_id in enumerate(eval_ids, start=1)
        },
        "selected_tile_block_counts": selected_counts,
        "train_tile_selection": selected["train_tile_selection"],
        "eval_tile_selection": selected["eval_tile_selection"],
        "max_blocks": int(max_blocks),
        "total_timesteps": int(total_timesteps),
        "eval_max_steps": int(eval_max_steps),
        "seeds": normalized_seeds,
        "seed_count": len(normalized_seeds),
        "seed_ranks": {
            str(seed): rank for rank, seed in enumerate(normalized_seeds, start=1)
        },
        "claim_boundary": PHASE62_CLAIM_BOUNDARY,
    }


def build_phase62_d4_d6_analysis(
    summary_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    metadata: Mapping[str, object] | None = None,
    bootstrap_iterations: int = 5000,
    random_seed: int = 62,
    pooled_positive_threshold: float = 0.5,
) -> dict[str, object]:
    if int(bootstrap_iterations) <= 0:
        raise ValueError("bootstrap_iterations must be positive")

    rows = _load_summary_rows(summary_rows_or_csv)
    trained_rows = [
        row for row in rows if str(row.get("row_type", "")) == "trained_policy"
    ]
    comparable_rows = [
        row
        for row in trained_rows
        if str(row.get("variant_id", "")) in set(PHASE62_ALLOWED_VARIANTS)
    ]
    ignored_rows = [
        row
        for row in trained_rows
        if str(row.get("variant_id", "")) not in set(PHASE62_ALLOWED_VARIANTS)
    ]
    metadata_map = {} if metadata is None else dict(metadata)
    eval_tile_ids = _metadata_string_list(
        metadata_map,
        "eval_tile_ids",
        fallback=_unique_strings(comparable_rows, "eval_tile_id"),
    )
    seeds = _metadata_int_list(
        metadata_map,
        "seeds",
        fallback=_unique_ints(comparable_rows, "seed"),
    )
    coverage_issues = _coverage_issues(
        comparable_rows,
        variants=PHASE62_PRIMARY_VARIANTS,
        eval_tile_ids=eval_tile_ids,
        seeds=seeds,
    )
    delta_rows = _d4_d6_delta_rows(comparable_rows, eval_tile_ids, seeds)
    primary_delta_rows = [
        row for row in delta_rows if row["comparison_role"] == "primary"
    ]
    matched_deltas = _matched_delta_summaries(delta_rows)
    pooled_primary = _delta_summary(
        [float(row["d4_minus_d6_reward"]) for row in primary_delta_rows],
        bootstrap_iterations=int(bootstrap_iterations),
        random_seed=int(random_seed),
    )
    cluster_rows = _cluster_rows(primary_delta_rows)
    cluster_summary = _cluster_summary(cluster_rows)
    signed_rank_summary = _signed_rank_summary(cluster_rows)
    status = _phase62_status(
        matched_deltas,
        pooled_primary,
        coverage_issues,
        pooled_positive_threshold=float(pooled_positive_threshold),
    )
    return {
        "phase": "phase62_d4_d6_matched_ppo_analysis",
        "variants": list(PHASE62_PRIMARY_VARIANTS),
        "optional_variants": list(PHASE62_OPTIONAL_VARIANTS),
        "primary_comparisons": [
            {"d4_variant_id": left, "d6_variant_id": right}
            for left, right in PHASE62_PRIMARY_COMPARISONS
        ],
        "optional_comparisons": [
            {"d4_variant_id": left, "d6_variant_id": right}
            for left, right in PHASE62_OPTIONAL_COMPARISONS
        ],
        "eval_tile_ids": eval_tile_ids,
        "seeds": seeds,
        "source_rows": rows,
        "main_summary_rows": _main_summary_rows(comparable_rows),
        "ignored_historical_variant_row_count": len(ignored_rows),
        "delta_rows": delta_rows,
        "primary_delta_rows": primary_delta_rows,
        "mean_reward_by_variant": _mean_reward_by_variant(comparable_rows),
        "matched_deltas": matched_deltas,
        "pooled_primary_delta": pooled_primary,
        "cluster_rows": cluster_rows,
        "cluster_summary": cluster_summary,
        "signed_rank_summary": signed_rank_summary,
        "coverage_issues": coverage_issues,
        "phase62_d4_d6_status": status,
        "conclusion": _phase62_conclusion(status),
        "claim_boundary": PHASE62_CLAIM_BOUNDARY,
    }


def _load_summary_rows(
    summary_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(summary_rows_or_csv, (str, PathLike)):
        path = Path(summary_rows_or_csv)
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 62 summary CSV: {path}")
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
            key[2] in expected_variants
            and (key[0] not in expected_tiles or key[1] not in expected_seeds)
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


def _d4_d6_delta_rows(
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
            for d4_variant, d6_variant in PHASE62_ALL_COMPARISONS:
                d4_row = by_variant.get(d4_variant)
                d6_row = by_variant.get(d6_variant)
                if d4_row is None or d6_row is None:
                    continue
                d4_reward = _float_value(d4_row, "total_contract_reward")
                d6_reward = _float_value(d6_row, "total_contract_reward")
                delta = _round_float(d4_reward - d6_reward)
                delta_rows.append(
                    {
                        "d4_variant_id": d4_variant,
                        "d6_variant_id": d6_variant,
                        "comparison_role": (
                            "primary"
                            if (d4_variant, d6_variant)
                            in PHASE62_PRIMARY_COMPARISONS
                            else "optional_lineage"
                        ),
                        "eval_tile_id": str(eval_tile_id),
                        "seed": int(seed),
                        "d4_reward": _round_float(d4_reward),
                        "d6_reward": _round_float(d6_reward),
                        "d4_minus_d6_reward": delta,
                        "d4_improves_d6": delta > 0.0,
                        "train_timesteps": _optional_int(
                            d4_row,
                            "train_timesteps",
                            fallback=_optional_int(d6_row, "train_timesteps"),
                        ),
                        "eval_max_steps": _optional_int(
                            d4_row,
                            "eval_max_steps",
                            fallback=_optional_int(d6_row, "eval_max_steps"),
                        ),
                        "claim_boundary": PHASE62_CLAIM_BOUNDARY,
                    }
                )
    return delta_rows


def _matched_delta_summaries(
    delta_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    roles: dict[tuple[str, str], str] = {}
    for row in delta_rows:
        key = (str(row["d4_variant_id"]), str(row["d6_variant_id"]))
        grouped.setdefault(key, []).append(_float_value(row, "d4_minus_d6_reward"))
        roles[key] = str(row["comparison_role"])
    summaries: dict[str, dict[str, object]] = {}
    for (d4_variant, d6_variant), values in sorted(grouped.items()):
        summaries[f"{d4_variant}_minus_{d6_variant}"] = {
            "d4_variant_id": d4_variant,
            "d6_variant_id": d6_variant,
            "comparison_role": roles[(d4_variant, d6_variant)],
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
        grouped.setdefault(key, []).append(_float_value(row, "d4_minus_d6_reward"))
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
                "claim_boundary": PHASE62_CLAIM_BOUNDARY,
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


def _phase62_status(
    matched_deltas: Mapping[str, object],
    pooled: Mapping[str, object],
    coverage_issues: Mapping[str, object],
    pooled_positive_threshold: float,
) -> str:
    if _has_coverage_issues(coverage_issues):
        return "insufficient"
    primary_summaries = [
        matched_deltas.get(f"{d4_variant}_minus_{d6_variant}")
        for d4_variant, d6_variant in PHASE62_PRIMARY_COMPARISONS
    ]
    if not all(isinstance(item, Mapping) for item in primary_summaries):
        return "insufficient"
    primary_means = [
        float(item.get("mean_delta") or 0.0)
        for item in primary_summaries
        if isinstance(item, Mapping)
    ]
    pooled_mean = float(pooled.get("mean_delta") or 0.0)
    pooled_positive_fraction = float(pooled.get("positive_fraction") or 0.0)
    if (
        all(value > 0.0 for value in primary_means)
        and pooled_mean > 0.0
        and pooled_positive_fraction >= float(pooled_positive_threshold)
    ):
        return "d4_pca_advantage_over_d6_supported"
    if all(value < 0.0 for value in primary_means) and pooled_mean < 0.0:
        return "d6_random_projection_advantage"
    return "d4_d6_not_distinguishable"


def _phase62_conclusion(status: str) -> str:
    if status == "d4_pca_advantage_over_d6_supported":
        return (
            "Phase 62 conclusion: D4P8/D4P16 outperform D6R8/D6R16 under "
            "the bounded base-reward held-out PPO protocol."
        )
    if status == "d6_random_projection_advantage":
        return (
            "Phase 62 conclusion: D6 raw-B1 random projection controls "
            "outperform D4 PCA compressed states under this protocol."
        )
    if status == "d4_d6_not_distinguishable":
        return (
            "Phase 62 conclusion: complete matched rows do not distinguish D4 "
            "PCA states from D6 raw-B1 random projection controls."
        )
    return "Phase 62 conclusion: insufficient rows for a D4/D6 decision."


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
                "claim_boundary": PHASE62_CLAIM_BOUNDARY,
            }
        )
    return summary_rows


def _mean_reward_by_variant(rows: list[dict[str, object]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for variant_id in PHASE62_ALLOWED_VARIANTS:
        values = [
            _float_value(row, "total_contract_reward")
            for row in rows
            if str(row.get("variant_id", "")) == str(variant_id)
        ]
        if values:
            result[str(variant_id)] = _mean_or_none(values)
    return result


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


def _exact_signed_rank_p(
    ranks: Sequence[float],
    observed_positive_rank_sum: float,
) -> float:
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


def _normalize_phase62_variants(variants: Sequence[str] | str) -> list[str]:
    if isinstance(variants, str):
        values = [part.strip() for part in variants.split(",")]
    else:
        values = [str(item).strip() for item in variants]
    normalized = [value.upper() for value in values if value]
    if not normalized:
        raise ValueError("At least one Phase 62 variant must be requested")
    allowed = set(PHASE62_ALLOWED_VARIANTS)
    unsupported = [variant for variant in normalized if variant not in allowed]
    if unsupported:
        raise ValueError(f"unsupported Phase 62 variants: {unsupported}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Phase 62 variants must be unique")
    return normalized


def _phase62_variant_source_dirs(
    variants: Sequence[str],
    phase8_output_dir: Path | str,
    phase61_output_dir: Path | str,
) -> dict[str, str]:
    source_dirs: dict[str, str] = {}
    for variant_id in variants:
        if str(variant_id).startswith("D4"):
            source_dirs[str(variant_id)] = str(Path(phase8_output_dir))
        else:
            source_dirs[str(variant_id)] = str(Path(phase61_output_dir))
    return source_dirs

def write_phase62_d4_d6_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "phase62_d4_d6_matched_ppo_summary.csv"
    traces_path = output_path / "phase62_d4_d6_matched_ppo_traces.json"
    delta_path = output_path / "phase62_d4_d6_delta_table.csv"
    cluster_path = output_path / "phase62_d4_d6_cluster_summary.csv"
    comparison_path = output_path / "phase62_d4_d6_matched_ppo.json"
    readiness_path = output_path / "phase62_d4_d6_matched_ppo.md"

    _write_summary_csv(
        summary_path,
        analysis.get("summaries", analysis.get("source_rows", [])),
    )
    traces_path.write_text(
        json.dumps(_json_ready(analysis.get("traces", {})), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv_mapping_rows(
        delta_path,
        PHASE62_DELTA_FIELDNAMES,
        analysis.get("delta_rows"),
        "delta_rows",
    )
    _write_csv_mapping_rows(
        cluster_path,
        PHASE62_CLUSTER_FIELDNAMES,
        analysis.get("cluster_rows"),
        "cluster_rows",
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
    readiness_path.write_text(_phase62_readiness_markdown(analysis), encoding="utf-8")
    return {
        "summary_csv": summary_path,
        "traces_json": traces_path,
        "delta_csv": delta_path,
        "cluster_csv": cluster_path,
        "comparison_json": comparison_path,
        "readiness_md": readiness_path,
    }


def run_phase62_d4_d6_evaluation(
    phase8_output_dir: Path | str,
    phase61_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] | str = PHASE62_PRIMARY_VARIANTS,
    existing_summary_csv: Path | str | None = None,
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 5,
    total_timesteps: int = 4096,
    eval_max_steps: int = 8,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
    bootstrap_iterations: int = 5000,
    random_seed: int = 62,
) -> dict[str, object]:
    contract = build_phase62_d4_d6_contract(
        phase8_output_dir=phase8_output_dir,
        phase61_output_dir=phase61_output_dir,
        tile_index_csv=tile_index_csv,
        variants=variants,
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
        total_timesteps=total_timesteps,
        eval_max_steps=eval_max_steps,
        seeds=seeds,
    )
    summaries: list[dict[str, object]] = []
    traces: dict[str, dict[str, dict[str, dict[str, list[dict[str, object]]]]]] = {
        "trained_policy": {},
        "first_valid": {},
        "seeded_random": {},
    }
    train_n_blocks = int(
        contract["selected_tile_block_counts"][str(contract["train_tile_id"])]
    )
    for variant_id in contract["variants"]:
        for seed in contract["seeds"]:
            train_tiled = _load_phase62_tiled_variant_input(
                contract,
                str(contract["train_tile_id"]),
                str(variant_id),
            )
            train_env = Phase25PaddedTileEnv(
                train_tiled,
                max_blocks=int(contract["max_blocks"]),
                max_steps=int(contract["total_timesteps"]),
            )
            train_env.reset(seed=int(seed))
            model = _train_maskable_ppo_model(
                train_env,
                seed=int(seed),
                total_timesteps=int(contract["total_timesteps"]),
            )
            for eval_tile_id in contract["eval_tile_ids"]:
                eval_tiled = _load_phase62_tiled_variant_input(
                    contract,
                    str(eval_tile_id),
                    str(variant_id),
                )
                eval_tile_rank = int(contract["eval_tile_ranks"][str(eval_tile_id)])
                seed_rank = int(contract["seed_ranks"][str(int(seed))])
                trained_summary, trained_steps = _evaluate_trained_policy(
                    model,
                    eval_tiled,
                    train_tile_id=str(contract["train_tile_id"]),
                    train_n_blocks=train_n_blocks,
                    max_blocks=int(contract["max_blocks"]),
                    eval_tile_rank=eval_tile_rank,
                    phase25_seed_rank=seed_rank,
                    eval_max_steps=int(contract["eval_max_steps"]),
                    train_timesteps=int(contract["total_timesteps"]),
                    seed=int(seed),
                )
                trained_summary["claim_boundary"] = PHASE62_CLAIM_BOUNDARY
                summaries.append(trained_summary)
                _store_trace(
                    traces,
                    "trained_policy",
                    str(variant_id),
                    str(eval_tile_id),
                    int(seed),
                    trained_steps,
                )
                for policy_id in ("first_valid", "seeded_random"):
                    baseline_summary, baseline_steps = _evaluate_baseline_policy(
                        eval_tiled,
                        policy_id=policy_id,
                        train_tile_id=str(contract["train_tile_id"]),
                        train_n_blocks=train_n_blocks,
                        max_blocks=int(contract["max_blocks"]),
                        eval_tile_rank=eval_tile_rank,
                        phase25_seed_rank=seed_rank,
                        eval_max_steps=int(contract["eval_max_steps"]),
                        train_timesteps=int(contract["total_timesteps"]),
                        seed=int(seed),
                    )
                    baseline_summary["claim_boundary"] = PHASE62_CLAIM_BOUNDARY
                    summaries.append(baseline_summary)
                    _store_trace(
                        traces,
                        policy_id,
                        str(variant_id),
                        str(eval_tile_id),
                        int(seed),
                        baseline_steps,
                    )
    analysis_rows = list(summaries)
    if existing_summary_csv is not None:
        analysis_rows = _load_summary_rows(existing_summary_csv) + analysis_rows
    analysis = build_phase62_d4_d6_analysis(
        analysis_rows,
        metadata={
            "eval_tile_ids": contract["eval_tile_ids"],
            "seeds": contract["seeds"],
        },
        bootstrap_iterations=int(bootstrap_iterations),
        random_seed=int(random_seed),
    )
    analysis["contract"] = contract
    analysis["summaries"] = analysis_rows
    analysis["new_summaries"] = summaries
    analysis["traces"] = traces
    analysis["dependencies"] = _dependency_metadata()
    return analysis


def _load_phase62_tiled_variant_input(
    contract: Mapping[str, object],
    tile_id: str,
    variant_id: str,
):
    variant_source_dirs = contract.get("variant_source_dirs")
    if not isinstance(variant_source_dirs, Mapping):
        raise ValueError("Phase 62 contract is missing variant source routing")
    source_dir = variant_source_dirs.get(variant_id)
    if source_dir is None:
        raise ValueError(f"Phase 62 contract has no source for variant {variant_id}")
    return load_tiled_variant_input(
        source_dir,
        str(contract["tile_index_csv"]),
        tile_id,
        variant_id=variant_id,
    )


def _phase62_readiness_markdown(analysis: Mapping[str, object]) -> str:
    matched_deltas = analysis.get("matched_deltas")
    if not isinstance(matched_deltas, Mapping):
        matched_deltas = {}
    pooled = analysis.get("pooled_primary_delta")
    if not isinstance(pooled, Mapping):
        pooled = {}
    cluster = analysis.get("cluster_summary")
    if not isinstance(cluster, Mapping):
        cluster = {}
    signed_rank = analysis.get("signed_rank_summary")
    if not isinstance(signed_rank, Mapping):
        signed_rank = {}

    lines = [
        "# Phase 62 D4/D6 Matched PPO Evaluation",
        "",
        f"Status: {analysis.get('phase62_d4_d6_status', '')}",
        "",
        "D4/D6 matched PPO evaluation conclusion:",
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
            f"{key}: role={value.get('comparison_role')}, "
            f"mean={value.get('mean_delta')}, "
            f"positive={value.get('positive_count')} / {value.get('total_count')}"
        )
    lines.extend(
        [
            "",
            "Pooled primary D4-D6R delta:",
            "- "
            f"mean={pooled.get('mean_delta')}, "
            f"positive={pooled.get('positive_count')} / {pooled.get('total_count')}, "
            f"bootstrap CI95=[{pooled.get('bootstrap_ci95_low')}, "
            f"{pooled.get('bootstrap_ci95_high')}], "
            f"sign-test p={pooled.get('one_sided_sign_test_p')}",
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
            str(analysis.get("claim_boundary", PHASE62_CLAIM_BOUNDARY)),
            "",
        ]
    )
    return "\n".join(lines)


def _write_summary_csv(path: Path, rows: object) -> None:
    if not isinstance(rows, list):
        raise ValueError("Phase 62 analysis is missing summary rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Phase 62 summary rows must be objects")
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDNAMES})


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    row_key: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 62 analysis is missing {row_key}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 62 {row_key} rows must be objects")
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
