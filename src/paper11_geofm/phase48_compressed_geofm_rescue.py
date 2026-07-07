from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
import json
import statistics
from os import PathLike
from pathlib import Path

from .padded_heldout_policy import SUMMARY_FIELDNAMES


PHASE48_CLAIM_BOUNDARY = (
    "Phase 48 is a read-only compressed GeoFM rescue audit over existing "
    "base-reward held-out Bishan summary rows. It tests D4P8/D4P16 as "
    "compressed GeoFM candidate routes against B0, raw B1, random D2, and "
    "shuffled D3 under the same summary protocol; it does not enable "
    "suitability reward, does not test B2/B3, does not test cross-region "
    "transfer, and does not support submission-level planning-performance "
    "claims."
)

PHASE48_COMPRESSED_VARIANTS = ("D4P8", "D4P16")
PHASE48_COMPARATORS = ("B0", "B1", "D2", "D3")
PHASE48_REQUIRED_VARIANTS = (*PHASE48_COMPARATORS, *PHASE48_COMPRESSED_VARIANTS)

PHASE48_DELTA_FIELDNAMES = [
    "compressed_variant_id",
    "comparator_variant_id",
    "eval_tile_id",
    "seed",
    "compressed_reward",
    "comparator_reward",
    "compressed_minus_comparator_reward",
    "compressed_improves_comparator",
    "train_timesteps",
    "eval_max_steps",
    "claim_boundary",
]


def build_phase48_compressed_geofm_rescue_analysis(
    summary_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    metadata: Mapping[str, object] | None = None,
    mean_delta_tolerance: float = 1e-9,
    pooled_positive_threshold: float = 0.5,
) -> dict[str, object]:
    rows = _load_summary_rows(summary_rows_or_csv)
    trained_rows = [row for row in rows if str(row.get("row_type", "")) == "trained_policy"]
    metadata_map = {} if metadata is None else dict(metadata)

    variants = _metadata_string_list(
        metadata_map,
        "variants",
        fallback=_unique_strings(trained_rows, "variant_id"),
    )
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
    train_timesteps = _metadata_int(
        metadata_map,
        ("train_timesteps", "total_timesteps"),
        trained_rows,
        "train_timesteps",
    )
    eval_max_steps = _metadata_int(
        metadata_map,
        ("eval_max_steps",),
        trained_rows,
        "eval_max_steps",
    )

    expected_variants = _expected_variants(variants)
    coverage_issues = _coverage_issues(
        trained_rows,
        variants=expected_variants,
        eval_tile_ids=eval_tile_ids,
        seeds=seeds,
    )
    delta_rows = _compressed_delta_rows(
        trained_rows,
        compressed_variants=PHASE48_COMPRESSED_VARIANTS,
        comparators=PHASE48_COMPARATORS,
        eval_tile_ids=eval_tile_ids,
        seeds=seeds,
    )
    learned_policy = _policy_summary(
        trained_rows,
        variants=variants,
        delta_rows=delta_rows,
    )
    pooled_delta = _pooled_delta_summary(delta_rows)
    status = _phase48_status(
        learned_policy,
        pooled_delta,
        coverage_issues,
        mean_delta_tolerance=float(mean_delta_tolerance),
        pooled_positive_threshold=float(pooled_positive_threshold),
    )

    analysis: dict[str, object] = {
        "phase": "phase48_compressed_geofm_rescue_analysis",
        "variants": variants,
        "compressed_variants": list(PHASE48_COMPRESSED_VARIANTS),
        "comparators": list(PHASE48_COMPARATORS),
        "eval_tile_ids": eval_tile_ids,
        "seeds": seeds,
        "train_timesteps": train_timesteps,
        "eval_max_steps": eval_max_steps,
        "source_rows": rows,
        "main_summary_rows": _main_summary_rows(rows),
        "delta_rows": delta_rows,
        "learned_policy": learned_policy,
        "pooled_compressed_control_delta": pooled_delta,
        "coverage_issues": coverage_issues,
        "phase48_compressed_geofm_status": status,
        "conclusion": _phase48_conclusion(status),
        "mean_delta_tolerance": float(mean_delta_tolerance),
        "pooled_positive_threshold": float(pooled_positive_threshold),
        "claim_boundary": PHASE48_CLAIM_BOUNDARY,
    }
    if metadata is not None:
        analysis["metadata"] = metadata_map
    return analysis


def write_phase48_compressed_geofm_rescue_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / "phase48_compressed_geofm_rescue_summary.csv"
    comparison_path = output_path / "phase48_compressed_geofm_rescue_comparison.json"
    delta_path = output_path / "phase48_compressed_geofm_rescue_delta_table.csv"
    readiness_path = output_path / "phase48_compressed_geofm_rescue_readiness.md"

    _write_summary_csv(
        summary_path,
        analysis.get("summaries", analysis.get("source_rows", [])),
    )
    comparison = {
        key: value
        for key, value in dict(analysis).items()
        if key not in {"summaries", "source_rows"}
    }
    comparison_path.write_text(
        json.dumps(_json_ready(comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv_mapping_rows(
        delta_path,
        PHASE48_DELTA_FIELDNAMES,
        analysis.get("delta_rows"),
        "delta_rows",
    )
    readiness_path.write_text(
        _phase48_readiness_markdown(analysis),
        encoding="utf-8",
    )
    return {
        "summary_csv": summary_path,
        "comparison_json": comparison_path,
        "delta_csv": delta_path,
        "readiness_md": readiness_path,
    }


def _load_summary_rows(
    summary_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(summary_rows_or_csv, (str, PathLike)):
        path = Path(summary_rows_or_csv)
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 48 summary CSV: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    rows: list[dict[str, object]] = []
    for row in summary_rows_or_csv:
        if not isinstance(row, Mapping):
            raise ValueError("Phase 48 summary rows must be objects")
        rows.append(dict(row))
    return rows


def _expected_variants(variants: Sequence[str]) -> list[str]:
    observed = {str(variant) for variant in variants}
    ordered = [variant for variant in PHASE48_REQUIRED_VARIANTS if variant in observed]
    for variant in PHASE48_REQUIRED_VARIANTS:
        if variant not in ordered:
            ordered.append(variant)
    return ordered


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
    observed_keys: set[tuple[str, int, str]] = set()
    duplicate_keys: set[tuple[str, int, str]] = set()
    unexpected_keys: set[tuple[str, int, str]] = set()
    expected_variants = {str(item) for item in variants}
    expected_tiles = {str(item) for item in eval_tile_ids}
    expected_seeds = {int(item) for item in seeds}
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


def _compressed_delta_rows(
    rows: list[dict[str, object]],
    compressed_variants: Sequence[str],
    comparators: Sequence[str],
    eval_tile_ids: Sequence[str],
    seeds: Sequence[int],
) -> list[dict[str, object]]:
    indexed: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (str(row.get("eval_tile_id", "")), _int_value(row, "seed"))
        indexed.setdefault(key, {})
        indexed[key][str(row.get("variant_id", ""))] = row

    delta_rows: list[dict[str, object]] = []
    for eval_tile_id in eval_tile_ids:
        for seed in seeds:
            by_variant = indexed.get((str(eval_tile_id), int(seed)), {})
            for compressed_variant in compressed_variants:
                compressed_row = by_variant.get(str(compressed_variant))
                if compressed_row is None:
                    continue
                compressed_reward = _float_value(
                    compressed_row,
                    "total_contract_reward",
                )
                for comparator in comparators:
                    comparator_row = by_variant.get(str(comparator))
                    if comparator_row is None:
                        continue
                    comparator_reward = _float_value(
                        comparator_row,
                        "total_contract_reward",
                    )
                    delta = _round_float(compressed_reward - comparator_reward)
                    delta_rows.append(
                        {
                            "compressed_variant_id": str(compressed_variant),
                            "comparator_variant_id": str(comparator),
                            "eval_tile_id": str(eval_tile_id),
                            "seed": int(seed),
                            "compressed_reward": _round_float(compressed_reward),
                            "comparator_reward": _round_float(comparator_reward),
                            "compressed_minus_comparator_reward": delta,
                            "compressed_improves_comparator": delta > 0.0,
                            "train_timesteps": _optional_int(
                                compressed_row,
                                "train_timesteps",
                                fallback=_optional_int(comparator_row, "train_timesteps"),
                            ),
                            "eval_max_steps": _optional_int(
                                compressed_row,
                                "eval_max_steps",
                                fallback=_optional_int(comparator_row, "eval_max_steps"),
                            ),
                            "claim_boundary": PHASE48_CLAIM_BOUNDARY,
                        }
                    )
    return delta_rows


def _policy_summary(
    rows: list[dict[str, object]],
    variants: Sequence[str],
    delta_rows: list[dict[str, object]],
) -> dict[str, object]:
    mean_reward_by_variant: dict[str, float] = {}
    for variant_id in variants:
        values = [
            _float_value(row, "total_contract_reward")
            for row in rows
            if str(row.get("variant_id", "")) == str(variant_id)
        ]
        if values:
            mean_reward_by_variant[str(variant_id)] = _round_float(
                sum(values) / len(values)
            )
    return {
        "mean_reward_by_variant": mean_reward_by_variant,
        "compressed_deltas": _delta_summaries(delta_rows),
    }


def _delta_summaries(
    delta_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in delta_rows:
        key = (
            str(row["compressed_variant_id"]),
            str(row["comparator_variant_id"]),
        )
        grouped.setdefault(key, []).append(row)

    summaries: dict[str, dict[str, object]] = {}
    for (compressed_variant, comparator), rows in sorted(grouped.items()):
        summaries[f"{compressed_variant}_minus_{comparator}"] = _summarize_delta_rows(
            rows
        )
    return summaries


def _pooled_delta_summary(
    delta_rows: list[dict[str, object]],
) -> dict[str, object]:
    return _summarize_delta_rows(delta_rows)


def _summarize_delta_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    deltas = [_float_value(row, "compressed_minus_comparator_reward") for row in rows]
    positive_count = sum(
        1 for row in rows if bool(row.get("compressed_improves_comparator"))
    )
    total_count = len(deltas)
    return {
        "mean_reward_delta": _mean_or_none(deltas),
        "std_reward_delta": _std_or_none(deltas),
        "positive_tile_seed_count": positive_count,
        "total_tile_seed_count": total_count,
        "positive_fraction": _round_float(positive_count / total_count)
        if total_count
        else None,
    }


def _phase48_status(
    learned_policy: Mapping[str, object],
    pooled_delta: Mapping[str, object],
    coverage_issues: Mapping[str, object],
    mean_delta_tolerance: float,
    pooled_positive_threshold: float,
) -> str:
    if _has_coverage_issues(coverage_issues):
        return "insufficient"

    compressed_deltas = learned_policy.get("compressed_deltas")
    if not isinstance(compressed_deltas, Mapping):
        return "insufficient"

    if _all_compressed_means_positive(
        compressed_deltas,
        mean_delta_tolerance,
    ) and _pooled_supports_route(
        pooled_delta,
        mean_delta_tolerance,
        pooled_positive_threshold,
    ):
        return "compressed_geofm_route_supported"

    if _any_compressed_variant_recovers_raw_and_b0(
        compressed_deltas,
        mean_delta_tolerance,
    ):
        return "compressed_geofm_route_partial"

    return "compressed_geofm_route_not_supported"


def _all_compressed_means_positive(
    compressed_deltas: Mapping[str, object],
    tolerance: float,
) -> bool:
    for compressed_variant in PHASE48_COMPRESSED_VARIANTS:
        for comparator in PHASE48_COMPARATORS:
            summary = compressed_deltas.get(f"{compressed_variant}_minus_{comparator}")
            if not isinstance(summary, Mapping):
                return False
            if _mean_delta(summary) <= tolerance:
                return False
    return True


def _pooled_supports_route(
    pooled_delta: Mapping[str, object],
    tolerance: float,
    positive_threshold: float,
) -> bool:
    return (
        _mean_delta(pooled_delta) > tolerance
        and _positive_fraction(pooled_delta) >= positive_threshold
    )


def _any_compressed_variant_recovers_raw_and_b0(
    compressed_deltas: Mapping[str, object],
    tolerance: float,
) -> bool:
    for compressed_variant in PHASE48_COMPRESSED_VARIANTS:
        required = [
            compressed_deltas.get(f"{compressed_variant}_minus_B0"),
            compressed_deltas.get(f"{compressed_variant}_minus_B1"),
        ]
        if all(
            isinstance(summary, Mapping) and _mean_delta(summary) > tolerance
            for summary in required
        ):
            return True
    return False


def _phase48_conclusion(status: str) -> str:
    if status == "compressed_geofm_route_supported":
        return (
            "Phase 48 conclusion: compressed GeoFM route supported under the "
            "current Bishan base-reward held-out protocol. D4P8/D4P16 improve "
            "over B0, raw B1, random D2, and shuffled D3 in mean learned-policy "
            "reward. This narrows the old negative result to raw direct "
            "injection unsupported; suitability reward remains blocked without "
            "independent labels."
        )
    if status == "compressed_geofm_route_partial":
        return (
            "Phase 48 conclusion: compressed GeoFM route partially supported. "
            "At least one compressed GeoFM variant recovers the B0/raw-B1 gap, "
            "but the random or shuffled representation-control comparison is "
            "not yet strong enough for the compressed route claim."
        )
    if status == "compressed_geofm_route_not_supported":
        return (
            "Phase 48 conclusion: compressed GeoFM route not supported under "
            "this summary set. The result does not rescue the representation "
            "hypothesis beyond the raw B1 negative evidence."
        )
    return (
        "Phase 48 conclusion: insufficient coverage for a compressed GeoFM "
        "route decision. Required B0/B1/D2/D3/D4P8/D4P16 tile-seed rows are "
        "missing or duplicated."
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
    summary_rows: list[dict[str, object]] = []
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
                "claim_boundary": PHASE48_CLAIM_BOUNDARY,
            }
        )
    return summary_rows


def _phase48_readiness_markdown(analysis: Mapping[str, object]) -> str:
    learned = analysis.get("learned_policy")
    if not isinstance(learned, Mapping):
        learned = {}
    mean_rewards = learned.get("mean_reward_by_variant")
    if not isinstance(mean_rewards, Mapping):
        mean_rewards = {}
    compressed_deltas = learned.get("compressed_deltas")
    if not isinstance(compressed_deltas, Mapping):
        compressed_deltas = {}
    pooled = analysis.get("pooled_compressed_control_delta")
    if not isinstance(pooled, Mapping):
        pooled = {}

    lines = [
        "# Phase 48 Compressed GeoFM Rescue",
        "",
        f"Status: {analysis.get('phase48_compressed_geofm_status', '')}",
        "",
        "Compressed GeoFM route conclusion:",
        str(analysis.get("conclusion", "")),
        "",
        "Scope: read-only audit over existing base-reward held-out Bishan summary rows.",
        "",
        "Mean learned-policy reward by variant:",
    ]
    for variant_id in sorted(mean_rewards):
        lines.append(f"- {variant_id}: {mean_rewards[variant_id]}")
    lines.extend(["", "Compressed candidate deltas:"])
    for key in sorted(compressed_deltas):
        value = compressed_deltas[key]
        if not isinstance(value, Mapping):
            continue
        lines.append(
            "- "
            f"{key}: {value.get('mean_reward_delta')} "
            f"({value.get('positive_tile_seed_count')} / "
            f"{value.get('total_tile_seed_count')} positive)"
        )
    lines.extend(
        [
            "",
            "Pooled compressed-control delta:",
            "- "
            f"mean={pooled.get('mean_reward_delta')}, "
            f"positive={pooled.get('positive_tile_seed_count')} / "
            f"{pooled.get('total_tile_seed_count')}",
            "",
            "Claim boundary:",
            str(analysis.get("claim_boundary", PHASE48_CLAIM_BOUNDARY)),
            "",
        ]
    )
    return "\n".join(lines)


def _write_summary_csv(path: Path, rows: object) -> None:
    if not isinstance(rows, list):
        raise ValueError("Phase 48 analysis is missing summary rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Phase 48 summary rows must be objects")
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDNAMES})


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    row_key: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 48 analysis is missing {row_key}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 48 {row_key} rows must be objects")
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _metadata_string_list(
    metadata: Mapping[str, object],
    key: str,
    fallback: list[str],
) -> list[str]:
    value = metadata.get(key)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
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
    if isinstance(value, Iterable):
        return [int(item) for item in value if str(item).strip()]
    return fallback


def _metadata_int(
    metadata: Mapping[str, object],
    keys: tuple[str, ...],
    rows: list[dict[str, object]],
    row_field: str,
) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip() != "":
            return int(value)
    for row in rows:
        value = row.get(row_field)
        if value is not None and str(value).strip() != "":
            return int(value)
    return None


def _has_coverage_issues(coverage_issues: Mapping[str, object]) -> bool:
    return any(
        bool(coverage_issues.get(key))
        for key in (
            "missing_variant_rows",
            "unexpected_variant_rows",
            "duplicate_variant_rows",
        )
    )


def _mean_delta(summary: Mapping[str, object]) -> float:
    value = summary.get("mean_reward_delta")
    return float(value) if value is not None else 0.0


def _positive_fraction(summary: Mapping[str, object]) -> float:
    value = summary.get("positive_fraction")
    return float(value) if value is not None else 0.0


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


def _unique_strings(rows: list[dict[str, object]], field: str) -> list[str]:
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


def _unique_ints(rows: list[dict[str, object]], field: str) -> list[int]:
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


def _variant_key_dicts(keys: set[tuple[str, int, str]]) -> list[dict[str, object]]:
    return [
        {"eval_tile_id": eval_tile_id, "seed": seed, "variant_id": variant_id}
        for eval_tile_id, seed, variant_id in sorted(
            keys,
            key=lambda item: (item[0], item[1], item[2]),
        )
    ]


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _round_float(value: float) -> float:
    return round(float(value), 10)
