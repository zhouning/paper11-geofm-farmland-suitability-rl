from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import math
import random
import statistics
from os import PathLike
from pathlib import Path


PHASE49_CLAIM_BOUNDARY = (
    "Phase 49 is a read-only statistical robustness audit over existing "
    "Phase 48 compressed GeoFM delta rows. It tests sign-test, bootstrap, and "
    "leave-one tile/seed sensitivity for the compressed base-reward route; it "
    "does not enable suitability reward, does not test B2/B3, does not test "
    "cross-region transfer, and does not support independent agronomic "
    "suitability claims."
)

PER_COMPARISON_FIELDNAMES = [
    "comparison_id",
    "compressed_variant_id",
    "comparator_variant_id",
    "mean_delta",
    "std_delta",
    "positive_count",
    "total_count",
    "positive_fraction",
    "one_sided_sign_test_p",
    "bootstrap_ci95_low",
    "bootstrap_ci95_high",
    "claim_boundary",
]

LEAVE_ONE_FIELDNAMES = [
    "group_type",
    "held_out_group",
    "mean_delta",
    "positive_count",
    "total_count",
    "positive_fraction",
    "claim_boundary",
]


def build_phase49_compressed_route_robustness(
    delta_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    bootstrap_iterations: int = 5000,
    random_seed: int = 49,
    alpha: float = 0.05,
) -> dict[str, object]:
    if int(bootstrap_iterations) <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    rows = _load_delta_rows(delta_rows_or_csv)
    deltas = [_float_value(row, "compressed_minus_comparator_reward") for row in rows]
    per_comparison = _per_comparison_summaries(
        rows,
        bootstrap_iterations=int(bootstrap_iterations),
        random_seed=int(random_seed),
    )
    pooled_delta = _delta_summary(
        deltas,
        bootstrap_iterations=int(bootstrap_iterations),
        random_seed=int(random_seed),
    )
    leave_one_tile_rows = _leave_one_rows(rows, "eval_tile_id", "tile")
    leave_one_seed_rows = _leave_one_rows(rows, "seed", "seed")
    leave_one_rows = [*leave_one_tile_rows, *leave_one_seed_rows]
    leave_one_tile_summary = _leave_one_summary(leave_one_tile_rows)
    leave_one_seed_summary = _leave_one_summary(leave_one_seed_rows)
    status = _phase49_status(
        per_comparison,
        pooled_delta,
        leave_one_tile_summary,
        leave_one_seed_summary,
        alpha=float(alpha),
    )
    return {
        "phase": "phase49_compressed_route_robustness",
        "source_row_count": len(rows),
        "bootstrap_iterations": int(bootstrap_iterations),
        "random_seed": int(random_seed),
        "alpha": float(alpha),
        "per_comparison": per_comparison,
        "per_comparison_rows": _per_comparison_rows(per_comparison),
        "pooled_delta": pooled_delta,
        "leave_one_rows": leave_one_rows,
        "leave_one_tile_summary": leave_one_tile_summary,
        "leave_one_seed_summary": leave_one_seed_summary,
        "phase49_robustness_status": status,
        "conclusion": _phase49_conclusion(status),
        "claim_boundary": PHASE49_CLAIM_BOUNDARY,
    }


def write_phase49_compressed_route_robustness_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_path = output_path / "phase49_compressed_route_robustness.json"
    per_comparison_path = output_path / "phase49_per_comparison_robustness.csv"
    leave_one_path = output_path / "phase49_leave_one_sensitivity.csv"
    readiness_path = output_path / "phase49_compressed_route_robustness.md"

    comparison_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv_mapping_rows(
        per_comparison_path,
        PER_COMPARISON_FIELDNAMES,
        analysis.get("per_comparison_rows"),
        "per_comparison_rows",
    )
    _write_csv_mapping_rows(
        leave_one_path,
        LEAVE_ONE_FIELDNAMES,
        analysis.get("leave_one_rows"),
        "leave_one_rows",
    )
    readiness_path.write_text(_readiness_markdown(analysis), encoding="utf-8")
    return {
        "comparison_json": comparison_path,
        "per_comparison_csv": per_comparison_path,
        "leave_one_csv": leave_one_path,
        "readiness_md": readiness_path,
    }


def _load_delta_rows(
    delta_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(delta_rows_or_csv, (str, PathLike)):
        path = Path(delta_rows_or_csv)
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 49 delta CSV: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    rows = []
    for row in delta_rows_or_csv:
        if not isinstance(row, Mapping):
            raise ValueError("Phase 49 delta rows must be objects")
        rows.append(dict(row))
    return rows


def _per_comparison_summaries(
    rows: list[dict[str, object]],
    bootstrap_iterations: int,
    random_seed: int,
) -> dict[str, dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (
            str(row.get("compressed_variant_id", "")),
            str(row.get("comparator_variant_id", "")),
        )
        grouped.setdefault(key, []).append(
            _float_value(row, "compressed_minus_comparator_reward")
        )
    summaries = {}
    for index, ((compressed, comparator), values) in enumerate(sorted(grouped.items())):
        summaries[f"{compressed}_minus_{comparator}"] = {
            "compressed_variant_id": compressed,
            "comparator_variant_id": comparator,
            **_delta_summary(
                values,
                bootstrap_iterations=bootstrap_iterations,
                random_seed=random_seed + index + 1,
            ),
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


def _one_sided_sign_test_p(positive_count: int, total_count: int) -> float | None:
    if total_count <= 0:
        return None
    tail = sum(math.comb(total_count, k) for k in range(positive_count, total_count + 1))
    return _round_float(tail / (2**total_count))


def _leave_one_rows(
    rows: list[dict[str, object]],
    field: str,
    group_type: str,
) -> list[dict[str, object]]:
    groups = sorted({str(row.get(field, "")) for row in rows})
    result = []
    for group in groups:
        kept = [
            _float_value(row, "compressed_minus_comparator_reward")
            for row in rows
            if str(row.get(field, "")) != group
        ]
        summary = _simple_delta_summary(kept)
        result.append(
            {
                "group_type": group_type,
                "held_out_group": group,
                "mean_delta": summary["mean_delta"],
                "positive_count": summary["positive_count"],
                "total_count": summary["total_count"],
                "positive_fraction": summary["positive_fraction"],
                "claim_boundary": PHASE49_CLAIM_BOUNDARY,
            }
        )
    return result


def _leave_one_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    means = [
        float(row["mean_delta"])
        for row in rows
        if row.get("mean_delta") is not None
    ]
    return {
        "group_count": len(rows),
        "min_mean_delta": _round_float(min(means)) if means else None,
        "all_leave_one_means_positive": all(value > 0.0 for value in means)
        if means
        else False,
    }


def _simple_delta_summary(values: Sequence[float]) -> dict[str, object]:
    deltas = [float(value) for value in values]
    positive_count = sum(1 for value in deltas if value > 0.0)
    total_count = len(deltas)
    return {
        "mean_delta": _mean_or_none(deltas),
        "positive_count": positive_count,
        "total_count": total_count,
        "positive_fraction": _round_float(positive_count / total_count)
        if total_count
        else None,
    }


def _phase49_status(
    per_comparison: Mapping[str, object],
    pooled_delta: Mapping[str, object],
    leave_one_tile_summary: Mapping[str, object],
    leave_one_seed_summary: Mapping[str, object],
    alpha: float,
) -> str:
    all_comparisons_positive = all(
        isinstance(summary, Mapping)
        and float(summary.get("mean_delta") or 0.0) > 0.0
        for summary in per_comparison.values()
    )
    if not all_comparisons_positive:
        return "compressed_route_fragile"
    if (
        float(pooled_delta.get("mean_delta") or 0.0) > 0.0
        and float(pooled_delta.get("one_sided_sign_test_p") or 1.0) <= float(alpha)
        and float(pooled_delta.get("bootstrap_ci95_low") or 0.0) > 0.0
        and bool(leave_one_tile_summary.get("all_leave_one_means_positive"))
        and bool(leave_one_seed_summary.get("all_leave_one_means_positive"))
    ):
        return "compressed_route_statistically_robust"
    return "compressed_route_fragile"


def _phase49_conclusion(status: str) -> str:
    if status == "compressed_route_statistically_robust":
        return (
            "Phase 49 conclusion: the Phase 48 compressed GeoFM route remains "
            "positive under pooled sign-test, bootstrap, and leave-one "
            "tile/seed sensitivity checks."
        )
    return (
        "Phase 49 conclusion: the compressed GeoFM route has positive evidence "
        "but does not clear all statistical robustness checks."
    )


def _per_comparison_rows(
    per_comparison: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = []
    for comparison_id, summary in sorted(per_comparison.items()):
        if not isinstance(summary, Mapping):
            continue
        rows.append(
            {
                "comparison_id": comparison_id,
                "compressed_variant_id": summary.get("compressed_variant_id"),
                "comparator_variant_id": summary.get("comparator_variant_id"),
                "mean_delta": summary.get("mean_delta"),
                "std_delta": summary.get("std_delta"),
                "positive_count": summary.get("positive_count"),
                "total_count": summary.get("total_count"),
                "positive_fraction": summary.get("positive_fraction"),
                "one_sided_sign_test_p": summary.get("one_sided_sign_test_p"),
                "bootstrap_ci95_low": summary.get("bootstrap_ci95_low"),
                "bootstrap_ci95_high": summary.get("bootstrap_ci95_high"),
                "claim_boundary": PHASE49_CLAIM_BOUNDARY,
            }
        )
    return rows


def _readiness_markdown(analysis: Mapping[str, object]) -> str:
    pooled = analysis.get("pooled_delta")
    if not isinstance(pooled, Mapping):
        pooled = {}
    lines = [
        "# Phase 49 Compressed Route Robustness",
        "",
        f"Status: {analysis.get('phase49_robustness_status', '')}",
        "",
        "Compressed route robustness:",
        str(analysis.get("conclusion", "")),
        "",
        "Pooled delta:",
        "- "
        f"mean={pooled.get('mean_delta')}, "
        f"positive={pooled.get('positive_count')} / {pooled.get('total_count')}, "
        f"sign-test p={pooled.get('one_sided_sign_test_p')}, "
        f"bootstrap CI95=[{pooled.get('bootstrap_ci95_low')}, "
        f"{pooled.get('bootstrap_ci95_high')}]",
        "",
        "Claim boundary:",
        str(analysis.get("claim_boundary", PHASE49_CLAIM_BOUNDARY)),
        "",
    ]
    return "\n".join(lines)


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    row_key: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 49 analysis is missing {row_key}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 49 {row_key} rows must be objects")
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _float_value(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing numeric field: {field}")
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
    return value


def _round_float(value: float) -> float:
    return round(float(value), 10)
