from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
import csv
import json
import math
import random
from os import PathLike
from pathlib import Path


PHASE53_CLAIM_BOUNDARY = (
    "Phase 53 is a read-only cluster mean support audit over Phase 50 "
    "tile-seed cluster deltas from the expanded Phase 52 replication. It uses "
    "an exact one-sided sign-flip mean test, cluster bootstrap confidence "
    "intervals, and leave-one influence checks; it does not enable suitability "
    "reward, does not test B2/B3, does not test cross-region transfer, and "
    "does not validate independent agronomic suitability."
)

LEAVE_ONE_FIELDNAMES = [
    "group_type",
    "held_out_group",
    "mean_delta",
    "cluster_count",
    "claim_boundary",
]


def build_phase53_cluster_mean_support(
    cluster_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    bootstrap_iterations: int = 5000,
    random_seed: int = 53,
    alpha: float = 0.05,
) -> dict[str, object]:
    if int(bootstrap_iterations) <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    rows = _load_cluster_rows(cluster_rows_or_csv)
    values = [_float_value(row, "mean_cluster_delta") for row in rows]
    mean_delta = _mean(values)
    low, high = _bootstrap_mean_ci(
        values,
        iterations=int(bootstrap_iterations),
        random_seed=int(random_seed),
    )
    exact_p = _exact_sign_flip_mean_p(values, mean_delta)
    leave_one_rows = [
        *_leave_one_cluster_rows(rows),
        *_leave_one_group_rows(rows, "eval_tile_id", "tile"),
        *_leave_one_group_rows(rows, "seed", "seed"),
    ]
    influence_summary = _influence_summary(leave_one_rows)
    status = _phase53_status(
        mean_delta=mean_delta,
        exact_p=exact_p,
        bootstrap_low=low,
        influence_summary=influence_summary,
        alpha=float(alpha),
    )
    return {
        "phase": "phase53_cluster_mean_support",
        "cluster_count": len(values),
        "mean_cluster_delta": mean_delta,
        "exact_sign_flip_mean_p": exact_p,
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "bootstrap_iterations": int(bootstrap_iterations),
        "random_seed": int(random_seed),
        "alpha": float(alpha),
        "leave_one_rows": leave_one_rows,
        "influence_summary": influence_summary,
        "phase53_cluster_mean_status": status,
        "conclusion": _phase53_conclusion(status),
        "claim_boundary": PHASE53_CLAIM_BOUNDARY,
    }


def write_phase53_cluster_mean_support_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_path = output_path / "phase53_cluster_mean_support.json"
    leave_one_path = output_path / "phase53_leave_one_influence.csv"
    readiness_path = output_path / "phase53_cluster_mean_support.md"

    comparison_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
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
        "leave_one_csv": leave_one_path,
        "readiness_md": readiness_path,
    }


def _load_cluster_rows(
    cluster_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(cluster_rows_or_csv, (str, PathLike)):
        path = Path(cluster_rows_or_csv)
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 53 cluster CSV: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return [dict(row) for row in cluster_rows_or_csv]


def _bootstrap_mean_ci(
    values: Sequence[float],
    iterations: int,
    random_seed: int,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(int(random_seed))
    means = []
    n = len(values)
    for _ in range(int(iterations)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low_index = max(0, int(math.floor(0.025 * (len(means) - 1))))
    high_index = min(len(means) - 1, int(math.ceil(0.975 * (len(means) - 1))))
    return _round_float(means[low_index]), _round_float(means[high_index])


def _exact_sign_flip_mean_p(values: Sequence[float], observed_mean: float) -> float | None:
    if not values:
        return None
    magnitudes = [abs(float(value)) for value in values if float(value) != 0.0]
    if not magnitudes:
        return None
    total = 0
    at_least_observed = 0
    n = len(magnitudes)
    for signs in product((-1.0, 1.0), repeat=n):
        mean_value = sum(sign * value for sign, value in zip(signs, magnitudes)) / n
        total += 1
        if mean_value >= float(observed_mean) - 1e-12:
            at_least_observed += 1
    return _round_float(at_least_observed / total)


def _leave_one_cluster_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    values = [_float_value(row, "mean_cluster_delta") for row in rows]
    result = []
    for index, row in enumerate(rows):
        kept = [value for kept_index, value in enumerate(values) if kept_index != index]
        result.append(
            {
                "group_type": "cluster",
                "held_out_group": f"{row.get('eval_tile_id')}|seed_{row.get('seed')}",
                "mean_delta": _mean(kept),
                "cluster_count": len(kept),
                "claim_boundary": PHASE53_CLAIM_BOUNDARY,
            }
        )
    return result


def _leave_one_group_rows(
    rows: Sequence[Mapping[str, object]],
    field: str,
    group_type: str,
) -> list[dict[str, object]]:
    groups = sorted({str(row.get(field, "")) for row in rows})
    result = []
    for group in groups:
        kept = [
            _float_value(row, "mean_cluster_delta")
            for row in rows
            if str(row.get(field, "")) != group
        ]
        result.append(
            {
                "group_type": group_type,
                "held_out_group": group,
                "mean_delta": _mean(kept),
                "cluster_count": len(kept),
                "claim_boundary": PHASE53_CLAIM_BOUNDARY,
            }
        )
    return result


def _influence_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_group: dict[str, list[float]] = {"cluster": [], "tile": [], "seed": []}
    for row in rows:
        group_type = str(row.get("group_type", ""))
        if group_type in by_group:
            by_group[group_type].append(_float_value(row, "mean_delta"))
    all_values = [value for values in by_group.values() for value in values]
    return {
        "min_leave_one_cluster_mean": _min_or_none(by_group["cluster"]),
        "min_leave_one_tile_mean": _min_or_none(by_group["tile"]),
        "min_leave_one_seed_mean": _min_or_none(by_group["seed"]),
        "all_leave_one_means_positive": bool(all_values)
        and all(value > 0.0 for value in all_values),
    }


def _phase53_status(
    mean_delta: float | None,
    exact_p: float | None,
    bootstrap_low: float | None,
    influence_summary: Mapping[str, object],
    alpha: float,
) -> str:
    if (
        mean_delta is not None
        and mean_delta > 0.0
        and exact_p is not None
        and exact_p <= float(alpha)
        and bootstrap_low is not None
        and bootstrap_low > 0.0
        and bool(influence_summary.get("all_leave_one_means_positive"))
    ):
        return "cluster_mean_support"
    return "cluster_mean_directional"


def _phase53_conclusion(status: str) -> str:
    if status == "cluster_mean_support":
        return (
            "Phase 53 conclusion: the expanded compressed-route cluster mean "
            "is supported by exact sign-flip, bootstrap, and leave-one "
            "influence checks."
        )
    return (
        "Phase 53 conclusion: the expanded compressed-route cluster mean "
        "remains directional but does not clear all cluster mean checks."
    )


def _readiness_markdown(analysis: Mapping[str, object]) -> str:
    influence = analysis.get("influence_summary")
    if not isinstance(influence, Mapping):
        influence = {}
    return "\n".join(
        [
            "# Phase 53 Cluster Mean Support",
            "",
            f"Status: {analysis.get('phase53_cluster_mean_status', '')}",
            "",
            "Cluster mean support:",
            str(analysis.get("conclusion", "")),
            "",
            "Exact and bootstrap result:",
            "- "
            f"mean={analysis.get('mean_cluster_delta')}, "
            f"sign-flip p={analysis.get('exact_sign_flip_mean_p')}, "
            f"bootstrap CI95=[{analysis.get('bootstrap_ci95_low')}, "
            f"{analysis.get('bootstrap_ci95_high')}]",
            "",
            "Leave-one influence:",
            "- "
            f"min cluster={influence.get('min_leave_one_cluster_mean')}, "
            f"min tile={influence.get('min_leave_one_tile_mean')}, "
            f"min seed={influence.get('min_leave_one_seed_mean')}",
            "",
            "Claim boundary:",
            str(analysis.get("claim_boundary", PHASE53_CLAIM_BOUNDARY)),
            "",
        ]
    )


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    row_key: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 53 analysis is missing {row_key}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 53 {row_key} rows must be objects")
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _float_value(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing numeric field: {field}")
    return float(value)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return _round_float(sum(float(value) for value in values) / len(values))


def _min_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return _round_float(min(values))


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
