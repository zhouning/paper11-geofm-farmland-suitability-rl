from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
import csv
import json
from os import PathLike
from pathlib import Path


PHASE51_CLAIM_BOUNDARY = (
    "Phase 51 is a read-only magnitude-sensitive cluster audit over Phase 50 "
    "tile-seed cluster deltas. It uses an exact one-sided signed-rank test; it "
    "does not enable suitability reward, does not test B2/B3, does not test "
    "cross-region transfer, and does not validate independent agronomic "
    "suitability."
)

RANK_FIELDNAMES = [
    "eval_tile_id",
    "seed",
    "mean_cluster_delta",
    "abs_rank",
    "positive_rank",
    "claim_boundary",
]


def build_phase51_cluster_magnitude_support(
    cluster_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    alpha: float = 0.05,
) -> dict[str, object]:
    rows = _load_cluster_rows(cluster_rows_or_csv)
    rank_rows = _rank_rows(rows)
    values = [float(row["mean_cluster_delta"]) for row in rank_rows]
    positive_rank_sum = sum(float(row["abs_rank"]) for row in rank_rows if row["positive_rank"])
    total_rank_sum = sum(float(row["abs_rank"]) for row in rank_rows)
    p_value = _exact_signed_rank_p([float(row["abs_rank"]) for row in rank_rows], positive_rank_sum)
    mean_delta = _round_float(sum(values) / len(values)) if values else None
    status = (
        "cluster_magnitude_support"
        if mean_delta is not None and mean_delta > 0.0 and p_value <= float(alpha)
        else "cluster_magnitude_directional"
    )
    return {
        "phase": "phase51_cluster_magnitude_support",
        "cluster_count": len(rank_rows),
        "mean_cluster_delta": mean_delta,
        "positive_rank_sum": _int_if_whole(positive_rank_sum),
        "total_rank_sum": _int_if_whole(total_rank_sum),
        "one_sided_signed_rank_p": p_value,
        "alpha": float(alpha),
        "rank_rows": rank_rows,
        "phase51_magnitude_status": status,
        "conclusion": _phase51_conclusion(status),
        "claim_boundary": PHASE51_CLAIM_BOUNDARY,
    }


def write_phase51_cluster_magnitude_support_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_path = output_path / "phase51_cluster_magnitude_support.json"
    rank_path = output_path / "phase51_cluster_signed_rank.csv"
    readiness_path = output_path / "phase51_cluster_magnitude_support.md"
    comparison_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv_mapping_rows(rank_path, RANK_FIELDNAMES, analysis.get("rank_rows"), "rank_rows")
    readiness_path.write_text(_readiness_markdown(analysis), encoding="utf-8")
    return {
        "comparison_json": comparison_path,
        "rank_csv": rank_path,
        "readiness_md": readiness_path,
    }


def _load_cluster_rows(
    cluster_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(cluster_rows_or_csv, (str, PathLike)):
        path = Path(cluster_rows_or_csv)
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 51 cluster CSV: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return [dict(row) for row in cluster_rows_or_csv]


def _rank_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    nonzero = [
        (abs(_float_value(row, "mean_cluster_delta")), index, row)
        for index, row in enumerate(rows)
        if _float_value(row, "mean_cluster_delta") != 0.0
    ]
    ranked = []
    ranks_by_index = {}
    for rank, (_, index, _) in enumerate(sorted(nonzero), start=1):
        ranks_by_index[index] = rank
    for index, row in enumerate(rows):
        if index not in ranks_by_index:
            continue
        delta = _float_value(row, "mean_cluster_delta")
        ranked.append(
            {
                "eval_tile_id": str(row.get("eval_tile_id", "")),
                "seed": int(row.get("seed", 0)),
                "mean_cluster_delta": _round_float(delta),
                "abs_rank": ranks_by_index[index],
                "positive_rank": delta > 0.0,
                "claim_boundary": PHASE51_CLAIM_BOUNDARY,
            }
        )
    return ranked


def _exact_signed_rank_p(ranks: Sequence[float], observed_positive_rank_sum: float) -> float:
    total = 0
    at_least_observed = 0
    for signs in product((0, 1), repeat=len(ranks)):
        rank_sum = sum(rank for rank, sign in zip(ranks, signs) if sign)
        total += 1
        if rank_sum >= observed_positive_rank_sum:
            at_least_observed += 1
    return _round_float(at_least_observed / total)


def _phase51_conclusion(status: str) -> str:
    if status == "cluster_magnitude_support":
        return (
            "Phase 51 conclusion: the cluster-level compressed-route effect is "
            "supported by an exact magnitude-sensitive signed-rank test."
        )
    return (
        "Phase 51 conclusion: the cluster-level compressed-route effect remains "
        "directional but does not clear the signed-rank threshold."
    )


def _readiness_markdown(analysis: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Phase 51 Cluster Magnitude Support",
            "",
            f"Status: {analysis.get('phase51_magnitude_status', '')}",
            "",
            "Exact signed-rank result:",
            "- "
            f"positive rank sum={analysis.get('positive_rank_sum')}, "
            f"total rank sum={analysis.get('total_rank_sum')}, "
            f"p={analysis.get('one_sided_signed_rank_p')}",
            "",
            "Claim boundary:",
            str(analysis.get("claim_boundary", PHASE51_CLAIM_BOUNDARY)),
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
        raise ValueError(f"Phase 51 analysis is missing {row_key}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _float_value(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing numeric field: {field}")
    return float(value)


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _int_if_whole(value: float) -> int | float:
    return int(value) if float(value).is_integer() else _round_float(value)


def _round_float(value: float) -> float:
    return round(float(value), 10)
