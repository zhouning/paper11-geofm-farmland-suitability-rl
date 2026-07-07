from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import math
from os import PathLike
from pathlib import Path


PHASE50_CLAIM_BOUNDARY = (
    "Phase 50 is a read-only tile-seed cluster-level audit over existing "
    "Phase 48 compressed GeoFM delta rows. It checks whether the compressed "
    "route remains positive after aggregating non-independent comparison rows "
    "to tile-seed clusters; it does not enable suitability reward, does not "
    "test B2/B3, does not test cross-region transfer, and does not validate "
    "independent agronomic suitability."
)

CLUSTER_FIELDNAMES = [
    "eval_tile_id",
    "seed",
    "cluster_delta_count",
    "mean_cluster_delta",
    "cluster_positive",
    "claim_boundary",
]


def build_phase50_cluster_level_robustness(
    delta_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    alpha: float = 0.05,
) -> dict[str, object]:
    rows = _load_delta_rows(delta_rows_or_csv)
    cluster_rows = _cluster_rows(rows)
    summary = _cluster_summary(cluster_rows)
    status = _phase50_status(summary, alpha=float(alpha))
    return {
        "phase": "phase50_cluster_level_robustness",
        "source_row_count": len(rows),
        "alpha": float(alpha),
        "cluster_rows": cluster_rows,
        "cluster_summary": summary,
        "phase50_cluster_status": status,
        "conclusion": _phase50_conclusion(status),
        "claim_boundary": PHASE50_CLAIM_BOUNDARY,
    }


def write_phase50_cluster_level_robustness_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_path = output_path / "phase50_cluster_level_robustness.json"
    cluster_path = output_path / "phase50_cluster_delta_summary.csv"
    readiness_path = output_path / "phase50_cluster_level_robustness.md"

    comparison_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv_mapping_rows(
        cluster_path,
        CLUSTER_FIELDNAMES,
        analysis.get("cluster_rows"),
        "cluster_rows",
    )
    readiness_path.write_text(_readiness_markdown(analysis), encoding="utf-8")
    return {
        "comparison_json": comparison_path,
        "cluster_csv": cluster_path,
        "readiness_md": readiness_path,
    }


def _load_delta_rows(
    delta_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(delta_rows_or_csv, (str, PathLike)):
        path = Path(delta_rows_or_csv)
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 50 delta CSV: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    rows = []
    for row in delta_rows_or_csv:
        if not isinstance(row, Mapping):
            raise ValueError("Phase 50 delta rows must be objects")
        rows.append(dict(row))
    return rows


def _cluster_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        key = (str(row.get("eval_tile_id", "")), _int_value(row, "seed"))
        grouped.setdefault(key, []).append(
            _float_value(row, "compressed_minus_comparator_reward")
        )
    cluster_rows = []
    for eval_tile_id, seed in sorted(grouped):
        values = grouped[(eval_tile_id, seed)]
        mean_delta = _round_float(sum(values) / len(values))
        cluster_rows.append(
            {
                "eval_tile_id": eval_tile_id,
                "seed": seed,
                "cluster_delta_count": len(values),
                "mean_cluster_delta": mean_delta,
                "cluster_positive": mean_delta > 0.0,
                "claim_boundary": PHASE50_CLAIM_BOUNDARY,
            }
        )
    return cluster_rows


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


def _phase50_status(summary: Mapping[str, object], alpha: float) -> str:
    mean_delta = float(summary.get("mean_cluster_delta") or 0.0)
    positive_fraction = float(summary.get("positive_cluster_fraction") or 0.0)
    p_value = summary.get("one_sided_sign_test_p")
    if mean_delta <= 0.0 or positive_fraction <= 0.5:
        return "cluster_not_supported"
    if p_value is not None and float(p_value) <= float(alpha):
        return "cluster_statistical_support"
    return "cluster_directional_support"


def _phase50_conclusion(status: str) -> str:
    if status == "cluster_statistical_support":
        return (
            "Phase 50 conclusion: compressed GeoFM route support remains "
            "statistically significant after aggregating to tile-seed clusters."
        )
    if status == "cluster_directional_support":
        return (
            "Phase 50 conclusion: compressed GeoFM route support remains "
            "positive after tile-seed clustering, but the small cluster count "
            "does not clear the alpha threshold."
        )
    return (
        "Phase 50 conclusion: compressed GeoFM route support does not survive "
        "tile-seed cluster aggregation."
    )


def _one_sided_sign_test_p(positive_count: int, total_count: int) -> float | None:
    if total_count <= 0:
        return None
    tail = sum(math.comb(total_count, k) for k in range(positive_count, total_count + 1))
    return _round_float(tail / (2**total_count))


def _readiness_markdown(analysis: Mapping[str, object]) -> str:
    summary = analysis.get("cluster_summary")
    if not isinstance(summary, Mapping):
        summary = {}
    lines = [
        "# Phase 50 Cluster-Level Robustness",
        "",
        f"Status: {analysis.get('phase50_cluster_status', '')}",
        "",
        "Tile-seed cluster conclusion:",
        str(analysis.get("conclusion", "")),
        "",
        "Cluster summary:",
        "- "
        f"mean={summary.get('mean_cluster_delta')}, "
        f"positive={summary.get('positive_cluster_count')} / "
        f"{summary.get('cluster_count')}, "
        f"sign-test p={summary.get('one_sided_sign_test_p')}",
        "",
        "Claim boundary:",
        str(analysis.get("claim_boundary", PHASE50_CLAIM_BOUNDARY)),
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
        raise ValueError(f"Phase 50 analysis is missing {row_key}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 50 {row_key} rows must be objects")
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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
