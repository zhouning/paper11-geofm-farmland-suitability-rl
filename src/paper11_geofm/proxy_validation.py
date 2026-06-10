from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


PHASE9_CLAIM_BOUNDARY = (
    "Phase 9 is a weak-label proxy-validation report for suitability_proxy; "
    "it does not prove agronomic validity, train a policy, evaluate a policy, "
    "or report planning performance."
)
DEFAULT_LABEL_COLUMNS = (
    "stable_farmland_label",
    "high_standard_farmland_label",
)


def build_phase9_proxy_validation_report(
    phase2_output_dir: Path | str,
    label_columns: Sequence[str] = DEFAULT_LABEL_COLUMNS,
) -> dict[str, object]:
    output_dir = Path(phase2_output_dir)
    block_table = output_dir / "block_geofm_features.csv"
    rows = _read_block_rows(block_table)
    suitability_values = _extract_suitability(rows, block_table)
    requested = [str(column) for column in label_columns]
    labels = {column: _label_report(rows, column) for column in requested}
    available = [
        column
        for column in requested
        if labels[column]["interpretation"] != "label_unavailable"
    ]

    return {
        "phase": "phase9_proxy_validation_report",
        "phase2_output_dir": str(output_dir),
        "block_table": block_table.name,
        "label_columns_requested": requested,
        "label_columns_available": available,
        "label_columns_missing": [
            column for column in requested if column not in available
        ],
        "n_blocks": len(rows),
        "suitability_summary": _suitability_summary(suitability_values),
        "labels": labels,
        "claim_boundary": PHASE9_CLAIM_BOUNDARY,
    }


def write_phase9_proxy_validation_report(
    report: Mapping[str, object],
    output_dir: Path | str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "phase9_proxy_validation_report.json"
    report_path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path


def _read_block_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 2 block feature table: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _extract_suitability(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        values.append(_parse_required_float(row.get("suitability_proxy"), path))
    return values


def _parse_required_float(value: Any, path: Path) -> float:
    if value is None or str(value).strip() == "":
        raise ValueError(
            f"Phase 9 requires numeric suitability_proxy values in {path}"
        )
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Phase 9 requires numeric suitability_proxy values in {path}"
        ) from exc


def _suitability_summary(values: Sequence[float]) -> dict[str, int | float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Phase 9 requires at least one numeric suitability_proxy value")

    return {
        "count": int(array.size),
        "min": _rounded(float(array.min())),
        "max": _rounded(float(array.max())),
        "mean": _rounded(float(array.mean())),
        "std": _rounded(float(array.std())),
        "q25": _rounded(float(np.quantile(array, 0.25))),
        "median": _rounded(float(np.quantile(array, 0.50))),
        "q75": _rounded(float(np.quantile(array, 0.75))),
    }


def _label_report(
    rows: Sequence[Mapping[str, Any]],
    column: str,
) -> dict[str, object]:
    positives: list[float] = []
    negatives: list[float] = []
    parseable_labels = 0

    for row in rows:
        label = _parse_binary_label(row.get(column))
        if label is None:
            continue
        suitability = _parse_optional_float(row.get("suitability_proxy"))
        if suitability is None:
            continue
        parseable_labels += 1
        if label == 1:
            positives.append(suitability)
        else:
            negatives.append(suitability)

    positive_mean = _mean_or_none(positives)
    negative_mean = _mean_or_none(negatives)
    mean_difference = _mean_difference(positives, negatives)
    rank_auc = _rank_auc_or_none(positives, negatives)

    if parseable_labels == 0:
        interpretation = "label_unavailable"
    elif not positives or not negatives:
        interpretation = "insufficient_label_variation"
    elif (
        mean_difference is not None
        and rank_auc is not None
        and mean_difference > 0.0
        and rank_auc >= 0.5
    ):
        interpretation = "positive_alignment"
    else:
        interpretation = "negative_or_no_alignment"

    return {
        "validation_available": bool(positives and negatives),
        "valid_label_count": parseable_labels,
        "missing_label_count": len(rows) - parseable_labels,
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "positive_suitability_mean": positive_mean,
        "negative_suitability_mean": negative_mean,
        "mean_difference": mean_difference,
        "rank_auc": rank_auc,
        "suitability_quantiles_by_label": {
            "positive": _quantiles(positives),
            "negative": _quantiles(negatives),
        },
        "interpretation": interpretation,
    }


def _parse_binary_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes"}:
        return 1
    if text in {"0", "0.0", "false", "no"}:
        return 0
    return None


def _parse_optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return _rounded(float(np.mean(values)))


def _mean_difference(
    positives: Sequence[float],
    negatives: Sequence[float],
) -> float | None:
    if not positives or not negatives:
        return None
    return _rounded(float(np.mean(positives) - np.mean(negatives)))


def _rank_auc_or_none(
    positives: Sequence[float],
    negatives: Sequence[float],
) -> float | None:
    if not positives or not negatives:
        return None

    favorable_pairs = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                favorable_pairs += 1.0
            elif positive == negative:
                favorable_pairs += 0.5
    return _rounded(favorable_pairs / (len(positives) * len(negatives)))


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": _rounded(float(array.min())),
        "q25": _rounded(float(np.quantile(array, 0.25))),
        "median": _rounded(float(np.quantile(array, 0.50))),
        "q75": _rounded(float(np.quantile(array, 0.75))),
        "max": _rounded(float(array.max())),
    }


def _rounded(value: float) -> float:
    return round(float(value), 10)

