from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .block_schema import summarize_phase2_readiness


CLAIM_BOUNDARY = (
    "The suitability_proxy is derived from latent remote-sensing embeddings "
    "and does not directly measure soil quality, fertility, or irrigation access."
)
WEAK_LABEL_CLAIM_BOUNDARY = (
    "Weak-label validation is a diagnostic proxy check for whether the "
    "suitability_proxy is directionally aligned with available planning labels; "
    "it is not proof of agronomic validity or direct measurement of soil quality, "
    "fertility, or irrigation access."
)

BASE_COLUMNS = [
    "region_id",
    "pixel_count",
    "row_min",
    "row_max",
    "col_min",
    "col_max",
]
EMBEDDING_COLUMNS = [f"embedding_mean_{idx:02d}" for idx in range(64)]
METRIC_COLUMNS = [
    "embedding_std_mean",
    "temporal_stability",
    "suitability_proxy",
]
BLOCK_BASE_COLUMNS = [
    "block_id",
    "pixel_count",
    "pixel_weight_sum",
    "row_min",
    "row_max",
    "col_min",
    "col_max",
]
WEAK_LABEL_COLUMNS = [
    "stable_farmland_label",
    "high_standard_farmland_label",
]


def write_phase1_artifacts(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    summary: Mapping[str, object],
) -> dict[str, Path]:
    """Write Phase 1 CSV and JSON artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    region_table = output_dir / "region_features.csv"
    summary_path = output_dir / "summary.json"

    fieldnames = _fieldnames(rows)
    with region_table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    suitability = np.array(
        [float(row["suitability_proxy"]) for row in rows], dtype=np.float64
    )
    output_summary = dict(summary)
    output_summary.update(
        {
            "n_regions": len(rows),
            "region_table": region_table.name,
            "claim_boundary": CLAIM_BOUNDARY,
            "suitability_min": float(suitability.min()) if suitability.size else None,
            "suitability_max": float(suitability.max()) if suitability.size else None,
            "suitability_mean": float(suitability.mean()) if suitability.size else None,
        }
    )
    summary_path.write_text(
        json.dumps(output_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {"region_table": region_table, "summary": summary_path}


def write_phase2_artifacts(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    summary: Mapping[str, object],
) -> dict[str, Path]:
    """Write Phase 2 block-level CSV and JSON artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    block_table = output_dir / "block_geofm_features.csv"
    summary_path = output_dir / "summary.json"
    weak_label_validation_path = output_dir / "weak_label_validation.json"
    fieldnames = _phase2_fieldnames(rows)

    with block_table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    suitability = np.array(
        [float(row["suitability_proxy"]) for row in rows if "suitability_proxy" in row],
        dtype=np.float64,
    )
    output_summary = dict(summary)
    output_summary.pop("weak_label_validation", None)
    artifact_paths = {"block_table": block_table, "summary": summary_path}
    weak_label_validation = _build_weak_label_validation(rows)
    if weak_label_validation is not None:
        weak_label_validation_path.write_text(
            json.dumps(weak_label_validation, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        output_summary["weak_label_validation"] = weak_label_validation_path.name
        artifact_paths["weak_label_validation"] = weak_label_validation_path
    elif weak_label_validation_path.exists():
        weak_label_validation_path.unlink()

    output_summary.update(
        {
            "n_blocks": len(rows),
            "block_table": block_table.name,
            "feature_readiness": summarize_phase2_readiness(rows),
            "claim_boundary": CLAIM_BOUNDARY,
            "suitability_min": float(suitability.min()) if suitability.size else None,
            "suitability_max": float(suitability.max()) if suitability.size else None,
            "suitability_mean": float(suitability.mean()) if suitability.size else None,
        }
    )
    summary_path.write_text(
        json.dumps(output_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return artifact_paths


def _fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    known = BASE_COLUMNS + EMBEDDING_COLUMNS + METRIC_COLUMNS
    extras = sorted({key for row in rows for key in row if key not in known})
    return [field for field in known if any(field in row for row in rows)] + extras


def _phase2_fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    known = BLOCK_BASE_COLUMNS + EMBEDDING_COLUMNS + METRIC_COLUMNS
    extras = sorted({key for row in rows for key in row if key not in known})
    return [field for field in known if any(field in row for row in rows)] + extras


def _build_weak_label_validation(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    summaries: dict[str, dict[str, int | float | None]] = {}
    label_columns = sorted(
        column
        for column in WEAK_LABEL_COLUMNS
        if any(column in row for row in rows)
    )

    for column in label_columns:
        positives: list[float] = []
        negatives: list[float] = []
        for row in rows:
            if "suitability_proxy" not in row:
                continue
            label = _parse_binary_label(row.get(column))
            if label is None:
                continue
            suitability = float(row["suitability_proxy"])
            if label == 1:
                positives.append(suitability)
            else:
                negatives.append(suitability)

        summaries[column] = {
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "positive_suitability_mean": _mean_or_none(positives),
            "negative_suitability_mean": _mean_or_none(negatives),
            "mean_difference": _mean_difference(positives, negatives),
            "rank_auc": _rank_auc_or_none(positives, negatives),
        }

    if not summaries:
        return None

    return {
        "validation_available": True,
        "label_columns": label_columns,
        "labels": summaries,
        "claim_boundary": WEAK_LABEL_CLAIM_BOUNDARY,
    }


def _parse_binary_label(value: object) -> int | None:
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


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(float(np.mean(values)), 10)


def _mean_difference(
    positives: Sequence[float],
    negatives: Sequence[float],
) -> float | None:
    if not positives or not negatives:
        return None
    return round(float(np.mean(positives) - np.mean(negatives)), 10)


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
    return round(favorable_pairs / (len(positives) * len(negatives)), 10)
