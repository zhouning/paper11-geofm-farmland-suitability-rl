from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .block_schema import EMBEDDING_COLUMNS, EXPLICIT_FEATURE_COLUMNS
from .drl_inputs import load_variant_input


PHASE36_SUITABILITY_PROXY_CLAIM_BOUNDARY = (
    "Phase 36 is a read-only weak-label suitability-proxy validation over "
    "existing feature tables. It does not run policy training, does not alter "
    "rewards, does not enable suitability reward, does not test B2/B3 planning "
    "performance, and does not prove agronomic validity."
)

DEFAULT_PHASE36_LABEL_COLUMNS = (
    "current_farmland_label",
    "farmland_or_orchard_label",
    "low_slope_farmland_label",
)

LEAKAGE_RISK_LABELS = {
    "current_farmland_label",
    "farmland_or_orchard_label",
    "low_slope_farmland_label",
}

PHASE36_LABEL_FIELDNAMES = [
    "label_column",
    "available",
    "usable",
    "valid_label_count",
    "positive_count",
    "negative_count",
    "positive_rate",
    "train_count",
    "eval_count",
    "train_positive_count",
    "train_negative_count",
    "eval_positive_count",
    "eval_negative_count",
    "split_source",
    "label_leakage_risk",
    "claim_boundary",
]

PHASE36_MODEL_FIELDNAMES = [
    "label_column",
    "feature_family",
    "validation_status",
    "feature_count",
    "train_count",
    "eval_count",
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "accuracy",
    "positive_rate_eval",
    "top_coefficients",
    "claim_boundary",
]


def build_phase36_suitability_proxy_validation(
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str | None = None,
    normalized_controls_dir: Path | str | None = None,
    label_columns: Sequence[str] | str = DEFAULT_PHASE36_LABEL_COLUMNS,
    min_delta: float = 0.02,
) -> dict[str, object]:
    phase2_dir = Path(phase2_output_dir)
    block_rows = _read_csv_rows(
        phase2_dir / "block_geofm_features.csv",
        "Phase 2 block feature CSV",
    )
    requested_labels = _normalize_csvish_values(label_columns)
    available_labels = [
        label for label in requested_labels if _column_available(block_rows, label)
    ]
    if not available_labels:
        raise ValueError("Phase 36 no requested label columns are available")

    feature_families = _build_feature_families(
        phase2_dir,
        Path(phase8_output_dir) if phase8_output_dir is not None else None,
        Path(normalized_controls_dir) if normalized_controls_dir is not None else None,
    )
    if not feature_families:
        raise ValueError("Phase 36 found no usable feature families")

    label_summaries: dict[str, dict[str, object]] = {}
    model_rows: list[dict[str, object]] = []
    for label in requested_labels:
        summary = _label_summary(block_rows, label)
        label_summaries[label] = summary
        if not summary["usable"]:
            continue
        labels_by_block = _labels_by_block(block_rows, label)
        split = _split_for_blocks(block_rows, labels_by_block)
        for family in feature_families:
            model_rows.append(_evaluate_family(label, labels_by_block, split, family))

    row_count_summary = {
        "block_rows": len(block_rows),
        "feature_families": len(feature_families),
        "label_summaries": len(label_summaries),
        "model_rows": len(model_rows),
    }
    status = _phase36_status(model_rows, min_delta=float(min_delta))
    return {
        "phase": "phase36_suitability_proxy_validation",
        "phase36_proxy_validation_status": status,
        "source_paths": {
            "phase2_output_dir": str(phase2_dir),
            "phase8_output_dir": str(Path(phase8_output_dir))
            if phase8_output_dir is not None
            else None,
            "normalized_controls_dir": str(Path(normalized_controls_dir))
            if normalized_controls_dir is not None
            else None,
        },
        "label_columns_requested": requested_labels,
        "label_columns_available": available_labels,
        "feature_families": [family["feature_family"] for family in feature_families],
        "min_delta": float(min_delta),
        "row_counts": row_count_summary,
        "label_summaries": label_summaries,
        "label_summary_rows": list(label_summaries.values()),
        "model_rows": model_rows,
        "interpretation": _phase36_interpretation(status),
        "claim_boundary": PHASE36_SUITABILITY_PROXY_CLAIM_BOUNDARY,
    }


def write_phase36_suitability_proxy_validation_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    label_summary_path = output_path / "phase36_label_summary.csv"
    model_summary_path = output_path / "phase36_model_summary.csv"
    diagnosis_json_path = output_path / "phase36_suitability_proxy_validation.json"
    diagnosis_md_path = output_path / "phase36_suitability_proxy_validation.md"

    _write_csv_mapping_rows(
        label_summary_path,
        PHASE36_LABEL_FIELDNAMES,
        analysis.get("label_summary_rows"),
        "label_summary_rows",
    )
    _write_csv_mapping_rows(
        model_summary_path,
        PHASE36_MODEL_FIELDNAMES,
        analysis.get("model_rows"),
        "model_rows",
    )
    diagnosis_json_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    diagnosis_md_path.write_text(_phase36_markdown(analysis), encoding="utf-8")
    return {
        "label_summary_csv": label_summary_path,
        "model_summary_csv": model_summary_path,
        "diagnosis_json": diagnosis_json_path,
        "diagnosis_md": diagnosis_md_path,
    }


def _build_feature_families(
    phase2_dir: Path,
    phase8_dir: Path | None,
    normalized_dir: Path | None,
) -> list[dict[str, object]]:
    families: list[dict[str, object]] = []
    b0 = _load_variant_or_none(phase2_dir, "B0")
    b1 = _load_variant_or_none(phase2_dir, "B1")
    b2 = _load_variant_or_none(phase2_dir, "B2")
    if b0 is not None:
        families.append(_family_from_variant("explicit_only", b0))
    if b1 is not None:
        families.append(_family_from_variant("raw_geofm_only", b1, EMBEDDING_COLUMNS))
        families.append(_family_from_variant("explicit_plus_raw_geofm", b1))
    if b2 is not None:
        families.append(_family_from_variant("suitability_proxy_only", b2, ["suitability_proxy"]))
        families.append(_family_from_variant("explicit_plus_suitability_proxy", b2))
    if phase8_dir is not None:
        for variant_id, family_id in (
            ("D2", "explicit_plus_random_geofm"),
            ("D3", "explicit_plus_shuffled_geofm"),
            ("D4P8", "explicit_plus_pca8_geofm"),
            ("D4P16", "explicit_plus_pca16_geofm"),
        ):
            variant = _load_variant_or_none(phase8_dir, variant_id)
            if variant is not None:
                families.append(_family_from_variant(family_id, variant))
    if normalized_dir is not None:
        for variant_id, family_id in (
            ("N1Z", "explicit_plus_normalized_geofm_zscore"),
            ("N1ZR", "explicit_plus_normalized_geofm_zscore_row_l2"),
        ):
            variant = _load_variant_or_none(normalized_dir, variant_id)
            if variant is not None:
                families.append(_family_from_variant(family_id, variant))
    return families


def _load_variant_or_none(output_dir: Path, variant_id: str):
    try:
        return load_variant_input(output_dir, variant_id)
    except (FileNotFoundError, ValueError):
        return None


def _family_from_variant(
    family_id: str,
    variant_input,
    selected_columns: Sequence[str] | None = None,
) -> dict[str, object]:
    if selected_columns is None:
        matrix = np.asarray(variant_input.state_matrix, dtype=float)
        columns = list(variant_input.feature_columns)
    else:
        indexes = [variant_input.feature_columns.index(column) for column in selected_columns]
        matrix = np.asarray(variant_input.state_matrix[:, indexes], dtype=float)
        columns = [str(column) for column in selected_columns]
    return {
        "feature_family": family_id,
        "block_ids": list(variant_input.block_ids),
        "feature_columns": columns,
        "matrix": matrix,
    }


def _label_summary(
    block_rows: Sequence[Mapping[str, object]],
    label_column: str,
) -> dict[str, object]:
    if not _column_available(block_rows, label_column):
        return {
            "label_column": label_column,
            "available": False,
            "usable": False,
            "valid_label_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_rate": "",
            "train_count": 0,
            "eval_count": 0,
            "train_positive_count": 0,
            "train_negative_count": 0,
            "eval_positive_count": 0,
            "eval_negative_count": 0,
            "split_source": "unavailable",
            "label_leakage_risk": _label_leakage_risk(label_column),
            "claim_boundary": PHASE36_SUITABILITY_PROXY_CLAIM_BOUNDARY,
        }
    labels = _labels_by_block(block_rows, label_column)
    split = _split_for_blocks(block_rows, labels)
    train_labels = [labels[block_id] for block_id in split["train_block_ids"]]
    eval_labels = [labels[block_id] for block_id in split["eval_block_ids"]]
    positives = sum(1 for label in labels.values() if label == 1)
    negatives = sum(1 for label in labels.values() if label == 0)
    return {
        "label_column": label_column,
        "available": True,
        "usable": _has_binary_variation(train_labels) and _has_binary_variation(eval_labels),
        "valid_label_count": len(labels),
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": _round_float(positives / len(labels)) if labels else "",
        "train_count": len(train_labels),
        "eval_count": len(eval_labels),
        "train_positive_count": sum(1 for label in train_labels if label == 1),
        "train_negative_count": sum(1 for label in train_labels if label == 0),
        "eval_positive_count": sum(1 for label in eval_labels if label == 1),
        "eval_negative_count": sum(1 for label in eval_labels if label == 0),
        "split_source": split["split_source"],
        "label_leakage_risk": _label_leakage_risk(label_column),
        "claim_boundary": PHASE36_SUITABILITY_PROXY_CLAIM_BOUNDARY,
    }


def _evaluate_family(
    label_column: str,
    labels_by_block: Mapping[str, int],
    split: Mapping[str, object],
    family: Mapping[str, object],
) -> dict[str, object]:
    block_ids = [str(block_id) for block_id in family["block_ids"]]
    matrix = np.asarray(family["matrix"], dtype=float)
    block_index = {block_id: index for index, block_id in enumerate(block_ids)}
    train_ids = [
        block_id
        for block_id in split["train_block_ids"]
        if block_id in block_index and block_id in labels_by_block
    ]
    eval_ids = [
        block_id
        for block_id in split["eval_block_ids"]
        if block_id in block_index and block_id in labels_by_block
    ]
    train_y = np.asarray([labels_by_block[block_id] for block_id in train_ids], dtype=int)
    eval_y = np.asarray([labels_by_block[block_id] for block_id in eval_ids], dtype=int)
    base_row = {
        "label_column": label_column,
        "feature_family": family["feature_family"],
        "feature_count": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "train_count": len(train_ids),
        "eval_count": len(eval_ids),
        "positive_rate_eval": _round_float(float(np.mean(eval_y))) if eval_y.size else "",
        "claim_boundary": PHASE36_SUITABILITY_PROXY_CLAIM_BOUNDARY,
    }
    if not _has_binary_variation(train_y) or not _has_binary_variation(eval_y):
        return {
            **base_row,
            "validation_status": "insufficient_label_variation",
            "roc_auc": "",
            "average_precision": "",
            "balanced_accuracy": "",
            "accuracy": "",
            "top_coefficients": [],
        }
    train_x = matrix[[block_index[block_id] for block_id in train_ids], :]
    eval_x = matrix[[block_index[block_id] for block_id in eval_ids], :]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=0,
            solver="liblinear",
        ),
    )
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(eval_x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        **base_row,
        "validation_status": "evaluated",
        "roc_auc": _round_float(roc_auc_score(eval_y, probabilities)),
        "average_precision": _round_float(average_precision_score(eval_y, probabilities)),
        "balanced_accuracy": _round_float(balanced_accuracy_score(eval_y, predictions)),
        "accuracy": _round_float(accuracy_score(eval_y, predictions)),
        "top_coefficients": _top_coefficients(model, family),
    }


def _top_coefficients(model, family: Mapping[str, object]) -> list[dict[str, object]]:
    columns = [str(column) for column in family["feature_columns"]]
    coefficients = np.asarray(model.named_steps["logisticregression"].coef_[0], dtype=float)
    indexes = np.argsort(np.abs(coefficients))[::-1][:5]
    return [
        {
            "feature": columns[int(index)] if int(index) < len(columns) else str(index),
            "coefficient": _round_float(coefficients[int(index)]),
        }
        for index in indexes
    ]


def _phase36_status(
    model_rows: Sequence[Mapping[str, object]],
    min_delta: float,
) -> str:
    evaluated = [
        row for row in model_rows if str(row.get("validation_status")) == "evaluated"
    ]
    if not evaluated:
        return "insufficient_proxy_labels"
    labels = sorted({str(row["label_column"]) for row in evaluated})
    for label in labels:
        rows = {
            str(row["feature_family"]): row
            for row in evaluated
            if str(row["label_column"]) == label
        }
        explicit = rows.get("explicit_only")
        if explicit is None:
            continue
        explicit_auc = _metric(explicit, "roc_auc")
        explicit_ap = _metric(explicit, "average_precision")
        control_aucs = [
            _metric(rows[family], "roc_auc")
            for family in ("explicit_plus_random_geofm", "explicit_plus_shuffled_geofm")
            if family in rows
        ]
        control_aps = [
            _metric(rows[family], "average_precision")
            for family in ("explicit_plus_random_geofm", "explicit_plus_shuffled_geofm")
            if family in rows
        ]
        control_auc = max(control_aucs) if control_aucs else explicit_auc
        control_ap = max(control_aps) if control_aps else explicit_ap
        for family in (
            "explicit_plus_raw_geofm",
            "explicit_plus_normalized_geofm_zscore",
            "explicit_plus_normalized_geofm_zscore_row_l2",
        ):
            candidate = rows.get(family)
            if candidate is None:
                continue
            if (
                _metric(candidate, "roc_auc") >= max(explicit_auc, control_auc) + min_delta
                and _metric(candidate, "average_precision")
                >= max(explicit_ap, control_ap) + min_delta
            ):
                return "proxy_signal_supported_for_bounded_reward_smoke"
    return "proxy_signal_not_supported"


def _phase36_interpretation(status: str) -> str:
    if status == "proxy_signal_supported_for_bounded_reward_smoke":
        return (
            "At least one GeoFM-derived family improves explicit-only and "
            "diagnostic controls by the configured threshold for a usable weak "
            "label. This can justify a bounded B2/B3 reward smoke, not final "
            "suitability or planning-performance claims."
        )
    if status == "proxy_signal_not_supported":
        return (
            "Usable weak labels exist, but GeoFM-derived families do not exceed "
            "explicit-only and diagnostic controls by the configured threshold."
        )
    return "No requested label has usable binary train/evaluation coverage."


def _phase36_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Phase 36 Suitability-Proxy Validation",
        "",
        f"Status: {analysis.get('phase36_proxy_validation_status', '')}",
        "",
        "## Label Summary",
        "",
        "| Label | Usable | Train / Eval | Positives / Negatives | Leakage risk |",
        "|---|---:|---:|---:|---|",
    ]
    label_rows = analysis.get("label_summary_rows")
    if isinstance(label_rows, list):
        for row in label_rows:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| {label} | {usable} | {train} / {eval} | {pos} / {neg} | {risk} |".format(
                    label=row.get("label_column", ""),
                    usable=row.get("usable", ""),
                    train=row.get("train_count", ""),
                    eval=row.get("eval_count", ""),
                    pos=row.get("positive_count", ""),
                    neg=row.get("negative_count", ""),
                    risk=row.get("label_leakage_risk", ""),
                )
            )
    lines.extend(["", "## Model Summary", ""])
    model_rows = analysis.get("model_rows")
    if isinstance(model_rows, list):
        grouped: dict[str, list[Mapping[str, object]]] = {}
        for row in model_rows:
            if isinstance(row, Mapping):
                grouped.setdefault(str(row.get("label_column", "")), []).append(row)
        for label in sorted(grouped):
            lines.extend(
                [
                    f"### {label}",
                    "",
                    "| Feature family | ROC AUC | AP | Balanced accuracy | Status |",
                    "|---|---:|---:|---:|---|",
                ]
            )
            for row in sorted(grouped[label], key=lambda item: str(item.get("feature_family", ""))):
                lines.append(
                    "| {family} | {auc} | {ap} | {bal} | {status} |".format(
                        family=row.get("feature_family", ""),
                        auc=row.get("roc_auc", ""),
                        ap=row.get("average_precision", ""),
                        bal=row.get("balanced_accuracy", ""),
                        status=row.get("validation_status", ""),
                    )
                )
            lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            str(analysis.get("interpretation", "")),
            "",
            "## Boundary",
            "",
            str(analysis.get("claim_boundary", "")),
            "",
            "High scores on DLTB-derived labels may reflect explicit-feature leakage and must not be treated as agronomic validation.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_csv_rows(path: Path, label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _column_available(rows: Sequence[Mapping[str, object]], column: str) -> bool:
    return any(column in row and str(row.get(column, "")).strip() != "" for row in rows)


def _labels_by_block(
    rows: Sequence[Mapping[str, object]],
    label_column: str,
) -> dict[str, int]:
    labels: dict[str, int] = {}
    for row in rows:
        block_id = str(row.get("block_id", "")).strip()
        label = _parse_binary_label(row.get(label_column))
        if block_id and label is not None:
            labels[block_id] = label
    return labels


def _split_for_blocks(
    rows: Sequence[Mapping[str, object]],
    labels_by_block: Mapping[str, int],
) -> dict[str, object]:
    split_by_block: dict[str, str] = {}
    for row in rows:
        block_id = str(row.get("block_id", "")).strip()
        if block_id not in labels_by_block:
            continue
        split_text = str(row.get("split", "")).strip().lower()
        if split_text:
            split_by_block[block_id] = split_text
    train_ids = [
        block_id
        for block_id, split in split_by_block.items()
        if split in {"train", "training"}
    ]
    eval_ids = [
        block_id
        for block_id, split in split_by_block.items()
        if split in {"test", "val", "valid", "validation", "eval", "evaluation"}
    ]
    if train_ids and eval_ids:
        return {
            "split_source": "split_column",
            "train_block_ids": sorted(train_ids),
            "eval_block_ids": sorted(eval_ids),
        }
    ordered = sorted(labels_by_block)
    eval_ids = ordered[::5]
    train_ids = [block_id for block_id in ordered if block_id not in set(eval_ids)]
    if not eval_ids and ordered:
        eval_ids = ordered[-1:]
        train_ids = ordered[:-1]
    return {
        "split_source": "deterministic_modulo_split",
        "train_block_ids": train_ids,
        "eval_block_ids": eval_ids,
    }


def _parse_binary_label(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes"}:
        return 1
    if text in {"0", "0.0", "false", "no"}:
        return 0
    return None


def _has_binary_variation(values: Sequence[int] | np.ndarray) -> bool:
    clean = [int(value) for value in values]
    return 0 in clean and 1 in clean


def _metric(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return float("-inf")
    return float(value)


def _label_leakage_risk(label_column: str) -> str:
    if label_column in LEAKAGE_RISK_LABELS:
        return "explicit_label_leakage_risk"
    return "not_flagged"


def _normalize_csvish_values(values: Sequence[str] | str) -> list[str]:
    if isinstance(values, str):
        raw = values.split(",")
    else:
        raw = []
        for value in values:
            raw.extend(str(value).split(","))
    return [str(value).strip() for value in raw if str(value).strip()]


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    label: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 36 analysis is missing {label}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 36 {label} contains a non-mapping row")
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, list):
        return json.dumps(_json_ready(value), sort_keys=True)
    return value


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _round_float(value: float | int) -> float:
    return round(float(value), 10)
