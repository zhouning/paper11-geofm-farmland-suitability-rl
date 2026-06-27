from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .block_schema import EMBEDDING_COLUMNS
from .drl_inputs import load_variant_input


PHASE38_PROXY_REBUILD_CLAIM_BOUNDARY = (
    "Phase 38 rebuilds diagnostic proxy classifiers from existing block feature "
    "tables. It does not run PPO, does not alter rewards, does not report trained "
    "policy performance, and does not prove agronomic suitability."
)

DEFAULT_PHASE38_LABEL_COLUMNS = (
    "current_farmland_label",
    "farmland_or_orchard_label",
    "low_slope_farmland_label",
)

LEAKAGE_RISK_LABELS = {
    "current_farmland_label",
    "farmland_or_orchard_label",
    "low_slope_farmland_label",
}

VALID_LABEL_CLASSIFICATIONS = {
    "explicit_label_leakage_risk",
    "candidate_independent_proxy",
    "independent_validation_label",
}

DEFAULT_MODEL_FAMILIES = (
    "logistic_elastic_net",
    "random_forest",
    "hist_gradient_boosting",
)

GEOFM_CANDIDATE_FAMILIES = {
    "raw_geofm_only",
    "explicit_plus_raw_geofm",
    "explicit_plus_normalized_geofm_zscore",
    "explicit_plus_normalized_geofm_zscore_row_l2",
}

CONTROL_FAMILIES = {
    "explicit_plus_random_geofm",
    "explicit_plus_shuffled_geofm",
}


def build_phase38_proxy_rebuild(
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str | None = None,
    normalized_controls_dir: Path | str | None = None,
    label_columns: Sequence[str] | str = DEFAULT_PHASE38_LABEL_COLUMNS,
    label_classifications: Mapping[str, str] | str | None = None,
    model_families: Sequence[str] | str = DEFAULT_MODEL_FAMILIES,
    min_auc_delta: float = 0.02,
    min_ap_delta: float = 0.02,
) -> dict[str, object]:
    phase2_dir = Path(phase2_output_dir)
    block_rows = _read_csv_rows(
        phase2_dir / "block_geofm_features.csv",
        "Phase 2 block feature CSV",
    )
    requested_labels = _normalize_csvish_values(label_columns)
    requested_models = _normalize_csvish_values(model_families)
    _validate_model_families(requested_models)
    classifications = _classifications_for_labels(
        requested_labels,
        _parse_label_classifications(label_classifications),
    )
    available_labels = [
        label for label in requested_labels if _column_available(block_rows, label)
    ]
    if not available_labels:
        raise ValueError("Phase 38 no requested label columns are available")

    feature_families = _build_feature_families(
        phase2_dir,
        Path(phase8_output_dir) if phase8_output_dir is not None else None,
        Path(normalized_controls_dir) if normalized_controls_dir is not None else None,
    )
    if not feature_families:
        raise ValueError("Phase 38 found no usable feature families")

    label_summaries: dict[str, dict[str, object]] = {}
    model_rows: list[dict[str, object]] = []
    rebuilt_proxy_score_rows: list[dict[str, object]] = []
    for label in requested_labels:
        label_classification = classifications[label]
        summary = _label_summary(block_rows, label, label_classification)
        label_summaries[label] = summary
        if not summary["usable"]:
            continue
        labels_by_block = _labels_by_block(block_rows, label)
        split = _split_for_blocks(block_rows, labels_by_block)
        for family in feature_families:
            for model_family in requested_models:
                model_row, score_rows = _evaluate_family(
                    label,
                    label_classification,
                    labels_by_block,
                    split,
                    family,
                    model_family,
                )
                model_rows.append(model_row)
                rebuilt_proxy_score_rows.extend(score_rows)

    status = _phase38_status(
        model_rows,
        min_auc_delta=float(min_auc_delta),
        min_ap_delta=float(min_ap_delta),
    )
    row_counts = {
        "block_rows": len(block_rows),
        "feature_families": len(feature_families),
        "label_summaries": len(label_summaries),
        "model_rows": len(model_rows),
        "rebuilt_proxy_score_rows": len(rebuilt_proxy_score_rows),
    }
    return {
        "phase": "phase38_proxy_rebuild",
        "phase38_proxy_rebuild_status": status,
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
        "label_classifications": classifications,
        "model_families": requested_models,
        "feature_families": [family["feature_family"] for family in feature_families],
        "min_auc_delta": float(min_auc_delta),
        "min_ap_delta": float(min_ap_delta),
        "row_counts": row_counts,
        "label_summaries": label_summaries,
        "label_summary_rows": list(label_summaries.values()),
        "model_rows": model_rows,
        "rebuilt_proxy_score_rows": rebuilt_proxy_score_rows,
        "interpretation": _phase38_interpretation(status),
        "claim_boundary": PHASE38_PROXY_REBUILD_CLAIM_BOUNDARY,
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
        families.append(
            _family_from_variant("suitability_proxy_only", b2, ["suitability_proxy"])
        )
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
        indexes = [
            variant_input.feature_columns.index(column)
            for column in selected_columns
            if column in variant_input.feature_columns
        ]
        matrix = np.asarray(variant_input.state_matrix[:, indexes], dtype=float)
        columns = [str(variant_input.feature_columns[index]) for index in indexes]
    return {
        "feature_family": family_id,
        "block_ids": list(variant_input.block_ids),
        "feature_columns": columns,
        "matrix": matrix,
    }


def _label_summary(
    block_rows: Sequence[Mapping[str, object]],
    label_column: str,
    label_classification: str,
) -> dict[str, object]:
    if not _column_available(block_rows, label_column):
        return {
            "label_column": label_column,
            "label_classification": label_classification,
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
            "claim_boundary": PHASE38_PROXY_REBUILD_CLAIM_BOUNDARY,
        }
    labels = _labels_by_block(block_rows, label_column)
    split = _split_for_blocks(block_rows, labels)
    train_labels = [labels[block_id] for block_id in split["train_block_ids"]]
    eval_labels = [labels[block_id] for block_id in split["eval_block_ids"]]
    positives = sum(1 for label in labels.values() if label == 1)
    negatives = sum(1 for label in labels.values() if label == 0)
    return {
        "label_column": label_column,
        "label_classification": label_classification,
        "available": True,
        "usable": _has_binary_variation(train_labels)
        and _has_binary_variation(eval_labels),
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
        "claim_boundary": PHASE38_PROXY_REBUILD_CLAIM_BOUNDARY,
    }


def _evaluate_family(
    label_column: str,
    label_classification: str,
    labels_by_block: Mapping[str, int],
    split: Mapping[str, object],
    family: Mapping[str, object],
    model_family: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
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
        "label_classification": label_classification,
        "model_family": model_family,
        "feature_family": family["feature_family"],
        "feature_count": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "train_count": len(train_ids),
        "eval_count": len(eval_ids),
        "positive_rate_eval": _round_float(float(np.mean(eval_y))) if eval_y.size else "",
        "claim_boundary": PHASE38_PROXY_REBUILD_CLAIM_BOUNDARY,
    }
    if (
        not _has_binary_variation(train_y)
        or not _has_binary_variation(eval_y)
        or matrix.ndim != 2
        or matrix.shape[1] == 0
        or model_family not in DEFAULT_MODEL_FAMILIES
    ):
        return (
            {
                **base_row,
                "validation_status": "insufficient_label_variation",
                "roc_auc": "",
                "average_precision": "",
                "balanced_accuracy": "",
                "accuracy": "",
                "calibration_bins": [],
                "top_diagnostics": [],
            },
            [],
        )

    train_x = matrix[[block_index[block_id] for block_id in train_ids], :]
    eval_x = matrix[[block_index[block_id] for block_id in eval_ids], :]
    model = _model_for_family(model_family)
    model.fit(train_x, train_y)
    eval_probabilities = _positive_probabilities(model, eval_x)
    predictions = (eval_probabilities >= 0.5).astype(int)
    model_row = {
        **base_row,
        "validation_status": "evaluated",
        "roc_auc": _round_float(roc_auc_score(eval_y, eval_probabilities)),
        "average_precision": _round_float(
            average_precision_score(eval_y, eval_probabilities)
        ),
        "balanced_accuracy": _round_float(
            balanced_accuracy_score(eval_y, predictions)
        ),
        "accuracy": _round_float(accuracy_score(eval_y, predictions)),
        "calibration_bins": _calibration_bins(eval_probabilities, eval_y),
        "top_diagnostics": _top_diagnostics(model, model_family, family),
    }

    all_indexes = [
        block_index[block_id]
        for block_id in block_ids
        if block_id in labels_by_block
    ]
    all_ids = [block_id for block_id in block_ids if block_id in labels_by_block]
    all_probabilities = _positive_probabilities(model, matrix[all_indexes, :])
    split_role_by_block = {
        **{str(block_id): "train" for block_id in split["train_block_ids"]},
        **{str(block_id): "eval" for block_id in split["eval_block_ids"]},
    }
    score_rows = [
        {
            "label_column": label_column,
            "label_classification": label_classification,
            "model_family": model_family,
            "feature_family": str(family["feature_family"]),
            "block_id": block_id,
            "split_role": split_role_by_block.get(block_id, ""),
            "label_value": labels_by_block[block_id],
            "rebuilt_proxy_score": _round_float(float(probability)),
        }
        for block_id, probability in zip(all_ids, all_probabilities, strict=True)
    ]
    return model_row, score_rows


def _validate_model_families(model_families: Sequence[str]) -> None:
    invalid = sorted(
        {
            model_family
            for model_family in model_families
            if model_family not in DEFAULT_MODEL_FAMILIES
        }
    )
    if invalid:
        raise ValueError(f"Phase 38 unknown model families: {invalid}")


def _model_for_family(model_family: str):
    if model_family == "logistic_elastic_net":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                class_weight="balanced",
                max_iter=5000,
                random_state=0,
            ),
        )
    if model_family == "random_forest":
        return RandomForestClassifier(
            n_estimators=80,
            max_depth=6,
            class_weight="balanced",
            random_state=0,
            n_jobs=1,
        )
    if model_family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.05,
            random_state=0,
        )
    raise ValueError(f"Unknown Phase 38 model family: {model_family}")


def _calibration_bins(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    bin_count: int = 5,
) -> list[dict[str, object]]:
    probability_array = np.asarray(probabilities, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    bins: list[dict[str, object]] = []
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    for index in range(bin_count):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == bin_count - 1:
            mask = (probability_array >= lower) & (probability_array <= upper)
        else:
            mask = (probability_array >= lower) & (probability_array < upper)
        if not np.any(mask):
            bins.append(
                {
                    "bin": index,
                    "count": 0,
                    "mean_probability": "",
                    "positive_rate": "",
                }
            )
            continue
        bins.append(
            {
                "bin": index,
                "count": int(np.sum(mask)),
                "mean_probability": _round_float(
                    float(np.mean(probability_array[mask]))
                ),
                "positive_rate": _round_float(float(np.mean(label_array[mask]))),
            }
        )
    return bins


def _top_diagnostics(
    model,
    model_family: str,
    family: Mapping[str, object],
    limit: int = 5,
) -> list[dict[str, object]]:
    feature_columns = [str(column) for column in family["feature_columns"]]
    if model_family == "logistic_elastic_net":
        estimator = model.named_steps.get("logisticregression")
        if estimator is None or not hasattr(estimator, "coef_"):
            return []
        values = np.asarray(estimator.coef_[0], dtype=float)
        label = "coefficient"
    elif model_family == "random_forest" and hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        label = "importance"
    else:
        return []
    indexes = np.argsort(np.abs(values))[::-1][:limit]
    return [
        {
            "feature": feature_columns[int(index)]
            if int(index) < len(feature_columns)
            else str(index),
            label: _round_float(values[int(index)]),
        }
        for index in indexes
    ]


def _positive_probabilities(model, matrix: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(matrix)
    if probabilities.ndim == 1:
        return np.asarray(probabilities, dtype=float)
    return np.asarray(probabilities[:, 1], dtype=float)


def _phase38_status(
    model_rows: Sequence[Mapping[str, object]],
    min_auc_delta: float,
    min_ap_delta: float,
) -> str:
    evaluated = [
        row for row in model_rows if str(row.get("validation_status")) == "evaluated"
    ]
    if not evaluated:
        return "proxy_rebuild_inputs_insufficient"
    non_leakage_rows = [
        row
        for row in evaluated
        if str(row.get("label_classification")) != "explicit_label_leakage_risk"
    ]
    if not non_leakage_rows:
        return "proxy_rebuild_diagnostic_only"

    label_model_pairs = sorted(
        {
            (str(row.get("label_column")), str(row.get("model_family")))
            for row in non_leakage_rows
        }
    )
    for label_column, model_family in label_model_pairs:
        rows = {
            str(row["feature_family"]): row
            for row in evaluated
            if str(row.get("label_column")) == label_column
            and str(row.get("model_family")) == model_family
        }
        explicit = rows.get("explicit_only")
        if explicit is None:
            continue
        missing_controls = [
            family for family in CONTROL_FAMILIES if family not in rows
        ]
        if missing_controls:
            continue
        explicit_auc = _metric(explicit, "roc_auc")
        explicit_ap = _metric(explicit, "average_precision")
        control_auc = max(
            _metric(rows[family], "roc_auc") for family in CONTROL_FAMILIES
        )
        control_ap = max(
            _metric(rows[family], "average_precision")
            for family in CONTROL_FAMILIES
        )
        for family_id in GEOFM_CANDIDATE_FAMILIES:
            candidate = rows.get(family_id)
            if candidate is None:
                continue
            if (
                _metric(candidate, "roc_auc")
                >= max(explicit_auc, control_auc) + min_auc_delta
                and _metric(candidate, "average_precision")
                >= max(explicit_ap, control_ap) + min_ap_delta
            ):
                return "proxy_rebuild_supported_for_bounded_b2_b3_smoke"
    return "proxy_rebuild_diagnostic_only"


def _phase38_interpretation(status: str) -> str:
    if status == "proxy_rebuild_supported_for_bounded_b2_b3_smoke":
        return (
            "At least one non-leakage label has a GeoFM-derived rebuilt proxy "
            "that improves over explicit-only and random or shuffled controls by "
            "the configured ROC AUC and AP deltas."
        )
    if status == "proxy_rebuild_inputs_insufficient":
        return "No requested label/model/family combination had usable binary train and evaluation coverage."
    return (
        "Phase 38 remains diagnostic only: either evaluated labels were explicit "
        "leakage risks or GeoFM-derived rebuilt proxies did not clear the control "
        "thresholds."
    )


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


def _normalize_csvish_values(values: Sequence[str] | str) -> list[str]:
    if isinstance(values, str):
        raw = values.split(",")
    else:
        raw = []
        for value in values:
            raw.extend(str(value).split(","))
    return [str(value).strip() for value in raw if str(value).strip()]


def _parse_label_classifications(
    label_classifications: Mapping[str, str] | str | None,
) -> dict[str, str]:
    if label_classifications is None:
        return {}
    if isinstance(label_classifications, str):
        parsed: dict[str, str] = {}
        for item in _normalize_csvish_values(label_classifications):
            if ":" not in item:
                raise ValueError(f"Invalid Phase 38 label classification: {item}")
            label, classification = item.split(":", 1)
            parsed[label.strip()] = classification.strip()
    else:
        parsed = {
            str(label).strip(): str(classification).strip()
            for label, classification in label_classifications.items()
        }
    invalid = sorted(
        {
            classification
            for classification in parsed.values()
            if classification not in VALID_LABEL_CLASSIFICATIONS
        }
    )
    if invalid:
        raise ValueError(f"Invalid Phase 38 label classification names: {invalid}")
    return {label: classification for label, classification in parsed.items() if label}


def _classifications_for_labels(
    labels: Sequence[str],
    overrides: Mapping[str, str],
) -> dict[str, str]:
    classifications: dict[str, str] = {}
    for label in labels:
        if label in overrides:
            classifications[label] = overrides[label]
        elif label in LEAKAGE_RISK_LABELS:
            classifications[label] = "explicit_label_leakage_risk"
        else:
            classifications[label] = "explicit_label_leakage_risk"
    return classifications


def _metric(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return float("-inf")
    return float(value)


def _round_float(value: float | int) -> float:
    return round(float(value), 10)
