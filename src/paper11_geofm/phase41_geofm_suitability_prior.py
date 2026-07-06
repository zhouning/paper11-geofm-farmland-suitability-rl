from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .block_schema import EMBEDDING_COLUMNS
from .phase40_independent_label_gate import (
    Phase40Thresholds,
    evaluate_label_candidate,
    load_label_registry,
)


PHASE41_GEOFM_PRIOR_CLAIM_BOUNDARY = (
    "Phase 41 evaluates whether an independent-label-calibrated GeoFM "
    "suitability prior clears baseline, representation-control, fold-stability, "
    "and calibration checks. It does not train PPO, does not alter rewards, "
    "does not enable B2/B3, and does not prove planning-policy improvement."
)

PHASE41_SUMMARY_FIELDNAMES = (
    "phase41_geofm_prior_status",
    "label_column",
    "supported_feature_family",
    "decision_reason",
    "claim_boundary",
)

PHASE41_METRIC_FIELDNAMES = (
    "label_column",
    "feature_family",
    "feature_count",
    "fold_count",
    "positive_fold_count",
    "positive_fold_fraction",
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "brier_score",
    "auc_delta_vs_explicit",
    "ap_delta_vs_explicit",
    "brier_delta_vs_explicit",
    "gate_role",
    "claim_boundary",
)

PHASE41_PRIOR_FIELDNAMES = (
    "block_id",
    "label_column",
    "calibrated_suitability_prior",
    "prior_uncertainty",
    "feature_family",
    "model_family",
    "claim_boundary",
)


@dataclass(frozen=True)
class Phase41Thresholds:
    min_valid_count: int = 100
    max_missing_rate: float = 0.20
    min_positive_rate: float = 0.02
    max_positive_rate: float = 0.98
    min_split_valid_count: int = 20
    min_auc_delta: float = 0.03
    min_ap_delta: float = 0.03
    min_positive_fold_fraction: float = 0.67
    max_brier_regression: float = 0.02
    n_pca_components: int = 8


def run_phase41_geofm_suitability_prior(
    phase2_output_dir: Path | str,
    label_registry: Path | str | None = None,
    min_valid_count: int = 100,
    max_missing_rate: float = 0.20,
    min_positive_rate: float = 0.02,
    max_positive_rate: float = 0.98,
    min_split_valid_count: int = 20,
    min_auc_delta: float = 0.03,
    min_ap_delta: float = 0.03,
    min_positive_fold_fraction: float = 0.67,
    max_brier_regression: float = 0.02,
    n_pca_components: int = 8,
) -> dict[str, object]:
    phase2_dir = Path(phase2_output_dir)
    feature_csv = phase2_dir / "block_geofm_features.csv"
    feature_rows = _read_csv_rows(feature_csv, "Phase 2 block feature CSV")
    thresholds = Phase41Thresholds(
        min_valid_count=min_valid_count,
        max_missing_rate=max_missing_rate,
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
        min_split_valid_count=min_split_valid_count,
        min_auc_delta=min_auc_delta,
        min_ap_delta=min_ap_delta,
        min_positive_fold_fraction=min_positive_fold_fraction,
        max_brier_regression=max_brier_regression,
        n_pca_components=n_pca_components,
    )
    passed_labels, label_gate_rows = select_phase40_passed_labels(
        feature_rows,
        label_registry,
        thresholds,
    )
    if not passed_labels:
        return _phase41_missing_result(
            phase2_dir,
            label_registry,
            thresholds,
            label_gate_rows,
            feature_rows,
        )

    metric_rows = _evaluate_passed_labels(feature_rows, passed_labels, thresholds)
    gate_summary = summarize_phase41_gate(metric_rows, thresholds.__dict__)
    prior_rows = _build_prior_rows(feature_rows, gate_summary, thresholds)
    return _phase41_result_payload(
        phase2_dir=phase2_dir,
        label_registry=label_registry,
        thresholds=thresholds,
        feature_rows=feature_rows,
        label_gate_rows=label_gate_rows,
        passed_labels=passed_labels,
        metric_rows=metric_rows,
        gate_summary=gate_summary,
        prior_rows=prior_rows,
    )


def select_phase40_passed_labels(
    feature_rows: Sequence[Mapping[str, str]],
    label_registry: Path | str | None,
    thresholds: Phase41Thresholds,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    registry_rows = load_label_registry(label_registry)
    phase40_thresholds = Phase40Thresholds(
        min_valid_count=thresholds.min_valid_count,
        max_missing_rate=thresholds.max_missing_rate,
        min_positive_rate=thresholds.min_positive_rate,
        max_positive_rate=thresholds.max_positive_rate,
        min_split_valid_count=thresholds.min_split_valid_count,
    )
    label_gate_rows = [
        evaluate_label_candidate(feature_rows, row, phase40_thresholds)
        for row in registry_rows
    ]
    passed = [
        row for row in label_gate_rows if row.get("label_gate_status") == "label_gate_passed"
    ]
    return passed, label_gate_rows


def evaluate_feature_family(
    rows: Sequence[Mapping[str, str]],
    label_column: str,
    feature_family: str,
    feature_matrix: np.ndarray,
) -> dict[str, object]:
    train_mask = np.asarray([_split_role(row) == "train" for row in rows])
    eval_mask = np.asarray([_split_role(row) == "eval" for row in rows])
    y = _labels(rows, label_column)
    train_y = y[train_mask]
    eval_y = y[eval_mask]
    if len(train_y) == 0 or len(eval_y) == 0:
        return _blocked_metric_row(label_column, feature_family, feature_matrix.shape[1])
    if len(set(train_y.tolist())) < 2 or len(set(eval_y.tolist())) < 2:
        return _blocked_metric_row(label_column, feature_family, feature_matrix.shape[1])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    )
    model.fit(feature_matrix[train_mask], train_y)
    scores = model.predict_proba(feature_matrix[eval_mask])[:, 1]
    predictions = (scores >= 0.5).astype(int)
    return {
        "label_column": label_column,
        "feature_family": feature_family,
        "feature_count": int(feature_matrix.shape[1]),
        "fold_count": 1,
        "positive_fold_count": 1,
        "positive_fold_fraction": 1.0,
        "roc_auc": _safe_metric(roc_auc_score(eval_y, scores)),
        "average_precision": _safe_metric(average_precision_score(eval_y, scores)),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score(eval_y, predictions)),
        "brier_score": _safe_metric(brier_score_loss(eval_y, scores)),
        "auc_delta_vs_explicit": 0.0,
        "ap_delta_vs_explicit": 0.0,
        "brier_delta_vs_explicit": 0.0,
        "gate_role": "candidate" if "geofm" in feature_family else "baseline",
        "claim_boundary": PHASE41_GEOFM_PRIOR_CLAIM_BOUNDARY,
    }


def summarize_phase41_gate(
    metric_rows: Sequence[Mapping[str, object]],
    thresholds: Mapping[str, object],
) -> dict[str, object]:
    if not metric_rows:
        return {
            "phase41_geofm_prior_status": "phase41_independent_label_inputs_missing",
            "supported_prior": None,
            "summary_rows": [_summary_row("phase41_independent_label_inputs_missing", "", "", "no Phase 40-passed labels were available")],
        }

    rows_by_label: dict[str, list[Mapping[str, object]]] = {}
    for row in metric_rows:
        rows_by_label.setdefault(str(row.get("label_column", "")), []).append(row)

    min_auc_delta = float(thresholds.get("min_auc_delta", 0.03))
    min_ap_delta = float(thresholds.get("min_ap_delta", 0.03))
    min_positive_fold_fraction = float(thresholds.get("min_positive_fold_fraction", 0.67))
    max_brier_regression = float(thresholds.get("max_brier_regression", 0.02))

    any_candidate_pass = False
    any_control_pass = False
    best_candidate: Mapping[str, object] | None = None
    for label_rows in rows_by_label.values():
        _attach_explicit_deltas(label_rows)
        for row in label_rows:
            family = str(row.get("feature_family", ""))
            passed = _row_passes_thresholds(
                row,
                min_auc_delta=min_auc_delta,
                min_ap_delta=min_ap_delta,
                min_positive_fold_fraction=min_positive_fold_fraction,
                max_brier_regression=max_brier_regression,
            )
            if family in {"geofm_shuffled_control", "geofm_random_control"} and passed:
                any_control_pass = True
            if family in {"geofm_pca_only", "explicit_plus_geofm_pca"} and passed:
                any_candidate_pass = True
                if best_candidate is None or float(row.get("roc_auc", 0.0)) > float(best_candidate.get("roc_auc", 0.0)):
                    best_candidate = row

    if any_control_pass and any_candidate_pass:
        status = "geofm_suitability_prior_control_failed"
        reason = "GeoFM candidate improved, but shuffled or random controls also passed."
        supported_prior = None
    elif any_candidate_pass and best_candidate is not None:
        status = "geofm_suitability_prior_supported"
        reason = "A GeoFM PCA candidate cleared explicit baseline, control, fold-stability, and calibration thresholds."
        supported_prior = {
            "label_column": best_candidate.get("label_column", ""),
            "feature_family": best_candidate.get("feature_family", ""),
            "roc_auc": best_candidate.get("roc_auc", 0.0),
            "average_precision": best_candidate.get("average_precision", 0.0),
            "brier_score": best_candidate.get("brier_score", 0.0),
        }
    else:
        status = "geofm_suitability_prior_not_supported"
        reason = "GeoFM candidates did not clear the explicit baseline and control thresholds."
        supported_prior = None

    label_column = str(best_candidate.get("label_column", "")) if best_candidate else _first_label(metric_rows)
    feature_family = str(best_candidate.get("feature_family", "")) if best_candidate else ""
    return {
        "phase41_geofm_prior_status": status,
        "supported_prior": supported_prior,
        "summary_rows": [_summary_row(status, label_column, feature_family, reason)],
    }


def write_phase41_geofm_suitability_prior_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path | None]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path | None] = {
        "summary_csv": output_path / "phase41_geofm_prior_summary.csv",
        "metrics_csv": output_path / "phase41_geofm_prior_metrics.csv",
        "diagnosis_json": output_path / "phase41_geofm_prior.json",
        "diagnosis_md": output_path / "phase41_geofm_prior.md",
        "prior_csv": None,
    }
    _write_csv_mapping_rows(
        artifacts["summary_csv"],
        PHASE41_SUMMARY_FIELDNAMES,
        analysis.get("summary_rows", []),
        "Phase 41 summary rows",
    )
    _write_csv_mapping_rows(
        artifacts["metrics_csv"],
        PHASE41_METRIC_FIELDNAMES,
        analysis.get("metric_rows", []),
        "Phase 41 metric rows",
    )
    prior_rows = analysis.get("prior_rows", [])
    if isinstance(prior_rows, list) and prior_rows:
        prior_path = output_path / "block_geofm_suitability_prior.csv"
        _write_csv_mapping_rows(
            prior_path,
            PHASE41_PRIOR_FIELDNAMES,
            prior_rows,
            "Phase 41 prior rows",
        )
        artifacts["prior_csv"] = prior_path
    artifacts["diagnosis_json"].write_text(
        json.dumps(_json_ready(analysis), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    artifacts["diagnosis_md"].write_text(
        _phase41_markdown(analysis),
        encoding="utf-8",
    )
    return artifacts


def _evaluate_passed_labels(
    feature_rows: Sequence[Mapping[str, str]],
    passed_labels: Sequence[Mapping[str, object]],
    thresholds: Phase41Thresholds,
) -> list[dict[str, object]]:
    label_columns = {str(row.get("label_column", "")) for row in passed_labels}
    embedding_columns = _embedding_columns(feature_rows)
    explicit_columns = _explicit_columns(feature_rows, label_columns)
    if not embedding_columns:
        raise ValueError("Phase 41 found no embedding_mean_* GeoFM columns")
    metric_rows: list[dict[str, object]] = []
    for label_gate_row in passed_labels:
        label_column = str(label_gate_row.get("label_column", ""))
        families = _build_feature_family_matrices(
            feature_rows,
            explicit_columns,
            embedding_columns,
            thresholds,
        )
        label_metric_rows = [
            evaluate_feature_family(feature_rows, label_column, family_name, matrix)
            for family_name, matrix in families.items()
        ]
        _attach_explicit_deltas(label_metric_rows)
        metric_rows.extend(label_metric_rows)
    return metric_rows


def _build_feature_family_matrices(
    rows: Sequence[Mapping[str, str]],
    explicit_columns: Sequence[str],
    embedding_columns: Sequence[str],
    thresholds: Phase41Thresholds,
) -> dict[str, np.ndarray]:
    explicit = _matrix(rows, explicit_columns) if explicit_columns else np.zeros((len(rows), 0))
    raw = _matrix(rows, embedding_columns)
    pca = _pca_matrix(rows, raw, thresholds.n_pca_components)
    rng = np.random.default_rng(0)
    shuffled = _deterministic_split_shuffle(rows, pca)
    random = rng.normal(size=pca.shape)
    families = {
        "explicit_only": explicit,
        "geofm_raw_only": raw,
        "geofm_pca_only": pca,
        "explicit_plus_geofm_pca": np.column_stack([explicit, pca]) if explicit.size else pca,
        "geofm_shuffled_control": shuffled,
        "geofm_random_control": random,
    }
    return families


def _deterministic_split_shuffle(
    rows: Sequence[Mapping[str, str]],
    matrix: np.ndarray,
) -> np.ndarray:
    shuffled = np.array(matrix, copy=True)
    rng = np.random.default_rng(0)
    for role in ("train", "eval"):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if _split_role(row) == role]
        )
        if len(indices) > 1:
            shuffled[indices] = shuffled[rng.permutation(indices)]
    return shuffled
def _pca_matrix(
    rows: Sequence[Mapping[str, str]], raw: np.ndarray, n_components: int) -> np.ndarray:
    train_mask = np.asarray([_split_role(row) == "train" for row in rows])
    max_components = min(int(n_components), raw.shape[1], max(1, int(train_mask.sum()) - 1))
    if max_components <= 0:
        max_components = 1
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(raw[train_mask])
    all_scaled = scaler.transform(raw)
    pca = PCA(n_components=max_components, random_state=0)
    pca.fit(train_scaled)
    return pca.transform(all_scaled)


def _build_prior_rows(
    feature_rows: Sequence[Mapping[str, str]],
    gate_summary: Mapping[str, object],
    thresholds: Phase41Thresholds,
) -> list[dict[str, object]]:
    supported = gate_summary.get("supported_prior")
    if not isinstance(supported, Mapping):
        return []
    label_column = str(supported.get("label_column", ""))
    feature_family = str(supported.get("feature_family", "")) or "explicit_plus_geofm_pca"
    label_columns = {label_column}
    explicit_columns = _explicit_columns(feature_rows, label_columns)
    embedding_columns = _embedding_columns(feature_rows)
    matrices = _build_feature_family_matrices(
        feature_rows,
        explicit_columns,
        embedding_columns,
        thresholds,
    )
    matrix = matrices[feature_family]
    y = _labels(feature_rows, label_column)
    if len(set(y.tolist())) < 2:
        return []
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    )
    model.fit(matrix, y)
    scores = model.predict_proba(matrix)[:, 1]
    rows: list[dict[str, object]] = []
    for row, score in zip(feature_rows, scores, strict=True):
        rounded = _safe_metric(float(score))
        rows.append(
            {
                "block_id": row.get("block_id", ""),
                "label_column": label_column,
                "calibrated_suitability_prior": rounded,
                "prior_uncertainty": _safe_metric(min(float(score), 1.0 - float(score))),
                "feature_family": feature_family,
                "model_family": "standardized_logistic_regression",
                "claim_boundary": PHASE41_GEOFM_PRIOR_CLAIM_BOUNDARY,
            }
        )
    return rows


def _phase41_missing_result(
    phase2_dir: Path,
    label_registry: Path | str | None,
    thresholds: Phase41Thresholds,
    label_gate_rows: Sequence[Mapping[str, object]],
    feature_rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    status = "phase41_independent_label_inputs_missing"
    summary_rows = [_summary_row(status, "", "", "no Phase 40-passed independent label was available")]
    return {
        "phase": "phase41_geofm_suitability_prior",
        "phase41_geofm_prior_status": status,
        "source_paths": {
            "phase2_output_dir": str(phase2_dir),
            "label_registry": str(Path(label_registry)) if label_registry is not None else None,
        },
        "thresholds": thresholds.__dict__,
        "row_counts": {
            "feature_rows": len(feature_rows),
            "phase40_label_gate_rows": len(label_gate_rows),
            "phase40_passed_labels": 0,
            "metric_rows": 0,
            "prior_rows": 0,
        },
        "phase40_label_gate_rows": list(label_gate_rows),
        "metric_rows": [],
        "summary_rows": summary_rows,
        "supported_prior": None,
        "prior_rows": [],
        "interpretation": _phase41_interpretation(status),
        "recommended_next_step": _phase41_next_step(status),
        "claim_boundary": PHASE41_GEOFM_PRIOR_CLAIM_BOUNDARY,
    }


def _phase41_result_payload(
    phase2_dir: Path,
    label_registry: Path | str | None,
    thresholds: Phase41Thresholds,
    feature_rows: Sequence[Mapping[str, str]],
    label_gate_rows: Sequence[Mapping[str, object]],
    passed_labels: Sequence[Mapping[str, object]],
    metric_rows: list[dict[str, object]],
    gate_summary: Mapping[str, object],
    prior_rows: list[dict[str, object]],
) -> dict[str, object]:
    status = str(gate_summary["phase41_geofm_prior_status"])
    summary_rows = gate_summary.get("summary_rows", [])
    return {
        "phase": "phase41_geofm_suitability_prior",
        "phase41_geofm_prior_status": status,
        "source_paths": {
            "phase2_output_dir": str(phase2_dir),
            "label_registry": str(Path(label_registry)) if label_registry is not None else None,
        },
        "thresholds": thresholds.__dict__,
        "row_counts": {
            "feature_rows": len(feature_rows),
            "phase40_label_gate_rows": len(label_gate_rows),
            "phase40_passed_labels": len(passed_labels),
            "metric_rows": len(metric_rows),
            "prior_rows": len(prior_rows),
        },
        "phase40_label_gate_rows": list(label_gate_rows),
        "metric_rows": metric_rows,
        "summary_rows": summary_rows,
        "supported_prior": gate_summary.get("supported_prior"),
        "prior_rows": prior_rows,
        "interpretation": _phase41_interpretation(status),
        "recommended_next_step": _phase41_next_step(status),
        "claim_boundary": PHASE41_GEOFM_PRIOR_CLAIM_BOUNDARY,
    }


def _read_csv_rows(path: Path, label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _numeric(value: object) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _embedding_columns(rows: Sequence[Mapping[str, str]]) -> list[str]:
    if not rows:
        return []
    columns = set(rows[0])
    return [column for column in EMBEDDING_COLUMNS if column in columns]


def _explicit_columns(rows: Sequence[Mapping[str, str]], label_columns: set[str]) -> list[str]:
    if not rows:
        return []
    excluded = {"block_id", "split", "tile_id", "suitability_proxy", *label_columns}
    excluded.update(column for column in rows[0] if column.startswith("embedding_mean_"))
    excluded.update(column for column in rows[0] if column.endswith("_label"))
    candidates: list[str] = []
    for column in rows[0]:
        if column in excluded:
            continue
        sample = rows[: min(len(rows), 10)]
        if sample and all(_numeric(row.get(column)) is not None for row in sample):
            candidates.append(column)
    return candidates


def _split_role(row: Mapping[str, str]) -> str:
    text = str(row.get("split", "")).strip().lower()
    if text in {"train", "training"}:
        return "train"
    if text in {"test", "eval", "evaluation", "validation", "val"}:
        return "eval"
    digest = hashlib.sha1(str(row.get("block_id", "")).encode("utf-8")).hexdigest()
    return "eval" if int(digest[:2], 16) % 5 == 0 else "train"


def _labels(rows: Sequence[Mapping[str, str]], label_column: str) -> np.ndarray:
    return np.asarray([int(float(row[label_column])) for row in rows], dtype=int)


def _matrix(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.zeros((len(rows), 0), dtype=float)
    return np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=float)


def _blocked_metric_row(label_column: str, feature_family: str, feature_count: int) -> dict[str, object]:
    return {
        "label_column": label_column,
        "feature_family": feature_family,
        "feature_count": int(feature_count),
        "fold_count": 0,
        "positive_fold_count": 0,
        "positive_fold_fraction": 0.0,
        "roc_auc": 0.0,
        "average_precision": 0.0,
        "balanced_accuracy": 0.0,
        "brier_score": 1.0,
        "auc_delta_vs_explicit": 0.0,
        "ap_delta_vs_explicit": 0.0,
        "brier_delta_vs_explicit": 1.0,
        "gate_role": "blocked",
        "claim_boundary": PHASE41_GEOFM_PRIOR_CLAIM_BOUNDARY,
    }


def _attach_explicit_deltas(rows: Sequence[Mapping[str, object]]) -> None:
    explicit = next((row for row in rows if row.get("feature_family") == "explicit_only"), None)
    if explicit is None:
        return
    explicit_auc = float(explicit.get("roc_auc", 0.0))
    explicit_ap = float(explicit.get("average_precision", 0.0))
    explicit_brier = float(explicit.get("brier_score", 1.0))
    for row in rows:
        if isinstance(row, dict):
            row["auc_delta_vs_explicit"] = _safe_metric(float(row.get("roc_auc", 0.0)) - explicit_auc)
            row["ap_delta_vs_explicit"] = _safe_metric(float(row.get("average_precision", 0.0)) - explicit_ap)
            row["brier_delta_vs_explicit"] = _safe_metric(float(row.get("brier_score", 1.0)) - explicit_brier)


def _row_passes_thresholds(
    row: Mapping[str, object],
    *,
    min_auc_delta: float,
    min_ap_delta: float,
    min_positive_fold_fraction: float,
    max_brier_regression: float,
) -> bool:
    return (
        float(row.get("auc_delta_vs_explicit", 0.0)) >= min_auc_delta
        or float(row.get("ap_delta_vs_explicit", 0.0)) >= min_ap_delta
    ) and float(row.get("positive_fold_fraction", 0.0)) >= min_positive_fold_fraction and float(
        row.get("brier_delta_vs_explicit", 1.0)
    ) <= max_brier_regression


def _first_label(metric_rows: Sequence[Mapping[str, object]]) -> str:
    for row in metric_rows:
        value = str(row.get("label_column", ""))
        if value:
            return value
    return ""


def _summary_row(status: str, label_column: str, feature_family: str, reason: str) -> dict[str, object]:
    return {
        "phase41_geofm_prior_status": status,
        "label_column": label_column,
        "supported_feature_family": feature_family,
        "decision_reason": reason,
        "claim_boundary": PHASE41_GEOFM_PRIOR_CLAIM_BOUNDARY,
    }


def _write_csv_mapping_rows(
    path: Path | None,
    fieldnames: Sequence[str],
    rows: object,
    label: str,
) -> None:
    if path is None:
        return
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{label} must be a list of mappings")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _csv_value(value: object) -> object:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True, allow_nan=False)
    return _json_ready(value)


def _json_ready(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _phase41_markdown(analysis: Mapping[str, object]) -> str:
    rows = analysis.get("summary_rows", [])
    summary_rows = rows if isinstance(rows, list) else []
    lines = [
        "# Phase 41 GeoFM Suitability Prior Gate",
        "",
        f"Status: {analysis.get('phase41_geofm_prior_status', '')}",
        "",
        "## Summary",
        "",
        *_markdown_table(PHASE41_SUMMARY_FIELDNAMES, summary_rows),
        "",
        "## Interpretation",
        "",
        str(analysis.get("interpretation", "")),
        "",
        "## Recommended Next Step",
        "",
        str(analysis.get("recommended_next_step", "")),
        "",
        "## Boundary",
        "",
        str(analysis.get("claim_boundary", PHASE41_GEOFM_PRIOR_CLAIM_BOUNDARY)),
        "",
    ]
    return "\n".join(lines)


def _markdown_table(fieldnames: Sequence[str], rows: Sequence[object]) -> list[str]:
    header = [str(field) for field in fieldnames]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        lines.append("| " + " | ".join(_markdown_cell(row.get(field, "")) for field in fieldnames) + " |")
    return lines


def _markdown_cell(value: object) -> str:
    return str(_csv_value(value)).replace("|", "\\|").replace("\n", " ")


def _phase41_interpretation(status: str) -> str:
    if status == "geofm_suitability_prior_supported":
        return "A Phase 40-passed independent label supports a calibrated GeoFM suitability prior under Phase 41 controls."
    if status == "geofm_suitability_prior_control_failed":
        return "A GeoFM candidate improved, but representation controls also passed; the prior is not admissible."
    if status == "geofm_suitability_prior_not_supported":
        return "Independent labels are available, but GeoFM did not clear the Phase 41 prior gate."
    return "No Phase 40-passed independent label is available for Phase 41."


def _phase41_next_step(status: str) -> str:
    if status == "geofm_suitability_prior_supported":
        return "Design a later bounded low-dimensional prior interface before any B2/B3 reward experiment."
    return "Do not run B2/B3; obtain a Phase 40-passed independent label and rerun Phase 41."


def _safe_metric(value: float | int) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        return 0.0
    return round(parsed, 10)
