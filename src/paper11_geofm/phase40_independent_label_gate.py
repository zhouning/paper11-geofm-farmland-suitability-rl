from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path


PHASE40_INDEPENDENT_LABEL_GATE_CLAIM_BOUNDARY = (
    "Phase 40 is a go/no-go independent-label gate. It validates whether a "
    "registered non-leakage label can justify a later Phase 38 proxy-rebuild "
    "rerun. It does not train PPO, does not alter rewards, does not enable "
    "B2/B3, and does not prove suitability or planning-performance improvement."
)

PHASE40_TRAIN_SPLIT_VALUES = {"train", "training"}
PHASE40_EVAL_SPLIT_VALUES = {"test", "eval", "evaluation", "validation", "val"}

PHASE40_REGISTRY_FIELDNAMES = (
    "label_column",
    "label_source",
    "source_type",
    "independence_level",
    "allowed_eval_roles",
    "provenance_note",
    "license_or_access",
    "expected_positive_definition",
)

PHASE40_LABEL_GATE_FIELDNAMES = (
    "label_column",
    "label_gate_status",
    "label_source",
    "source_type",
    "independence_level",
    "allowed_eval_roles",
    "valid_label_count",
    "missing_count",
    "missing_rate",
    "positive_count",
    "negative_count",
    "positive_rate",
    "train_valid_count",
    "eval_valid_count",
    "train_positive_count",
    "train_negative_count",
    "eval_positive_count",
    "eval_negative_count",
    "decision_reason",
    "claim_boundary",
)

INDEPENDENT_SOURCE_TYPES = {
    "external_field_survey",
    "external_agronomic",
    "external_soil",
    "external_irrigation",
    "external_yield",
    "external_high_standard_farmland",
    "external_retention_or_policy",
    "remote_sensing_independent_product",
}

DIAGNOSTIC_SOURCE_TYPES = {
    "diagnostic_internal",
    "dltb_derived",
    "slope_derived",
    "source_metadata",
    "geofm_derived",
    "unknown",
}

VALID_SOURCE_TYPES = INDEPENDENT_SOURCE_TYPES | DIAGNOSTIC_SOURCE_TYPES
PASSING_INDEPENDENCE_LEVELS = {"independent", "partially_independent"}
DIAGNOSTIC_INDEPENDENCE_LEVELS = {"diagnostic_only", "leakage_risk", "unknown"}
VALID_INDEPENDENCE_LEVELS = (
    PASSING_INDEPENDENCE_LEVELS | DIAGNOSTIC_INDEPENDENCE_LEVELS
)


@dataclass(frozen=True)
class Phase40Thresholds:
    min_valid_count: int = 100
    max_missing_rate: float = 0.20
    min_positive_rate: float = 0.02
    max_positive_rate: float = 0.98
    min_split_valid_count: int = 20


def run_phase40_independent_label_gate(
    phase2_output_dir: Path | str,
    label_registry: Path | str | None = None,
    min_valid_count: int = 100,
    max_missing_rate: float = 0.20,
    min_positive_rate: float = 0.02,
    max_positive_rate: float = 0.98,
    min_split_valid_count: int = 20,
) -> dict[str, object]:
    phase2_dir = Path(phase2_output_dir)
    feature_csv = phase2_dir / "block_geofm_features.csv"
    fieldnames, feature_rows = _read_csv_table(
        feature_csv,
        "Phase 2 block feature CSV",
    )
    if "block_id" not in fieldnames:
        raise ValueError(f"Phase 40 Phase 2 CSV is missing block_id: {feature_csv}")
    if "split" not in fieldnames:
        raise ValueError(f"Phase 40 Phase 2 CSV is missing split: {feature_csv}")

    thresholds = Phase40Thresholds(
        min_valid_count=min_valid_count,
        max_missing_rate=max_missing_rate,
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
        min_split_valid_count=min_split_valid_count,
    )
    registry_rows = load_label_registry(label_registry)
    label_gate_rows = [
        evaluate_label_candidate(feature_rows, row, thresholds)
        for row in registry_rows
    ]
    status = summarize_phase40_gate(label_gate_rows)
    return {
        "phase": "phase40_independent_label_gate",
        "phase40_independent_label_gate_status": status,
        "source_paths": {
            "phase2_output_dir": str(phase2_dir),
            "phase2_block_geofm_features_csv": str(feature_csv),
            "label_registry": str(Path(label_registry))
            if label_registry is not None
            else None,
        },
        "thresholds": thresholds.__dict__,
        "row_counts": {
            "feature_rows": len(feature_rows),
            "registry_rows": len(registry_rows),
            "label_gate_rows": len(label_gate_rows),
        },
        "label_gate_rows": label_gate_rows,
        "interpretation": _phase40_interpretation(status),
        "claim_boundary": PHASE40_INDEPENDENT_LABEL_GATE_CLAIM_BOUNDARY,
        "recommended_next_step": _phase40_next_step(status),
    }


def load_label_registry(label_registry: Path | str | None) -> list[dict[str, str]]:
    if label_registry is None:
        return []
    registry_path = Path(label_registry)
    if not registry_path.exists():
        raise ValueError(f"Missing Phase 40 label registry: {registry_path}")
    suffix = registry_path.suffix.lower()
    if suffix == ".csv":
        _, rows = _read_csv_table(registry_path, "Phase 40 label registry CSV")
    elif suffix == ".json":
        rows = _read_json_registry_rows(registry_path)
    else:
        raise ValueError(
            f"Unsupported Phase 40 label registry extension: {registry_path}"
        )
    return [
        _normalize_registry_row(row, registry_path, index)
        for index, row in enumerate(rows, start=2)
    ]


def evaluate_label_candidate(
    feature_rows: Sequence[Mapping[str, str]],
    registry_row: Mapping[str, str],
    thresholds: Phase40Thresholds,
) -> dict[str, object]:
    label_column = str(registry_row.get("label_column", "")).strip()
    if not label_column:
        return _blocked_row(
            registry_row,
            "label_gate_blocked",
            "registry row has blank label_column",
        )
    if not feature_rows or label_column not in feature_rows[0]:
        return _blocked_row(
            registry_row,
            "label_missing",
            f"label column {label_column!r} is missing from the feature table",
        )

    labels: list[int] = []
    train_labels: list[int] = []
    eval_labels: list[int] = []
    missing_count = 0
    positive_definition = (
        str(registry_row.get("expected_positive_definition", "1")).strip() or "1"
    )
    for row in feature_rows:
        parsed = _parse_label(row.get(label_column), positive_definition)
        if parsed is None:
            missing_count += 1
            continue
        labels.append(parsed)
        split_role = _split_role(row.get("split"))
        if split_role == "train":
            train_labels.append(parsed)
        elif split_role == "eval":
            eval_labels.append(parsed)

    valid_count = len(labels)
    positive_count = sum(1 for label in labels if label == 1)
    negative_count = sum(1 for label in labels if label == 0)
    train_positive = sum(1 for label in train_labels if label == 1)
    train_negative = sum(1 for label in train_labels if label == 0)
    eval_positive = sum(1 for label in eval_labels if label == 1)
    eval_negative = sum(1 for label in eval_labels if label == 0)
    missing_rate = missing_count / len(feature_rows) if feature_rows else 1.0
    positive_rate = positive_count / valid_count if valid_count else 0.0
    source_type = str(registry_row.get("source_type", "")).strip()
    independence_level = str(registry_row.get("independence_level", "")).strip()
    allowed_roles = _normalize_csvish_values(
        registry_row.get("allowed_eval_roles", "")
    )
    available_eval_roles = {
        str(row.get("split", "")).strip().lower()
        for row in feature_rows
        if _split_role(row.get("split")) == "eval"
    }

    failure_reasons = _label_failure_reasons(
        valid_count,
        missing_rate,
        positive_rate,
        len(train_labels),
        len(eval_labels),
        source_type,
        independence_level,
        allowed_roles,
        available_eval_roles,
        thresholds,
    )
    if not failure_reasons:
        status = "label_gate_passed"
        reason = (
            "label passed independent source, balance, missingness, and split "
            "coverage gates"
        )
    elif _is_diagnostic_source(source_type, independence_level):
        status = "label_gate_diagnostic_only"
        reason = (
            "label is computable but not independent enough for B2/B3: "
            + "; ".join(failure_reasons)
        )
    else:
        status = "label_gate_blocked"
        reason = "; ".join(failure_reasons)

    return {
        "label_column": label_column,
        "label_gate_status": status,
        "label_source": registry_row.get("label_source", ""),
        "source_type": source_type,
        "independence_level": independence_level,
        "allowed_eval_roles": ",".join(allowed_roles),
        "valid_label_count": valid_count,
        "missing_count": missing_count,
        "missing_rate": _round_float(missing_rate),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": _round_float(positive_rate),
        "train_valid_count": len(train_labels),
        "eval_valid_count": len(eval_labels),
        "train_positive_count": train_positive,
        "train_negative_count": train_negative,
        "eval_positive_count": eval_positive,
        "eval_negative_count": eval_negative,
        "decision_reason": reason,
        "claim_boundary": PHASE40_INDEPENDENT_LABEL_GATE_CLAIM_BOUNDARY,
    }


def summarize_phase40_gate(label_gate_rows: Sequence[Mapping[str, object]]) -> str:
    if not label_gate_rows:
        return "independent_label_inputs_missing"
    statuses = {str(row.get("label_gate_status", "")) for row in label_gate_rows}
    if "label_gate_passed" in statuses:
        return "independent_label_gate_passed"
    if "label_gate_diagnostic_only" in statuses:
        return "independent_label_gate_diagnostic_only"
    return "independent_label_gate_blocked"


def write_phase40_independent_label_gate_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "label_gate_summary_csv": output_path / "phase40_label_gate_summary.csv",
        "diagnosis_json": output_path / "phase40_independent_label_gate.json",
        "diagnosis_md": output_path / "phase40_independent_label_gate.md",
    }
    _write_csv_mapping_rows(
        artifacts["label_gate_summary_csv"],
        PHASE40_LABEL_GATE_FIELDNAMES,
        analysis.get("label_gate_rows", []),
        "Phase 40 label gate rows",
    )
    artifacts["diagnosis_json"].write_text(
        json.dumps(
            _json_ready(analysis),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    artifacts["diagnosis_md"].write_text(_phase40_markdown(analysis), encoding="utf-8")
    return artifacts


def _normalize_registry_row(
    row: Mapping[str, object],
    registry_path: Path,
    row_index: int,
) -> dict[str, str]:
    normalized = {
        field: str(row.get(field, "")).strip()
        for field in PHASE40_REGISTRY_FIELDNAMES
    }
    if not normalized["label_column"]:
        raise ValueError(
            f"Phase 40 label registry row {row_index} has blank label_column: "
            f"{registry_path}"
        )
    source_type = normalized["source_type"]
    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError(
            f"Phase 40 unsupported source_type {source_type!r} for label "
            f"{normalized['label_column']}"
        )
    independence_level = normalized["independence_level"]
    if independence_level not in VALID_INDEPENDENCE_LEVELS:
        raise ValueError(
            f"Phase 40 unsupported independence_level {independence_level!r} "
            f"for label {normalized['label_column']}"
        )
    return normalized


def _label_failure_reasons(
    valid_count: int,
    missing_rate: float,
    positive_rate: float,
    train_count: int,
    eval_count: int,
    source_type: str,
    independence_level: str,
    allowed_roles: Sequence[str],
    available_eval_roles: set[str],
    thresholds: Phase40Thresholds,
) -> list[str]:
    reasons: list[str] = []
    if valid_count < thresholds.min_valid_count:
        reasons.append(
            f"valid_label_count {valid_count} is below min_valid_count "
            f"{thresholds.min_valid_count}"
        )
    if missing_rate > thresholds.max_missing_rate:
        reasons.append(
            f"missing_rate {missing_rate:.10f} exceeds max_missing_rate "
            f"{thresholds.max_missing_rate:.10f}"
        )
    if (
        positive_rate < thresholds.min_positive_rate
        or positive_rate > thresholds.max_positive_rate
    ):
        reasons.append(
            f"positive_rate {positive_rate:.10f} is outside "
            f"[{thresholds.min_positive_rate:.10f}, "
            f"{thresholds.max_positive_rate:.10f}]"
        )
    if train_count < thresholds.min_split_valid_count:
        reasons.append(
            f"train_valid_count {train_count} is below min_split_valid_count "
            f"{thresholds.min_split_valid_count}"
        )
    if eval_count < thresholds.min_split_valid_count:
        reasons.append(
            f"eval_valid_count {eval_count} is below min_split_valid_count "
            f"{thresholds.min_split_valid_count}"
        )
    if source_type not in INDEPENDENT_SOURCE_TYPES:
        reasons.append(f"source_type {source_type!r} is not independent enough")
    if independence_level not in PASSING_INDEPENDENCE_LEVELS:
        reasons.append(
            f"independence_level {independence_level!r} is not independent enough"
        )
    if allowed_roles and not set(allowed_roles).intersection(available_eval_roles):
        reasons.append(
            "allowed_eval_roles do not include an available evaluation split role"
        )
    if not allowed_roles:
        reasons.append("allowed_eval_roles is empty")
    return reasons


def _blocked_row(
    registry_row: Mapping[str, str],
    status: str,
    reason: str,
) -> dict[str, object]:
    return {
        "label_column": str(registry_row.get("label_column", "")).strip(),
        "label_gate_status": status,
        "label_source": registry_row.get("label_source", ""),
        "source_type": registry_row.get("source_type", ""),
        "independence_level": registry_row.get("independence_level", ""),
        "allowed_eval_roles": registry_row.get("allowed_eval_roles", ""),
        "valid_label_count": 0,
        "missing_count": 0,
        "missing_rate": "",
        "positive_count": 0,
        "negative_count": 0,
        "positive_rate": "",
        "train_valid_count": 0,
        "eval_valid_count": 0,
        "train_positive_count": 0,
        "train_negative_count": 0,
        "eval_positive_count": 0,
        "eval_negative_count": 0,
        "decision_reason": reason,
        "claim_boundary": PHASE40_INDEPENDENT_LABEL_GATE_CLAIM_BOUNDARY,
    }


def _read_csv_table(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise ValueError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, [dict(row) for row in reader]


def _read_json_registry_rows(registry_path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Phase 40 label registry JSON is invalid: {registry_path}"
        ) from exc
    if isinstance(payload, list):
        if not all(isinstance(row, Mapping) for row in payload):
            raise ValueError(
                f"Phase 40 label registry JSON rows must be objects: {registry_path}"
            )
        return [dict(row) for row in payload]
    if isinstance(payload, Mapping):
        rows = []
        for label_column, row in payload.items():
            if not isinstance(row, Mapping):
                raise ValueError(
                    "Phase 40 label registry JSON entry "
                    f"{label_column!r} is not an object: {registry_path}"
                )
            normalized = dict(row)
            normalized.setdefault("label_column", str(label_column))
            rows.append(normalized)
        return rows
    raise ValueError(
        f"Phase 40 label registry JSON must be a list or object: {registry_path}"
    )


def _parse_label(value: object, positive_definition: str) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    positive = str(positive_definition).strip().lower()
    if text == "":
        return None
    if text == positive or text in {"1", "1.0", "true", "yes", "y"}:
        return 1
    if text in {"0", "0.0", "false", "no", "n"}:
        return 0
    return None


def _split_role(value: object) -> str | None:
    split_text = str(value or "").strip().lower()
    if split_text in PHASE40_TRAIN_SPLIT_VALUES:
        return "train"
    if split_text in PHASE40_EVAL_SPLIT_VALUES:
        return "eval"
    return None


def _normalize_csvish_values(value: object) -> list[str]:
    return [
        item.strip().lower()
        for item in str(value or "").split(",")
        if item.strip()
    ]


def _is_diagnostic_source(source_type: str, independence_level: str) -> bool:
    return (
        source_type in DIAGNOSTIC_SOURCE_TYPES
        or independence_level in DIAGNOSTIC_INDEPENDENCE_LEVELS
    )


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    label: str,
) -> None:
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{label} must be a list of mappings")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _csv_value(row.get(field, "")) for field in fieldnames}
            )


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


def _phase40_markdown(analysis: Mapping[str, object]) -> str:
    rows = analysis.get("label_gate_rows", [])
    table_rows = rows if isinstance(rows, list) else []
    lines = [
        "# Phase 40 Independent Label Gate",
        "",
        f"Status: {analysis.get('phase40_independent_label_gate_status', '')}",
        "",
        "## Label Gate Summary",
        "",
        *_markdown_table(
            (
                "label_column",
                "label_gate_status",
                "source_type",
                "independence_level",
                "valid_label_count",
                "positive_rate",
                "decision_reason",
            ),
            table_rows,
        ),
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
        str(
            analysis.get(
                "claim_boundary",
                PHASE40_INDEPENDENT_LABEL_GATE_CLAIM_BOUNDARY,
            )
        ),
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
        lines.append(
            "| "
            + " | ".join(_markdown_cell(row.get(field, "")) for field in fieldnames)
            + " |"
        )
    return lines


def _markdown_cell(value: object) -> str:
    return str(_csv_value(value)).replace("|", "\\|").replace("\n", " ")


def _phase40_interpretation(status: str) -> str:
    if status == "independent_label_gate_passed":
        return (
            "At least one registered independent label passed the Phase 40 "
            "admission gate for a later Phase 38 rerun."
        )
    if status == "independent_label_gate_diagnostic_only":
        return (
            "At least one label is computable, but no label is independent "
            "enough to unlock suitability-reward work."
        )
    if status == "independent_label_gate_blocked":
        return "A registry was supplied, but no label passed the independent-label gate."
    return "No usable independent-label registry was supplied."


def _phase40_next_step(status: str) -> str:
    if status == "independent_label_gate_passed":
        return (
            "Rerun Phase 38 proxy rebuild with the accepted independent label "
            "before any B2/B3 smoke."
        )
    return (
        "Do not run B2/B3 or claim suitability reward readiness; obtain an "
        "independent label or frame the manuscript as diagnostic evidence."
    )


def _round_float(value: float | int) -> float:
    return round(float(value), 10)
