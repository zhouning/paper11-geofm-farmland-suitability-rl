from __future__ import annotations

from collections.abc import Sequence
import csv
from pathlib import Path


PHASE39_INDEPENDENT_LABEL_AUDIT_CLAIM_BOUNDARY = (
    "Phase 39 audits candidate independent label inputs from existing Phase 2 "
    "block tables and optional external label CSVs. It does not train PPO, does "
    "not alter rewards, does not run Phase 38, and does not prove agronomic "
    "suitability."
)

DEFAULT_PHASE39_LABEL_COLUMNS = (
    "current_farmland_label",
    "farmland_or_orchard_label",
    "low_slope_farmland_label",
    "source_bsm",
    "source_category",
    "source_dlbm",
    "source_dlmc",
)

BUILT_IN_PROVENANCE_BY_LABEL = {
    "current_farmland_label": "explicit_label_leakage_risk",
    "farmland_or_orchard_label": "explicit_label_leakage_risk",
    "low_slope_farmland_label": "explicit_label_leakage_risk",
    "source_bsm": "source_field_leakage_risk",
    "source_category": "source_field_leakage_risk",
    "source_dlbm": "source_field_leakage_risk",
    "source_dlmc": "source_field_leakage_risk",
}

VALID_PHASE39_PROVENANCE_CLASSES = {
    "explicit_label_leakage_risk",
    "source_field_leakage_risk",
    "candidate_independent_proxy",
    "independent_validation_label",
    "unclassified",
}

INDEPENDENT_PHASE39_PROVENANCE_CLASSES = {
    "candidate_independent_proxy",
    "independent_validation_label",
}

PHASE39_TRAIN_SPLIT_VALUES = {"train", "training"}

PHASE39_EVAL_SPLIT_VALUES = {
    "test",
    "eval",
    "evaluation",
    "validation",
    "val",
    "valid",
}

PHASE39_REGISTRY_FIELDNAMES = (
    "label_column",
    "source_path",
    "provenance_class",
    "description",
    "external_source_name",
    "independence_rationale",
    "allowed_for_phase38_rerun",
)


def build_phase39_independent_label_audit(
    phase2_output_dir: Path | str,
    external_label_csvs: Sequence[Path | str] | Path | str | None = None,
    label_registry: Path | str | None = None,
    label_columns: Sequence[str] | str = DEFAULT_PHASE39_LABEL_COLUMNS,
) -> dict[str, object]:
    phase2_dir = Path(phase2_output_dir)
    phase2_csv = phase2_dir / "block_geofm_features.csv"
    phase2_fieldnames, block_rows = _read_csv_table(
        phase2_csv,
        "Phase 2 block feature CSV",
    )
    if "block_id" not in phase2_fieldnames:
        raise ValueError(f"Phase 39 Phase 2 CSV is missing block_id: {phase2_csv}")

    requested_labels = _normalize_csvish_values(label_columns)
    if not requested_labels:
        raise ValueError("Phase 39 requires at least one requested label column")

    external_paths = _normalize_paths(external_label_csvs)
    external_sources_by_label = _join_external_label_csvs(
        block_rows,
        external_paths,
    )
    registry_entries = _read_label_registry(label_registry)

    available_columns = set(phase2_fieldnames)
    available_columns.update(external_sources_by_label.keys())
    for row in block_rows:
        available_columns.update(row.keys())
    missing_labels = [
        label for label in requested_labels if label not in available_columns
    ]
    if missing_labels:
        raise ValueError(f"Phase 39 requested missing label columns: {missing_labels}")

    label_inventory_rows: list[dict[str, object]] = []
    label_readiness: dict[str, dict[str, object]] = {}
    registry_template_rows: list[dict[str, object]] = []
    for label_column in requested_labels:
        registry_entry = registry_entries.get(label_column)
        provenance_class = _provenance_for_label(label_column, registry_entry)
        readiness_row = _label_readiness_row(
            block_rows,
            label_column,
            provenance_class,
            registry_entry,
            external_sources_by_label,
        )
        label_readiness[label_column] = readiness_row
        label_inventory_rows.append(
            _label_inventory_row(
                label_column,
                provenance_class,
                registry_entry,
                external_sources_by_label,
            )
        )
        registry_template_rows.append(
            _registry_template_row(label_column, provenance_class, registry_entry)
        )

    label_readiness_rows = list(label_readiness.values())
    status = _phase39_status(label_readiness_rows)
    return {
        "phase": "phase39_independent_label_audit",
        "phase39_independent_label_audit_status": status,
        "source_paths": {
            "phase2_output_dir": str(phase2_dir),
            "phase2_block_geofm_features_csv": str(phase2_csv),
            "external_label_csvs": [str(path) for path in external_paths],
            "label_registry": str(Path(label_registry))
            if label_registry is not None
            else None,
        },
        "label_columns_requested": requested_labels,
        "row_counts": {
            "block_rows": len(block_rows),
            "external_label_csvs": len(external_paths),
            "external_label_columns": len(external_sources_by_label),
            "registry_rows": len(registry_entries),
            "label_inventory_rows": len(label_inventory_rows),
            "label_readiness_rows": len(label_readiness_rows),
        },
        "label_inventory_rows": label_inventory_rows,
        "label_readiness_rows": label_readiness_rows,
        "label_readiness": label_readiness,
        "registry_template_rows": registry_template_rows,
        "interpretation": _phase39_interpretation(status),
        "claim_boundary": PHASE39_INDEPENDENT_LABEL_AUDIT_CLAIM_BOUNDARY,
    }


def _read_csv_table(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise ValueError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, [dict(row) for row in reader]


def _normalize_paths(
    paths: Sequence[Path | str] | Path | str | None,
) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _join_external_label_csvs(
    block_rows: list[dict[str, str]],
    external_paths: Sequence[Path],
) -> dict[str, list[str]]:
    sources_by_label: dict[str, list[str]] = {}
    block_rows_by_id = {
        str(row.get("block_id", "")).strip(): row
        for row in block_rows
        if str(row.get("block_id", "")).strip()
    }
    for external_path in external_paths:
        fieldnames, external_rows = _read_csv_table(
            external_path,
            "external label CSV",
        )
        if "block_id" not in fieldnames:
            raise ValueError(
                f"Phase 39 external label CSV is missing block_id: {external_path}"
            )
        label_columns = [column for column in fieldnames if column != "block_id"]
        for label_column in label_columns:
            sources_by_label.setdefault(label_column, []).append(str(external_path))

        seen_block_ids: set[str] = set()
        for external_row in external_rows:
            block_id = str(external_row.get("block_id", "")).strip()
            if not block_id:
                raise ValueError(
                    f"Phase 39 external label CSV contains a blank block_id: {external_path}"
                )
            if block_id in seen_block_ids:
                raise ValueError(
                    f"Phase 39 external label CSV has duplicate block_id {block_id}: "
                    f"{external_path}"
                )
            seen_block_ids.add(block_id)
            block_row = block_rows_by_id.get(block_id)
            if block_row is None:
                continue
            for label_column in label_columns:
                block_row[label_column] = str(external_row.get(label_column, ""))
    return sources_by_label


def _read_label_registry(
    label_registry: Path | str | None,
) -> dict[str, dict[str, str]]:
    if label_registry is None:
        return {}
    registry_path = Path(label_registry)
    _, registry_rows = _read_csv_table(registry_path, "Phase 39 label registry CSV")
    entries: dict[str, dict[str, str]] = {}
    for row_index, row in enumerate(registry_rows, start=2):
        label_column = str(row.get("label_column", "")).strip()
        if not label_column:
            raise ValueError(
                "Phase 39 label registry row "
                f"{row_index} has blank or missing label_column: {registry_path}"
            )
        provenance_class = str(row.get("provenance_class", "")).strip()
        if provenance_class not in VALID_PHASE39_PROVENANCE_CLASSES:
            raise ValueError(
                "Phase 39 unsupported provenance class "
                f"{provenance_class!r} for label {label_column}"
            )
        if label_column in entries:
            raise ValueError(
                f"Phase 39 label registry has duplicate label_column {label_column}"
            )
        entries[label_column] = {
            field: str(row.get(field, "")).strip()
            for field in PHASE39_REGISTRY_FIELDNAMES
        }
    return entries


def _provenance_for_label(
    label_column: str,
    registry_entry: dict[str, str] | None,
) -> str:
    built_in = BUILT_IN_PROVENANCE_BY_LABEL.get(label_column)
    if built_in is not None:
        return built_in
    if registry_entry is not None:
        return str(registry_entry.get("provenance_class", "")).strip()
    return "unclassified"


def _label_readiness_row(
    block_rows: Sequence[dict[str, str]],
    label_column: str,
    provenance_class: str,
    registry_entry: dict[str, str] | None,
    external_sources_by_label: dict[str, list[str]],
) -> dict[str, object]:
    labels_by_block: dict[str, int] = {}
    split_by_block: dict[str, str | None] = {}
    for row in block_rows:
        block_id = str(row.get("block_id", "")).strip()
        if not block_id:
            continue
        label = _parse_binary_label(row.get(label_column))
        if label is None:
            continue
        labels_by_block[block_id] = label
        split_by_block[block_id] = _split_role(row.get("split"))

    train_labels = [
        labels_by_block[block_id]
        for block_id, split_role in split_by_block.items()
        if split_role == "train"
    ]
    eval_labels = [
        labels_by_block[block_id]
        for block_id, split_role in split_by_block.items()
        if split_role == "eval"
    ]
    positive_count = sum(1 for label in labels_by_block.values() if label == 1)
    negative_count = sum(1 for label in labels_by_block.values() if label == 0)
    train_positive_count = sum(1 for label in train_labels if label == 1)
    train_negative_count = sum(1 for label in train_labels if label == 0)
    eval_positive_count = sum(1 for label in eval_labels if label == 1)
    eval_negative_count = sum(1 for label in eval_labels if label == 0)
    join_missing_count = (
        _join_missing_count(block_rows, label_column)
        if label_column in external_sources_by_label
        else 0
    )
    usable = (
        train_positive_count > 0
        and train_negative_count > 0
        and eval_positive_count > 0
        and eval_negative_count > 0
    )
    registry_entry_present = registry_entry is not None
    registry_allows_rerun = (
        _parse_bool(registry_entry.get("allowed_for_phase38_rerun"))
        if registry_entry is not None
        else False
    )
    allowed_for_phase38_rerun = (
        usable
        and registry_entry_present
        and registry_allows_rerun
        and provenance_class in INDEPENDENT_PHASE39_PROVENANCE_CLASSES
        and join_missing_count == 0
    )
    return {
        "label_column": label_column,
        "provenance_class": provenance_class,
        "registry_entry_present": registry_entry_present,
        "source_path": _source_path_for_label(
            label_column,
            registry_entry,
            external_sources_by_label,
        ),
        "description": _registry_value(registry_entry, "description"),
        "external_source_name": _registry_value(
            registry_entry,
            "external_source_name",
        ),
        "independence_rationale": _registry_value(
            registry_entry,
            "independence_rationale",
        ),
        "registry_allowed_for_phase38_rerun": registry_allows_rerun,
        "join_missing_count": join_missing_count,
        "usable": usable,
        "valid_label_count": len(labels_by_block),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": _round_float(positive_count / len(labels_by_block))
        if labels_by_block
        else "",
        "train_count": len(train_labels),
        "eval_count": len(eval_labels),
        "train_positive_count": train_positive_count,
        "train_negative_count": train_negative_count,
        "eval_positive_count": eval_positive_count,
        "eval_negative_count": eval_negative_count,
        "split_source": "split_column",
        "allowed_for_phase38_rerun": allowed_for_phase38_rerun,
        "decision_reason": _decision_reason(
            usable,
            len(labels_by_block),
            train_positive_count,
            train_negative_count,
            eval_positive_count,
            eval_negative_count,
            provenance_class,
            registry_entry_present,
            registry_allows_rerun,
            join_missing_count,
        ),
        "claim_boundary": PHASE39_INDEPENDENT_LABEL_AUDIT_CLAIM_BOUNDARY,
    }


def _label_inventory_row(
    label_column: str,
    provenance_class: str,
    registry_entry: dict[str, str] | None,
    external_sources_by_label: dict[str, list[str]],
) -> dict[str, object]:
    return {
        "label_column": label_column,
        "provenance_class": provenance_class,
        "registry_entry_present": registry_entry is not None,
        "source_path": _source_path_for_label(
            label_column,
            registry_entry,
            external_sources_by_label,
        ),
        "description": _registry_value(registry_entry, "description"),
        "external_source_name": _registry_value(
            registry_entry,
            "external_source_name",
        ),
        "independence_rationale": _registry_value(
            registry_entry,
            "independence_rationale",
        ),
    }


def _registry_template_row(
    label_column: str,
    provenance_class: str,
    registry_entry: dict[str, str] | None,
) -> dict[str, object]:
    return {
        "label_column": label_column,
        "source_path": _registry_value(registry_entry, "source_path"),
        "provenance_class": provenance_class,
        "description": _registry_value(registry_entry, "description"),
        "external_source_name": _registry_value(
            registry_entry,
            "external_source_name",
        ),
        "independence_rationale": _registry_value(
            registry_entry,
            "independence_rationale",
        ),
        "allowed_for_phase38_rerun": _parse_bool(
            registry_entry.get("allowed_for_phase38_rerun")
        )
        if registry_entry is not None
        else False,
    }


def _source_path_for_label(
    label_column: str,
    registry_entry: dict[str, str] | None,
    external_sources_by_label: dict[str, list[str]],
) -> str:
    registry_source = _registry_value(registry_entry, "source_path")
    if registry_source:
        return registry_source
    sources = external_sources_by_label.get(label_column, [])
    if sources:
        return ",".join(sources)
    return "phase2:block_geofm_features.csv"


def _registry_value(registry_entry: dict[str, str] | None, key: str) -> str:
    if registry_entry is None:
        return ""
    return str(registry_entry.get(key, "")).strip()


def _split_role(value: object) -> str | None:
    split_text = str(value or "").strip().lower()
    if split_text in PHASE39_TRAIN_SPLIT_VALUES:
        return "train"
    if split_text in PHASE39_EVAL_SPLIT_VALUES:
        return "eval"
    return None


def _join_missing_count(
    block_rows: Sequence[dict[str, str]],
    label_column: str,
) -> int:
    return sum(
        1
        for row in block_rows
        if str(row.get(label_column, "")).strip() == ""
    )


def _parse_binary_label(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes"}:
        return 1
    if text in {"0", "0.0", "false", "no"}:
        return 0
    return None


def _parse_bool(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "1.0", "true", "yes", "y"}


def _decision_reason(
    usable: bool,
    valid_label_count: int,
    train_positive_count: int,
    train_negative_count: int,
    eval_positive_count: int,
    eval_negative_count: int,
    provenance_class: str,
    registry_entry_present: bool,
    registry_allows_rerun: bool,
    join_missing_count: int,
) -> str:
    if join_missing_count > 0:
        return f"label has {join_missing_count} missing joined labels from external sources"
    if usable and provenance_class in INDEPENDENT_PHASE39_PROVENANCE_CLASSES:
        if registry_entry_present and registry_allows_rerun:
            return (
                "usable registered independent label with both positive and "
                "negative labels in train and evaluation subsets"
            )
        if registry_entry_present:
            return "usable independent label, but registry does not allow Phase 38 rerun"
        return "usable independent label, but no registry entry is present"
    if usable and provenance_class == "unclassified":
        return "usable unclassified label needs provenance review before Phase 38 rerun"
    if usable:
        return "usable binary label, but provenance class is not eligible for Phase 38 rerun"
    if valid_label_count == 0:
        return "no binary labels were found for the requested label column"
    return (
        "label does not have both positive and negative labels in train and "
        "evaluation subsets "
        f"(train_pos={train_positive_count}, train_neg={train_negative_count}, "
        f"eval_pos={eval_positive_count}, eval_neg={eval_negative_count})"
    )


def _phase39_status(label_readiness_rows: Sequence[dict[str, object]]) -> str:
    if any(bool(row.get("allowed_for_phase38_rerun")) for row in label_readiness_rows):
        return "independent_labels_ready_for_phase38_rerun"
    if any(
        str(row.get("provenance_class")) in INDEPENDENT_PHASE39_PROVENANCE_CLASSES
        and (
            not bool(row.get("usable"))
            or int(row.get("join_missing_count") or 0) > 0
        )
        for row in label_readiness_rows
    ):
        return "independent_label_inputs_insufficient"
    if any(
        bool(row.get("usable"))
        and str(row.get("provenance_class"))
        in {"unclassified", *INDEPENDENT_PHASE39_PROVENANCE_CLASSES}
        and not bool(row.get("allowed_for_phase38_rerun"))
        for row in label_readiness_rows
    ):
        return "candidate_proxy_labels_need_review"
    return "independent_label_inputs_missing"


def _phase39_interpretation(status: str) -> str:
    if status == "independent_labels_ready_for_phase38_rerun":
        return (
            "At least one registered independent or candidate proxy label has "
            "binary train/evaluation coverage and can be supplied to Phase 38."
        )
    if status == "candidate_proxy_labels_need_review":
        return (
            "At least one usable non-leakage label exists, but it needs registry "
            "review or an explicit registry allowance before Phase 38 can use it."
        )
    if status == "independent_label_inputs_insufficient":
        return (
            "A registered independent or candidate proxy label exists, but it "
            "does not have both positive and negative examples in train and "
            "evaluation subsets."
        )
    return (
        "No requested label is a usable registered independent input for a "
        "Phase 38 rerun."
    )


def _normalize_csvish_values(values: Sequence[str] | str) -> list[str]:
    if isinstance(values, str):
        raw = values.split(",")
    else:
        raw = []
        for value in values:
            raw.extend(str(value).split(","))
    return [str(value).strip() for value in raw if str(value).strip()]


def _round_float(value: float | int) -> float:
    return round(float(value), 10)
