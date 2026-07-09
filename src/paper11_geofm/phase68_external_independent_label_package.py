from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import math
from pathlib import Path

from paper11_geofm.phase40_independent_label_gate import Phase40Thresholds


PHASE68_EXTERNAL_LABEL_PACKAGE_CLAIM_BOUNDARY = (
    "Phase 68 builds and audits an external independent-label package before "
    "Phase 40/41 or reward-redesign work. It does not train PPO, does not alter "
    "rewards, does not enable B2/B3, does not prove suitability, and does not "
    "justify formal submission-level claims."
)

PHASE68_REGISTRY_FIELDNAMES = (
    "label_column",
    "label_source",
    "source_type",
    "independence_level",
    "allowed_eval_roles",
    "provenance_note",
    "license_or_access",
    "expected_positive_definition",
    "source_owner",
    "collection_date_or_period",
    "spatial_join_method",
    "original_unit",
    "label_scale",
    "missing_value_policy",
    "known_overlap_with_dltb_slope_or_source_metadata",
    "contact_or_access_note",
)

PHASE68_EXTERNAL_LABEL_TEMPLATE_FIELDNAMES = (
    "block_id",
    "external_independent_label",
    "label_missing_reason",
    "source_record_id",
)

PHASE68_LABEL_PREFLIGHT_FIELDNAMES = (
    "label_column",
    "label_preflight_status",
    "label_source",
    "source_type",
    "independence_level",
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
    "unjoined_external_count",
    "decision_reason",
    "claim_boundary",
)

PHASE68_PACKAGE_SUMMARY_FIELDNAMES = (
    "phase68_status",
    "phase2_block_rows",
    "external_label_csv_count",
    "registry_rows",
    "ready_label_count",
    "invalid_label_count",
    "diagnostic_label_count",
    "recommended_next_step",
    "claim_boundary",
)


def build_phase68_external_independent_label_package(
    phase2_output_dir: Path | str,
    external_label_csvs: Sequence[Path | str] | Path | str | None = None,
    label_registry: Path | str | None = None,
    validation_mode: bool = False,
    min_valid_count: int = 100,
    max_missing_rate: float = 0.20,
    min_positive_rate: float = 0.02,
    max_positive_rate: float = 0.98,
    min_split_valid_count: int = 20,
) -> dict[str, object]:
    phase2_dir = Path(phase2_output_dir)
    phase2_csv = phase2_dir / "block_geofm_features.csv"
    phase2_fieldnames, phase2_rows = _read_csv_table(
        phase2_csv,
        "Phase 2 block feature CSV",
    )
    if "block_id" not in phase2_fieldnames:
        raise ValueError(f"Phase 68 Phase 2 CSV is missing block_id: {phase2_csv}")
    if "split" not in phase2_fieldnames:
        raise ValueError(f"Phase 68 Phase 2 CSV is missing split: {phase2_csv}")

    external_paths = _normalize_paths(external_label_csvs)
    thresholds = Phase40Thresholds(
        min_valid_count=min_valid_count,
        max_missing_rate=max_missing_rate,
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
        min_split_valid_count=min_split_valid_count,
    )
    template_rows = _external_label_template_rows(phase2_rows)
    registry_template_rows = [_default_registry_template_row()]

    if not external_paths and label_registry is None:
        status = (
            "external_label_inputs_missing"
            if validation_mode
            else "external_label_package_ready"
        )
        label_preflight_rows: list[dict[str, object]] = []
        registry_rows: list[dict[str, str]] = []
    elif validation_mode and (not external_paths or label_registry is None):
        status = "external_label_inputs_missing"
        label_preflight_rows = []
        registry_rows = []
    else:
        _external_values_by_label, _sources_by_label, _unjoined_by_label = (
            _load_external_label_csvs(
                phase2_rows,
                external_paths,
            )
        )
        registry_rows = _load_phase68_registry(label_registry)
        label_preflight_rows = []
        status = "external_label_inputs_invalid"

    summary_rows = [
        _package_summary_row(
            status=status,
            phase2_block_rows=len(phase2_rows),
            external_label_csv_count=len(external_paths),
            registry_rows=len(registry_rows),
            label_preflight_rows=label_preflight_rows,
        )
    ]
    return {
        "phase": "phase68_external_independent_label_package",
        "phase68_status": status,
        "source_paths": {
            "phase2_output_dir": str(phase2_dir),
            "phase2_block_geofm_features_csv": str(phase2_csv),
            "external_label_csvs": [str(path) for path in external_paths],
            "label_registry": str(Path(label_registry))
            if label_registry is not None
            else None,
        },
        "thresholds": thresholds.__dict__,
        "row_counts": {
            "phase2_block_rows": len(phase2_rows),
            "template_rows": len(template_rows),
            "external_label_csvs": len(external_paths),
            "registry_rows": len(registry_rows),
            "label_preflight_rows": len(label_preflight_rows),
        },
        "external_label_template_rows": template_rows,
        "registry_template_rows": registry_template_rows,
        "label_preflight_rows": label_preflight_rows,
        "package_summary_rows": summary_rows,
        "interpretation": _phase68_interpretation(status),
        "recommended_next_step": _phase68_next_step(status),
        "claim_boundary": PHASE68_EXTERNAL_LABEL_PACKAGE_CLAIM_BOUNDARY,
    }


def write_phase68_external_independent_label_package_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "external_label_template_csv": output_path
        / "phase68_external_label_template.csv",
        "label_registry_template_csv": output_path
        / "phase68_label_registry_template.csv",
        "package_readme_md": output_path / "phase68_external_label_package_readme.md",
        "label_preflight_csv": output_path / "phase68_label_preflight.csv",
        "package_summary_csv": output_path / "phase68_package_summary.csv",
        "diagnosis_json": output_path
        / "phase68_external_independent_label_package.json",
        "diagnosis_md": output_path
        / "phase68_external_independent_label_package.md",
    }
    _write_csv_mapping_rows(
        artifacts["external_label_template_csv"],
        PHASE68_EXTERNAL_LABEL_TEMPLATE_FIELDNAMES,
        analysis.get("external_label_template_rows", []),
        "Phase 68 external label template rows",
    )
    _write_csv_mapping_rows(
        artifacts["label_registry_template_csv"],
        PHASE68_REGISTRY_FIELDNAMES,
        analysis.get("registry_template_rows", []),
        "Phase 68 registry template rows",
    )
    _write_csv_mapping_rows(
        artifacts["label_preflight_csv"],
        PHASE68_LABEL_PREFLIGHT_FIELDNAMES,
        analysis.get("label_preflight_rows", []),
        "Phase 68 label preflight rows",
    )
    _write_csv_mapping_rows(
        artifacts["package_summary_csv"],
        PHASE68_PACKAGE_SUMMARY_FIELDNAMES,
        analysis.get("package_summary_rows", []),
        "Phase 68 package summary rows",
    )
    artifacts["package_readme_md"].write_text(
        _phase68_package_readme(),
        encoding="utf-8",
    )
    artifacts["diagnosis_json"].write_text(
        json.dumps(_json_ready(analysis), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    artifacts["diagnosis_md"].write_text(_phase68_markdown(analysis), encoding="utf-8")
    return artifacts


def _read_csv_table(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise ValueError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _normalize_paths(paths: Sequence[Path | str] | Path | str | None) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _load_external_label_csvs(
    phase2_rows: Sequence[Mapping[str, str]],
    external_paths: Sequence[Path],
) -> tuple[dict[str, dict[str, str]], dict[str, list[str]], dict[str, int]]:
    phase2_block_ids = {
        str(row.get("block_id", "")).strip()
        for row in phase2_rows
        if str(row.get("block_id", "")).strip()
    }
    values_by_label: dict[str, dict[str, str]] = {}
    sources_by_label: dict[str, list[str]] = {}
    unjoined_by_label: dict[str, int] = {}
    for external_path in external_paths:
        fieldnames, external_rows = _read_csv_table(
            external_path,
            "Phase 68 external label CSV",
        )
        if "block_id" not in fieldnames:
            raise ValueError(
                f"Phase 68 external label CSV is missing block_id: {external_path}"
            )
        label_columns = [field for field in fieldnames if field != "block_id"]
        seen_block_ids: set[str] = set()
        for external_row in external_rows:
            block_id = str(external_row.get("block_id", "")).strip()
            if not block_id:
                raise ValueError(
                    "Phase 68 external label CSV contains a blank block_id: "
                    f"{external_path}"
                )
            if block_id in seen_block_ids:
                raise ValueError(
                    "Phase 68 external label CSV has duplicate block_id "
                    f"{block_id}: {external_path}"
                )
            seen_block_ids.add(block_id)
            joined = block_id in phase2_block_ids
            for label_column in label_columns:
                sources_by_label.setdefault(label_column, []).append(str(external_path))
                if not joined:
                    unjoined_by_label[label_column] = (
                        unjoined_by_label.get(label_column, 0) + 1
                    )
                    continue
                values_by_label.setdefault(label_column, {})[block_id] = str(
                    external_row.get(label_column, "")
                )
    return values_by_label, sources_by_label, unjoined_by_label


def _load_phase68_registry(label_registry: Path | str | None) -> list[dict[str, str]]:
    if label_registry is None:
        return []
    registry_path = Path(label_registry)
    if not registry_path.exists():
        raise ValueError(f"Missing Phase 68 label registry: {registry_path}")
    if registry_path.suffix.lower() == ".csv":
        _, rows = _read_csv_table(registry_path, "Phase 68 label registry CSV")
    elif registry_path.suffix.lower() == ".json":
        rows = _read_json_registry_rows(registry_path)
    else:
        raise ValueError(
            f"Unsupported Phase 68 label registry extension: {registry_path}"
        )
    normalized_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=2):
        normalized = {
            field: str(row.get(field, "")).strip()
            for field in PHASE68_REGISTRY_FIELDNAMES
        }
        if not normalized["label_column"]:
            raise ValueError(
                f"Phase 68 label registry row {index} has blank label_column: "
                f"{registry_path}"
            )
        normalized_rows.append(normalized)
    return normalized_rows


def _read_json_registry_rows(registry_path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Phase 68 label registry JSON is invalid: {registry_path}"
        ) from exc
    if isinstance(payload, list):
        if not all(isinstance(row, Mapping) for row in payload):
            raise ValueError(
                f"Phase 68 label registry JSON rows must be objects: {registry_path}"
            )
        return [dict(row) for row in payload]
    if isinstance(payload, Mapping):
        rows: list[dict[str, object]] = []
        for label_column, row in payload.items():
            if not isinstance(row, Mapping):
                raise ValueError(
                    "Phase 68 label registry JSON entry "
                    f"{label_column!r} is not an object: {registry_path}"
                )
            normalized = dict(row)
            normalized.setdefault("label_column", str(label_column))
            rows.append(normalized)
        return rows
    raise ValueError(
        f"Phase 68 label registry JSON must be a list or object: {registry_path}"
    )


def _external_label_template_rows(
    phase2_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "block_id": str(row.get("block_id", "")).strip(),
            "external_independent_label": "",
            "label_missing_reason": "",
            "source_record_id": "",
        }
        for row in phase2_rows
    ]


def _default_registry_template_row() -> dict[str, str]:
    return {
        "label_column": "external_independent_label",
        "label_source": "name_of_external_dataset_or_survey",
        "source_type": "external_soil",
        "independence_level": "independent",
        "allowed_eval_roles": "test,validation,eval",
        "provenance_note": (
            "Describe why this label is not derived from DLTB, slope, source "
            "metadata, explicit planning features, GeoFM, or model predictions."
        ),
        "license_or_access": (
            "Describe license, access permission, or restricted-access handling."
        ),
        "expected_positive_definition": "1",
        "source_owner": "organization_or_data_owner",
        "collection_date_or_period": "YYYY or YYYY-YYYY",
        "spatial_join_method": "block_id_join",
        "original_unit": "block",
        "label_scale": "binary",
        "missing_value_policy": "blank means missing",
        "known_overlap_with_dltb_slope_or_source_metadata": "none",
        "contact_or_access_note": "contact or access note",
    }


def _package_summary_row(
    status: str,
    phase2_block_rows: int,
    external_label_csv_count: int,
    registry_rows: int,
    label_preflight_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "phase68_status": status,
        "phase2_block_rows": phase2_block_rows,
        "external_label_csv_count": external_label_csv_count,
        "registry_rows": registry_rows,
        "ready_label_count": sum(
            1
            for row in label_preflight_rows
            if row.get("label_preflight_status") == "label_ready_for_phase40"
        ),
        "invalid_label_count": sum(
            1
            for row in label_preflight_rows
            if row.get("label_preflight_status") == "label_inputs_invalid"
        ),
        "diagnostic_label_count": sum(
            1
            for row in label_preflight_rows
            if row.get("label_preflight_status") == "label_diagnostic_only"
        ),
        "recommended_next_step": _phase68_next_step(status),
        "claim_boundary": PHASE68_EXTERNAL_LABEL_PACKAGE_CLAIM_BOUNDARY,
    }


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


def _phase68_package_readme() -> str:
    return "\n".join(
        [
            "# Phase 68 External Label Package",
            "",
            (
                "Provide a block-level CSV with `block_id` and one or more "
                "external independent label columns."
            ),
            (
                "The `block_id` values must come from the Paper11 Phase 2 "
                "`block_geofm_features.csv` table."
            ),
            (
                "Complete `phase68_label_registry_template.csv` so Phase 68 "
                "can check Phase 40-compatible source type, independence level, "
                "positive definition, access, and provenance."
            ),
            (
                "Acceptable sources include field survey, soil, irrigation, "
                "yield, high-standard-farmland, retention, policy outcome, or "
                "independent remote-sensing products."
            ),
            (
                "DLTB-derived, slope-derived, source-metadata-derived, "
                "GeoFM-derived, or model-generated labels remain diagnostic-only."
            ),
            "",
        ]
    )


def _phase68_markdown(analysis: Mapping[str, object]) -> str:
    rows = analysis.get("label_preflight_rows", [])
    table_rows = rows if isinstance(rows, list) else []
    lines = [
        "# Phase 68 External Independent Label Package",
        "",
        f"Status: {analysis.get('phase68_status', '')}",
        "",
        "## Label Preflight",
        "",
        *_markdown_table(
            (
                "label_column",
                "label_preflight_status",
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
                PHASE68_EXTERNAL_LABEL_PACKAGE_CLAIM_BOUNDARY,
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


def _phase68_interpretation(status: str) -> str:
    if status == "external_label_package_ready":
        return (
            "Templates and documentation are ready for an external independent "
            "label provider."
        )
    if status == "external_label_inputs_missing":
        return (
            "Validation mode was requested, but an external label CSV or "
            "registry is missing."
        )
    if status == "phase40_ready_to_rerun_with_external_label":
        return "At least one supplied external label appears ready for a Phase 40 rerun."
    if status == "independent_label_route_blocked":
        return (
            "Supplied labels are diagnostic-only or leakage-risk and cannot "
            "unlock Phase 40/41."
        )
    return "Supplied external label inputs failed Phase 68 preflight checks."


def _phase68_next_step(status: str) -> str:
    if status == "external_label_package_ready":
        return (
            "Provide a completed external label CSV and registry, then rerun "
            "Phase 68 in validation mode."
        )
    if status == "external_label_inputs_missing":
        return (
            "Missing external label input: supply both an external label CSV "
            "and a Phase 40-compatible registry, then rerun validation mode."
        )
    if status == "phase40_ready_to_rerun_with_external_label":
        return (
            "Rerun Phase 40 with the accepted external label registry before "
            "Phase 41 or reward redesign."
        )
    return "Do not run B2/B3 or reward redesign; fix the external label package first."
