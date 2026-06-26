from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path


PHASE37_DECISION_ALIGNMENT_CLAIM_BOUNDARY = (
    "Phase 37 is a read-only decision-alignment diagnostic over existing "
    "Phase 34, Phase 35, and optional Phase 36 artifacts; it does not run new "
    "policy training, does not alter rewards, does not enable suitability "
    "reward, does not test B2/B3, and does not support final "
    "submission-level planning-performance claims."
)

PHASE37_CASE_FIELDNAMES = [
    "case_id",
    "case_role",
    "eval_tile_id",
    "seed",
    "variant_id",
    "comparator_variant_id",
    "stability_class",
    "summary_reward_gap",
    "spatial_pattern",
    "action_overlap_pattern",
    "selected_block_jaccard",
    "base_planning_reward_gap",
    "suitability_proxy_gap",
    "low_slope_farmland_label_gap",
    "current_farmland_label_gap",
    "slope_mean_gap",
    "slope_max_gap",
    "proxy_alignment_pattern",
    "phase36_proxy_validation_status",
    "claim_boundary",
]

PHASE37_SUMMARY_FIELDNAMES = [
    "summary_group",
    "group_case_role",
    "group_eval_tile_id",
    "group_variant_id",
    "group_comparator_variant_id",
    "group_spatial_pattern",
    "group_action_overlap_pattern",
    "case_count",
    "mean_summary_reward_gap",
    "mean_base_planning_reward_gap",
    "mean_suitability_proxy_gap",
    "mean_low_slope_farmland_label_gap",
    "mean_current_farmland_label_gap",
    "mean_slope_mean_gap",
    "mean_slope_max_gap",
    "proxy_or_label_alignment_count",
    "claim_boundary",
]


def build_phase37_decision_alignment(
    phase34_cases_csv: Path | str,
    phase34_blocks_csv: Path | str,
    phase35_cases_csv: Path | str,
    phase36_diagnosis_json: Path | str | None = None,
) -> dict[str, object]:
    phase34_cases_path = Path(phase34_cases_csv)
    phase34_blocks_path = Path(phase34_blocks_csv)
    phase35_cases_path = Path(phase35_cases_csv)
    phase36_path = (
        Path(phase36_diagnosis_json) if phase36_diagnosis_json is not None else None
    )

    phase34_cases = _read_csv_rows(phase34_cases_path, "Phase 34 case CSV")
    phase34_blocks = _read_csv_rows(phase34_blocks_path, "Phase 34 block CSV")
    phase35_cases = _read_csv_rows(phase35_cases_path, "Phase 35 case CSV")
    phase36_status = _phase36_status(phase36_path)

    phase34_by_case = _index_by_case_id(phase34_cases)
    phase35_by_case = _index_by_case_id(phase35_cases)
    phase34_blocks_by_case = _group_by_case_id(phase34_blocks)

    case_rows: list[dict[str, object]] = []
    for case_id in sorted(set(phase34_by_case) & set(phase35_by_case)):
        phase34_case = phase34_by_case[case_id]
        phase35_case = phase35_by_case[case_id]
        blocks = phase34_blocks_by_case.get(case_id, [])
        case_rows.append(
            _case_row(case_id, phase34_case, phase35_case, blocks, phase36_status)
        )

    summary_rows = _summary_rows(case_rows)
    status = _phase37_status(case_rows)
    return {
        "phase": "phase37_decision_alignment",
        "phase37_decision_alignment_status": status,
        "phase36_proxy_validation_status": phase36_status,
        "source_paths": {
            "phase34_cases_csv": str(phase34_cases_path),
            "phase34_blocks_csv": str(phase34_blocks_path),
            "phase35_cases_csv": str(phase35_cases_path),
            "phase36_diagnosis_json": str(phase36_path) if phase36_path else None,
        },
        "row_counts": {
            "phase34_cases": len(phase34_cases),
            "phase34_blocks": len(phase34_blocks),
            "phase35_cases": len(phase35_cases),
            "case_rows": len(case_rows),
            "summary_rows": len(summary_rows),
        },
        "case_rows": case_rows,
        "summary_rows": summary_rows,
        "interpretation": _phase37_interpretation(status),
        "claim_boundary": PHASE37_DECISION_ALIGNMENT_CLAIM_BOUNDARY,
    }


def write_phase37_decision_alignment_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    case_rows = _validate_mapping_rows(analysis.get("case_rows"), "case_rows")
    summary_rows = _validate_mapping_rows(analysis.get("summary_rows"), "summary_rows")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "case_alignment_csv": output_path / "phase37_decision_alignment_cases.csv",
        "summary_csv": output_path / "phase37_decision_alignment_summary.csv",
        "diagnosis_json": output_path / "phase37_decision_alignment.json",
        "diagnosis_md": output_path / "phase37_decision_alignment.md",
    }
    _write_csv_rows(artifacts["case_alignment_csv"], case_rows, PHASE37_CASE_FIELDNAMES)
    _write_csv_rows(artifacts["summary_csv"], summary_rows, PHASE37_SUMMARY_FIELDNAMES)
    artifacts["diagnosis_json"].write_text(
        json.dumps(_json_ready(analysis), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifacts["diagnosis_md"].write_text(
        _decision_alignment_markdown(analysis),
        encoding="utf-8",
    )
    return artifacts


def _case_row(
    case_id: str,
    phase34_case: Mapping[str, object],
    phase35_case: Mapping[str, object],
    blocks: Sequence[Mapping[str, object]],
    phase36_status: str,
) -> dict[str, object]:
    variant_blocks = [row for row in blocks if _has_value(row, "variant_step")]
    comparator_blocks = [row for row in blocks if _has_value(row, "comparator_step")]
    suitability_gap = _gap(
        phase34_case,
        "variant_mean_suitability_proxy",
        "comparator_mean_suitability_proxy",
    )
    low_slope_gap = _gap(
        phase34_case,
        "variant_mean_low_slope_farmland_label",
        "comparator_mean_low_slope_farmland_label",
    )
    current_farmland_gap = _set_gap(
        variant_blocks,
        comparator_blocks,
        "current_farmland_label",
    )
    slope_mean_gap = _set_gap(variant_blocks, comparator_blocks, "slope_mean")
    row = {
        "case_id": case_id,
        "case_role": str(phase34_case.get("case_role", "")),
        "eval_tile_id": str(phase34_case.get("eval_tile_id", "")),
        "seed": _optional_int(phase34_case, "seed"),
        "variant_id": str(phase34_case.get("variant_id", "")),
        "comparator_variant_id": str(phase34_case.get("comparator_variant_id", "")),
        "stability_class": str(phase34_case.get("stability_class", "")),
        "summary_reward_gap": _optional_float(phase35_case, "summary_reward_gap"),
        "spatial_pattern": str(phase34_case.get("spatial_pattern", "")),
        "action_overlap_pattern": str(phase35_case.get("action_overlap_pattern", "")),
        "selected_block_jaccard": _optional_float(
            phase35_case,
            "selected_block_jaccard",
        ),
        "base_planning_reward_gap": _gap(
            phase34_case,
            "variant_mean_base_planning_reward",
            "comparator_mean_base_planning_reward",
        ),
        "suitability_proxy_gap": suitability_gap,
        "low_slope_farmland_label_gap": low_slope_gap,
        "current_farmland_label_gap": current_farmland_gap,
        "slope_mean_gap": slope_mean_gap,
        "slope_max_gap": _set_gap(variant_blocks, comparator_blocks, "slope_max"),
        "proxy_alignment_pattern": _proxy_alignment_pattern(
            suitability_gap,
            low_slope_gap,
            current_farmland_gap,
            slope_mean_gap,
        ),
        "phase36_proxy_validation_status": phase36_status,
        "claim_boundary": PHASE37_DECISION_ALIGNMENT_CLAIM_BOUNDARY,
    }
    return row


def _summary_rows(case_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: list[tuple[str, dict[str, object], list[Mapping[str, object]]]] = [
        ("all_cases", {}, list(case_rows)),
        (
            "phase33_positive_case",
            {"group_case_role": "phase33_positive_case"},
            [
                row
                for row in case_rows
                if str(row.get("case_role", "")) == "phase33_positive_case"
            ],
        ),
        (
            "phase33_failure_case",
            {"group_case_role": "phase33_failure_case"},
            [
                row
                for row in case_rows
                if str(row.get("case_role", "")) == "phase33_failure_case"
            ],
        ),
    ]
    for key, rows in _case_groups(case_rows).items():
        groups.append((_case_group_name(key), _case_group_fields(key), rows))
    return [_summary_row(name, fields, rows) for name, fields, rows in groups if rows]


def _case_groups(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str, str, str, str], list[Mapping[str, object]]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = _case_group_key(row)
        grouped.setdefault(key, []).append(row)
    return grouped


def _case_group_key(row: Mapping[str, object]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("case_role", "")),
        str(row.get("eval_tile_id", "")),
        str(row.get("variant_id", "")),
        str(row.get("comparator_variant_id", "")),
        str(row.get("spatial_pattern", "")),
        str(row.get("action_overlap_pattern", "")),
    )


def _case_group_name(key: tuple[str, str, str, str, str, str]) -> str:
    return "case_group|" + "|".join(key)


def _case_group_fields(key: tuple[str, str, str, str, str, str]) -> dict[str, object]:
    return {
        "group_case_role": key[0],
        "group_eval_tile_id": key[1],
        "group_variant_id": key[2],
        "group_comparator_variant_id": key[3],
        "group_spatial_pattern": key[4],
        "group_action_overlap_pattern": key[5],
    }


def _summary_row(
    name: str,
    group_fields: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "summary_group": name,
        "group_case_role": group_fields.get("group_case_role", ""),
        "group_eval_tile_id": group_fields.get("group_eval_tile_id", ""),
        "group_variant_id": group_fields.get("group_variant_id", ""),
        "group_comparator_variant_id": group_fields.get("group_comparator_variant_id", ""),
        "group_spatial_pattern": group_fields.get("group_spatial_pattern", ""),
        "group_action_overlap_pattern": group_fields.get("group_action_overlap_pattern", ""),
        "case_count": len(rows),
        "mean_summary_reward_gap": _mean_field(rows, "summary_reward_gap"),
        "mean_base_planning_reward_gap": _mean_field(rows, "base_planning_reward_gap"),
        "mean_suitability_proxy_gap": _mean_field(rows, "suitability_proxy_gap"),
        "mean_low_slope_farmland_label_gap": _mean_field(
            rows,
            "low_slope_farmland_label_gap",
        ),
        "mean_current_farmland_label_gap": _mean_field(
            rows,
            "current_farmland_label_gap",
        ),
        "mean_slope_mean_gap": _mean_field(rows, "slope_mean_gap"),
        "mean_slope_max_gap": _mean_field(rows, "slope_max_gap"),
        "proxy_or_label_alignment_count": sum(
            1
            for row in rows
            if row.get("proxy_alignment_pattern") == "proxy_or_label_alignment"
        ),
        "claim_boundary": PHASE37_DECISION_ALIGNMENT_CLAIM_BOUNDARY,
    }


def _phase37_status(case_rows: Sequence[Mapping[str, object]]) -> str:
    if not case_rows:
        return "decision_alignment_inputs_insufficient"
    positive_rows = [
        row
        for row in case_rows
        if str(row.get("case_role", "")) == "phase33_positive_case"
    ]
    failure_rows = [
        row
        for row in case_rows
        if str(row.get("case_role", "")) == "phase33_failure_case"
    ]
    if not positive_rows or not failure_rows:
        return "decision_alignment_inputs_insufficient"
    # The support gate is narrower than proxy_alignment_pattern by design.
    gate_fields = ("suitability_proxy_gap", "low_slope_farmland_label_gap")
    positive_supported = any(
        _proxy_rebuild_signal_positive(group_rows, gate_fields)
        for group_rows in _case_groups(positive_rows).values()
    )
    failure_supported = any(
        _proxy_rebuild_signal_positive(group_rows, gate_fields)
        for group_rows in _case_groups(failure_rows).values()
    )
    if positive_supported and not failure_supported:
        return "decision_alignment_supported_for_proxy_rebuild"
    return "decision_alignment_not_supported"


def _phase37_interpretation(status: str) -> str:
    if status == "decision_alignment_supported_for_proxy_rebuild":
        return (
            "At least one positive Phase 33 case group shows positive "
            "suitability-proxy or low-slope label alignment, while no "
            "failure-case group shows the same status-gate signal. This remains "
            "a conservative Phase 37 diagnostic rule."
        )
    if status == "decision_alignment_not_supported":
        return (
            "Phase 37 did not find conservative decision-alignment support: "
            "either no positive case group met the status gate, or at least one "
            "failure-case group also showed a positive status-gate signal."
        )
    return "Phase 37 could not join enough input cases for decision alignment."


def _validate_mapping_rows(value: object, key: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"analysis['{key}'] must be a list")
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ValueError(f"analysis['{key}'][{index}] must be a Mapping")
    return value


def _write_csv_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _decision_alignment_markdown(analysis: Mapping[str, object]) -> str:
    status = str(analysis.get("phase37_decision_alignment_status", ""))
    phase36_status = str(analysis.get("phase36_proxy_validation_status", ""))
    interpretation = str(analysis.get("interpretation", ""))
    claim_boundary = str(
        analysis.get("claim_boundary", PHASE37_DECISION_ALIGNMENT_CLAIM_BOUNDARY)
    )
    summary_rows = analysis.get("summary_rows")
    if not isinstance(summary_rows, list):
        raise ValueError("analysis['summary_rows'] must be a list")

    lines = [
        "# Phase 37 Decision-Alignment",
        "",
        f"Status: {status}",
        f"Phase36 status: {phase36_status}",
        "",
        "## Case Summary",
        "",
    ]
    lines.extend(_summary_markdown_table(summary_rows))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation,
            "",
            "## Claim Boundary",
            "",
            claim_boundary,
            "",
        ]
    )
    return "\n".join(lines)


def _summary_markdown_table(rows: Sequence[Mapping[str, object]]) -> list[str]:
    columns = [
        "summary_group",
        "case_count",
        "mean_summary_reward_gap",
        "mean_suitability_proxy_gap",
        "mean_low_slope_farmland_label_gap",
        "proxy_or_label_alignment_count",
    ]
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        table.append(
            "| "
            + " | ".join(_markdown_cell(row.get(column, "")) for column in columns)
            + " |"
        )
    return table


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _read_csv_rows(path: Path, label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _phase36_status(path: Path | None) -> str:
    if path is None:
        return "phase36_not_supplied"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        return str(payload.get("phase36_proxy_validation_status", ""))
    return ""


def _index_by_case_id(rows: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if case_id:
            indexed[case_id] = row
    return indexed


def _group_by_case_id(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if case_id:
            grouped.setdefault(case_id, []).append(row)
    return grouped


def _gap(row: Mapping[str, object], variant_field: str, comparator_field: str) -> float | str:
    variant = _optional_float(row, variant_field)
    comparator = _optional_float(row, comparator_field)
    if variant == "" or comparator == "":
        return ""
    return _round_float(float(variant) - float(comparator))


def _set_gap(
    variant_rows: Sequence[Mapping[str, object]],
    comparator_rows: Sequence[Mapping[str, object]],
    field: str,
) -> float | str:
    variant_mean = _mean_optional([_optional_float(row, field) for row in variant_rows])
    comparator_mean = _mean_optional(
        [_optional_float(row, field) for row in comparator_rows]
    )
    if variant_mean == "" or comparator_mean == "":
        return ""
    return _round_float(float(variant_mean) - float(comparator_mean))


def _proxy_alignment_pattern(
    suitability_gap: object,
    low_slope_gap: object,
    current_farmland_gap: object,
    slope_mean_gap: object,
) -> str:
    if (
        _is_positive(suitability_gap)
        or _is_positive(low_slope_gap)
        or _is_positive(current_farmland_gap)
        or _is_negative(slope_mean_gap)
    ):
        return "proxy_or_label_alignment"
    return "no_proxy_alignment"


def _proxy_rebuild_signal_positive(
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> bool:
    return any(_is_positive(_mean_field(rows, field)) for field in fields)


def _mean_field(rows: Sequence[Mapping[str, object]], field: str) -> float | str:
    return _mean_optional([_optional_float(row, field) for row in rows])


def _mean_optional(values: Sequence[object]) -> float | str:
    clean = [float(value) for value in values if _has_numeric(value)]
    if not clean:
        return ""
    return _round_float(sum(clean) / len(clean))


def _optional_float(row: Mapping[str, object], field: str) -> float | str:
    if not _has_value(row, field):
        return ""
    return _round_float(float(row[field]))


def _optional_int(row: Mapping[str, object], field: str) -> int | str:
    if not _has_value(row, field):
        return ""
    return int(float(row[field]))


def _has_value(row: Mapping[str, object], field: str) -> bool:
    return field in row and str(row.get(field, "")).strip() != ""


def _has_numeric(value: object) -> bool:
    return value is not None and str(value).strip() != ""


def _is_positive(value: object) -> bool:
    return _has_numeric(value) and float(value) > 0.0


def _is_negative(value: object) -> bool:
    return _has_numeric(value) and float(value) < 0.0


def _round_float(value: float | int) -> float:
    return round(float(value), 10)
