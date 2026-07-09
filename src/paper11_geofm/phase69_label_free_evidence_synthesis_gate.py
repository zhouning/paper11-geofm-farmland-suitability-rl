from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import math
from pathlib import Path


PHASE69_LABEL_FREE_SYNTHESIS_CLAIM_BOUNDARY = (
    "Phase 69 is a read-only label-free evidence synthesis gate over existing "
    "Paper11 artifacts. It does not train PPO, does not alter rewards, does not "
    "enable B2/B3, does not validate suitability, and does not justify formal "
    "submission-level claims."
)

PHASE69_EVIDENCE_AXIS_FIELDNAMES = (
    "axis_id",
    "axis_class",
    "axis_status",
    "source_phases",
    "primary_metric",
    "primary_value",
    "decision_reason",
    "claim_boundary",
)

PHASE69_CLAIM_BOUNDARY_FIELDNAMES = (
    "claim_id",
    "claim_status",
    "supporting_axis_ids",
    "blocking_axis_ids",
    "decision_reason",
    "claim_boundary",
)


def build_phase69_label_free_evidence_synthesis_gate(
    phase60_json: Path | str,
    phase57_json: Path | str,
    phase59_json: Path | str,
    phase62_json: Path | str,
    phase66_json: Path | str,
    phase67_json: Path | str,
    phase68_json: Path | str,
) -> dict[str, object]:
    phase60 = _read_json_object(phase60_json, "Phase 60 JSON")
    phase57 = _read_json_object(phase57_json, "Phase 57 JSON")
    phase59 = _read_json_object(phase59_json, "Phase 59 JSON")
    phase62 = _read_json_object(phase62_json, "Phase 62 JSON")
    phase66 = _read_json_object(phase66_json, "Phase 66 JSON")
    phase67 = _read_json_object(phase67_json, "Phase 67 JSON")
    phase68 = _read_json_object(phase68_json, "Phase 68 JSON")

    evidence_axis_rows = [
        _route_support_axis(phase60),
        _mechanism_support_axis(phase57),
        _mechanism_limits_axis(phase59, phase62),
        _reward_target_limits_axis(phase66, phase67),
        _external_label_state_axis(phase68),
    ]
    status = _phase69_status(evidence_axis_rows)
    claim_boundary_rows = _claim_boundary_rows(status, evidence_axis_rows)
    return {
        "phase": "phase69_label_free_evidence_synthesis_gate",
        "phase69_status": status,
        "source_paths": {
            "phase60_json": str(Path(phase60_json)),
            "phase57_json": str(Path(phase57_json)),
            "phase59_json": str(Path(phase59_json)),
            "phase62_json": str(Path(phase62_json)),
            "phase66_json": str(Path(phase66_json)),
            "phase67_json": str(Path(phase67_json)),
            "phase68_json": str(Path(phase68_json)),
        },
        "row_counts": {
            "evidence_axis_rows": len(evidence_axis_rows),
            "claim_boundary_rows": len(claim_boundary_rows),
        },
        "evidence_axis_rows": evidence_axis_rows,
        "claim_boundary_rows": claim_boundary_rows,
        "allowed_claim": _allowed_claim(status),
        "blocked_claims": [
            row["claim_id"]
            for row in claim_boundary_rows
            if row["claim_status"] == "blocked"
        ],
        "recommended_next_step": _phase69_next_step(status),
        "claim_boundary": PHASE69_LABEL_FREE_SYNTHESIS_CLAIM_BOUNDARY,
    }


def write_phase69_label_free_evidence_synthesis_gate_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "evidence_axes_csv": output_path / "phase69_evidence_axes.csv",
        "claim_boundary_matrix_csv": output_path / "phase69_claim_boundary_matrix.csv",
        "diagnosis_json": output_path / "phase69_label_free_evidence_synthesis_gate.json",
        "diagnosis_md": output_path / "phase69_label_free_evidence_synthesis_gate.md",
    }
    _write_csv_mapping_rows(
        artifacts["evidence_axes_csv"],
        PHASE69_EVIDENCE_AXIS_FIELDNAMES,
        analysis.get("evidence_axis_rows", []),
        "Phase 69 evidence axis rows",
    )
    _write_csv_mapping_rows(
        artifacts["claim_boundary_matrix_csv"],
        PHASE69_CLAIM_BOUNDARY_FIELDNAMES,
        analysis.get("claim_boundary_rows", []),
        "Phase 69 claim boundary rows",
    )
    artifacts["diagnosis_json"].write_text(
        json.dumps(_json_ready(analysis), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    artifacts["diagnosis_md"].write_text(_phase69_markdown(analysis), encoding="utf-8")
    return artifacts


def _read_json_object(path: Path | str, label: str) -> dict[str, object]:
    json_path = Path(path)
    if not json_path.exists():
        raise ValueError(f"Missing {label}: {json_path}")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {json_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {json_path}")
    return dict(payload)


def _route_support_axis(phase60: Mapping[str, object]) -> dict[str, object]:
    axes = phase60.get("attribution_axes", [])
    route = _find_axis(axes, "compressed_route_performance")
    cluster = _find_axis(axes, "cluster_level_robustness")
    supported = (
        route.get("axis_status") == "supported"
        and cluster.get("axis_status") == "supported"
    )
    mean_delta = _as_float(route.get("primary_value"))
    return _axis_row(
        axis_id="route_support",
        axis_class="support" if supported else "missing",
        axis_status="supported" if supported else "missing_or_not_supported",
        source_phases="phase48_phase52_expanded,phase53",
        primary_metric="pooled_or_cluster_mean_delta",
        primary_value=mean_delta,
        decision_reason=(
            "Compressed route and cluster robustness are supported."
            if supported
            else "Compressed route support or cluster robustness is missing."
        ),
    )


def _mechanism_support_axis(phase57: Mapping[str, object]) -> dict[str, object]:
    status = str(phase57.get("phase57_mechanism_status", ""))
    supported = status == "compressed_geometry_consistent"
    return _axis_row(
        axis_id="mechanism_support",
        axis_class="support" if supported else "missing",
        axis_status=status or "missing",
        source_phases="phase57",
        primary_metric="phase57_mechanism_status",
        primary_value=status,
        decision_reason=(
            "Compressed geometry is consistent with the low-dimensional route."
            if supported
            else "Compressed geometry support is missing or not supported."
        ),
    )


def _mechanism_limits_axis(
    phase59: Mapping[str, object],
    phase62: Mapping[str, object],
) -> dict[str, object]:
    phase59_status = str(phase59.get("phase59_matched_dimension_status", ""))
    phase62_status = str(phase62.get("phase62_d4_d6_status", ""))
    limiting = (
        phase59_status == "matched_dimension_geofm_not_supported"
        and phase62_status == "d6_random_projection_advantage"
    )
    return _axis_row(
        axis_id="mechanism_limits",
        axis_class="limit" if limiting else "missing",
        axis_status=f"{phase59_status};{phase62_status}",
        source_phases="phase59,phase62",
        primary_metric="matched_control_limit_status",
        primary_value=f"{phase59_status};{phase62_status}",
        decision_reason=(
            "Matched-dimension and D6 controls block GeoFM-specific/PCA-optimality claims."
            if limiting
            else "Matched-control limiting evidence is missing or inconsistent."
        ),
    )


def _reward_target_limits_axis(
    phase66: Mapping[str, object],
    phase67: Mapping[str, object],
) -> dict[str, object]:
    phase66_status = str(phase66.get("phase66_status", ""))
    phase67_status = str(phase67.get("phase67_status", ""))
    blocked = (
        phase66_status == "base_reward_target_masks_geofm_signal"
        and phase67_status == "independent_label_required_before_reward_redesign"
    )
    return _axis_row(
        axis_id="reward_target_limits",
        axis_class="blocked" if blocked else "missing",
        axis_status=f"{phase66_status};{phase67_status}",
        source_phases="phase66,phase67",
        primary_metric="reward_target_block_status",
        primary_value=f"{phase66_status};{phase67_status}",
        decision_reason=(
            "Base reward masks GeoFM signal and candidate-target route requires independent labels."
            if blocked
            else "Reward/target limiting evidence is missing or inconsistent."
        ),
    )


def _external_label_state_axis(phase68: Mapping[str, object]) -> dict[str, object]:
    status = str(phase68.get("phase68_status", ""))
    blocked = status == "external_label_package_ready"
    return _axis_row(
        axis_id="external_label_state",
        axis_class="blocked" if blocked else "support",
        axis_status=status or "missing",
        source_phases="phase68",
        primary_metric="phase68_status",
        primary_value=status,
        decision_reason=(
            "External-label package is ready, but no external label CSV or registry has been supplied."
            if blocked
            else "External label state is not template-only."
        ),
    )


def _axis_row(
    axis_id: str,
    axis_class: str,
    axis_status: str,
    source_phases: str,
    primary_metric: str,
    primary_value: object,
    decision_reason: str,
) -> dict[str, object]:
    return {
        "axis_id": axis_id,
        "axis_class": axis_class,
        "axis_status": axis_status,
        "source_phases": source_phases,
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "decision_reason": decision_reason,
        "claim_boundary": PHASE69_LABEL_FREE_SYNTHESIS_CLAIM_BOUNDARY,
    }


def _phase69_status(evidence_axis_rows: Sequence[Mapping[str, object]]) -> str:
    rows = {str(row.get("axis_id")): row for row in evidence_axis_rows}
    route_supported = rows.get("route_support", {}).get("axis_class") == "support"
    mechanism_supported = rows.get("mechanism_support", {}).get("axis_class") == "support"
    if not route_supported or not mechanism_supported:
        return "label_free_evidence_insufficient"
    has_limits = rows.get("mechanism_limits", {}).get("axis_class") == "limit"
    has_reward_block = rows.get("reward_target_limits", {}).get("axis_class") == "blocked"
    has_label_block = rows.get("external_label_state", {}).get("axis_class") == "blocked"
    if has_limits or has_reward_block or has_label_block:
        return "claim_must_be_narrowed_to_low_dimensional_route"
    return "bounded_label_free_algorithm_claim_supported"


def _claim_boundary_rows(
    status: str,
    evidence_axis_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    allowed = status in {
        "bounded_label_free_algorithm_claim_supported",
        "claim_must_be_narrowed_to_low_dimensional_route",
    }
    return [
        _claim_row(
            "bounded_low_dimensional_route",
            "allowed" if allowed else "blocked",
            "route_support,mechanism_support",
            "",
            "Bounded low-dimensional compressed state route is allowed under label-free evidence."
            if allowed
            else "Required label-free support axes are missing.",
        ),
        _claim_row(
            "raw_b1_superiority",
            "blocked",
            "",
            "mechanism_limits",
            "Raw B1 superiority is not supported.",
        ),
        _claim_row(
            "pca_optimality",
            "blocked",
            "",
            "mechanism_limits",
            "Matched controls block PCA optimality.",
        ),
        _claim_row(
            "geofm_specific_matched_dimension_superiority",
            "blocked",
            "",
            "mechanism_limits",
            "Matched-dimension evidence blocks this claim.",
        ),
        _claim_row(
            "suitability_reward_readiness",
            "blocked",
            "",
            "reward_target_limits,external_label_state",
            "Suitability reward requires independent labels.",
        ),
        _claim_row(
            "b2_b3_reward_integration",
            "blocked",
            "",
            "reward_target_limits,external_label_state",
            "B2/B3 reward integration remains blocked.",
        ),
        _claim_row(
            "external_independent_label_passed",
            "blocked",
            "",
            "external_label_state",
            "Phase 68 is template-ready only.",
        ),
        _claim_row(
            "independent_agronomic_suitability",
            "blocked",
            "",
            "external_label_state",
            "No independent agronomic label has passed.",
        ),
        _claim_row(
            "cross_region_transfer",
            "blocked",
            "",
            "not_tested",
            "Cross-region transfer is not tested.",
        ),
        _claim_row(
            "formal_submission_readiness",
            "out_of_scope",
            "",
            "paper_not_revised",
            "Formal submission readiness is outside Phase 69.",
        ),
    ]


def _claim_row(
    claim_id: str,
    claim_status: str,
    supporting_axis_ids: str,
    blocking_axis_ids: str,
    decision_reason: str,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "claim_status": claim_status,
        "supporting_axis_ids": supporting_axis_ids,
        "blocking_axis_ids": blocking_axis_ids,
        "decision_reason": decision_reason,
        "claim_boundary": PHASE69_LABEL_FREE_SYNTHESIS_CLAIM_BOUNDARY,
    }


def _find_axis(axes: object, axis_id: str) -> dict[str, object]:
    if not isinstance(axes, list):
        return {}
    for row in axes:
        if isinstance(row, Mapping) and row.get("axis_id") == axis_id:
            return dict(row)
    return {}


def _allowed_claim(status: str) -> str:
    if status == "label_free_evidence_insufficient":
        return "No label-free algorithm claim is supported by the required axes."
    return "bounded low-dimensional compressed state route under the Bishan base-reward protocol."


def _phase69_next_step(status: str) -> str:
    if status == "label_free_evidence_insufficient":
        return "Do not revise manuscript claims; repair missing label-free evidence first."
    return "Keep claims narrowed; obtain external independent labels before suitability reward or reward redesign."


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


def _as_float(value: object) -> float | str:
    try:
        return round(float(value), 10)
    except (TypeError, ValueError):
        return ""


def _phase69_markdown(analysis: Mapping[str, object]) -> str:
    axes = analysis.get("evidence_axis_rows", [])
    claims = analysis.get("claim_boundary_rows", [])
    axis_rows = axes if isinstance(axes, list) else []
    claim_rows = claims if isinstance(claims, list) else []
    lines = [
        "# Phase 69 Label-Free Evidence Synthesis Gate",
        "",
        f"Status: {analysis.get('phase69_status', '')}",
        "",
        "## Allowed Claim",
        "",
        str(analysis.get("allowed_claim", "")),
        "",
        "## Evidence Axes",
        "",
        *_markdown_table(("axis_id", "axis_class", "axis_status", "decision_reason"), axis_rows),
        "",
        "## Claim Boundary Matrix",
        "",
        *_markdown_table(("claim_id", "claim_status", "decision_reason"), claim_rows),
        "",
        "## Recommended Next Step",
        "",
        str(analysis.get("recommended_next_step", "")),
        "",
        "## Boundary",
        "",
        str(analysis.get("claim_boundary", PHASE69_LABEL_FREE_SYNTHESIS_CLAIM_BOUNDARY)),
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