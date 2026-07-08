from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path


PHASE60_CLAIM_BOUNDARY = (
    "Phase 60 is a read-only information-vs-optimization attribution audit over "
    "existing Paper11 Phase 48/53/57/59 artifacts. It reconciles compressed-route "
    "performance, cluster robustness, representation geometry, and matched-dimension "
    "controls; it does not retrain RL policies, does not enable B2/B3, does not "
    "add suitability reward, does not test transfer, and does not validate "
    "independent agronomic suitability."
)

PHASE60_AXIS_FIELDNAMES = [
    "axis_id",
    "axis_label",
    "axis_status",
    "source_phase",
    "primary_metric",
    "primary_value",
    "support_threshold",
    "interpretation",
    "claim_boundary",
]


def build_phase60_information_optimization_attribution(
    phase48: Mapping[str, object],
    phase53: Mapping[str, object],
    phase57: Mapping[str, object],
    phase59: Mapping[str, object],
    source_paths: Mapping[str, object] | None = None,
) -> dict[str, object]:
    axes = [
        _compressed_route_axis(phase48),
        _cluster_robustness_axis(phase53),
        _compressed_geometry_axis(phase57),
        _matched_dimension_axis(phase59),
    ]
    status = _phase60_status(axes)
    return {
        "phase": "phase60_information_optimization_attribution",
        "phase60_attribution_status": status,
        "attribution_axes": axes,
        "source_paths": {} if source_paths is None else dict(source_paths),
        "claim_boundary_recommendation": _claim_boundary_recommendation(status),
        "next_experiment_recommendation": _next_experiment_recommendation(status),
        "conclusion": _phase60_conclusion(status),
        "claim_boundary": PHASE60_CLAIM_BOUNDARY,
    }


def _compressed_route_axis(phase48: Mapping[str, object]) -> dict[str, object]:
    pooled = _required_mapping(phase48, "pooled_compressed_control_delta")
    coverage = _required_mapping(phase48, "coverage_issues")
    mean_delta = _required_float(pooled, "mean_reward_delta")
    positive_fraction = _required_float(pooled, "positive_fraction")
    status = str(phase48.get("phase48_compressed_geofm_status", ""))
    supported = (
        status == "compressed_geofm_route_supported"
        and mean_delta > 0.0
        and positive_fraction >= 0.5
        and not _has_coverage_issues(coverage)
    )
    return _axis_row(
        axis_id="compressed_route_performance",
        axis_label="D4P8/D4P16 versus B0/B1/D2/D3",
        axis_status="supported" if supported else "not_supported",
        source_phase="phase48_phase52_expanded",
        primary_metric="pooled_mean_delta",
        primary_value=mean_delta,
        support_threshold="> 0 and positive_fraction >= 0.5",
        interpretation=(
            "Compressed route performance is supported against earlier controls."
            if supported
            else "Compressed route performance is not supported against earlier controls."
        ),
    )


def _cluster_robustness_axis(phase53: Mapping[str, object]) -> dict[str, object]:
    summary = _required_mapping(phase53, "cluster_mean_summary")
    mean_delta = _required_float(summary, "mean_cluster_delta")
    p_value = _optional_float(summary.get("exact_sign_flip_mean_p"))
    status = str(phase53.get("phase53_cluster_mean_status", ""))
    supported = (
        status == "cluster_mean_support"
        and mean_delta > 0.0
        and (p_value is None or p_value < 0.05)
    )
    return _axis_row(
        axis_id="cluster_level_robustness",
        axis_label="Expanded cluster mean robustness",
        axis_status="supported" if supported else "not_supported",
        source_phase="phase53",
        primary_metric="cluster_mean_delta",
        primary_value=mean_delta,
        support_threshold="> 0 and exact p < 0.05 when present",
        interpretation=(
            "Cluster-level support is present."
            if supported
            else "Cluster-level support is not established."
        ),
    )


def _compressed_geometry_axis(phase57: Mapping[str, object]) -> dict[str, object]:
    geometry_rows = _required_list(phase57, "geometry_rows")
    reward_rows = _required_list(phase57, "reward_gain_rows")
    geometry = {
        str(row.get("variant_id")): row
        for row in geometry_rows
        if isinstance(row, Mapping)
    }
    gains = {
        str(row.get("compressed_variant_id")): row
        for row in reward_rows
        if isinstance(row, Mapping)
    }
    raw_rank = _required_float(_required_mapping(geometry, "B1"), "effective_rank")
    d4p8_rank = _required_float(_required_mapping(geometry, "D4P8"), "effective_rank")
    d4p16_rank = _required_float(_required_mapping(geometry, "D4P16"), "effective_rank")
    d4p8_retention = _required_float(
        _required_mapping(geometry, "D4P8"),
        "raw_variance_retention",
    )
    d4p16_retention = _required_float(
        _required_mapping(geometry, "D4P16"),
        "raw_variance_retention",
    )
    d4p8_gain = _required_float(_required_mapping(gains, "D4P8"), "mean_delta")
    d4p16_gain = _required_float(_required_mapping(gains, "D4P16"), "mean_delta")
    phase_status = str(phase57.get("phase57_mechanism_status", ""))
    supported = (
        phase_status == "compressed_geometry_consistent"
        and d4p8_rank < raw_rank
        and d4p16_rank < raw_rank
        and d4p8_retention > 0.0
        and d4p16_retention > 0.0
        and d4p8_gain > 0.0
        and d4p16_gain > 0.0
    )
    return _axis_row(
        axis_id="compressed_geometry_consistency",
        axis_label="Lower-rank compressed GeoFM geometry",
        axis_status="supported" if supported else "not_supported",
        source_phase="phase57",
        primary_metric="max_compressed_effective_rank",
        primary_value=max(d4p8_rank, d4p16_rank),
        support_threshold="below raw B1 effective rank with positive reward gains",
        interpretation=(
            "Compressed geometry is consistent with the low-dimensional route."
            if supported
            else "Compressed geometry does not fully support the low-dimensional route."
        ),
    )


def _matched_dimension_axis(phase59: Mapping[str, object]) -> dict[str, object]:
    learned = _required_mapping(phase59, "learned_policy")
    matched = _required_mapping(learned, "matched_deltas")
    pooled = _required_mapping(phase59, "pooled_matched_control_delta")
    coverage = _required_mapping(phase59, "coverage_issues")
    required = (
        "D4P8_minus_D5R8",
        "D4P8_minus_D5S8",
        "D4P16_minus_D5R16",
        "D4P16_minus_D5S16",
    )
    means = [
        _required_float(_required_mapping(matched, key), "mean_delta")
        for key in required
    ]
    pooled_mean = _required_float(pooled, "mean_delta")
    positive_fraction = _required_float(pooled, "positive_fraction")
    phase_status = str(phase59.get("phase59_matched_dimension_status", ""))
    supported = (
        phase_status == "matched_dimension_geofm_supported"
        and all(value > 0.0 for value in means)
        and pooled_mean > 0.0
        and positive_fraction >= 0.5
        and not _has_coverage_issues(coverage)
    )
    return _axis_row(
        axis_id="geofm_specific_matched_dimension",
        axis_label="D4P8/D4P16 versus same-dimension controls",
        axis_status="supported" if supported else "not_supported",
        source_phase="phase59",
        primary_metric="pooled_matched_control_mean_delta",
        primary_value=pooled_mean,
        support_threshold="> 0 with all matched comparisons positive",
        interpretation=(
            "GeoFM-specific matched-dimension advantage is supported."
            if supported
            else "GeoFM-specific matched-dimension advantage is not supported."
        ),
    )


def _phase60_status(axes: Sequence[Mapping[str, object]]) -> str:
    by_axis = {str(row["axis_id"]): str(row["axis_status"]) for row in axes}
    early_axes = (
        "compressed_route_performance",
        "cluster_level_robustness",
        "compressed_geometry_consistency",
    )
    if any(by_axis.get(axis) != "supported" for axis in early_axes):
        return "low_dimensional_route_uncertain"
    if by_axis.get("geofm_specific_matched_dimension") == "supported":
        return "geofm_specific_information_supported"
    return "mechanism_claim_narrowed"


def _axis_row(
    axis_id,
    axis_label,
    axis_status,
    source_phase,
    primary_metric,
    primary_value,
    support_threshold,
    interpretation,
):
    return {
        "axis_id": axis_id,
        "axis_label": axis_label,
        "axis_status": axis_status,
        "source_phase": source_phase,
        "primary_metric": primary_metric,
        "primary_value": round(float(primary_value), 10),
        "support_threshold": support_threshold,
        "interpretation": interpretation,
        "claim_boundary": PHASE60_CLAIM_BOUNDARY,
    }


def _required_mapping(
    mapping: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Phase 60 missing object field: {key}")
    return value


def _required_list(mapping: Mapping[str, object], key: str) -> list[object]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Phase 60 missing list field: {key}")
    return value


def _required_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Phase 60 missing numeric field: {key}")
    return float(value)


def _optional_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _has_coverage_issues(coverage: Mapping[str, object]) -> bool:
    return any(
        bool(coverage.get(key))
        for key in (
            "missing_variant_rows",
            "duplicate_variant_rows",
            "unexpected_variant_rows",
        )
    )


def _claim_boundary_recommendation(status: str) -> str:
    if status == "geofm_specific_information_supported":
        return "allow_geofm_specific_matched_dimension_claim"
    if status == "mechanism_claim_narrowed":
        return "narrow_to_low_dimensional_route"
    if status == "low_dimensional_route_uncertain":
        return "do_not_claim_compressed_route_until_resolved"
    return "insufficient_evidence"


def _next_experiment_recommendation(status: str) -> str:
    if status == "mechanism_claim_narrowed":
        return "optional_d6_geofm_projection_controls_before_stronger_mechanism_claim"
    if status == "low_dimensional_route_uncertain":
        return "repair_or_repeat_core_compressed_route_evidence"
    if status == "geofm_specific_information_supported":
        return "no_additional_matched_dimension_control_required_for_this_claim"
    return "repair_missing_input_artifacts"


def _phase60_conclusion(status: str) -> str:
    if status == "geofm_specific_information_supported":
        return (
            "Phase 60 conclusion: all current axes support a GeoFM-specific "
            "matched-dimension information claim."
        )
    if status == "mechanism_claim_narrowed":
        return (
            "Phase 60 conclusion: the compressed low-dimensional route remains "
            "supported, but the current evidence does not support a GeoFM-specific "
            "matched-dimension advantage."
        )
    if status == "low_dimensional_route_uncertain":
        return (
            "Phase 60 conclusion: the low-dimensional compressed route is not "
            "sufficiently supported by the required upstream axes."
        )
    return "Phase 60 conclusion: insufficient input evidence for attribution."


def build_phase60_information_optimization_attribution_from_paths(
    phase48_json: Path | str,
    phase53_json: Path | str,
    phase57_json: Path | str,
    phase59_json: Path | str,
) -> dict[str, object]:
    paths = {
        "phase48": str(Path(phase48_json)),
        "phase53": str(Path(phase53_json)),
        "phase57": str(Path(phase57_json)),
        "phase59": str(Path(phase59_json)),
    }
    return build_phase60_information_optimization_attribution(
        phase48=_load_json(phase48_json, "Phase 48 JSON"),
        phase53=_load_json(phase53_json, "Phase 53 JSON"),
        phase57=_load_json(phase57_json, "Phase 57 JSON"),
        phase59=_load_json(phase59_json, "Phase 59 JSON"),
        source_paths=paths,
    )


def write_phase60_information_optimization_attribution_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_path = output_path / "phase60_information_optimization_attribution.json"
    axes_path = output_path / "phase60_attribution_axes.csv"
    readiness_path = output_path / "phase60_information_optimization_attribution.md"

    comparison_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_axes_csv(axes_path, analysis.get("attribution_axes"))
    readiness_path.write_text(_phase60_markdown(analysis), encoding="utf-8")
    return {
        "comparison_json": comparison_path,
        "axes_csv": axes_path,
        "readiness_md": readiness_path,
    }


def _load_json(path_or_str: Path | str, label: str) -> dict[str, object]:
    path = Path(path_or_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _write_axes_csv(path: Path, rows: object) -> None:
    if not isinstance(rows, list):
        raise ValueError("Phase 60 analysis is missing attribution_axes")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PHASE60_AXIS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Phase 60 attribution axis rows must be objects")
            writer.writerow(
                {field: row.get(field, "") for field in PHASE60_AXIS_FIELDNAMES}
            )


def _phase60_markdown(analysis: Mapping[str, object]) -> str:
    axes = analysis.get("attribution_axes")
    if not isinstance(axes, list):
        axes = []
    lines = [
        "# Phase 60 Information-vs-Optimization Attribution",
        "",
        f"Status: {analysis.get('phase60_attribution_status', '')}",
        "",
        "Attribution conclusion:",
        str(analysis.get("conclusion", "")),
        "",
        "Attribution axes:",
    ]
    for row in axes:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "- "
            f"{row.get('axis_id')}: {row.get('axis_status')} "
            f"({row.get('primary_metric')}={row.get('primary_value')}). "
            f"{row.get('interpretation')}"
        )
    lines.extend(
        [
            "",
            "Claim-boundary recommendation:",
            str(analysis.get("claim_boundary_recommendation", "")),
            "",
            "Next-experiment recommendation:",
            str(analysis.get("next_experiment_recommendation", "")),
            "",
            "Claim boundary:",
            str(analysis.get("claim_boundary", PHASE60_CLAIM_BOUNDARY)),
            "",
        ]
    )
    return "\n".join(lines)


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
