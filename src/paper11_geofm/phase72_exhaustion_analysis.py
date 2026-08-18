from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
from pathlib import Path

from .phase72b_protocol import canonical_json_sha256


PHASE72_EXHAUSTION_CLAIM_BOUNDARY = (
    "Phase 72 exhaustion analysis is a read-only audit of the completed Phase "
    "72A, Phase 72B, and separately frozen exhaustion evidence. It does not "
    "train Phase 72C, run planning, alter rewards, revise the formal "
    "manuscript, or establish a complete scientific exhaustion of every "
    "future-aware GeoFM design."
)

CRITERION_FIELDS = (
    "criterion_id",
    "criterion_class",
    "criterion_status",
    "description",
    "evidence",
    "quantified_result",
    "source_artifacts",
    "claim_boundary",
)

CLAIM_FIELDS = (
    "claim_id",
    "claim_status",
    "decision_reason",
    "supporting_criteria",
    "blocking_criteria",
    "claim_boundary",
)

HASH_FIELDS = (
    "artifact_name",
    "expected_sha256",
    "actual_sha256",
    "hash_status",
)

_RECEIPT_REQUIRED_ARTIFACTS = {
    "phase72b_metrics.csv",
    "phase72b_predictions.csv",
    "phase72b_calibration.csv",
    "phase72b_bootstrap_deltas.csv",
    "phase72b_control_comparison.csv",
    "phase72b_confirmation_control_manifest.csv",
    "phase72b_transfer_summary.csv",
    "phase72b_information_gain_screen.json",
    "phase72b_information_gain_screen.md",
}

_TWO_YEAR_RECEIPT_REQUIRED_ARTIFACTS = {
    "phase72_two_year_metrics.csv",
    "phase72_two_year_predictions.csv",
    "phase72_two_year_bootstrap_deltas.csv",
    "phase72_two_year_control_comparison.csv",
    "phase72_two_year_transfer_summary.csv",
    "phase72_two_year_spatial_summary.csv",
    "phase72_two_year_endpoint_screen.json",
    "phase72_two_year_endpoint_screen.md",
}

_RESIDUAL_RECEIPT_REQUIRED_ARTIFACTS = {
    "phase72_explicit_residual_metrics.csv",
    "phase72_explicit_residual_predictions.csv",
    "phase72_explicit_residual_bootstrap_deltas.csv",
    "phase72_explicit_residual_control_comparison.csv",
    "phase72_explicit_residual_transfer_summary.csv",
    "phase72_explicit_residual_spatial_summary.csv",
    "phase72_explicit_residual_screen.json",
    "phase72_explicit_residual_screen.md",
}


def build_phase72_exhaustion_analysis(
    *,
    phase72a_json: Path | str,
    phase72a_summary_csv: Path | str,
    phase72a_review_csv: Path | str,
    phase72b_json: Path | str,
    phase72b_protocol_json: Path | str,
    phase72b_metrics_csv: Path | str,
    phase72b_control_csv: Path | str,
    phase72b_transfer_csv: Path | str,
    phase72b_receipt_json: Path | str,
    phase72b_receipt_sha256: Path | str,
    phase72b_confirmation_dir: Path | str,
    phase72_two_year_json: Path | str | None = None,
    phase72_two_year_receipt_json: Path | str | None = None,
    phase72_two_year_receipt_sha256: Path | str | None = None,
    phase72_two_year_confirmation_dir: Path | str | None = None,
    phase72_residual_json: Path | str | None = None,
    phase72_residual_receipt_json: Path | str | None = None,
    phase72_residual_receipt_sha256: Path | str | None = None,
    phase72_residual_confirmation_dir: Path | str | None = None,
) -> dict[str, object]:
    phase72a = _read_json_object(phase72a_json, "Phase 72A JSON")
    phase72b = _read_json_object(phase72b_json, "Phase 72B JSON")
    protocol = _read_json_object(phase72b_protocol_json, "Phase 72B protocol JSON")
    receipt = _read_json_object(phase72b_receipt_json, "Phase 72B receipt JSON")
    summary_rows = _read_csv_rows(phase72a_summary_csv, "Phase 72A summary CSV")
    review_rows = _read_csv_rows(phase72a_review_csv, "Phase 72A review CSV")
    metric_rows = _read_csv_rows(phase72b_metrics_csv, "Phase 72B metrics CSV")
    control_rows = _read_csv_rows(phase72b_control_csv, "Phase 72B control CSV")
    transfer_rows = _read_csv_rows(phase72b_transfer_csv, "Phase 72B transfer CSV")

    phase72a_status = _required_status(phase72a, "phase72a_status", "Phase 72A JSON")
    phase72b_status = _required_status(phase72b, "phase72b_status", "Phase 72B JSON")
    receipt_status = _required_status(
        receipt, "phase72b_status", "Phase 72B receipt JSON"
    )

    hash_rows, integrity_blockers = _audit_receipt_artifacts(
        receipt,
        Path(phase72b_confirmation_dir),
        required_artifacts=_RECEIPT_REQUIRED_ARTIFACTS,
        label="Phase 72B",
    )
    receipt_hash_row, receipt_hash_blocker = _audit_receipt_self(
        receipt,
        Path(phase72b_receipt_sha256),
        label="Phase 72B",
    )
    hash_rows.append(receipt_hash_row)
    if receipt_hash_blocker:
        integrity_blockers.append(receipt_hash_blocker)
    if receipt_status != phase72b_status:
        integrity_blockers.append(
            "Phase 72B receipt status does not match the information-gain JSON"
        )

    two_year_paths = (
        phase72_two_year_json,
        phase72_two_year_receipt_json,
        phase72_two_year_receipt_sha256,
        phase72_two_year_confirmation_dir,
    )
    two_year_evidence = None
    if any(path is not None for path in two_year_paths):
        if any(path is None for path in two_year_paths):
            raise ValueError(
                "All Phase 72 two-year evidence paths must be supplied together"
            )
        two_year = _read_json_object(
            phase72_two_year_json, "Phase 72 two-year JSON"
        )
        two_year_receipt = _read_json_object(
            phase72_two_year_receipt_json,
            "Phase 72 two-year receipt JSON",
        )
        two_year_status = _required_status(
            two_year,
            "phase72_two_year_status",
            "Phase 72 two-year JSON",
        )
        receipt_two_year_status = _required_status(
            two_year_receipt,
            "phase72_two_year_status",
            "Phase 72 two-year receipt JSON",
        )
        two_year_hash_rows, two_year_blockers = _audit_receipt_artifacts(
            two_year_receipt,
            Path(phase72_two_year_confirmation_dir),
            required_artifacts=_TWO_YEAR_RECEIPT_REQUIRED_ARTIFACTS,
            label="Phase 72 two-year",
        )
        two_year_receipt_hash_row, two_year_receipt_hash_blocker = (
            _audit_receipt_self(
                two_year_receipt,
                Path(phase72_two_year_receipt_sha256),
                label="Phase 72 two-year",
                receipt_name="phase72_two_year_confirmation_receipt.json",
            )
        )
        two_year_hash_rows.append(two_year_receipt_hash_row)
        hash_rows.extend(two_year_hash_rows)
        integrity_blockers.extend(two_year_blockers)
        if two_year_receipt_hash_blocker:
            integrity_blockers.append(two_year_receipt_hash_blocker)
        if receipt_two_year_status != two_year_status:
            integrity_blockers.append(
                "Phase 72 two-year receipt status does not match the screen JSON"
            )
        endpoint_results = two_year.get("endpoint_results", {})
        if not isinstance(endpoint_results, Mapping):
            raise ValueError(
                "Phase 72 two-year endpoint results must be an object"
            )
        two_year_evidence = {
            "status": two_year_status,
            "endpoint_statuses": {
                str(endpoint): str(result.get("phase72b_status", ""))
                for endpoint, result in endpoint_results.items()
                if isinstance(result, Mapping)
            },
            "counts": dict(two_year.get("counts", {})),
            "receipt_artifact_rows": len(two_year_hash_rows),
        }

    residual_paths = (
        phase72_residual_json,
        phase72_residual_receipt_json,
        phase72_residual_receipt_sha256,
        phase72_residual_confirmation_dir,
    )
    residual_evidence = None
    if any(path is not None for path in residual_paths):
        if any(path is None for path in residual_paths):
            raise ValueError(
                "All Phase 72 explicit residual evidence paths must be supplied together"
            )
        residual = _read_json_object(
            phase72_residual_json, "Phase 72 explicit residual JSON"
        )
        residual_receipt = _read_json_object(
            phase72_residual_receipt_json,
            "Phase 72 explicit residual receipt JSON",
        )
        residual_status = _required_status(
            residual,
            "phase72_explicit_residual_status",
            "Phase 72 explicit residual JSON",
        )
        receipt_residual_status = _required_status(
            residual_receipt,
            "phase72_explicit_residual_status",
            "Phase 72 explicit residual receipt JSON",
        )
        residual_hash_rows, residual_blockers = _audit_receipt_artifacts(
            residual_receipt,
            Path(phase72_residual_confirmation_dir),
            required_artifacts=_RESIDUAL_RECEIPT_REQUIRED_ARTIFACTS,
            label="Phase 72 explicit residual",
        )
        residual_receipt_hash_row, residual_receipt_hash_blocker = (
            _audit_receipt_self(
                residual_receipt,
                Path(phase72_residual_receipt_sha256),
                label="Phase 72 explicit residual",
                receipt_name=(
                    "phase72_explicit_residual_confirmation_receipt.json"
                ),
            )
        )
        residual_hash_rows.append(residual_receipt_hash_row)
        hash_rows.extend(residual_hash_rows)
        integrity_blockers.extend(residual_blockers)
        residual_result_entries = [
            dict(entry)
            for entry in residual_receipt.get("artifacts", [])
            if isinstance(entry, Mapping)
            and str(entry.get("name"))
            == "phase72_explicit_residual_screen.json"
        ]
        if (
            len(residual_result_entries) != 1
            or _sha256(Path(phase72_residual_json))
            != str(residual_result_entries[0].get("sha256", "")).lower()
        ):
            integrity_blockers.append(
                "Phase 72 explicit residual JSON is not the receipt-bound result artifact"
            )
        if residual_receipt_hash_blocker:
            integrity_blockers.append(residual_receipt_hash_blocker)
        if receipt_residual_status != residual_status:
            integrity_blockers.append(
                "Phase 72 explicit residual receipt status does not match the screen JSON"
            )
        for field in ("prepared_sha256", "selected_models_sha256"):
            if residual_receipt.get(field) != residual.get(field):
                integrity_blockers.append(
                    f"Phase 72 explicit residual receipt binding mismatch: {field}"
                )
        if (
            residual.get("phase72c_allowed") is not False
            or residual_receipt.get("phase72c_allowed") is not False
        ):
            integrity_blockers.append(
                "Phase 72 explicit residual evidence must keep Phase 72C closed"
            )
        endpoint_results = residual.get("endpoint_results", {})
        if not isinstance(endpoint_results, Mapping):
            raise ValueError(
                "Phase 72 explicit residual endpoint results must be an object"
            )
        residual_evidence = {
            "status": residual_status,
            "endpoint_statuses": {
                str(endpoint): str(result.get("phase72b_status", ""))
                for endpoint, result in endpoint_results.items()
                if isinstance(result, Mapping)
            },
            "counts": dict(residual.get("counts", {})),
            "receipt_artifact_rows": len(residual_hash_rows),
        }

    labels = _label_evidence(phase72a, summary_rows, review_rows)
    models = _model_evidence(metric_rows, protocol)
    screen = _screen_evidence(phase72b, protocol, control_rows, transfer_rows)
    criteria = _criteria_rows(
        labels,
        models,
        screen,
        phase72a_status,
        phase72b_status,
        two_year_evidence,
        residual_evidence,
    )
    claims = _claim_boundary_rows(criteria, integrity_blockers, phase72b_status)

    unresolved = [
        row["criterion_id"]
        for row in criteria
        if row["criterion_status"] in {"data_gap", "not_evaluated", "partially_evaluated"}
    ]
    negative = [
        row["criterion_id"]
        for row in criteria
        if row["criterion_status"] in {"evaluated_negative", "evaluated_mixed"}
    ]
    if integrity_blockers or phase72a_status != "phase72a_label_inputs_ready":
        exhaustion_status = "phase72_exhaustion_inputs_not_ready"
    elif unresolved:
        exhaustion_status = "phase72_exhaustion_criteria_not_fully_evaluated"
    else:
        exhaustion_status = "phase72_exhaustion_criteria_evaluated"

    route_closed = phase72b_status == "geofm_information_not_supported"
    return {
        "phase": "phase72_exhaustion_analysis",
        "phase72_exhaustion_status": exhaustion_status,
        "phase72b_status": phase72b_status,
        "route_decision": (
            "phase72_route_closed_at_phase72b_gate"
            if route_closed
            else "phase72_route_requires_review"
        ),
        "phase72c_allowed": False,
        "phase72a_status": phase72a_status,
        "source_paths": {
            "phase72a_json": str(Path(phase72a_json)),
            "phase72a_summary_csv": str(Path(phase72a_summary_csv)),
            "phase72a_review_csv": str(Path(phase72a_review_csv)),
            "phase72b_json": str(Path(phase72b_json)),
            "phase72b_protocol_json": str(Path(phase72b_protocol_json)),
            "phase72b_metrics_csv": str(Path(phase72b_metrics_csv)),
            "phase72b_control_csv": str(Path(phase72b_control_csv)),
            "phase72b_transfer_csv": str(Path(phase72b_transfer_csv)),
            "phase72b_receipt_json": str(Path(phase72b_receipt_json)),
            "phase72b_receipt_sha256": str(Path(phase72b_receipt_sha256)),
            "phase72b_confirmation_dir": str(Path(phase72b_confirmation_dir)),
            **(
                {
                    "phase72_two_year_json": str(Path(phase72_two_year_json)),
                    "phase72_two_year_receipt_json": str(
                        Path(phase72_two_year_receipt_json)
                    ),
                    "phase72_two_year_receipt_sha256": str(
                        Path(phase72_two_year_receipt_sha256)
                    ),
                    "phase72_two_year_confirmation_dir": str(
                        Path(phase72_two_year_confirmation_dir)
                    ),
                }
                if two_year_evidence is not None
                else {}
            ),
            **(
                {
                    "phase72_residual_json": str(Path(phase72_residual_json)),
                    "phase72_residual_receipt_json": str(
                        Path(phase72_residual_receipt_json)
                    ),
                    "phase72_residual_receipt_sha256": str(
                        Path(phase72_residual_receipt_sha256)
                    ),
                    "phase72_residual_confirmation_dir": str(
                        Path(phase72_residual_confirmation_dir)
                    ),
                }
                if residual_evidence is not None
                else {}
            ),
        },
        "counts": {
            "criteria": len(criteria),
            "claims": len(claims),
            "unresolved_criteria": len(unresolved),
            "negative_or_mixed_criteria": len(negative),
            "integrity_blockers": len(integrity_blockers),
            "receipt_hash_rows": len(hash_rows),
            **labels["counts"],
            **models["counts"],
            **screen["counts"],
        },
        "label_evidence": labels,
        "model_evidence": models,
        "screen_evidence": screen,
        "two_year_evidence": two_year_evidence,
        "residual_evidence": residual_evidence,
        "artifact_hash_rows": hash_rows,
        "integrity_blockers": integrity_blockers,
        "criteria_rows": criteria,
        "claim_boundary_rows": claims,
        "unresolved_criteria": unresolved,
        "negative_or_mixed_criteria": negative,
        "recommended_next_step": (
            "Do not begin Phase 72C. Preserve the Phase 72B negative gate and "
            "the negative two-year result; treat the explicit residual result "
            "as mixed rather than stable support, and record the remaining "
            "product, noise, temporal-neural, and planning evidence as unresolved."
        ),
        "claim_boundary": PHASE72_EXHAUSTION_CLAIM_BOUNDARY,
    }


def write_phase72_exhaustion_analysis_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "criteria_csv": output_path / "phase72_exhaustion_criteria.csv",
        "claim_boundary_csv": output_path / "phase72_exhaustion_claim_boundary.csv",
        "artifact_hashes_csv": output_path / "phase72_exhaustion_artifact_hashes.csv",
        "analysis_json": output_path / "phase72_exhaustion_analysis.json",
        "analysis_md": output_path / "phase72_exhaustion_analysis.md",
    }
    _write_csv_mapping_rows(
        artifacts["criteria_csv"],
        CRITERION_FIELDS,
        analysis.get("criteria_rows", []),
        "Phase 72 exhaustion criteria",
    )
    _write_csv_mapping_rows(
        artifacts["claim_boundary_csv"],
        CLAIM_FIELDS,
        analysis.get("claim_boundary_rows", []),
        "Phase 72 exhaustion claim boundary",
    )
    _write_csv_mapping_rows(
        artifacts["artifact_hashes_csv"],
        HASH_FIELDS,
        analysis.get("artifact_hash_rows", []),
        "Phase 72B receipt artifact hashes",
    )
    artifacts["analysis_json"].write_text(
        json.dumps(_json_ready(analysis), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    artifacts["analysis_md"].write_text(_markdown(analysis), encoding="utf-8")
    return artifacts


def _label_evidence(
    phase72a: Mapping[str, object],
    summary_rows: Sequence[Mapping[str, str]],
    review_rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    manifest_rows = [
        row
        for audit in phase72a.get("region_audits", [])
        if isinstance(audit, Mapping)
        for row in audit.get("file_manifest_rows", [])
        if isinstance(row, Mapping)
    ]
    label_rows = [
        row
        for row in manifest_rows
        if row.get("asset_type") == "label"
        and _as_bool(row.get("independent_label"))
    ]
    source_ids = sorted(
        {
            str(row.get("source_id", ""))
            for row in label_rows
            if row.get("source_id")
        }
    )
    horizons = sorted({str(row.get("horizon", "")) for row in summary_rows if row.get("horizon")})
    region_ids = sorted({str(row.get("region_id", "")) for row in summary_rows if row.get("region_id")})
    review_fields = ("review_label", "review_source", "review_source_id", "review_date", "review_confidence")
    reviewed_rows = sum(
        any(str(row.get(field, "")).strip() for field in review_fields)
        for row in review_rows
    )
    return {
        "label_source_ids": source_ids,
        "independent_label_source_count": len(source_ids),
        "horizons_in_label_package": horizons,
        "regions_in_label_package": region_ids,
        "manual_review_rows": len(review_rows),
        "manual_review_completed_rows": reviewed_rows,
        "phase72a_sample_rows": phase72a.get("row_counts", {}).get("sample_rows", 0),
        "counts": {
            "independent_label_sources": len(source_ids),
            "label_package_horizons": len(horizons),
            "label_package_regions": len(region_ids),
            "manual_review_rows": len(review_rows),
            "manual_review_completed_rows": reviewed_rows,
        },
    }


def _model_evidence(
    metric_rows: Sequence[Mapping[str, str]],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    variants = sorted({str(row.get("variant_id", "")) for row in metric_rows if row.get("variant_id")})
    families = sorted({str(row.get("model_family", "")) for row in metric_rows if row.get("model_family")})
    expected = [str(value) for value in protocol.get("variants", [])]
    missing_expected = sorted(set(expected) - set(variants))
    return {
        "variants": variants,
        "model_families": families,
        "expected_variants": expected,
        "missing_expected_variants": missing_expected,
        "has_residual_variant": any("residual" in value.lower() for value in variants),
        "has_temporal_neural_family": any(
            token in family.lower()
            for family in families
            for token in ("mlp", "neural", "transformer", "temporal_net")
        ),
        "counts": {
            "metric_rows": len(metric_rows),
            "variants": len(variants),
            "model_families": len(families),
        },
    }


def _screen_evidence(
    phase72b: Mapping[str, object],
    protocol: Mapping[str, object],
    control_rows: Sequence[Mapping[str, str]],
    transfer_rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    evidence = phase72b.get("evidence", {})
    controls = [row for row in evidence.get("controls", []) if isinstance(row, Mapping)]
    transfers = [row for row in evidence.get("transfers", []) if isinstance(row, Mapping)]
    spatial_regions = [row for row in evidence.get("spatial_regions", []) if isinstance(row, Mapping)]
    spatial_folds = [row for row in evidence.get("spatial_folds", []) if isinstance(row, Mapping)]
    temporal = next((row for row in controls if row.get("control_id") == "temporal_order_shuffle"), {})
    return {
        "protocol_years": protocol.get("years", {}),
        "protocol_spatial": protocol.get("spatial", {}),
        "protocol_control_seeds": protocol.get("controls", {}).get("seeds", []),
        "phase72b_counts": phase72b.get("counts", {}),
        "controls": controls,
        "transfers": transfers,
        "spatial_regions": spatial_regions,
        "spatial_fold_count": len(spatial_folds),
        "control_csv_rows": len(control_rows),
        "transfer_csv_rows": len(transfer_rows),
        "temporal_order_control_passed": bool(temporal.get("passed", False)),
        "all_transfer_axes_passed": bool(transfers) and all(row.get("passed") is True for row in transfers),
        "all_spatial_regions_passed": bool(spatial_regions) and all(row.get("passed") is True for row in spatial_regions),
        "counts": {
            "control_axes": len(controls),
            "transfer_axes": len(transfers),
            "spatial_regions": len(spatial_regions),
            "spatial_folds": len(spatial_folds),
        },
    }


def _criteria_rows(
    labels: Mapping[str, object],
    models: Mapping[str, object],
    screen: Mapping[str, object],
    phase72a_status: str,
    phase72b_status: str,
    two_year_evidence: Mapping[str, object] | None = None,
    residual_evidence: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    source_count = int(labels["independent_label_source_count"])
    horizons = set(labels["horizons_in_label_package"])
    variants = set(models["variants"])
    families = set(models["model_families"])
    controls = {str(row.get("control_id")): row for row in screen["controls"]}
    transfers = screen["transfers"]
    spatial_regions = screen["spatial_regions"]
    two_year_status = (
        "" if two_year_evidence is None else str(two_year_evidence["status"])
    )
    residual_status = (
        "" if residual_evidence is None else str(residual_evidence["status"])
    )
    if two_year_evidence is None:
        horizon_status = (
            "partially_evaluated"
            if {"1y", "2y", "continuous_2y"}.issubset(horizons)
            else "data_gap"
        )
        horizon_evidence = (
            f"Label package contains horizons: {', '.join(sorted(horizons))}; "
            "Phase 72B metrics are one-year only."
        )
    else:
        horizon_status = "evaluated_complete"
        horizon_evidence = (
            f"Label package horizons: {', '.join(sorted(horizons))}; "
            f"separate two-year screen status: {two_year_status}."
        )
    rows = [
        _criterion(
            "independent_annual_products",
            "label_coverage",
            "data_gap" if source_count < 2 else "evaluated_complete",
            "At least two independent annual land-cover products where accessible.",
            f"{source_count} independent source(s): {', '.join(labels['label_source_ids']) or 'none'}.",
            source_count,
            "phase72a_temporal_label_package.json",
        ),
        _criterion(
            "one_and_two_year_endpoints",
            "horizon_coverage",
            horizon_status,
            "One-year and two-year outcomes must both be evaluated, not only assembled.",
            horizon_evidence,
            ",".join(sorted(horizons)),
            "phase72a_package_summary.csv;phase72b_metrics.csv;phase72_two_year_endpoint_screen.json",
        ),
        _criterion(
            "two_year_prediction_outcome_gate",
            "prediction_outcome",
            (
                "not_evaluated"
                if two_year_evidence is None
                else "evaluated_negative"
                if two_year_status == "two_year_geofm_information_not_supported"
                else "evaluated_mixed"
                if two_year_status == "two_year_geofm_information_mixed"
                else "evaluated_complete"
                if two_year_status == "two_year_geofm_information_supported"
                else "not_evaluated"
            ),
            "Both frozen two-year endpoints must support representation-specific GeoFM information.",
            (
                "No separate two-year screen was supplied."
                if two_year_evidence is None
                else f"Separate two-year screen status: {two_year_status}; endpoint statuses: {two_year_evidence['endpoint_statuses']}."
            ),
            two_year_status or "absent",
            "phase72_two_year_endpoint_screen.json;phase72_two_year_confirmation_receipt.json",
        ),
        _criterion(
            "explicit_residual_model",
            "model_coverage",
            (
                "evaluated_mixed"
                if residual_status == "explicit_residual_information_mixed"
                else "evaluated_negative"
                if residual_status
                == "explicit_residual_information_not_supported"
                else "evaluated_complete"
                if residual_status == "explicit_residual_information_supported"
                else "evaluated_complete"
                if any("residual" in value.lower() for value in variants)
                else "not_evaluated"
            ),
            "An explicit residual risk model must be evaluated.",
            (
                "No separate explicit residual screen was supplied."
                if residual_evidence is None
                and not any("residual" in value.lower() for value in variants)
                else "Residual variant present in the Phase 72B metrics."
                if residual_evidence is None
                else (
                    f"Explicit residual screen status: {residual_status}; "
                    f"endpoint statuses: {residual_evidence['endpoint_statuses']}."
                )
            ),
            residual_status
            or (
                "residual"
                if any("residual" in value.lower() for value in variants)
                else "absent"
            ),
            (
                "phase72_explicit_residual_screen.json;"
                "phase72_explicit_residual_confirmation_receipt.json"
                if residual_evidence is not None
                else "phase72b_metrics.csv;phase72b_protocol.json"
            ),
        ),
        _criterion(
            "temporal_neural_model",
            "model_coverage",
            "evaluated_complete" if any(token in family.lower() for family in families for token in ("mlp", "neural", "transformer", "temporal_net")) else "not_evaluated",
            "A temporal neural model must be evaluated before claiming full GeoFM-STaR exhaustion.",
            f"Frozen model families: {', '.join(sorted(families)) or 'none'}.",
            ",".join(sorted(families)),
            "phase72b_metrics.csv",
        ),
        _criterion(
            "bidirectional_cross_region_transfer",
            "validation_coverage",
            "evaluated_negative" if transfers and not screen["all_transfer_axes_passed"] else "evaluated_complete",
            "Both Bishan-to-Dongxing and Dongxing-to-Bishan transfer axes must be evaluated.",
            f"{len(transfers)} transfer axes; all passed={screen['all_transfer_axes_passed']}.",
            len(transfers),
            "phase72b_transfer_summary.csv;phase72b_information_gain_screen.json",
        ),
        _criterion(
            "buffered_spatial_validation",
            "validation_coverage",
            "evaluated_mixed" if spatial_regions and not screen["all_spatial_regions_passed"] else "evaluated_complete",
            "Buffered spatial validation must cover both regions and remain stable.",
            f"{len(screen['spatial_regions'])} region summaries and {screen['spatial_fold_count']} folds; all regions passed={screen['all_spatial_regions_passed']}.",
            f"regions={len(spatial_regions)},folds={screen['spatial_fold_count']}",
            "phase72b_information_gain_screen.json;phase72b_protocol.json",
        ),
        _criterion(
            "strict_temporal_and_representation_controls",
            "control_coverage",
            "evaluated_negative" if controls and not screen["temporal_order_control_passed"] else "evaluated_complete",
            "Temporal-order, spatial-shuffle, and same-dimension random controls must be evaluated.",
            f"{len(controls)} control axes; temporal-order control passed={screen['temporal_order_control_passed']}.",
            ",".join(sorted(controls)),
            "phase72b_control_comparison.csv;phase72b_information_gain_screen.json",
        ),
        _criterion(
            "label_resolution_disagreement_noise_sensitivity",
            "label_quality",
            "not_evaluated" if int(labels["manual_review_completed_rows"]) == 0 or source_count < 2 else "partially_evaluated",
            "Label-source disagreement, resolution, and noise sensitivity must be audited.",
            f"{labels['manual_review_completed_rows']} of {labels['manual_review_rows']} manual-review rows completed; {source_count} product source(s).",
            f"reviewed={labels['manual_review_completed_rows']}/{labels['manual_review_rows']},sources={source_count}",
            "phase72a_manual_review_frame.csv;phase72a_temporal_label_package.json",
        ),
        _criterion(
            "prediction_outcome_gate",
            "prediction_outcome",
            "evaluated_negative" if phase72b_status == "geofm_information_not_supported" else "not_evaluated",
            "The low-cost prediction gate must support representation-specific GeoFM information.",
            f"Official Phase 72B status: {phase72b_status}.",
            phase72b_status,
            "phase72b_information_gain_screen.json",
        ),
        _criterion(
            "constrained_planning_outcomes",
            "planning_coverage",
            "not_evaluated",
            "Future observed persistence under constrained planning must be evaluated.",
            "Phase 72B claim boundary and artifacts contain no planning run or hidden-outcome planner comparison.",
            "absent",
            "phase72b_information_gain_screen.json;phase72b_confirmation_receipt.json",
        ),
    ]
    return rows


def _claim_boundary_rows(
    criteria: Sequence[Mapping[str, object]],
    integrity_blockers: Sequence[str],
    phase72b_status: str,
) -> list[dict[str, object]]:
    statuses = {str(row["criterion_id"]): str(row["criterion_status"]) for row in criteria}
    unresolved = [key for key, value in statuses.items() if value in {"data_gap", "not_evaluated", "partially_evaluated"}]
    negative = [key for key, value in statuses.items() if value in {"evaluated_negative", "evaluated_mixed"}]
    return [
        {
            "claim_id": "phase72b_low_cost_screen_negative",
            "claim_status": "allowed" if phase72b_status == "geofm_information_not_supported" and not integrity_blockers else "blocked",
            "decision_reason": "The receipt-bound Phase 72B screen is complete and officially negative." if not integrity_blockers else "Receipt integrity blockers prevent relying on the Phase 72B screen.",
            "supporting_criteria": "prediction_outcome_gate,strict_temporal_and_representation_controls,bidirectional_cross_region_transfer",
            "blocking_criteria": "",
            "claim_boundary": PHASE72_EXHAUSTION_CLAIM_BOUNDARY,
        },
        {
            "claim_id": "complete_future_aware_geofm_exhaustion",
            "claim_status": "blocked",
            "decision_reason": "The exhaustion criteria still contain unresolved data, model, label-quality, horizon, or planning axes.",
            "supporting_criteria": ",".join(negative),
            "blocking_criteria": ",".join(unresolved),
            "claim_boundary": PHASE72_EXHAUSTION_CLAIM_BOUNDARY,
        },
        {
            "claim_id": "phase72c_geofm_star_training",
            "claim_status": "blocked",
            "decision_reason": "The Phase 72B gate is negative and the approved transition explicitly forbids Phase 72C.",
            "supporting_criteria": "prediction_outcome_gate",
            "blocking_criteria": "strict_temporal_and_representation_controls,prediction_outcome_gate",
            "claim_boundary": PHASE72_EXHAUSTION_CLAIM_BOUNDARY,
        },
        {
            "claim_id": "future_stability_planning_claim",
            "claim_status": "blocked",
            "decision_reason": "No constrained planner or hidden future-outcome evaluation was run.",
            "supporting_criteria": "",
            "blocking_criteria": "constrained_planning_outcomes",
            "claim_boundary": PHASE72_EXHAUSTION_CLAIM_BOUNDARY,
        },
        {
            "claim_id": "formal_submission_revision",
            "claim_status": "out_of_scope",
            "decision_reason": "Phase 72 exhaustion analysis is not a formal manuscript revision stage.",
            "supporting_criteria": "",
            "blocking_criteria": "",
            "claim_boundary": PHASE72_EXHAUSTION_CLAIM_BOUNDARY,
        },
    ]


def _criterion(
    criterion_id: str,
    criterion_class: str,
    criterion_status: str,
    description: str,
    evidence: str,
    quantified_result: object,
    source_artifacts: str,
) -> dict[str, object]:
    return {
        "criterion_id": criterion_id,
        "criterion_class": criterion_class,
        "criterion_status": criterion_status,
        "description": description,
        "evidence": evidence,
        "quantified_result": quantified_result,
        "source_artifacts": source_artifacts,
        "claim_boundary": PHASE72_EXHAUSTION_CLAIM_BOUNDARY,
    }


def _audit_receipt_artifacts(
    receipt: Mapping[str, object],
    confirmation_dir: Path,
    *,
    required_artifacts: set[str],
    label: str,
) -> tuple[list[dict[str, str]], list[str]]:
    artifact_entries = receipt.get("artifacts", [])
    if not isinstance(artifact_entries, Sequence) or isinstance(artifact_entries, (str, bytes)):
        return [], [f"{label} receipt artifacts must be a list"]
    names = {str(row.get("name")) for row in artifact_entries if isinstance(row, Mapping)}
    blockers = []
    if names != required_artifacts:
        blockers.append(
            f"{label} receipt artifact set does not match the required files"
        )
    rows = []
    for entry in artifact_entries:
        if not isinstance(entry, Mapping):
            blockers.append(f"{label} receipt contains a non-object artifact entry")
            continue
        name = str(entry.get("name", ""))
        expected = str(entry.get("sha256", "")).lower()
        safe_name = bool(name) and Path(name).name == name
        path = confirmation_dir / name if safe_name else confirmation_dir
        actual = _sha256(path) if safe_name and path.is_file() else ""
        status = "match" if expected and actual == expected else "missing_or_mismatch"
        if not safe_name:
            blockers.append(f"Receipt artifact name is not a safe basename: {name}")
        if status != "match":
            blockers.append(f"Receipt artifact hash mismatch or missing: {name}")
        rows.append({"artifact_name": name, "expected_sha256": expected, "actual_sha256": actual, "hash_status": status})
    return rows, blockers


def _audit_receipt_self(
    receipt: Mapping[str, object],
    receipt_hash_path: Path,
    *,
    label: str,
    receipt_name: str = "phase72b_confirmation_receipt.json",
) -> tuple[dict[str, str], str]:
    expected = ""
    if receipt_hash_path.is_file():
        expected = receipt_hash_path.read_text(encoding="ascii").strip().lower()
    actual = canonical_json_sha256(receipt)
    status = "match" if len(expected) == 64 and expected == actual else "missing_or_mismatch"
    row = {
        "artifact_name": receipt_name,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "hash_status": status,
    }
    blocker = (
        ""
        if status == "match"
        else f"{label} receipt canonical hash mismatch or sidecar missing"
    )
    return row, blocker


def _read_json_object(path: Path | str, label: str) -> dict[str, object]:
    json_path = Path(path)
    if not json_path.is_file():
        raise ValueError(f"Missing {label}: {json_path}")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {json_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {json_path}")
    return dict(payload)


def _read_csv_rows(path: Path | str, label: str) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise ValueError(f"Missing {label}: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{label} is empty: {csv_path}")
    return [dict(row) for row in rows]


def _required_status(payload: Mapping[str, object], field: str, label: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is missing required status field: {field}")
    return value


def _sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    label: str,
) -> None:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"{label} contains a non-object row")
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _markdown(analysis: Mapping[str, object]) -> str:
    criteria = analysis.get("criteria_rows", [])
    claims = analysis.get("claim_boundary_rows", [])
    lines = [
        "# Phase 72 Exhaustion Analysis",
        "",
        f"Status: `{analysis.get('phase72_exhaustion_status', '')}`",
        f"Route decision: `{analysis.get('route_decision', '')}`",
        "",
        "## Interpretation",
        "",
        "This is a read-only evidence-coverage audit. It separates negative "
        "Phase 72B evidence from criteria that were not evaluated and therefore "
        "does not promote the result to a complete exhaustion claim.",
        "",
        "## Criteria",
        "",
        "| Criterion | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in criteria if isinstance(criteria, Sequence) else []:
        if isinstance(row, Mapping):
            lines.append(f"| `{row.get('criterion_id', '')}` | `{row.get('criterion_status', '')}` | {row.get('evidence', '')} |")
    lines.extend(["", "## Claim Boundary", "", "| Claim | Status | Reason |", "| --- | --- | --- |"])
    for row in claims if isinstance(claims, Sequence) else []:
        if isinstance(row, Mapping):
            lines.append(f"| `{row.get('claim_id', '')}` | `{row.get('claim_status', '')}` | {row.get('decision_reason', '')} |")
    lines.extend(
        [
            "",
            "## Required Transition",
            "",
            str(analysis.get("recommended_next_step", "")),
            "",
            "## Claim Boundary",
            "",
            str(analysis.get("claim_boundary", "")),
            "",
        ]
    )
    return "\n".join(lines)
