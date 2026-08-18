from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path


PHASE72_CLAIM_DRIFT_CLAIM_BOUNDARY = (
    "Phase 72 claim-drift audit is a read-only comparison between formal "
    "manuscript wording and later Paper11 evidence. It does not edit the "
    "manuscript, train Phase 72C, alter rewards, or promote blocked GeoFM, "
    "suitability, transfer, or future-planning claims."
)

CLAIM_FIELDS = (
    "claim_id",
    "claim_class",
    "claim_status",
    "manuscript_anchor",
    "manuscript_lines",
    "current_evidence",
    "evidence_sources",
    "recommended_action",
    "claim_boundary",
)

_ANCHORS = {
    "abstract_compressed_claim": "These results support a bounded conclusion: GeoFM information improved farmland layout optimization",
    "introduction_compressed_claim": "PCA-compressed GeoFM representations, however, improved held-out learned-policy reward",
    "discussion_compressed_claim": "The main finding is that GeoFM information was useful only after representation control",
    "conclusion_compressed_claim": "GeoFM improved the learned planning policy when represented through controlled compressed state features",
}


def build_phase72_claim_drift_audit(
    *,
    manuscript_md: Path | str,
    phase60_json: Path | str,
    phase62_json: Path | str,
    phase69_json: Path | str,
    phase71_json: Path | str,
    phase72b_json: Path | str,
    phase72_exhaustion_json: Path | str,
) -> dict[str, object]:
    manuscript_path = Path(manuscript_md)
    manuscript_text = _read_text(manuscript_path, "formal manuscript")
    phase60 = _read_json_object(phase60_json, "Phase 60 JSON")
    phase62 = _read_json_object(phase62_json, "Phase 62 JSON")
    phase69 = _read_json_object(phase69_json, "Phase 69 JSON")
    phase71 = _read_json_object(phase71_json, "Phase 71 JSON")
    phase72b = _read_json_object(phase72b_json, "Phase 72B JSON")
    exhaustion = _read_json_object(phase72_exhaustion_json, "Phase 72 exhaustion JSON")

    statuses = {
        "phase60": _required_status(phase60, "phase60_attribution_status", "Phase 60 JSON"),
        "phase62": _required_status(phase62, "phase62_d4_d6_status", "Phase 62 JSON"),
        "phase69": _required_status(phase69, "phase69_status", "Phase 69 JSON"),
        "phase71": _required_status(phase71, "phase71_status", "Phase 71 JSON"),
        "phase72b": _required_status(phase72b, "phase72b_status", "Phase 72B JSON"),
        "exhaustion": _required_status(
            exhaustion,
            "phase72_exhaustion_status",
            "Phase 72 exhaustion JSON",
        ),
    }
    anchors = _find_anchors(manuscript_text)
    claims = _claim_rows(statuses, anchors)
    drift_claims = [
        row["claim_id"]
        for row in claims
        if row["claim_status"] in {"needs_narrowing", "blocked"}
    ]
    missing_anchors = [claim_id for claim_id, lines in anchors.items() if not lines]
    if missing_anchors:
        audit_status = "claim_drift_inputs_incomplete"
    elif any(row["claim_status"] == "needs_narrowing" for row in claims):
        audit_status = "claim_drift_requires_narrowing"
    else:
        audit_status = "formal_claims_aligned"

    return {
        "phase": "phase72_claim_drift_audit",
        "phase72_claim_drift_status": audit_status,
        "formal_manuscript_path": str(manuscript_path),
        "source_paths": {
            "phase60_json": str(Path(phase60_json)),
            "phase62_json": str(Path(phase62_json)),
            "phase69_json": str(Path(phase69_json)),
            "phase71_json": str(Path(phase71_json)),
            "phase72b_json": str(Path(phase72b_json)),
            "phase72_exhaustion_json": str(Path(phase72_exhaustion_json)),
        },
        "source_statuses": statuses,
        "claim_rows": claims,
        "drift_claims": drift_claims,
        "missing_anchors": missing_anchors,
        "counts": {
            "claims": len(claims),
            "drift_claims": len(drift_claims),
            "missing_anchors": len(missing_anchors),
        },
        "recommended_action": (
            "Keep the formal manuscript unchanged for this audit. If a future "
            "revision is authorized, narrow the compressed-route wording to a "
            "Bishan base-reward low-dimensional state result and do not call it "
            "GeoFM-specific, PCA-optimal, transferable, suitable, or future-aware."
        ),
        "claim_boundary": PHASE72_CLAIM_DRIFT_CLAIM_BOUNDARY,
    }


def write_phase72_claim_drift_audit_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "claims_csv": output / "phase72_claim_drift_claims.csv",
        "audit_json": output / "phase72_claim_drift_audit.json",
        "audit_md": output / "phase72_claim_drift_audit.md",
    }
    _write_csv_mapping_rows(
        artifacts["claims_csv"],
        CLAIM_FIELDS,
        analysis.get("claim_rows", []),
        "Phase 72 claim drift rows",
    )
    artifacts["audit_json"].write_text(
        json.dumps(_json_ready(analysis), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    artifacts["audit_md"].write_text(_markdown(analysis), encoding="utf-8")
    return artifacts


def _claim_rows(
    statuses: Mapping[str, str],
    anchors: Mapping[str, Sequence[int]],
) -> list[dict[str, object]]:
    phase62_block = statuses["phase62"] == "d6_random_projection_advantage"
    phase72b_block = statuses["phase72b"] == "geofm_information_not_supported"
    suitability_block = statuses["phase69"] == "claim_must_be_narrowed_to_low_dimensional_route"
    target_masked = statuses["phase71"] == "ranker_improves_but_target_masks_geofm"
    return [
        _claim(
            "real_bishan_planning_workflow",
            "workflow",
            "supported",
            "Formal manuscript describes a reproducible real-Bishan planning workflow; later phases do not contradict this technical capability.",
            "Phase 60/62/71 remain within the real Bishan planning protocol.",
            "phase60,phase62,phase71",
            "retain_with_scope",
            anchors,
        ),
        _claim(
            "bounded_low_dimensional_compressed_route",
            "algorithm_result",
            "bounded_supported",
            "The earlier compressed route remains supported against B0, raw B1, random D2, and shuffled D3 under the Bishan base-reward protocol.",
            "Phase 60 is mechanism_claim_narrowed; the positive route is bounded to the low-dimensional representation setting.",
            "phase60,phase69",
            "retain_bounded_wording",
            anchors,
        ),
        _claim(
            "geofm_specific_compressed_information",
            "attribution",
            "blocked" if phase62_block else "needs_narrowing",
            "A stronger reading that the compressed route is uniquely GeoFM-informative is not supported.",
            "Phase 60 reports matched-dimension GeoFM not supported and Phase 62 reports d6_random_projection_advantage.",
            "phase60,phase62",
            "block_or_replace_with_low_dimensional_route",
            anchors,
        ),
        _claim(
            "pca_optimality",
            "attribution",
            "blocked",
            "PCA is not established as the optimal or uniquely valid compression method.",
            "Phase 62 D4-minus-D6 deltas are negative and the claim boundary explicitly rejects PCA optimality.",
            "phase62,phase69",
            "block",
            anchors,
        ),
        _claim(
            "suitability_reward_or_agronomic_value",
            "suitability",
            "blocked" if suitability_block else "needs_narrowing",
            "The formal manuscript must not imply independent agronomic suitability or B2/B3 reward readiness.",
            "Phase 69 keeps suitability reward and independent agronomic suitability blocked; Phase 71 target gains are explicit-base-target gains.",
            "phase69,phase71",
            "block",
            anchors,
        ),
        _claim(
            "cross_region_transfer",
            "generalization",
            "blocked" if phase72b_block else "needs_narrowing",
            "The result cannot be presented as transferable beyond the current Bishan protocol.",
            "Phase 72B status is geofm_information_not_supported and both zero-shot transfer directions failed.",
            "phase72b,phase72_exhaustion",
            "block",
            anchors,
        ),
        _claim(
            "future_aware_prediction_or_planning",
            "future_target",
            "blocked" if phase72b_block else "needs_narrowing",
            "No supported future-aware GeoFM prediction or planning claim exists.",
            "Phase 72B stopped before Phase 72C and the exhaustion audit records planning outcomes as not_evaluated.",
            "phase72b,phase72_exhaustion",
            "block",
            anchors,
        ),
        _claim(
            "formal_manuscript_current_wording",
            "alignment",
            "needs_narrowing" if (phase62_block or target_masked or phase72b_block) else "supported",
            "Several formal sentences use broad 'GeoFM information improved' wording while later controls narrow attribution to a low-dimensional Bishan route.",
            "Later evidence requires explicit separation of compressed-route performance from GeoFM-specific information, suitability, transfer, and future-planning claims.",
            "phase60,phase62,phase69,phase71,phase72b,phase72_exhaustion",
            "review_before_any_future_manuscript_revision",
            anchors,
        ),
    ]


def _claim(
    claim_id: str,
    claim_class: str,
    claim_status: str,
    anchor: str,
    evidence: str,
    evidence_sources: str,
    recommended_action: str,
    anchors: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    matched = [
        f"{key}:{','.join(str(line) for line in anchors.get(key, []))}"
        for key in anchors
        if anchors.get(key)
    ]
    return {
        "claim_id": claim_id,
        "claim_class": claim_class,
        "claim_status": claim_status,
        "manuscript_anchor": anchor,
        "manuscript_lines": ";".join(matched),
        "current_evidence": evidence,
        "evidence_sources": evidence_sources,
        "recommended_action": recommended_action,
        "claim_boundary": PHASE72_CLAIM_DRIFT_CLAIM_BOUNDARY,
    }


def _find_anchors(text: str) -> dict[str, list[int]]:
    lines = text.splitlines()
    return {
        claim_id: [index for index, line in enumerate(lines, start=1) if anchor in line]
        for claim_id, anchor in _ANCHORS.items()
    }


def _read_text(path: Path, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    return path.read_text(encoding="utf-8")


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


def _required_status(payload: Mapping[str, object], field: str, label: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is missing required status field: {field}")
    return value


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
    claims = analysis.get("claim_rows", [])
    lines = [
        "# Phase 72 Claim-Drift Audit",
        "",
        f"Status: `{analysis.get('phase72_claim_drift_status', '')}`",
        "",
        "This read-only audit compares formal manuscript wording with later "
        "Paper11 evidence. It does not edit the formal manuscript.",
        "",
        "| Claim | Status | Recommended action |",
        "| --- | --- | --- |",
    ]
    for row in claims if isinstance(claims, Sequence) else []:
        if isinstance(row, Mapping):
            lines.append(
                f"| `{row.get('claim_id', '')}` | `{row.get('claim_status', '')}` | "
                f"{row.get('recommended_action', '')} |"
            )
    lines.extend(
        [
            "",
            "## Required Action",
            "",
            str(analysis.get("recommended_action", "")),
            "",
            "## Claim Boundary",
            "",
            str(analysis.get("claim_boundary", "")),
            "",
        ]
    )
    return "\n".join(lines)
