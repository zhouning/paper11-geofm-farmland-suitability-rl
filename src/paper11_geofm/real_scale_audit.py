from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PHASE12_CLAIM_BOUNDARY = (
    "Phase 12 audits real DLTB-derived artifact readiness and scale; it does "
    "not train, tune, evaluate, or compare a DRL policy and does not override "
    "the Phase 10 reward-readiness gate."
)
DEFAULT_FLAT_OBSERVATION_THRESHOLD = 1_000_000
REQUIRED_VARIANTS = ("B0", "B1", "B2", "B3")
FLOAT32_BYTES = 4
MIB = 1024 * 1024


def build_phase12_real_scale_audit(
    phase11_summary_path: Path | str,
    phase2_output_dir: Path | str,
    phase9_report_path: Path | str,
    phase10_gate_path: Path | str,
    flat_observation_threshold: int = DEFAULT_FLAT_OBSERVATION_THRESHOLD,
) -> dict[str, object]:
    threshold = int(flat_observation_threshold)
    if threshold <= 0:
        raise ValueError("flat_observation_threshold must be positive")

    phase11_path = Path(phase11_summary_path)
    phase2_dir = Path(phase2_output_dir)
    phase9_path = Path(phase9_report_path)
    phase10_path = Path(phase10_gate_path)

    phase11 = _read_json_object(phase11_path)
    phase2_summary = _read_json_object(phase2_dir / "summary.json")
    variant_manifest = _read_json_object(phase2_dir / "experiment_variants.json")
    phase9 = _read_json_object(phase9_path)
    phase10 = _read_json_object(phase10_path)

    n_blocks = _int_value(phase2_summary.get("n_blocks"))
    rows_exported = _int_value(phase11.get("rows_exported"))
    variants = _build_variant_audits(variant_manifest, threshold)
    max_observation_dimension = max(
        (int(variant["observation_dimension"]) for variant in variants.values()),
        default=0,
    )
    max_estimated_observation_mib = max(
        (
            float(variant["estimated_observation_mib"])
            for variant in variants.values()
        ),
        default=0.0,
    )

    required_variants_ready = all(
        bool(variants.get(variant_id, {}).get("ready"))
        and int(variants.get(variant_id, {}).get("row_count", -1)) == n_blocks
        for variant_id in REQUIRED_VARIANTS
    )
    row_count_match = n_blocks == rows_exported
    real_feature_tables_ready = bool(
        rows_exported > 0 and n_blocks > 0 and row_count_match and required_variants_ready
    )
    representation_only_smoke_allowed = bool(
        real_feature_tables_ready
        and bool(variants.get("B0", {}).get("ready"))
        and bool(variants.get("B1", {}).get("ready"))
    )
    suitability_reward_allowed = _suitability_reward_allowed(phase10)
    exceeds_flat_threshold = max_observation_dimension > threshold
    flat_full_scale_training_ready = bool(
        real_feature_tables_ready
        and suitability_reward_allowed
        and not exceeds_flat_threshold
    )
    requires_tiled_or_hierarchical_env = bool(
        real_feature_tables_ready and exceeds_flat_threshold
    )

    return {
        "phase": "phase12_real_dltb_scale_audit",
        "phase11_summary": str(phase11_path),
        "phase2_output_dir": str(phase2_dir),
        "phase9_report": str(phase9_path),
        "phase10_gate": str(phase10_path),
        "flat_observation_threshold": threshold,
        "n_blocks": n_blocks,
        "phase11": {
            "rows_read_in_bbox": _int_value(phase11.get("rows_read_in_bbox")),
            "rows_exported": rows_exported,
            "category_counts": _mapping_to_plain_dict(
                phase11.get("category_counts", {})
            ),
            "label_positive_counts": _mapping_to_plain_dict(
                phase11.get("label_positive_counts", {})
            ),
        },
        "phase2": {
            "n_blocks": n_blocks,
            "feature_groups_present": list(
                phase2_summary.get("feature_groups_present", [])
            ),
            "feature_readiness": _mapping_to_plain_dict(
                phase2_summary.get("feature_readiness", {})
            ),
        },
        "phase9": {
            "n_blocks": _int_value(phase9.get("n_blocks")),
            "labels": _label_summaries(phase9.get("labels", {})),
        },
        "phase10": _phase10_summary(phase10),
        "variants": variants,
        "required_variants": list(REQUIRED_VARIANTS),
        "all_required_variants_ready": required_variants_ready,
        "phase11_phase2_row_count_match": row_count_match,
        "max_observation_dimension": max_observation_dimension,
        "max_estimated_observation_mib": round(max_estimated_observation_mib, 6),
        "real_feature_tables_ready": real_feature_tables_ready,
        "representation_only_smoke_allowed": representation_only_smoke_allowed,
        "suitability_reward_allowed": suitability_reward_allowed,
        "flat_full_scale_training_ready": flat_full_scale_training_ready,
        "requires_tiled_or_hierarchical_env": requires_tiled_or_hierarchical_env,
        "recommendation": _recommendation(
            real_feature_tables_ready,
            suitability_reward_allowed,
            requires_tiled_or_hierarchical_env,
        ),
        "claim_boundary": PHASE12_CLAIM_BOUNDARY,
    }


def write_phase12_real_scale_audit(
    report: Mapping[str, object],
    output_dir: Path | str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "phase12_real_dltb_scale_audit.json"
    report_path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path


def _build_variant_audits(
    variant_manifest: Mapping[str, object],
    threshold: int,
) -> dict[str, dict[str, object]]:
    variants = variant_manifest.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError("Phase 2 experiment_variants.json is missing variants")

    audits: dict[str, dict[str, object]] = {}
    for variant_id in REQUIRED_VARIANTS:
        variant = variants.get(variant_id)
        if not isinstance(variant, Mapping):
            audits[variant_id] = {
                "ready": False,
                "missing": ["variant_metadata"],
                "row_count": 0,
                "n_features": 0,
                "reward_mode": "",
                "feature_table": None,
                "state_groups": [],
                "observation_dimension": 0,
                "estimated_observation_mib": 0.0,
                "within_flat_observation_threshold": True,
            }
            continue
        row_count = _int_value(variant.get("row_count"))
        required_columns = variant.get("required_columns", [])
        if not isinstance(required_columns, list):
            raise ValueError(f"Variant {variant_id} required_columns must be a list")
        n_features = len(required_columns)
        observation_dimension = row_count * n_features + 3
        audits[variant_id] = {
            "ready": bool(variant.get("ready")),
            "missing": list(variant.get("missing", [])),
            "row_count": row_count,
            "n_features": n_features,
            "reward_mode": str(variant.get("reward", "")),
            "feature_table": variant.get("feature_table"),
            "state_groups": list(variant.get("state_groups", [])),
            "observation_dimension": observation_dimension,
            "estimated_observation_mib": round(
                observation_dimension * FLOAT32_BYTES / MIB,
                6,
            ),
            "within_flat_observation_threshold": observation_dimension <= threshold,
        }
    return audits


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 12 input artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Phase 12 input artifact must be a JSON object: {path}")
    return payload


def _label_summaries(labels: object) -> dict[str, dict[str, object]]:
    if not isinstance(labels, Mapping):
        return {}
    summaries: dict[str, dict[str, object]] = {}
    for label, payload in labels.items():
        if not isinstance(payload, Mapping):
            continue
        summaries[str(label)] = {
            "interpretation": str(payload.get("interpretation", "")),
            "rank_auc": _optional_float(payload.get("rank_auc")),
            "mean_difference": _optional_float(payload.get("mean_difference")),
            "positive_count": _optional_int(payload.get("positive_count")),
            "negative_count": _optional_int(payload.get("negative_count")),
        }
    return summaries


def _phase10_summary(phase10: Mapping[str, object]) -> dict[str, object]:
    return {
        "status": str(phase10.get("status", "")),
        "recommendation": str(phase10.get("recommendation", "")),
        "passing_label_count": _int_value(phase10.get("passing_label_count")),
        "failing_label_count": _int_value(phase10.get("failing_label_count")),
        "insufficient_label_count": _int_value(
            phase10.get("insufficient_label_count")
        ),
        "labels": _mapping_to_plain_dict(phase10.get("labels", {})),
    }


def _suitability_reward_allowed(phase10: Mapping[str, object]) -> bool:
    return (
        str(phase10.get("status", "")) == "ready_for_suitability_reward"
        and str(phase10.get("recommendation", ""))
        != "do_not_enable_suitability_reward"
    )


def _recommendation(
    real_feature_tables_ready: bool,
    suitability_reward_allowed: bool,
    requires_tiled_or_hierarchical_env: bool,
) -> str:
    if not real_feature_tables_ready:
        return "repair_real_feature_tables_before_downstream_experiments"
    if not suitability_reward_allowed and requires_tiled_or_hierarchical_env:
        return (
            "continue_real_dltb_representation_only_analysis; "
            "keep_suitability_reward_disabled; "
            "design_tiled_or_hierarchical_env_before_full_scale_training"
        )
    if not suitability_reward_allowed:
        return (
            "continue_real_dltb_representation_only_analysis; "
            "keep_suitability_reward_disabled"
        )
    if requires_tiled_or_hierarchical_env:
        return "design_tiled_or_hierarchical_env_before_full_scale_training"
    return "flat_full_scale_training_gate_passed_for_smoke_only"


def _mapping_to_plain_dict(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _plain_json_value(item) for key, item in value.items()}


def _plain_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _mapping_to_plain_dict(value)
    if isinstance(value, list):
        return [_plain_json_value(item) for item in value]
    return value


def _int_value(value: object) -> int:
    if value is None or str(value).strip() == "":
        return 0
    return int(value)


def _optional_int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)
