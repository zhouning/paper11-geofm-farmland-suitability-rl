from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .planning_reward import (
    BASE_PLANNING_REWARD_EVIDENCE,
    BASE_PLANNING_REWARD_IMPLEMENTED,
)


PHASE18_CLAIM_BOUNDARY = (
    "Phase 18 is a planning-reward readiness audit; it does not implement a "
    "planning reward, train, tune, evaluate, or compare a DRL policy, enable "
    "suitability reward, or report planning performance."
)


def build_phase18_planning_reward_readiness(
    phase2_output_dir: Path | str,
    phase10_gate_path: Path | str,
    phase12_audit_path: Path | str,
    phase17_readiness_path: Path | str | None = None,
) -> dict[str, object]:
    phase2_dir = Path(phase2_output_dir)
    phase10_path = Path(phase10_gate_path)
    phase12_path = Path(phase12_audit_path)

    variant_manifest = _read_json_object(phase2_dir / "experiment_variants.json")
    phase10 = _read_json_object(phase10_path)
    phase12 = _read_json_object(phase12_path)
    phase17 = _read_optional_json_object(phase17_readiness_path)

    base_variants = _base_variant_summary(variant_manifest)
    base_reward_implemented, base_reward_evidence = _base_reward_evidence()
    phase10_allows_suitability_reward = _phase10_allows_suitability_reward(phase10)
    suitability_reward_allowed = bool(phase12.get("suitability_reward_allowed", False))
    flat_full_scale_training_ready = bool(
        phase12.get("flat_full_scale_training_ready", False)
    )
    tiled_maskableppo_api_ready = _tiled_api_ready(phase17)
    blocked_reasons = _blocked_reasons(
        base_variants,
        base_reward_implemented=base_reward_implemented,
        real_feature_tables_ready=bool(phase12.get("real_feature_tables_ready", False)),
        suitability_reward_allowed=suitability_reward_allowed,
        flat_full_scale_training_ready=flat_full_scale_training_ready,
        tiled_maskableppo_api_ready=tiled_maskableppo_api_ready,
    )
    performance_experiment_ready = not blocked_reasons

    return {
        "phase": "phase18_planning_reward_readiness",
        "phase2_output_dir": str(phase2_dir),
        "phase10_gate": str(phase10_path),
        "phase12_audit": str(phase12_path),
        "phase17_readiness": (
            str(Path(phase17_readiness_path))
            if phase17_readiness_path is not None
            else None
        ),
        "base_variants_ready": {
            variant_id: bool(summary["ready"])
            for variant_id, summary in base_variants.items()
        },
        "base_reward_modes": {
            variant_id: str(summary["reward_mode"])
            for variant_id, summary in base_variants.items()
        },
        "base_planning_reward_implemented": base_reward_implemented,
        "base_planning_reward_evidence": base_reward_evidence,
        "phase10_status": str(phase10.get("status", "")),
        "phase10_recommendation": str(phase10.get("recommendation", "")),
        "phase10_allows_suitability_reward": phase10_allows_suitability_reward,
        "real_feature_tables_ready": bool(
            phase12.get("real_feature_tables_ready", False)
        ),
        "representation_only_smoke_allowed": bool(
            phase12.get("representation_only_smoke_allowed", False)
        ),
        "suitability_reward_allowed": suitability_reward_allowed,
        "flat_full_scale_training_ready": flat_full_scale_training_ready,
        "requires_tiled_or_hierarchical_env": bool(
            phase12.get("requires_tiled_or_hierarchical_env", False)
        ),
        "n_blocks": _optional_int(phase12.get("n_blocks")),
        "max_observation_dimension": _optional_int(
            phase12.get("max_observation_dimension")
        ),
        "tiled_maskableppo_api_ready": tiled_maskableppo_api_ready,
        "tiled_maskableppo_status": _phase17_status(phase17),
        "performance_experiment_ready": performance_experiment_ready,
        "blocked_reasons": blocked_reasons,
        "recommended_next_step": _recommended_next_step(
            blocked_reasons,
            base_reward_implemented=base_reward_implemented,
        ),
        "claim_boundary": PHASE18_CLAIM_BOUNDARY,
    }


def write_phase18_planning_reward_readiness(
    report: Mapping[str, object],
    output_dir: Path | str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / "phase18_planning_reward_readiness.json"
    artifact_path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact_path


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 18 input artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Phase 18 input artifact must be a JSON object: {path}")
    return payload


def _read_optional_json_object(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _read_json_object(Path(path))


def _base_variant_summary(
    variant_manifest: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    variants = variant_manifest.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError("Phase 2 experiment_variants.json is missing variants")

    summaries: dict[str, dict[str, object]] = {}
    for variant_id in ("B0", "B1"):
        variant = variants.get(variant_id)
        if not isinstance(variant, Mapping):
            summaries[variant_id] = {
                "ready": False,
                "reward_mode": "",
                "missing": ["variant_metadata"],
            }
            continue
        summaries[variant_id] = {
            "ready": bool(variant.get("ready")),
            "reward_mode": str(variant.get("reward", "")),
            "missing": list(variant.get("missing", [])),
        }
    return summaries


def _base_reward_evidence() -> tuple[bool, str]:
    return BASE_PLANNING_REWARD_IMPLEMENTED, BASE_PLANNING_REWARD_EVIDENCE


def _phase10_allows_suitability_reward(phase10: Mapping[str, object]) -> bool:
    return (
        str(phase10.get("status", ""))
        in {"ready_for_suitability_reward", "ready_for_suitability_reward_smoke"}
        and str(phase10.get("recommendation", ""))
        != "do_not_enable_suitability_reward"
    )


def _tiled_api_ready(phase17: Mapping[str, object] | None) -> bool:
    if phase17 is None:
        return False
    return (
        str(phase17.get("readiness_status", ""))
        == "passed_tiled_maskableppo_smoke"
        and bool(phase17.get("masking_supported", False))
        and bool(phase17.get("predicted_action_valid", False))
    )


def _phase17_status(phase17: Mapping[str, object] | None) -> str:
    if phase17 is None:
        return "not_supplied"
    return str(phase17.get("readiness_status", "unknown"))


def _blocked_reasons(
    base_variants: Mapping[str, Mapping[str, object]],
    base_reward_implemented: bool,
    real_feature_tables_ready: bool,
    suitability_reward_allowed: bool,
    flat_full_scale_training_ready: bool,
    tiled_maskableppo_api_ready: bool,
) -> list[str]:
    reasons: list[str] = []
    if not all(bool(summary.get("ready")) for summary in base_variants.values()):
        reasons.append("base_variants_not_ready")
    if any(
        str(summary.get("reward_mode")) != "base_planning_reward"
        for summary in base_variants.values()
    ):
        reasons.append("base_variants_do_not_use_base_planning_reward")
    if not base_reward_implemented:
        reasons.append("base_planning_reward_not_implemented")
    if not real_feature_tables_ready:
        reasons.append("real_feature_tables_not_ready")
    if not suitability_reward_allowed:
        reasons.append("suitability_reward_not_allowed")
    if not flat_full_scale_training_ready:
        reasons.append("flat_full_scale_training_not_ready")
    if not flat_full_scale_training_ready and not tiled_maskableppo_api_ready:
        reasons.append("tiled_maskableppo_api_not_ready")
    return reasons


def _recommended_next_step(
    blocked_reasons: list[str],
    base_reward_implemented: bool,
) -> str:
    if not base_reward_implemented:
        return "implement_real_tiled_planning_reward_before_policy_evaluation"
    if "suitability_reward_not_allowed" in blocked_reasons:
        return "resolve_suitability_reward_gate_before_suitability_reward_experiments"
    if "tiled_maskableppo_api_not_ready" in blocked_reasons:
        return "repair_tiled_maskableppo_readiness_before_policy_evaluation"
    if blocked_reasons:
        return "repair_readiness_blockers_before_policy_evaluation"
    return "ready_for_bounded_planning_performance_experiment"


def _optional_int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)
