from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path


PHASE10_CLAIM_BOUNDARY = (
    "Phase 10 is a reward-readiness gate for suitability_proxy; it does not "
    "train, tune, evaluate, or report a DRL policy, and it does not prove "
    "agronomic validity."
)
DEFAULT_REQUIRED_LABELS = (
    "stable_farmland_label",
    "high_standard_farmland_label",
)


def build_phase10_reward_readiness_gate(
    phase9_report_path: Path | str,
    required_labels: Sequence[str] = DEFAULT_REQUIRED_LABELS,
    min_rank_auc: float = 0.5,
    min_mean_difference: float = 0.0,
) -> dict[str, object]:
    report_path = Path(phase9_report_path)
    report = _load_phase9_report(report_path)
    labels = report["labels"]
    requested = [str(label) for label in required_labels]
    thresholds = {
        "min_rank_auc": float(min_rank_auc),
        "min_mean_difference": float(min_mean_difference),
        "require_positive_interpretation": True,
    }
    label_results = {
        label: _evaluate_label(
            label,
            labels.get(label),
            min_rank_auc=float(min_rank_auc),
            min_mean_difference=float(min_mean_difference),
        )
        for label in requested
    }
    status, recommendation, reasons = _reduce_gate(label_results)

    return {
        "phase": "phase10_reward_readiness_gate",
        "phase9_report": str(report_path),
        "required_labels": requested,
        "thresholds": thresholds,
        "status": status,
        "recommendation": recommendation,
        "passing_label_count": sum(
            1 for result in label_results.values() if result["passes_gate"]
        ),
        "failing_label_count": sum(
            1 for result in label_results.values() if result["category"] == "failing"
        ),
        "insufficient_label_count": sum(
            1
            for result in label_results.values()
            if result["category"] == "insufficient"
        ),
        "labels": label_results,
        "reasons": reasons,
        "claim_boundary": PHASE10_CLAIM_BOUNDARY,
    }


def write_phase10_reward_readiness_gate(
    gate: Mapping[str, object],
    output_dir: Path | str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    gate_path = output_path / "phase10_reward_readiness_gate.json"
    gate_path.write_text(
        json.dumps(dict(gate), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return gate_path


def _load_phase9_report(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 9 proxy-validation report: {path}")

    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("phase") != "phase9_proxy_validation_report":
        raise ValueError("Phase 10 requires a Phase 9 proxy-validation report")
    labels = report.get("labels")
    if not isinstance(labels, Mapping):
        raise ValueError("Phase 9 report is missing labels")
    return report


def _evaluate_label(
    label: str,
    payload: object,
    min_rank_auc: float,
    min_mean_difference: float,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        return _label_result(
            label,
            payload=None,
            available=False,
            passes_gate=False,
            category="insufficient",
            reason=f"{label} is missing from the Phase 9 report.",
        )

    interpretation = str(payload.get("interpretation", "label_unavailable"))
    validation_available = bool(payload.get("validation_available"))
    rank_auc = _optional_float(payload.get("rank_auc"))
    mean_difference = _optional_float(payload.get("mean_difference"))

    if not validation_available or interpretation in {
        "label_unavailable",
        "insufficient_label_variation",
    }:
        return _label_result(
            label,
            payload=payload,
            available=True,
            passes_gate=False,
            category="insufficient",
            reason=(
                f"{label} has insufficient weak-label evidence for reward use "
                f"({interpretation})."
            ),
        )

    passes_gate = (
        interpretation == "positive_alignment"
        and rank_auc is not None
        and rank_auc >= min_rank_auc
        and mean_difference is not None
        and mean_difference > min_mean_difference
    )
    if passes_gate:
        return _label_result(
            label,
            payload=payload,
            available=True,
            passes_gate=True,
            category="passing",
            reason=f"{label} passed suitability proxy alignment gate.",
        )
    return _label_result(
        label,
        payload=payload,
        available=True,
        passes_gate=False,
        category="failing",
        reason=(
            f"{label} failed suitability proxy alignment gate "
            f"(interpretation={interpretation}, rank_auc={rank_auc}, "
            f"mean_difference={mean_difference})."
        ),
    )


def _label_result(
    label: str,
    payload: Mapping[str, object] | None,
    available: bool,
    passes_gate: bool,
    category: str,
    reason: str,
) -> dict[str, object]:
    payload = payload or {}
    return {
        "available": available,
        "validation_available": bool(payload.get("validation_available", False)),
        "interpretation": str(payload.get("interpretation", "label_unavailable")),
        "rank_auc": payload.get("rank_auc"),
        "mean_difference": payload.get("mean_difference"),
        "positive_count": int(payload.get("positive_count", 0) or 0),
        "negative_count": int(payload.get("negative_count", 0) or 0),
        "passes_gate": passes_gate,
        "category": category,
        "reason": reason,
    }


def _reduce_gate(
    label_results: Mapping[str, Mapping[str, object]],
) -> tuple[str, str, list[str]]:
    reasons = [str(result["reason"]) for result in label_results.values()]
    if label_results and all(result["passes_gate"] for result in label_results.values()):
        return (
            "ready_for_suitability_reward_smoke",
            "allow_bounded_suitability_reward_smoke",
            reasons,
        )
    if all(result["category"] == "insufficient" for result in label_results.values()):
        return (
            "insufficient_evidence",
            "collect_or_rebuild_weak_labels_before_reward_use",
            reasons,
        )
    return (
        "not_ready_for_suitability_reward",
        "do_not_enable_suitability_reward",
        reasons,
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
