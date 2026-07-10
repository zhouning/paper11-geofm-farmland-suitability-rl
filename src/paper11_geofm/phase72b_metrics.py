from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from .phase72b_protocol import PHASE72B_CLAIM_BOUNDARY


def _round(value: float) -> float:
    return round(float(value), 12)


def expected_calibration_error(
    y_true: Sequence[int], probability: Sequence[float], bins: int = 10
) -> float:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("Invalid Phase 72B calibration inputs")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Phase 72B probabilities must be finite in [0, 1]")
    order = np.argsort(p, kind="mergesort")
    groups = np.array_split(order, min(int(bins), len(y)))
    value = sum(
        len(group)
        / len(y)
        * abs(float(y[group].mean()) - float(p[group].mean()))
        for group in groups
        if len(group)
    )
    return _round(value)


def phase72b_metrics(
    y_true: Sequence[int],
    probability: Sequence[float],
    *,
    threshold: float,
    budgets: Sequence[float],
    ece_bins: int,
    budget_thresholds: Mapping[str, float] | None = None,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("Invalid Phase 72B metric inputs")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("Phase 72B outcomes must be binary")
    predicted = (p >= float(threshold)).astype(np.int8)
    result = {
        "average_precision": _round(average_precision_score(y, p)),
        "brier": _round(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, ece_bins),
        "roc_auc": (
            _round(roc_auc_score(y, p))
            if len(np.unique(y)) == 2
            else float("nan")
        ),
        "f1": _round(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": _round(
            balanced_accuracy_score(y, predicted)
        ),
    }
    prevalence = float(y.mean())
    for budget in budgets:
        budget_value = float(budget)
        budget_label = f"{int(round(100 * budget_value))}pct"
        k = max(1, int(np.ceil(len(y) * budget_value)))
        selected = np.argsort(-p, kind="mergesort")[:k]
        result[f"capture_at_{budget_label}"] = _round(
            y[selected].sum() / max(1, y.sum())
        )
        result[f"precision_at_{budget_label}"] = _round(y[selected].mean())
        result[f"lift_at_{budget_label}"] = _round(
            float(y[selected].mean()) / prevalence if prevalence > 0 else 0.0
        )
        if budget_thresholds is not None and budget_label in budget_thresholds:
            decision_threshold = float(budget_thresholds[budget_label])
            selected_mask = p >= decision_threshold
            true_positive = float(np.sum(selected_mask & (y == 1)))
            false_positive = float(np.sum(selected_mask & (y == 0)))
            odds = decision_threshold / max(1e-12, 1 - decision_threshold)
            result[f"net_benefit_at_{budget_label}"] = _round(
                true_positive / len(y) - false_positive / len(y) * odds
            )
    return result


def paired_block_bootstrap(
    y_true: Sequence[int],
    explicit_probability: Sequence[float],
    geofm_probability: Sequence[float],
    sample_rows: Sequence[Mapping[str, object]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.int8)
    explicit = np.asarray(explicit_probability, dtype=np.float64)
    geofm = np.asarray(geofm_probability, dtype=np.float64)
    if not (y.shape == explicit.shape == geofm.shape):
        raise ValueError("Phase 72B bootstrap arrays must align")
    if len(sample_rows) != len(y):
        raise ValueError("Phase 72B bootstrap rows must align")

    clusters: dict[tuple[str, str], list[int]] = {}
    regions: dict[str, list[tuple[str, str]]] = {}
    for index, row in enumerate(sample_rows):
        key = (str(row["region_id"]), str(row["spatial_block_id"]))
        clusters.setdefault(key, []).append(index)
    for key in clusters:
        regions.setdefault(key[0], []).append(key)
    for region_id in regions:
        regions[region_id] = sorted(regions[region_id])

    rng = np.random.default_rng(int(seed))
    ap_deltas = []
    brier_deltas = []
    for _ in range(int(iterations)):
        selected_indexes = []
        for region_id in sorted(regions):
            region_clusters = regions[region_id]
            sampled_positions = rng.integers(
                0, len(region_clusters), size=len(region_clusters)
            )
            for position in sampled_positions:
                selected_indexes.extend(
                    clusters[region_clusters[int(position)]]
                )
        indexes = np.asarray(selected_indexes, dtype=np.int64)
        sampled_y = y[indexes]
        if len(np.unique(sampled_y)) < 2:
            continue
        sampled_explicit = explicit[indexes]
        sampled_geofm = geofm[indexes]
        ap_deltas.append(
            average_precision_score(sampled_y, sampled_geofm)
            - average_precision_score(sampled_y, sampled_explicit)
        )
        brier_deltas.append(
            brier_score_loss(sampled_y, sampled_explicit)
            - brier_score_loss(sampled_y, sampled_geofm)
        )
    minimum_valid = max(1, int(np.ceil(int(iterations) * 0.8)))
    if len(ap_deltas) < minimum_valid:
        raise ValueError(
            "Phase 72B bootstrap has insufficient valid replicates: "
            f"{len(ap_deltas)} / {iterations}"
        )
    ap = np.asarray(ap_deltas, dtype=np.float64)
    brier = np.asarray(brier_deltas, dtype=np.float64)
    return {
        "iterations_requested": int(iterations),
        "iterations_valid": len(ap),
        "n_clusters": len(clusters),
        "ap_delta_mean": _round(ap.mean()),
        "ap_delta_ci_low": _round(np.quantile(ap, 0.025)),
        "ap_delta_ci_high": _round(np.quantile(ap, 0.975)),
        "brier_delta_mean": _round(brier.mean()),
        "brier_delta_ci_low": _round(np.quantile(brier, 0.025)),
        "brier_delta_ci_high": _round(np.quantile(brier, 0.975)),
    }


def build_phase72b_gate(
    *,
    pooled_delta: Mapping[str, float],
    pooled_bootstrap: Mapping[str, float],
    control_rows: Sequence[Mapping[str, object]],
    transfer_rows: Sequence[Mapping[str, object]],
    spatial_rows: Sequence[Mapping[str, object]],
    leakage_ok: bool,
    gates: Mapping[str, float],
) -> dict[str, object]:
    if not leakage_ok:
        return {
            "phase72b_status": "phase72b_inputs_not_ready",
            "reasons": ["leakage audit failed"],
            "claim_boundary": PHASE72B_CLAIM_BOUNDARY,
        }
    practical_checks = {
        "ap": float(pooled_delta["ap_delta"])
        >= float(gates["ap_vs_explicit"]),
        "brier": float(pooled_delta["brier_delta"])
        >= float(gates["brier_vs_explicit"]),
        "ece": float(pooled_delta["ece_delta"])
        >= float(gates["ece_vs_explicit"]),
    }
    practical = sum(practical_checks.values()) >= 2
    statistical = (
        float(pooled_bootstrap["ap_delta_ci_low"]) > 0
        or float(pooled_bootstrap["brier_delta_ci_low"]) > 0
    )
    controls = bool(control_rows) and all(
        float(row["ap_delta"]) >= float(gates["ap_vs_control"])
        and float(row["brier_delta"])
        >= float(gates["brier_vs_control"])
        for row in control_rows
    )
    checks = {
        "practical": practical,
        "statistical": statistical,
        "controls": controls,
    }
    if not all(checks.values()):
        return {
            "phase72b_status": "geofm_information_not_supported",
            "reasons": ["pooled practical/statistical/control gate failed"],
            "checks": {**checks, "practical_metrics": practical_checks},
            "claim_boundary": PHASE72B_CLAIM_BOUNDARY,
        }
    transfer = len(transfer_rows) == 2 and all(
        (
            float(row["ap_delta"]) >= float(gates["transfer_ap_gain"])
            or float(row["brier_delta"])
            >= float(gates["transfer_brier_gain"])
        )
        and float(row["ap_delta"]) >= -float(gates["transfer_ap_harm"])
        and float(row["brier_delta"])
        >= -float(gates["transfer_brier_harm"])
        for row in transfer_rows
    )
    spatial = bool(spatial_rows) and all(
        float(row["ap_delta"]) >= 0 or float(row["brier_delta"]) >= 0
        for row in spatial_rows
    )
    supported = transfer and spatial
    return {
        "phase72b_status": (
            "geofm_information_supported"
            if supported
            else "geofm_information_mixed"
        ),
        "reasons": [] if supported else ["spatial or transfer heterogeneity"],
        "checks": {
            **checks,
            "practical_metrics": practical_checks,
            "transfer": transfer,
            "spatial": spatial,
        },
        "claim_boundary": PHASE72B_CLAIM_BOUNDARY,
    }
