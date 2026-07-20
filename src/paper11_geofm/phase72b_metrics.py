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

from .phase72b_protocol import PHASE72B_CLAIM_BOUNDARY, PHASE72B_GATES


_EXPECTED_CONTROL_IDS = {
    "temporal_order_shuffle",
    "spatial_shuffle",
    "random_projection",
}
_EXPECTED_TRANSFER_AXES = {
    "bishan_to_dongxing",
    "dongxing_to_bishan",
}
_EXPECTED_SPATIAL_REGIONS = {"bishan", "dongxing"}


def _round(value: float) -> float:
    return round(float(value), 12)


def _binary_array(values: Sequence[int], *, context: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Phase 72B {context} must be binary") from exc
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"Invalid Phase 72B {context}")
    if not np.isfinite(array).all() or not np.isin(array, (0.0, 1.0)).all():
        raise ValueError(f"Phase 72B {context} must be binary")
    return array.astype(np.int8)


def _probability_array(
    values: Sequence[float], *, context: str
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Phase 72B {context} probabilities are invalid"
        ) from exc
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"Invalid Phase 72B {context} probabilities")
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError(
            f"Phase 72B {context} probabilities must be finite in [0, 1]"
        )
    return array


def _positive_integer(value: object, *, context: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Phase 72B {context} must be a positive integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Phase 72B {context} must be a positive integer"
        ) from exc
    if converted <= 0 or converted != value:
        raise ValueError(f"Phase 72B {context} must be a positive integer")
    return converted


def expected_calibration_error(
    y_true: Sequence[int], probability: Sequence[float], bins: int = 10
) -> float:
    y = _binary_array(y_true, context="calibration outcomes")
    p = _probability_array(probability, context="calibration")
    if y.shape != p.shape:
        raise ValueError("Invalid Phase 72B calibration inputs")
    bin_count = _positive_integer(bins, context="calibration bins")
    order = np.argsort(p, kind="mergesort")
    groups = np.array_split(order, min(bin_count, len(y)))
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
    y = _binary_array(y_true, context="metric outcomes")
    p = _probability_array(probability, context="metric")
    if y.shape != p.shape:
        raise ValueError("Invalid Phase 72B metric inputs")
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or not 0 <= threshold_value <= 1:
        raise ValueError("Phase 72B threshold must be finite in [0, 1]")
    budget_values = [float(value) for value in budgets]
    if not budget_values or any(
        not np.isfinite(value) or not 0 < value <= 1
        for value in budget_values
    ):
        raise ValueError("Phase 72B budgets must be finite in (0, 1]")
    bin_count = _positive_integer(ece_bins, context="ECE bins")
    predicted = (p >= threshold_value).astype(np.int8)
    result = {
        "average_precision": _round(average_precision_score(y, p)),
        "brier": _round(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, bin_count),
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
    for budget_value in budget_values:
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
            if (
                not np.isfinite(decision_threshold)
                or not 0 <= decision_threshold <= 1
            ):
                raise ValueError(
                    "Phase 72B budget threshold must be finite in [0, 1]"
                )
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
    y = _binary_array(y_true, context="bootstrap outcomes")
    explicit = _probability_array(
        explicit_probability, context="bootstrap explicit"
    )
    geofm = _probability_array(
        geofm_probability, context="bootstrap GeoFM"
    )
    if not (y.shape == explicit.shape == geofm.shape):
        raise ValueError("Phase 72B bootstrap arrays must align")
    if len(sample_rows) != len(y):
        raise ValueError("Phase 72B bootstrap rows must align")
    iteration_count = _positive_integer(
        iterations, context="bootstrap iterations"
    )

    clusters: dict[tuple[str, str], list[int]] = {}
    regions: dict[str, list[tuple[str, str]]] = {}
    for index, row in enumerate(sample_rows):
        region_id = str(row.get("region_id", "")).strip()
        block_id = str(row.get("spatial_block_id", "")).strip()
        if not region_id:
            raise ValueError("Phase 72B bootstrap region ID is blank")
        if not block_id:
            raise ValueError("Phase 72B bootstrap spatial block ID is blank")
        key = (region_id, block_id)
        clusters.setdefault(key, []).append(index)
    for key in clusters:
        regions.setdefault(key[0], []).append(key)
    for region_id in regions:
        regions[region_id] = sorted(regions[region_id])

    rng = np.random.default_rng(int(seed))
    ap_deltas = []
    brier_deltas = []
    for _ in range(iteration_count):
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
    minimum_valid = max(1, int(np.ceil(iteration_count * 0.8)))
    if len(ap_deltas) < minimum_valid:
        raise ValueError(
            "Phase 72B bootstrap has insufficient valid replicates: "
            f"{len(ap_deltas)} / {iteration_count}"
        )
    ap = np.asarray(ap_deltas, dtype=np.float64)
    brier = np.asarray(brier_deltas, dtype=np.float64)
    return {
        "iterations_requested": iteration_count,
        "iterations_valid": len(ap),
        "n_clusters": len(clusters),
        "ap_delta_mean": _round(ap.mean()),
        "ap_delta_ci_low": _round(np.quantile(ap, 0.025)),
        "ap_delta_ci_high": _round(np.quantile(ap, 0.975)),
        "brier_delta_mean": _round(brier.mean()),
        "brier_delta_ci_low": _round(np.quantile(brier, 0.025)),
        "brier_delta_ci_high": _round(np.quantile(brier, 0.975)),
    }


def _gate_number(
    record: Mapping[str, object],
    field: str,
    *,
    context: str,
    blockers: list[str],
) -> float | None:
    try:
        value = float(record[field])
    except (KeyError, TypeError, ValueError):
        blockers.append(f"missing or invalid {context} {field}")
        return None
    if not np.isfinite(value):
        blockers.append(f"non-finite {context} {field}")
        return None
    return value


def _gate_deltas(
    row: Mapping[str, object], *, context: str, blockers: list[str]
) -> dict[str, float | None]:
    return {
        field: _gate_number(
            row, field, context=context, blockers=blockers
        )
        for field in ("ap_delta", "brier_delta", "ece_delta")
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
    blockers = []
    if not leakage_ok:
        blockers.append("leakage audit failed")
    try:
        supplied_gates = {
            str(key): float(value) for key, value in dict(gates).items()
        }
    except (TypeError, ValueError):
        supplied_gates = {}
    if supplied_gates != PHASE72B_GATES:
        blockers.append("frozen gate thresholds mismatch")

    gate_fields = tuple(PHASE72B_GATES)
    thresholds = {
        field: _gate_number(
            gates, field, context="gate threshold", blockers=blockers
        )
        for field in gate_fields
    }
    pooled_deltas = _gate_deltas(
        pooled_delta, context="pooled delta", blockers=blockers
    )
    bootstrap_values = {
        field: _gate_number(
            pooled_bootstrap,
            field,
            context="pooled bootstrap",
            blockers=blockers,
        )
        for field in ("ap_delta_ci_low", "brier_delta_ci_low")
    }
    practical_checks = {
        metric: (
            pooled_deltas[f"{metric}_delta"] is not None
            and thresholds[f"{metric}_vs_explicit"] is not None
            and pooled_deltas[f"{metric}_delta"]
            >= thresholds[f"{metric}_vs_explicit"]
        )
        for metric in ("ap", "brier", "ece")
    }
    practical = sum(practical_checks.values()) >= 2
    statistical_checks = {
        metric: (
            bootstrap_values[f"{metric}_delta_ci_low"] is not None
            and bootstrap_values[f"{metric}_delta_ci_low"] > 0
        )
        for metric in ("ap", "brier")
    }
    statistical = any(statistical_checks.values())
    pooled_directions = [
        metric for metric in ("ap", "brier") if practical_checks[metric]
    ]

    control_audits = []
    control_ids = []
    for raw_row in control_rows:
        row = dict(raw_row)
        control_id = str(row.get("control_id", "")).strip()
        control_ids.append(control_id)
        deltas = _gate_deltas(
            row,
            context=f"control {control_id or '<blank>'}",
            blockers=blockers,
        )
        delta_checks = {
            "ap": (
                deltas["ap_delta"] is not None
                and thresholds["ap_vs_control"] is not None
                and deltas["ap_delta"] >= thresholds["ap_vs_control"]
            ),
            "brier": (
                deltas["brier_delta"] is not None
                and thresholds["brier_vs_control"] is not None
                and deltas["brier_delta"]
                >= thresholds["brier_vs_control"]
            ),
        }
        control_audits.append(
            {
                "control_id": control_id,
                "deltas": deltas,
                "delta_checks": delta_checks,
                "passed": all(delta_checks.values()),
            }
        )
    if set(control_ids) != _EXPECTED_CONTROL_IDS or len(control_ids) != len(
        _EXPECTED_CONTROL_IDS
    ):
        blockers.append("control evidence identities are incomplete or duplicate")
    controls = (
        not blockers
        and len(control_audits) == len(_EXPECTED_CONTROL_IDS)
        and all(row["passed"] for row in control_audits)
    )

    transfer_audits = []
    transfer_axes = []
    for raw_row in transfer_rows:
        row = dict(raw_row)
        axis_id = str(row.get("axis_id", "")).strip()
        transfer_axes.append(axis_id)
        deltas = _gate_deltas(
            row,
            context=f"transfer {axis_id or '<blank>'}",
            blockers=blockers,
        )
        gain_checks = {
            "ap": (
                deltas["ap_delta"] is not None
                and thresholds["transfer_ap_gain"] is not None
                and deltas["ap_delta"] >= thresholds["transfer_ap_gain"]
            ),
            "brier": (
                deltas["brier_delta"] is not None
                and thresholds["transfer_brier_gain"] is not None
                and deltas["brier_delta"]
                >= thresholds["transfer_brier_gain"]
            ),
        }
        no_harm_checks = {
            "ap": (
                deltas["ap_delta"] is not None
                and thresholds["transfer_ap_harm"] is not None
                and deltas["ap_delta"] >= -thresholds["transfer_ap_harm"]
            ),
            "brier": (
                deltas["brier_delta"] is not None
                and thresholds["transfer_brier_harm"] is not None
                and deltas["brier_delta"]
                >= -thresholds["transfer_brier_harm"]
            ),
        }
        transfer_audits.append(
            {
                "axis_id": axis_id,
                "deltas": deltas,
                "gain_checks": gain_checks,
                "no_harm_checks": no_harm_checks,
                "passed": any(gain_checks.values())
                and all(no_harm_checks.values()),
            }
        )
    if set(transfer_axes) != _EXPECTED_TRANSFER_AXES or len(
        transfer_axes
    ) != len(_EXPECTED_TRANSFER_AXES):
        blockers.append("transfer evidence identities are incomplete or duplicate")
    transfer = (
        set(transfer_axes) == _EXPECTED_TRANSFER_AXES
        and len(transfer_axes) == len(_EXPECTED_TRANSFER_AXES)
        and all(row["passed"] for row in transfer_audits)
    )

    spatial_audits = []
    spatial_axes = []
    folds_by_region = {region_id: set() for region_id in _EXPECTED_SPATIAL_REGIONS}
    for raw_row in spatial_rows:
        row = dict(raw_row)
        axis_id = str(row.get("axis_id", "")).strip()
        region_id = str(row.get("region_id", "")).strip()
        spatial_axes.append(axis_id)
        prefix = f"spatial_{region_id}_fold"
        fold_id = None
        if region_id not in _EXPECTED_SPATIAL_REGIONS:
            blockers.append(f"unexpected spatial region: {region_id or '<blank>'}")
        elif not axis_id.startswith(prefix):
            blockers.append(f"spatial axis identity mismatch: {axis_id}")
        else:
            try:
                fold_id = int(axis_id.removeprefix(prefix))
            except ValueError:
                blockers.append(f"invalid spatial fold identity: {axis_id}")
            else:
                if fold_id not in range(5):
                    blockers.append(f"invalid spatial fold identity: {axis_id}")
                elif axis_id != f"spatial_{region_id}_fold{fold_id}":
                    blockers.append(
                        f"non-canonical spatial fold identity: {axis_id}"
                    )
                else:
                    folds_by_region[region_id].add(fold_id)
        rows = _gate_number(
            row,
            "rows",
            context=f"spatial {axis_id or '<blank>'}",
            blockers=blockers,
        )
        if rows is not None and rows <= 0:
            blockers.append(f"non-positive spatial row count: {axis_id}")
        elif rows is not None and not rows.is_integer():
            blockers.append(f"non-integer spatial row count: {axis_id}")
        deltas = _gate_deltas(
            row,
            context=f"spatial {axis_id or '<blank>'}",
            blockers=blockers,
        )
        direction_checks = {
            metric: deltas[f"{metric}_delta"] is not None
            and deltas[f"{metric}_delta"] >= 0
            for metric in pooled_directions
        }
        spatial_audits.append(
            {
                "axis_id": axis_id,
                "region_id": region_id,
                "fold_id": fold_id,
                "rows": rows,
                "deltas": deltas,
                "direction_checks": direction_checks,
                "passed": bool(direction_checks)
                and any(direction_checks.values()),
            }
        )
    if len(spatial_axes) != len(set(spatial_axes)):
        blockers.append("spatial evidence identities are duplicate")
    for region_id, folds in folds_by_region.items():
        if len(folds) < 2:
            blockers.append(
                f"spatial evidence requires at least two folds for {region_id}"
            )

    spatial_region_audits = []
    for region_id in sorted(_EXPECTED_SPATIAL_REGIONS):
        region_rows = [
            row for row in spatial_audits if row["region_id"] == region_id
        ]
        weight_sum = sum(
            float(row["rows"])
            for row in region_rows
            if row["rows"] is not None and row["rows"] > 0
        )
        aggregate_deltas = {}
        for metric in ("ap", "brier", "ece"):
            values = [
                (float(row["rows"]), row["deltas"].get(f"{metric}_delta"))
                for row in region_rows
                if row["rows"] is not None
                and row["rows"] > 0
                and row["deltas"].get(f"{metric}_delta") is not None
            ]
            aggregate_deltas[f"{metric}_delta"] = (
                _round(sum(weight * float(value) for weight, value in values) / weight_sum)
                if values and sum(weight for weight, _ in values) == weight_sum
                else None
            )
        direction_checks = {
            metric: aggregate_deltas[f"{metric}_delta"] is not None
            and aggregate_deltas[f"{metric}_delta"] >= 0
            for metric in pooled_directions
        }
        spatial_region_audits.append(
            {
                "region_id": region_id,
                "fold_count": len(folds_by_region[region_id]),
                "rows": int(weight_sum),
                "deltas": aggregate_deltas,
                "direction_checks": direction_checks,
                "passed": bool(direction_checks)
                and any(direction_checks.values()),
            }
        )
    spatial_coverage = all(
        len(folds) >= 2 for folds in folds_by_region.values()
    ) and len(spatial_axes) == len(set(spatial_axes))
    spatial = (
        spatial_coverage
        and bool(spatial_audits)
        and all(row["passed"] for row in spatial_audits)
        and all(row["passed"] for row in spatial_region_audits)
    )

    input_ready = not blockers
    controls = input_ready and all(
        row["passed"] for row in control_audits
    )
    supported = transfer and spatial
    checks = {
        "input_ready": input_ready,
        "leakage": bool(leakage_ok),
        "practical": practical,
        "statistical": statistical,
        "controls": controls,
        "transfer": transfer,
        "spatial": spatial,
        "practical_metrics": practical_checks,
        "statistical_metrics": statistical_checks,
    }
    evidence = {
        "input_blockers": list(dict.fromkeys(blockers)),
        "pooled": {
            "deltas": pooled_deltas,
            "bootstrap": bootstrap_values,
            "practical_metrics": practical_checks,
            "statistical_metrics": statistical_checks,
            "spatial_directions": pooled_directions,
        },
        "controls": control_audits,
        "transfers": transfer_audits,
        "spatial_folds": spatial_audits,
        "spatial_regions": spatial_region_audits,
    }
    if not input_ready:
        status = "phase72b_inputs_not_ready"
        reasons = evidence["input_blockers"]
    elif not (practical and statistical and controls):
        status = "geofm_information_not_supported"
        reasons = ["pooled practical/statistical/control gate failed"]
    elif not supported:
        status = "geofm_information_mixed"
        reasons = ["spatial or transfer heterogeneity"]
    else:
        status = "geofm_information_supported"
        reasons = []
    return {
        "phase72b_status": status,
        "reasons": reasons,
        "checks": checks,
        "evidence": evidence,
        "claim_boundary": PHASE72B_CLAIM_BOUNDARY,
    }
