from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

from .planning_reward import compute_base_planning_reward_from_matrix_row
from .tiled_inputs import load_tiled_variant_input


PHASE64_CLAIM_BOUNDARY = (
    "Phase 64 is a read-only set-policy error-diagnosis and standardization-gate "
    "phase. It uses Phase 63 base-reward artifacts to diagnose behavior-cloned "
    "set-policy errors and decide whether a train-tile-fitted standardization "
    "rerun is justified. It does not enable suitability reward, does not test "
    "B2/B3, does not test transfer, does not prove GeoFM advantage or PCA "
    "optimality, and does not justify formal submission-level claims."
)

PHASE64_STATUS_STANDARDIZATION = "standardization_route_supported"
PHASE64_STATUS_CAPACITY = "bc_training_capacity_limited"
PHASE64_STATUS_NOT_HELPFUL = "geofm_features_not_helpful_under_set_policy"
PHASE64_STATUS_INCONCLUSIVE = "diagnostic_inconclusive"

PHASE64_WEAK_TOP1_THRESHOLD = 0.25
PHASE64_WEAK_TOPK_THRESHOLD = 0.50
PHASE64_STD_RATIO_THRESHOLD = 100.0
PHASE64_MEAN_SCALE_RATIO_THRESHOLD = 10.0
PHASE64_Z_SHIFT_THRESHOLD = 3.0
PHASE64_EFFECTIVE_RANK_FRACTION_THRESHOLD = 0.30
PHASE64_PC1_SHARE_THRESHOLD = 0.80

PHASE64_CONVERGENCE_FIELDNAMES = [
    "variant_id",
    "train_tile_id",
    "seed",
    "first_epoch",
    "final_epoch",
    "best_epoch",
    "first_loss",
    "final_loss",
    "best_loss",
    "final_top1_accuracy",
    "best_top1_accuracy",
    "final_topk_hit_rate",
    "best_topk_hit_rate",
    "loss_delta",
    "claim_boundary",
]


def _split_semicolon_values(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _round_float(value: object, digits: int = 10) -> float:
    rounded = round(float(value), digits)
    compact = round(rounded, 6)
    if abs(rounded - compact) < 5.0e-8:
        return compact
    return rounded


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def _safe_int(value: object, default: int = 0) -> int:
    if value is None or str(value).strip() == "":
        return int(default)
    return int(float(value))


def _load_csv_rows(path: Path | str, label: str) -> list[dict[str, object]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {label}: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_json_object(path: Path | str, label: str) -> dict[str, object]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {label}: {json_path}")
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return loaded


def _group_key(row: Mapping[str, object], fields: Sequence[str]) -> tuple[object, ...]:
    return tuple(row.get(field, "") for field in fields)


def build_phase64_convergence_summary(
    history_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(history_rows_or_csv, (str, Path)):
        history_rows = _load_csv_rows(history_rows_or_csv, "Phase 63 BC history CSV")
    else:
        history_rows = [dict(row) for row in history_rows_or_csv]

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in history_rows:
        key = _group_key(row, ("variant_id", "train_tile_id", "seed"))
        grouped.setdefault(key, []).append(dict(row))

    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: _safe_int(row.get("epoch")))
        first = rows[0]
        final = rows[-1]
        best = min(rows, key=lambda row: _safe_float(row.get("loss"), math.inf))
        first_loss = _safe_float(first.get("loss"))
        final_loss = _safe_float(final.get("loss"))
        output.append(
            {
                "variant_id": str(key[0]),
                "train_tile_id": str(key[1]),
                "seed": _safe_int(key[2]),
                "first_epoch": _safe_int(first.get("epoch")),
                "final_epoch": _safe_int(final.get("epoch")),
                "best_epoch": _safe_int(best.get("epoch")),
                "first_loss": _round_float(first_loss),
                "final_loss": _round_float(final_loss),
                "best_loss": _round_float(best.get("loss")),
                "final_top1_accuracy": _round_float(final.get("top1_accuracy")),
                "best_top1_accuracy": _round_float(
                    max(_safe_float(row.get("top1_accuracy")) for row in rows)
                ),
                "final_topk_hit_rate": _round_float(final.get("topk_hit_rate")),
                "best_topk_hit_rate": _round_float(
                    max(_safe_float(row.get("topk_hit_rate")) for row in rows)
                ),
                "loss_delta": _round_float(final_loss - first_loss),
                "claim_boundary": PHASE64_CLAIM_BOUNDARY,
            }
        )
    return output
