from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path


PHASE33_BUDGET_ROBUSTNESS_CLAIM_BOUNDARY = (
    "Phase 33 is a bounded budget-robustness follow-up over Phase 30 "
    "normalized-B1 and compressed-control artifacts; it does not enable "
    "suitability reward, does not test B2/B3, does not test cross-region "
    "transfer, and does not support final submission-level planning-performance "
    "claims."
)

BUDGET_TRANSITION_FIELDNAMES = [
    "budget_label",
    "train_timesteps",
    "eval_max_steps",
    "phase30_status",
    "comparison_count",
    "closed_gap_count",
    "best_mean_delta",
    "best_gap_key",
    "best_mean_delta_change_from_previous",
    "claim_boundary",
]

FOCAL_GAP_TRANSITION_FIELDNAMES = [
    "budget_label",
    "train_timesteps",
    "variant_id",
    "comparator_variant_id",
    "mean_reward_delta",
    "positive_tile_seed_count",
    "total_tile_seed_count",
    "positive_fraction",
    "mean_delta_change_from_previous",
    "positive_count_change_from_previous",
    "gap_closed",
    "claim_boundary",
]

TILE_SEED_STABILITY_FIELDNAMES = [
    "variant_id",
    "comparator_variant_id",
    "eval_tile_id",
    "seed",
    "lower_budget_label",
    "higher_budget_label",
    "lower_train_timesteps",
    "higher_train_timesteps",
    "lower_delta",
    "higher_delta",
    "delta_change",
    "lower_positive",
    "higher_positive",
    "stability_class",
]

STABILITY_CLASSES = (
    "stable_positive",
    "stable_negative",
    "flip_to_positive",
    "flip_to_negative",
    "incomplete",
)


def build_phase33_budget_robustness(
    phase30_comparison_json_paths: Sequence[Path | str],
) -> dict[str, object]:
    if len(phase30_comparison_json_paths) < 2:
        raise ValueError("Phase 33 requires at least two Phase 30 comparison JSONs")

    budgets = [
        _phase30_budget_record(Path(path)) for path in phase30_comparison_json_paths
    ]
    budgets.sort(key=lambda item: (int(item["train_timesteps"]), str(item["source_path"])))
    budget_rows = _budget_transition_rows(budgets)
    gap_rows = _focal_gap_transition_rows(budgets)
    lower = budgets[0]
    higher = budgets[-1]
    stability_rows = _tile_seed_stability_rows(lower, higher)
    stability_counts = _stability_counts(stability_rows)
    status = _phase33_status(budget_rows, gap_rows, stability_counts)
    return {
        "phase": "phase33_budget_robustness",
        "source_phase30_comparison_jsons": [
            str(record["source_path"]) for record in budgets
        ],
        "ordered_budgets": [
            {
                "budget_label": record["budget_label"],
                "train_timesteps": record["train_timesteps"],
                "eval_max_steps": record["eval_max_steps"],
                "phase30_status": record["phase30_status"],
            }
            for record in budgets
        ],
        "budget_transition_rows": budget_rows,
        "focal_gap_transition_rows": gap_rows,
        "tile_seed_stability_rows": stability_rows,
        "tile_seed_stability_counts": stability_counts,
        "phase33_budget_status": status,
        "interpretation": _phase33_interpretation(status),
        "claim_boundary": PHASE33_BUDGET_ROBUSTNESS_CLAIM_BOUNDARY,
    }


def write_phase33_budget_robustness_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    budget_transition_path = output_path / "phase33_budget_transition.csv"
    focal_gap_transition_path = output_path / "phase33_focal_gap_transition.csv"
    tile_seed_stability_path = output_path / "phase33_tile_seed_stability.csv"
    summary_json_path = output_path / "phase33_budget_robustness.json"
    summary_md_path = output_path / "phase33_budget_robustness.md"

    _write_csv_mapping_rows(
        budget_transition_path,
        BUDGET_TRANSITION_FIELDNAMES,
        analysis.get("budget_transition_rows"),
        "budget_transition_rows",
    )
    _write_csv_mapping_rows(
        focal_gap_transition_path,
        FOCAL_GAP_TRANSITION_FIELDNAMES,
        analysis.get("focal_gap_transition_rows"),
        "focal_gap_transition_rows",
    )
    _write_csv_mapping_rows(
        tile_seed_stability_path,
        TILE_SEED_STABILITY_FIELDNAMES,
        analysis.get("tile_seed_stability_rows"),
        "tile_seed_stability_rows",
    )
    summary_json_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_md_path.write_text(_phase33_markdown(analysis), encoding="utf-8")
    return {
        "budget_transition_csv": budget_transition_path,
        "focal_gap_transition_csv": focal_gap_transition_path,
        "tile_seed_stability_csv": tile_seed_stability_path,
        "summary_json": summary_json_path,
        "summary_md": summary_md_path,
    }


def write_phase33_matched_baseline_comparison(
    baseline_phase30_comparison_json: Path | str,
    target_phase30_comparison_json: Path | str,
    output_path: Path | str,
) -> Path:
    baseline_path = Path(baseline_phase30_comparison_json)
    target_path = Path(target_phase30_comparison_json)
    baseline = _read_json_object(baseline_path)
    target = _read_json_object(target_path)
    target_keys = _delta_row_keys(_delta_rows(target, target_path))
    baseline_rows = [
        row
        for row in _delta_rows(baseline, baseline_path)
        if _delta_row_key(row) in target_keys
    ]
    if not baseline_rows:
        raise ValueError(
            "Phase 33 could not build a matched baseline subset; no baseline "
            "delta rows match the high-budget coverage"
        )
    subset = {
        key: value
        for key, value in dict(baseline).items()
        if key not in {"delta_rows", "learned_policy"}
    }
    subset["delta_rows"] = baseline_rows
    subset["learned_policy"] = {
        "mean_reward_by_variant": (
            baseline.get("learned_policy", {}).get("mean_reward_by_variant", {})
            if isinstance(baseline.get("learned_policy"), Mapping)
            else {}
        ),
        "focal_deltas": _focal_delta_summary_payload(baseline_rows),
    }
    subset["phase33_subset_source"] = str(baseline_path)
    subset["phase33_subset_target"] = str(target_path)
    subset["phase33_subset_row_count"] = len(baseline_rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_ready(subset), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def _phase30_budget_record(path: Path) -> dict[str, object]:
    payload = _read_json_object(path)
    delta_rows = _delta_rows(payload, path)
    train_timesteps = _metadata_int(payload.get("train_timesteps"), delta_rows, "train_timesteps")
    eval_max_steps = _metadata_int(payload.get("eval_max_steps"), delta_rows, "eval_max_steps")
    focal_deltas = _focal_delta_summaries(payload, delta_rows)
    tile_seed_index = _tile_seed_index(delta_rows)
    return {
        "source_path": str(path),
        "budget_label": f"{train_timesteps}_steps",
        "train_timesteps": train_timesteps,
        "eval_max_steps": eval_max_steps,
        "phase30_status": str(payload.get("phase30_normalized_b1_status", "")),
        "focal_deltas": focal_deltas,
        "tile_seed_index": tile_seed_index,
    }


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 33 input comparison JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Phase 33 input JSON must be an object")
    return value


def _delta_rows(payload: Mapping[str, object], path: Path) -> list[dict[str, object]]:
    rows = payload.get("delta_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Phase 30 comparison is missing delta_rows: {path}")
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"Phase 30 delta rows must be objects: {path}")
        normalized.append(dict(row))
    return normalized


def _metadata_int(
    value: object,
    rows: list[dict[str, object]],
    row_field: str,
) -> int:
    if value is not None and str(value).strip() != "":
        return int(value)
    for row in rows:
        row_value = row.get(row_field)
        if row_value is not None and str(row_value).strip() != "":
            return int(row_value)
    raise ValueError(f"Phase 33 cannot determine {row_field}")


def _focal_delta_summaries(
    payload: Mapping[str, object],
    delta_rows: list[dict[str, object]],
) -> dict[tuple[str, str], dict[str, object]]:
    learned = payload.get("learned_policy")
    if isinstance(learned, Mapping) and isinstance(learned.get("focal_deltas"), Mapping):
        summaries = {}
        for key, value in learned["focal_deltas"].items():
            if not isinstance(value, Mapping):
                continue
            parsed = _parse_gap_key(str(key))
            if parsed is None:
                continue
            variant_id, comparator_id = parsed
            summaries[(variant_id, comparator_id)] = {
                "variant_id": variant_id,
                "comparator_variant_id": comparator_id,
                "mean_reward_delta": _round_float(
                    _optional_float(value, "mean_reward_delta") or 0.0
                ),
                "positive_tile_seed_count": int(
                    value.get("positive_tile_seed_count") or 0
                ),
                "total_tile_seed_count": int(
                    value.get("total_tile_seed_count") or 0
                ),
                "positive_fraction": _optional_float(value, "positive_fraction"),
            }
        if summaries:
            return summaries

    grouped: dict[tuple[str, str], list[float]] = {}
    for row in delta_rows:
        key = (
            str(row.get("variant_id", "")),
            str(row.get("comparator_variant_id", "")),
        )
        grouped.setdefault(key, []).append(_float_value(row, "variant_minus_comparator_reward"))
    return {
        key: {
            "variant_id": key[0],
            "comparator_variant_id": key[1],
            "mean_reward_delta": _mean(values),
            "positive_tile_seed_count": sum(1 for value in values if value > 0.0),
            "total_tile_seed_count": len(values),
            "positive_fraction": _round_float(
                sum(1 for value in values if value > 0.0) / len(values)
            )
            if values
            else None,
        }
        for key, values in grouped.items()
    }


def _parse_gap_key(key: str) -> tuple[str, str] | None:
    if "_minus_" not in key:
        return None
    variant_id, comparator_id = key.split("_minus_", 1)
    if not variant_id or not comparator_id:
        return None
    return variant_id, comparator_id


def _tile_seed_index(
    rows: list[dict[str, object]],
) -> dict[tuple[str, str, str, int], dict[str, object]]:
    index: dict[tuple[str, str, str, int], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row.get("variant_id", "")),
            str(row.get("comparator_variant_id", "")),
            str(row.get("eval_tile_id", "")),
            _int_value(row, "seed"),
        )
        if key in index:
            raise ValueError(
                "Phase 33 requires unique variant/comparator/tile/seed delta rows"
            )
        index[key] = row
    return index


def _delta_row_key(row: Mapping[str, object]) -> tuple[str, str, str, int]:
    return (
        str(row.get("variant_id", "")),
        str(row.get("comparator_variant_id", "")),
        str(row.get("eval_tile_id", "")),
        _int_value(row, "seed"),
    )


def _delta_row_keys(rows: list[dict[str, object]]) -> set[tuple[str, str, str, int]]:
    return {_delta_row_key(row) for row in rows}


def _focal_delta_summary_payload(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["variant_id"]), str(row["comparator_variant_id"]))
        grouped.setdefault(key, []).append(_float_value(row, "variant_minus_comparator_reward"))
    result = {}
    for (variant_id, comparator_id), values in sorted(grouped.items()):
        positive_count = sum(1 for value in values if value > 0.0)
        total_count = len(values)
        result[f"{variant_id}_minus_{comparator_id}"] = {
            "mean_reward_delta": _mean(values),
            "std_reward_delta": 0.0,
            "positive_tile_seed_count": positive_count,
            "total_tile_seed_count": total_count,
            "positive_fraction": _round_float(positive_count / total_count)
            if total_count
            else None,
        }
    return result


def _budget_transition_rows(
    budgets: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    previous_best = None
    for budget in budgets:
        focal_deltas = _mapping_value(budget, "focal_deltas")
        comparable = list(focal_deltas.values())
        closed = [
            item
            for item in comparable
            if _optional_float(item, "mean_reward_delta") is not None
            and float(item["mean_reward_delta"]) >= -1e-9
        ]
        best = _best_gap(comparable)
        best_delta = None if best is None else best["mean_reward_delta"]
        row = {
            "budget_label": budget["budget_label"],
            "train_timesteps": budget["train_timesteps"],
            "eval_max_steps": budget["eval_max_steps"],
            "phase30_status": budget["phase30_status"],
            "comparison_count": len(comparable),
            "closed_gap_count": len(closed),
            "best_mean_delta": best_delta,
            "best_gap_key": _gap_key(best) if best is not None else "",
            "best_mean_delta_change_from_previous": None,
            "claim_boundary": PHASE33_BUDGET_ROBUSTNESS_CLAIM_BOUNDARY,
        }
        if previous_best is not None and best_delta is not None:
            row["best_mean_delta_change_from_previous"] = _round_float(
                float(best_delta) - float(previous_best)
            )
        rows.append(row)
        previous_best = best_delta
    return rows


def _focal_gap_transition_rows(
    budgets: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    previous_by_gap: dict[tuple[str, str], dict[str, object]] = {}
    for budget in budgets:
        focal_deltas = _mapping_value(budget, "focal_deltas")
        for key in sorted(focal_deltas):
            summary = dict(focal_deltas[key])
            previous = previous_by_gap.get(key)
            mean_delta = _optional_float(summary, "mean_reward_delta")
            positive_count = int(summary.get("positive_tile_seed_count") or 0)
            row = {
                "budget_label": budget["budget_label"],
                "train_timesteps": budget["train_timesteps"],
                "variant_id": key[0],
                "comparator_variant_id": key[1],
                "mean_reward_delta": mean_delta,
                "positive_tile_seed_count": positive_count,
                "total_tile_seed_count": int(summary.get("total_tile_seed_count") or 0),
                "positive_fraction": _optional_float(summary, "positive_fraction"),
                "mean_delta_change_from_previous": None,
                "positive_count_change_from_previous": None,
                "gap_closed": mean_delta is not None and mean_delta >= -1e-9,
                "claim_boundary": PHASE33_BUDGET_ROBUSTNESS_CLAIM_BOUNDARY,
            }
            if previous is not None and mean_delta is not None:
                previous_delta = _optional_float(previous, "mean_reward_delta")
                if previous_delta is not None:
                    row["mean_delta_change_from_previous"] = _round_float(
                        mean_delta - previous_delta
                    )
                row["positive_count_change_from_previous"] = positive_count - int(
                    previous.get("positive_tile_seed_count") or 0
                )
            rows.append(row)
            previous_by_gap[key] = summary
    return rows


def _tile_seed_stability_rows(
    lower: Mapping[str, object],
    higher: Mapping[str, object],
) -> list[dict[str, object]]:
    lower_index = _mapping_value(lower, "tile_seed_index")
    higher_index = _mapping_value(higher, "tile_seed_index")
    keys = sorted(set(lower_index) | set(higher_index))
    rows = []
    for variant_id, comparator_id, eval_tile_id, seed in keys:
        lower_row = lower_index.get((variant_id, comparator_id, eval_tile_id, seed))
        higher_row = higher_index.get((variant_id, comparator_id, eval_tile_id, seed))
        lower_delta = (
            _float_value(lower_row, "variant_minus_comparator_reward")
            if isinstance(lower_row, Mapping)
            else None
        )
        higher_delta = (
            _float_value(higher_row, "variant_minus_comparator_reward")
            if isinstance(higher_row, Mapping)
            else None
        )
        stability_class = _stability_class(lower_delta, higher_delta)
        rows.append(
            {
                "variant_id": variant_id,
                "comparator_variant_id": comparator_id,
                "eval_tile_id": eval_tile_id,
                "seed": seed,
                "lower_budget_label": lower["budget_label"],
                "higher_budget_label": higher["budget_label"],
                "lower_train_timesteps": lower["train_timesteps"],
                "higher_train_timesteps": higher["train_timesteps"],
                "lower_delta": lower_delta,
                "higher_delta": higher_delta,
                "delta_change": _round_float(higher_delta - lower_delta)
                if lower_delta is not None and higher_delta is not None
                else None,
                "lower_positive": lower_delta > 0.0 if lower_delta is not None else None,
                "higher_positive": higher_delta > 0.0 if higher_delta is not None else None,
                "stability_class": stability_class,
            }
        )
    return rows


def _stability_class(
    lower_delta: float | None,
    higher_delta: float | None,
) -> str:
    if lower_delta is None or higher_delta is None:
        return "incomplete"
    lower_positive = lower_delta > 0.0
    higher_positive = higher_delta > 0.0
    if lower_positive and higher_positive:
        return "stable_positive"
    if not lower_positive and not higher_positive:
        return "stable_negative"
    if not lower_positive and higher_positive:
        return "flip_to_positive"
    return "flip_to_negative"


def _stability_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts = {key: 0 for key in STABILITY_CLASSES}
    for row in rows:
        key = str(row.get("stability_class", "incomplete"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _phase33_status(
    budget_rows: list[dict[str, object]],
    gap_rows: list[dict[str, object]],
    stability_counts: Mapping[str, int],
) -> str:
    if len(budget_rows) < 2 or int(stability_counts.get("incomplete", 0)) > 0:
        return "insufficient"
    higher_rows = [
        row
        for row in gap_rows
        if int(row.get("train_timesteps", 0)) == int(budget_rows[-1]["train_timesteps"])
    ]
    if not higher_rows:
        return "insufficient"
    closed = [
        row for row in higher_rows if bool(row.get("gap_closed"))
    ]
    improved = [
        row
        for row in higher_rows
        if row.get("mean_delta_change_from_previous") is not None
        and float(row["mean_delta_change_from_previous"]) > 1e-9
    ]
    if len(closed) == len(higher_rows):
        return "budget_closes_compressed_gap"
    if improved:
        return "budget_improves_but_not_closed"
    return "budget_not_explanatory"


def _phase33_interpretation(status: str) -> str:
    if status == "budget_closes_compressed_gap":
        return (
            "The higher budget closes all tracked normalized-B1 versus "
            "compressed-control gaps in this bounded result set."
        )
    if status == "budget_improves_but_not_closed":
        return (
            "The higher budget improves at least one tracked gap, but at least "
            "one compressed-control gap remains open."
        )
    if status == "budget_not_explanatory":
        return (
            "The higher budget does not improve the tracked normalized-B1 "
            "compressed-control gaps."
        )
    return "Phase 33 cannot compare budgets because coverage is incomplete."


def _phase33_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Phase 33 Budget Robustness",
        "",
        f"Status: {analysis.get('phase33_budget_status', '')}",
        "",
        "## Focal Gap Transition",
        "",
        "| Budget | Gap | Mean delta | Change | Positive |",
        "|---|---|---:|---:|---:|",
    ]
    for row in analysis.get("focal_gap_transition_rows", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {budget} | {variant} - {comp} | `{delta}` | `{change}` | {pos}/{total} |".format(
                budget=row.get("budget_label", ""),
                variant=row.get("variant_id", ""),
                comp=row.get("comparator_variant_id", ""),
                delta=row.get("mean_reward_delta", ""),
                change=row.get("mean_delta_change_from_previous", ""),
                pos=row.get("positive_tile_seed_count", ""),
                total=row.get("total_tile_seed_count", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(analysis.get("interpretation", "")),
            "",
            "## Claim Boundary",
            "",
            str(analysis.get("claim_boundary", "")),
            "",
        ]
    )
    return "\n".join(lines)


def _best_gap(rows: list[object]) -> Mapping[str, object] | None:
    mappings = [row for row in rows if isinstance(row, Mapping)]
    if not mappings:
        return None
    return max(
        mappings,
        key=lambda row: float(row.get("mean_reward_delta") or float("-inf")),
    )


def _gap_key(row: Mapping[str, object]) -> str:
    return f"{row.get('variant_id', '')}_minus_{row.get('comparator_variant_id', '')}"


def _mapping_value(
    mapping: Mapping[str, object],
    key: str,
) -> dict:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Phase 33 record is missing {key}")
    return value


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    row_key: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 33 analysis is missing {row_key}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 33 {row_key} rows must be objects")
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    return value


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _optional_float(row: Mapping[str, object], field: str) -> float | None:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _float_value(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing numeric field: {field}")
    return float(value)


def _int_value(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing integer field: {field}")
    return int(value)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return _round_float(sum(float(value) for value in values) / len(values))


def _round_float(value: float) -> float:
    return round(float(value), 10)
