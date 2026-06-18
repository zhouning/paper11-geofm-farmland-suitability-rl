from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path


PHASE27_CLAIM_BOUNDARY = (
    "Phase 27 is a read-only diagnosis of existing Phase 26 B0/B1 padded "
    "held-out Bishan learned-policy artifacts; it does not run new training, "
    "does not enable suitability reward, does not test B2/B3, and does not "
    "support cross-region transfer or final submission-level claims."
)

PHASE27_REMAINING_EVIDENCE_GAPS = [
    "representation_controls_against_random_shuffled_pca_features",
    "repeated_or_intermediate_budget_stability_sweep",
    "suitability_proxy_validation_before_reward_use",
    "held_out_region_transfer_evaluation",
    "spatial_case_maps_and_uncertainty",
]

BUDGET_TRANSITION_FIELDNAMES = [
    "budget_label",
    "train_timesteps",
    "eval_max_steps",
    "b1_minus_b0_mean_reward",
    "positive_tile_seed_count",
    "total_tile_seed_count",
    "positive_fraction",
    "phase26_claim_status",
    "mean_delta_change_from_previous",
    "positive_count_change_from_previous",
    "claim_boundary",
]

TILE_SEED_STABILITY_FIELDNAMES = [
    "eval_tile_id",
    "seed",
    "lower_budget_label",
    "higher_budget_label",
    "lower_train_timesteps",
    "higher_train_timesteps",
    "lower_b1_minus_b0_reward",
    "higher_b1_minus_b0_reward",
    "delta_change",
    "lower_b1_improves_b0",
    "higher_b1_improves_b0",
    "stability_class",
    "diagnostic_note",
]

STABILITY_CLASSES = (
    "stable_positive",
    "stable_negative",
    "flip_to_positive",
    "flip_to_negative",
    "incomplete",
)


def build_phase27_stability_diagnosis(
    phase26_comparison_json_paths: Sequence[Path | str],
) -> dict[str, object]:
    if len(phase26_comparison_json_paths) < 2:
        raise ValueError("Phase 27 requires at least two Phase 26 comparison JSONs")

    budgets = [
        _phase26_budget_record(Path(path)) for path in phase26_comparison_json_paths
    ]
    budgets.sort(key=lambda item: (int(item["train_timesteps"]), str(item["source_path"])))

    budget_transition_rows = _budget_transition_rows(budgets)
    lower = budgets[0]
    higher = budgets[-1]
    tile_seed_stability_rows = _tile_seed_stability_rows(lower, higher)
    stability_counts = _stability_counts(tile_seed_stability_rows)
    status = _phase27_diagnostic_status(budget_transition_rows, stability_counts)

    return {
        "phase": "phase27_b0_b1_stability_diagnosis",
        "source_phase26_comparison_jsons": [
            str(record["source_path"]) for record in budgets
        ],
        "ordered_budgets": [
            {
                "budget_label": record["budget_label"],
                "train_timesteps": record["train_timesteps"],
                "eval_max_steps": record["eval_max_steps"],
                "phase26_claim_status": record["phase26_claim_status"],
            }
            for record in budgets
        ],
        "budget_transition_rows": budget_transition_rows,
        "tile_seed_stability_rows": tile_seed_stability_rows,
        "stability_counts": stability_counts,
        "per_tile_transition_summary": _group_transition_summary(
            tile_seed_stability_rows,
            "eval_tile_id",
        ),
        "per_seed_transition_summary": _group_transition_summary(
            tile_seed_stability_rows,
            "seed",
        ),
        "phase27_diagnostic_status": status,
        "recommendation": _phase27_recommendation(status, stability_counts),
        "remaining_evidence_gaps": list(PHASE27_REMAINING_EVIDENCE_GAPS),
        "claim_boundary": PHASE27_CLAIM_BOUNDARY,
    }


def write_phase27_stability_diagnosis_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    budget_transition_path = output_path / "phase27_budget_transition_table.csv"
    tile_seed_stability_path = output_path / "phase27_tile_seed_stability.csv"
    diagnostic_summary_path = output_path / "phase27_diagnostic_summary.json"
    diagnostic_readiness_path = output_path / "phase27_diagnostic_readiness.md"

    _write_csv_mapping_rows(
        budget_transition_path,
        BUDGET_TRANSITION_FIELDNAMES,
        analysis.get("budget_transition_rows"),
        "budget_transition_rows",
    )
    _write_csv_mapping_rows(
        tile_seed_stability_path,
        TILE_SEED_STABILITY_FIELDNAMES,
        analysis.get("tile_seed_stability_rows"),
        "tile_seed_stability_rows",
    )
    diagnostic_summary_path.write_text(
        json.dumps(dict(analysis), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    diagnostic_readiness_path.write_text(
        _phase27_diagnostic_readiness_markdown(analysis),
        encoding="utf-8",
    )

    return {
        "budget_transition_csv": budget_transition_path,
        "tile_seed_stability_csv": tile_seed_stability_path,
        "diagnostic_summary_json": diagnostic_summary_path,
        "diagnostic_readiness_md": diagnostic_readiness_path,
    }


def _phase26_budget_record(path: Path) -> dict[str, object]:
    comparison = _read_json_object(path)
    tile_seed_rows = _tile_seed_rows(comparison, path)
    train_timesteps = _metadata_int(
        comparison.get("train_timesteps"),
        tile_seed_rows,
        "train_timesteps",
    )
    eval_max_steps = _metadata_int(
        comparison.get("eval_max_steps"),
        tile_seed_rows,
        "eval_max_steps",
    )
    learned = comparison.get("learned_policy")
    if not isinstance(learned, Mapping):
        learned = {}
    deltas = [_float_value(row, "b1_minus_b0_reward") for row in tile_seed_rows]
    positive_count = sum(1 for value in deltas if value > 0)
    total_count = len(deltas)
    mean_delta = learned.get("B1_minus_B0_mean_reward")
    if mean_delta is None:
        mean_delta = _mean_or_none(deltas)
    positive_fraction = learned.get("positive_fraction")
    if positive_fraction is None:
        positive_fraction = _round_float(positive_count / total_count) if total_count else None

    return {
        "source_path": str(path),
        "budget_label": f"{train_timesteps}_steps",
        "train_timesteps": train_timesteps,
        "eval_max_steps": eval_max_steps,
        "b1_minus_b0_mean_reward": _round_float(float(mean_delta))
        if mean_delta is not None
        else None,
        "positive_tile_seed_count": int(
            learned.get("positive_tile_seed_count", positive_count)
        ),
        "total_tile_seed_count": int(
            learned.get("total_tile_seed_count", total_count)
        ),
        "positive_fraction": _round_float(float(positive_fraction))
        if positive_fraction is not None
        else None,
        "phase26_claim_status": str(comparison.get("phase26_claim_status", "")),
        "tile_seed_rows": tile_seed_rows,
        "tile_seed_index": _tile_seed_index(tile_seed_rows),
    }


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 27 input comparison JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Phase 27 input JSON must be an object")
    return value


def _tile_seed_rows(
    comparison: Mapping[str, object],
    path: Path,
) -> list[dict[str, object]]:
    rows = comparison.get("tile_seed_delta_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Phase 26 comparison is missing tile_seed_delta_rows: {path}")
    normalized: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"Phase 26 tile-seed rows must be objects: {path}")
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
    raise ValueError(f"Phase 27 cannot determine {row_field}")


def _tile_seed_index(
    rows: list[dict[str, object]],
) -> dict[tuple[str, int], dict[str, object]]:
    index: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        key = (str(row.get("eval_tile_id", "")), _int_value(row, "seed"))
        if key in index:
            raise ValueError(
                "Phase 27 requires unique tile-seed rows; duplicate "
                f"{key[0]} seed {key[1]}"
            )
        index[key] = row
    return index


def _budget_transition_rows(
    budgets: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous: dict[str, object] | None = None
    for record in budgets:
        mean_delta = record["b1_minus_b0_mean_reward"]
        positive_count = record["positive_tile_seed_count"]
        row = {
            "budget_label": record["budget_label"],
            "train_timesteps": record["train_timesteps"],
            "eval_max_steps": record["eval_max_steps"],
            "b1_minus_b0_mean_reward": mean_delta,
            "positive_tile_seed_count": positive_count,
            "total_tile_seed_count": record["total_tile_seed_count"],
            "positive_fraction": record["positive_fraction"],
            "phase26_claim_status": record["phase26_claim_status"],
            "mean_delta_change_from_previous": None,
            "positive_count_change_from_previous": None,
            "claim_boundary": PHASE27_CLAIM_BOUNDARY,
        }
        if previous is not None:
            row["mean_delta_change_from_previous"] = _round_float(
                float(mean_delta) - float(previous["b1_minus_b0_mean_reward"])
            )
            row["positive_count_change_from_previous"] = int(positive_count) - int(
                previous["positive_tile_seed_count"]
            )
        rows.append(row)
        previous = record
    return rows


def _tile_seed_stability_rows(
    lower: Mapping[str, object],
    higher: Mapping[str, object],
) -> list[dict[str, object]]:
    lower_index = _mapping_value(lower, "tile_seed_index")
    higher_index = _mapping_value(higher, "tile_seed_index")
    keys = sorted(
        set(lower_index) | set(higher_index),
        key=lambda item: (item[0], item[1]),
    )
    rows: list[dict[str, object]] = []
    for eval_tile_id, seed in keys:
        lower_row = lower_index.get((eval_tile_id, seed))
        higher_row = higher_index.get((eval_tile_id, seed))
        lower_delta = (
            _float_value(lower_row, "b1_minus_b0_reward")
            if isinstance(lower_row, Mapping)
            else None
        )
        higher_delta = (
            _float_value(higher_row, "b1_minus_b0_reward")
            if isinstance(higher_row, Mapping)
            else None
        )
        stability_class = _stability_class(lower_delta, higher_delta)
        delta_change = (
            _round_float(float(higher_delta) - float(lower_delta))
            if lower_delta is not None and higher_delta is not None
            else None
        )
        rows.append(
            {
                "eval_tile_id": eval_tile_id,
                "seed": seed,
                "lower_budget_label": lower["budget_label"],
                "higher_budget_label": higher["budget_label"],
                "lower_train_timesteps": lower["train_timesteps"],
                "higher_train_timesteps": higher["train_timesteps"],
                "lower_b1_minus_b0_reward": lower_delta,
                "higher_b1_minus_b0_reward": higher_delta,
                "delta_change": delta_change,
                "lower_b1_improves_b0": lower_delta > 0
                if lower_delta is not None
                else None,
                "higher_b1_improves_b0": higher_delta > 0
                if higher_delta is not None
                else None,
                "stability_class": stability_class,
                "diagnostic_note": _diagnostic_note(stability_class),
            }
        )
    return rows


def _mapping_value(
    mapping: Mapping[str, object],
    key: str,
) -> dict[tuple[str, int], dict[str, object]]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Phase 27 budget record is missing {key}")
    return value


def _stability_class(
    lower_delta: float | None,
    higher_delta: float | None,
) -> str:
    if lower_delta is None or higher_delta is None:
        return "incomplete"
    lower_positive = lower_delta > 0
    higher_positive = higher_delta > 0
    if lower_positive and higher_positive:
        return "stable_positive"
    if not lower_positive and not higher_positive:
        return "stable_negative"
    if not lower_positive and higher_positive:
        return "flip_to_positive"
    return "flip_to_negative"


def _diagnostic_note(stability_class: str) -> str:
    notes = {
        "stable_positive": "B1 remains above B0 across budgets for this tile-seed pair.",
        "stable_negative": "B1 remains non-positive relative to B0 across budgets for this tile-seed pair.",
        "flip_to_positive": "B1 changes from non-positive to positive as budget increases.",
        "flip_to_negative": "B1 changes from positive to non-positive as budget increases.",
        "incomplete": "This tile-seed pair is missing from one budget and cannot be compared.",
    }
    return notes[stability_class]


def _stability_counts(
    rows: list[dict[str, object]],
) -> dict[str, int]:
    counts = {key: 0 for key in STABILITY_CLASSES}
    for row in rows:
        stability_class = str(row.get("stability_class", "incomplete"))
        counts[stability_class] = counts.get(stability_class, 0) + 1
    return counts


def _group_transition_summary(
    rows: list[dict[str, object]],
    group_field: str,
) -> dict[str, dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        key = str(row.get(group_field, ""))
        groups.setdefault(key, []).append(row)

    summary: dict[str, dict[str, object]] = {}
    for key, group_rows in groups.items():
        changes = [
            float(row["delta_change"])
            for row in group_rows
            if row.get("delta_change") is not None
        ]
        counts = _stability_counts(group_rows)
        summary[key] = {
            "paired_count": len(changes),
            "mean_delta_change": _mean_or_none(changes),
            "stability_counts": counts,
        }
    return summary


def _phase27_diagnostic_status(
    budget_transition_rows: list[dict[str, object]],
    stability_counts: Mapping[str, int],
) -> str:
    if len(budget_transition_rows) < 2 or int(stability_counts.get("incomplete", 0)) > 0:
        return "insufficient"
    higher = budget_transition_rows[-1]
    higher_delta = higher.get("b1_minus_b0_mean_reward")
    higher_fraction = higher.get("positive_fraction")
    if higher_delta is None or higher_fraction is None:
        return "insufficient"
    all_not_supported = all(
        row.get("phase26_claim_status") == "not_supported"
        for row in budget_transition_rows
    )
    if (
        all_not_supported
        and float(higher_delta) <= 0
        and float(higher_fraction) < 0.6
    ):
        return "budget_not_explanatory"
    if float(higher_delta) > 0 and float(higher_fraction) >= 0.6:
        return "budget_promising_stable"
    if float(higher_delta) > 0 and float(higher_fraction) < 0.6:
        return "budget_promising_unstable"
    return "budget_not_explanatory"


def _phase27_recommendation(
    status: str,
    stability_counts: Mapping[str, int],
) -> str:
    if status == "budget_promising_stable":
        return (
            "Treat budget as a plausible contributor, then replicate with "
            "representation controls before making any B1 superiority claim."
        )
    if status == "budget_promising_unstable":
        return (
            "Run repeated or intermediate budgets because the mean improves "
            "without stable tile-seed support."
        )
    if status == "insufficient":
        return (
            "Repair Phase 26 coverage before interpreting budget sensitivity."
        )
    if int(stability_counts.get("flip_to_negative", 0)) > 0:
        return (
            "Do not extend the B1 superiority claim; prioritize representation "
            "controls and a repeated-budget stability sweep."
        )
    return (
        "Budget alone does not explain the current negative evidence; prioritize "
        "representation controls and suitability-proxy validation."
    )


def _phase27_diagnostic_readiness_markdown(
    analysis: Mapping[str, object],
) -> str:
    rows = analysis.get("budget_transition_rows")
    if not isinstance(rows, list):
        rows = []
    stability_counts = analysis.get("stability_counts")
    if not isinstance(stability_counts, Mapping):
        stability_counts = {}

    lines = [
        "# Phase 27 Diagnostic Readiness",
        "",
        f"Status: {analysis.get('phase27_diagnostic_status', '')}",
        "",
        "Budget transition:",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "- "
            f"{row.get('train_timesteps')} steps: "
            f"B1-B0 mean delta {row.get('b1_minus_b0_mean_reward')}, "
            f"positive tile-seed count "
            f"{row.get('positive_tile_seed_count')} / "
            f"{row.get('total_tile_seed_count')}, "
            f"Phase 26 status {row.get('phase26_claim_status')}"
        )

    lines.extend(
        [
            "",
            "Tile-seed stability:",
            (
                "- stable_positive: "
                f"{stability_counts.get('stable_positive', 0)}"
            ),
            (
                "- stable_negative: "
                f"{stability_counts.get('stable_negative', 0)}"
            ),
            (
                "- flip_to_positive: "
                f"{stability_counts.get('flip_to_positive', 0)}"
            ),
            (
                "- flip_to_negative: "
                f"{stability_counts.get('flip_to_negative', 0)}"
            ),
            (
                "- incomplete: "
                f"{stability_counts.get('incomplete', 0)}"
            ),
            "",
            "Interpretation:",
            str(analysis.get("recommendation", "")),
            "",
            str(analysis.get("claim_boundary", PHASE27_CLAIM_BOUNDARY)),
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: list[str],
    rows: object,
    row_key: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 27 analysis is missing a {row_key} list")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 27 {row_key} rows must be objects")
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    return value


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


def _round_float(value: float) -> float:
    return round(float(value), 10)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return _round_float(sum(values) / len(values))
