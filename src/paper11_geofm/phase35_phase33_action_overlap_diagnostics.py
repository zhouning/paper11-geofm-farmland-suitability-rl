from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path


PHASE35_ACTION_OVERLAP_CLAIM_BOUNDARY = (
    "Phase 35 is a read-only action-overlap diagnostic over existing Phase 33 "
    "matched pilot artifacts; it does not run new policy training, does not "
    "alter rewards, does not enable suitability reward, does not test B2/B3, "
    "and does not support final submission-level planning-performance claims."
)

PHASE35_CASE_FIELDNAMES = [
    "case_id",
    "case_role",
    "eval_tile_id",
    "seed",
    "variant_id",
    "comparator_variant_id",
    "stability_class",
    "lower_delta",
    "higher_delta",
    "delta_change",
    "variant_summary_reward",
    "comparator_summary_reward",
    "summary_reward_gap",
    "variant_step_source",
    "comparator_step_source",
    "variant_step_count",
    "comparator_step_count",
    "shared_block_count",
    "union_block_count",
    "selected_block_jaccard",
    "same_step_match_count",
    "mean_abs_shared_step_displacement",
    "max_abs_shared_step_displacement",
    "variant_trace_cumulative_reward",
    "comparator_trace_cumulative_reward",
    "trace_cumulative_reward_gap",
    "first_step_reward_gap",
    "action_overlap_pattern",
    "source_phase33_output_dir",
    "claim_boundary",
]

PHASE35_STEP_FIELDNAMES = [
    "case_id",
    "step",
    "eval_tile_id",
    "seed",
    "variant_id",
    "comparator_variant_id",
    "variant_block_id",
    "comparator_block_id",
    "same_step_block_match",
    "variant_step_reward",
    "comparator_step_reward",
    "step_reward_gap",
    "variant_cumulative_reward",
    "comparator_cumulative_reward",
    "cumulative_reward_gap",
    "variant_step_source",
    "comparator_step_source",
    "claim_boundary",
]


def build_phase35_phase33_action_overlap_diagnostics(
    phase33_output_dirs: Sequence[Path | str],
    *,
    variants: Sequence[str] | str = ("N1Z", "N1ZR"),
    comparators: Sequence[str] | str = ("B1", "D4P8", "D4P16"),
) -> dict[str, object]:
    output_dirs = [Path(path) for path in phase33_output_dirs]
    if not output_dirs:
        raise ValueError("Phase 35 requires at least one Phase 33 output directory")

    variant_filter = set(_normalize_csvish_values(variants))
    comparator_filter = set(_normalize_csvish_values(comparators))
    case_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []

    for output_dir in output_dirs:
        summary_csv = output_dir / "phase30_high_budget" / "phase30_normalized_b1_summary.csv"
        traces_json = output_dir / "phase30_high_budget" / "phase30_normalized_b1_traces.json"
        stability_csv = output_dir / "phase33_tile_seed_stability.csv"
        summary_rows = _trained_summary_index(
            _read_csv_rows(summary_csv, "Phase 33 high-budget summary CSV")
        )
        traces = _read_trace_payload(traces_json)
        stability_rows = _read_csv_rows(stability_csv, "Phase 33 tile-seed stability CSV")
        for stability in stability_rows:
            variant_id = str(stability.get("variant_id", "")).strip()
            comparator_id = str(stability.get("comparator_variant_id", "")).strip()
            if variant_filter and variant_id not in variant_filter:
                continue
            if comparator_filter and comparator_id not in comparator_filter:
                continue
            tile_id = str(stability.get("eval_tile_id", "")).strip()
            seed = _int_value(stability, "seed")
            variant_summary = summary_rows.get((variant_id, tile_id, seed))
            comparator_summary = summary_rows.get((comparator_id, tile_id, seed))
            if variant_summary is None or comparator_summary is None:
                continue
            variant_steps = _ordered_steps(
                traces,
                variant_id,
                tile_id,
                seed,
                fallback_blocks=_selected_block_ids(variant_summary),
            )
            comparator_steps = _ordered_steps(
                traces,
                comparator_id,
                tile_id,
                seed,
                fallback_blocks=_selected_block_ids(comparator_summary),
            )
            if not variant_steps["steps"] or not comparator_steps["steps"]:
                continue
            case_row = _case_row(
                stability,
                variant_summary,
                comparator_summary,
                variant_steps,
                comparator_steps,
                output_dir,
            )
            case_rows.append(case_row)
            step_rows.extend(_step_rows(case_row, variant_steps, comparator_steps))

    status = (
        "action_overlap_diagnostics_ready"
        if case_rows and step_rows
        else "action_overlap_diagnostics_insufficient"
    )
    return {
        "phase": "phase35_phase33_action_overlap_diagnostics",
        "phase35_action_overlap_status": status,
        "source_paths": {
            "phase33_output_dirs": [str(path) for path in output_dirs],
        },
        "variants": sorted(variant_filter),
        "comparators": sorted(comparator_filter),
        "row_counts": {
            "phase33_output_dirs": len(output_dirs),
            "case_rows": len(case_rows),
            "step_rows": len(step_rows),
        },
        "case_rows": case_rows,
        "step_rows": step_rows,
        "interpretation": _phase35_interpretation(status),
        "claim_boundary": PHASE35_ACTION_OVERLAP_CLAIM_BOUNDARY,
    }


def write_phase35_phase33_action_overlap_diagnostics_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    case_summary_path = output_path / "phase35_action_overlap_cases.csv"
    step_alignment_path = output_path / "phase35_action_overlap_steps.csv"
    diagnosis_json_path = output_path / "phase35_action_overlap_diagnostics.json"
    diagnosis_md_path = output_path / "phase35_action_overlap_diagnostics.md"

    _write_csv_mapping_rows(
        case_summary_path,
        PHASE35_CASE_FIELDNAMES,
        analysis.get("case_rows"),
        "case_rows",
    )
    _write_csv_mapping_rows(
        step_alignment_path,
        PHASE35_STEP_FIELDNAMES,
        analysis.get("step_rows"),
        "step_rows",
    )
    diagnosis_json_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    diagnosis_md_path.write_text(_phase35_markdown(analysis), encoding="utf-8")
    return {
        "case_summary_csv": case_summary_path,
        "step_alignment_csv": step_alignment_path,
        "diagnosis_json": diagnosis_json_path,
        "diagnosis_md": diagnosis_md_path,
    }


def _case_row(
    stability: Mapping[str, object],
    variant_summary: Mapping[str, object],
    comparator_summary: Mapping[str, object],
    variant_steps: Mapping[str, object],
    comparator_steps: Mapping[str, object],
    output_dir: Path,
) -> dict[str, object]:
    variant_step_rows = _step_sequence(variant_steps)
    comparator_step_rows = _step_sequence(comparator_steps)
    variant_positions = _block_positions(variant_step_rows)
    comparator_positions = _block_positions(comparator_step_rows)
    variant_blocks = set(variant_positions)
    comparator_blocks = set(comparator_positions)
    shared = variant_blocks & comparator_blocks
    union = variant_blocks | comparator_blocks
    displacements = [
        abs(variant_positions[block_id] - comparator_positions[block_id])
        for block_id in shared
    ]
    same_step_matches = sum(
        1
        for variant_step, comparator_step in zip(variant_step_rows, comparator_step_rows)
        if str(variant_step.get("selected_block_id", "")).strip()
        == str(comparator_step.get("selected_block_id", "")).strip()
    )
    variant_trace_total = _trace_reward_sum(variant_step_rows)
    comparator_trace_total = _trace_reward_sum(comparator_step_rows)
    trace_gap = (
        _round_float(variant_trace_total - comparator_trace_total)
        if variant_trace_total is not None and comparator_trace_total is not None
        else ""
    )
    first_gap = _first_step_reward_gap(variant_step_rows, comparator_step_rows)
    variant_reward = _float_value(variant_summary, "total_contract_reward")
    comparator_reward = _float_value(comparator_summary, "total_contract_reward")
    summary_gap = _round_float(variant_reward - comparator_reward)
    tile_id = str(stability.get("eval_tile_id", "")).strip()
    seed = _int_value(stability, "seed")
    variant_id = str(stability.get("variant_id", "")).strip()
    comparator_id = str(stability.get("comparator_variant_id", "")).strip()
    return {
        "case_id": f"{tile_id}|{seed}|{variant_id}|{comparator_id}",
        "case_role": _case_role(_float_value(stability, "higher_delta")),
        "eval_tile_id": tile_id,
        "seed": seed,
        "variant_id": variant_id,
        "comparator_variant_id": comparator_id,
        "stability_class": str(stability.get("stability_class", "")),
        "lower_delta": _float_value(stability, "lower_delta"),
        "higher_delta": _float_value(stability, "higher_delta"),
        "delta_change": _optional_float(stability, "delta_change"),
        "variant_summary_reward": _round_float(variant_reward),
        "comparator_summary_reward": _round_float(comparator_reward),
        "summary_reward_gap": summary_gap,
        "variant_step_source": str(variant_steps["source"]),
        "comparator_step_source": str(comparator_steps["source"]),
        "variant_step_count": len(variant_step_rows),
        "comparator_step_count": len(comparator_step_rows),
        "shared_block_count": len(shared),
        "union_block_count": len(union),
        "selected_block_jaccard": _round_float(len(shared) / len(union)) if union else 1.0,
        "same_step_match_count": same_step_matches,
        "mean_abs_shared_step_displacement": _mean(displacements),
        "max_abs_shared_step_displacement": max(displacements) if displacements else "",
        "variant_trace_cumulative_reward": _csv_optional_float(variant_trace_total),
        "comparator_trace_cumulative_reward": _csv_optional_float(comparator_trace_total),
        "trace_cumulative_reward_gap": trace_gap,
        "first_step_reward_gap": first_gap,
        "action_overlap_pattern": _action_overlap_pattern(
            shared,
            union,
            summary_gap,
        ),
        "source_phase33_output_dir": str(output_dir),
        "claim_boundary": PHASE35_ACTION_OVERLAP_CLAIM_BOUNDARY,
    }


def _step_rows(
    case_row: Mapping[str, object],
    variant_steps: Mapping[str, object],
    comparator_steps: Mapping[str, object],
) -> list[dict[str, object]]:
    variant_step_rows = _step_sequence(variant_steps)
    comparator_step_rows = _step_sequence(comparator_steps)
    rows: list[dict[str, object]] = []
    variant_cumulative = 0.0
    comparator_cumulative = 0.0
    max_steps = max(len(variant_step_rows), len(comparator_step_rows))
    for index in range(max_steps):
        variant_step = variant_step_rows[index] if index < len(variant_step_rows) else {}
        comparator_step = (
            comparator_step_rows[index] if index < len(comparator_step_rows) else {}
        )
        variant_reward = _optional_float(variant_step, "reward")
        comparator_reward = _optional_float(comparator_step, "reward")
        variant_cumulative += variant_reward or 0.0
        comparator_cumulative += comparator_reward or 0.0
        variant_block = str(variant_step.get("selected_block_id", "")).strip()
        comparator_block = str(comparator_step.get("selected_block_id", "")).strip()
        has_rewards = variant_reward is not None and comparator_reward is not None
        rows.append(
            {
                "case_id": case_row["case_id"],
                "step": index + 1,
                "eval_tile_id": case_row["eval_tile_id"],
                "seed": case_row["seed"],
                "variant_id": case_row["variant_id"],
                "comparator_variant_id": case_row["comparator_variant_id"],
                "variant_block_id": variant_block,
                "comparator_block_id": comparator_block,
                "same_step_block_match": variant_block == comparator_block and bool(variant_block),
                "variant_step_reward": _csv_optional_float(variant_reward),
                "comparator_step_reward": _csv_optional_float(comparator_reward),
                "step_reward_gap": _round_float(variant_reward - comparator_reward)
                if has_rewards
                else "",
                "variant_cumulative_reward": _round_float(variant_cumulative)
                if _has_any_reward(variant_step_rows[: index + 1])
                else "",
                "comparator_cumulative_reward": _round_float(comparator_cumulative)
                if _has_any_reward(comparator_step_rows[: index + 1])
                else "",
                "cumulative_reward_gap": _round_float(
                    variant_cumulative - comparator_cumulative
                )
                if has_rewards
                else "",
                "variant_step_source": variant_steps["source"],
                "comparator_step_source": comparator_steps["source"],
                "claim_boundary": PHASE35_ACTION_OVERLAP_CLAIM_BOUNDARY,
            }
        )
    return rows


def _read_csv_rows(path: Path, label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_trace_payload(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 33 high-budget traces JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("traces"), Mapping):
        return payload["traces"]
    return payload


def _trained_summary_index(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, int], Mapping[str, object]]:
    indexed: dict[tuple[str, str, int], Mapping[str, object]] = {}
    for row in rows:
        if str(row.get("row_type", "")).strip() != "trained_policy":
            continue
        variant_id = str(row.get("variant_id", "")).strip()
        tile_id = str(row.get("eval_tile_id", "")).strip()
        if not variant_id or not tile_id:
            continue
        key = (variant_id, tile_id, _int_value(row, "seed"))
        if key in indexed:
            raise ValueError(
                "Phase 35 requires unique trained_policy rows for "
                f"{variant_id} {tile_id} seed {key[2]}"
            )
        indexed[key] = row
    return indexed


def _ordered_steps(
    traces: object,
    variant_id: str,
    tile_id: str,
    seed: int,
    *,
    fallback_blocks: Sequence[str],
) -> dict[str, object]:
    trace_steps = _trace_steps(traces, variant_id, tile_id, seed)
    if trace_steps:
        return {"source": "trace", "steps": trace_steps}
    return {
        "source": "summary_selected_block_ids",
        "steps": [
            {"step": index, "selected_block_id": block_id}
            for index, block_id in enumerate(fallback_blocks, start=1)
            if str(block_id).strip()
        ],
    }


def _trace_steps(
    traces: object,
    variant_id: str,
    tile_id: str,
    seed: int,
) -> list[dict[str, object]]:
    if not isinstance(traces, Mapping):
        return []
    trained = traces.get("trained_policy")
    if not isinstance(trained, Mapping):
        return []
    variant = trained.get(variant_id)
    if not isinstance(variant, Mapping):
        return []
    tile = variant.get(tile_id)
    if not isinstance(tile, Mapping):
        return []
    steps = tile.get(str(seed))
    if not isinstance(steps, list):
        return []
    result = []
    for fallback_index, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping):
            continue
        normalized = dict(step)
        normalized.setdefault("step", fallback_index)
        result.append(normalized)
    return result


def _selected_block_ids(row: Mapping[str, object]) -> list[str]:
    value = row.get("selected_block_ids", "")
    if isinstance(value, list):
        parts = [str(item) for item in value]
    else:
        parts = str(value).split(";")
    return [part.strip() for part in parts if part.strip()]


def _step_sequence(step_payload: Mapping[str, object]) -> list[dict[str, object]]:
    value = step_payload.get("steps")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Phase 35 step payload is missing steps")
    return [dict(step) for step in value if isinstance(step, Mapping)]


def _block_positions(steps: Sequence[Mapping[str, object]]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for index, step in enumerate(steps, start=1):
        block_id = str(step.get("selected_block_id", "")).strip()
        if block_id and block_id not in positions:
            positions[block_id] = index
    return positions


def _trace_reward_sum(steps: Sequence[Mapping[str, object]]) -> float | None:
    if not _has_any_reward(steps):
        return None
    return _round_float(sum(_optional_float(step, "reward") or 0.0 for step in steps))


def _has_any_reward(steps: Sequence[Mapping[str, object]]) -> bool:
    return any(_optional_float(step, "reward") is not None for step in steps)


def _first_step_reward_gap(
    variant_steps: Sequence[Mapping[str, object]],
    comparator_steps: Sequence[Mapping[str, object]],
) -> float | str:
    if not variant_steps or not comparator_steps:
        return ""
    variant_reward = _optional_float(variant_steps[0], "reward")
    comparator_reward = _optional_float(comparator_steps[0], "reward")
    if variant_reward is None or comparator_reward is None:
        return ""
    return _round_float(variant_reward - comparator_reward)


def _case_role(higher_delta: float) -> str:
    if higher_delta > 0.0:
        return "phase33_positive_case"
    if higher_delta < 0.0:
        return "phase33_failure_case"
    return "phase33_neutral_case"


def _action_overlap_pattern(
    shared: set[str],
    union: set[str],
    summary_gap: float,
) -> str:
    if not union:
        return "action_overlap_incomplete"
    overlap_fraction = len(shared) / len(union)
    if overlap_fraction >= 0.999 and summary_gap > 0.0:
        return "same_blocks_positive_gap"
    if overlap_fraction >= 0.999 and summary_gap < 0.0:
        return "same_blocks_negative_gap"
    if overlap_fraction > 0.0 and summary_gap > 0.0:
        return "partial_overlap_positive_gap"
    if overlap_fraction > 0.0 and summary_gap < 0.0:
        return "partial_overlap_negative_gap"
    if summary_gap > 0.0:
        return "disjoint_positive_gap"
    if summary_gap < 0.0:
        return "disjoint_negative_gap"
    return "descriptive"


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    label: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 35 analysis is missing {label}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 35 {label} contains a non-mapping row")
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _phase35_interpretation(status: str) -> str:
    if status == "action_overlap_diagnostics_ready":
        return (
            "Phase 35 compares Phase 33 high-budget selected-block overlap and "
            "action order for normalized-B1 variants and matched comparators. "
            "It is diagnostic evidence only."
        )
    return "Phase 35 could not assemble complete action-overlap diagnostics."


def _phase35_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Phase 35 Phase 33 Action-Overlap Diagnostics",
        "",
        f"Status: {analysis.get('phase35_action_overlap_status', '')}",
        "",
        "## Case Summary",
        "",
        "| Case | Stability | Jaccard | Summary gap | Pattern |",
        "|---|---|---:|---:|---|",
    ]
    for row in analysis.get("case_rows", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| `{case}` | {stability} | `{jaccard}` | `{gap}` | {pattern} |".format(
                case=row.get("case_id", ""),
                stability=row.get("stability_class", ""),
                jaccard=row.get("selected_block_jaccard", ""),
                gap=row.get("summary_reward_gap", ""),
                pattern=row.get("action_overlap_pattern", ""),
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


def _normalize_csvish_values(values: Sequence[str] | str) -> list[str]:
    if isinstance(values, str):
        raw = values.split(",")
    else:
        raw = []
        for value in values:
            raw.extend(str(value).split(","))
    return [str(value).strip() for value in raw if str(value).strip()]


def _has_value(row: Mapping[str, object], field: str) -> bool:
    return field in row and str(row.get(field, "")).strip() != ""


def _optional_float(row: Mapping[str, object], field: str) -> float | None:
    if not _has_value(row, field):
        return None
    return _float_value(row, field)


def _float_value(row: Mapping[str, object], field: str) -> float:
    try:
        return float(row[field])
    except KeyError as exc:
        raise ValueError(f"Missing numeric field {field}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric field {field}: {row.get(field)!r}") from exc


def _int_value(row: Mapping[str, object], field: str) -> int:
    try:
        return int(float(row[field]))
    except KeyError as exc:
        raise ValueError(f"Missing integer field {field}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer field {field}: {row.get(field)!r}") from exc


def _mean(values: Sequence[float | int]) -> float | str:
    clean = [float(value) for value in values]
    if not clean:
        return ""
    return _round_float(sum(clean) / len(clean))


def _csv_optional_float(value: float | None) -> float | str:
    if value is None:
        return ""
    return _round_float(value)


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


def _round_float(value: float | int) -> float:
    return round(float(value), 10)
