from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path

from .planning_reward import BASE_PLANNING_REWARD_REQUIRED_COLUMNS


PHASE32_ACTION_ORDER_CLAIM_BOUNDARY = (
    "Phase 32 is a read-only action-order diagnostic over existing Phase 28, "
    "Phase 30, and Phase 31 artifacts; it does not run new policy training, "
    "does not alter rewards, does not enable suitability reward, does not test "
    "B2/B3, and does not support final submission-level planning-performance "
    "claims."
)

PHASE32_STEP_ALIGNMENT_FIELDNAMES = [
    "case_id",
    "step",
    "focal_variant",
    "comparator_variant",
    "focal_block_id",
    "comparator_block_id",
    "same_step_block_match",
    "focal_reward",
    "comparator_reward",
    "step_reward_gap",
    "focal_cumulative_reward",
    "comparator_cumulative_reward",
    "cumulative_reward_gap",
    "claim_boundary",
]

PHASE32_CASE_SUMMARY_FIELDNAMES = [
    "case_id",
    "case_rank",
    "case_role",
    "eval_tile_id",
    "seed",
    "focal_variant",
    "comparator_variant",
    "focal_step_count",
    "comparator_step_count",
    "shared_block_count",
    "union_block_count",
    "selected_block_jaccard",
    "mean_abs_shared_step_displacement",
    "max_abs_shared_step_displacement",
    "same_step_match_count",
    "focal_cumulative_reward",
    "comparator_cumulative_reward",
    "cumulative_reward_gap",
    "first_step_reward_gap",
    "diagnostic_pattern",
    "claim_boundary",
]

PHASE32_TILE_POOL_FIELDNAMES = [
    "case_id",
    "eval_tile_id",
    "tile_block_count",
    "focal_selected_count",
    "comparator_selected_count",
    "tile_low_slope_farmland_mean",
    "focal_low_slope_farmland_mean",
    "comparator_low_slope_farmland_mean",
    "tile_base_reward_mean",
    "focal_base_reward_mean",
    "comparator_base_reward_mean",
    "tile_suitability_mean",
    "focal_suitability_mean",
    "comparator_suitability_mean",
    "claim_boundary",
]


def build_phase32_action_order_diagnostics(
    ranked_cases_csv: Path | str,
    focal_traces_json: Path | str,
    comparator_traces_json: Path | str,
    phase2_features_csv: Path | str,
    tile_index_csv: Path | str,
    *,
    top_k: int = 6,
) -> dict[str, object]:
    ranked_cases = _read_csv_rows(Path(ranked_cases_csv), "Phase 31 ranked cases CSV")
    if not ranked_cases:
        raise ValueError("Phase 32 requires ranked case rows")
    selected_cases = ranked_cases[: max(int(top_k), 0)]
    focal_traces = _read_json(Path(focal_traces_json), "focal traces JSON")
    comparator_traces = _read_json(Path(comparator_traces_json), "comparator traces JSON")
    features = _read_feature_table(Path(phase2_features_csv))
    tile_index = _read_table_by_id(Path(tile_index_csv), "tile_id", "Phase 13 tile index CSV")

    step_alignment_rows: list[dict[str, object]] = []
    case_summary_rows: list[dict[str, object]] = []
    tile_pool_rows: list[dict[str, object]] = []
    for case in selected_cases:
        focal_variant = str(case.get("variant_id", "")).strip()
        comparator_variant = str(case.get("comparator_variant_id", "")).strip()
        tile_id = str(case.get("eval_tile_id", "")).strip()
        seed = _int_value(case, "seed")
        focal_steps = _trace_steps(focal_traces, focal_variant, tile_id, seed)
        comparator_steps = _trace_steps(
            comparator_traces,
            comparator_variant,
            tile_id,
            seed,
        )
        if not focal_steps or not comparator_steps:
            continue
        step_alignment_rows.extend(
            _step_alignment_rows(
                case,
                focal_steps,
                comparator_steps,
            )
        )
        case_summary_rows.append(
            _case_summary_row(
                case,
                focal_steps,
                comparator_steps,
            )
        )
        tile_pool_rows.append(
            _tile_pool_composition_row(
                case,
                focal_steps,
                comparator_steps,
                features,
                tile_index,
            )
        )

    status = (
        "action_order_diagnostics_ready"
        if step_alignment_rows and case_summary_rows and tile_pool_rows
        else "action_order_diagnostics_insufficient"
    )
    return {
        "phase": "phase32_action_order_diagnostics",
        "phase32_action_order_status": status,
        "source_paths": {
            "ranked_cases_csv": str(Path(ranked_cases_csv)),
            "focal_traces_json": str(Path(focal_traces_json)),
            "comparator_traces_json": str(Path(comparator_traces_json)),
            "phase2_features_csv": str(Path(phase2_features_csv)),
            "tile_index_csv": str(Path(tile_index_csv)),
        },
        "top_k": int(top_k),
        "row_counts": {
            "ranked_case_rows": len(ranked_cases),
            "selected_case_rows": len(selected_cases),
            "step_alignment_rows": len(step_alignment_rows),
            "case_summary_rows": len(case_summary_rows),
            "tile_pool_composition_rows": len(tile_pool_rows),
        },
        "step_alignment_rows": step_alignment_rows,
        "case_summary_rows": case_summary_rows,
        "tile_pool_composition_rows": tile_pool_rows,
        "interpretation": _phase32_interpretation(status),
        "claim_boundary": PHASE32_ACTION_ORDER_CLAIM_BOUNDARY,
    }


def write_phase32_action_order_diagnostics_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    step_alignment_path = output_path / "phase32_step_alignment.csv"
    case_summary_path = output_path / "phase32_case_summary.csv"
    tile_pool_path = output_path / "phase32_tile_pool_composition.csv"
    diagnosis_json_path = output_path / "phase32_action_order_diagnostics.json"
    diagnosis_md_path = output_path / "phase32_action_order_diagnostics.md"

    _write_csv_mapping_rows(
        step_alignment_path,
        PHASE32_STEP_ALIGNMENT_FIELDNAMES,
        analysis.get("step_alignment_rows"),
        "step_alignment_rows",
    )
    _write_csv_mapping_rows(
        case_summary_path,
        PHASE32_CASE_SUMMARY_FIELDNAMES,
        analysis.get("case_summary_rows"),
        "case_summary_rows",
    )
    _write_csv_mapping_rows(
        tile_pool_path,
        PHASE32_TILE_POOL_FIELDNAMES,
        analysis.get("tile_pool_composition_rows"),
        "tile_pool_composition_rows",
    )
    diagnosis_json_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    diagnosis_md_path.write_text(_phase32_markdown(analysis), encoding="utf-8")
    return {
        "step_alignment_csv": step_alignment_path,
        "case_summary_csv": case_summary_path,
        "tile_pool_csv": tile_pool_path,
        "diagnosis_json": diagnosis_json_path,
        "diagnosis_md": diagnosis_md_path,
    }


def _read_csv_rows(path: Path, label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path, label: str) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_feature_table(path: Path) -> dict[str, dict[str, object]]:
    rows = _read_csv_rows(path, "Phase 2 feature CSV")
    table: dict[str, dict[str, object]] = {}
    for row_number, row in enumerate(rows, start=2):
        block_id = str(row.get("block_id", "")).strip()
        if not block_id:
            raise ValueError(f"Feature table is missing block_id at {path}:{row_number}")
        if block_id in table:
            raise ValueError(f"Duplicate block_id in feature table {path}: {block_id}")
        table[block_id] = dict(row)
    if not table:
        raise ValueError(f"Feature table has no rows: {path}")
    return table


def _read_table_by_id(path: Path, key_field: str, label: str) -> dict[str, dict[str, object]]:
    rows = _read_csv_rows(path, label)
    table: dict[str, dict[str, object]] = {}
    for row in rows:
        key = str(row.get(key_field, "")).strip()
        if key:
            table[key] = dict(row)
    if not table:
        raise ValueError(f"{label} has no keyed rows")
    return table


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
    return [dict(step) for step in steps if isinstance(step, Mapping)]


def _step_alignment_rows(
    case: Mapping[str, object],
    focal_steps: Sequence[Mapping[str, object]],
    comparator_steps: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    focal_cumulative = 0.0
    comparator_cumulative = 0.0
    max_steps = max(len(focal_steps), len(comparator_steps))
    for index in range(max_steps):
        focal_step = focal_steps[index] if index < len(focal_steps) else {}
        comparator_step = comparator_steps[index] if index < len(comparator_steps) else {}
        focal_reward = _optional_float(focal_step, "reward")
        comparator_reward = _optional_float(comparator_step, "reward")
        focal_cumulative += focal_reward or 0.0
        comparator_cumulative += comparator_reward or 0.0
        focal_block = str(focal_step.get("selected_block_id", "")).strip()
        comparator_block = str(comparator_step.get("selected_block_id", "")).strip()
        rows.append(
            {
                "case_id": str(case.get("case_id", "")),
                "step": index + 1,
                "focal_variant": str(case.get("variant_id", "")),
                "comparator_variant": str(case.get("comparator_variant_id", "")),
                "focal_block_id": focal_block,
                "comparator_block_id": comparator_block,
                "same_step_block_match": focal_block == comparator_block and bool(focal_block),
                "focal_reward": _round_float(focal_reward) if focal_reward is not None else None,
                "comparator_reward": _round_float(comparator_reward)
                if comparator_reward is not None
                else None,
                "step_reward_gap": _round_float(
                    (focal_reward or 0.0) - (comparator_reward or 0.0)
                ),
                "focal_cumulative_reward": _round_float(focal_cumulative),
                "comparator_cumulative_reward": _round_float(comparator_cumulative),
                "cumulative_reward_gap": _round_float(
                    focal_cumulative - comparator_cumulative
                ),
                "claim_boundary": PHASE32_ACTION_ORDER_CLAIM_BOUNDARY,
            }
        )
    return rows


def _case_summary_row(
    case: Mapping[str, object],
    focal_steps: Sequence[Mapping[str, object]],
    comparator_steps: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    focal_positions = _block_positions(focal_steps)
    comparator_positions = _block_positions(comparator_steps)
    focal_blocks = set(focal_positions)
    comparator_blocks = set(comparator_positions)
    shared = focal_blocks & comparator_blocks
    union = focal_blocks | comparator_blocks
    displacements = [
        abs(focal_positions[block_id] - comparator_positions[block_id])
        for block_id in shared
    ]
    focal_total = sum(_optional_float(step, "reward") or 0.0 for step in focal_steps)
    comparator_total = sum(
        _optional_float(step, "reward") or 0.0 for step in comparator_steps
    )
    same_step_matches = sum(
        1
        for focal_step, comparator_step in zip(focal_steps, comparator_steps)
        if str(focal_step.get("selected_block_id", "")).strip()
        == str(comparator_step.get("selected_block_id", "")).strip()
    )
    first_gap = (
        (_optional_float(focal_steps[0], "reward") or 0.0)
        - (_optional_float(comparator_steps[0], "reward") or 0.0)
        if focal_steps and comparator_steps
        else None
    )
    return {
        "case_id": str(case.get("case_id", "")),
        "case_rank": _int_value(case, "case_rank"),
        "case_role": str(case.get("case_role", "")),
        "eval_tile_id": str(case.get("eval_tile_id", "")),
        "seed": _int_value(case, "seed"),
        "focal_variant": str(case.get("variant_id", "")),
        "comparator_variant": str(case.get("comparator_variant_id", "")),
        "focal_step_count": len(focal_steps),
        "comparator_step_count": len(comparator_steps),
        "shared_block_count": len(shared),
        "union_block_count": len(union),
        "selected_block_jaccard": _round_float(len(shared) / len(union)) if union else 1.0,
        "mean_abs_shared_step_displacement": _mean(displacements),
        "max_abs_shared_step_displacement": max(displacements) if displacements else None,
        "same_step_match_count": same_step_matches,
        "focal_cumulative_reward": _round_float(focal_total),
        "comparator_cumulative_reward": _round_float(comparator_total),
        "cumulative_reward_gap": _round_float(focal_total - comparator_total),
        "first_step_reward_gap": _round_float(first_gap) if first_gap is not None else None,
        "diagnostic_pattern": _diagnostic_pattern(
            shared,
            union,
            displacements,
            focal_total - comparator_total,
        ),
        "claim_boundary": PHASE32_ACTION_ORDER_CLAIM_BOUNDARY,
    }


def _tile_pool_composition_row(
    case: Mapping[str, object],
    focal_steps: Sequence[Mapping[str, object]],
    comparator_steps: Sequence[Mapping[str, object]],
    features: Mapping[str, Mapping[str, object]],
    tile_index: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    tile_id = str(case.get("eval_tile_id", "")).strip()
    tile_row = tile_index.get(tile_id)
    if not isinstance(tile_row, Mapping):
        raise ValueError(f"Tile {tile_id} is missing from tile index")
    tile_blocks = [
        part.strip()
        for part in str(tile_row.get("block_ids", "")).split(";")
        if part.strip()
    ]
    focal_blocks = _trace_block_ids(focal_steps)
    comparator_blocks = _trace_block_ids(comparator_steps)
    return {
        "case_id": str(case.get("case_id", "")),
        "eval_tile_id": tile_id,
        "tile_block_count": len(tile_blocks),
        "focal_selected_count": len(focal_blocks),
        "comparator_selected_count": len(comparator_blocks),
        "tile_low_slope_farmland_mean": _feature_mean(
            tile_blocks,
            features,
            "low_slope_farmland_label",
        ),
        "focal_low_slope_farmland_mean": _feature_mean(
            focal_blocks,
            features,
            "low_slope_farmland_label",
        ),
        "comparator_low_slope_farmland_mean": _feature_mean(
            comparator_blocks,
            features,
            "low_slope_farmland_label",
        ),
        "tile_base_reward_mean": _mean([_base_reward(features[block]) for block in tile_blocks]),
        "focal_base_reward_mean": _mean([_base_reward(features[block]) for block in focal_blocks]),
        "comparator_base_reward_mean": _mean(
            [_base_reward(features[block]) for block in comparator_blocks]
        ),
        "tile_suitability_mean": _feature_mean(tile_blocks, features, "suitability_proxy"),
        "focal_suitability_mean": _feature_mean(
            focal_blocks,
            features,
            "suitability_proxy",
        ),
        "comparator_suitability_mean": _feature_mean(
            comparator_blocks,
            features,
            "suitability_proxy",
        ),
        "claim_boundary": PHASE32_ACTION_ORDER_CLAIM_BOUNDARY,
    }


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    label: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 32 analysis is missing {label}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 32 {label} contains a non-mapping row")
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _phase32_interpretation(status: str) -> str:
    if status == "action_order_diagnostics_ready":
        return (
            "Phase 32 compares action order and local block composition for "
            "Phase 31 cases. It is diagnostic evidence only."
        )
    return "Phase 32 could not assemble complete action-order diagnostics."


def _phase32_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Phase 32 Action-Order Diagnostics",
        "",
        f"Status: {analysis.get('phase32_action_order_status', '')}",
        "",
        "## Case Summary",
        "",
        "| Case | Pattern | Cum. gap | First-step gap | Mean displacement |",
        "|---|---|---:|---:|---:|",
    ]
    for row in analysis.get("case_summary_rows", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| `{case}` | {pattern} | `{gap}` | `{first}` | `{disp}` |".format(
                case=row.get("case_id", ""),
                pattern=row.get("diagnostic_pattern", ""),
                gap=row.get("cumulative_reward_gap", ""),
                first=row.get("first_step_reward_gap", ""),
                disp=row.get("mean_abs_shared_step_displacement", ""),
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


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _block_positions(steps: Sequence[Mapping[str, object]]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for index, step in enumerate(steps, start=1):
        block_id = str(step.get("selected_block_id", "")).strip()
        if block_id and block_id not in positions:
            positions[block_id] = index
    return positions


def _trace_block_ids(steps: Sequence[Mapping[str, object]]) -> list[str]:
    return [
        str(step.get("selected_block_id", "")).strip()
        for step in steps
        if str(step.get("selected_block_id", "")).strip()
    ]


def _feature_mean(
    block_ids: Sequence[str],
    features: Mapping[str, Mapping[str, object]],
    field: str,
) -> float | None:
    values = []
    for block_id in block_ids:
        row = features.get(block_id)
        if not isinstance(row, Mapping):
            raise ValueError(f"Block {block_id} is missing from Phase 2 features")
        if field in row and str(row.get(field, "")).strip() != "":
            values.append(_float_value(row, field))
    return _mean(values)


def _base_reward(row: Mapping[str, object]) -> float:
    missing = [
        column
        for column in BASE_PLANNING_REWARD_REQUIRED_COLUMNS
        if column not in row or str(row[column]).strip() == ""
    ]
    if missing:
        raise ValueError(
            "Phase 32 requires explicit feature columns: "
            f"{', '.join(missing)}"
        )
    low_slope_farmland_or_orchard = 0.35 * _clip01(
        _float_value(row, "explicit_feature_16")
    )
    current_farmland_or_orchard = 0.20 * max(
        _clip01(_float_value(row, "explicit_feature_04")),
        _clip01(_float_value(row, "explicit_feature_07")),
    )
    low_slope = 0.10 * _clip01(_float_value(row, "explicit_feature_13"))
    area_score = 0.10 * _clip01(_float_value(row, "explicit_feature_00") / 5.0)
    mean_slope_penalty = -0.15 * _clip01(
        _float_value(row, "explicit_feature_01") / 25.0
    )
    max_slope_penalty = -0.05 * _clip01(
        _float_value(row, "explicit_feature_02") / 35.0
    )
    built_up_penalty = -0.10 * _clip01(_float_value(row, "explicit_feature_09"))
    water_penalty = -0.10 * _clip01(_float_value(row, "explicit_feature_10"))
    return (
        low_slope_farmland_or_orchard
        + current_farmland_or_orchard
        + low_slope
        + area_score
        + mean_slope_penalty
        + max_slope_penalty
        + built_up_penalty
        + water_penalty
    )


def _diagnostic_pattern(
    shared: set[str],
    union: set[str],
    displacements: Sequence[int],
    cumulative_gap: float,
) -> str:
    if union and len(shared) == len(union) and any(value > 0 for value in displacements):
        return "same_blocks_reordered"
    if union and len(shared) / len(union) >= 0.5 and cumulative_gap < 0.0:
        return "overlap_with_negative_gap"
    if union and len(shared) / len(union) < 0.5:
        return "different_blocks_selected"
    return "descriptive"


def _optional_float(row: Mapping[str, object], field: str) -> float | None:
    if field not in row or str(row.get(field, "")).strip() == "":
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


def _mean(values: Sequence[float | int]) -> float | None:
    clean = [float(value) for value in values]
    if not clean:
        return None
    return _round_float(sum(clean) / len(clean))


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _round_float(value: float | int) -> float:
    return round(float(value), 10)
