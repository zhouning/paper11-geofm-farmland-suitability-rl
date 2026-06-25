from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path

from .planning_reward import compute_base_planning_reward


PHASE34_CASE_MAP_CLAIM_BOUNDARY = (
    "Phase 34 is a read-only case-map diagnostic over existing Phase 33 "
    "matched pilot artifacts; it does not run new policy training, does not "
    "alter rewards, does not enable suitability reward, does not test B2/B3, "
    "and does not support final submission-level planning-performance claims."
)

PHASE34_CASE_FIELDNAMES = [
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
    "variant_reward",
    "comparator_reward",
    "variant_minus_comparator_reward",
    "selected_block_jaccard",
    "shared_selected_block_count",
    "variant_selected_block_count",
    "comparator_selected_block_count",
    "variant_mean_base_planning_reward",
    "comparator_mean_base_planning_reward",
    "variant_mean_suitability_proxy",
    "comparator_mean_suitability_proxy",
    "variant_mean_low_slope_farmland_label",
    "comparator_mean_low_slope_farmland_label",
    "variant_row_min",
    "variant_row_max",
    "variant_col_min",
    "variant_col_max",
    "comparator_row_min",
    "comparator_row_max",
    "comparator_col_min",
    "comparator_col_max",
    "spatial_pattern",
    "source_phase33_output_dir",
    "claim_boundary",
]

PHASE34_BLOCK_FIELDNAMES = [
    "case_id",
    "eval_tile_id",
    "seed",
    "variant_id",
    "comparator_variant_id",
    "selection_role",
    "block_id",
    "variant_step",
    "comparator_step",
    "row_min",
    "row_max",
    "col_min",
    "col_max",
    "row_center",
    "col_center",
    "base_planning_reward",
    "suitability_proxy",
    "current_farmland_label",
    "low_slope_farmland_label",
    "slope_mean",
    "slope_max",
    "area_m2",
    "claim_boundary",
]


def build_phase34_case_map_diagnostics(
    phase33_output_dirs: Sequence[Path | str],
    phase2_features_csv: Path | str,
    tile_index_csv: Path | str,
    *,
    variants: Sequence[str] | str = ("N1Z", "N1ZR"),
    comparators: Sequence[str] | str = ("B1", "D4P8", "D4P16"),
) -> dict[str, object]:
    output_dirs = [Path(path) for path in phase33_output_dirs]
    if not output_dirs:
        raise ValueError("Phase 34 requires at least one Phase 33 output directory")

    variant_filter = set(_normalize_csvish_values(variants))
    comparator_filter = set(_normalize_csvish_values(comparators))
    tile_index = _read_table_by_id(
        Path(tile_index_csv),
        "tile_id",
        "Phase 13 tile index CSV",
    )

    pending_cases: list[dict[str, object]] = []
    selected_block_ids: set[str] = set()
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
            variant_blocks = _selected_block_ids(variant_summary)
            comparator_blocks = _selected_block_ids(comparator_summary)
            if not variant_blocks or not comparator_blocks:
                continue
            selected_block_ids.update(variant_blocks)
            selected_block_ids.update(comparator_blocks)
            pending_cases.append(
                {
                    "source_phase33_output_dir": str(output_dir),
                    "stability": dict(stability),
                    "variant_summary": dict(variant_summary),
                    "comparator_summary": dict(comparator_summary),
                    "variant_blocks": variant_blocks,
                    "comparator_blocks": comparator_blocks,
                    "variant_steps": _block_steps(
                        traces,
                        variant_id,
                        tile_id,
                        seed,
                        fallback_blocks=variant_blocks,
                    ),
                    "comparator_steps": _block_steps(
                        traces,
                        comparator_id,
                        tile_id,
                        seed,
                        fallback_blocks=comparator_blocks,
                    ),
                }
            )

    features = _read_feature_subset(Path(phase2_features_csv), selected_block_ids)
    case_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    for case in pending_cases:
        case_row = _case_row(case, features, tile_index)
        case_rows.append(case_row)
        block_rows.extend(_case_block_rows(case, features, case_row))

    status = (
        "case_map_diagnostics_ready"
        if case_rows and block_rows
        else "case_map_diagnostics_insufficient"
    )
    return {
        "phase": "phase34_case_map_diagnostics",
        "phase34_case_map_status": status,
        "source_paths": {
            "phase33_output_dirs": [str(path) for path in output_dirs],
            "phase2_features_csv": str(Path(phase2_features_csv)),
            "tile_index_csv": str(Path(tile_index_csv)),
        },
        "variants": sorted(variant_filter),
        "comparators": sorted(comparator_filter),
        "row_counts": {
            "phase33_output_dirs": len(output_dirs),
            "case_rows": len(case_rows),
            "case_map_block_rows": len(block_rows),
            "selected_feature_rows": len(features),
        },
        "case_rows": case_rows,
        "case_map_block_rows": block_rows,
        "interpretation": _phase34_interpretation(status),
        "claim_boundary": PHASE34_CASE_MAP_CLAIM_BOUNDARY,
    }


def write_phase34_case_map_diagnostics_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    case_summary_path = output_path / "phase34_case_map_cases.csv"
    case_map_blocks_path = output_path / "phase34_case_map_blocks.csv"
    diagnosis_json_path = output_path / "phase34_case_map_diagnostics.json"
    diagnosis_md_path = output_path / "phase34_case_map_diagnostics.md"

    _write_csv_mapping_rows(
        case_summary_path,
        PHASE34_CASE_FIELDNAMES,
        analysis.get("case_rows"),
        "case_rows",
    )
    _write_csv_mapping_rows(
        case_map_blocks_path,
        PHASE34_BLOCK_FIELDNAMES,
        analysis.get("case_map_block_rows"),
        "case_map_block_rows",
    )
    diagnosis_json_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    diagnosis_md_path.write_text(_phase34_markdown(analysis), encoding="utf-8")
    return {
        "case_summary_csv": case_summary_path,
        "case_map_blocks_csv": case_map_blocks_path,
        "diagnosis_json": diagnosis_json_path,
        "diagnosis_md": diagnosis_md_path,
    }


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
                "Phase 34 requires unique trained_policy rows for "
                f"{variant_id} {tile_id} seed {key[2]}"
            )
        indexed[key] = row
    return indexed


def _read_feature_subset(
    path: Path,
    required_block_ids: set[str],
) -> dict[str, dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 2 feature CSV: {path}")
    if not required_block_ids:
        return {}
    table: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            block_id = str(row.get("block_id", "")).strip()
            if block_id in required_block_ids:
                table[block_id] = dict(row)
    missing = sorted(required_block_ids - set(table))
    if missing:
        raise ValueError(
            "Phase 34 selected blocks are missing from Phase 2 features: "
            f"{', '.join(missing[:5])}"
        )
    return table


def _case_row(
    case: Mapping[str, object],
    features: Mapping[str, Mapping[str, object]],
    tile_index: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    stability = _mapping_value(case, "stability")
    variant_summary = _mapping_value(case, "variant_summary")
    comparator_summary = _mapping_value(case, "comparator_summary")
    variant_blocks = _sequence_value(case, "variant_blocks")
    comparator_blocks = _sequence_value(case, "comparator_blocks")

    tile_id = str(stability.get("eval_tile_id", "")).strip()
    if tile_id not in tile_index:
        raise ValueError(f"Tile {tile_id} is missing from Phase 13 tile index")
    seed = _int_value(stability, "seed")
    variant_id = str(stability.get("variant_id", "")).strip()
    comparator_id = str(stability.get("comparator_variant_id", "")).strip()
    case_id = f"{tile_id}|{seed}|{variant_id}|{comparator_id}"
    shared = set(variant_blocks) & set(comparator_blocks)
    union = set(variant_blocks) | set(comparator_blocks)
    variant_reward = _float_value(variant_summary, "total_contract_reward")
    comparator_reward = _float_value(comparator_summary, "total_contract_reward")
    variant_metrics = _block_metrics(variant_blocks, features)
    comparator_metrics = _block_metrics(comparator_blocks, features)
    return {
        "case_id": case_id,
        "case_role": _case_role(_float_value(stability, "higher_delta")),
        "eval_tile_id": tile_id,
        "seed": seed,
        "variant_id": variant_id,
        "comparator_variant_id": comparator_id,
        "stability_class": str(stability.get("stability_class", "")),
        "lower_delta": _float_value(stability, "lower_delta"),
        "higher_delta": _float_value(stability, "higher_delta"),
        "delta_change": _optional_float(stability, "delta_change"),
        "variant_reward": _round_float(variant_reward),
        "comparator_reward": _round_float(comparator_reward),
        "variant_minus_comparator_reward": _round_float(
            variant_reward - comparator_reward
        ),
        "selected_block_jaccard": _round_float(len(shared) / len(union))
        if union
        else 1.0,
        "shared_selected_block_count": len(shared),
        "variant_selected_block_count": len(variant_blocks),
        "comparator_selected_block_count": len(comparator_blocks),
        "variant_mean_base_planning_reward": variant_metrics["mean_base_planning_reward"],
        "comparator_mean_base_planning_reward": comparator_metrics[
            "mean_base_planning_reward"
        ],
        "variant_mean_suitability_proxy": variant_metrics["mean_suitability_proxy"],
        "comparator_mean_suitability_proxy": comparator_metrics[
            "mean_suitability_proxy"
        ],
        "variant_mean_low_slope_farmland_label": variant_metrics[
            "mean_low_slope_farmland_label"
        ],
        "comparator_mean_low_slope_farmland_label": comparator_metrics[
            "mean_low_slope_farmland_label"
        ],
        "variant_row_min": variant_metrics["row_min"],
        "variant_row_max": variant_metrics["row_max"],
        "variant_col_min": variant_metrics["col_min"],
        "variant_col_max": variant_metrics["col_max"],
        "comparator_row_min": comparator_metrics["row_min"],
        "comparator_row_max": comparator_metrics["row_max"],
        "comparator_col_min": comparator_metrics["col_min"],
        "comparator_col_max": comparator_metrics["col_max"],
        "spatial_pattern": _spatial_pattern(
            variant_metrics["mean_base_planning_reward"],
            comparator_metrics["mean_base_planning_reward"],
        ),
        "source_phase33_output_dir": str(case.get("source_phase33_output_dir", "")),
        "claim_boundary": PHASE34_CASE_MAP_CLAIM_BOUNDARY,
    }


def _case_block_rows(
    case: Mapping[str, object],
    features: Mapping[str, Mapping[str, object]],
    case_row: Mapping[str, object],
) -> list[dict[str, object]]:
    variant_blocks = _sequence_value(case, "variant_blocks")
    comparator_blocks = _sequence_value(case, "comparator_blocks")
    variant_steps = _mapping_value(case, "variant_steps")
    comparator_steps = _mapping_value(case, "comparator_steps")
    variant_set = set(variant_blocks)
    comparator_set = set(comparator_blocks)
    ordered_blocks = _ordered_union(variant_blocks, comparator_blocks)
    rows: list[dict[str, object]] = []
    for block_id in ordered_blocks:
        row = _feature_row(features, block_id)
        role = _selection_role(block_id, variant_set, comparator_set)
        rows.append(
            {
                "case_id": case_row["case_id"],
                "eval_tile_id": case_row["eval_tile_id"],
                "seed": case_row["seed"],
                "variant_id": case_row["variant_id"],
                "comparator_variant_id": case_row["comparator_variant_id"],
                "selection_role": role,
                "block_id": block_id,
                "variant_step": variant_steps.get(block_id),
                "comparator_step": comparator_steps.get(block_id),
                "row_min": _int_value(row, "row_min"),
                "row_max": _int_value(row, "row_max"),
                "col_min": _int_value(row, "col_min"),
                "col_max": _int_value(row, "col_max"),
                "row_center": _round_float(
                    (_float_value(row, "row_min") + _float_value(row, "row_max")) / 2.0
                ),
                "col_center": _round_float(
                    (_float_value(row, "col_min") + _float_value(row, "col_max")) / 2.0
                ),
                "base_planning_reward": compute_base_planning_reward(row),
                "suitability_proxy": _optional_float(row, "suitability_proxy"),
                "current_farmland_label": _optional_float(
                    row,
                    "current_farmland_label",
                ),
                "low_slope_farmland_label": _optional_float(
                    row,
                    "low_slope_farmland_label",
                ),
                "slope_mean": _optional_float(row, "slope_mean"),
                "slope_max": _optional_float(row, "slope_max"),
                "area_m2": _optional_float(row, "area_m2"),
                "claim_boundary": PHASE34_CASE_MAP_CLAIM_BOUNDARY,
            }
        )
    return rows


def _block_metrics(
    block_ids: Sequence[object],
    features: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    rows = [_feature_row(features, str(block_id)) for block_id in block_ids]
    return {
        "mean_base_planning_reward": _mean(
            [compute_base_planning_reward(row) for row in rows]
        ),
        "mean_suitability_proxy": _mean(
            [
                _float_value(row, "suitability_proxy")
                for row in rows
                if _has_value(row, "suitability_proxy")
            ]
        ),
        "mean_low_slope_farmland_label": _mean(
            [
                _float_value(row, "low_slope_farmland_label")
                for row in rows
                if _has_value(row, "low_slope_farmland_label")
            ]
        ),
        "row_min": min(_int_value(row, "row_min") for row in rows),
        "row_max": max(_int_value(row, "row_max") for row in rows),
        "col_min": min(_int_value(row, "col_min") for row in rows),
        "col_max": max(_int_value(row, "col_max") for row in rows),
    }


def _selected_block_ids(row: Mapping[str, object]) -> list[str]:
    value = row.get("selected_block_ids", "")
    if isinstance(value, list):
        parts = [str(item) for item in value]
    else:
        parts = str(value).split(";")
    return [part.strip() for part in parts if part.strip()]


def _block_steps(
    traces: object,
    variant_id: str,
    tile_id: str,
    seed: int,
    *,
    fallback_blocks: Sequence[str],
) -> dict[str, int]:
    steps = _trace_steps(traces, variant_id, tile_id, seed)
    if steps:
        result: dict[str, int] = {}
        for position, step in enumerate(steps, start=1):
            block_id = str(step.get("selected_block_id", "")).strip()
            if block_id and block_id not in result:
                result[block_id] = position
        return result
    return {
        str(block_id): index
        for index, block_id in enumerate(fallback_blocks, start=1)
        if str(block_id).strip()
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
    return [dict(step) for step in steps if isinstance(step, Mapping)]


def _case_role(higher_delta: float) -> str:
    if higher_delta > 0.0:
        return "phase33_positive_case"
    if higher_delta < 0.0:
        return "phase33_failure_case"
    return "phase33_neutral_case"


def _spatial_pattern(
    variant_mean_base_reward: object,
    comparator_mean_base_reward: object,
) -> str:
    if variant_mean_base_reward is None or comparator_mean_base_reward is None:
        return "spatial_composition_incomplete"
    variant_value = float(variant_mean_base_reward)
    comparator_value = float(comparator_mean_base_reward)
    if variant_value > comparator_value + 1e-9:
        return "variant_selects_higher_base_reward_blocks"
    if variant_value < comparator_value - 1e-9:
        return "variant_selects_lower_base_reward_blocks"
    return "variant_matches_comparator_base_reward"


def _selection_role(
    block_id: str,
    variant_blocks: set[str],
    comparator_blocks: set[str],
) -> str:
    in_variant = block_id in variant_blocks
    in_comparator = block_id in comparator_blocks
    if in_variant and in_comparator:
        return "shared"
    if in_variant:
        return "variant_only"
    return "comparator_only"


def _ordered_union(
    variant_blocks: Sequence[object],
    comparator_blocks: Sequence[object],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for block_id in [*variant_blocks, *comparator_blocks]:
        normalized = str(block_id).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _feature_row(
    features: Mapping[str, Mapping[str, object]],
    block_id: str,
) -> Mapping[str, object]:
    row = features.get(block_id)
    if not isinstance(row, Mapping):
        raise ValueError(f"Block {block_id} is missing from Phase 2 features")
    return row


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    label: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 34 analysis is missing {label}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 34 {label} contains a non-mapping row")
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _phase34_interpretation(status: str) -> str:
    if status == "case_map_diagnostics_ready":
        return (
            "Phase 34 joins Phase 33 matched cases to selected-block spatial "
            "extents and base-reward composition for case-map inspection. It "
            "is diagnostic evidence only."
        )
    return "Phase 34 could not assemble complete case-map diagnostics."


def _phase34_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Phase 34 Case-Map Diagnostics",
        "",
        f"Status: {analysis.get('phase34_case_map_status', '')}",
        "",
        "## Case Summary",
        "",
        "| Case | Stability | Higher delta | Jaccard | Spatial pattern |",
        "|---|---|---:|---:|---|",
    ]
    for row in analysis.get("case_rows", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| `{case}` | {stability} | `{delta}` | `{jaccard}` | {pattern} |".format(
                case=row.get("case_id", ""),
                stability=row.get("stability_class", ""),
                delta=row.get("higher_delta", ""),
                jaccard=row.get("selected_block_jaccard", ""),
                pattern=row.get("spatial_pattern", ""),
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


def _mapping_value(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Phase 34 case is missing {key}")
    return value


def _sequence_value(mapping: Mapping[str, object], key: str) -> list[str]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Phase 34 case is missing {key}")
    return [str(item) for item in value]


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


def _mean(values: Sequence[float | int]) -> float | None:
    clean = [float(value) for value in values]
    if not clean:
        return None
    return _round_float(sum(clean) / len(clean))


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
