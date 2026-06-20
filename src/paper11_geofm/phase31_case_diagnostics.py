from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path

from .planning_reward import BASE_PLANNING_REWARD_REQUIRED_COLUMNS


PHASE31_CASE_DIAGNOSTICS_CLAIM_BOUNDARY = (
    "Phase 31 is a read-only case diagnostic over existing Phase 30 artifacts; "
    "it does not run new policy training, does not alter rewards, does not "
    "enable suitability reward, does not test B2/B3, and does not support "
    "final submission-level planning-performance claims."
)

PHASE31_RANKED_CASE_FIELDNAMES = [
    "case_id",
    "case_rank",
    "case_role",
    "eval_tile_id",
    "seed",
    "variant_id",
    "comparator_variant_id",
    "variant_reward",
    "comparator_reward",
    "variant_minus_comparator_reward",
    "abs_variant_minus_comparator_reward",
    "selected_block_jaccard",
    "shared_selected_block_count",
    "variant_selected_block_count",
    "comparator_selected_block_count",
    "trace_step_count",
    "claim_boundary",
]

PHASE31_SELECTED_BLOCK_FIELDNAMES = [
    "case_id",
    "variant_id",
    "selected_block_count",
    "mean_base_planning_reward",
    "mean_current_farmland_label",
    "mean_farmland_or_orchard_label",
    "mean_low_slope_farmland_label",
    "mean_slope_mean",
    "mean_slope_max",
    "mean_suitability_proxy",
    "mean_area_m2",
    "selected_block_ids",
    "claim_boundary",
]

PHASE31_TILE_GEOMETRY_FIELDNAMES = [
    "case_id",
    "eval_tile_id",
    "tile_row",
    "tile_col",
    "tile_n_blocks",
    "tile_min_grid_row",
    "tile_max_grid_row",
    "tile_min_grid_col",
    "tile_max_grid_col",
    "selected_mapping_min_row",
    "selected_mapping_max_row",
    "selected_mapping_min_col",
    "selected_mapping_max_col",
    "selected_mapping_count",
    "claim_boundary",
]


def build_phase31_case_diagnostics(
    summary_csv: Path | str,
    traces_json: Path | str,
    phase2_features_csv: Path | str,
    tile_index_csv: Path | str,
    block_mapping_csv: Path | str,
    *,
    focal_variant: str = "N1ZR",
    comparator_variant: str = "B1",
    top_k: int = 6,
) -> dict[str, object]:
    summary_rows = _read_csv_rows(Path(summary_csv), "Phase 30 summary CSV")
    trained_rows = [
        row for row in summary_rows if str(row.get("row_type", "")) == "trained_policy"
    ]
    if not trained_rows:
        raise ValueError("Phase 31 requires trained_policy rows")

    traces = _read_json(Path(traces_json), "Phase 30 traces JSON")
    features = _read_feature_table(Path(phase2_features_csv))
    tile_index = _read_table_by_id(Path(tile_index_csv), "tile_id", "Phase 13 tile index CSV")
    block_mapping = _read_block_mapping(Path(block_mapping_csv))

    indexed = _summary_index(trained_rows)
    candidate_rows = _candidate_case_rows(
        indexed,
        traces,
        focal_variant=str(focal_variant),
        comparator_variant=str(comparator_variant),
    )
    ranked_case_rows = _rank_cases(candidate_rows, int(top_k))
    selected_block_rows = _selected_block_summary_rows(
        ranked_case_rows,
        indexed,
        features,
    )
    tile_geometry_rows = _tile_geometry_rows(
        ranked_case_rows,
        indexed,
        tile_index,
        block_mapping,
    )
    status = (
        "case_diagnostics_ready"
        if ranked_case_rows and selected_block_rows and tile_geometry_rows
        else "case_diagnostics_insufficient"
    )

    return {
        "phase": "phase31_case_diagnostics",
        "phase31_case_diagnostic_status": status,
        "source_paths": {
            "summary_csv": str(Path(summary_csv)),
            "traces_json": str(Path(traces_json)),
            "phase2_features_csv": str(Path(phase2_features_csv)),
            "tile_index_csv": str(Path(tile_index_csv)),
            "block_mapping_csv": str(Path(block_mapping_csv)),
        },
        "focal_variant": str(focal_variant),
        "comparator_variant": str(comparator_variant),
        "top_k": int(top_k),
        "row_counts": {
            "summary_rows": len(summary_rows),
            "trained_policy_rows": len(trained_rows),
            "candidate_case_rows": len(candidate_rows),
            "ranked_case_rows": len(ranked_case_rows),
            "selected_block_summary_rows": len(selected_block_rows),
            "tile_geometry_rows": len(tile_geometry_rows),
        },
        "ranked_case_rows": ranked_case_rows,
        "selected_block_summary_rows": selected_block_rows,
        "tile_geometry_rows": tile_geometry_rows,
        "interpretation": _phase31_interpretation(status),
        "claim_boundary": PHASE31_CASE_DIAGNOSTICS_CLAIM_BOUNDARY,
    }


def write_phase31_case_diagnostics_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ranked_cases_path = output_path / "phase31_ranked_cases.csv"
    selected_blocks_path = output_path / "phase31_selected_blocks.csv"
    tile_geometry_path = output_path / "phase31_tile_geometry.csv"
    diagnosis_json_path = output_path / "phase31_case_diagnostics.json"
    diagnosis_md_path = output_path / "phase31_case_diagnostics.md"

    _write_csv_mapping_rows(
        ranked_cases_path,
        PHASE31_RANKED_CASE_FIELDNAMES,
        analysis.get("ranked_case_rows"),
        "ranked_case_rows",
    )
    _write_csv_mapping_rows(
        selected_blocks_path,
        PHASE31_SELECTED_BLOCK_FIELDNAMES,
        analysis.get("selected_block_summary_rows"),
        "selected_block_summary_rows",
    )
    _write_csv_mapping_rows(
        tile_geometry_path,
        PHASE31_TILE_GEOMETRY_FIELDNAMES,
        analysis.get("tile_geometry_rows"),
        "tile_geometry_rows",
    )
    diagnosis_json_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    diagnosis_md_path.write_text(_phase31_markdown(analysis), encoding="utf-8")
    return {
        "ranked_cases_csv": ranked_cases_path,
        "selected_blocks_csv": selected_blocks_path,
        "tile_geometry_csv": tile_geometry_path,
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


def _read_block_mapping(path: Path) -> dict[str, list[dict[str, object]]]:
    rows = _read_csv_rows(path, "Phase 11 block mapping CSV")
    mapping: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        block_id = str(row.get("block_id", "")).strip()
        if not block_id:
            continue
        mapping.setdefault(block_id, []).append(dict(row))
    if not mapping:
        raise ValueError("Phase 31 block mapping has no block rows")
    return mapping


def _summary_index(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int], dict[str, Mapping[str, object]]]:
    indexed: dict[tuple[str, int], dict[str, Mapping[str, object]]] = {}
    for row in rows:
        variant_id = str(row.get("variant_id", "")).strip()
        if not variant_id:
            continue
        key = (str(row.get("eval_tile_id", "")).strip(), _int_value(row, "seed"))
        variants = indexed.setdefault(key, {})
        if variant_id in variants:
            raise ValueError(
                "Phase 31 requires unique trained rows for "
                f"{key[0]} seed {key[1]} variant {variant_id}"
            )
        variants[variant_id] = row
    return indexed


def _candidate_case_rows(
    indexed: Mapping[tuple[str, int], Mapping[str, Mapping[str, object]]],
    traces: object,
    *,
    focal_variant: str,
    comparator_variant: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tile_id, seed in sorted(indexed, key=lambda item: (item[0], item[1])):
        variants = indexed[(tile_id, seed)]
        variant_row = variants.get(focal_variant)
        comparator_row = variants.get(comparator_variant)
        if variant_row is None or comparator_row is None:
            continue
        variant_blocks = _selected_block_ids(variant_row)
        comparator_blocks = _selected_block_ids(comparator_row)
        shared = set(variant_blocks) & set(comparator_blocks)
        union = set(variant_blocks) | set(comparator_blocks)
        variant_reward = _float_value(variant_row, "total_contract_reward")
        comparator_reward = _float_value(comparator_row, "total_contract_reward")
        delta = variant_reward - comparator_reward
        case_id = f"{tile_id}|{seed}|{focal_variant}|{comparator_variant}"
        rows.append(
            {
                "case_id": case_id,
                "case_role": _case_role(delta),
                "eval_tile_id": tile_id,
                "seed": seed,
                "variant_id": focal_variant,
                "comparator_variant_id": comparator_variant,
                "variant_reward": _round_float(variant_reward),
                "comparator_reward": _round_float(comparator_reward),
                "variant_minus_comparator_reward": _round_float(delta),
                "abs_variant_minus_comparator_reward": _round_float(abs(delta)),
                "selected_block_jaccard": _round_float(len(shared) / len(union))
                if union
                else 1.0,
                "shared_selected_block_count": len(shared),
                "variant_selected_block_count": len(variant_blocks),
                "comparator_selected_block_count": len(comparator_blocks),
                "trace_step_count": _trace_step_count(
                    traces,
                    focal_variant,
                    tile_id,
                    seed,
                ),
                "claim_boundary": PHASE31_CASE_DIAGNOSTICS_CLAIM_BOUNDARY,
            }
        )
    return rows


def _rank_cases(
    candidate_rows: Sequence[Mapping[str, object]],
    top_k: int,
) -> list[dict[str, object]]:
    requested = max(int(top_k), 0)
    positives = [
        dict(row)
        for row in candidate_rows
        if float(row.get("variant_minus_comparator_reward", 0.0)) > 0.0
    ]
    negatives = [
        dict(row)
        for row in candidate_rows
        if float(row.get("variant_minus_comparator_reward", 0.0)) < 0.0
    ]
    zeros = [
        dict(row)
        for row in candidate_rows
        if float(row.get("variant_minus_comparator_reward", 0.0)) == 0.0
    ]
    positives.sort(
        key=lambda row: (
            -float(row["variant_minus_comparator_reward"]),
            str(row["eval_tile_id"]),
            int(row["seed"]),
        )
    )
    negatives.sort(
        key=lambda row: (
            float(row["variant_minus_comparator_reward"]),
            str(row["eval_tile_id"]),
            int(row["seed"]),
        )
    )
    zeros.sort(key=lambda row: (str(row["eval_tile_id"]), int(row["seed"])))
    positive_limit = (requested + 1) // 2
    negative_limit = requested // 2
    ordered = positives[:positive_limit] + negatives[:negative_limit]
    if len(ordered) < requested:
        selected = {str(row["case_id"]) for row in ordered}
        fillers = [
            row
            for row in [*positives[positive_limit:], *negatives[negative_limit:], *zeros]
            if str(row["case_id"]) not in selected
        ]
        fillers.sort(
            key=lambda row: (
                -float(row["abs_variant_minus_comparator_reward"]),
                str(row["eval_tile_id"]),
                int(row["seed"]),
            )
        )
        ordered.extend(fillers[: requested - len(ordered)])
    ranked: list[dict[str, object]] = []
    for rank, row in enumerate(ordered, start=1):
        output = dict(row)
        output["case_rank"] = rank
        ranked.append(output)
    return ranked


def _selected_block_summary_rows(
    ranked_case_rows: Sequence[Mapping[str, object]],
    indexed: Mapping[tuple[str, int], Mapping[str, Mapping[str, object]]],
    features: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in ranked_case_rows:
        key = (str(case["eval_tile_id"]), int(case["seed"]))
        variants = indexed.get(key, {})
        for variant_id in (
            str(case["variant_id"]),
            str(case["comparator_variant_id"]),
        ):
            summary_row = variants.get(variant_id)
            if summary_row is None:
                continue
            block_ids = _selected_block_ids(summary_row)
            block_rows = []
            for block_id in block_ids:
                feature_row = features.get(block_id)
                if not isinstance(feature_row, Mapping):
                    raise ValueError(
                        f"Selected block {block_id} is missing from Phase 2 features"
                    )
                block_rows.append(feature_row)
            rows.append(
                {
                    "case_id": str(case["case_id"]),
                    "variant_id": variant_id,
                    "selected_block_count": len(block_ids),
                    "mean_base_planning_reward": _mean(
                        [_base_planning_reward(row) for row in block_rows]
                    ),
                    "mean_current_farmland_label": _mean(
                        [_float_value(row, "current_farmland_label") for row in block_rows]
                    ),
                    "mean_farmland_or_orchard_label": _mean(
                        [
                            _float_value(row, "farmland_or_orchard_label")
                            for row in block_rows
                        ]
                    ),
                    "mean_low_slope_farmland_label": _mean(
                        [
                            _float_value(row, "low_slope_farmland_label")
                            for row in block_rows
                        ]
                    ),
                    "mean_slope_mean": _mean(
                        [_float_value(row, "slope_mean") for row in block_rows]
                    ),
                    "mean_slope_max": _mean(
                        [_float_value(row, "slope_max") for row in block_rows]
                    ),
                    "mean_suitability_proxy": _mean(
                        [
                            _float_value(row, "suitability_proxy")
                            for row in block_rows
                            if "suitability_proxy" in row
                            and str(row.get("suitability_proxy", "")).strip() != ""
                        ]
                    ),
                    "mean_area_m2": _mean(
                        [
                            _float_value(row, "area_m2")
                            for row in block_rows
                            if "area_m2" in row and str(row.get("area_m2", "")).strip() != ""
                        ]
                    ),
                    "selected_block_ids": ";".join(block_ids),
                    "claim_boundary": PHASE31_CASE_DIAGNOSTICS_CLAIM_BOUNDARY,
                }
            )
    return rows


def _tile_geometry_rows(
    ranked_case_rows: Sequence[Mapping[str, object]],
    indexed: Mapping[tuple[str, int], Mapping[str, Mapping[str, object]]],
    tile_index: Mapping[str, Mapping[str, object]],
    block_mapping: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in ranked_case_rows:
        tile_id = str(case["eval_tile_id"])
        tile_row = tile_index.get(tile_id)
        if not isinstance(tile_row, Mapping):
            raise ValueError(f"Tile {tile_id} is missing from Phase 13 tile index")
        key = (tile_id, int(case["seed"]))
        variants = indexed.get(key, {})
        selected_ids = []
        row = variants.get(str(case["variant_id"]))
        if row is not None:
            selected_ids.extend(_selected_block_ids(row))
        mapping_rows = [
            mapping_row
            for block_id in selected_ids
            for mapping_row in block_mapping.get(block_id, [])
        ]
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "eval_tile_id": tile_id,
                "tile_row": _int_value(tile_row, "tile_row"),
                "tile_col": _int_value(tile_row, "tile_col"),
                "tile_n_blocks": _int_value(tile_row, "n_blocks"),
                "tile_min_grid_row": _int_value(tile_row, "min_grid_row"),
                "tile_max_grid_row": _int_value(tile_row, "max_grid_row"),
                "tile_min_grid_col": _int_value(tile_row, "min_grid_col"),
                "tile_max_grid_col": _int_value(tile_row, "max_grid_col"),
                "selected_mapping_min_row": _min_mapping_int(mapping_rows, "row"),
                "selected_mapping_max_row": _max_mapping_int(mapping_rows, "row"),
                "selected_mapping_min_col": _min_mapping_int(mapping_rows, "col"),
                "selected_mapping_max_col": _max_mapping_int(mapping_rows, "col"),
                "selected_mapping_count": len(mapping_rows),
                "claim_boundary": PHASE31_CASE_DIAGNOSTICS_CLAIM_BOUNDARY,
            }
        )
    return rows


def _selected_block_ids(row: Mapping[str, object]) -> list[str]:
    value = row.get("selected_block_ids", "")
    if isinstance(value, list):
        parts = [str(item) for item in value]
    else:
        parts = str(value).split(";")
    return [part.strip() for part in parts if part.strip()]


def _trace_step_count(
    traces: object,
    variant_id: str,
    tile_id: str,
    seed: int,
) -> int:
    if not isinstance(traces, Mapping):
        return 0
    trained = traces.get("trained_policy")
    if not isinstance(trained, Mapping):
        return 0
    variant = trained.get(variant_id)
    if not isinstance(variant, Mapping):
        return 0
    tile = variant.get(tile_id)
    if not isinstance(tile, Mapping):
        return 0
    steps = tile.get(str(seed))
    if not isinstance(steps, list):
        return 0
    return len(steps)


def _case_role(delta: float) -> str:
    if delta > 0.0:
        return "strong_positive"
    if delta < 0.0:
        return "failure_case"
    return "neutral_case"


def _base_planning_reward(row: Mapping[str, object]) -> float:
    missing = [
        column
        for column in BASE_PLANNING_REWARD_REQUIRED_COLUMNS
        if column not in row or str(row[column]).strip() == ""
    ]
    if missing:
        raise ValueError(
            "Phase 31 case diagnostics requires explicit features: "
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


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    label: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 31 analysis is missing {label}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 31 {label} contains a non-mapping row")
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _phase31_interpretation(status: str) -> str:
    if status == "case_diagnostics_ready":
        return (
            "Phase 31 ranks informative Phase 30 tile-seed cases and summarizes "
            "selected-block composition and tile geometry for spatial inspection. "
            "It is diagnostic evidence only."
        )
    return (
        "Phase 31 could not assemble complete case, selected-block, and tile "
        "geometry diagnostics from the provided artifacts."
    )


def _phase31_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Phase 31 Case Diagnostics",
        "",
        f"Status: {analysis.get('phase31_case_diagnostic_status', '')}",
        "",
        f"Focal comparison: {analysis.get('focal_variant', '')} - "
        f"{analysis.get('comparator_variant', '')}",
        "",
        "## Ranked Cases",
        "",
        "| Rank | Case | Role | Delta | Jaccard | Trace steps |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in analysis.get("ranked_case_rows", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {rank} | `{case}` | {role} | `{delta}` | `{jaccard}` | {steps} |".format(
                rank=row.get("case_rank", ""),
                case=row.get("case_id", ""),
                role=row.get("case_role", ""),
                delta=row.get("variant_minus_comparator_reward", ""),
                jaccard=row.get("selected_block_jaccard", ""),
                steps=row.get("trace_step_count", ""),
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


def _mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values]
    if not clean:
        return None
    return _round_float(sum(clean) / len(clean))


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


def _min_mapping_int(rows: Sequence[Mapping[str, object]], field: str) -> int | None:
    if not rows:
        return None
    return min(_int_value(row, field) for row in rows)


def _max_mapping_int(rows: Sequence[Mapping[str, object]], field: str) -> int | None:
    if not rows:
        return None
    return max(_int_value(row, field) for row in rows)


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _round_float(value: float) -> float:
    return round(float(value), 10)
