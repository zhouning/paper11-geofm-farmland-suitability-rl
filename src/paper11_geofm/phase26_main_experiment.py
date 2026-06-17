from __future__ import annotations

from collections.abc import Mapping
import csv
import json
import statistics
from pathlib import Path


PHASE26_CLAIM_BOUNDARY = (
    "Phase 26 is a main empirical analysis package for B0/B1 padded held-out "
    "Bishan tile learned-policy results under the deterministic base planning "
    "reward; it does not enable suitability reward, does not test B2/B3, and "
    "does not support cross-region transfer or final submission-level claims."
)

PHASE26_REMAINING_EVIDENCE_GAPS = [
    "longer_budget_replication_if_1024_steps_is_used",
    "suitability_reward_validation_before_B2_B3",
    "held_out_region_transfer_evaluation",
    "spatial_case_maps_and_uncertainty",
    "submission_level_ablation_and_robustness_package",
]

MAIN_SUMMARY_FIELDNAMES = [
    "row_type",
    "variant_id",
    "eval_tile_id",
    "seed_count",
    "mean_total_contract_reward",
    "std_total_contract_reward",
    "min_total_contract_reward",
    "max_total_contract_reward",
    "train_timesteps",
    "eval_max_steps",
    "claim_boundary",
]

DELTA_FIELDNAMES = [
    "eval_tile_id",
    "seed",
    "b0_reward",
    "b1_reward",
    "b1_minus_b0_reward",
    "b1_improves_b0",
    "train_timesteps",
    "eval_max_steps",
]


def build_phase26_main_empirical_analysis(
    phase25_output_dir: Path | str,
) -> dict[str, object]:
    phase25_path = Path(phase25_output_dir)
    summary_path = phase25_path / "phase25_padded_heldout_policy_summary.csv"
    comparison_path = phase25_path / "phase25_padded_heldout_policy_comparison.json"

    rows = _read_summary_rows(summary_path)
    comparison = _read_json_object(comparison_path)

    eval_tile_ids = _string_list(
        comparison.get("eval_tile_ids"),
        fallback=_unique_strings(rows, "eval_tile_id"),
    )
    variants = _string_list(
        comparison.get("variants"),
        fallback=_unique_strings(rows, "variant_id"),
    )
    seeds = _int_list(comparison.get("seeds"), fallback=_unique_ints(rows, "seed"))
    train_timesteps = _metadata_int(
        comparison,
        ("train_timesteps", "total_timesteps"),
        rows,
        "train_timesteps",
    )
    eval_max_steps = _metadata_int(
        comparison,
        ("eval_max_steps",),
        rows,
        "eval_max_steps",
    )

    main_rows = _main_summary_rows(rows)
    coverage_issues = _coverage_issues(rows, eval_tile_ids, seeds)
    delta_rows = _tile_seed_delta_rows(rows, coverage_issues)
    learned = _learned_policy_summary(delta_rows, coverage_issues)
    expected_total = len(eval_tile_ids) * len(seeds)

    return {
        "phase": "phase26_main_empirical_experiment",
        "source_phase25": {
            "summary_csv": str(summary_path),
            "comparison_json": str(comparison_path),
        },
        "train_tile_id": str(
            comparison.get("train_tile_id") or _first_nonempty(rows, "train_tile_id")
        ),
        "eval_tile_ids": eval_tile_ids,
        "variants": variants,
        "seeds": seeds,
        "train_timesteps": train_timesteps,
        "eval_max_steps": eval_max_steps,
        "main_summary_rows": main_rows,
        "tile_seed_delta_rows": delta_rows,
        "learned_policy": learned,
        "baselines": _baseline_summaries(rows),
        "phase26_claim_status": _phase26_claim_status(learned, expected_total),
        "remaining_evidence_gaps": list(PHASE26_REMAINING_EVIDENCE_GAPS),
        "claim_boundary": PHASE26_CLAIM_BOUNDARY,
    }


def write_phase26_main_empirical_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    main_summary_path = output_path / "phase26_main_summary.csv"
    delta_path = output_path / "phase26_tile_seed_delta_table.csv"
    comparison_path = output_path / "phase26_main_comparison.json"
    claim_readiness_path = output_path / "phase26_claim_readiness.md"

    _write_csv_mapping_rows(
        main_summary_path,
        MAIN_SUMMARY_FIELDNAMES,
        analysis.get("main_summary_rows"),
        "main_summary_rows",
    )
    _write_csv_mapping_rows(
        delta_path,
        DELTA_FIELDNAMES,
        analysis.get("tile_seed_delta_rows"),
        "tile_seed_delta_rows",
    )
    comparison_path.write_text(
        json.dumps(dict(analysis), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    claim_readiness_path.write_text(
        _phase26_claim_readiness_markdown(analysis),
        encoding="utf-8",
    )

    return {
        "main_summary_csv": main_summary_path,
        "tile_seed_delta_csv": delta_path,
        "comparison_json": comparison_path,
        "claim_readiness_md": claim_readiness_path,
    }


def _read_summary_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 26 input summary CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 26 input JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Phase 26 JSON input must be an object")
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


def _main_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row.get("row_type", "")),
            str(row.get("variant_id", "")),
            str(row.get("eval_tile_id", "")),
        )
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (row_type, variant_id, eval_tile_id), group_rows in groups.items():
        values = [_float_value(row, "total_contract_reward") for row in group_rows]
        seeds = {_int_value(row, "seed") for row in group_rows}
        std_reward = statistics.pstdev(values) if len(values) > 1 else 0.0
        summary_rows.append(
            {
                "row_type": row_type,
                "variant_id": variant_id,
                "eval_tile_id": eval_tile_id,
                "seed_count": len(seeds),
                "mean_total_contract_reward": _round_float(sum(values) / len(values)),
                "std_total_contract_reward": _round_float(std_reward),
                "min_total_contract_reward": _round_float(min(values)),
                "max_total_contract_reward": _round_float(max(values)),
                "train_timesteps": _int_value(group_rows[0], "train_timesteps"),
                "eval_max_steps": _int_value(group_rows[0], "eval_max_steps"),
                "claim_boundary": PHASE26_CLAIM_BOUNDARY,
            }
        )
    return summary_rows


def _tile_seed_delta_rows(
    rows: list[dict[str, object]],
    coverage_issues: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    duplicate_keys = set()
    if isinstance(coverage_issues, Mapping):
        duplicate_keys = {
            (
                str(item.get("eval_tile_id", "")),
                int(item.get("seed", 0)),
                str(item.get("variant_id", "")),
            )
            for item in coverage_issues.get("duplicate_variant_rows", [])
            if isinstance(item, Mapping)
        }

    paired: dict[tuple[str, int], dict[str, object]] = {}
    order: list[tuple[str, int]] = []
    for row in rows:
        if row.get("row_type") != "trained_policy":
            continue
        eval_tile_id = str(row.get("eval_tile_id", ""))
        seed = _int_value(row, "seed")
        variant_id = str(row.get("variant_id", ""))
        if (eval_tile_id, seed, variant_id) in duplicate_keys:
            continue
        key = (eval_tile_id, seed)
        if key not in paired:
            paired[key] = {
                "eval_tile_id": eval_tile_id,
                "seed": seed,
                "rewards": {},
                "train_timesteps": _int_value(row, "train_timesteps"),
                "eval_max_steps": _int_value(row, "eval_max_steps"),
            }
            order.append(key)
        rewards = paired[key]["rewards"]
        if isinstance(rewards, dict):
            rewards[variant_id] = _float_value(
                row,
                "total_contract_reward",
            )

    delta_rows: list[dict[str, object]] = []
    for key in order:
        record = paired[key]
        rewards = record["rewards"]
        if not isinstance(rewards, dict) or "B0" not in rewards or "B1" not in rewards:
            continue
        b0_reward = float(rewards["B0"])
        b1_reward = float(rewards["B1"])
        delta = _round_float(b1_reward - b0_reward)
        delta_rows.append(
            {
                "eval_tile_id": record["eval_tile_id"],
                "seed": record["seed"],
                "b0_reward": _round_float(b0_reward),
                "b1_reward": _round_float(b1_reward),
                "b1_minus_b0_reward": delta,
                "b1_improves_b0": delta > 0,
                "train_timesteps": record["train_timesteps"],
                "eval_max_steps": record["eval_max_steps"],
            }
        )
    return delta_rows


def _learned_policy_summary(
    delta_rows: list[dict[str, object]],
    coverage_issues: Mapping[str, object] | None = None,
) -> dict[str, object]:
    deltas = [_float_value(row, "b1_minus_b0_reward") for row in delta_rows]
    positive_count = sum(1 for row in delta_rows if bool(row.get("b1_improves_b0")))
    total_count = len(delta_rows)
    mean_delta = _mean_or_none(deltas)
    std_delta = (
        _round_float(statistics.pstdev(deltas))
        if len(deltas) > 1
        else (0.0 if deltas else None)
    )
    return {
        "B1_minus_B0_mean_reward": mean_delta,
        "B1_minus_B0_std_reward": std_delta,
        "positive_tile_seed_count": positive_count,
        "total_tile_seed_count": total_count,
        "positive_fraction": (
            _round_float(positive_count / total_count) if total_count else None
        ),
        "per_tile_mean_delta": _mean_delta_by_field(delta_rows, "eval_tile_id"),
        "per_seed_mean_delta": _mean_delta_by_field(delta_rows, "seed"),
        "coverage_issues": _empty_coverage_issues()
        if coverage_issues is None
        else dict(coverage_issues),
    }


def _baseline_summaries(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        row_type: _policy_summary(rows, row_type)
        for row_type in ("first_valid", "seeded_random")
    }


def _phase26_claim_status(learned: Mapping[str, object], expected_total: int) -> str:
    total = int(learned.get("total_tile_seed_count") or 0)
    delta = learned.get("B1_minus_B0_mean_reward")
    positive_fraction = learned.get("positive_fraction")
    coverage_issues = learned.get("coverage_issues")
    if isinstance(coverage_issues, Mapping) and any(
        bool(coverage_issues.get(key))
        for key in (
            "missing_tile_seed_pairs",
            "unexpected_tile_seed_pairs",
            "duplicate_variant_rows",
        )
    ):
        return "insufficient"
    if (
        total <= 0
        or total < int(expected_total)
        or delta is None
        or positive_fraction is None
    ):
        return "insufficient"
    if float(delta) <= 0:
        return "not_supported"
    if float(positive_fraction) >= 0.6:
        return "pilot_supported"
    return "mixed"


def _policy_summary(rows: list[dict[str, object]], row_type: str) -> dict[str, object]:
    policy_rows = [row for row in rows if row.get("row_type") == row_type]
    mean_reward_by_variant: dict[str, float] = {}
    for variant_id in _unique_strings(policy_rows, "variant_id"):
        values = [
            _float_value(row, "total_contract_reward")
            for row in policy_rows
            if str(row.get("variant_id", "")) == variant_id
        ]
        if values:
            mean_reward_by_variant[variant_id] = _round_float(
                sum(values) / len(values)
            )

    delta_rows = _variant_delta_rows(policy_rows)
    deltas = [_float_value(row, "b1_minus_b0_reward") for row in delta_rows]
    return {
        "mean_reward_by_variant": mean_reward_by_variant,
        "B1_minus_B0_mean_reward": _mean_or_none(deltas),
        "B1_minus_B0_std_reward": (
            _round_float(statistics.pstdev(deltas))
            if len(deltas) > 1
            else (0.0 if deltas else None)
        ),
        "paired_tile_seed_count": len(delta_rows),
        "per_tile_mean_delta": _mean_delta_by_field(delta_rows, "eval_tile_id"),
        "per_seed_mean_delta": _mean_delta_by_field(delta_rows, "seed"),
    }


def _variant_delta_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    patched_rows: list[dict[str, object]] = []
    for row in rows:
        patched = dict(row)
        patched["row_type"] = "trained_policy"
        patched_rows.append(patched)
    return _tile_seed_delta_rows(patched_rows)


def _coverage_issues(
    rows: list[dict[str, object]],
    eval_tile_ids: list[str],
    seeds: list[int],
) -> dict[str, object]:
    expected_pairs = {(str(tile_id), int(seed)) for tile_id in eval_tile_ids for seed in seeds}
    observed_pairs: set[tuple[str, int]] = set()
    seen_variant_keys: set[tuple[str, int, str]] = set()
    duplicate_variant_keys: set[tuple[str, int, str]] = set()

    for row in rows:
        if row.get("row_type") != "trained_policy":
            continue
        eval_tile_id = str(row.get("eval_tile_id", ""))
        seed = _int_value(row, "seed")
        variant_id = str(row.get("variant_id", ""))
        observed_pairs.add((eval_tile_id, seed))
        variant_key = (eval_tile_id, seed, variant_id)
        if variant_key in seen_variant_keys:
            duplicate_variant_keys.add(variant_key)
        seen_variant_keys.add(variant_key)

    return {
        "missing_tile_seed_pairs": _pair_dicts(expected_pairs - observed_pairs),
        "unexpected_tile_seed_pairs": _pair_dicts(observed_pairs - expected_pairs),
        "duplicate_variant_rows": _variant_key_dicts(duplicate_variant_keys),
    }


def _empty_coverage_issues() -> dict[str, object]:
    return {
        "missing_tile_seed_pairs": [],
        "unexpected_tile_seed_pairs": [],
        "duplicate_variant_rows": [],
    }


def _pair_dicts(pairs: set[tuple[str, int]]) -> list[dict[str, object]]:
    return [
        {"eval_tile_id": eval_tile_id, "seed": seed}
        for eval_tile_id, seed in sorted(pairs, key=lambda item: (item[0], item[1]))
    ]


def _variant_key_dicts(keys: set[tuple[str, int, str]]) -> list[dict[str, object]]:
    return [
        {"eval_tile_id": eval_tile_id, "seed": seed, "variant_id": variant_id}
        for eval_tile_id, seed, variant_id in sorted(
            keys,
            key=lambda item: (item[0], item[1], item[2]),
        )
    ]


def _mean_delta_by_field(
    delta_rows: list[dict[str, object]],
    field: str,
) -> dict[object, float]:
    groups: dict[object, list[float]] = {}
    for row in delta_rows:
        key = row.get(field)
        groups.setdefault(key, []).append(_float_value(row, "b1_minus_b0_reward"))
    return {key: _round_float(sum(values) / len(values)) for key, values in groups.items()}


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return _round_float(sum(values) / len(values))


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: list[str],
    rows: object,
    row_key: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 26 analysis is missing a {row_key} list")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 26 {row_key} rows must be objects")
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _phase26_claim_readiness_markdown(analysis: Mapping[str, object]) -> str:
    learned = analysis.get("learned_policy")
    if not isinstance(learned, Mapping):
        learned = {}
    gaps = analysis.get("remaining_evidence_gaps")
    if not isinstance(gaps, list):
        gaps = []

    lines = [
        "# Phase 26 Claim Readiness",
        "",
        f"Status: {analysis.get('phase26_claim_status', '')}",
        "",
        (
            "B1-B0 learned-policy mean reward delta: "
            f"{learned.get('B1_minus_B0_mean_reward')}"
        ),
        (
            "Positive tile-seed count: "
            f"{learned.get('positive_tile_seed_count')} / "
            f"{learned.get('total_tile_seed_count')}"
        ),
        "",
        str(analysis.get("claim_boundary", PHASE26_CLAIM_BOUNDARY)),
        "",
        "Remaining evidence gaps:",
    ]
    readable_gaps = {
        "longer_budget_replication_if_1024_steps_is_used": (
            "longer budget replication if 1024 steps is used"
        ),
        "suitability_reward_validation_before_B2_B3": (
            "suitability reward validation before B2/B3"
        ),
        "held_out_region_transfer_evaluation": (
            "held-out region transfer evaluation"
        ),
        "spatial_case_maps_and_uncertainty": "spatial case maps and uncertainty",
        "submission_level_ablation_and_robustness_package": (
            "submission-level ablation and robustness package"
        ),
    }
    for gap in gaps:
        lines.append(f"- {readable_gaps.get(str(gap), str(gap))}")
    lines.append("")
    return "\n".join(lines)


def _metadata_int(
    metadata: Mapping[str, object],
    keys: tuple[str, ...],
    rows: list[dict[str, object]],
    row_field: str,
) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip() != "":
            return int(value)
    first = _first_nonempty(rows, row_field)
    return int(first) if first is not None else None


def _first_nonempty(
    rows: list[dict[str, object]],
    field: str,
) -> object | None:
    for row in rows:
        value = row.get(field)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _unique_strings(rows: list[dict[str, object]], field: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        text = str(value)
        if text not in seen:
            values.append(text)
            seen.add(text)
    return values


def _unique_ints(rows: list[dict[str, object]], field: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for row in rows:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        number = int(value)
        if number not in seen:
            values.append(number)
            seen.add(number)
    return values


def _string_list(value: object, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return fallback


def _int_list(value: object, fallback: list[int]) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    return fallback
