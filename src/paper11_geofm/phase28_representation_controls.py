from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
import json
import statistics
import warnings
from os import PathLike
from pathlib import Path

from .padded_heldout_policy import (
    SUMMARY_FIELDNAMES,
    Phase25PaddedTileEnv,
    _dependency_metadata,
    _evaluate_baseline_policy,
    _evaluate_trained_policy,
    _normalize_seeds,
    _select_train_eval_tiles,
    _store_trace,
)
from .tiled_inputs import load_tiled_variant_input


PHASE28_CLAIM_BOUNDARY = (
    "Phase 28 is a representation-control diagnostic over B0/B1/D2/D3/D4 "
    "base-reward padded held-out Bishan policy runs; it does not enable "
    "suitability reward, does not test B2/B3, does not test cross-region "
    "transfer, and does not support submission-level planning-performance "
    "claims."
)

PHASE28_REMAINING_EVIDENCE_GAPS = [
    "full_training_protocol_replication",
    "held_out_region_transfer_evaluation",
    "suitability_reward_validation_before_B2_B3",
    "spatial_case_maps_and_uncertainty",
    "submission_level_ablation_and_robustness_package",
]

PHASE28_DEFAULT_VARIANTS = ("B0", "B1", "D2", "D3", "D4P8", "D4P16")
PHASE28_ALLOWED_VARIANTS = PHASE28_DEFAULT_VARIANTS
PHASE28_PRIMARY_COMPARATORS = ("B0", "D2", "D3")
PHASE28_REQUIRED_CONTROL_VARIANTS = PHASE28_DEFAULT_VARIANTS

TILE_SEED_DELTA_FIELDNAMES = [
    "comparator_variant_id",
    "eval_tile_id",
    "seed",
    "b1_reward",
    "comparator_reward",
    "b1_minus_comparator_reward",
    "b1_improves_comparator",
    "train_timesteps",
    "eval_max_steps",
    "claim_boundary",
]


def build_phase28_representation_control_contract(
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] | str = PHASE28_DEFAULT_VARIANTS,
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 3,
    total_timesteps: int = 32,
    eval_max_steps: int = 4,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
) -> dict[str, object]:
    if int(total_timesteps) <= 0:
        raise ValueError("total_timesteps must be positive")
    if int(eval_max_steps) <= 0:
        raise ValueError("eval_max_steps must be positive")

    normalized_variants = _normalize_phase28_variants(variants)
    normalized_seeds = _normalize_seeds(seeds)
    selected = _select_train_eval_tiles(
        Path(tile_index_csv),
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
    )
    eval_ids = list(selected["eval_tile_ids"])
    selected_counts = dict(selected["selected_tile_block_counts"])
    max_blocks = max(int(selected_counts[tile_id]) for tile_id in selected_counts)
    eval_tile_ranks = {
        str(tile_id): rank for rank, tile_id in enumerate(eval_ids, start=1)
    }
    seed_ranks = {
        str(seed): rank for rank, seed in enumerate(normalized_seeds, start=1)
    }
    variant_source_dirs = _phase28_variant_source_dirs(
        normalized_variants,
        phase2_output_dir=phase2_output_dir,
        phase8_output_dir=phase8_output_dir,
    )

    train_id = str(selected["train_tile_id"])
    return {
        "phase": "phase28_representation_control_evaluation",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "phase8_output_dir": str(Path(phase8_output_dir)),
        "tile_index_csv": str(Path(tile_index_csv)),
        "variants": normalized_variants,
        "variant_source_dirs": variant_source_dirs,
        "train_tile_id": train_id,
        "train_tile_ids": [train_id],
        "eval_tile_ids": eval_ids,
        "eval_tile_count": len(eval_ids),
        "eval_tile_ranks": eval_tile_ranks,
        "selected_tile_block_counts": selected_counts,
        "train_tile_selection": selected["train_tile_selection"],
        "eval_tile_selection": selected["eval_tile_selection"],
        "padded_policy_status": "enabled_phase28_representation_controls",
        "learned_policy_evaluation_scope": (
            "padded_variable_size_heldout_tile_representation_control_diagnostic"
        ),
        "max_blocks": int(max_blocks),
        "total_timesteps": int(total_timesteps),
        "eval_max_steps": int(eval_max_steps),
        "seeds": normalized_seeds,
        "seed_count": len(normalized_seeds),
        "seed_ranks": seed_ranks,
        "claim_boundary": PHASE28_CLAIM_BOUNDARY,
        "remaining_evidence_gaps": list(PHASE28_REMAINING_EVIDENCE_GAPS),
    }


def run_phase28_representation_control_evaluation(
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] | str = PHASE28_DEFAULT_VARIANTS,
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 3,
    total_timesteps: int = 32,
    eval_max_steps: int = 4,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
    compression_match_tolerance: float = 1e-9,
) -> dict[str, object]:
    contract = build_phase28_representation_control_contract(
        phase2_output_dir=phase2_output_dir,
        phase8_output_dir=phase8_output_dir,
        tile_index_csv=tile_index_csv,
        variants=variants,
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
        total_timesteps=total_timesteps,
        eval_max_steps=eval_max_steps,
        seeds=seeds,
    )
    summaries: list[dict[str, object]] = []
    traces: dict[str, dict[str, dict[str, dict[str, list[dict[str, object]]]]]] = {
        "trained_policy": {},
        "first_valid": {},
        "seeded_random": {},
    }

    for variant_id in contract["variants"]:
        for seed in contract["seeds"]:
            train_tiled = _load_phase28_tiled_variant_input(
                contract,
                str(contract["train_tile_id"]),
                str(variant_id),
            )
            train_env = Phase25PaddedTileEnv(
                train_tiled,
                max_blocks=int(contract["max_blocks"]),
                max_steps=int(contract["total_timesteps"]),
            )
            train_env.reset(seed=int(seed))
            model = _train_maskable_ppo_model(
                train_env,
                seed=int(seed),
                total_timesteps=int(contract["total_timesteps"]),
            )

            for eval_tile_id in contract["eval_tile_ids"]:
                eval_tiled = _load_phase28_tiled_variant_input(
                    contract,
                    str(eval_tile_id),
                    str(variant_id),
                )
                eval_tile_rank = int(contract["eval_tile_ranks"][str(eval_tile_id)])
                seed_rank = int(contract["seed_ranks"][str(int(seed))])
                train_n_blocks = int(
                    contract["selected_tile_block_counts"][
                        str(contract["train_tile_id"])
                    ]
                )
                trained_summary, trained_steps = _evaluate_trained_policy(
                    model,
                    eval_tiled,
                    train_tile_id=str(contract["train_tile_id"]),
                    train_n_blocks=train_n_blocks,
                    max_blocks=int(contract["max_blocks"]),
                    eval_tile_rank=eval_tile_rank,
                    phase25_seed_rank=seed_rank,
                    eval_max_steps=int(contract["eval_max_steps"]),
                    train_timesteps=int(contract["total_timesteps"]),
                    seed=int(seed),
                )
                trained_summary["claim_boundary"] = PHASE28_CLAIM_BOUNDARY
                summaries.append(trained_summary)
                _store_trace(
                    traces,
                    "trained_policy",
                    str(variant_id),
                    str(eval_tile_id),
                    int(seed),
                    trained_steps,
                )

                for policy_id in ("first_valid", "seeded_random"):
                    baseline_summary, baseline_steps = _evaluate_baseline_policy(
                        eval_tiled,
                        policy_id=policy_id,
                        train_tile_id=str(contract["train_tile_id"]),
                        train_n_blocks=train_n_blocks,
                        max_blocks=int(contract["max_blocks"]),
                        eval_tile_rank=eval_tile_rank,
                        phase25_seed_rank=seed_rank,
                        eval_max_steps=int(contract["eval_max_steps"]),
                        train_timesteps=int(contract["total_timesteps"]),
                        seed=int(seed),
                    )
                    baseline_summary["claim_boundary"] = PHASE28_CLAIM_BOUNDARY
                    summaries.append(baseline_summary)
                    _store_trace(
                        traces,
                        policy_id,
                        str(variant_id),
                        str(eval_tile_id),
                        int(seed),
                        baseline_steps,
                    )

    analysis = build_phase28_representation_control_analysis(
        summaries,
        compression_match_tolerance=compression_match_tolerance,
        metadata={
            "phase2_output_dir": contract["phase2_output_dir"],
            "phase8_output_dir": contract["phase8_output_dir"],
            "tile_index_csv": contract["tile_index_csv"],
            "variants": list(contract["variants"]),
            "eval_tile_ids": list(contract["eval_tile_ids"]),
            "seeds": list(contract["seeds"]),
            "total_timesteps": int(contract["total_timesteps"]),
            "eval_max_steps": int(contract["eval_max_steps"]),
        },
    )
    return {
        **contract,
        **analysis,
        "training_completed": True,
        "all_evaluations_completed": all(
            bool(row["terminated"]) or bool(row["truncated"]) for row in summaries
        ),
        "summary_count": len(summaries),
        "summaries": summaries,
        "traces": traces,
        "dependencies": _dependency_metadata(),
        "claim_boundary": PHASE28_CLAIM_BOUNDARY,
    }


def build_phase28_representation_control_analysis(
    summary_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    compression_match_tolerance: float = 1e-9,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    rows = _load_summary_rows(summary_rows_or_csv)
    trained_rows = [row for row in rows if str(row.get("row_type", "")) == "trained_policy"]
    metadata_map = {} if metadata is None else dict(metadata)

    variants = _metadata_string_list(
        metadata_map,
        "variants",
        fallback=_unique_strings(trained_rows, "variant_id"),
    )
    eval_tile_ids = _metadata_string_list(
        metadata_map,
        "eval_tile_ids",
        fallback=_unique_strings(trained_rows, "eval_tile_id"),
    )
    seeds = _metadata_int_list(
        metadata_map,
        "seeds",
        fallback=_unique_ints(trained_rows, "seed"),
    )
    train_timesteps = _metadata_int(
        metadata_map,
        ("train_timesteps", "total_timesteps"),
        trained_rows,
        "train_timesteps",
    )
    eval_max_steps = _metadata_int(
        metadata_map,
        ("eval_max_steps",),
        trained_rows,
        "eval_max_steps",
    )

    coverage_issues = _coverage_issues(
        trained_rows,
        variants=variants,
        eval_tile_ids=eval_tile_ids,
        seeds=seeds,
        metadata=metadata_map,
    )
    delta_rows = _tile_seed_delta_rows(
        trained_rows,
        variants=variants,
        eval_tile_ids=eval_tile_ids,
        seeds=seeds,
    )
    learned_policy = _policy_summary(
        trained_rows,
        variants=variants,
        delta_rows=delta_rows,
        coverage_issues=coverage_issues,
    )
    status = _phase28_diagnostic_status(
        learned_policy,
        coverage_issues,
        compression_match_tolerance=float(compression_match_tolerance),
    )

    analysis: dict[str, object] = {
        "phase": "phase28_representation_control_analysis",
        "variants": variants,
        "eval_tile_ids": eval_tile_ids,
        "seeds": seeds,
        "train_timesteps": train_timesteps,
        "eval_max_steps": eval_max_steps,
        "source_rows": rows,
        "main_summary_rows": _main_summary_rows(rows),
        "tile_seed_delta_rows": delta_rows,
        "learned_policy": learned_policy,
        "baselines": _baseline_summaries(rows, variants=variants),
        "coverage_issues": coverage_issues,
        "phase28_diagnostic_status": status,
        "compression_match_tolerance": float(compression_match_tolerance),
        "remaining_evidence_gaps": list(PHASE28_REMAINING_EVIDENCE_GAPS),
        "claim_boundary": PHASE28_CLAIM_BOUNDARY,
    }
    if metadata is not None:
        analysis["metadata"] = metadata_map
    return analysis


def write_phase28_representation_control_artifacts(
    protocol_or_analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / "phase28_representation_control_summary.csv"
    traces_path = output_path / "phase28_representation_control_traces.json"
    comparison_path = output_path / "phase28_representation_control_comparison.json"
    delta_path = output_path / "phase28_tile_seed_delta_table.csv"
    readiness_path = output_path / "phase28_control_readiness.md"

    _write_phase25_summary_csv(
        summary_path,
        protocol_or_analysis.get("summaries", protocol_or_analysis.get("source_rows", [])),
    )
    traces_path.write_text(
        json.dumps(
            _json_ready(protocol_or_analysis.get("traces", {})),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    comparison = _comparison_payload(protocol_or_analysis)
    comparison_path.write_text(
        json.dumps(_json_ready(comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv_mapping_rows(
        delta_path,
        TILE_SEED_DELTA_FIELDNAMES,
        protocol_or_analysis.get("tile_seed_delta_rows"),
        "tile_seed_delta_rows",
    )
    readiness_path.write_text(
        _phase28_control_readiness_markdown(protocol_or_analysis),
        encoding="utf-8",
    )

    return {
        "summary_csv": summary_path,
        "traces_json": traces_path,
        "comparison_json": comparison_path,
        "tile_seed_delta_csv": delta_path,
        "control_readiness_md": readiness_path,
    }


def _load_summary_rows(
    summary_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(summary_rows_or_csv, (str, PathLike)):
        path = Path(summary_rows_or_csv)
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 28 summary CSV: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    rows: list[dict[str, object]] = []
    for row in summary_rows_or_csv:
        if not isinstance(row, Mapping):
            raise ValueError("Phase 28 summary rows must be objects")
        rows.append(dict(row))
    return rows


def _normalize_phase28_variants(variants: Sequence[str] | str) -> list[str]:
    if isinstance(variants, str):
        values = [part.strip() for part in variants.split(",")]
    else:
        values = [str(item).strip() for item in variants]
    normalized = [item.upper() for item in values if item]
    if not normalized:
        raise ValueError("At least one Phase 28 variant must be requested")
    unsupported = [
        variant for variant in normalized if variant not in PHASE28_ALLOWED_VARIANTS
    ]
    if unsupported:
        raise ValueError(f"unsupported Phase 28 variants: {unsupported}")
    if "B1" not in normalized:
        raise ValueError("Phase 28 representation controls requires B1")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Phase 28 variants must be unique")
    return normalized


def _phase28_variant_source_dirs(
    variants: Sequence[str],
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str,
) -> dict[str, str]:
    source_dirs: dict[str, str] = {}
    for variant_id in variants:
        if variant_id in {"B0", "B1"}:
            source_dirs[str(variant_id)] = str(Path(phase2_output_dir))
        else:
            source_dirs[str(variant_id)] = str(Path(phase8_output_dir))
    return source_dirs


def _load_phase28_tiled_variant_input(
    contract: Mapping[str, object],
    tile_id: str,
    variant_id: str,
):
    variant_source_dirs = contract.get("variant_source_dirs")
    if not isinstance(variant_source_dirs, Mapping):
        raise ValueError("Phase 28 contract is missing variant source routing")
    source_dir = variant_source_dirs.get(variant_id)
    if source_dir is None:
        raise ValueError(f"Phase 28 contract has no source for variant {variant_id}")
    return load_tiled_variant_input(
        source_dir,
        str(contract["tile_index_csv"]),
        tile_id,
        variant_id=variant_id,
    )


def _train_maskable_ppo_model(
    train_env: Phase25PaddedTileEnv,
    seed: int,
    total_timesteps: int,
):
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import is_masking_supported
    except ImportError as exc:
        raise RuntimeError(
            "Phase 28 representation-control evaluation requires "
            "stable-baselines3 and sb3-contrib"
        ) from exc
    if not is_masking_supported(train_env):
        raise ValueError("Phase 28 train env does not expose action_masks")
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        seed=int(seed),
        device="cpu",
        verbose=0,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        gamma=0.99,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="XPU device count is zero!.*",
            category=UserWarning,
        )
        model.learn(total_timesteps=int(total_timesteps))
    return model


def _main_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    order: list[tuple[str, str, str]] = []
    for row in rows:
        key = (
            str(row.get("row_type", "")),
            str(row.get("variant_id", "")),
            str(row.get("eval_tile_id", "")),
        )
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for key in order:
        row_type, variant_id, eval_tile_id = key
        group_rows = groups[key]
        rewards = [_float_value(row, "total_contract_reward") for row in group_rows]
        seeds = {_int_value(row, "seed") for row in group_rows}
        summary_rows.append(
            {
                "row_type": row_type,
                "variant_id": variant_id,
                "eval_tile_id": eval_tile_id,
                "seed_count": len(seeds),
                "mean_total_contract_reward": _mean_or_none(rewards),
                "std_total_contract_reward": _std_or_none(rewards),
                "min_total_contract_reward": _round_float(min(rewards)),
                "max_total_contract_reward": _round_float(max(rewards)),
                "train_timesteps": _optional_int(group_rows[0], "train_timesteps"),
                "eval_max_steps": _optional_int(group_rows[0], "eval_max_steps"),
                "claim_boundary": PHASE28_CLAIM_BOUNDARY,
            }
        )
    return summary_rows


def _tile_seed_delta_rows(
    rows: list[dict[str, object]],
    variants: list[str],
    eval_tile_ids: list[str],
    seeds: list[int],
) -> list[dict[str, object]]:
    indexed: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
    for row in rows:
        variant_id = str(row.get("variant_id", ""))
        key = (str(row.get("eval_tile_id", "")), _int_value(row, "seed"))
        indexed.setdefault(key, {})
        indexed[key].setdefault(variant_id, row)

    comparator_variants = [
        variant
        for variant in variants
        if variant != "B1" and variant in PHASE28_ALLOWED_VARIANTS
    ]
    delta_rows: list[dict[str, object]] = []
    for eval_tile_id in eval_tile_ids:
        for seed in seeds:
            by_variant = indexed.get((eval_tile_id, int(seed)), {})
            b1_row = by_variant.get("B1")
            if b1_row is None:
                continue
            b1_reward = _float_value(b1_row, "total_contract_reward")
            for comparator_variant in comparator_variants:
                comparator_row = by_variant.get(comparator_variant)
                if comparator_row is None:
                    continue
                comparator_reward = _float_value(
                    comparator_row,
                    "total_contract_reward",
                )
                delta = _round_float(b1_reward - comparator_reward)
                delta_rows.append(
                    {
                        "comparator_variant_id": comparator_variant,
                        "eval_tile_id": eval_tile_id,
                        "seed": int(seed),
                        "b1_reward": _round_float(b1_reward),
                        "comparator_reward": _round_float(comparator_reward),
                        "b1_minus_comparator_reward": delta,
                        "b1_improves_comparator": delta > 0.0,
                        "train_timesteps": _optional_int(
                            b1_row,
                            "train_timesteps",
                            fallback=_optional_int(comparator_row, "train_timesteps"),
                        ),
                        "eval_max_steps": _optional_int(
                            b1_row,
                            "eval_max_steps",
                            fallback=_optional_int(comparator_row, "eval_max_steps"),
                        ),
                        "claim_boundary": PHASE28_CLAIM_BOUNDARY,
                    }
                )
    return delta_rows


def _policy_summary(
    rows: list[dict[str, object]],
    variants: list[str],
    delta_rows: list[dict[str, object]] | None = None,
    coverage_issues: Mapping[str, object] | None = None,
) -> dict[str, object]:
    mean_reward_by_variant: dict[str, float] = {}
    for variant_id in variants:
        values = [
            _float_value(row, "total_contract_reward")
            for row in rows
            if str(row.get("variant_id", "")) == variant_id
        ]
        if values:
            mean_reward_by_variant[variant_id] = _round_float(sum(values) / len(values))

    comparator_delta_rows = (
        delta_rows
        if delta_rows is not None
        else _tile_seed_delta_rows(
            rows,
            variants=variants,
            eval_tile_ids=_unique_strings(rows, "eval_tile_id"),
            seeds=_unique_ints(rows, "seed"),
        )
    )
    comparator_deltas = _comparator_delta_summaries(
        comparator_delta_rows,
        comparator_variants=[
            variant
            for variant in variants
            if variant != "B1" and variant in PHASE28_ALLOWED_VARIANTS
        ],
    )
    summary: dict[str, object] = {
        "mean_reward_by_variant": mean_reward_by_variant,
        "comparator_deltas": comparator_deltas,
    }
    b0_summary = comparator_deltas.get("B1_minus_B0")
    if isinstance(b0_summary, Mapping):
        summary["B1_minus_B0_mean_reward"] = b0_summary.get("mean_reward_delta")
        summary["B1_minus_B0_std_reward"] = b0_summary.get("std_reward_delta")
        summary["positive_tile_seed_count"] = b0_summary.get("positive_tile_seed_count")
        summary["total_tile_seed_count"] = b0_summary.get("total_tile_seed_count")
        summary["positive_fraction"] = b0_summary.get("positive_fraction")
    if coverage_issues is not None:
        summary["coverage_issues"] = dict(coverage_issues)
    return summary


def _comparator_delta_summaries(
    delta_rows: list[dict[str, object]],
    comparator_variants: list[str],
) -> dict[str, dict[str, object]]:
    rows_by_comparator: dict[str, list[dict[str, object]]] = {
        variant: [] for variant in comparator_variants
    }
    for row in delta_rows:
        comparator = str(row.get("comparator_variant_id", ""))
        rows_by_comparator.setdefault(comparator, []).append(row)

    summaries: dict[str, dict[str, object]] = {}
    for comparator in comparator_variants:
        rows = rows_by_comparator.get(comparator, [])
        deltas = [_float_value(row, "b1_minus_comparator_reward") for row in rows]
        total_count = len(deltas)
        positive_count = sum(
            1 for row in rows if bool(row.get("b1_improves_comparator"))
        )
        summaries[f"B1_minus_{comparator}"] = {
            "mean_reward_delta": _mean_or_none(deltas),
            "std_reward_delta": _std_or_none(deltas),
            "positive_tile_seed_count": positive_count,
            "total_tile_seed_count": total_count,
            "positive_fraction": (
                _round_float(positive_count / total_count) if total_count else None
            ),
        }
    return summaries


def _baseline_summaries(
    rows: list[dict[str, object]],
    variants: list[str],
) -> dict[str, object]:
    baselines: dict[str, object] = {}
    for row_type in _unique_strings(rows, "row_type"):
        if row_type == "trained_policy":
            continue
        baseline_rows = [row for row in rows if str(row.get("row_type", "")) == row_type]
        baselines[row_type] = _policy_summary(baseline_rows, variants=variants)
    return baselines


def _coverage_issues(
    rows: list[dict[str, object]],
    variants: list[str],
    eval_tile_ids: list[str],
    seeds: list[int],
    metadata: Mapping[str, object],
) -> dict[str, object]:
    expected_variants = _expected_variants(variants, metadata)
    expected_tiles = set(eval_tile_ids)
    expected_seeds = {int(seed) for seed in seeds}
    expected_keys = {
        (eval_tile_id, seed, variant_id)
        for eval_tile_id in expected_tiles
        for seed in expected_seeds
        for variant_id in expected_variants
    }

    observed_keys: set[tuple[str, int, str]] = set()
    duplicate_keys: set[tuple[str, int, str]] = set()
    unexpected_keys: set[tuple[str, int, str]] = set()
    for row in rows:
        eval_tile_id = str(row.get("eval_tile_id", ""))
        seed = _int_value(row, "seed")
        variant_id = str(row.get("variant_id", ""))
        key = (eval_tile_id, seed, variant_id)
        if key in observed_keys:
            duplicate_keys.add(key)
        observed_keys.add(key)
        if (
            variant_id not in expected_variants
            or eval_tile_id not in expected_tiles
            or seed not in expected_seeds
        ):
            unexpected_keys.add(key)

    comparable_observed = {
        key
        for key in observed_keys
        if key[2] in expected_variants
        and key[0] in expected_tiles
        and key[1] in expected_seeds
    }
    return {
        "missing_variant_rows": _variant_key_dicts(expected_keys - comparable_observed),
        "unexpected_variant_rows": _variant_key_dicts(unexpected_keys),
        "duplicate_variant_rows": _variant_key_dicts(duplicate_keys),
    }


def _expected_variants(
    variants: list[str],
    metadata: Mapping[str, object],
) -> set[str]:
    if "variants" in metadata:
        return {
            str(variant)
            for variant in _metadata_string_list(metadata, "variants", fallback=[])
            if str(variant) in PHASE28_ALLOWED_VARIANTS
        }
    observed = {variant for variant in variants if variant in PHASE28_ALLOWED_VARIANTS}
    if {"B0", "B1", "D2", "D3"}.issubset(observed):
        return set(PHASE28_REQUIRED_CONTROL_VARIANTS)
    return observed


def _phase28_diagnostic_status(
    learned_policy: Mapping[str, object],
    coverage_issues: Mapping[str, object],
    compression_match_tolerance: float,
) -> str:
    if _has_coverage_issues(coverage_issues):
        return "insufficient"

    comparator_deltas = learned_policy.get("comparator_deltas")
    if not isinstance(comparator_deltas, Mapping):
        return "insufficient"
    required = {
        comparator: comparator_deltas.get(f"B1_minus_{comparator}")
        for comparator in PHASE28_PRIMARY_COMPARATORS
    }
    if not all(_has_delta_summary(summary) for summary in required.values()):
        return "insufficient"

    b0 = required["B0"]
    d2 = required["D2"]
    d3 = required["D3"]
    assert isinstance(b0, Mapping)
    assert isinstance(d2, Mapping)
    assert isinstance(d3, Mapping)

    if (
        _beats_comparator(b0, compression_match_tolerance)
        and _beats_comparator(d2, compression_match_tolerance)
        and _beats_comparator(d3, compression_match_tolerance)
        and _positive_fraction(d2) >= 0.6
        and _positive_fraction(d3) >= 0.6
    ):
        return "representation_signal_supported"

    if _compression_matches_raw(comparator_deltas, compression_match_tolerance):
        return "compression_matches_raw"

    if not _beats_comparator(d2, compression_match_tolerance) and not _beats_comparator(
        d3,
        compression_match_tolerance,
    ):
        return "representation_signal_not_distinguishable"

    return "representation_signal_control_limited"


def _has_delta_summary(summary: object) -> bool:
    if not isinstance(summary, Mapping):
        return False
    if summary.get("mean_reward_delta") is None:
        return False
    return int(summary.get("total_tile_seed_count") or 0) > 0


def _beats_comparator(
    summary: Mapping[str, object],
    tolerance: float,
) -> bool:
    delta = summary.get("mean_reward_delta")
    return delta is not None and float(delta) > float(tolerance)


def _compression_matches_raw(
    comparator_deltas: Mapping[str, object],
    tolerance: float,
) -> bool:
    for comparator in ("D4P8", "D4P16"):
        summary = comparator_deltas.get(f"B1_minus_{comparator}")
        if not isinstance(summary, Mapping):
            continue
        delta = summary.get("mean_reward_delta")
        if delta is not None and float(delta) <= float(tolerance):
            return True
    return False


def _positive_fraction(summary: Mapping[str, object]) -> float:
    value = summary.get("positive_fraction")
    return 0.0 if value is None else float(value)


def _has_coverage_issues(coverage_issues: Mapping[str, object]) -> bool:
    return any(
        bool(coverage_issues.get(key))
        for key in (
            "missing_variant_rows",
            "unexpected_variant_rows",
            "duplicate_variant_rows",
        )
    )


def _write_phase25_summary_csv(path: Path, rows: object) -> None:
    if not isinstance(rows, list):
        raise ValueError("Phase 28 protocol is missing a summaries list")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Phase 28 summary rows must be objects")
            output_row = {field: row.get(field, "") for field in SUMMARY_FIELDNAMES}
            selected = output_row.get("selected_block_ids")
            if isinstance(selected, list):
                output_row["selected_block_ids"] = ";".join(str(item) for item in selected)
            writer.writerow(output_row)


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: list[str],
    rows: object,
    row_key: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 28 analysis is missing a {row_key} list")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 28 {row_key} rows must be objects")
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _comparison_payload(protocol_or_analysis: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in dict(protocol_or_analysis).items()
        if key not in {"summaries", "traces"}
    }


def _phase28_control_readiness_markdown(analysis: Mapping[str, object]) -> str:
    learned = analysis.get("learned_policy")
    if not isinstance(learned, Mapping):
        learned = {}
    comparator_deltas = learned.get("comparator_deltas")
    if not isinstance(comparator_deltas, Mapping):
        comparator_deltas = {}
    gaps = analysis.get("remaining_evidence_gaps")
    if not isinstance(gaps, list):
        gaps = []

    lines = [
        "# Phase 28 Control Readiness",
        "",
        f"Status: {analysis.get('phase28_diagnostic_status', '')}",
        "",
        "Primary comparator deltas:",
    ]
    for comparator in PHASE28_PRIMARY_COMPARATORS:
        summary = comparator_deltas.get(f"B1_minus_{comparator}")
        if isinstance(summary, Mapping):
            lines.append(
                "- "
                f"B1 minus {comparator}: "
                f"{summary.get('mean_reward_delta')} "
                f"({summary.get('positive_tile_seed_count')} / "
                f"{summary.get('total_tile_seed_count')} positive)"
            )
    lines.extend(
        [
            "",
            str(analysis.get("claim_boundary", PHASE28_CLAIM_BOUNDARY)),
            "",
            "Unsafe wording:",
            "- Do not claim planning improvement, transfer, or submission readiness from this diagnostic alone.",
            "",
            "Remaining evidence gaps:",
        ]
    )
    for gap in gaps:
        lines.append(f"- {gap}")
    lines.append("")
    return "\n".join(lines)


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _metadata_string_list(
    metadata: Mapping[str, object],
    key: str,
    fallback: list[str],
) -> list[str]:
    value = metadata.get(key)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item).strip()]
    return fallback


def _metadata_int_list(
    metadata: Mapping[str, object],
    key: str,
    fallback: list[int],
) -> list[int]:
    value = metadata.get(key)
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [int(item) for item in value if str(item).strip()]
    return fallback


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


def _optional_int(
    row: Mapping[str, object],
    field: str,
    fallback: int | None = None,
) -> int | None:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        return fallback
    return int(value)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return _round_float(sum(values) / len(values))


def _std_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return _round_float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _round_float(value: float) -> float:
    return round(float(value), 10)


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


def _variant_key_dicts(keys: set[tuple[str, int, str]]) -> list[dict[str, object]]:
    return [
        {"eval_tile_id": eval_tile_id, "seed": seed, "variant_id": variant_id}
        for eval_tile_id, seed, variant_id in sorted(
            keys,
            key=lambda item: (item[0], item[1], item[2]),
        )
    ]
