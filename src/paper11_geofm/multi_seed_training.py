from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .bounded_tiled_training import (
    PHASE20_CROSS_TILE_BLOCKER,
    SUMMARY_FIELDNAMES as PHASE20_SUMMARY_FIELDNAMES,
    build_phase20_bounded_training_contract,
    run_phase20_bounded_tiled_training,
)


PHASE23_CLAIM_BOUNDARY = (
    "Phase 23 is a bounded multi-seed same-tile B0/B1 MaskablePPO training "
    "pilot under the deterministic base planning reward; it strengthens "
    "learned-policy execution evidence relative to Phase 20, but does not "
    "prove GeoFM superiority, suitability-reward benefit, cross-region "
    "transfer, or submission-level planning performance."
)

REMAINING_EVIDENCE_GAPS = [
    "cross_tile_or_variable_size_learned_policy_evaluation",
    "longer_training_budget_and_hyperparameter_sensitivity",
    "suitability_reward_validation_before_B2_B3",
    "held_out_region_transfer_evaluation",
    "ablation_controls_and_spatial_case_maps",
]

SUMMARY_FIELDNAMES = []
for _field in PHASE20_SUMMARY_FIELDNAMES:
    SUMMARY_FIELDNAMES.append(_field)
    if _field == "seed":
        SUMMARY_FIELDNAMES.append("phase23_seed_rank")


def build_phase23_multi_seed_training_contract(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] = ("B0", "B1"),
    train_tile_id: str | None = None,
    eval_tile_id: str | None = None,
    total_timesteps: int = 8,
    eval_max_steps: int = 4,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
) -> dict[str, object]:
    normalized_seeds = _normalize_seeds(seeds)
    seed_ranks = {str(seed): rank for rank, seed in enumerate(normalized_seeds, start=1)}
    phase20_contract = build_phase20_bounded_training_contract(
        phase2_output_dir,
        tile_index_csv,
        variants=variants,
        train_tile_id=train_tile_id,
        eval_tile_id=eval_tile_id,
        total_timesteps=total_timesteps,
        eval_max_steps=eval_max_steps,
        seed=normalized_seeds[0],
    )
    return {
        **phase20_contract,
        "phase": "phase23_multi_seed_training",
        "seeds": normalized_seeds,
        "seed_count": len(normalized_seeds),
        "seed_ranks": seed_ranks,
        "learned_policy_evaluation_scope": "multi_seed_same_tile_b0_b1_training_pilot",
        "cross_tile_evaluation_status": "blocked_variable_observation_shape",
        "cross_tile_blocker": PHASE20_CROSS_TILE_BLOCKER,
        "claim_boundary": PHASE23_CLAIM_BOUNDARY,
        "remaining_evidence_gaps": REMAINING_EVIDENCE_GAPS,
    }


def run_phase23_multi_seed_training(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] = ("B0", "B1"),
    train_tile_id: str | None = None,
    eval_tile_id: str | None = None,
    total_timesteps: int = 8,
    eval_max_steps: int = 4,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
) -> dict[str, object]:
    contract = build_phase23_multi_seed_training_contract(
        phase2_output_dir,
        tile_index_csv,
        variants=variants,
        train_tile_id=train_tile_id,
        eval_tile_id=eval_tile_id,
        total_timesteps=total_timesteps,
        eval_max_steps=eval_max_steps,
        seeds=seeds,
    )

    summaries: list[dict[str, object]] = []
    traces: dict[str, dict[str, dict[str, list[dict[str, object]]]]] = {
        "trained_policy": {},
        "first_valid": {},
        "seeded_random": {},
    }
    dependencies: dict[str, object] | None = None

    for seed in contract["seeds"]:
        phase20 = run_phase20_bounded_tiled_training(
            phase2_output_dir,
            tile_index_csv,
            variants=contract["variants"],
            train_tile_id=str(contract["train_tile_id"]),
            eval_tile_id=str(contract["eval_tile_id"]),
            total_timesteps=int(contract["total_timesteps"]),
            eval_max_steps=int(contract["eval_max_steps"]),
            seed=int(seed),
        )
        dependencies = phase20.get("dependencies") if isinstance(phase20, Mapping) else None
        seed_rank = int(contract["seed_ranks"][str(int(seed))])
        for summary in phase20["summaries"]:
            if not isinstance(summary, Mapping):
                raise ValueError("Phase 20 summary rows must be objects")
            summaries.append(_phase23_summary(summary, seed_rank))

        phase20_traces = phase20.get("traces", {})
        if not isinstance(phase20_traces, Mapping):
            raise ValueError("Phase 20 traces must be an object")
        _merge_seed_traces(traces, phase20_traces, int(seed))

    comparison = _build_comparison(summaries, contract)
    return {
        **contract,
        "training_completed": True,
        "all_evaluations_completed": all(
            bool(row["terminated"]) or bool(row["truncated"]) for row in summaries
        ),
        "summary_count": len(summaries),
        "summaries": summaries,
        "traces": traces,
        "comparison": comparison,
        "dependencies": dependencies or {},
    }


def write_phase23_multi_seed_training_artifacts(
    protocol: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "phase23_multi_seed_training_summary.csv"
    traces_path = output_path / "phase23_multi_seed_training_traces.json"
    comparison_path = output_path / "phase23_multi_seed_training_comparison.json"

    summaries = protocol.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("Phase 23 protocol is missing a summaries list")

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for summary in summaries:
            if not isinstance(summary, Mapping):
                raise ValueError("Phase 23 summary rows must be objects")
            row = {field: summary.get(field, "") for field in SUMMARY_FIELDNAMES}
            selected = row.get("selected_block_ids")
            if isinstance(selected, list):
                row["selected_block_ids"] = ";".join(str(item) for item in selected)
            writer.writerow(row)

    traces_path.write_text(
        json.dumps(dict(protocol), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    comparison = protocol.get("comparison")
    if not isinstance(comparison, Mapping):
        raise ValueError("Phase 23 protocol is missing a comparison object")
    comparison_path.write_text(
        json.dumps(dict(comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "summary_csv": summary_path,
        "traces_json": traces_path,
        "comparison_json": comparison_path,
    }


def _normalize_seeds(
    seeds: Sequence[int | str] | str | int | None,
) -> list[int]:
    if seeds is None:
        values: list[int | str] = [0, 1, 2]
    elif isinstance(seeds, str):
        values = [part.strip() for part in seeds.split(",")]
    elif isinstance(seeds, int):
        values = [seeds]
    else:
        values = list(seeds)

    normalized: list[int] = []
    for value in values:
        if str(value).strip() == "":
            continue
        normalized.append(int(value))
    if not normalized:
        raise ValueError("At least one Phase 23 seed must be requested")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Phase 23 seeds must be unique")
    return normalized


def _phase23_summary(
    summary: Mapping[str, object],
    seed_rank: int,
) -> dict[str, object]:
    row = dict(summary)
    row["phase23_seed_rank"] = int(seed_rank)
    row["claim_boundary"] = PHASE23_CLAIM_BOUNDARY
    return row


def _merge_seed_traces(
    traces: dict[str, dict[str, dict[str, list[dict[str, object]]]]],
    seed_traces: Mapping[str, object],
    seed: int,
) -> None:
    for row_type in traces:
        policy_traces = seed_traces.get(row_type, {})
        if not isinstance(policy_traces, Mapping):
            continue
        for variant_id, steps in policy_traces.items():
            if not isinstance(steps, list):
                raise ValueError("Phase 20 trace leaves must be step lists")
            variant_traces = traces[row_type].setdefault(str(variant_id), {})
            variant_traces[str(int(seed))] = steps


def _build_comparison(
    summaries: list[dict[str, object]],
    contract: Mapping[str, object],
) -> dict[str, object]:
    policy_means: dict[str, dict[str, float]] = {}
    for row_type in ("trained_policy", "first_valid", "seeded_random"):
        policy_means[row_type] = {}
        for variant_id in contract["variants"]:
            values = [
                float(row["total_contract_reward"])
                for row in summaries
                if row["row_type"] == row_type and row["variant_id"] == variant_id
            ]
            if values:
                policy_means[row_type][str(variant_id)] = _round_float(
                    sum(values) / len(values)
                )

    learned = dict(policy_means["trained_policy"])
    b1_minus_b0 = None
    if "B0" in learned and "B1" in learned:
        b1_minus_b0 = _round_float(learned["B1"] - learned["B0"])

    return {
        "phase": "phase23_multi_seed_training_comparison",
        "train_tile_id": contract["train_tile_id"],
        "eval_tile_id": contract["eval_tile_id"],
        "variants": list(contract["variants"]),
        "seeds": list(contract["seeds"]),
        "seed_count": int(contract["seed_count"]),
        "policies": ["trained_policy", "first_valid", "seeded_random"],
        "summary_count": len(summaries),
        "learned_policy": {
            "mean_reward_by_variant": learned,
            "B1_minus_B0_mean_reward": b1_minus_b0,
        },
        "baselines": {
            "first_valid": policy_means["first_valid"],
            "seeded_random": policy_means["seeded_random"],
        },
        "remaining_evidence_gaps": list(REMAINING_EVIDENCE_GAPS),
        "claim_boundary": PHASE23_CLAIM_BOUNDARY,
    }


def _round_float(value: float) -> float:
    return round(float(value), 10)
