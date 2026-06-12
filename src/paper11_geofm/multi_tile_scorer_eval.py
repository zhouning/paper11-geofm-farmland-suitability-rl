from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .cross_tile_block_scorer import (
    SUMMARY_FIELDNAMES as PHASE21_SUMMARY_FIELDNAMES,
)
from .cross_tile_block_scorer import (
    _evaluate_baseline_policy,
    _evaluate_learned_scorer,
    _fit_ridge_block_scorer,
    _normalize_variants,
    _read_tile_rows,
)
from .tiled_inputs import load_tiled_variant_input


PHASE22_CLAIM_BOUNDARY = (
    "Phase 22 is a bounded multi-tile, multi-seed per-block scorer evaluation "
    "pilot; it broadens the Phase 21 variable-block-count scorer interface "
    "check across several evaluation tiles and seeds, does not enable "
    "suitability reward or PPO training, and does not support final "
    "policy-performance, cross-region transfer, or GeoFM-superiority claims."
)

SUMMARY_FIELDNAMES = []
for _field in PHASE21_SUMMARY_FIELDNAMES:
    SUMMARY_FIELDNAMES.append(_field)
    if _field == "eval_tile_id":
        SUMMARY_FIELDNAMES.append("eval_tile_rank")


def build_phase22_multi_tile_scorer_eval_contract(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] = ("B0", "B1"),
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 2,
    ridge_alpha: float = 1e-6,
    eval_max_steps: int = 4,
    seeds: Sequence[int | str] | str | int | None = (0, 1),
) -> dict[str, object]:
    if float(ridge_alpha) < 0.0:
        raise ValueError("ridge_alpha must be non-negative")
    if int(eval_max_steps) <= 0:
        raise ValueError("eval_max_steps must be positive")

    normalized_variants = _normalize_variants(variants)
    normalized_seeds = _normalize_seeds(seeds)
    selected = _select_train_eval_tiles(
        Path(tile_index_csv),
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
    )

    eval_tile_ids_out = list(selected["eval_tile_ids"])
    eval_tile_ranks = {
        str(tile_id): rank + 1 for rank, tile_id in enumerate(eval_tile_ids_out)
    }
    return {
        "phase": "phase22_multi_tile_scorer_eval",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "tile_index_csv": str(Path(tile_index_csv)),
        "variants": normalized_variants,
        "train_tile_id": selected["train_tile_id"],
        "eval_tile_ids": eval_tile_ids_out,
        "eval_tile_count": len(eval_tile_ids_out),
        "eval_tile_ranks": eval_tile_ranks,
        "train_tile_selection": selected["train_tile_selection"],
        "eval_tile_selection": selected["eval_tile_selection"],
        "learned_policy_evaluation_scope": "multi_tile_multi_seed_per_block_scorer_pilot",
        "multi_tile_evaluation_status": "executed_distinct_tiles",
        "ridge_alpha": float(ridge_alpha),
        "eval_max_steps": int(eval_max_steps),
        "seeds": normalized_seeds,
        "seed_count": len(normalized_seeds),
        "claim_boundary": PHASE22_CLAIM_BOUNDARY,
    }


def run_phase22_multi_tile_scorer_eval(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] = ("B0", "B1"),
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 2,
    ridge_alpha: float = 1e-6,
    eval_max_steps: int = 4,
    seeds: Sequence[int | str] | str | int | None = (0, 1),
) -> dict[str, object]:
    contract = build_phase22_multi_tile_scorer_eval_contract(
        phase2_output_dir,
        tile_index_csv,
        variants=variants,
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
        ridge_alpha=ridge_alpha,
        eval_max_steps=eval_max_steps,
        seeds=seeds,
    )

    summaries: list[dict[str, object]] = []
    traces: dict[str, dict[str, dict[str, dict[str, list[dict[str, object]]]]]] = {
        "learned_block_scorer": {},
        "first_valid": {},
        "seeded_random": {},
    }
    model_metadata: dict[str, dict[str, object]] = {}
    eval_tile_ranks = dict(contract["eval_tile_ranks"])

    for variant_id in contract["variants"]:
        train_tiled = load_tiled_variant_input(
            phase2_output_dir,
            tile_index_csv,
            str(contract["train_tile_id"]),
            variant_id=str(variant_id),
        )
        scorer, metadata = _fit_ridge_block_scorer(
            train_tiled,
            ridge_alpha=float(contract["ridge_alpha"]),
        )
        model_metadata[str(variant_id)] = metadata

        for eval_tile_id in contract["eval_tile_ids"]:
            eval_tiled = load_tiled_variant_input(
                phase2_output_dir,
                tile_index_csv,
                str(eval_tile_id),
                variant_id=str(variant_id),
            )
            if train_tiled.state_matrix.shape[1] != eval_tiled.state_matrix.shape[1]:
                raise ValueError(
                    "Phase 22 train/evaluation per-block feature dimensions differ: "
                    f"{train_tiled.state_matrix.shape[1]} vs {eval_tiled.state_matrix.shape[1]}"
                )
            eval_tile_rank = int(eval_tile_ranks[str(eval_tile_id)])

            for seed in contract["seeds"]:
                learned_summary, learned_steps = _evaluate_learned_scorer(
                    scorer,
                    eval_tiled,
                    train_tile_id=str(contract["train_tile_id"]),
                    train_n_blocks=len(train_tiled.block_ids),
                    ridge_alpha=float(contract["ridge_alpha"]),
                    eval_max_steps=int(contract["eval_max_steps"]),
                    seed=int(seed),
                )
                summaries.append(_phase22_summary(learned_summary, eval_tile_rank))
                _store_trace(
                    traces,
                    "learned_block_scorer",
                    str(variant_id),
                    str(eval_tile_id),
                    int(seed),
                    learned_steps,
                )

                for policy_id in ("first_valid", "seeded_random"):
                    baseline_summary, baseline_steps = _evaluate_baseline_policy(
                        eval_tiled,
                        policy_id=policy_id,
                        train_tile_id=str(contract["train_tile_id"]),
                        train_n_blocks=len(train_tiled.block_ids),
                        ridge_alpha=float(contract["ridge_alpha"]),
                        eval_max_steps=int(contract["eval_max_steps"]),
                        seed=int(seed),
                    )
                    summaries.append(
                        _phase22_summary(baseline_summary, eval_tile_rank)
                    )
                    _store_trace(
                        traces,
                        policy_id,
                        str(variant_id),
                        str(eval_tile_id),
                        int(seed),
                        baseline_steps,
                    )

    return {
        **contract,
        "all_evaluations_completed": all(
            bool(row["terminated"]) or bool(row["truncated"]) for row in summaries
        ),
        "summary_count": len(summaries),
        "summaries": summaries,
        "traces": traces,
        "model_metadata": model_metadata,
    }


def write_phase22_multi_tile_scorer_eval_artifacts(
    protocol: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "phase22_multi_tile_scorer_eval_summary.csv"
    traces_path = output_path / "phase22_multi_tile_scorer_eval_traces.json"

    summaries = protocol.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("Phase 22 protocol is missing a summaries list")

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for summary in summaries:
            if not isinstance(summary, Mapping):
                raise ValueError("Phase 22 summary rows must be objects")
            row = {field: summary.get(field, "") for field in SUMMARY_FIELDNAMES}
            selected = row.get("selected_block_ids")
            if isinstance(selected, list):
                row["selected_block_ids"] = ";".join(str(item) for item in selected)
            writer.writerow(row)

    traces_path.write_text(
        json.dumps(dict(protocol), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"summary_csv": summary_path, "traces_json": traces_path}


def _select_train_eval_tiles(
    tile_index_csv: Path,
    train_tile_id: str | None,
    eval_tile_ids: Sequence[str] | str | None,
    max_eval_tiles: int,
) -> dict[str, object]:
    rows = sorted(_read_tile_rows(tile_index_csv), key=lambda row: -int(row["n_blocks"]))
    train_selection = "explicit" if train_tile_id else "largest"
    train_id = str(train_tile_id).strip() if train_tile_id else str(rows[0]["tile_id"])
    known = {str(row["tile_id"]) for row in rows}
    if train_id not in known:
        raise ValueError(f"Train tile ID not found: {train_id}")

    explicit_eval_ids = _normalize_eval_tile_ids(eval_tile_ids)
    if explicit_eval_ids:
        eval_selection = "explicit"
        selected_eval_ids = explicit_eval_ids
    else:
        if int(max_eval_tiles) <= 0:
            raise ValueError("max_eval_tiles must be positive")
        eval_selection = "largest_distinct"
        selected_eval_ids = [
            str(row["tile_id"]) for row in rows if str(row["tile_id"]) != train_id
        ][: int(max_eval_tiles)]

    if not selected_eval_ids:
        raise ValueError("Phase 22 requires at least one distinct evaluation tile")

    seen: set[str] = set()
    for eval_id in selected_eval_ids:
        if eval_id in seen:
            raise ValueError(f"Duplicate evaluation tile ID: {eval_id}")
        seen.add(eval_id)
        if eval_id not in known:
            raise ValueError(f"Evaluation tile ID not found: {eval_id}")
        if eval_id == train_id:
            raise ValueError("Phase 22 train and evaluation tiles must be distinct")

    return {
        "train_tile_id": train_id,
        "eval_tile_ids": selected_eval_ids,
        "train_tile_selection": train_selection,
        "eval_tile_selection": eval_selection,
    }


def _normalize_eval_tile_ids(
    eval_tile_ids: Sequence[str] | str | None,
) -> list[str]:
    if eval_tile_ids is None:
        return []
    if isinstance(eval_tile_ids, str):
        values = [part.strip() for part in eval_tile_ids.split(",")]
    else:
        values = [str(item).strip() for item in eval_tile_ids]
    return [item for item in values if item]


def _normalize_seeds(
    seeds: Sequence[int | str] | str | int | None,
) -> list[int]:
    if seeds is None:
        values: list[int | str] = [0, 1]
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
        raise ValueError("At least one Phase 22 seed must be requested")
    return normalized


def _phase22_summary(
    summary: Mapping[str, object],
    eval_tile_rank: int,
) -> dict[str, object]:
    row = dict(summary)
    row["eval_tile_rank"] = int(eval_tile_rank)
    row["claim_boundary"] = PHASE22_CLAIM_BOUNDARY
    return row


def _store_trace(
    traces: dict[str, dict[str, dict[str, dict[str, list[dict[str, object]]]]]],
    row_type: str,
    variant_id: str,
    eval_tile_id: str,
    seed: int,
    steps: list[dict[str, object]],
) -> None:
    variant_traces = traces[row_type].setdefault(variant_id, {})
    tile_traces = variant_traces.setdefault(eval_tile_id, {})
    tile_traces[str(int(seed))] = steps
