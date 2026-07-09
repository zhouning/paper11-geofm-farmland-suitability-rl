from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np

from paper11_geofm.phase63_set_policy_oracle_pretraining import (
    Phase63SetPolicyScorer,
    _round_float,
    build_phase63_model_inputs,
    build_phase63_oracle_trajectory,
)
from paper11_geofm.planning_reward import compute_base_planning_reward_from_matrix_row
from paper11_geofm.tiled_inputs import TiledVariantInput


PHASE70_CLAIM_BOUNDARY = (
    "Phase 70 is a standardized set-policy rerun under the existing Bishan "
    "base-reward protocol. It standardizes model inputs with train-tile-fitted "
    "parameters while preserving original features for reward and oracle "
    "scoring. It does not alter rewards, enable B2/B3, validate suitability, "
    "prove PCA optimality, or justify formal submission-level claims."
)

PHASE70_STATUS_GEOFM = "standardization_improves_geofm_set_policy_route"
PHASE70_STATUS_ARCHITECTURE = "standardization_improves_architecture_not_geofm"
PHASE70_STATUS_NOT_SUFFICIENT = "standardization_not_sufficient"
PHASE70_STATUS_INCOMPLETE = "standardized_rerun_incomplete"


@dataclass(frozen=True)
class Phase70StandardizedTiledInput:
    tiled_input: TiledVariantInput
    model_matrix: np.ndarray
    reward_matrix: np.ndarray
    standardization: Mapping[str, object]


def fit_phase70_standardization(tiled_input: TiledVariantInput) -> dict[str, object]:
    matrix = np.asarray(tiled_input.state_matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Phase 70 standardization requires a non-empty 2D matrix")
    means = np.nanmean(matrix, axis=0)
    scales = np.nanstd(matrix, axis=0)
    safe_scales = np.where(np.isfinite(scales) & (np.abs(scales) >= 1.0e-8), scales, 1.0)
    return {
        "variant_id": str(tiled_input.variant_id),
        "tile_id": str(tiled_input.tile_id),
        "feature_columns": list(tiled_input.feature_columns),
        "means": [_round_float(value) for value in means.tolist()],
        "scales": [_round_float(value) for value in safe_scales.tolist()],
        "claim_boundary": PHASE70_CLAIM_BOUNDARY,
    }


def apply_phase70_standardization(
    tiled_input: TiledVariantInput,
    params: Mapping[str, object],
) -> Phase70StandardizedTiledInput:
    feature_columns = tuple(str(value) for value in params.get("feature_columns", []))
    if feature_columns != tuple(tiled_input.feature_columns):
        raise ValueError("Phase 70 standardization feature columns do not match input")
    matrix = np.asarray(tiled_input.state_matrix, dtype=np.float32)
    means = np.asarray(params.get("means", []), dtype=np.float32)
    scales = np.asarray(params.get("scales", []), dtype=np.float32)
    if means.shape[0] != matrix.shape[1] or scales.shape[0] != matrix.shape[1]:
        raise ValueError("Phase 70 standardization parameter length does not match input")
    safe_scales = np.where(np.isfinite(scales) & (np.abs(scales) >= 1.0e-8), scales, 1.0)
    model_matrix = (matrix - means) / safe_scales
    model_matrix = np.nan_to_num(model_matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32,
        copy=False,
    )
    return Phase70StandardizedTiledInput(
        tiled_input=tiled_input,
        model_matrix=model_matrix,
        reward_matrix=matrix.astype(np.float32, copy=True),
        standardization=dict(params),
    )


def _model_view(tiled: Phase70StandardizedTiledInput) -> TiledVariantInput:
    return TiledVariantInput(
        tile_id=tiled.tiled_input.tile_id,
        variant_id=tiled.tiled_input.variant_id,
        block_ids=tiled.tiled_input.block_ids,
        feature_columns=tiled.tiled_input.feature_columns,
        state_matrix=tiled.model_matrix.astype(np.float32, copy=True),
        reward_mode=tiled.tiled_input.reward_mode,
        state_groups=tiled.tiled_input.state_groups,
        source_table=tiled.tiled_input.source_table,
        tile_index_csv=tiled.tiled_input.tile_index_csv,
        claim_boundary=tiled.tiled_input.claim_boundary,
    )


def _reward_view(tiled: Phase70StandardizedTiledInput) -> TiledVariantInput:
    return TiledVariantInput(
        tile_id=tiled.tiled_input.tile_id,
        variant_id=tiled.tiled_input.variant_id,
        block_ids=tiled.tiled_input.block_ids,
        feature_columns=tiled.tiled_input.feature_columns,
        state_matrix=tiled.reward_matrix.astype(np.float32, copy=True),
        reward_mode=tiled.tiled_input.reward_mode,
        state_groups=tiled.tiled_input.state_groups,
        source_table=tiled.tiled_input.source_table,
        tile_index_csv=tiled.tiled_input.tile_index_csv,
        claim_boundary=tiled.tiled_input.claim_boundary,
    )


def build_phase70_oracle_trajectory(
    standardized_input: Phase70StandardizedTiledInput,
    eval_max_steps: int,
) -> dict[str, object]:
    oracle = build_phase63_oracle_trajectory(_reward_view(standardized_input), eval_max_steps)
    oracle["claim_boundary"] = PHASE70_CLAIM_BOUNDARY
    oracle["phase70_standardized_input"] = True
    return oracle


def _phase70_model_inputs(
    standardized_input: Phase70StandardizedTiledInput,
    selected_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    return build_phase63_model_inputs(_model_view(standardized_input), selected_indices)


def build_phase70_bc_examples(
    standardized_input: Phase70StandardizedTiledInput,
    eval_max_steps: int,
) -> list[dict[str, object]]:
    trajectory = build_phase70_oracle_trajectory(standardized_input, eval_max_steps)
    examples: list[dict[str, object]] = []
    selected: list[int] = []
    for step in trajectory["steps"]:
        action_index = int(step["action_index"])
        inputs = _phase70_model_inputs(standardized_input, selected)
        examples.append(
            {
                "block_features": inputs["block_features"],
                "valid_mask": inputs["valid_mask"],
                "selected_mask": inputs["selected_mask"],
                "target_action": action_index,
            }
        )
        selected.append(action_index)
    return examples


def train_phase70_standardized_behavior_cloner(
    standardized_input: Phase70StandardizedTiledInput,
    seed: int,
    eval_max_steps: int,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    top_k: int = 3,
    device: str = "cpu",
):
    import random
    import torch
    import torch.nn.functional as F

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    examples = build_phase70_bc_examples(standardized_input, eval_max_steps)
    if not examples:
        raise ValueError("Phase 70 behavior cloning requires at least one example")
    model = Phase63SetPolicyScorer(
        n_features=len(standardized_input.tiled_input.feature_columns),
        hidden_dim=int(hidden_dim),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    history: list[dict[str, object]] = []
    for epoch in range(1, int(epochs) + 1):
        losses = []
        correct = 0
        topk_hits = 0
        for example in examples:
            block_features = torch.tensor(
                example["block_features"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            valid_mask = torch.tensor(
                example["valid_mask"], dtype=torch.bool, device=device
            ).unsqueeze(0)
            selected_mask = torch.tensor(
                example["selected_mask"], dtype=torch.bool, device=device
            ).unsqueeze(0)
            target = torch.tensor(
                [int(example["target_action"])], dtype=torch.long, device=device
            )
            logits = model(block_features, valid_mask, selected_mask)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            predicted = int(torch.argmax(logits.detach(), dim=1).item())
            correct += int(predicted == int(example["target_action"]))
            k = min(int(top_k), logits.shape[1])
            topk = torch.topk(logits.detach(), k=k, dim=1).indices[0].cpu().tolist()
            topk_hits += int(int(example["target_action"]) in topk)
        history.append(
            {
                "variant_id": str(standardized_input.tiled_input.variant_id),
                "train_tile_id": str(standardized_input.tiled_input.tile_id),
                "seed": int(seed),
                "epoch": int(epoch),
                "loss": _round_float(statistics.mean(losses)),
                "top1_accuracy": _round_float(correct / len(examples)),
                "topk_hit_rate": _round_float(topk_hits / len(examples)),
                "learning_rate": float(learning_rate),
                "hidden_dim": int(hidden_dim),
                "phase70_standardized_input": True,
                "claim_boundary": PHASE70_CLAIM_BOUNDARY,
            }
        )
    model.eval()
    return model, history


def rollout_phase70_standardized_greedy_policy(
    model,
    standardized_input: Phase70StandardizedTiledInput,
    train_tile_id: str,
    eval_tile_rank: int,
    seed: int,
    phase70_seed_rank: int,
    eval_max_steps: int,
    device: str = "cpu",
) -> dict[str, object]:
    import torch

    selected: list[int] = []
    selected_block_ids: list[str] = []
    rewards: list[float] = []
    invalid_action_count = 0
    for _step_index in range(min(int(eval_max_steps), len(standardized_input.tiled_input.block_ids))):
        inputs = _phase70_model_inputs(standardized_input, selected)
        available = inputs["available_mask"]
        if not bool(available.any()):
            break
        with torch.no_grad():
            logits = model(
                torch.tensor(
                    inputs["block_features"], dtype=torch.float32, device=device
                ).unsqueeze(0),
                torch.tensor(
                    inputs["valid_mask"], dtype=torch.bool, device=device
                ).unsqueeze(0),
                torch.tensor(
                    inputs["selected_mask"], dtype=torch.bool, device=device
                ).unsqueeze(0),
            )
        action = int(torch.argmax(logits, dim=1).item())
        if action in selected or not bool(available[action]):
            invalid_action_count += 1
            valid_indices = [
                int(index) for index, flag in enumerate(available.tolist()) if flag
            ]
            action = valid_indices[0]
        selected.append(action)
        selected_block_ids.append(str(standardized_input.tiled_input.block_ids[action]))
        rewards.append(
            compute_base_planning_reward_from_matrix_row(
                standardized_input.tiled_input.feature_columns,
                standardized_input.reward_matrix[action],
            )
        )
    oracle = build_phase70_oracle_trajectory(standardized_input, eval_max_steps)
    total_reward = _round_float(sum(rewards))
    oracle_total = float(oracle["total_oracle_reward"])
    oracle_gap = _round_float(oracle_total - total_reward)
    oracle_gap_fraction = _round_float(oracle_gap / max(abs(oracle_total), 1.0e-9))
    terminated = len(selected) == len(standardized_input.tiled_input.block_ids)
    return {
        "row_type": "bc_greedy_policy",
        "variant_id": str(standardized_input.tiled_input.variant_id),
        "train_tile_id": str(train_tile_id),
        "eval_tile_id": str(standardized_input.tiled_input.tile_id),
        "eval_tile_rank": int(eval_tile_rank),
        "seed": int(seed),
        "phase70_seed_rank": int(phase70_seed_rank),
        "eval_max_steps": int(eval_max_steps),
        "n_blocks": len(standardized_input.tiled_input.block_ids),
        "n_features": len(standardized_input.tiled_input.feature_columns),
        "episode_steps": len(selected),
        "terminated": bool(terminated),
        "truncated": not terminated,
        "all_actions_valid": invalid_action_count == 0,
        "invalid_action_count": int(invalid_action_count),
        "total_contract_reward": total_reward,
        "oracle_total_reward": _round_float(oracle_total),
        "oracle_gap": oracle_gap,
        "oracle_gap_fraction": oracle_gap_fraction,
        "selected_block_ids": ";".join(selected_block_ids),
        "selected_action_indices": ";".join(str(index) for index in selected),
        "phase70_standardized_input": True,
        "claim_boundary": PHASE70_CLAIM_BOUNDARY,
    }

PHASE70_PARAMETER_FIELDNAMES = (
    "variant_id",
    "tile_id",
    "feature_name",
    "mean",
    "scale",
    "claim_boundary",
)

PHASE70_ROLLOUT_FIELDNAMES = (
    "row_type",
    "variant_id",
    "train_tile_id",
    "eval_tile_id",
    "eval_tile_rank",
    "seed",
    "phase70_seed_rank",
    "eval_max_steps",
    "n_blocks",
    "n_features",
    "episode_steps",
    "terminated",
    "truncated",
    "all_actions_valid",
    "invalid_action_count",
    "total_contract_reward",
    "oracle_total_reward",
    "oracle_gap",
    "oracle_gap_fraction",
    "selected_block_ids",
    "selected_action_indices",
    "phase70_standardized_input",
    "claim_boundary",
)

PHASE70_HISTORY_FIELDNAMES = (
    "variant_id",
    "train_tile_id",
    "seed",
    "epoch",
    "loss",
    "top1_accuracy",
    "topk_hit_rate",
    "learning_rate",
    "hidden_dim",
    "phase70_standardized_input",
    "claim_boundary",
)

PHASE70_ORACLE_FIELDNAMES = (
    "variant_id",
    "tile_role",
    "tile_id",
    "seed",
    "eval_max_steps",
    "n_blocks",
    "n_features",
    "episode_steps",
    "terminated",
    "total_oracle_reward",
    "top_k_reward_ceiling",
    "selected_block_ids",
    "action_indices",
    "claim_boundary",
)

PHASE70_DELTA_FIELDNAMES = (
    "variant_id",
    "eval_tile_id",
    "seed",
    "phase70_reward",
    "phase63_reward",
    "phase70_minus_phase63_reward",
    "phase70_oracle_gap_fraction",
    "phase63_oracle_gap_fraction",
    "claim_boundary",
)


def build_phase70_standardization_parameter_rows(
    params_by_variant: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant_id in sorted(params_by_variant):
        params = params_by_variant[variant_id]
        columns = list(params.get("feature_columns", []))
        means = list(params.get("means", []))
        scales = list(params.get("scales", []))
        for feature_name, mean, scale in zip(columns, means, scales):
            rows.append(
                {
                    "variant_id": variant_id,
                    "tile_id": params.get("tile_id", ""),
                    "feature_name": feature_name,
                    "mean": mean,
                    "scale": scale,
                    "claim_boundary": PHASE70_CLAIM_BOUNDARY,
                }
            )
    return rows


def _rollout_key(row: Mapping[str, object]) -> tuple[str, str, int]:
    return (
        str(row.get("variant_id", "")),
        str(row.get("eval_tile_id", "")),
        int(row.get("seed", 0)),
    )


def _coverage_issues(
    rows: Sequence[Mapping[str, object]],
    variants: Sequence[str],
    eval_tile_ids: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, list[dict[str, object]]]:
    counts: dict[tuple[str, str, int], int] = {}
    for row in rows:
        if str(row.get("row_type", "")) != "bc_greedy_policy":
            continue
        key = _rollout_key(row)
        counts[key] = counts.get(key, 0) + 1
    missing = []
    duplicate = []
    expected = {
        (str(variant), str(tile_id), int(seed))
        for variant in variants
        for tile_id in eval_tile_ids
        for seed in seeds
    }
    for variant_id, tile_id, seed in sorted(expected):
        count = counts.get((variant_id, tile_id, seed), 0)
        if count == 0:
            missing.append({"variant_id": variant_id, "eval_tile_id": tile_id, "seed": seed})
        elif count > 1:
            duplicate.append(
                {
                    "variant_id": variant_id,
                    "eval_tile_id": tile_id,
                    "seed": seed,
                    "count": count,
                }
            )
    unexpected = [
        {"variant_id": key[0], "eval_tile_id": key[1], "seed": key[2], "count": count}
        for key, count in sorted(counts.items())
        if key not in expected
    ]
    return {
        "missing_rollout_rows": missing,
        "duplicate_rollout_rows": duplicate,
        "unexpected_rollout_rows": unexpected,
    }


def _mean_by_variant(rows: Sequence[Mapping[str, object]], value_field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        variant_id = str(row.get("variant_id", ""))
        value = row.get(value_field, "")
        if not variant_id or str(value).strip() == "":
            continue
        grouped.setdefault(variant_id, []).append(float(value))
    return {key: _round_float(statistics.mean(values)) for key, values in sorted(grouped.items())}


def _numeric_summary(values: Sequence[float]) -> dict[str, object]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "mean_delta": 0.0,
            "positive_count": 0,
            "total_count": 0,
            "min_delta": 0.0,
            "max_delta": 0.0,
        }
    return {
        "mean_delta": _round_float(statistics.mean(numbers)),
        "positive_count": sum(1 for value in numbers if value > 0.0),
        "total_count": len(numbers),
        "min_delta": _round_float(min(numbers)),
        "max_delta": _round_float(max(numbers)),
    }


def _paired_delta_summary(
    rows: Sequence[Mapping[str, object]],
    pairs: Sequence[tuple[str, str]],
) -> dict[str, object]:
    index = {
        _rollout_key(row): float(row.get("total_contract_reward", 0.0))
        for row in rows
        if str(row.get("row_type", "")) == "bc_greedy_policy"
    }
    values = []
    tile_seed_keys = sorted({(key[1], key[2]) for key in index})
    for left, right in pairs:
        for tile_id, seed in tile_seed_keys:
            left_key = (left, tile_id, seed)
            right_key = (right, tile_id, seed)
            if left_key in index and right_key in index:
                values.append(index[left_key] - index[right_key])
    return _numeric_summary(values)


def _delta_rows(
    phase70_rows: Sequence[Mapping[str, object]],
    phase63_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    baseline = {
        _rollout_key(row): row
        for row in phase63_rows
        if str(row.get("row_type", "")) == "bc_greedy_policy"
    }
    rows = []
    for row in phase70_rows:
        if str(row.get("row_type", "")) != "bc_greedy_policy":
            continue
        key = _rollout_key(row)
        old = baseline.get(key)
        if old is None:
            continue
        phase70_reward = float(row.get("total_contract_reward", 0.0))
        phase63_reward = float(old.get("total_contract_reward", 0.0))
        rows.append(
            {
                "variant_id": key[0],
                "eval_tile_id": key[1],
                "seed": key[2],
                "phase70_reward": _round_float(phase70_reward),
                "phase63_reward": _round_float(phase63_reward),
                "phase70_minus_phase63_reward": _round_float(phase70_reward - phase63_reward),
                "phase70_oracle_gap_fraction": _round_float(row.get("oracle_gap_fraction", 0.0)),
                "phase63_oracle_gap_fraction": _round_float(old.get("oracle_gap_fraction", 0.0)),
                "claim_boundary": PHASE70_CLAIM_BOUNDARY,
            }
        )
    return rows


def _phase70_status(
    coverage: Mapping[str, object],
    delta_summary: Mapping[str, object],
    d4_b0_summary: Mapping[str, object],
    d4_d6_summary: Mapping[str, object],
) -> str:
    if coverage["missing_rollout_rows"] or coverage["duplicate_rollout_rows"]:
        return PHASE70_STATUS_INCOMPLETE
    architecture_improved = (
        float(delta_summary["mean_delta"]) > 0.0
        and int(delta_summary["positive_count"]) * 2 >= int(delta_summary["total_count"])
    )
    geofm_improved = (
        float(d4_b0_summary["mean_delta"]) > 0.0
        and int(d4_b0_summary["positive_count"]) * 2 >= int(d4_b0_summary["total_count"])
        and float(d4_d6_summary["mean_delta"]) > 0.0
        and int(d4_d6_summary["positive_count"]) * 2 >= int(d4_d6_summary["total_count"])
    )
    if architecture_improved and geofm_improved:
        return PHASE70_STATUS_GEOFM
    if architecture_improved:
        return PHASE70_STATUS_ARCHITECTURE
    return PHASE70_STATUS_NOT_SUFFICIENT


def build_phase70_standardized_set_policy_comparison(
    phase70_rollout_rows: Sequence[Mapping[str, object]],
    phase63_rollout_rows: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    metadata = dict(metadata or {})
    variants = [str(value) for value in metadata.get("variants", [])] or sorted(
        _mean_by_variant(phase70_rollout_rows, "total_contract_reward")
    )
    eval_tile_ids = [str(value) for value in metadata.get("eval_tile_ids", [])] or sorted(
        {str(row.get("eval_tile_id", "")) for row in phase70_rollout_rows}
    )
    seeds = [int(value) for value in metadata.get("seeds", [])] or sorted(
        {int(row.get("seed", 0)) for row in phase70_rollout_rows}
    )
    coverage = _coverage_issues(phase70_rollout_rows, variants, eval_tile_ids, seeds)
    delta_table = _delta_rows(phase70_rollout_rows, phase63_rollout_rows)
    delta_summary = _numeric_summary(
        [float(row["phase70_minus_phase63_reward"]) for row in delta_table]
    )
    d4_b0 = _paired_delta_summary(phase70_rollout_rows, (("D4P8", "B0"), ("D4P16", "B0")))
    d4_d6 = _paired_delta_summary(phase70_rollout_rows, (("D4P8", "D6R8"), ("D4P16", "D6R16")))
    status = _phase70_status(coverage, delta_summary, d4_b0, d4_d6)
    return {
        "phase": "phase70_standardized_set_policy_rerun",
        "phase70_status": status,
        "variants": variants,
        "eval_tile_ids": eval_tile_ids,
        "seeds": seeds,
        "coverage_issues": coverage,
        "mean_standardized_reward_by_variant": _mean_by_variant(
            phase70_rollout_rows, "total_contract_reward"
        ),
        "mean_phase70_minus_phase63_by_variant": _mean_by_variant(
            delta_table, "phase70_minus_phase63_reward"
        ),
        "standardized_minus_phase63_summary": delta_summary,
        "d4_b0_delta_summary": d4_b0,
        "d4_d6_delta_summary": d4_d6,
        "delta_rows": delta_table,
        "recommended_next_step": _phase70_next_step(status),
        "claim_boundary": PHASE70_CLAIM_BOUNDARY,
    }


def _phase70_next_step(status: str) -> str:
    if status == PHASE70_STATUS_GEOFM:
        return "Use Phase 70 as the next bounded algorithm evidence gate; still do not broaden suitability claims without independent labels."
    if status == PHASE70_STATUS_ARCHITECTURE:
        return "Keep the set-policy architecture route, but do not claim GeoFM-specific advantage from standardized features."
    if status == PHASE70_STATUS_INCOMPLETE:
        return "Repair missing standardized rerun coverage before interpreting Phase 70."
    return "Treat standardization as insufficient and design a different algorithm route."


def write_phase70_standardized_set_policy_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "standardization_parameters_csv": output_path / "phase70_standardization_parameters.csv",
        "history_csv": output_path / "phase70_standardized_bc_training_history.csv",
        "rollout_csv": output_path / "phase70_standardized_bc_rollout_summary.csv",
        "oracle_summary_csv": output_path / "phase70_standardized_oracle_summary.csv",
        "delta_csv": output_path / "phase70_standardized_delta_table.csv",
        "comparison_json": output_path / "phase70_standardized_set_policy_comparison.json",
        "readiness_md": output_path / "phase70_standardized_set_policy_rerun.md",
    }
    _write_csv_rows(paths["standardization_parameters_csv"], PHASE70_PARAMETER_FIELDNAMES, analysis.get("standardization_parameter_rows", []))
    _write_csv_rows(paths["history_csv"], PHASE70_HISTORY_FIELDNAMES, analysis.get("history_rows", []))
    _write_csv_rows(paths["rollout_csv"], PHASE70_ROLLOUT_FIELDNAMES, analysis.get("rollout_rows", []))
    _write_csv_rows(paths["oracle_summary_csv"], PHASE70_ORACLE_FIELDNAMES, analysis.get("oracle_summary_rows", []))
    _write_csv_rows(paths["delta_csv"], PHASE70_DELTA_FIELDNAMES, analysis.get("delta_rows", []))
    comparison = {
        key: value
        for key, value in analysis.items()
        if key not in {"standardization_parameter_rows", "history_rows", "rollout_rows", "oracle_summary_rows"}
    }
    paths["comparison_json"].write_text(
        json.dumps(_json_ready(comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["readiness_md"].write_text(_phase70_markdown(analysis), encoding="utf-8")
    return paths


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: object) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 70 rows must be a list for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 70 CSV rows must be objects for {path.name}")
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _phase70_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Phase 70 Standardized Set-Policy Rerun",
        "",
        f"Status: {analysis.get('phase70_status', '')}",
        "",
        "Mean standardized reward by variant:",
    ]
    for variant_id, value in dict(analysis.get("mean_standardized_reward_by_variant", {})).items():
        lines.append(f"- {variant_id}: {value}")
    lines.extend(
        [
            "",
            f"Phase 70 minus Phase 63 summary: {analysis.get('standardized_minus_phase63_summary', {})}",
            f"D4/B0 delta summary: {analysis.get('d4_b0_delta_summary', {})}",
            f"D4/D6 delta summary: {analysis.get('d4_d6_delta_summary', {})}",
            "",
            "Recommended next step:",
            str(analysis.get("recommended_next_step", "")),
            "",
            "Claim boundary:",
            str(analysis.get("claim_boundary", PHASE70_CLAIM_BOUNDARY)),
            "",
        ]
    )
    return "\n".join(lines)