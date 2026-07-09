from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
import statistics

import numpy as np
import torch
from torch.nn import functional as F

from .phase63_set_policy_oracle_pretraining import (
    PHASE63_D4_B0_COMPARISONS,
    PHASE63_D4_D6_COMPARISONS,
    PHASE63_DELTA_FIELDNAMES,
    PHASE63_HISTORY_FIELDNAMES,
    PHASE63_ROLLOUT_FIELDNAMES,
    Phase63SetPolicyScorer,
    build_phase63_model_inputs,
    build_phase63_oracle_trajectory,
    build_phase63_set_policy_analysis,
)
from .planning_reward import compute_base_planning_reward_from_matrix_row
from .tiled_inputs import load_tiled_variant_input


PHASE65_CLAIM_BOUNDARY = (
    "Phase 65 is a base-reward train-tile-fitted standardized set-policy "
    "behavior-cloning rerun. Standardization is applied only to policy model "
    "inputs; oracle targets and rollout rewards remain computed from raw "
    "unstandardized feature matrices. It does not enable suitability reward, "
    "does not test B2/B3, does not test transfer, does not prove GeoFM "
    "advantage or PCA optimality, and does not justify formal submission-level "
    "claims."
)

PHASE65_STATUS_GEOFM = "standardization_improves_geofm_set_policy"
PHASE65_STATUS_ALL_VARIANTS = "standardization_improves_all_variants_no_geofm_advantage"
PHASE65_STATUS_NOT_HELPFUL = "standardization_not_helpful"
PHASE65_STATUS_INCONCLUSIVE = "standardization_hurts_or_inconclusive"
PHASE65_STATUS_INSUFFICIENT = "insufficient"

PHASE65_EPSILON = 1.0e-12


@dataclass(frozen=True)
class Phase65Standardizer:
    variant_id: str
    train_tile_id: str
    feature_columns: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    safe_std: np.ndarray
    zero_variance_feature_count: int
    epsilon: float = PHASE65_EPSILON

    def transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("Phase 65 standardizer expects a 2-D state matrix")
        if values.shape[1] != len(self.feature_columns):
            raise ValueError("Phase 65 state matrix feature count does not match transform")
        return ((values - self.mean) / self.safe_std).astype(np.float32, copy=True)

    def to_json_row(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "train_tile_id": self.train_tile_id,
            "n_features": len(self.feature_columns),
            "zero_variance_feature_count": int(self.zero_variance_feature_count),
            "epsilon": float(self.epsilon),
            "feature_columns": list(self.feature_columns),
            "mean": [round(float(value), 10) for value in self.mean.tolist()],
            "std": [round(float(value), 10) for value in self.std.tolist()],
            "safe_std": [round(float(value), 10) for value in self.safe_std.tolist()],
            "claim_boundary": PHASE65_CLAIM_BOUNDARY,
        }


def fit_phase65_train_tile_standardizer(
    tiled_input,
    epsilon: float = PHASE65_EPSILON,
) -> Phase65Standardizer:
    matrix = np.asarray(tiled_input.state_matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Phase 65 train tile state matrix must be 2-D")
    if matrix.shape[0] <= 0:
        raise ValueError("Phase 65 train tile has no blocks")
    if matrix.shape[1] <= 0:
        raise ValueError("Phase 65 train tile has no feature columns")
    mean = np.mean(matrix, axis=0).astype(np.float32)
    std = np.std(matrix, axis=0, ddof=0).astype(np.float32)
    safe_std = np.where(std > float(epsilon), std, 1.0).astype(np.float32)
    return Phase65Standardizer(
        variant_id=str(tiled_input.variant_id),
        train_tile_id=str(tiled_input.tile_id),
        feature_columns=tuple(str(column) for column in tiled_input.feature_columns),
        mean=mean,
        std=std,
        safe_std=safe_std,
        zero_variance_feature_count=int(np.sum(std <= float(epsilon))),
        epsilon=float(epsilon),
    )


def apply_phase65_standardizer(tiled_input, standardizer: Phase65Standardizer):
    if str(tiled_input.variant_id) != standardizer.variant_id:
        raise ValueError(
            "Phase 65 standardizer variant mismatch: "
            f"{tiled_input.variant_id} != {standardizer.variant_id}"
        )
    if tuple(tiled_input.feature_columns) != standardizer.feature_columns:
        raise ValueError("Phase 65 standardizer feature columns do not match tiled input")
    standardized = standardizer.transform_matrix(tiled_input.state_matrix)
    return replace(
        tiled_input,
        state_matrix=standardized,
        claim_boundary=PHASE65_CLAIM_BOUNDARY,
    )


def _validate_aligned_tiled_inputs(raw_tiled, standardized_tiled) -> None:
    if tuple(raw_tiled.block_ids) != tuple(standardized_tiled.block_ids):
        raise ValueError("Phase 65 raw and standardized block IDs are not aligned")
    if tuple(raw_tiled.feature_columns) != tuple(standardized_tiled.feature_columns):
        raise ValueError("Phase 65 raw and standardized feature columns are not aligned")
    if str(raw_tiled.tile_id) != str(standardized_tiled.tile_id):
        raise ValueError("Phase 65 raw and standardized tile IDs are not aligned")
    if str(raw_tiled.variant_id) != str(standardized_tiled.variant_id):
        raise ValueError("Phase 65 raw and standardized variant IDs are not aligned")


def build_phase65_bc_examples(
    raw_tiled_input,
    standardizer: Phase65Standardizer,
    eval_max_steps: int,
) -> list[dict[str, object]]:
    standardized_tiled = apply_phase65_standardizer(raw_tiled_input, standardizer)
    _validate_aligned_tiled_inputs(raw_tiled_input, standardized_tiled)
    trajectory = build_phase63_oracle_trajectory(raw_tiled_input, eval_max_steps)
    examples: list[dict[str, object]] = []
    selected: list[int] = []
    for step in trajectory["steps"]:
        action_index = int(step["action_index"])
        inputs = build_phase63_model_inputs(standardized_tiled, selected)
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


def _round_float(value: object, digits: int = 10) -> float:
    rounded = round(float(value), digits)
    compact = round(rounded, 6)
    if abs(rounded - compact) < 5.0e-8:
        return compact
    return rounded


def train_phase65_behavior_cloner(
    raw_tiled_input,
    standardizer: Phase65Standardizer,
    seed: int,
    eval_max_steps: int,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    top_k: int = 3,
    device: str = "cpu",
) -> tuple[Phase63SetPolicyScorer, list[dict[str, object]]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    examples = build_phase65_bc_examples(raw_tiled_input, standardizer, eval_max_steps)
    if not examples:
        raise ValueError("Phase 65 behavior cloning requires at least one example")
    model = Phase63SetPolicyScorer(
        n_features=len(raw_tiled_input.feature_columns),
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
                example["block_features"],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            valid_mask = torch.tensor(
                example["valid_mask"],
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0)
            selected_mask = torch.tensor(
                example["selected_mask"],
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0)
            target = torch.tensor(
                [int(example["target_action"])],
                dtype=torch.long,
                device=device,
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
                "variant_id": str(raw_tiled_input.variant_id),
                "train_tile_id": str(raw_tiled_input.tile_id),
                "seed": int(seed),
                "epoch": int(epoch),
                "loss": _round_float(statistics.mean(losses)),
                "top1_accuracy": _round_float(correct / len(examples)),
                "topk_hit_rate": _round_float(topk_hits / len(examples)),
                "learning_rate": float(learning_rate),
                "hidden_dim": int(hidden_dim),
                "claim_boundary": PHASE65_CLAIM_BOUNDARY,
            }
        )
    model.eval()
    return model, history


def rollout_phase65_greedy_policy(
    model: Phase63SetPolicyScorer,
    raw_tiled_input,
    standardizer: Phase65Standardizer,
    train_tile_id: str,
    eval_tile_rank: int,
    seed: int,
    phase65_seed_rank: int,
    eval_max_steps: int,
    device: str = "cpu",
) -> dict[str, object]:
    standardized_tiled = apply_phase65_standardizer(raw_tiled_input, standardizer)
    _validate_aligned_tiled_inputs(raw_tiled_input, standardized_tiled)
    selected: list[int] = []
    selected_block_ids: list[str] = []
    rewards: list[float] = []
    invalid_action_count = 0
    for _step_index in range(min(int(eval_max_steps), len(raw_tiled_input.block_ids))):
        inputs = build_phase63_model_inputs(standardized_tiled, selected)
        available = inputs["available_mask"]
        if not bool(available.any()):
            break
        with torch.no_grad():
            logits = model(
                torch.tensor(
                    inputs["block_features"],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0),
                torch.tensor(
                    inputs["valid_mask"],
                    dtype=torch.bool,
                    device=device,
                ).unsqueeze(0),
                torch.tensor(
                    inputs["selected_mask"],
                    dtype=torch.bool,
                    device=device,
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
        selected_block_ids.append(str(raw_tiled_input.block_ids[action]))
        rewards.append(
            compute_base_planning_reward_from_matrix_row(
                raw_tiled_input.feature_columns,
                raw_tiled_input.state_matrix[action],
            )
        )
    oracle = build_phase63_oracle_trajectory(raw_tiled_input, eval_max_steps)
    total_reward = _round_float(sum(rewards))
    oracle_total = float(oracle["total_oracle_reward"])
    oracle_gap = _round_float(oracle_total - total_reward)
    oracle_gap_fraction = _round_float(oracle_gap / max(abs(oracle_total), 1.0e-9))
    terminated = len(selected) == len(raw_tiled_input.block_ids)
    return {
        "row_type": "bc_greedy_policy",
        "variant_id": str(raw_tiled_input.variant_id),
        "train_tile_id": str(train_tile_id),
        "eval_tile_id": str(raw_tiled_input.tile_id),
        "eval_tile_rank": int(eval_tile_rank),
        "seed": int(seed),
        "phase63_seed_rank": int(phase65_seed_rank),
        "eval_max_steps": int(eval_max_steps),
        "n_blocks": len(raw_tiled_input.block_ids),
        "n_features": len(raw_tiled_input.feature_columns),
        "episode_steps": len(selected),
        "terminated": bool(terminated),
        "truncated": bool(not terminated and len(selected) >= int(eval_max_steps)),
        "all_actions_valid": bool(invalid_action_count == 0),
        "invalid_action_count": int(invalid_action_count),
        "total_contract_reward": total_reward,
        "oracle_total_reward": _round_float(oracle_total),
        "oracle_gap": oracle_gap,
        "oracle_gap_fraction": oracle_gap_fraction,
        "selected_block_ids": ";".join(selected_block_ids),
        "selected_action_indices": ";".join(str(index) for index in selected),
        "claim_boundary": PHASE65_CLAIM_BOUNDARY,
    }


PHASE65_PAIRWISE_FIELDNAMES = [
    "variant_id",
    "eval_tile_id",
    "seed",
    "standardized_bc_reward",
    "unstandardized_bc_reward",
    "standardized_minus_unstandardized_reward",
    "standardized_oracle_gap_fraction",
    "unstandardized_oracle_gap_fraction",
    "standardized_minus_unstandardized_oracle_gap_fraction",
    "self_improves_unstandardized",
    "claim_boundary",
]


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def _safe_int(value: object, default: int = 0) -> int:
    if value is None or str(value).strip() == "":
        return int(default)
    return int(float(value))


def _rollout_key(row: Mapping[str, object]) -> tuple[str, str, int]:
    return (
        str(row.get("variant_id", "")),
        str(row.get("eval_tile_id", "")),
        _safe_int(row.get("seed")),
    )


def _index_rollout_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[tuple[str, str, int], Mapping[str, object]], list[dict[str, object]]]:
    index: dict[tuple[str, str, int], Mapping[str, object]] = {}
    duplicates: list[dict[str, object]] = []
    for row in rows:
        if str(row.get("row_type", "")) != "bc_greedy_policy":
            continue
        key = _rollout_key(row)
        if key in index:
            duplicates.append(
                {"variant_id": key[0], "eval_tile_id": key[1], "seed": key[2]}
            )
        index[key] = row
    return index, duplicates


def build_phase65_standardization_pairwise_rows(
    standardized_rows: Sequence[Mapping[str, object]],
    unstandardized_rows: Sequence[Mapping[str, object]],
    variants: Sequence[str],
    eval_tile_ids: Sequence[str],
    seeds: Sequence[int],
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    standardized_index, standardized_duplicates = _index_rollout_rows(standardized_rows)
    unstandardized_index, unstandardized_duplicates = _index_rollout_rows(unstandardized_rows)
    rows: list[dict[str, object]] = []
    missing_standardized = []
    missing_unstandardized = []
    expected = {
        (str(variant), str(tile_id), int(seed))
        for variant in variants
        for tile_id in eval_tile_ids
        for seed in seeds
    }
    for key in sorted(expected):
        standardized = standardized_index.get(key)
        unstandardized = unstandardized_index.get(key)
        if standardized is None:
            missing_standardized.append(
                {"variant_id": key[0], "eval_tile_id": key[1], "seed": key[2]}
            )
            continue
        if unstandardized is None:
            missing_unstandardized.append(
                {"variant_id": key[0], "eval_tile_id": key[1], "seed": key[2]}
            )
            continue
        standardized_reward = _safe_float(standardized.get("total_contract_reward"))
        unstandardized_reward = _safe_float(unstandardized.get("total_contract_reward"))
        standardized_gap = _safe_float(standardized.get("oracle_gap_fraction"))
        unstandardized_gap = _safe_float(unstandardized.get("oracle_gap_fraction"))
        reward_delta = _round_float(standardized_reward - unstandardized_reward)
        rows.append(
            {
                "variant_id": key[0],
                "eval_tile_id": key[1],
                "seed": key[2],
                "standardized_bc_reward": _round_float(standardized_reward),
                "unstandardized_bc_reward": _round_float(unstandardized_reward),
                "standardized_minus_unstandardized_reward": reward_delta,
                "standardized_oracle_gap_fraction": _round_float(standardized_gap),
                "unstandardized_oracle_gap_fraction": _round_float(unstandardized_gap),
                "standardized_minus_unstandardized_oracle_gap_fraction": _round_float(
                    standardized_gap - unstandardized_gap
                ),
                "self_improves_unstandardized": bool(reward_delta > 0.0),
                "claim_boundary": PHASE65_CLAIM_BOUNDARY,
            }
        )
    unexpected_standardized = [
        {"variant_id": key[0], "eval_tile_id": key[1], "seed": key[2]}
        for key in sorted(standardized_index)
        if key not in expected
    ]
    unexpected_unstandardized = [
        {"variant_id": key[0], "eval_tile_id": key[1], "seed": key[2]}
        for key in sorted(unstandardized_index)
        if key not in expected
    ]
    return rows, {
        "missing_standardized_rows": missing_standardized,
        "missing_unstandardized_rows": missing_unstandardized,
        "duplicate_standardized_rows": standardized_duplicates,
        "duplicate_unstandardized_rows": unstandardized_duplicates,
        "unexpected_standardized_rows": unexpected_standardized,
        "unexpected_unstandardized_rows": unexpected_unstandardized,
    }


def _numeric_delta_summary(values: Sequence[float]) -> dict[str, object]:
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


def _paired_variant_delta_rows(
    rows: Sequence[Mapping[str, object]],
    comparisons: Sequence[tuple[str, str]],
    value_field: str,
    output_field: str,
) -> list[dict[str, object]]:
    index = {
        (
            str(row.get("variant_id", "")),
            str(row.get("eval_tile_id", "")),
            _safe_int(row.get("seed")),
        ): _safe_float(row.get(value_field))
        for row in rows
        if str(row.get("row_type", "")) == "bc_greedy_policy"
    }
    output = []
    tile_seed_keys = sorted({(key[1], key[2]) for key in index})
    for left, right in comparisons:
        for tile_id, seed in tile_seed_keys:
            left_key = (left, tile_id, seed)
            right_key = (right, tile_id, seed)
            if left_key not in index or right_key not in index:
                continue
            delta = _round_float(index[left_key] - index[right_key])
            output.append(
                {
                    "left_variant_id": left,
                    "right_variant_id": right,
                    "eval_tile_id": tile_id,
                    "seed": seed,
                    output_field: delta,
                    "left_improves_right": bool(delta > 0.0),
                    "claim_boundary": PHASE65_CLAIM_BOUNDARY,
                }
            )
    return output


def _coverage_has_issues(coverage: Mapping[str, object]) -> bool:
    return any(bool(value) for value in coverage.values())


def _phase65_status(
    coverage: Mapping[str, object],
    overall_summary: Mapping[str, object],
    d4_self_summary: Mapping[str, object],
    d4_b0_summary: Mapping[str, object],
    d4_d6_summary: Mapping[str, object],
) -> str:
    if _coverage_has_issues(coverage):
        return PHASE65_STATUS_INSUFFICIENT
    overall_positive = float(overall_summary["mean_delta"]) > 0.0
    d4_self_positive = float(d4_self_summary["mean_delta"]) > 0.0
    d4_b0_positive = float(d4_b0_summary["mean_delta"]) > 0.0
    d4_d6_positive = float(d4_d6_summary["mean_delta"]) > 0.0
    if d4_self_positive and d4_b0_positive and d4_d6_positive:
        return PHASE65_STATUS_GEOFM
    if overall_positive:
        return PHASE65_STATUS_ALL_VARIANTS
    if not overall_positive and (not d4_b0_positive or not d4_d6_positive):
        return PHASE65_STATUS_NOT_HELPFUL
    return PHASE65_STATUS_INCONCLUSIVE


def build_phase65_standardization_comparison(
    standardized_rows: Sequence[Mapping[str, object]],
    unstandardized_rows: Sequence[Mapping[str, object]],
    variants: Sequence[str],
    eval_tile_ids: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, object]:
    pairwise_rows, coverage = build_phase65_standardization_pairwise_rows(
        standardized_rows,
        unstandardized_rows,
        variants=variants,
        eval_tile_ids=eval_tile_ids,
        seeds=seeds,
    )
    overall = _numeric_delta_summary(
        [float(row["standardized_minus_unstandardized_reward"]) for row in pairwise_rows]
    )
    d4_self = _numeric_delta_summary(
        [
            float(row["standardized_minus_unstandardized_reward"])
            for row in pairwise_rows
            if str(row["variant_id"]).startswith("D4")
        ]
    )
    d4_b0_rows = _paired_variant_delta_rows(
        standardized_rows,
        PHASE63_D4_B0_COMPARISONS,
        value_field="total_contract_reward",
        output_field="left_minus_right_reward",
    )
    d4_d6_rows = _paired_variant_delta_rows(
        standardized_rows,
        PHASE63_D4_D6_COMPARISONS,
        value_field="total_contract_reward",
        output_field="left_minus_right_reward",
    )
    d4_b0 = _numeric_delta_summary(
        [float(row["left_minus_right_reward"]) for row in d4_b0_rows]
    )
    d4_d6 = _numeric_delta_summary(
        [float(row["left_minus_right_reward"]) for row in d4_d6_rows]
    )
    status = _phase65_status(coverage, overall, d4_self, d4_b0, d4_d6)
    return {
        "phase": "phase65_standardization_comparison",
        "phase65_status": status,
        "pairwise_delta_rows": pairwise_rows,
        "coverage_issues": coverage,
        "overall_standardized_minus_unstandardized_summary": overall,
        "d4_standardized_minus_unstandardized_summary": d4_self,
        "d4_b0_delta_rows": d4_b0_rows,
        "d4_d6_delta_rows": d4_d6_rows,
        "d4_b0_delta_summary": d4_b0,
        "d4_d6_delta_summary": d4_d6,
        "claim_boundary": PHASE65_CLAIM_BOUNDARY,
    }


def _write_csv_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


def _phase65_markdown(analysis: Mapping[str, object]) -> str:
    comparison = dict(analysis.get("standardization_comparison", {}))
    phase63_style = dict(analysis.get("phase63_style_analysis", {}))
    lines = [
        "# Phase 65 Standardized Set-Policy BC Rerun",
        "",
        f"Status: {comparison.get('phase65_status', '')}",
        "",
        "Mean standardized BC reward by variant:",
    ]
    for variant_id, value in dict(phase63_style.get("mean_bc_reward_by_variant", {})).items():
        lines.append(f"- {variant_id}: {value}")
    lines.extend(
        [
            "",
            f"Overall standardized-minus-unstandardized summary: {comparison.get('overall_standardized_minus_unstandardized_summary', {})}",
            f"D4 standardized-minus-unstandardized summary: {comparison.get('d4_standardized_minus_unstandardized_summary', {})}",
            f"D4/B0 delta summary after standardization: {comparison.get('d4_b0_delta_summary', {})}",
            f"D4/D6 delta summary after standardization: {comparison.get('d4_d6_delta_summary', {})}",
            f"Oracle gap summary after standardization: {phase63_style.get('oracle_gap_fraction_summary', {})}",
            "",
            "Claim boundary:",
            str(analysis.get("claim_boundary", PHASE65_CLAIM_BOUNDARY)),
            "",
        ]
    )
    return "\n".join(lines)


def write_phase65_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "standardization_stats_json": output_path / "phase65_standardization_stats.json",
        "history_csv": output_path / "phase65_bc_training_history.csv",
        "rollout_csv": output_path / "phase65_bc_rollout_summary.csv",
        "comparison_json": output_path / "phase65_set_policy_comparison.json",
        "pairwise_delta_csv": output_path / "phase65_standardization_pairwise_delta.csv",
        "readiness_md": output_path / "phase65_standardized_set_policy_bc_rerun.md",
    }
    paths["standardization_stats_json"].write_text(
        json.dumps(
            _json_ready(analysis.get("standardization_stats", [])),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_csv_rows(paths["history_csv"], PHASE63_HISTORY_FIELDNAMES, analysis.get("history_rows", []))
    _write_csv_rows(paths["rollout_csv"], PHASE63_ROLLOUT_FIELDNAMES, analysis.get("rollout_rows", []))
    pairwise_rows = dict(analysis.get("standardization_comparison", {})).get(
        "pairwise_delta_rows",
        [],
    )
    _write_csv_rows(paths["pairwise_delta_csv"], PHASE65_PAIRWISE_FIELDNAMES, pairwise_rows)
    comparison = {
        key: value
        for key, value in dict(analysis).items()
        if key not in {"history_rows", "rollout_rows"}
    }
    status = dict(analysis.get("standardization_comparison", {})).get(
        "phase65_status",
        PHASE65_STATUS_INSUFFICIENT,
    )
    comparison["phase65_status"] = status
    paths["comparison_json"].write_text(
        json.dumps(_json_ready(comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["readiness_md"].write_text(_phase65_markdown(analysis), encoding="utf-8")
    return paths


def _load_json_object(path: Path | str, label: str) -> dict[str, object]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {label}: {json_path}")
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return loaded


def _load_csv_rows(path: Path | str, label: str) -> list[dict[str, object]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {label}: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _normalize_optional_paths(paths: Sequence[Path | str] | str | None) -> list[Path | str]:
    if paths is None:
        return []
    if isinstance(paths, str):
        return [part.strip() for part in paths.split(",") if part.strip()]
    return [path for path in paths if str(path).strip()]


def _contract_string_list(contract: Mapping[str, object], key: str) -> list[str]:
    value = contract.get(key)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item).strip()]
    return []


def _contract_int_list(contract: Mapping[str, object], key: str) -> list[int]:
    value = contract.get(key)
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, Sequence):
        return [int(item) for item in value if str(item).strip()]
    return []


def _load_phase65_tiled_variant_input(
    contract: Mapping[str, object],
    tile_id: str,
    variant_id: str,
):
    variant_source_dirs = contract.get("variant_source_dirs")
    if not isinstance(variant_source_dirs, Mapping):
        raise ValueError("Phase 65 contract is missing variant_source_dirs")
    source_dir = variant_source_dirs.get(variant_id)
    if source_dir is None:
        raise ValueError(f"Phase 65 contract has no source for variant {variant_id}")
    return load_tiled_variant_input(
        source_dir,
        str(contract["tile_index_csv"]),
        tile_id,
        variant_id=variant_id,
    )


def run_phase65_standardized_set_policy_bc_rerun(
    phase63_comparison_json: Path | str,
    phase63_rollout_csv: Path | str,
    existing_flattened_summary_csvs: Sequence[Path | str] | str | None = None,
) -> dict[str, object]:
    phase63_comparison = _load_json_object(
        phase63_comparison_json,
        "Phase 63 comparison JSON",
    )
    contract = phase63_comparison.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Phase 63 comparison JSON is missing contract metadata")
    unstandardized_rows = _load_csv_rows(phase63_rollout_csv, "Phase 63 rollout CSV")
    variants = _contract_string_list(contract, "variants")
    eval_tile_ids = _contract_string_list(contract, "eval_tile_ids")
    seeds = _contract_int_list(contract, "seeds")
    train_tile_id = str(contract.get("train_tile_id", ""))
    if not variants:
        raise ValueError("Phase 65 contract has no variants")
    if not eval_tile_ids:
        raise ValueError("Phase 65 contract has no eval_tile_ids")
    if not seeds:
        raise ValueError("Phase 65 contract has no seeds")
    if not train_tile_id:
        raise ValueError("Phase 65 contract has no train_tile_id")
    eval_tile_ranks = {
        str(tile_id): int(rank)
        for tile_id, rank in dict(contract.get("eval_tile_ranks", {})).items()
    }
    seed_ranks = {
        str(seed): int(rank)
        for seed, rank in dict(contract.get("seed_ranks", {})).items()
    }
    history_rows: list[dict[str, object]] = []
    rollout_rows: list[dict[str, object]] = []
    standardization_stats: list[dict[str, object]] = []
    for variant_id in variants:
        raw_train = _load_phase65_tiled_variant_input(contract, train_tile_id, variant_id)
        standardizer = fit_phase65_train_tile_standardizer(raw_train)
        standardization_stats.append(standardizer.to_json_row())
        for seed in seeds:
            model, history = train_phase65_behavior_cloner(
                raw_train,
                standardizer,
                seed=int(seed),
                eval_max_steps=int(contract["eval_max_steps"]),
                epochs=int(contract["bc_epochs"]),
                learning_rate=float(contract["learning_rate"]),
                hidden_dim=int(contract["hidden_dim"]),
                top_k=int(contract["top_k"]),
            )
            history_rows.extend(history)
            for eval_tile_id in eval_tile_ids:
                raw_eval = _load_phase65_tiled_variant_input(
                    contract,
                    eval_tile_id,
                    variant_id,
                )
                rollout_rows.append(
                    rollout_phase65_greedy_policy(
                        model,
                        raw_tiled_input=raw_eval,
                        standardizer=standardizer,
                        train_tile_id=train_tile_id,
                        eval_tile_rank=eval_tile_ranks.get(str(eval_tile_id), 0),
                        seed=int(seed),
                        phase65_seed_rank=seed_ranks.get(str(int(seed)), 0),
                        eval_max_steps=int(contract["eval_max_steps"]),
                    )
                )
    phase63_style_analysis = build_phase63_set_policy_analysis(
        rollout_rows,
        existing_flattened_summary_csvs=existing_flattened_summary_csvs,
        metadata={"variants": variants, "eval_tile_ids": eval_tile_ids, "seeds": seeds},
    )
    standardization_comparison = build_phase65_standardization_comparison(
        rollout_rows,
        unstandardized_rows,
        variants=variants,
        eval_tile_ids=eval_tile_ids,
        seeds=seeds,
    )
    return {
        "phase": "phase65_standardized_set_policy_bc_rerun",
        "phase63_comparison_json": str(Path(phase63_comparison_json)),
        "phase63_rollout_csv": str(Path(phase63_rollout_csv)),
        "existing_flattened_summary_csvs": [
            str(Path(path)) for path in _normalize_optional_paths(existing_flattened_summary_csvs)
        ],
        "contract": dict(contract),
        "standardization_stats": standardization_stats,
        "history_rows": history_rows,
        "rollout_rows": rollout_rows,
        "phase63_style_analysis": phase63_style_analysis,
        "standardization_comparison": standardization_comparison,
        "claim_boundary": PHASE65_CLAIM_BOUNDARY,
    }
