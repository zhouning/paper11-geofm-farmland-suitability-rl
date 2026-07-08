from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from os import PathLike
from pathlib import Path
import random
import statistics

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .padded_heldout_policy import (
    _dependency_metadata,
    _normalize_seeds,
    _select_train_eval_tiles,
)
from .planning_reward import compute_base_planning_reward_from_matrix_row
from .tiled_inputs import load_tiled_variant_input


PHASE63_CLAIM_BOUNDARY = (
    "Phase 63 is a base-reward set-policy oracle-pretraining experiment. "
    "It tests whether task-aware block scoring and deterministic oracle behavior "
    "cloning improve candidate-block selection under existing Bishan tile inputs. "
    "It does not enable suitability reward, does not test B2/B3, does not test "
    "cross-region transfer, does not prove independent agronomic suitability, "
    "does not prove PCA optimality, and does not justify final submission-level "
    "planning-performance claims."
)

PHASE63_DEFAULT_VARIANTS = ("B0", "D4P8", "D4P16", "D6R8", "D6R16")
PHASE63_ALLOWED_VARIANTS = PHASE63_DEFAULT_VARIANTS
PHASE63_D4_D6_COMPARISONS = (("D4P8", "D6R8"), ("D4P16", "D6R16"))
PHASE63_D4_B0_COMPARISONS = (("D4P8", "B0"), ("D4P16", "B0"))

PHASE63_ORACLE_FIELDNAMES = [
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
]

PHASE63_HISTORY_FIELDNAMES = [
    "variant_id",
    "train_tile_id",
    "seed",
    "epoch",
    "loss",
    "top1_accuracy",
    "topk_hit_rate",
    "learning_rate",
    "hidden_dim",
    "claim_boundary",
]

PHASE63_ROLLOUT_FIELDNAMES = [
    "row_type",
    "variant_id",
    "train_tile_id",
    "eval_tile_id",
    "eval_tile_rank",
    "seed",
    "phase63_seed_rank",
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
    "claim_boundary",
]

PHASE63_DELTA_FIELDNAMES = [
    "variant_id",
    "eval_tile_id",
    "seed",
    "bc_reward",
    "oracle_total_reward",
    "oracle_gap",
    "oracle_gap_fraction",
    "flattened_reward",
    "bc_minus_flattened_reward",
    "bc_improves_flattened",
    "claim_boundary",
]


def build_phase63_set_policy_contract(
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str,
    phase61_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] | str = PHASE63_DEFAULT_VARIANTS,
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 5,
    eval_max_steps: int = 8,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
    bc_epochs: int = 80,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    top_k: int = 3,
) -> dict[str, object]:
    if int(eval_max_steps) <= 0:
        raise ValueError("eval_max_steps must be positive")
    if int(bc_epochs) <= 0:
        raise ValueError("bc_epochs must be positive")
    if float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if int(hidden_dim) <= 0:
        raise ValueError("hidden_dim must be positive")
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive")

    normalized_variants = _normalize_phase63_variants(variants)
    normalized_seeds = _normalize_seeds(seeds)
    selected = _select_train_eval_tiles(
        Path(tile_index_csv),
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
    )
    eval_ids = list(selected["eval_tile_ids"])
    train_id = str(selected["train_tile_id"])
    selected_counts = dict(selected["selected_tile_block_counts"])
    return {
        "phase": "phase63_set_policy_oracle_pretraining",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "phase8_output_dir": str(Path(phase8_output_dir)),
        "phase61_output_dir": str(Path(phase61_output_dir)),
        "tile_index_csv": str(Path(tile_index_csv)),
        "variants": normalized_variants,
        "variant_source_dirs": _phase63_variant_source_dirs(
            normalized_variants,
            phase2_output_dir=phase2_output_dir,
            phase8_output_dir=phase8_output_dir,
            phase61_output_dir=phase61_output_dir,
        ),
        "train_tile_id": train_id,
        "train_tile_ids": [train_id],
        "eval_tile_ids": eval_ids,
        "eval_tile_count": len(eval_ids),
        "eval_tile_ranks": {
            str(tile_id): rank for rank, tile_id in enumerate(eval_ids, start=1)
        },
        "selected_tile_block_counts": selected_counts,
        "train_tile_selection": selected["train_tile_selection"],
        "eval_tile_selection": selected["eval_tile_selection"],
        "max_blocks": max(int(count) for count in selected_counts.values()),
        "eval_max_steps": int(eval_max_steps),
        "seeds": normalized_seeds,
        "seed_count": len(normalized_seeds),
        "seed_ranks": {
            str(seed): rank for rank, seed in enumerate(normalized_seeds, start=1)
        },
        "bc_epochs": int(bc_epochs),
        "learning_rate": float(learning_rate),
        "hidden_dim": int(hidden_dim),
        "top_k": int(top_k),
        "claim_boundary": PHASE63_CLAIM_BOUNDARY,
    }


def build_phase63_model_inputs(
    tiled_input,
    selected_indices: Sequence[int] = (),
) -> dict[str, np.ndarray]:
    n_blocks = len(tiled_input.block_ids)
    selected = np.zeros(n_blocks, dtype=bool)
    for index in selected_indices:
        action_index = int(index)
        if action_index < 0 or action_index >= n_blocks:
            raise ValueError(f"Selected action out of range: {action_index}")
        if selected[action_index]:
            raise ValueError(f"Selected action repeated: {action_index}")
        selected[action_index] = True
    valid = np.ones(n_blocks, dtype=bool)
    available = np.logical_and(valid, ~selected)
    return {
        "block_features": tiled_input.state_matrix.astype(np.float32, copy=True),
        "valid_mask": valid,
        "selected_mask": selected,
        "available_mask": available,
    }


class Phase63SetPolicyScorer(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if int(n_features) <= 0:
            raise ValueError("n_features must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        self.n_features = int(n_features)
        self.hidden_dim = int(hidden_dim)
        self.block_encoder = nn.Sequential(
            nn.Linear(self.n_features + 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear((3 * self.n_features) + 3, self.hidden_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        block_features: torch.Tensor,
        valid_mask: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        if block_features.ndim != 3:
            raise ValueError("block_features must have shape [batch, blocks, features]")
        if block_features.shape[-1] != self.n_features:
            raise ValueError("block_features feature count does not match model")
        valid = valid_mask.to(dtype=torch.bool, device=block_features.device)
        selected = selected_mask.to(dtype=torch.bool, device=block_features.device)
        available = torch.logical_and(valid, torch.logical_not(selected))
        valid_f = valid.to(dtype=block_features.dtype).unsqueeze(-1)
        selected_f = selected.to(dtype=block_features.dtype).unsqueeze(-1)
        block_input = torch.cat([block_features, valid_f, selected_f], dim=-1)
        block_encoded = self.block_encoder(block_input)
        context = self._context_features(block_features, valid, selected, available)
        context_encoded = self.context_encoder(context).unsqueeze(1)
        context_encoded = context_encoded.expand(-1, block_features.shape[1], -1)
        logits = self.scorer(
            torch.cat([block_encoded, context_encoded], dim=-1)
        ).squeeze(-1)
        return logits.masked_fill(torch.logical_not(available), -1.0e9)

    def _context_features(
        self,
        block_features: torch.Tensor,
        valid_mask: torch.Tensor,
        selected_mask: torch.Tensor,
        available_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_mean = _masked_mean_tensor(block_features, valid_mask)
        selected_mean = _masked_mean_tensor(block_features, selected_mask)
        available_mean = _masked_mean_tensor(block_features, available_mask)
        denom = torch.clamp(
            valid_mask.sum(dim=1, keepdim=True).to(block_features.dtype),
            min=1.0,
        )
        valid_fraction = valid_mask.to(block_features.dtype).mean(dim=1, keepdim=True)
        selected_fraction = (
            selected_mask.sum(dim=1, keepdim=True).to(block_features.dtype) / denom
        )
        available_fraction = (
            available_mask.sum(dim=1, keepdim=True).to(block_features.dtype) / denom
        )
        return torch.cat(
            [
                valid_mean,
                selected_mean,
                available_mean,
                valid_fraction,
                selected_fraction,
                available_fraction,
            ],
            dim=1,
        )


def _masked_mean_tensor(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    denom = torch.clamp(weights.sum(dim=1), min=1.0)
    return (values * weights).sum(dim=1) / denom


def build_phase63_bc_examples(tiled_input, eval_max_steps: int) -> list[dict[str, object]]:
    trajectory = build_phase63_oracle_trajectory(tiled_input, eval_max_steps)
    examples: list[dict[str, object]] = []
    selected: list[int] = []
    for step in trajectory["steps"]:
        action_index = int(step["action_index"])
        inputs = build_phase63_model_inputs(tiled_input, selected)
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


def train_phase63_behavior_cloner(
    tiled_input,
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
    examples = build_phase63_bc_examples(tiled_input, eval_max_steps)
    if not examples:
        raise ValueError("Phase 63 behavior cloning requires at least one example")
    model = Phase63SetPolicyScorer(
        n_features=len(tiled_input.feature_columns),
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
                "variant_id": str(tiled_input.variant_id),
                "train_tile_id": str(tiled_input.tile_id),
                "seed": int(seed),
                "epoch": int(epoch),
                "loss": _round_float(statistics.mean(losses)),
                "top1_accuracy": _round_float(correct / len(examples)),
                "topk_hit_rate": _round_float(topk_hits / len(examples)),
                "learning_rate": float(learning_rate),
                "hidden_dim": int(hidden_dim),
                "claim_boundary": PHASE63_CLAIM_BOUNDARY,
            }
        )
    model.eval()
    return model, history


def rollout_phase63_greedy_policy(
    model: Phase63SetPolicyScorer,
    tiled_input,
    train_tile_id: str,
    eval_tile_rank: int,
    seed: int,
    phase63_seed_rank: int,
    eval_max_steps: int,
    device: str = "cpu",
) -> dict[str, object]:
    selected: list[int] = []
    selected_block_ids: list[str] = []
    rewards: list[float] = []
    invalid_action_count = 0
    for _step_index in range(min(int(eval_max_steps), len(tiled_input.block_ids))):
        inputs = build_phase63_model_inputs(tiled_input, selected)
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
        selected_block_ids.append(str(tiled_input.block_ids[action]))
        rewards.append(
            compute_base_planning_reward_from_matrix_row(
                tiled_input.feature_columns,
                tiled_input.state_matrix[action],
            )
        )
    oracle = build_phase63_oracle_trajectory(tiled_input, eval_max_steps)
    total_reward = _round_float(sum(rewards))
    oracle_total = float(oracle["total_oracle_reward"])
    oracle_gap = _round_float(oracle_total - total_reward)
    oracle_gap_fraction = _round_float(oracle_gap / max(abs(oracle_total), 1.0e-9))
    terminated = len(selected) == len(tiled_input.block_ids)
    return {
        "row_type": "bc_greedy_policy",
        "variant_id": str(tiled_input.variant_id),
        "train_tile_id": str(train_tile_id),
        "eval_tile_id": str(tiled_input.tile_id),
        "eval_tile_rank": int(eval_tile_rank),
        "seed": int(seed),
        "phase63_seed_rank": int(phase63_seed_rank),
        "eval_max_steps": int(eval_max_steps),
        "n_blocks": len(tiled_input.block_ids),
        "n_features": len(tiled_input.feature_columns),
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
        "claim_boundary": PHASE63_CLAIM_BOUNDARY,
    }


def build_phase63_oracle_trajectory(tiled_input, eval_max_steps: int) -> dict[str, object]:
    if int(eval_max_steps) <= 0:
        raise ValueError("eval_max_steps must be positive")
    rewards = _phase63_block_rewards(tiled_input)
    ranked = sorted(
        range(len(tiled_input.block_ids)),
        key=lambda index: (-rewards[index], str(tiled_input.block_ids[index]), index),
    )
    selected_indices = ranked[: min(int(eval_max_steps), len(ranked))]
    selected_block_ids = [str(tiled_input.block_ids[index]) for index in selected_indices]
    step_rewards = [_round_float(rewards[index]) for index in selected_indices]
    total = _round_float(sum(step_rewards))
    terminated = len(selected_indices) == len(tiled_input.block_ids)
    steps = []
    cumulative = 0.0
    for step_index, action_index in enumerate(selected_indices):
        cumulative = _round_float(cumulative + rewards[action_index])
        steps.append(
            {
                "step_index": int(step_index),
                "action_index": int(action_index),
                "block_id": str(tiled_input.block_ids[action_index]),
                "reward": _round_float(rewards[action_index]),
                "cumulative_reward": cumulative,
                "valid_action_count": int(len(tiled_input.block_ids) - step_index),
            }
        )
    return {
        "variant_id": str(tiled_input.variant_id),
        "tile_id": str(tiled_input.tile_id),
        "eval_max_steps": int(eval_max_steps),
        "n_blocks": len(tiled_input.block_ids),
        "n_features": len(tiled_input.feature_columns),
        "episode_steps": len(selected_indices),
        "terminated": bool(terminated),
        "action_indices": [int(index) for index in selected_indices],
        "selected_block_ids": selected_block_ids,
        "step_rewards": step_rewards,
        "steps": steps,
        "total_oracle_reward": total,
        "top_k_reward_ceiling": total,
        "claim_boundary": PHASE63_CLAIM_BOUNDARY,
    }


def _normalize_phase63_variants(variants: Sequence[str] | str) -> list[str]:
    if isinstance(variants, str):
        values = [part.strip().upper() for part in variants.split(",")]
    else:
        values = [str(item).strip().upper() for item in variants]
    normalized = [value for value in values if value]
    if not normalized:
        raise ValueError("At least one Phase 63 variant must be requested")
    unsupported = [value for value in normalized if value not in PHASE63_ALLOWED_VARIANTS]
    if unsupported:
        raise ValueError(f"Phase 63 unsupported variant: {unsupported[0]}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Phase 63 variants must be unique")
    return normalized


def _phase63_variant_source_dirs(
    variants: Sequence[str],
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str,
    phase61_output_dir: Path | str,
) -> dict[str, str]:
    source_dirs: dict[str, str] = {}
    for variant_id in variants:
        if variant_id == "B0":
            source_dirs[variant_id] = str(Path(phase2_output_dir))
        elif variant_id.startswith("D4"):
            source_dirs[variant_id] = str(Path(phase8_output_dir))
        elif variant_id.startswith("D6"):
            source_dirs[variant_id] = str(Path(phase61_output_dir))
        else:
            raise ValueError(f"Phase 63 has no source routing for {variant_id}")
    return source_dirs


def _phase63_block_rewards(tiled_input) -> list[float]:
    return [
        compute_base_planning_reward_from_matrix_row(
            tiled_input.feature_columns,
            tiled_input.state_matrix[index],
        )
        for index in range(len(tiled_input.block_ids))
    ]


def _round_float(value: object, digits: int = 10) -> float:
    rounded = round(float(value), digits)
    compact = round(rounded, 6)
    if abs(rounded - compact) < 5.0e-8:
        return compact
    return rounded



def build_phase63_set_policy_analysis(
    rollout_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    existing_flattened_rows: Sequence[Mapping[str, object]] | None = None,
    existing_flattened_summary_csvs: Sequence[Path | str] | str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    rollout_rows = _load_mapping_rows(rollout_rows_or_csv, "Phase 63 rollout")
    flattened_rows: list[dict[str, object]] = []
    if existing_flattened_rows is not None:
        flattened_rows.extend(dict(row) for row in existing_flattened_rows)
    for csv_path in _normalize_optional_paths(existing_flattened_summary_csvs):
        flattened_rows.extend(_load_mapping_rows(csv_path, "flattened PPO summary"))

    metadata_map = {} if metadata is None else dict(metadata)
    variants = _metadata_string_list(
        metadata_map,
        "variants",
        fallback=_unique_strings(rollout_rows, "variant_id"),
    )
    eval_tile_ids = _metadata_string_list(
        metadata_map,
        "eval_tile_ids",
        fallback=_unique_strings(rollout_rows, "eval_tile_id"),
    )
    seeds = _metadata_int_list(
        metadata_map,
        "seeds",
        fallback=_unique_ints(rollout_rows, "seed"),
    )
    coverage = _phase63_coverage_issues(rollout_rows, variants, eval_tile_ids, seeds)
    flattened_index = _flattened_reward_index(flattened_rows)
    delta_rows = _phase63_delta_rows(rollout_rows, flattened_index)
    architecture = _numeric_delta_summary(
        [
            float(row["bc_minus_flattened_reward"])
            for row in delta_rows
            if row["flattened_reward"] != ""
        ]
    )
    d4_b0_rows = _paired_variant_delta_rows(
        rollout_rows,
        PHASE63_D4_B0_COMPARISONS,
        value_field="total_contract_reward",
        output_field="left_minus_right_reward",
    )
    d4_d6_rows = _paired_variant_delta_rows(
        rollout_rows,
        PHASE63_D4_D6_COMPARISONS,
        value_field="total_contract_reward",
        output_field="left_minus_right_reward",
    )
    oracle_gaps = [float(row.get("oracle_gap_fraction", 1.0)) for row in rollout_rows]
    oracle_gap_summary = _numeric_delta_summary(oracle_gaps)
    d4_b0_summary = _numeric_delta_summary(
        [float(row["left_minus_right_reward"]) for row in d4_b0_rows]
    )
    d4_d6_summary = _numeric_delta_summary(
        [float(row["left_minus_right_reward"]) for row in d4_d6_rows]
    )
    status = _phase63_status(
        coverage,
        architecture,
        d4_b0_summary,
        oracle_gap_summary,
        has_flattened_baseline=bool(delta_rows),
    )
    return {
        "phase": "phase63_set_policy_analysis",
        "variants": variants,
        "eval_tile_ids": eval_tile_ids,
        "seeds": seeds,
        "rollout_rows": rollout_rows,
        "flattened_rows": flattened_rows,
        "delta_rows": delta_rows,
        "d4_b0_delta_rows": d4_b0_rows,
        "d4_d6_delta_rows": d4_d6_rows,
        "mean_bc_reward_by_variant": _mean_by_field(
            rollout_rows,
            "variant_id",
            "total_contract_reward",
        ),
        "mean_oracle_reward_by_variant": _mean_by_field(
            rollout_rows,
            "variant_id",
            "oracle_total_reward",
        ),
        "mean_flattened_reward_by_variant": _mean_by_field(
            flattened_rows,
            "variant_id",
            "total_contract_reward",
        ),
        "architecture_delta_summary": architecture,
        "d4_b0_delta_summary": d4_b0_summary,
        "d4_d6_delta_summary": d4_d6_summary,
        "oracle_gap_fraction_summary": oracle_gap_summary,
        "coverage_issues": coverage,
        "phase63_set_policy_status": status,
        "conclusion": _phase63_conclusion(status),
        "claim_boundary": PHASE63_CLAIM_BOUNDARY,
    }


def write_phase63_set_policy_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    oracle_json = output_path / "phase63_oracle_trajectories.json"
    oracle_summary_csv = output_path / "phase63_oracle_summary.csv"
    history_csv = output_path / "phase63_bc_training_history.csv"
    rollout_csv = output_path / "phase63_bc_rollout_summary.csv"
    comparison_json = output_path / "phase63_set_policy_comparison.json"
    delta_csv = output_path / "phase63_set_policy_delta_table.csv"
    readiness_md = output_path / "phase63_set_policy_oracle_pretraining.md"

    oracle_json.write_text(
        json.dumps(
            _json_ready(analysis.get("oracle_trajectories", [])),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_csv_mapping_rows(
        oracle_summary_csv,
        PHASE63_ORACLE_FIELDNAMES,
        analysis.get("oracle_summary_rows", []),
        "oracle_summary_rows",
    )
    _write_csv_mapping_rows(
        history_csv,
        PHASE63_HISTORY_FIELDNAMES,
        analysis.get("history_rows", []),
        "history_rows",
    )
    _write_csv_mapping_rows(
        rollout_csv,
        PHASE63_ROLLOUT_FIELDNAMES,
        analysis.get("rollout_rows", []),
        "rollout_rows",
    )
    _write_csv_mapping_rows(
        delta_csv,
        PHASE63_DELTA_FIELDNAMES,
        analysis.get("delta_rows", []),
        "delta_rows",
    )
    comparison = {
        key: value
        for key, value in dict(analysis).items()
        if key
        not in {
            "oracle_trajectories",
            "oracle_summary_rows",
            "history_rows",
            "rollout_rows",
            "flattened_rows",
        }
    }
    comparison_json.write_text(
        json.dumps(_json_ready(comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    readiness_md.write_text(_phase63_readiness_markdown(analysis), encoding="utf-8")
    return {
        "oracle_json": oracle_json,
        "oracle_summary_csv": oracle_summary_csv,
        "history_csv": history_csv,
        "rollout_csv": rollout_csv,
        "comparison_json": comparison_json,
        "delta_csv": delta_csv,
        "readiness_md": readiness_md,
    }


def run_phase63_set_policy_oracle_pretraining(
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str,
    phase61_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] | str = PHASE63_DEFAULT_VARIANTS,
    existing_flattened_summary_csvs: Sequence[Path | str] | str | None = None,
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 5,
    eval_max_steps: int = 8,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
    bc_epochs: int = 80,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    top_k: int = 3,
) -> dict[str, object]:
    contract = build_phase63_set_policy_contract(
        phase2_output_dir=phase2_output_dir,
        phase8_output_dir=phase8_output_dir,
        phase61_output_dir=phase61_output_dir,
        tile_index_csv=tile_index_csv,
        variants=variants,
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
        eval_max_steps=eval_max_steps,
        seeds=seeds,
        bc_epochs=bc_epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        top_k=top_k,
    )
    oracle_trajectories = []
    oracle_summary_rows = []
    history_rows = []
    rollout_rows = []
    for variant_id in contract["variants"]:
        train_tiled = _load_phase63_tiled_variant_input(
            contract,
            str(contract["train_tile_id"]),
            str(variant_id),
        )
        for seed in contract["seeds"]:
            model, history = train_phase63_behavior_cloner(
                train_tiled,
                seed=int(seed),
                eval_max_steps=int(contract["eval_max_steps"]),
                epochs=int(contract["bc_epochs"]),
                learning_rate=float(contract["learning_rate"]),
                hidden_dim=int(contract["hidden_dim"]),
                top_k=int(contract["top_k"]),
            )
            history_rows.extend(history)
            for eval_tile_id in contract["eval_tile_ids"]:
                eval_tiled = _load_phase63_tiled_variant_input(
                    contract,
                    str(eval_tile_id),
                    str(variant_id),
                )
                oracle = build_phase63_oracle_trajectory(
                    eval_tiled,
                    int(contract["eval_max_steps"]),
                )
                oracle_trajectories.append(oracle)
                oracle_summary_rows.append(
                    _phase63_oracle_summary_row(oracle, seed=int(seed), tile_role="eval")
                )
                rollout_rows.append(
                    rollout_phase63_greedy_policy(
                        model,
                        eval_tiled,
                        train_tile_id=str(contract["train_tile_id"]),
                        eval_tile_rank=int(contract["eval_tile_ranks"][str(eval_tile_id)]),
                        seed=int(seed),
                        phase63_seed_rank=int(contract["seed_ranks"][str(int(seed))]),
                        eval_max_steps=int(contract["eval_max_steps"]),
                    )
                )
    analysis = build_phase63_set_policy_analysis(
        rollout_rows,
        existing_flattened_summary_csvs=existing_flattened_summary_csvs,
        metadata={
            "variants": contract["variants"],
            "eval_tile_ids": contract["eval_tile_ids"],
            "seeds": contract["seeds"],
        },
    )
    analysis["contract"] = contract
    analysis["oracle_trajectories"] = oracle_trajectories
    analysis["oracle_summary_rows"] = oracle_summary_rows
    analysis["history_rows"] = history_rows
    analysis["rollout_rows"] = rollout_rows
    analysis["dependencies"] = _dependency_metadata()
    analysis["dependencies"]["torch"] = torch.__version__
    return analysis


def _load_mapping_rows(
    rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    label: str,
) -> list[dict[str, object]]:
    if isinstance(rows_or_csv, (str, PathLike)):
        path = Path(rows_or_csv)
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} CSV: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return [dict(row) for row in rows_or_csv]


def _normalize_optional_paths(paths: Sequence[Path | str] | str | None) -> list[Path | str]:
    if paths is None:
        return []
    if isinstance(paths, str):
        return [part.strip() for part in paths.split(",") if part.strip()]
    return [path for path in paths if str(path).strip()]


def _metadata_string_list(
    metadata: Mapping[str, object],
    key: str,
    fallback: list[str],
) -> list[str]:
    value = metadata.get(key)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Sequence):
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
    if isinstance(value, Sequence):
        return [int(item) for item in value if str(item).strip()]
    return fallback


def _unique_strings(rows: Sequence[Mapping[str, object]], field: str) -> list[str]:
    seen: list[str] = []
    for row in rows:
        value = str(row.get(field, "")).strip()
        if value and value not in seen:
            seen.append(value)
    return seen


def _unique_ints(rows: Sequence[Mapping[str, object]], field: str) -> list[int]:
    values = []
    for row in rows:
        text = str(row.get(field, "")).strip()
        if not text:
            continue
        value = int(text)
        if value not in values:
            values.append(value)
    return values


def _phase63_coverage_issues(
    rows: Sequence[Mapping[str, object]],
    variants: Sequence[str],
    eval_tile_ids: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, list[dict[str, object]]]:
    counts: dict[tuple[str, str, int], int] = {}
    for row in rows:
        if str(row.get("row_type", "")) != "bc_greedy_policy":
            continue
        key = (
            str(row.get("variant_id", "")),
            str(row.get("eval_tile_id", "")),
            int(row.get("seed", 0)),
        )
        counts[key] = counts.get(key, 0) + 1
    missing = []
    duplicate = []
    for variant_id in variants:
        for tile_id in eval_tile_ids:
            for seed in seeds:
                key = (str(variant_id), str(tile_id), int(seed))
                count = counts.get(key, 0)
                if count == 0:
                    missing.append(
                        {"variant_id": variant_id, "eval_tile_id": tile_id, "seed": seed}
                    )
                elif count > 1:
                    duplicate.append(
                        {
                            "variant_id": variant_id,
                            "eval_tile_id": tile_id,
                            "seed": seed,
                            "count": count,
                        }
                    )
    expected = {
        (str(variant), str(tile_id), int(seed))
        for variant in variants
        for tile_id in eval_tile_ids
        for seed in seeds
    }
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


def _flattened_reward_index(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str, int], float]:
    values: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        if str(row.get("row_type", "")) != "trained_policy":
            continue
        key = (
            str(row.get("variant_id", "")),
            str(row.get("eval_tile_id", "")),
            int(row.get("seed", 0)),
        )
        values.setdefault(key, []).append(float(row.get("total_contract_reward", 0.0)))
    return {key: _round_float(statistics.mean(items)) for key, items in values.items()}


def _phase63_delta_rows(
    rollout_rows: Sequence[Mapping[str, object]],
    flattened_index: Mapping[tuple[str, str, int], float],
) -> list[dict[str, object]]:
    rows = []
    for row in rollout_rows:
        if str(row.get("row_type", "")) != "bc_greedy_policy":
            continue
        key = (
            str(row.get("variant_id", "")),
            str(row.get("eval_tile_id", "")),
            int(row.get("seed", 0)),
        )
        flattened = flattened_index.get(key)
        bc_reward = float(row.get("total_contract_reward", 0.0))
        delta = "" if flattened is None else _round_float(bc_reward - flattened)
        rows.append(
            {
                "variant_id": key[0],
                "eval_tile_id": key[1],
                "seed": key[2],
                "bc_reward": _round_float(bc_reward),
                "oracle_total_reward": _round_float(row.get("oracle_total_reward", 0.0)),
                "oracle_gap": _round_float(row.get("oracle_gap", 0.0)),
                "oracle_gap_fraction": _round_float(row.get("oracle_gap_fraction", 0.0)),
                "flattened_reward": "" if flattened is None else _round_float(flattened),
                "bc_minus_flattened_reward": delta,
                "bc_improves_flattened": "" if flattened is None else bool(float(delta) > 0.0),
                "claim_boundary": PHASE63_CLAIM_BOUNDARY,
            }
        )
    return rows


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
            int(row.get("seed", 0)),
        ): float(row.get(value_field, 0.0))
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
                    "claim_boundary": PHASE63_CLAIM_BOUNDARY,
                }
            )
    return output


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


def _mean_by_field(
    rows: Sequence[Mapping[str, object]],
    group_field: str,
    value_field: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        key = str(row.get(group_field, ""))
        if not key or str(row.get(value_field, "")).strip() == "":
            continue
        grouped.setdefault(key, []).append(float(row[value_field]))
    return {key: _round_float(statistics.mean(values)) for key, values in sorted(grouped.items())}


def _phase63_status(
    coverage: Mapping[str, object],
    architecture: Mapping[str, object],
    d4_b0_summary: Mapping[str, object],
    oracle_gap_summary: Mapping[str, object],
    has_flattened_baseline: bool,
) -> str:
    if coverage["missing_rollout_rows"] or coverage["duplicate_rollout_rows"]:
        return "insufficient"
    if not has_flattened_baseline:
        return (
            "set_policy_route_supported"
            if float(oracle_gap_summary["mean_delta"]) <= 0.2
            else "insufficient"
        )
    architecture_supported = (
        int(architecture["total_count"]) > 0
        and float(architecture["mean_delta"]) > 0.0
        and int(architecture["positive_count"]) * 2 >= int(architecture["total_count"])
    )
    geofm_supported = (
        int(d4_b0_summary["total_count"]) > 0
        and float(d4_b0_summary["mean_delta"]) > 0.0
        and int(d4_b0_summary["positive_count"]) * 2 >= int(d4_b0_summary["total_count"])
    )
    oracle_gap_small = float(oracle_gap_summary["mean_delta"]) <= 0.2
    if architecture_supported and geofm_supported and oracle_gap_small:
        return "geofm_set_policy_advantage"
    if architecture_supported and oracle_gap_small:
        return "architecture_improves_but_geofm_not_distinguished"
    return "set_policy_route_not_supported"


def _phase63_conclusion(status: str) -> str:
    conclusions = {
        "geofm_set_policy_advantage": (
            "Phase 63 supports the set-policy route and shows GeoFM-derived "
            "variants exceeding B0 under the set-policy protocol."
        ),
        "architecture_improves_but_geofm_not_distinguished": (
            "Phase 63 supports the set-policy architecture route, but does not "
            "separate GeoFM-derived variants from B0."
        ),
        "set_policy_route_supported": (
            "Phase 63 supports the set-policy route by closing the oracle gap, "
            "but flattened PPO baseline comparison is incomplete."
        ),
        "set_policy_route_not_supported": (
            "Phase 63 does not yet support the set-policy route under the current "
            "behavior-cloning setup."
        ),
        "insufficient": "Phase 63 has insufficient coverage for a decision.",
    }
    return conclusions.get(status, conclusions["insufficient"])


def _load_phase63_tiled_variant_input(
    contract: Mapping[str, object],
    tile_id: str,
    variant_id: str,
):
    variant_source_dirs = contract.get("variant_source_dirs")
    if not isinstance(variant_source_dirs, Mapping):
        raise ValueError("Phase 63 contract is missing variant source routing")
    source_dir = variant_source_dirs.get(variant_id)
    if source_dir is None:
        raise ValueError(f"Phase 63 contract has no source for variant {variant_id}")
    return load_tiled_variant_input(
        source_dir,
        str(contract["tile_index_csv"]),
        tile_id,
        variant_id=variant_id,
    )


def _phase63_oracle_summary_row(
    oracle: Mapping[str, object],
    seed: int,
    tile_role: str,
) -> dict[str, object]:
    return {
        "variant_id": oracle.get("variant_id", ""),
        "tile_role": tile_role,
        "tile_id": oracle.get("tile_id", ""),
        "seed": int(seed),
        "eval_max_steps": oracle.get("eval_max_steps", ""),
        "n_blocks": oracle.get("n_blocks", ""),
        "n_features": oracle.get("n_features", ""),
        "episode_steps": oracle.get("episode_steps", ""),
        "terminated": oracle.get("terminated", ""),
        "total_oracle_reward": oracle.get("total_oracle_reward", ""),
        "top_k_reward_ceiling": oracle.get("top_k_reward_ceiling", ""),
        "selected_block_ids": ";".join(str(item) for item in oracle.get("selected_block_ids", [])),
        "action_indices": ";".join(str(item) for item in oracle.get("action_indices", [])),
        "claim_boundary": PHASE63_CLAIM_BOUNDARY,
    }


def _phase63_readiness_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Phase 63 Set-Policy Oracle Pretraining",
        "",
        f"Status: {analysis.get('phase63_set_policy_status', '')}",
        "",
        "Conclusion:",
        str(analysis.get("conclusion", "")),
        "",
        "Mean behavior-cloned reward by variant:",
    ]
    for variant_id, value in dict(analysis.get("mean_bc_reward_by_variant", {})).items():
        lines.append(f"- {variant_id}: {value}")
    lines.extend(["", "Mean oracle reward by variant:"])
    for variant_id, value in dict(analysis.get("mean_oracle_reward_by_variant", {})).items():
        lines.append(f"- {variant_id}: {value}")
    lines.extend(
        [
            "",
            f"Architecture delta summary: {analysis.get('architecture_delta_summary', {})}",
            f"D4/B0 delta summary: {analysis.get('d4_b0_delta_summary', {})}",
            f"D4/D6 delta summary: {analysis.get('d4_d6_delta_summary', {})}",
            f"Oracle gap summary: {analysis.get('oracle_gap_fraction_summary', {})}",
            "",
            "Claim boundary:",
            str(analysis.get("claim_boundary", PHASE63_CLAIM_BOUNDARY)),
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    row_key: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 63 analysis is missing {row_key}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 63 {row_key} rows must be objects")
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
