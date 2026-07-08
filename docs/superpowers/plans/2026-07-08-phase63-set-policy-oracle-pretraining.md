# Phase 63 Set-Policy Oracle Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a bounded Phase 63 set-style block-selection policy with deterministic oracle behavior-cloning pretraining, then compare its base-reward rollouts against existing flattened PPO evidence.

**Architecture:** Add one focused Phase 63 module that owns contract routing, deterministic oracle trajectories, a PyTorch set-policy scorer, supervised behavior cloning, greedy rollout, analysis/status rules, and artifact writers. Add a thin experiment runner with `rollout-only`, `run-and-analyze`, and `analyze-only` modes. Keep formal manuscript files untouched until the algorithm and experiment evidence are stable.

**Tech Stack:** Python standard library, NumPy, PyTorch, existing `paper11_geofm.tiled_inputs`, existing `paper11_geofm.planning_reward`, existing tile selection helpers from `paper11_geofm.padded_heldout_policy`, CSV/JSON/Markdown writers, pytest.

---

## File Structure

- Create `src/paper11_geofm/phase63_set_policy_oracle_pretraining.py`.
  Owns constants, variant routing, contract builder, oracle trajectory builder, set-policy model, behavior-cloning training, greedy rollout, analysis/status rules, and artifact writers.
- Create `experiments/phase63_set_policy_oracle_pretraining/run_phase63_set_policy_oracle_pretraining.py`.
  Exposes `rollout-only`, `run-and-analyze`, and `analyze-only` modes.
- Create `tests/test_phase63_set_policy_oracle_pretraining.py`.
  Covers contract routing, oracle ordering, model masking, supervised loss reduction, greedy rollout validity, writer outputs, and CLI behavior.
- Create `paper/phase28_results/29_phase63_set_policy_oracle_pretraining.md` after the real run.
  Records Phase 63 algorithm evidence only.
- Modify `paper/phase28_results/README.md` after the real run.
  Adds the Phase 63 evidence entry and reproduction command.
- Modify `docs/superpowers/phase33_current_progress_handoff.md` after the real run.
  Records the current Phase 63 status and next entry point.

Do not modify:

- `paper/submission/final/Paper11_formal_conclusion_manuscript.md`
- `paper/submission/final/Paper11_formal_conclusion_manuscript.tex`
- `paper/submission/final/Paper11_formal_conclusion_manuscript.pdf`
- `paper/submission/final/Paper11_submission_metadata_template.md`

---

## Phase 63 Contract

Use this contract for the first implementation:

- Reward: existing deterministic `base_planning_reward`.
- Train/eval split: existing `_select_train_eval_tiles()` helper.
- Default variants: `B0,D4P8,D4P16,D6R8,D6R16`.
- Variant source routing:
  - `B0` from `phase2_output_dir`
  - `D4P8,D4P16` from `phase8_output_dir`
  - `D6R8,D6R16` from `phase61_output_dir`
- Default `eval_max_steps`: `8`.
- Default seeds: `0,1,2`.
- Default behavior-cloning epochs: `80`.
- Default learning rate: `0.001`.
- Default hidden dimension: `64`.
- Default top-k metric: `3`.
- First milestone: behavior-cloned greedy rollout only. PPO fine-tuning is not part of this plan.

Claim boundary constant:

```python
PHASE63_CLAIM_BOUNDARY = (
    "Phase 63 is a base-reward set-policy oracle-pretraining experiment. "
    "It tests whether task-aware block scoring and deterministic oracle behavior "
    "cloning improve candidate-block selection under existing Bishan tile inputs. "
    "It does not enable suitability reward, does not test B2/B3, does not test "
    "cross-region transfer, does not prove independent agronomic suitability, "
    "does not prove PCA optimality, and does not justify final submission-level "
    "planning-performance claims."
)
```

---

### Task 1: Contract Routing and Oracle Trajectories

**Files:**
- Create: `tests/test_phase63_set_policy_oracle_pretraining.py`
- Create: `src/paper11_geofm/phase63_set_policy_oracle_pretraining.py`

- [ ] **Step 1: Write failing tests for contract routing and oracle ordering**

Create `tests/test_phase63_set_policy_oracle_pretraining.py` with these imports and helpers:

```python
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _required_feature_columns() -> tuple[str, ...]:
    return (
        "explicit_feature_00",
        "explicit_feature_01",
        "explicit_feature_02",
        "explicit_feature_04",
        "explicit_feature_07",
        "explicit_feature_09",
        "explicit_feature_10",
        "explicit_feature_13",
        "explicit_feature_16",
    )


def _tiled_input(block_ids=("b2", "b1", "b3"), scores=(0.5, 0.5, 0.3), variant_id="B0"):
    from paper11_geofm.tiled_inputs import TiledVariantInput

    columns = _required_feature_columns()
    matrix = np.zeros((len(block_ids), len(columns)), dtype=np.float32)
    slope_farmland_index = columns.index("explicit_feature_16")
    for row_index, score in enumerate(scores):
        matrix[row_index, slope_farmland_index] = float(score)
    return TiledVariantInput(
        tile_id="tile_eval",
        variant_id=variant_id,
        block_ids=tuple(block_ids),
        feature_columns=columns,
        state_matrix=matrix,
        reward_mode="base_planning_reward",
        state_groups=("explicit_planning_features",),
        source_table=Path("variant_B0_features.csv"),
        tile_index_csv=Path("tiles.csv"),
    )
```

Add the first two tests:

```python
def test_phase63_contract_routes_b0_d4_and_d6_variants(tmp_path):
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        build_phase63_set_policy_contract,
    )

    tile_index = _write_csv(
        tmp_path / "tiles.csv",
        [
            {"tile_id": "tile_train", "block_ids": "b1;b2;b3;b4"},
            {"tile_id": "tile_eval", "block_ids": "b1;b2;b3"},
        ],
    )
    contract = build_phase63_set_policy_contract(
        phase2_output_dir=tmp_path / "phase2",
        phase8_output_dir=tmp_path / "phase8",
        phase61_output_dir=tmp_path / "phase61",
        tile_index_csv=tile_index,
        train_tile_id="tile_train",
        eval_tile_ids="tile_eval",
        variants="B0,D4P8,D4P16,D6R8,D6R16",
        seeds="0,1",
        eval_max_steps=2,
        bc_epochs=5,
        learning_rate=0.01,
        hidden_dim=16,
        top_k=2,
    )

    assert contract["phase"] == "phase63_set_policy_oracle_pretraining"
    assert contract["variants"] == ["B0", "D4P8", "D4P16", "D6R8", "D6R16"]
    assert contract["variant_source_dirs"]["B0"].endswith("phase2")
    assert contract["variant_source_dirs"]["D4P8"].endswith("phase8")
    assert contract["variant_source_dirs"]["D6R16"].endswith("phase61")
    assert contract["eval_tile_ids"] == ["tile_eval"]
    assert contract["seeds"] == [0, 1]
    assert contract["eval_max_steps"] == 2
    assert contract["bc_epochs"] == 5
    assert contract["top_k"] == 2


def test_phase63_oracle_uses_reward_descending_then_block_id_tiebreak():
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        build_phase63_oracle_trajectory,
    )

    trajectory = build_phase63_oracle_trajectory(
        _tiled_input(),
        eval_max_steps=3,
    )

    assert trajectory["action_indices"] == [1, 0, 2]
    assert trajectory["selected_block_ids"] == ["b1", "b2", "b3"]
    assert trajectory["step_rewards"] == [0.175, 0.175, 0.105]
    assert trajectory["total_oracle_reward"] == 0.455
    assert trajectory["top_k_reward_ceiling"] == 0.455
```

Add a termination test:

```python
def test_phase63_oracle_stops_at_eval_max_steps():
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        build_phase63_oracle_trajectory,
    )

    trajectory = build_phase63_oracle_trajectory(
        _tiled_input(block_ids=("b3", "b1", "b2"), scores=(0.1, 0.9, 0.8)),
        eval_max_steps=2,
    )

    assert trajectory["action_indices"] == [1, 2]
    assert trajectory["selected_block_ids"] == ["b1", "b2"]
    assert trajectory["episode_steps"] == 2
    assert trajectory["terminated"] is False
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests\test_phase63_set_policy_oracle_pretraining.py -q --basetemp=.pytest_tmp_phase63_task1_red -p no:cacheprovider
```

Expected: fails because `paper11_geofm.phase63_set_policy_oracle_pretraining` does not exist.

- [ ] **Step 3: Add the Phase 63 module with contract and oracle code**

Create `src/paper11_geofm/phase63_set_policy_oracle_pretraining.py` with these imports, constants, fieldnames, and functions:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import math
import random
import statistics
from os import PathLike
from pathlib import Path

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
```

Add constants:

```python
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
PHASE63_GEOM_VARIANTS = ("D4P8", "D4P16", "D6R8", "D6R16")
```

Add summary fieldnames:

```python
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
```

Add contract helpers:

```python
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
```

Add oracle helpers:

```python
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
                "step_index": step_index,
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


def _phase63_block_rewards(tiled_input) -> list[float]:
    return [
        compute_base_planning_reward_from_matrix_row(
            tiled_input.feature_columns,
            tiled_input.state_matrix[index],
        )
        for index in range(len(tiled_input.block_ids))
    ]


def _round_float(value: object, digits: int = 10) -> float:
    return round(float(value), digits)
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase63_set_policy_oracle_pretraining.py -q --basetemp=.pytest_tmp_phase63_task1_green -p no:cacheprovider
```

Expected: the three Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src\paper11_geofm\phase63_set_policy_oracle_pretraining.py tests\test_phase63_set_policy_oracle_pretraining.py
git commit -m "feat: add Phase 63 oracle contract"
```

---

### Task 2: Set-Policy Scorer and Masked Inputs

**Files:**
- Modify: `tests/test_phase63_set_policy_oracle_pretraining.py`
- Modify: `src/paper11_geofm/phase63_set_policy_oracle_pretraining.py`

- [ ] **Step 1: Add failing model input and mask tests**

Append these tests:

```python
def test_phase63_model_inputs_encode_valid_selected_and_available_masks():
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        build_phase63_model_inputs,
    )

    inputs = build_phase63_model_inputs(
        _tiled_input(block_ids=("b1", "b2", "b3"), scores=(0.9, 0.4, 0.2)),
        selected_indices=(1,),
    )

    assert inputs["block_features"].shape == (3, 9)
    assert inputs["valid_mask"].tolist() == [True, True, True]
    assert inputs["selected_mask"].tolist() == [False, True, False]
    assert inputs["available_mask"].tolist() == [True, False, True]


def test_phase63_set_policy_scorer_masks_selected_and_invalid_actions():
    import torch
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        Phase63SetPolicyScorer,
    )

    torch.manual_seed(63)
    model = Phase63SetPolicyScorer(n_features=9, hidden_dim=12)
    block_features = torch.zeros((1, 4, 9), dtype=torch.float32)
    valid_mask = torch.tensor([[True, True, False, False]])
    selected_mask = torch.tensor([[False, True, False, False]])

    logits = model(block_features, valid_mask, selected_mask)

    assert logits.shape == (1, 4)
    assert torch.isfinite(logits[0, 0])
    assert logits[0, 1].item() < -1e8
    assert logits[0, 2].item() < -1e8
    assert logits[0, 3].item() < -1e8
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests\test_phase63_set_policy_oracle_pretraining.py -q --basetemp=.pytest_tmp_phase63_task2_red -p no:cacheprovider
```

Expected: fails because `build_phase63_model_inputs` and `Phase63SetPolicyScorer` are missing.

- [ ] **Step 3: Add model input builder and scorer**

Add these definitions to `src/paper11_geofm/phase63_set_policy_oracle_pretraining.py`:

```python
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
        logits = self.scorer(torch.cat([block_encoded, context_encoded], dim=-1)).squeeze(-1)
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
        denom = torch.clamp(valid_mask.sum(dim=1, keepdim=True).to(block_features.dtype), min=1.0)
        valid_fraction = valid_mask.to(block_features.dtype).mean(dim=1, keepdim=True)
        selected_fraction = selected_mask.sum(dim=1, keepdim=True).to(block_features.dtype) / denom
        available_fraction = available_mask.sum(dim=1, keepdim=True).to(block_features.dtype) / denom
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
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase63_set_policy_oracle_pretraining.py -q --basetemp=.pytest_tmp_phase63_task2_green -p no:cacheprovider
```

Expected: Task 1 and Task 2 tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src\paper11_geofm\phase63_set_policy_oracle_pretraining.py tests\test_phase63_set_policy_oracle_pretraining.py
git commit -m "feat: add Phase 63 set policy scorer"
```

---

### Task 3: Behavior Cloning and Greedy Rollout

**Files:**
- Modify: `tests/test_phase63_set_policy_oracle_pretraining.py`
- Modify: `src/paper11_geofm/phase63_set_policy_oracle_pretraining.py`

- [ ] **Step 1: Add failing behavior-cloning tests**

Append these tests:

```python
def test_phase63_behavior_cloning_loss_decreases_on_tiny_tile():
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        train_phase63_behavior_cloner,
    )

    model, history = train_phase63_behavior_cloner(
        _tiled_input(block_ids=("b1", "b2", "b3", "b4"), scores=(0.9, 0.7, 0.2, 0.1)),
        seed=63,
        eval_max_steps=3,
        epochs=25,
        learning_rate=0.01,
        hidden_dim=16,
        top_k=2,
    )

    assert model.n_features == 9
    assert len(history) == 25
    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["top1_accuracy"] >= history[0]["top1_accuracy"]


def test_phase63_greedy_rollout_never_selects_invalid_or_repeated_actions():
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        rollout_phase63_greedy_policy,
        train_phase63_behavior_cloner,
    )

    tiled = _tiled_input(
        block_ids=("b3", "b1", "b2", "b4"),
        scores=(0.2, 0.9, 0.7, 0.1),
    )
    model, _history = train_phase63_behavior_cloner(
        tiled,
        seed=63,
        eval_max_steps=3,
        epochs=30,
        learning_rate=0.01,
        hidden_dim=16,
        top_k=2,
    )
    rollout = rollout_phase63_greedy_policy(
        model,
        tiled,
        train_tile_id="tile_train",
        eval_tile_rank=1,
        seed=63,
        phase63_seed_rank=1,
        eval_max_steps=3,
    )

    assert rollout["row_type"] == "bc_greedy_policy"
    assert rollout["all_actions_valid"] is True
    assert rollout["invalid_action_count"] == 0
    assert len(rollout["selected_action_indices"].split(";")) == 3
    assert len(set(rollout["selected_action_indices"].split(";"))) == 3
    assert float(rollout["total_contract_reward"]) > 0.0
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests\test_phase63_set_policy_oracle_pretraining.py -q --basetemp=.pytest_tmp_phase63_task3_red -p no:cacheprovider
```

Expected: fails because behavior-cloning and rollout functions are missing.

- [ ] **Step 3: Add behavior-cloning examples, training, and rollout**

Add these functions:

```python
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
            block_features = torch.tensor(example["block_features"], dtype=torch.float32, device=device).unsqueeze(0)
            valid_mask = torch.tensor(example["valid_mask"], dtype=torch.bool, device=device).unsqueeze(0)
            selected_mask = torch.tensor(example["selected_mask"], dtype=torch.bool, device=device).unsqueeze(0)
            target = torch.tensor([int(example["target_action"])], dtype=torch.long, device=device)
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
```

Add rollout:

```python
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
                torch.tensor(inputs["block_features"], dtype=torch.float32, device=device).unsqueeze(0),
                torch.tensor(inputs["valid_mask"], dtype=torch.bool, device=device).unsqueeze(0),
                torch.tensor(inputs["selected_mask"], dtype=torch.bool, device=device).unsqueeze(0),
            )
        action = int(torch.argmax(logits, dim=1).item())
        if action in selected or not bool(available[action]):
            invalid_action_count += 1
            valid_indices = [int(index) for index, flag in enumerate(available.tolist()) if flag]
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
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase63_set_policy_oracle_pretraining.py -q --basetemp=.pytest_tmp_phase63_task3_green -p no:cacheprovider
```

Expected: all Task 1-3 tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add src\paper11_geofm\phase63_set_policy_oracle_pretraining.py tests\test_phase63_set_policy_oracle_pretraining.py
git commit -m "feat: add Phase 63 behavior cloning rollout"
```

---

### Task 4: Analysis, Writers, Runner CLI

**Files:**
- Modify: `tests/test_phase63_set_policy_oracle_pretraining.py`
- Modify: `src/paper11_geofm/phase63_set_policy_oracle_pretraining.py`
- Create: `experiments/phase63_set_policy_oracle_pretraining/run_phase63_set_policy_oracle_pretraining.py`

- [ ] **Step 1: Add failing analysis and writer tests**

Append helpers:

```python
def _rollout_row(variant_id, reward, oracle=1.0, tile_id="tile_a", seed=0):
    return {
        "row_type": "bc_greedy_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": tile_id,
        "eval_tile_rank": 1 if tile_id == "tile_a" else 2,
        "seed": seed,
        "phase63_seed_rank": seed + 1,
        "eval_max_steps": 8,
        "n_blocks": 4,
        "n_features": 9,
        "episode_steps": 3,
        "terminated": False,
        "truncated": True,
        "all_actions_valid": True,
        "invalid_action_count": 0,
        "total_contract_reward": reward,
        "oracle_total_reward": oracle,
        "oracle_gap": oracle - reward,
        "oracle_gap_fraction": (oracle - reward) / oracle,
        "selected_block_ids": "b1;b2;b3",
        "selected_action_indices": "0;1;2",
        "claim_boundary": "fixture",
    }


def _flattened_row(variant_id, reward, tile_id="tile_a", seed=0):
    return {
        "row_type": "trained_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": tile_id,
        "seed": seed,
        "total_contract_reward": reward,
    }
```

Append analysis tests:

```python
def test_phase63_analysis_reports_architecture_improvement_with_complete_baseline():
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        build_phase63_set_policy_analysis,
    )

    pairs = [("tile_a", 0), ("tile_a", 1), ("tile_b", 0), ("tile_b", 1)]
    rollout_rows = []
    flattened_rows = []
    for tile_id, seed in pairs:
        rollout_rows.extend(
            [
                _rollout_row("B0", 1.10, tile_id=tile_id, seed=seed),
                _rollout_row("D4P8", 1.30, tile_id=tile_id, seed=seed),
                _rollout_row("D4P16", 1.35, tile_id=tile_id, seed=seed),
                _rollout_row("D6R8", 1.25, tile_id=tile_id, seed=seed),
                _rollout_row("D6R16", 1.28, tile_id=tile_id, seed=seed),
            ]
        )
        flattened_rows.extend(
            [
                _flattened_row("B0", 0.90, tile_id=tile_id, seed=seed),
                _flattened_row("D4P8", 1.00, tile_id=tile_id, seed=seed),
                _flattened_row("D4P16", 1.00, tile_id=tile_id, seed=seed),
                _flattened_row("D6R8", 1.00, tile_id=tile_id, seed=seed),
                _flattened_row("D6R16", 1.00, tile_id=tile_id, seed=seed),
            ]
        )
    analysis = build_phase63_set_policy_analysis(
        rollout_rows,
        existing_flattened_rows=flattened_rows,
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
    )

    assert analysis["phase63_set_policy_status"] == "geofm_set_policy_advantage"
    assert analysis["architecture_delta_summary"]["mean_delta"] > 0
    assert analysis["d4_b0_delta_summary"]["positive_count"] == 8
    assert analysis["coverage_issues"]["missing_rollout_rows"] == []
```

Append writer test:

```python
def test_phase63_writer_outputs_json_csv_and_markdown(tmp_path):
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        build_phase63_set_policy_analysis,
        write_phase63_set_policy_artifacts,
    )

    rollout_rows = [_rollout_row("B0", 1.0), _rollout_row("D4P8", 1.2)]
    flattened_rows = [_flattened_row("B0", 0.8), _flattened_row("D4P8", 0.9)]
    analysis = build_phase63_set_policy_analysis(
        rollout_rows,
        existing_flattened_rows=flattened_rows,
        metadata={"eval_tile_ids": ["tile_a"], "seeds": [0], "variants": ["B0", "D4P8"]},
    )
    paths = write_phase63_set_policy_artifacts(
        {
            **analysis,
            "oracle_trajectories": [],
            "oracle_summary_rows": [],
            "history_rows": [],
            "rollout_rows": rollout_rows,
        },
        tmp_path / "outputs",
    )

    assert paths["oracle_json"].name == "phase63_oracle_trajectories.json"
    assert paths["oracle_summary_csv"].name == "phase63_oracle_summary.csv"
    assert paths["history_csv"].name == "phase63_bc_training_history.csv"
    assert paths["rollout_csv"].name == "phase63_bc_rollout_summary.csv"
    assert paths["comparison_json"].name == "phase63_set_policy_comparison.json"
    assert paths["delta_csv"].name == "phase63_set_policy_delta_table.csv"
    assert paths["readiness_md"].name == "phase63_set_policy_oracle_pretraining.md"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase"] == "phase63_set_policy_analysis"
    markdown = paths["readiness_md"].read_text(encoding="utf-8")
    assert "Phase 63 Set-Policy Oracle Pretraining" in markdown
    assert "does not enable suitability reward" in markdown
```

- [ ] **Step 2: Add failing CLI tests**

Append:

```python
def test_phase63_cli_analyze_only(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase63_set_policy_oracle_pretraining"
        / "run_phase63_set_policy_oracle_pretraining.py"
    )
    spec = importlib.util.spec_from_file_location("phase63_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    rollout_csv = _write_csv(tmp_path / "rollout.csv", [_rollout_row("B0", 1.0)])
    flattened_csv = _write_csv(tmp_path / "flat.csv", [_flattened_row("B0", 0.8)])
    exit_code = module.main(
        [
            "--mode",
            "analyze-only",
            "--existing-rollout-csv",
            str(rollout_csv),
            "--existing-flattened-summary-csvs",
            str(flattened_csv),
            "--output-dir",
            str(tmp_path / "analysis"),
            "--eval-tile-ids",
            "tile_a",
            "--seeds",
            "0",
            "--variants",
            "B0",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Phase 63 set-policy status:" in stdout
    assert "phase63_set_policy_comparison.json" in stdout


def test_phase63_cli_rollout_only_parser_accepts_core_inputs():
    runner_path = (
        ROOT
        / "experiments"
        / "phase63_set_policy_oracle_pretraining"
        / "run_phase63_set_policy_oracle_pretraining.py"
    )
    spec = importlib.util.spec_from_file_location("phase63_runner_args", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "rollout-only",
            "--phase2-output-dir",
            "phase2",
            "--phase8-output-dir",
            "phase8",
            "--phase61-output-dir",
            "phase61",
            "--tile-index-csv",
            "tiles.csv",
            "--variants",
            "B0,D4P8",
            "--output-dir",
            "outputs",
        ]
    )

    assert args.mode == "rollout-only"
    assert args.variants == "B0,D4P8"
```

- [ ] **Step 3: Run tests to verify RED**

Run:

```powershell
python -m pytest tests\test_phase63_set_policy_oracle_pretraining.py -q --basetemp=.pytest_tmp_phase63_task4_red -p no:cacheprovider
```

Expected: analysis, writer, and runner tests fail.

- [ ] **Step 4: Add analysis helpers**

Add `build_phase63_set_policy_analysis(...)` with this contract:

```python
def build_phase63_set_policy_analysis(
    rollout_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    existing_flattened_rows: Sequence[Mapping[str, object]] | None = None,
    existing_flattened_summary_csvs: Sequence[Path | str] | str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    rollout_rows = _load_mapping_rows(rollout_rows_or_csv, "Phase 63 rollout")
    flattened_rows = []
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
        [float(row["bc_minus_flattened_reward"]) for row in delta_rows if row["flattened_reward"] != ""]
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
        "mean_bc_reward_by_variant": _mean_by_field(rollout_rows, "variant_id", "total_contract_reward"),
        "mean_oracle_reward_by_variant": _mean_by_field(rollout_rows, "variant_id", "oracle_total_reward"),
        "mean_flattened_reward_by_variant": _mean_by_field(flattened_rows, "variant_id", "total_contract_reward"),
        "architecture_delta_summary": architecture,
        "d4_b0_delta_summary": d4_b0_summary,
        "d4_d6_delta_summary": d4_d6_summary,
        "oracle_gap_fraction_summary": oracle_gap_summary,
        "coverage_issues": coverage,
        "phase63_set_policy_status": status,
        "conclusion": _phase63_conclusion(status),
        "claim_boundary": PHASE63_CLAIM_BOUNDARY,
    }
```

Add deterministic status rules:

```python
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
        return "set_policy_route_supported" if float(oracle_gap_summary["mean_delta"]) <= 0.2 else "insufficient"
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
```

Add helpers named in the analysis code:

- `_load_mapping_rows(rows_or_csv, label)`
- `_normalize_optional_paths(paths)`
- `_metadata_string_list(metadata, key, fallback)`
- `_metadata_int_list(metadata, key, fallback)`
- `_unique_strings(rows, field)`
- `_unique_ints(rows, field)`
- `_phase63_coverage_issues(rows, variants, eval_tile_ids, seeds)`
- `_flattened_reward_index(rows)`
- `_phase63_delta_rows(rollout_rows, flattened_index)`
- `_paired_variant_delta_rows(rows, comparisons, value_field, output_field)`
- `_numeric_delta_summary(values)`
- `_mean_by_field(rows, group_field, value_field)`
- `_phase63_conclusion(status)`
- `_json_ready(value)`

Use the helper style from `src/paper11_geofm/phase62_d4_d6_matched_ppo.py`: load CSV through `csv.DictReader`, convert `Path` to string in JSON, and round numeric summaries with `_round_float`.

- [ ] **Step 5: Add artifact writer**

Add `write_phase63_set_policy_artifacts(analysis, output_dir)`:

```python
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
        json.dumps(_json_ready(analysis.get("oracle_trajectories", [])), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv_mapping_rows(oracle_summary_csv, PHASE63_ORACLE_FIELDNAMES, analysis.get("oracle_summary_rows", []), "oracle_summary_rows")
    _write_csv_mapping_rows(history_csv, PHASE63_HISTORY_FIELDNAMES, analysis.get("history_rows", []), "history_rows")
    _write_csv_mapping_rows(rollout_csv, PHASE63_ROLLOUT_FIELDNAMES, analysis.get("rollout_rows", []), "rollout_rows")
    _write_csv_mapping_rows(delta_csv, PHASE63_DELTA_FIELDNAMES, analysis.get("delta_rows", []), "delta_rows")
    comparison = {
        key: value
        for key, value in dict(analysis).items()
        if key not in {"oracle_trajectories", "oracle_summary_rows", "history_rows", "rollout_rows", "flattened_rows"}
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
```

Add `_write_csv_mapping_rows(...)` and `_phase63_readiness_markdown(...)`. The Markdown must include:

- title `# Phase 63 Set-Policy Oracle Pretraining`
- status line
- mean behavior-cloned reward by variant
- mean oracle reward by variant
- architecture delta summary
- D4/B0 and D4/D6 delta summaries
- oracle gap summary
- claim boundary

- [ ] **Step 6: Add run wrapper**

Add `run_phase63_set_policy_oracle_pretraining(...)`:

```python
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
        train_tiled = _load_phase63_tiled_variant_input(contract, str(contract["train_tile_id"]), str(variant_id))
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
                eval_tiled = _load_phase63_tiled_variant_input(contract, str(eval_tile_id), str(variant_id))
                oracle = build_phase63_oracle_trajectory(eval_tiled, int(contract["eval_max_steps"]))
                oracle_trajectories.append(oracle)
                oracle_summary_rows.append(_phase63_oracle_summary_row(oracle, seed=int(seed), tile_role="eval"))
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
```

Add `_load_phase63_tiled_variant_input(contract, tile_id, variant_id)` and `_phase63_oracle_summary_row(oracle, seed, tile_role)`.

- [ ] **Step 7: Add CLI runner**

Create `experiments/phase63_set_policy_oracle_pretraining/run_phase63_set_policy_oracle_pretraining.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase63_set_policy_oracle_pretraining import (
    build_phase63_set_policy_analysis,
    run_phase63_set_policy_oracle_pretraining,
    write_phase63_set_policy_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        if args.mode in {"rollout-only", "run-and-analyze"}:
            protocol = run_phase63_set_policy_oracle_pretraining(
                phase2_output_dir=args.phase2_output_dir,
                phase8_output_dir=args.phase8_output_dir,
                phase61_output_dir=args.phase61_output_dir,
                tile_index_csv=args.tile_index_csv,
                variants=args.variants,
                existing_flattened_summary_csvs=args.existing_flattened_summary_csvs if args.mode == "run-and-analyze" else None,
                train_tile_id=args.train_tile_id,
                eval_tile_ids=args.eval_tile_ids,
                max_eval_tiles=args.max_eval_tiles,
                eval_max_steps=args.eval_max_steps,
                seeds=args.seeds,
                bc_epochs=args.bc_epochs,
                learning_rate=args.learning_rate,
                hidden_dim=args.hidden_dim,
                top_k=args.top_k,
            )
        else:
            protocol = build_phase63_set_policy_analysis(
                args.existing_rollout_csv,
                existing_flattened_summary_csvs=args.existing_flattened_summary_csvs,
                metadata={
                    "variants": args.variants,
                    "eval_tile_ids": args.eval_tile_ids,
                    "seeds": args.seeds,
                },
            )
        paths = write_phase63_set_policy_artifacts(protocol, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 63 set-policy status: {protocol['phase63_set_policy_status']}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Rollout CSV: {paths['rollout_csv']}")
    print(f"Delta CSV: {paths['delta_csv']}")
    print(f"Readiness Markdown: {paths['readiness_md']}")
    print(f"Claim boundary: {protocol['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 63 set-policy oracle-pretraining."
    )
    parser.add_argument(
        "--mode",
        choices=("rollout-only", "run-and-analyze", "analyze-only"),
        default="analyze-only",
    )
    parser.add_argument("--phase2-output-dir", type=Path, default=None)
    parser.add_argument("--phase8-output-dir", type=Path, default=None)
    parser.add_argument("--phase61-output-dir", type=Path, default=None)
    parser.add_argument("--tile-index-csv", type=Path, default=None)
    parser.add_argument("--existing-rollout-csv", type=Path, default=None)
    parser.add_argument("--existing-flattened-summary-csvs", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-tile-id", default=None)
    parser.add_argument("--variants", default="B0,D4P8,D4P16,D6R8,D6R16")
    parser.add_argument("--eval-tile-ids", default=None)
    parser.add_argument("--max-eval-tiles", type=int, default=5)
    parser.add_argument("--eval-max-steps", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--bc-epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    missing = []
    if args.mode in {"rollout-only", "run-and-analyze"}:
        for attr, flag in (
            ("phase2_output_dir", "--phase2-output-dir"),
            ("phase8_output_dir", "--phase8-output-dir"),
            ("phase61_output_dir", "--phase61-output-dir"),
            ("tile_index_csv", "--tile-index-csv"),
        ):
            if getattr(args, attr) is None:
                missing.append(flag)
    if args.mode == "analyze-only" and args.existing_rollout_csv is None:
        missing.append("--existing-rollout-csv")
    if missing:
        raise ValueError(f"{args.mode} requires " + ", ".join(missing))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase63_set_policy_oracle_pretraining.py -q --basetemp=.pytest_tmp_phase63_task4_green -p no:cacheprovider
```

Expected: all Phase 63 tests pass.

- [ ] **Step 9: Commit Task 4**

Run:

```powershell
git add src\paper11_geofm\phase63_set_policy_oracle_pretraining.py tests\test_phase63_set_policy_oracle_pretraining.py experiments\phase63_set_policy_oracle_pretraining\run_phase63_set_policy_oracle_pretraining.py
git commit -m "feat: add Phase 63 set policy analysis runner"
```

---

### Task 5: Real Phase 63 Bounded Run and Evidence Record

**Files:**
- Create: `paper/phase28_results/29_phase63_set_policy_oracle_pretraining.md`
- Modify: `paper/phase28_results/README.md`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [ ] **Step 1: Run targeted tests before the real experiment**

Run:

```powershell
python -m pytest tests\test_phase63_set_policy_oracle_pretraining.py tests\test_phase62_d4_d6_matched_ppo.py -q --basetemp=.pytest_tmp_phase63_pre_real -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the real Phase 63 bounded experiment**

Run from the repository root:

```powershell
python experiments\phase63_set_policy_oracle_pretraining\run_phase63_set_policy_oracle_pretraining.py --mode run-and-analyze --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --phase61-output-dir experiments\phase61_d6_geofm_projection_controls\outputs\phase52_full5_seed3 --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,D4P8,D4P16,D6R8,D6R16 --eval-tile-ids tile_r002_c003,tile_r005_c004,tile_r005_c003,tile_r000_c004,tile_r001_c004 --eval-max-steps 8 --seeds 0,1,2 --bc-epochs 80 --learning-rate 0.001 --hidden-dim 64 --top-k 3 --existing-flattened-summary-csvs experiments\phase52_expanded_cluster_replication\outputs\real_bishan_4096_5tiles\phase28_representation_control_summary.csv,experiments\phase62_d4_d6_matched_ppo\outputs\phase52_full5_seed3\phase62_d4_d6_matched_ppo_summary.csv --output-dir experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3
```

Expected generated artifacts:

- `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_oracle_trajectories.json`
- `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_oracle_summary.csv`
- `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_bc_training_history.csv`
- `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_bc_rollout_summary.csv`
- `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_set_policy_comparison.json`
- `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_set_policy_delta_table.csv`
- `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_set_policy_oracle_pretraining.md`

- [ ] **Step 3: Inspect the real status and key numbers**

Run:

```powershell
python -c "import json; p='experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_set_policy_comparison.json'; d=json.load(open(p, encoding='utf-8')); print(d['phase63_set_policy_status']); print(d['architecture_delta_summary']); print(d['d4_b0_delta_summary']); print(d['d4_d6_delta_summary']); print(d['oracle_gap_fraction_summary'])"
```

Expected: prints a Phase 63 status plus four summary dictionaries. If status is `insufficient`, inspect `coverage_issues` before writing the evidence record.

- [ ] **Step 4: Create the Phase 63 evidence document**

Create `paper/phase28_results/29_phase63_set_policy_oracle_pretraining.md` from the generated Markdown:

```powershell
Copy-Item -LiteralPath experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_set_policy_oracle_pretraining.md -Destination paper\phase28_results\29_phase63_set_policy_oracle_pretraining.md
```

Then add this reproduction block near the bottom of the paper-facing evidence document:

````markdown
## Reproduction

Run Phase 63 from the repository root after the Phase 2 B0 table, Phase 8 D4
tables, Phase 61 D6 tables, Phase 13 tile index, Phase 52 flattened PPO
summary, and Phase 62 D4/D6 flattened PPO summary exist:

```powershell
python experiments\phase63_set_policy_oracle_pretraining\run_phase63_set_policy_oracle_pretraining.py --mode run-and-analyze --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --phase61-output-dir experiments\phase61_d6_geofm_projection_controls\outputs\phase52_full5_seed3 --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,D4P8,D4P16,D6R8,D6R16 --eval-tile-ids tile_r002_c003,tile_r005_c004,tile_r005_c003,tile_r000_c004,tile_r001_c004 --eval-max-steps 8 --seeds 0,1,2 --bc-epochs 80 --learning-rate 0.001 --hidden-dim 64 --top-k 3 --existing-flattened-summary-csvs experiments\phase52_expanded_cluster_replication\outputs\real_bishan_4096_5tiles\phase28_representation_control_summary.csv,experiments\phase62_d4_d6_matched_ppo\outputs\phase52_full5_seed3\phase62_d4_d6_matched_ppo_summary.csv --output-dir experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3
```

## Boundary

Phase 63 is algorithm/model evidence under the existing deterministic
base-planning reward. It does not enable suitability reward, does not test
B2/B3, does not test cross-region transfer, does not prove independent
agronomic suitability, does not prove PCA optimality, and does not justify
formal submission-level planning-performance claims. No formal manuscript files
were changed in this phase.
````

- [ ] **Step 5: Update `paper/phase28_results/README.md`**

Add this bullet after the Phase 62 entry:

```markdown
- `29_phase63_set_policy_oracle_pretraining.md`: set-policy oracle-pretraining
  experiment testing whether explicit per-block scoring plus deterministic
  behavior cloning improves base-reward block selection over the flattened PPO
  route under the Phase 52 five-tile, three-seed protocol.
```

Add a short Phase 63 paragraph near the end of the reproduction history:

```markdown
Phase 63 then switches from representation-only diagnosis to algorithm/model
work. It trains a set-style block scorer from deterministic base-reward oracle
trajectories and rolls the behavior-cloned policy out on the same five held-out
tiles and three seeds. The status, architecture delta, D4/B0 delta, D4/D6
delta, and oracle-gap summaries are recorded in
`29_phase63_set_policy_oracle_pretraining.md`.
```

- [ ] **Step 6: Update the handoff document**

In `docs/superpowers/phase33_current_progress_handoff.md`, add a Phase 63 save block containing:

```markdown
## Phase 63 Set-Policy Oracle Pretraining

- Branch: `main`
- Formal manuscript files changed: no
- Implementation module: `src/paper11_geofm/phase63_set_policy_oracle_pretraining.py`
- Runner: `experiments/phase63_set_policy_oracle_pretraining/run_phase63_set_policy_oracle_pretraining.py`
- Evidence document: `paper/phase28_results/29_phase63_set_policy_oracle_pretraining.md`
- Generated output directory: `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3`
- Claim boundary: base-reward algorithm/model evidence only; no suitability reward, no B2/B3, no transfer, no PCA-optimality claim, no formal submission-level performance claim.
```

Add the exact `phase63_set_policy_status` and summary values printed in Step 3.

- [ ] **Step 7: Commit Task 5**

Run:

```powershell
git add paper\phase28_results\29_phase63_set_policy_oracle_pretraining.md paper\phase28_results\README.md docs\superpowers\phase33_current_progress_handoff.md
git commit -m "docs: record Phase 63 set policy evidence"
```

---

### Task 6: Final Verification and Push

**Files:**
- No new files unless verification exposes a real defect.

- [ ] **Step 1: Run targeted Phase 63 and regression tests**

Run:

```powershell
python -m pytest tests\test_phase63_set_policy_oracle_pretraining.py tests\test_phase62_d4_d6_matched_ppo.py tests\test_phase61_d6_geofm_projection_controls.py -q --basetemp=.pytest_tmp_phase63_final -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 2: Run repository smoke check**

Run:

```powershell
python scripts\smoke_check.py
```

Expected: smoke check passes.

- [ ] **Step 3: Run diff whitespace check**

Run:

```powershell
git diff --check
```

Expected: no output.

- [ ] **Step 4: Confirm formal manuscript files were not modified**

Run:

```powershell
git diff --name-only HEAD -- paper\submission\final
```

Expected: no output.

- [ ] **Step 5: Inspect Git state**

Run:

```powershell
git status --short --branch
```

Expected: branch is `main`; working tree is clean after commits.

- [ ] **Step 6: Push saved Phase 63 work**

Run:

```powershell
git push
```

Expected: `origin/main` receives the Phase 63 implementation and evidence commits.

---

## Execution Notes

- Keep Phase 63 BC-only for this implementation plan. Do not add PPO fine-tuning in these tasks.
- Keep generated outputs under `experiments/phase63_set_policy_oracle_pretraining/outputs/`.
- Do not strengthen the Paper11 manuscript claim from Phase 63 until evidence is reviewed after the real run.
- Do not enable suitability reward, B2/B3, transfer experiments, or formal manuscript edits in this phase.
- If `apply_patch` fails during execution on Windows, create a unified diff and apply it with `git apply --ignore-whitespace`; do not rewrite unrelated files.
