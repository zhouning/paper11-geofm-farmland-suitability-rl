# Phase 65 Standardized Set-Policy BC Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 65 as a controlled train-tile-fitted standardized set-policy BC rerun that keeps raw-feature oracle targets and raw-feature rewards comparable to Phase 63.

**Architecture:** Add one focused Phase 65 module that wraps Phase 63's set-policy scorer and analysis patterns while separating model-input matrices from raw reward matrices. A CLI runner will load the Phase 63 contract, fit one z-score transform per variant on the train tile, train and roll out standardized-input BC policies, compare them against Phase 63 unstandardized rollout rows, and write auditable JSON/CSV/Markdown artifacts.

**Tech Stack:** Python 3, NumPy, PyTorch, existing `paper11_geofm` tiled input loaders, existing Phase 63 set-policy scorer and analysis helpers, pytest, PowerShell commands using `D:\adk\.venv\Scripts\python.exe`.

---

## File Structure

- Create `src/paper11_geofm/phase65_standardized_set_policy_bc_rerun.py`
  - Owns Phase 65 claim boundary, standardization transform, standardized-input BC training, raw-reward rollout, paired Phase 63 comparison, status gate, artifact writing, and full run orchestration.
- Create `experiments/phase65_standardized_set_policy_bc_rerun/run_phase65_standardized_set_policy_bc_rerun.py`
  - Thin CLI wrapper, matching the Phase 64 runner style. It should call the Phase 65 module and print status plus artifact paths.
- Create `tests/test_phase65_standardized_set_policy_bc_rerun.py`
  - Covers train-only standardization, raw reward preservation, BC/rollout behavior on tiny fixtures, pairwise deltas, status gates, writer outputs, and CLI parser behavior.
- Create `paper/phase28_results/31_phase65_standardized_set_policy_bc_rerun.md`
  - Filled after the real Phase 65 run. It should report status and claim boundary only after generated artifacts exist.
- Do not modify `paper/submission/final/*`.

---

### Task 1: Standardizer And Raw-Reward Invariants

**Files:**
- Create: `src/paper11_geofm/phase65_standardized_set_policy_bc_rerun.py`
- Create: `tests/test_phase65_standardized_set_policy_bc_rerun.py`

- [ ] **Step 1: Write failing tests for train-fitted standardization**

Create `tests/test_phase65_standardized_set_policy_bc_rerun.py` with these initial tests and helpers:

```python
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _tiled_input(matrix, variant_id="D4P8", tile_id="tile_train"):
    from paper11_geofm.tiled_inputs import TiledVariantInput

    array = np.asarray(matrix, dtype=np.float32)
    feature_columns = tuple(f"feature_{index:02d}" for index in range(array.shape[1]))
    block_ids = tuple(f"b{index}" for index in range(array.shape[0]))
    return TiledVariantInput(
        tile_id=tile_id,
        variant_id=variant_id,
        block_ids=block_ids,
        feature_columns=feature_columns,
        state_matrix=array,
        reward_mode="base_planning_reward",
        state_groups=("synthetic",),
        source_table=Path(f"variant_{variant_id}_features.csv"),
        tile_index_csv=Path("tiles.csv"),
    )


def test_phase65_standardizer_fits_train_tile_and_applies_to_eval_without_eval_stats():
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        apply_phase65_standardizer,
        fit_phase65_train_tile_standardizer,
    )

    train = _tiled_input(
        [
            [1.0, 10.0, 5.0],
            [3.0, 14.0, 5.0],
            [5.0, 18.0, 5.0],
        ],
        variant_id="D4P8",
        tile_id="tile_train",
    )
    eval_tile = _tiled_input(
        [
            [7.0, 22.0, 5.0],
            [9.0, 26.0, 5.0],
        ],
        variant_id="D4P8",
        tile_id="tile_eval",
    )

    transform = fit_phase65_train_tile_standardizer(train)
    standardized_train = apply_phase65_standardizer(train, transform)
    standardized_eval = apply_phase65_standardizer(eval_tile, transform)

    np.testing.assert_allclose(transform.mean, np.array([3.0, 14.0, 5.0]))
    np.testing.assert_allclose(transform.safe_std[2], 1.0)
    np.testing.assert_allclose(
        standardized_train.state_matrix.mean(axis=0),
        np.array([0.0, 0.0, 0.0]),
        atol=1.0e-6,
    )
    expected_eval_first = np.array(
        [
            (7.0 - transform.mean[0]) / transform.safe_std[0],
            (22.0 - transform.mean[1]) / transform.safe_std[1],
            0.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(standardized_eval.state_matrix[0], expected_eval_first)
    assert standardized_eval.tile_id == "tile_eval"
    assert standardized_eval.variant_id == "D4P8"
    assert standardized_eval.block_ids == eval_tile.block_ids
    assert standardized_eval.feature_columns == eval_tile.feature_columns


def test_phase65_standardizer_rejects_mismatched_variant_and_columns():
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        apply_phase65_standardizer,
        fit_phase65_train_tile_standardizer,
    )

    transform = fit_phase65_train_tile_standardizer(
        _tiled_input([[1.0, 2.0], [3.0, 4.0]], variant_id="D4P8")
    )
    mismatched_variant = _tiled_input(
        [[1.0, 2.0], [3.0, 4.0]],
        variant_id="D6R8",
        tile_id="tile_eval",
    )
    mismatched_columns = _tiled_input(
        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
        variant_id="D4P8",
        tile_id="tile_eval",
    )

    try:
        apply_phase65_standardizer(mismatched_variant, transform)
    except ValueError as exc:
        assert "variant" in str(exc)
    else:
        raise AssertionError("Expected variant mismatch to fail")

    try:
        apply_phase65_standardizer(mismatched_columns, transform)
    except ValueError as exc:
        assert "feature columns" in str(exc)
    else:
        raise AssertionError("Expected feature-column mismatch to fail")


def test_phase65_standardized_inputs_do_not_change_raw_reward_or_oracle_targets():
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        build_phase63_oracle_trajectory,
    )
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        apply_phase65_standardizer,
        build_phase65_bc_examples,
        fit_phase65_train_tile_standardizer,
    )
    from paper11_geofm.tiled_inputs import TiledVariantInput

    columns = (
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
    matrix = np.zeros((3, len(columns)), dtype=np.float32)
    score_index = columns.index("explicit_feature_16")
    matrix[:, score_index] = np.array([0.9, 0.5, 0.1], dtype=np.float32)
    matrix[:, columns.index("explicit_feature_00")] = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    raw = TiledVariantInput(
        tile_id="tile_train",
        variant_id="D4P8",
        block_ids=("b1", "b2", "b3"),
        feature_columns=columns,
        state_matrix=matrix,
        reward_mode="base_planning_reward",
        state_groups=("synthetic",),
        source_table=Path("variant_D4P8_features.csv"),
        tile_index_csv=Path("tiles.csv"),
    )
    transform = fit_phase65_train_tile_standardizer(raw)
    standardized = apply_phase65_standardizer(raw, transform)

    raw_oracle = build_phase63_oracle_trajectory(raw, eval_max_steps=2)
    examples = build_phase65_bc_examples(raw, transform, eval_max_steps=2)

    assert not np.allclose(standardized.state_matrix, raw.state_matrix)
    assert [example["target_action"] for example in examples] == raw_oracle["action_indices"]
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase65_standardized_set_policy_bc_rerun.py -q --basetemp=.pytest_tmp_phase65_task1_red -p no:cacheprovider
```

Expected: `ModuleNotFoundError` or `ImportError` for `paper11_geofm.phase65_standardized_set_policy_bc_rerun`.

- [ ] **Step 3: Add the Phase 65 module with standardizer and BC examples**

Create `src/paper11_geofm/phase65_standardized_set_policy_bc_rerun.py` with this starting content:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, replace
import json
from os import PathLike
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


def fit_phase65_train_tile_standardizer(tiled_input, epsilon: float = PHASE65_EPSILON) -> Phase65Standardizer:
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
            f"Phase 65 standardizer variant mismatch: {tiled_input.variant_id} != {standardizer.variant_id}"
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


def build_phase65_bc_examples(raw_tiled_input, standardizer: Phase65Standardizer, eval_max_steps: int) -> list[dict[str, object]]:
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
```

- [ ] **Step 4: Run tests and verify Task 1 passes**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase65_standardized_set_policy_bc_rerun.py -q --basetemp=.pytest_tmp_phase65_task1_green -p no:cacheprovider
```

Expected: the three Phase 65 tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src\paper11_geofm\phase65_standardized_set_policy_bc_rerun.py tests\test_phase65_standardized_set_policy_bc_rerun.py
git commit -m "feat: add Phase 65 train-fitted standardizer"
```

Expected: commit succeeds with the new source and test file.

---

### Task 2: Standardized-Input Training And Raw-Reward Rollout

**Files:**
- Modify: `src/paper11_geofm/phase65_standardized_set_policy_bc_rerun.py`
- Modify: `tests/test_phase65_standardized_set_policy_bc_rerun.py`

- [ ] **Step 1: Add failing tests for training and rollout**

Append these tests to `tests/test_phase65_standardized_set_policy_bc_rerun.py`:

```python
def _reward_tiled_input(
    block_ids=("b3", "b1", "b2", "b4"),
    scores=(0.2, 0.9, 0.7, 0.1),
    scale_feature=(100.0, 200.0, 300.0, 400.0),
    variant_id="D4P8",
    tile_id="tile_eval",
):
    columns = (
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
    matrix = np.zeros((len(block_ids), len(columns)), dtype=np.float32)
    score_index = columns.index("explicit_feature_16")
    scale_index = columns.index("explicit_feature_00")
    for row_index, score in enumerate(scores):
        matrix[row_index, score_index] = float(score)
        matrix[row_index, scale_index] = float(scale_feature[row_index])
    from paper11_geofm.tiled_inputs import TiledVariantInput

    return TiledVariantInput(
        tile_id=tile_id,
        variant_id=variant_id,
        block_ids=tuple(block_ids),
        feature_columns=columns,
        state_matrix=matrix,
        reward_mode="base_planning_reward",
        state_groups=("synthetic",),
        source_table=Path(f"variant_{variant_id}_features.csv"),
        tile_index_csv=Path("tiles.csv"),
    )


def test_phase65_behavior_cloning_loss_decreases_with_standardized_inputs():
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        fit_phase65_train_tile_standardizer,
        train_phase65_behavior_cloner,
    )

    raw = _reward_tiled_input()
    transform = fit_phase65_train_tile_standardizer(raw)
    model, history = train_phase65_behavior_cloner(
        raw,
        transform,
        seed=65,
        eval_max_steps=3,
        epochs=30,
        learning_rate=0.01,
        hidden_dim=16,
        top_k=2,
    )

    assert model.n_features == len(raw.feature_columns)
    assert len(history) == 30
    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["topk_hit_rate"] >= history[0]["topk_hit_rate"]
    assert history[-1]["claim_boundary"].startswith("Phase 65")


def test_phase65_rollout_uses_standardized_logits_and_raw_rewards():
    from paper11_geofm.phase63_set_policy_oracle_pretraining import (
        build_phase63_oracle_trajectory,
    )
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        fit_phase65_train_tile_standardizer,
        rollout_phase65_greedy_policy,
        train_phase65_behavior_cloner,
    )

    raw = _reward_tiled_input()
    transform = fit_phase65_train_tile_standardizer(raw)
    model, _history = train_phase65_behavior_cloner(
        raw,
        transform,
        seed=65,
        eval_max_steps=3,
        epochs=35,
        learning_rate=0.01,
        hidden_dim=16,
        top_k=2,
    )
    rollout = rollout_phase65_greedy_policy(
        model,
        raw_tiled_input=raw,
        standardizer=transform,
        train_tile_id="tile_train",
        eval_tile_rank=1,
        seed=65,
        phase65_seed_rank=1,
        eval_max_steps=3,
    )
    oracle = build_phase63_oracle_trajectory(raw, eval_max_steps=3)

    assert rollout["row_type"] == "bc_greedy_policy"
    assert rollout["all_actions_valid"] is True
    assert rollout["invalid_action_count"] == 0
    assert rollout["oracle_total_reward"] == oracle["total_oracle_reward"]
    assert float(rollout["total_contract_reward"]) > 0.0
    assert rollout["claim_boundary"].startswith("Phase 65")
```

- [ ] **Step 2: Run the two new tests and verify they fail**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase65_standardized_set_policy_bc_rerun.py::test_phase65_behavior_cloning_loss_decreases_with_standardized_inputs tests\test_phase65_standardized_set_policy_bc_rerun.py::test_phase65_rollout_uses_standardized_logits_and_raw_rewards -q --basetemp=.pytest_tmp_phase65_task2_red -p no:cacheprovider
```

Expected: failures because `train_phase65_behavior_cloner` and `rollout_phase65_greedy_policy` do not exist.

- [ ] **Step 3: Implement standardized-input BC training and raw-reward rollout**

Append these functions to `src/paper11_geofm/phase65_standardized_set_policy_bc_rerun.py`:

```python
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
```

- [ ] **Step 4: Run Task 2 tests and verify they pass**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase65_standardized_set_policy_bc_rerun.py -q --basetemp=.pytest_tmp_phase65_task2_green -p no:cacheprovider
```

Expected: all current Phase 65 tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src\paper11_geofm\phase65_standardized_set_policy_bc_rerun.py tests\test_phase65_standardized_set_policy_bc_rerun.py
git commit -m "feat: add Phase 65 standardized BC rollout"
```

Expected: commit succeeds.

---

### Task 3: Pairwise Standardization Comparison And Status Gate

**Files:**
- Modify: `src/paper11_geofm/phase65_standardized_set_policy_bc_rerun.py`
- Modify: `tests/test_phase65_standardized_set_policy_bc_rerun.py`

- [ ] **Step 1: Add failing tests for pairwise deltas and status labels**

Append these helpers and tests:

```python
def _rollout_row(variant_id, reward, tile_id="tile_a", seed=0, gap_fraction=0.1):
    return {
        "row_type": "bc_greedy_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": tile_id,
        "eval_tile_rank": 1,
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
        "oracle_total_reward": 1.5,
        "oracle_gap": 1.5 - reward,
        "oracle_gap_fraction": gap_fraction,
        "selected_block_ids": "b1;b2;b3",
        "selected_action_indices": "0;1;2",
        "claim_boundary": "fixture",
    }


def test_phase65_pairwise_delta_reports_standardized_minus_unstandardized():
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        build_phase65_standardization_pairwise_rows,
    )

    standardized = [_rollout_row("D4P8", 1.30), _rollout_row("B0", 1.10)]
    unstandardized = [_rollout_row("D4P8", 1.00), _rollout_row("B0", 1.20)]

    rows, coverage = build_phase65_standardization_pairwise_rows(
        standardized,
        unstandardized,
        variants=["B0", "D4P8"],
        eval_tile_ids=["tile_a"],
        seeds=[0],
    )

    assert coverage["missing_standardized_rows"] == []
    assert coverage["missing_unstandardized_rows"] == []
    d4 = [row for row in rows if row["variant_id"] == "D4P8"][0]
    assert d4["standardized_minus_unstandardized_reward"] == 0.3
    assert d4["self_improves_unstandardized"] is True


def test_phase65_status_gate_covers_supported_all_variant_not_helpful_and_insufficient():
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        build_phase65_standardization_comparison,
    )

    variants = ["B0", "D4P8", "D4P16", "D6R8", "D6R16"]
    old_rows = [_rollout_row(variant, 1.0) for variant in variants]
    geofm_rows = [
        _rollout_row("B0", 1.05),
        _rollout_row("D4P8", 1.40),
        _rollout_row("D4P16", 1.35),
        _rollout_row("D6R8", 1.20),
        _rollout_row("D6R16", 1.25),
    ]
    all_variant_rows = [
        _rollout_row("B0", 1.30),
        _rollout_row("D4P8", 1.10),
        _rollout_row("D4P16", 1.12),
        _rollout_row("D6R8", 1.18),
        _rollout_row("D6R16", 1.19),
    ]
    not_helpful_rows = [_rollout_row(variant, 0.90) for variant in variants]

    geofm = build_phase65_standardization_comparison(
        geofm_rows,
        old_rows,
        variants=variants,
        eval_tile_ids=["tile_a"],
        seeds=[0],
    )
    all_variant = build_phase65_standardization_comparison(
        all_variant_rows,
        old_rows,
        variants=variants,
        eval_tile_ids=["tile_a"],
        seeds=[0],
    )
    not_helpful = build_phase65_standardization_comparison(
        not_helpful_rows,
        old_rows,
        variants=variants,
        eval_tile_ids=["tile_a"],
        seeds=[0],
    )
    insufficient = build_phase65_standardization_comparison(
        geofm_rows[:-1],
        old_rows,
        variants=variants,
        eval_tile_ids=["tile_a"],
        seeds=[0],
    )

    assert geofm["phase65_status"] == "standardization_improves_geofm_set_policy"
    assert all_variant["phase65_status"] == "standardization_improves_all_variants_no_geofm_advantage"
    assert not_helpful["phase65_status"] == "standardization_not_helpful"
    assert insufficient["phase65_status"] == "insufficient"
```

- [ ] **Step 2: Run the new comparison tests and verify they fail**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase65_standardized_set_policy_bc_rerun.py::test_phase65_pairwise_delta_reports_standardized_minus_unstandardized tests\test_phase65_standardized_set_policy_bc_rerun.py::test_phase65_status_gate_covers_supported_all_variant_not_helpful_and_insufficient -q --basetemp=.pytest_tmp_phase65_task3_red -p no:cacheprovider
```

Expected: failures because the comparison functions do not exist.

- [ ] **Step 3: Implement pairwise comparison and status gate**

Append these functions and fieldnames to the Phase 65 module:

```python
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


def _index_rollout_rows(rows: Sequence[Mapping[str, object]]) -> tuple[dict[tuple[str, str, int], Mapping[str, object]], list[dict[str, object]]]:
    index: dict[tuple[str, str, int], Mapping[str, object]] = {}
    duplicates: list[dict[str, object]] = []
    for row in rows:
        if str(row.get("row_type", "")) != "bc_greedy_policy":
            continue
        key = _rollout_key(row)
        if key in index:
            duplicates.append(
                {
                    "variant_id": key[0],
                    "eval_tile_id": key[1],
                    "seed": key[2],
                }
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
            missing_standardized.append({"variant_id": key[0], "eval_tile_id": key[1], "seed": key[2]})
            continue
        if unstandardized is None:
            missing_unstandardized.append({"variant_id": key[0], "eval_tile_id": key[1], "seed": key[2]})
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
                "standardized_minus_unstandardized_oracle_gap_fraction": _round_float(standardized_gap - unstandardized_gap),
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
```

- [ ] **Step 4: Run Task 3 tests and verify they pass**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase65_standardized_set_policy_bc_rerun.py -q --basetemp=.pytest_tmp_phase65_task3_green -p no:cacheprovider
```

Expected: all current Phase 65 tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add src\paper11_geofm\phase65_standardized_set_policy_bc_rerun.py tests\test_phase65_standardized_set_policy_bc_rerun.py
git commit -m "feat: add Phase 65 standardization comparison gate"
```

Expected: commit succeeds.

---

### Task 4: Artifact Writer And CLI Runner

**Files:**
- Modify: `src/paper11_geofm/phase65_standardized_set_policy_bc_rerun.py`
- Create: `experiments/phase65_standardized_set_policy_bc_rerun/run_phase65_standardized_set_policy_bc_rerun.py`
- Modify: `tests/test_phase65_standardized_set_policy_bc_rerun.py`

- [ ] **Step 1: Add failing tests for writer and CLI parser**

Append these tests:

```python
def test_phase65_writer_outputs_json_csv_and_markdown(tmp_path):
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        build_phase65_standardization_comparison,
        write_phase65_artifacts,
    )

    variants = ["B0", "D4P8"]
    standardized_rows = [_rollout_row("B0", 1.1), _rollout_row("D4P8", 1.3)]
    unstandardized_rows = [_rollout_row("B0", 1.0), _rollout_row("D4P8", 1.0)]
    comparison = build_phase65_standardization_comparison(
        standardized_rows,
        unstandardized_rows,
        variants=variants,
        eval_tile_ids=["tile_a"],
        seeds=[0],
    )
    analysis = {
        "phase": "phase65_standardized_set_policy_bc_rerun",
        "standardization_stats": [
            {
                "variant_id": "D4P8",
                "train_tile_id": "tile_train",
                "zero_variance_feature_count": 0,
                "claim_boundary": "Phase 65",
            }
        ],
        "history_rows": [],
        "rollout_rows": standardized_rows,
        "phase63_style_analysis": {
            "mean_bc_reward_by_variant": {"B0": 1.1, "D4P8": 1.3},
            "oracle_gap_fraction_summary": {"mean_delta": 0.1},
        },
        "standardization_comparison": comparison,
        "claim_boundary": "Phase 65",
    }

    paths = write_phase65_artifacts(analysis, tmp_path / "outputs")

    assert paths["standardization_stats_json"].name == "phase65_standardization_stats.json"
    assert paths["history_csv"].name == "phase65_bc_training_history.csv"
    assert paths["rollout_csv"].name == "phase65_bc_rollout_summary.csv"
    assert paths["comparison_json"].name == "phase65_set_policy_comparison.json"
    assert paths["pairwise_delta_csv"].name == "phase65_standardization_pairwise_delta.csv"
    assert paths["readiness_md"].name == "phase65_standardized_set_policy_bc_rerun.md"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase65_status"] in {
        "standardization_improves_geofm_set_policy",
        "standardization_improves_all_variants_no_geofm_advantage",
        "standardization_not_helpful",
        "standardization_hurts_or_inconclusive",
        "insufficient",
    }
    markdown = paths["readiness_md"].read_text(encoding="utf-8")
    assert "Phase 65 Standardized Set-Policy BC Rerun" in markdown


def test_phase65_cli_parser_accepts_required_inputs():
    runner_path = (
        ROOT
        / "experiments"
        / "phase65_standardized_set_policy_bc_rerun"
        / "run_phase65_standardized_set_policy_bc_rerun.py"
    )
    spec = importlib.util.spec_from_file_location("phase65_runner_args", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--phase63-comparison-json",
            "phase63_set_policy_comparison.json",
            "--phase63-rollout-csv",
            "phase63_bc_rollout_summary.csv",
            "--existing-flattened-summary-csvs",
            "phase52.csv,phase62.csv",
            "--output-dir",
            "outputs",
        ]
    )

    assert args.phase63_comparison_json == Path("phase63_set_policy_comparison.json")
    assert args.phase63_rollout_csv == Path("phase63_bc_rollout_summary.csv")
    assert args.output_dir == Path("outputs")
```

- [ ] **Step 2: Run writer and CLI tests and verify they fail**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase65_standardized_set_policy_bc_rerun.py::test_phase65_writer_outputs_json_csv_and_markdown tests\test_phase65_standardized_set_policy_bc_rerun.py::test_phase65_cli_parser_accepts_required_inputs -q --basetemp=.pytest_tmp_phase65_task4_red -p no:cacheprovider
```

Expected: failures because `write_phase65_artifacts` and the runner file do not exist.

- [ ] **Step 3: Implement writer helpers**

Append these helpers to the Phase 65 module:

```python
def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
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
        json.dumps(_json_ready(analysis.get("standardization_stats", [])), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv_rows(paths["history_csv"], PHASE63_HISTORY_FIELDNAMES, analysis.get("history_rows", []))
    _write_csv_rows(paths["rollout_csv"], PHASE63_ROLLOUT_FIELDNAMES, analysis.get("rollout_rows", []))
    pairwise_rows = dict(analysis.get("standardization_comparison", {})).get("pairwise_delta_rows", [])
    _write_csv_rows(paths["pairwise_delta_csv"], PHASE65_PAIRWISE_FIELDNAMES, pairwise_rows)
    comparison = {
        key: value
        for key, value in dict(analysis).items()
        if key not in {"history_rows", "rollout_rows"}
    }
    status = dict(analysis.get("standardization_comparison", {})).get("phase65_status", PHASE65_STATUS_INSUFFICIENT)
    comparison["phase65_status"] = status
    paths["comparison_json"].write_text(
        json.dumps(_json_ready(comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["readiness_md"].write_text(_phase65_markdown(analysis), encoding="utf-8")
    return paths
```

- [ ] **Step 4: Create the CLI runner**

Create `experiments/phase65_standardized_set_policy_bc_rerun/run_phase65_standardized_set_policy_bc_rerun.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
    run_phase65_standardized_set_policy_bc_rerun,
    write_phase65_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = run_phase65_standardized_set_policy_bc_rerun(
            phase63_comparison_json=args.phase63_comparison_json,
            phase63_rollout_csv=args.phase63_rollout_csv,
            existing_flattened_summary_csvs=args.existing_flattened_summary_csvs,
        )
        paths = write_phase65_artifacts(analysis, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    comparison = analysis["standardization_comparison"]
    print(f"Phase 65 status: {comparison['phase65_status']}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Rollout CSV: {paths['rollout_csv']}")
    print(f"Pairwise Delta CSV: {paths['pairwise_delta_csv']}")
    print(f"Readiness Markdown: {paths['readiness_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 65 standardized set-policy BC rerun."
    )
    parser.add_argument("--phase63-comparison-json", type=Path, required=True)
    parser.add_argument("--phase63-rollout-csv", type=Path, required=True)
    parser.add_argument("--existing-flattened-summary-csvs", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Add a temporary stub for run orchestration so parser imports**

Append this function to the module. Task 5 replaces the body with the real run loop.

```python
def run_phase65_standardized_set_policy_bc_rerun(
    phase63_comparison_json: Path | str,
    phase63_rollout_csv: Path | str,
    existing_flattened_summary_csvs: Sequence[Path | str] | str | None = None,
) -> dict[str, object]:
    raise ValueError("Phase 65 run orchestration requires a Phase 63 comparison JSON with contract metadata")
```

- [ ] **Step 6: Run tests and verify writer/parser pass**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase65_standardized_set_policy_bc_rerun.py -q --basetemp=.pytest_tmp_phase65_task4_green -p no:cacheprovider
```

Expected: all current Phase 65 tests pass.

- [ ] **Step 7: Commit Task 4**

Run:

```powershell
git add src\paper11_geofm\phase65_standardized_set_policy_bc_rerun.py experiments\phase65_standardized_set_policy_bc_rerun\run_phase65_standardized_set_policy_bc_rerun.py tests\test_phase65_standardized_set_policy_bc_rerun.py
git commit -m "feat: add Phase 65 artifact writer and CLI"
```

Expected: commit succeeds.

---

### Task 5: Full Phase 65 Run Orchestration

**Files:**
- Modify: `src/paper11_geofm/phase65_standardized_set_policy_bc_rerun.py`
- Modify: `tests/test_phase65_standardized_set_policy_bc_rerun.py`

- [ ] **Step 1: Add failing integration test on tiny artifacts**

Append this test:

```python
def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_variant_fixture(output_dir: Path, variant_id: str, rows: list[dict[str, float]], columns: tuple[str, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table = output_dir / f"variant_{variant_id}_features.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=["block_id", *columns])
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "variants": {
            variant_id: {
                "ready": True,
                "feature_table": table.name,
                "required_columns": list(columns),
                "reward": "base_planning_reward",
                "state_groups": ["synthetic"],
            }
        }
    }
    (output_dir / "experiment_variants.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_phase65_run_wrapper_loads_phase63_contract_and_writes_comparable_rows(tmp_path):
    from paper11_geofm.phase65_standardized_set_policy_bc_rerun import (
        run_phase65_standardized_set_policy_bc_rerun,
    )

    columns = (
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
    feature_rows = [
        {**{"block_id": "b1"}, **{column: 0.0 for column in columns}},
        {**{"block_id": "b2"}, **{column: 0.0 for column in columns}},
        {**{"block_id": "b3"}, **{column: 0.0 for column in columns}},
    ]
    feature_rows[0]["explicit_feature_16"] = 0.9
    feature_rows[1]["explicit_feature_16"] = 0.8
    feature_rows[2]["explicit_feature_16"] = 0.2
    phase2 = tmp_path / "phase2"
    _write_variant_fixture(phase2, "B0", feature_rows, columns)
    tile_index = _write_csv(
        tmp_path / "tiles.csv",
        [
            {"tile_id": "tile_train", "block_ids": "b1;b2;b3"},
            {"tile_id": "tile_eval", "block_ids": "b1;b2;b3"},
        ],
    )
    comparison = {
        "contract": {
            "phase2_output_dir": str(phase2),
            "phase8_output_dir": str(tmp_path / "phase8"),
            "phase61_output_dir": str(tmp_path / "phase61"),
            "tile_index_csv": str(tile_index),
            "variant_source_dirs": {"B0": str(phase2)},
            "variants": ["B0"],
            "train_tile_id": "tile_train",
            "eval_tile_ids": ["tile_eval"],
            "eval_tile_ranks": {"tile_eval": 1},
            "seeds": [0],
            "seed_ranks": {"0": 1},
            "eval_max_steps": 2,
            "bc_epochs": 8,
            "learning_rate": 0.01,
            "hidden_dim": 12,
            "top_k": 2,
        }
    }
    comparison_path = tmp_path / "phase63_set_policy_comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    old_rollout = _write_csv(
        tmp_path / "phase63_rollout.csv",
        [_rollout_row("B0", 0.1, tile_id="tile_eval", seed=0)],
    )

    analysis = run_phase65_standardized_set_policy_bc_rerun(
        phase63_comparison_json=comparison_path,
        phase63_rollout_csv=old_rollout,
    )

    assert analysis["phase"] == "phase65_standardized_set_policy_bc_rerun"
    assert len(analysis["standardization_stats"]) == 1
    assert len(analysis["history_rows"]) == 8
    assert len(analysis["rollout_rows"]) == 1
    assert analysis["standardization_comparison"]["coverage_issues"]["missing_standardized_rows"] == []
```

- [ ] **Step 2: Run the integration test and verify it fails**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase65_standardized_set_policy_bc_rerun.py::test_phase65_run_wrapper_loads_phase63_contract_and_writes_comparable_rows -q --basetemp=.pytest_tmp_phase65_task5_red -p no:cacheprovider
```

Expected: failure from the temporary orchestration stub.

- [ ] **Step 3: Replace the orchestration stub with real loading and run logic**

Replace the stub with these helpers and function:

```python
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


def _load_phase65_tiled_variant_input(contract: Mapping[str, object], tile_id: str, variant_id: str):
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
    phase63_comparison = _load_json_object(phase63_comparison_json, "Phase 63 comparison JSON")
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
                raw_eval = _load_phase65_tiled_variant_input(contract, eval_tile_id, variant_id)
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
```

- [ ] **Step 4: Run the integration test and verify it passes**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase65_standardized_set_policy_bc_rerun.py::test_phase65_run_wrapper_loads_phase63_contract_and_writes_comparable_rows -q --basetemp=.pytest_tmp_phase65_task5_green -p no:cacheprovider
```

Expected: test passes.

- [ ] **Step 5: Run all Phase 65 tests**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase65_standardized_set_policy_bc_rerun.py -q --basetemp=.pytest_tmp_phase65_task5_all -p no:cacheprovider
```

Expected: all Phase 65 tests pass.

- [ ] **Step 6: Commit Task 5**

Run:

```powershell
git add src\paper11_geofm\phase65_standardized_set_policy_bc_rerun.py tests\test_phase65_standardized_set_policy_bc_rerun.py
git commit -m "feat: add Phase 65 full rerun orchestration"
```

Expected: commit succeeds.

---

### Task 6: Real Phase 65 Run And Result Note

**Files:**
- Create: `paper/phase28_results/31_phase65_standardized_set_policy_bc_rerun.md`
- Generated ignored outputs under: `experiments/phase65_standardized_set_policy_bc_rerun/outputs/phase52_full5_seed3/`

- [ ] **Step 1: Run the full Phase 65 experiment**

Run from repository root:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase65_standardized_set_policy_bc_rerun\run_phase65_standardized_set_policy_bc_rerun.py --phase63-comparison-json experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_set_policy_comparison.json --phase63-rollout-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_bc_rollout_summary.csv --existing-flattened-summary-csvs experiments\phase52_expanded_cluster_replication\outputs\real_bishan_4096_5tiles\phase28_representation_control_summary.csv,experiments\phase62_d4_d6_matched_ppo\outputs\phase52_full5_seed3\phase62_d4_d6_matched_ppo_summary.csv --output-dir experiments\phase65_standardized_set_policy_bc_rerun\outputs\phase52_full5_seed3
```

Expected: exit code `0`, console prints `Phase 65 status:` plus paths for comparison JSON, rollout CSV, pairwise delta CSV, readiness Markdown, and claim boundary.

- [ ] **Step 2: Inspect generated Phase 65 status**

Run:

```powershell
Get-Content -Raw experiments\phase65_standardized_set_policy_bc_rerun\outputs\phase52_full5_seed3\phase65_set_policy_comparison.json
```

Expected: JSON contains `phase65_status`, `standardization_comparison`, `phase63_style_analysis`, and `claim_boundary`.

- [ ] **Step 3: Write the tracked result note**

Create `paper/phase28_results/31_phase65_standardized_set_policy_bc_rerun.md` from the generated Markdown and append the reproduction command:

```powershell
Copy-Item -LiteralPath experiments\phase65_standardized_set_policy_bc_rerun\outputs\phase52_full5_seed3\phase65_standardized_set_policy_bc_rerun.md -Destination paper\phase28_results\31_phase65_standardized_set_policy_bc_rerun.md
Add-Content -LiteralPath paper\phase28_results\31_phase65_standardized_set_policy_bc_rerun.md -Value @'

## Reproduction

Run Phase 65 from the repository root after Phase 63 full-run artifacts exist:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase65_standardized_set_policy_bc_rerun\run_phase65_standardized_set_policy_bc_rerun.py --phase63-comparison-json experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_set_policy_comparison.json --phase63-rollout-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_bc_rollout_summary.csv --existing-flattened-summary-csvs experiments\phase52_expanded_cluster_replication\outputs\real_bishan_4096_5tiles\phase28_representation_control_summary.csv,experiments\phase62_d4_d6_matched_ppo\outputs\phase52_full5_seed3\phase62_d4_d6_matched_ppo_summary.csv --output-dir experiments\phase65_standardized_set_policy_bc_rerun\outputs\phase52_full5_seed3
```

## Boundary

No formal manuscript files were changed in this phase.
'@
```

- [ ] **Step 4: Verify the result note has no angle-bracket placeholders**

Run:

```powershell
rg -n "copy phase65_status|one paragraph based|value from generated artifacts" paper\phase28_results\31_phase65_standardized_set_policy_bc_rerun.md
```

Expected: no output.

- [ ] **Step 5: Commit Task 6**

Run:

```powershell
git add paper\phase28_results\31_phase65_standardized_set_policy_bc_rerun.md
git commit -m "docs: record Phase 65 standardized rerun results"
```

Expected: commit succeeds. Generated `experiments/**/outputs/**` files should remain ignored unless repository policy says otherwise.

---

### Task 7: Regression Verification And Final Boundary Checks

**Files:**
- No new files unless a failing verification requires a targeted fix.

- [ ] **Step 1: Run Phase 65, Phase 64, and Phase 63 targeted tests**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase65_standardized_set_policy_bc_rerun.py tests\test_phase64_set_policy_error_diagnosis.py tests\test_phase63_set_policy_oracle_pretraining.py -q --basetemp=.pytest_tmp_phase65_final -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 2: Run smoke check**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe scripts\smoke_check.py
```

Expected: `Paper11 smoke check passed`.

- [ ] **Step 3: Check whitespace and conflict markers**

Run:

```powershell
git diff --check
```

Expected: no output.

- [ ] **Step 4: Confirm formal manuscript files are untouched**

Run:

```powershell
git diff --name-only HEAD -- paper\submission\final
```

Expected: no output.

- [ ] **Step 5: Confirm final git status**

Run:

```powershell
git status --short --branch
```

Expected: clean working tree on `main`. If local commits are ahead of `origin/main`, push after reviewing the commit list.

- [ ] **Step 6: Push completed Phase 65 work**

Run:

```powershell
git push
```

Expected: `main -> main` push succeeds.

