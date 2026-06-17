# Phase 25 Padded Held-Out Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first padded variable-size MaskablePPO held-out-tile B0/B1 learned-policy experiment for Paper11.

**Architecture:** Add a new Phase 25 module with a padded Gymnasium environment, held-out tile selection, MaskablePPO training/evaluation loops, deterministic baselines, and summary/trace/comparison artifacts. Keep Phase 25 restricted to B0/B1 under `base_planning_reward`; the padded contract removes the Phase 20/23 flat observation/action shape blocker without enabling suitability reward or B2/B3.

**Tech Stack:** Python standard library, NumPy, Gymnasium, Stable-Baselines3, sb3-contrib MaskablePPO, pytest, CSV/JSON artifact writing.

---

## File Structure

- Create: `tests/test_phase25_padded_heldout_policy.py`
  - Owns Phase 25 TDD coverage for padded env shape/masks, tile selection, tiny MaskablePPO run, writer, CLI, and comparison JSON.
- Create: `src/paper11_geofm/padded_heldout_policy.py`
  - Owns `Phase25PaddedTileEnv`, contract builder, runner, baseline evaluator, comparison aggregation, dependency metadata, and artifact writer.
- Create: `experiments/phase25_padded_heldout_policy/run_phase25_padded_heldout_policy.py`
  - Owns the reviewer-facing CLI for local smoke runs and Colab main runs.
- Modify: `README.md`
  - Add Phase 25 to the phase list, claim boundary, and reviewer command sequence.
- Modify: `reproducibility/REPRODUCTION_GUIDE.md`
  - Add Phase 25 Windows smoke and Colab Pro+ main-run commands.
- Modify: `reproducibility/FILE_MANIFEST.tsv`
  - Add the Phase 25 spec, plan, runtime module, runner, and tests.
- Modify: `paper/submission/01_ijaeog_submission_readiness.md`
  - Add guarded Phase 25 held-out learned-policy evidence status while keeping submission readiness `not_ready`.
- Modify: `paper/submission/02_draft_titles_highlights_declarations.md`
  - Add guarded manuscript language that separates Phase 25 held-out Bishan tile evidence from suitability-reward or cross-region claims.

Do not modify `src/paper11_geofm/drl_smoke_env.py` for Phase 25. Keep the padded contract isolated in `padded_heldout_policy.py` so Phase 4/14/20/23 behavior remains unchanged.

## Shared Constants and Data Contract

Use these constants in `src/paper11_geofm/padded_heldout_policy.py`:

```python
PHASE25_CLAIM_BOUNDARY = (
    "Phase 25 is a bounded padded variable-size held-out-tile B0/B1 "
    "MaskablePPO learned-policy pilot under the deterministic base planning "
    "reward; it tests distinct Bishan tiles, does not enable suitability "
    "reward, does not test B2/B3, and does not support cross-region transfer "
    "or submission-level planning-performance claims."
)

PHASE25_REMAINING_EVIDENCE_GAPS = [
    "longer_training_budget_and_hyperparameter_sensitivity",
    "suitability_reward_validation_before_B2_B3",
    "held_out_region_transfer_evaluation",
    "spatial_case_maps_and_uncertainty",
    "submission_level_ablation_and_robustness_package",
]

PHASE25_GLOBAL_FEATURE_COUNT = 5
```

Use this observation contract for `Phase25PaddedTileEnv`:

```text
observation_dim = max_blocks * n_features + max_blocks + max_blocks + 5
```

The flattened observation order is:

```text
padded_state_matrix.reshape(-1)
selected_mask
valid_block_mask
global_features
```

`global_features` order is:

```text
budget_remaining
step_fraction
valid_action_fraction
real_block_fraction
real_block_count_fraction
```

`real_block_fraction` and `real_block_count_fraction` both equal `n_real_blocks / max_blocks` for the current static tile. Keep both because the design spec names "real block fraction" and "real block count normalized by max_blocks" separately.

Use these summary CSV fields:

```python
SUMMARY_FIELDNAMES = [
    "row_type",
    "variant_id",
    "train_tile_id",
    "eval_tile_id",
    "eval_tile_rank",
    "seed",
    "phase25_seed_rank",
    "train_timesteps",
    "eval_max_steps",
    "max_blocks",
    "train_n_blocks",
    "eval_n_blocks",
    "n_features",
    "observation_shape",
    "action_space_n",
    "episode_steps",
    "terminated",
    "truncated",
    "all_actions_valid",
    "invalid_action_count",
    "total_contract_reward",
    "selected_block_ids",
    "claim_boundary",
]
```

---

### Task 1: Failing Padded Environment Tests

**Files:**
- Create: `tests/test_phase25_padded_heldout_policy.py`

- [ ] **Step 1: Write shared test helpers**

Copy the small Phase 22 fixture pattern and create helpers in `tests/test_phase25_padded_heldout_policy.py`:

```python
import csv
import faulthandler
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.filterwarnings("ignore:XPU device count is zero!:UserWarning")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id, slope_mean, farmland, suitability=0.75):
    row = {"block_id": block_id, "suitability_proxy": suitability}
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(dim) / 100.0
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = 0.0
    row.update(
        {
            "explicit_feature_00": 2.0,
            "explicit_feature_01": float(slope_mean),
            "explicit_feature_02": float(slope_mean) + 5.0,
            "explicit_feature_04": float(farmland),
            "explicit_feature_07": 0.0,
            "explicit_feature_09": 0.0,
            "explicit_feature_10": 0.0,
            "explicit_feature_13": 1.0 if slope_mean <= 15.0 else 0.0,
            "explicit_feature_16": float(farmland),
        }
    )
    return row


def _write_ready_phase2_outputs(output_dir: Path):
    from paper11_geofm.artifacts import write_phase2_artifacts

    return write_phase2_artifacts(
        [
            _complete_phase2_feature_row("b1", slope_mean=8.0, farmland=1.0),
            _complete_phase2_feature_row("b2", slope_mean=30.0, farmland=0.0),
            _complete_phase2_feature_row("b3", slope_mean=12.0, farmland=1.0),
            _complete_phase2_feature_row("b4", slope_mean=25.0, farmland=0.0),
            _complete_phase2_feature_row("b5", slope_mean=6.0, farmland=1.0),
            _complete_phase2_feature_row("b6", slope_mean=22.0, farmland=0.0),
        ],
        output_dir,
        {
            "metadata_source": "test",
            "base_year_requested": 2020,
            "base_year_used": 2020,
            "years": [2020],
            "grid_shape": [2, 3],
            "embedding_dim": 64,
            "mapping_mode": "test",
        },
    )


def _write_tile_index(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tile_id", "tile_row", "tile_col", "n_blocks", "block_ids"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "tile_id": "tile_r000_c000",
                "tile_row": 0,
                "tile_col": 0,
                "n_blocks": 1,
                "block_ids": "b6",
            }
        )
        writer.writerow(
            {
                "tile_id": "tile_r000_c001",
                "tile_row": 0,
                "tile_col": 1,
                "n_blocks": 3,
                "block_ids": "b1;b3;b5",
            }
        )
        writer.writerow(
            {
                "tile_id": "tile_r000_c002",
                "tile_row": 0,
                "tile_col": 2,
                "n_blocks": 2,
                "block_ids": "b2;b4",
            }
        )
    return path


@contextmanager
def _torch_windows_faulthandler_guard():
    was_enabled = faulthandler.is_enabled()
    if was_enabled:
        faulthandler.disable()
    try:
        yield
    finally:
        if was_enabled:
            faulthandler.enable(file=sys.__stderr__)


def _require_maskableppo_dependencies():
    with _torch_windows_faulthandler_guard():
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("sb3_contrib")
```

- [ ] **Step 2: Write padded reset and mask shape test**

Add a test that loads the two-block evaluation tile with `max_blocks=3` and asserts exact observation and mask behavior:

```python
def test_phase25_padded_env_uses_fixed_shape_and_masks_padded_rows(tmp_path):
    from paper11_geofm.padded_heldout_policy import Phase25PaddedTileEnv
    from paper11_geofm.tiled_inputs import load_tiled_variant_input

    _write_ready_phase2_outputs(tmp_path / "phase2")
    tile_index = _write_tile_index(tmp_path / "phase13_tile_index.csv")
    tiled = load_tiled_variant_input(
        tmp_path / "phase2",
        tile_index,
        "tile_r000_c002",
        variant_id="B0",
    )
    env = Phase25PaddedTileEnv(tiled, max_blocks=3, max_steps=2)

    obs, info = env.reset(seed=0)
    assert info["variant_id"] == "B0"
    assert info["tile_id"] == "tile_r000_c002"
    assert info["n_blocks"] == 2
    assert info["max_blocks"] == 3
    assert obs.shape == (3 * 17 + 3 + 3 + 5,)
    assert env.observation_space.shape == obs.shape
    assert env.action_space.n == 3
    assert env.action_masks().tolist() == [True, True, False]

    state_part = obs[: 3 * 17].reshape(3, 17)
    np.testing.assert_allclose(state_part[2], np.zeros(17, dtype=np.float32))
    selected_mask = obs[3 * 17 : 3 * 17 + 3]
    valid_block_mask = obs[3 * 17 + 3 : 3 * 17 + 6]
    np.testing.assert_allclose(selected_mask, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(valid_block_mask, [1.0, 1.0, 0.0])
```

- [ ] **Step 3: Write invalid action and selected-mask tests**

Assert that padded actions and repeated actions fail, and a valid step updates the selected mask:

```python
def test_phase25_padded_env_rejects_padded_and_repeated_actions(tmp_path):
    from paper11_geofm.padded_heldout_policy import Phase25PaddedTileEnv
    from paper11_geofm.tiled_inputs import load_tiled_variant_input

    _write_ready_phase2_outputs(tmp_path / "phase2")
    tile_index = _write_tile_index(tmp_path / "phase13_tile_index.csv")
    tiled = load_tiled_variant_input(
        tmp_path / "phase2",
        tile_index,
        "tile_r000_c002",
        variant_id="B0",
    )
    env = Phase25PaddedTileEnv(tiled, max_blocks=3, max_steps=2)
    env.reset(seed=0)

    with pytest.raises(ValueError, match="padded action"):
        env.step(2)

    next_obs, reward, terminated, truncated, step_info = env.step(0)
    assert isinstance(float(reward), float)
    assert terminated is False
    assert truncated is False
    assert step_info["selected_block_id"] == "b2"
    assert step_info["action_valid"] is True
    assert env.action_masks().tolist() == [False, True, False]

    selected_mask_start = 3 * 17
    selected_mask = next_obs[selected_mask_start : selected_mask_start + 3]
    np.testing.assert_allclose(selected_mask, [1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="already selected"):
        env.step(0)
```

- [ ] **Step 4: Verify RED**

Run:

```powershell
python -m pytest tests\test_phase25_padded_heldout_policy.py::test_phase25_padded_env_uses_fixed_shape_and_masks_padded_rows tests\test_phase25_padded_heldout_policy.py::test_phase25_padded_env_rejects_padded_and_repeated_actions -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'paper11_geofm.padded_heldout_policy'`.

- [ ] **Step 5: Commit RED tests**

Run:

```powershell
git add tests\test_phase25_padded_heldout_policy.py
git commit -m "test: add Phase 25 padded env contract tests"
```

### Task 2: Padded Environment Implementation

**Files:**
- Create: `src/paper11_geofm/padded_heldout_policy.py`
- Test: `tests/test_phase25_padded_heldout_policy.py`

- [ ] **Step 1: Implement module imports, constants, and environment constructor**

Create `src/paper11_geofm/padded_heldout_policy.py` with:

```python
from __future__ import annotations

import csv
import hashlib
import json
import warnings
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .planning_reward import compute_base_planning_reward_from_matrix_row
from .tiled_inputs import load_tiled_variant_input


PHASE25_CLAIM_BOUNDARY = (
    "Phase 25 is a bounded padded variable-size held-out-tile B0/B1 "
    "MaskablePPO learned-policy pilot under the deterministic base planning "
    "reward; it tests distinct Bishan tiles, does not enable suitability "
    "reward, does not test B2/B3, and does not support cross-region transfer "
    "or submission-level planning-performance claims."
)

PHASE25_REMAINING_EVIDENCE_GAPS = [
    "longer_training_budget_and_hyperparameter_sensitivity",
    "suitability_reward_validation_before_B2_B3",
    "held_out_region_transfer_evaluation",
    "spatial_case_maps_and_uncertainty",
    "submission_level_ablation_and_robustness_package",
]

PHASE25_GLOBAL_FEATURE_COUNT = 5
```

Implement the constructor:

```python
class Phase25PaddedTileEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, tiled_input, max_blocks: int, max_steps: int | None = None) -> None:
        super().__init__()
        if not tiled_input.block_ids:
            raise ValueError("Phase 25 padded env requires at least one block")
        if not tiled_input.feature_columns:
            raise ValueError("Phase 25 padded env requires at least one feature column")
        if str(tiled_input.reward_mode) != "base_planning_reward":
            raise ValueError("Phase 25 only supports base_planning_reward")

        self.tiled_input = tiled_input
        self.tile_id = str(tiled_input.tile_id)
        self.variant_id = str(tiled_input.variant_id)
        self.block_ids = tuple(tiled_input.block_ids)
        self.feature_columns = tuple(tiled_input.feature_columns)
        self.reward_mode = str(tiled_input.reward_mode)
        self.state_groups = tuple(tiled_input.state_groups)
        self.state_matrix = tiled_input.state_matrix.astype(np.float32, copy=True)
        self.n_blocks, self.n_features = self.state_matrix.shape
        self.max_blocks = int(max_blocks)
        if self.max_blocks < self.n_blocks:
            raise ValueError(
                f"max_blocks must cover tile block count: {self.max_blocks} < {self.n_blocks}"
            )
        self.max_steps = int(max_steps) if max_steps is not None else self.n_blocks
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

        obs_dim = self.max_blocks * self.n_features + self.max_blocks + self.max_blocks + PHASE25_GLOBAL_FEATURE_COUNT
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.max_blocks)
        self._selected = np.zeros(self.max_blocks, dtype=bool)
        self._valid_block_mask = np.zeros(self.max_blocks, dtype=bool)
        self._valid_block_mask[: self.n_blocks] = True
        self._step = 0
```

- [ ] **Step 2: Implement reset, step, action masks, and observation**

Add:

```python
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._selected = np.zeros(self.max_blocks, dtype=bool)
        self._step = 0
        return self._get_obs(), self._info()

    def step(self, action: int):
        action = int(action)
        if action < 0 or action >= self.max_blocks:
            raise ValueError(f"Action out of range: {action}")
        if not self._valid_block_mask[action]:
            raise ValueError(f"Action is a padded action: {action}")
        if self._selected[action]:
            raise ValueError(f"Action already selected: {action}")

        self._selected[action] = True
        self._step += 1
        reward = self._contract_reward(action)
        terminated = self._step >= self.max_steps or not self.action_masks().any()
        info = self._info()
        info.update(
            {
                "action": action,
                "action_valid": True,
                "selected_block_id": self.block_ids[action],
                "step": self._step,
                "valid_actions": int(self.action_masks().sum()),
                "terminated": bool(terminated),
            }
        )
        return self._get_obs(), reward, bool(terminated), False, info

    def action_masks(self) -> np.ndarray:
        return np.logical_and(self._valid_block_mask, ~self._selected).copy()

    def _get_obs(self) -> np.ndarray:
        padded = np.zeros((self.max_blocks, self.n_features), dtype=np.float32)
        padded[: self.n_blocks, :] = self.state_matrix
        return np.concatenate(
            [
                padded.reshape(-1),
                self._selected.astype(np.float32),
                self._valid_block_mask.astype(np.float32),
                self._global_features(),
            ]
        ).astype(np.float32)
```

- [ ] **Step 3: Implement global features, reward, and info**

Add:

```python
    def _global_features(self) -> np.ndarray:
        step_fraction = min(self._step / self.max_steps, 1.0)
        budget_remaining = max(1.0 - step_fraction, 0.0)
        valid_action_fraction = float(self.action_masks().sum() / self.max_blocks)
        real_block_fraction = float(self.n_blocks / self.max_blocks)
        real_block_count_fraction = float(self.n_blocks / self.max_blocks)
        return np.array(
            [
                budget_remaining,
                step_fraction,
                valid_action_fraction,
                real_block_fraction,
                real_block_count_fraction,
            ],
            dtype=np.float32,
        )

    def _contract_reward(self, action: int) -> float:
        return compute_base_planning_reward_from_matrix_row(
            self.feature_columns,
            self.state_matrix[action],
        )

    def _info(self) -> dict[str, object]:
        return {
            "phase": "phase25_padded_heldout_policy",
            "variant_id": self.variant_id,
            "tile_id": self.tile_id,
            "n_blocks": int(self.n_blocks),
            "n_features": int(self.n_features),
            "max_blocks": int(self.max_blocks),
            "reward_mode": self.reward_mode,
            "state_groups": self.state_groups,
            "claim_boundary": PHASE25_CLAIM_BOUNDARY,
        }
```

- [ ] **Step 4: Verify GREEN for padded env**

Run:

```powershell
python -m pytest tests\test_phase25_padded_heldout_policy.py::test_phase25_padded_env_uses_fixed_shape_and_masks_padded_rows tests\test_phase25_padded_heldout_policy.py::test_phase25_padded_env_rejects_padded_and_repeated_actions -q
```

Expected: PASS.

- [ ] **Step 5: Commit padded env**

Run:

```powershell
git add src\paper11_geofm\padded_heldout_policy.py tests\test_phase25_padded_heldout_policy.py
git commit -m "feat: add Phase 25 padded tile env"
```

### Task 3: Failing Contract, Runner, Writer, and CLI Tests

**Files:**
- Modify: `tests/test_phase25_padded_heldout_policy.py`

- [ ] **Step 1: Write contract and guardrail tests**

Add:

```python
def test_phase25_contract_selects_largest_train_and_distinct_eval_tiles(tmp_path):
    from paper11_geofm.padded_heldout_policy import (
        PHASE25_CLAIM_BOUNDARY,
        build_phase25_padded_heldout_policy_contract,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    contract = build_phase25_padded_heldout_policy_contract(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variants=("B0", "B1"),
        total_timesteps=8,
        eval_max_steps=2,
        seeds="0,1",
        max_eval_tiles=2,
    )

    assert contract["phase"] == "phase25_padded_heldout_policy"
    assert contract["variants"] == ["B0", "B1"]
    assert contract["train_tile_id"] == "tile_r000_c001"
    assert contract["eval_tile_ids"] == ["tile_r000_c002", "tile_r000_c000"]
    assert contract["eval_tile_ranks"] == {
        "tile_r000_c002": 1,
        "tile_r000_c000": 2,
    }
    assert contract["train_tile_selection"] == "largest"
    assert contract["eval_tile_selection"] == "largest_distinct"
    assert contract["padded_policy_status"] == "enabled_distinct_heldout_tiles"
    assert contract["max_blocks"] == 3
    assert contract["total_timesteps"] == 8
    assert contract["eval_max_steps"] == 2
    assert contract["seeds"] == [0, 1]
    assert contract["claim_boundary"] == PHASE25_CLAIM_BOUNDARY
```

Add guardrail coverage:

```python
def test_phase25_contract_rejects_suitability_variants_and_train_eval_overlap(tmp_path):
    from paper11_geofm.padded_heldout_policy import (
        build_phase25_padded_heldout_policy_contract,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    tile_index = _write_tile_index(tmp_path / "phase13_tile_index.csv")

    with pytest.raises(ValueError, match="B0/B1"):
        build_phase25_padded_heldout_policy_contract(
            tmp_path / "phase2",
            tile_index,
            variants=("B3",),
        )

    with pytest.raises(ValueError, match="must be distinct"):
        build_phase25_padded_heldout_policy_contract(
            tmp_path / "phase2",
            tile_index,
            train_tile_id="tile_r000_c001",
            eval_tile_ids="tile_r000_c001",
        )
```

- [ ] **Step 2: Write tiny run and comparison test**

Add a MaskablePPO-backed test:

```python
def test_phase25_runs_padded_heldout_policy_training_and_comparison(tmp_path):
    _require_maskableppo_dependencies()
    from paper11_geofm.padded_heldout_policy import (
        PHASE25_CLAIM_BOUNDARY,
        run_phase25_padded_heldout_policy,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with _torch_windows_faulthandler_guard():
        protocol = run_phase25_padded_heldout_policy(
            tmp_path / "phase2",
            _write_tile_index(tmp_path / "phase13_tile_index.csv"),
            variants=("B0", "B1"),
            total_timesteps=8,
            eval_max_steps=2,
            seeds=(0, 1),
            max_eval_tiles=2,
        )

    assert protocol["phase"] == "phase25_padded_heldout_policy"
    assert protocol["train_tile_id"] == "tile_r000_c001"
    assert protocol["eval_tile_ids"] == ["tile_r000_c002", "tile_r000_c000"]
    assert protocol["training_completed"] is True
    assert protocol["all_evaluations_completed"] is True
    assert protocol["summary_count"] == 24
    assert len(protocol["summaries"]) == 24
    assert all(row["max_blocks"] == 3 for row in protocol["summaries"])
    assert all(row["action_space_n"] == 3 for row in protocol["summaries"])
    assert all(row["all_actions_valid"] is True for row in protocol["summaries"])
    assert all(row["invalid_action_count"] == 0 for row in protocol["summaries"])
    assert all(row["claim_boundary"] == PHASE25_CLAIM_BOUNDARY for row in protocol["summaries"])
    assert protocol["comparison"]["learned_policy"]["B1_minus_B0_mean_reward"] is not None
    assert protocol["comparison"]["learned_policy"]["heldout_tile_B1_minus_B0_mean_reward"]
    assert protocol["comparison"]["pilot_result_status"] in {
        "B1_improves_B0",
        "B1_matches_B0",
        "B1_underperforms_B0",
    }
    assert protocol["traces"]["trained_policy"]["B0"]["tile_r000_c002"]["0"]
    assert protocol["traces"]["seeded_random"]["B1"]["tile_r000_c000"]["1"]
```

- [ ] **Step 3: Write writer test**

Add:

```python
def test_phase25_writer_outputs_summary_trace_and_comparison(tmp_path):
    from paper11_geofm.padded_heldout_policy import (
        PHASE25_CLAIM_BOUNDARY,
        write_phase25_padded_heldout_policy_artifacts,
    )

    protocol = {
        "phase": "phase25_padded_heldout_policy",
        "summaries": [
            {
                "row_type": "trained_policy",
                "variant_id": "B0",
                "train_tile_id": "tile_r000_c001",
                "eval_tile_id": "tile_r000_c002",
                "eval_tile_rank": 1,
                "seed": 0,
                "phase25_seed_rank": 1,
                "train_timesteps": 8,
                "eval_max_steps": 2,
                "max_blocks": 3,
                "train_n_blocks": 3,
                "eval_n_blocks": 2,
                "n_features": 17,
                "observation_shape": 62,
                "action_space_n": 3,
                "episode_steps": 2,
                "terminated": True,
                "truncated": False,
                "all_actions_valid": True,
                "invalid_action_count": 0,
                "total_contract_reward": 1.2,
                "selected_block_ids": ["b2", "b4"],
                "claim_boundary": PHASE25_CLAIM_BOUNDARY,
            }
        ],
        "traces": {"trained_policy": {"B0": {"tile_r000_c002": {"0": []}}}},
        "comparison": {
            "phase": "phase25_padded_heldout_policy_comparison",
            "pilot_result_status": "B1_matches_B0",
            "claim_boundary": PHASE25_CLAIM_BOUNDARY,
        },
        "claim_boundary": PHASE25_CLAIM_BOUNDARY,
    }

    paths = write_phase25_padded_heldout_policy_artifacts(
        protocol,
        tmp_path / "outputs",
    )

    assert paths["summary_csv"].name == "phase25_padded_heldout_policy_summary.csv"
    assert paths["traces_json"].name == "phase25_padded_heldout_policy_traces.json"
    assert paths["comparison_json"].name == "phase25_padded_heldout_policy_comparison.json"
    rows = list(csv.DictReader(paths["summary_csv"].open("r", encoding="utf-8")))
    assert rows[0]["selected_block_ids"] == "b2;b4"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["pilot_result_status"] == "B1_matches_B0"
```

- [ ] **Step 4: Write CLI test**

Add:

```python
def test_phase25_cli_writes_outputs_and_prints_summary(tmp_path, capsys):
    _require_maskableppo_dependencies()
    runner_path = (
        ROOT
        / "experiments"
        / "phase25_padded_heldout_policy"
        / "run_phase25_padded_heldout_policy.py"
    )
    spec = importlib.util.spec_from_file_location("phase25_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with _torch_windows_faulthandler_guard():
        exit_code = module.main(
            [
                "--phase2-output-dir",
                str(tmp_path / "phase2"),
                "--tile-index-csv",
                str(_write_tile_index(tmp_path / "phase13_tile_index.csv")),
                "--variants",
                "B0,B1",
                "--total-timesteps",
                "8",
                "--eval-max-steps",
                "2",
                "--seeds",
                "0,1",
                "--max-eval-tiles",
                "2",
                "--output-dir",
                str(tmp_path / "outputs"),
            ]
        )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Train tile: tile_r000_c001" in stdout
    assert "Held-out evaluation tiles: tile_r000_c002, tile_r000_c000" in stdout
    assert "Padded max blocks: 3" in stdout
    assert "Seeds: 0, 1" in stdout
    assert "Variants: B0, B1" in stdout
    assert "Summary rows: 24" in stdout
    assert "B1-B0 held-out learned-policy mean reward delta:" in stdout
    assert "phase25_padded_heldout_policy_comparison.json" in stdout
    assert "Claim boundary: Phase 25 is a bounded padded variable-size held-out-tile" in stdout
```

- [ ] **Step 5: Verify RED for non-env behavior**

Run:

```powershell
python -m pytest tests\test_phase25_padded_heldout_policy.py -q
```

Expected: padded env tests PASS; contract, run, writer, and CLI tests FAIL because the new functions and runner are not implemented.

- [ ] **Step 6: Commit expanded RED tests**

Run:

```powershell
git add tests\test_phase25_padded_heldout_policy.py
git commit -m "test: add Phase 25 held-out policy workflow tests"
```

### Task 4: Phase 25 Contract, Evaluation, Comparison, and Writer

**Files:**
- Modify: `src/paper11_geofm/padded_heldout_policy.py`
- Test: `tests/test_phase25_padded_heldout_policy.py`

- [ ] **Step 1: Implement normalizers and tile selection**

Add functions:

```python
def build_phase25_padded_heldout_policy_contract(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] = ("B0", "B1"),
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

    normalized_variants = _normalize_variants(variants)
    normalized_seeds = _normalize_seeds(seeds)
    selected = _select_train_eval_tiles(
        Path(tile_index_csv),
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
    )
    max_blocks = max(int(count) for count in selected["selected_tile_block_counts"].values())
    eval_tile_ids_out = list(selected["eval_tile_ids"])
    eval_tile_ranks = {
        str(tile_id): rank + 1 for rank, tile_id in enumerate(eval_tile_ids_out)
    }
    seed_ranks = {str(seed): rank + 1 for rank, seed in enumerate(normalized_seeds)}
    return {
        "phase": "phase25_padded_heldout_policy",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "tile_index_csv": str(Path(tile_index_csv)),
        "variants": normalized_variants,
        "train_tile_id": selected["train_tile_id"],
        "train_tile_ids": [selected["train_tile_id"]],
        "eval_tile_ids": eval_tile_ids_out,
        "eval_tile_count": len(eval_tile_ids_out),
        "eval_tile_ranks": eval_tile_ranks,
        "selected_tile_block_counts": dict(selected["selected_tile_block_counts"]),
        "train_tile_selection": selected["train_tile_selection"],
        "eval_tile_selection": selected["eval_tile_selection"],
        "padded_policy_status": "enabled_distinct_heldout_tiles",
        "learned_policy_evaluation_scope": "padded_variable_size_heldout_tile_b0_b1_training_pilot",
        "max_blocks": int(max_blocks),
        "total_timesteps": int(total_timesteps),
        "eval_max_steps": int(eval_max_steps),
        "seeds": normalized_seeds,
        "seed_count": len(normalized_seeds),
        "seed_ranks": seed_ranks,
        "claim_boundary": PHASE25_CLAIM_BOUNDARY,
        "remaining_evidence_gaps": list(PHASE25_REMAINING_EVIDENCE_GAPS),
    }
```

Implement `_normalize_variants`, `_normalize_seeds`, `_normalize_eval_tile_ids`, `_read_tile_rows`, and `_select_train_eval_tiles` with Phase 22 behavior:

```python
def _normalize_variants(variants: Sequence[str]) -> list[str]:
    normalized = [str(item).strip().upper() for item in variants]
    normalized = [item for item in normalized if item]
    if not normalized:
        raise ValueError("At least one Phase 25 variant must be requested")
    unsupported = [variant for variant in normalized if variant not in {"B0", "B1"}]
    if unsupported:
        raise ValueError(
            "Phase 25 is restricted to B0/B1 base-reward variants; "
            f"unsupported variant: {unsupported[0]}"
        )
    return normalized
```

`_select_train_eval_tiles` must sort tile rows by descending `n_blocks`, use the largest train tile by default, use the largest distinct evaluation tiles by default, reject unknown IDs, reject duplicate evaluation IDs, reject the train tile in evaluation IDs, and return `selected_tile_block_counts` for train plus evaluation tiles.

- [ ] **Step 2: Implement training and evaluation runner**

Add:

```python
def run_phase25_padded_heldout_policy(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] = ("B0", "B1"),
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 3,
    total_timesteps: int = 32,
    eval_max_steps: int = 4,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
) -> dict[str, object]:
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import is_masking_supported
    except ImportError as exc:
        raise RuntimeError(
            "Phase 25 padded held-out policy requires stable-baselines3 and sb3-contrib"
        ) from exc

    contract = build_phase25_padded_heldout_policy_contract(
        phase2_output_dir,
        tile_index_csv,
        variants=variants,
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
        total_timesteps=total_timesteps,
        eval_max_steps=eval_max_steps,
        seeds=seeds,
    )
```

Within the runner:

- Loop variants first, then seeds.
- Load the train tile once per variant using `load_tiled_variant_input`.
- Create `Phase25PaddedTileEnv(train_tiled, max_blocks=contract["max_blocks"], max_steps=contract["total_timesteps"])`.
- Call `train_env.reset(seed=seed)`.
- Use `is_masking_supported(train_env)` and raise `ValueError("Phase 25 train env does not expose action_masks")` if false.
- Train `MaskablePPO("MlpPolicy", train_env, seed=seed, device="cpu", verbose=0, n_steps=4, batch_size=4, n_epochs=1, gamma=0.99)`.
- Suppress the Windows XPU warning with the same `warnings.catch_warnings()` pattern used in Phase 20.
- Evaluate the trained model and both baselines on each held-out tile and seed.
- Store traces as `traces[row_type][variant_id][eval_tile_id][str(seed)]`.
- Return `comparison = _build_comparison(summaries, contract)`.

- [ ] **Step 3: Implement trained-policy and baseline evaluation helpers**

Implement:

```python
def _evaluate_trained_policy(
    model: Any,
    tiled,
    train_tile_id: str,
    train_n_blocks: int,
    max_blocks: int,
    eval_tile_rank: int,
    eval_max_steps: int,
    train_timesteps: int,
    seed: int,
    seed_rank: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    env = Phase25PaddedTileEnv(tiled, max_blocks=max_blocks, max_steps=eval_max_steps)
    obs, info = env.reset(seed=seed)
    steps: list[dict[str, object]] = []
    selected_block_ids: list[str] = []
    total_reward = 0.0
    terminated = False
    truncated = False
    invalid_action_count = 0

    while True:
        masks = env.action_masks()
        valid_actions = _valid_actions(masks)
        if not valid_actions:
            break
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        action_index = int(action)
        if action_index not in valid_actions:
            invalid_action_count += 1
            raise ValueError("Phase 25 trained policy selected an invalid action")
        obs, reward, terminated, truncated, step_info = env.step(action_index)
        reward_value = float(reward)
        total_reward += reward_value
        selected_block_id = str(step_info["selected_block_id"])
        selected_block_ids.append(selected_block_id)
        steps.append(_step_record(step_info, action_index, reward_value, env))
        if terminated or truncated:
            break
```

Return `_summary_row(...)` and steps. Implement `_evaluate_baseline_policy` with the same signature, except it selects actions through `_select_baseline_action(policy_id, valid_actions, rng)` and records `row_type=policy_id`.

Implement `_summary_row`, `_step_record`, `_valid_actions`, `_select_baseline_action`, `_rng_for`, `_round_float`, `_dependency_metadata`, and `_package_metadata`. Reuse the Phase 20/21 deterministic SHA-256 seeded random pattern.

- [ ] **Step 4: Implement comparison aggregation**

Create `_build_comparison(summaries, contract)` returning:

```python
{
    "phase": "phase25_padded_heldout_policy_comparison",
    "train_tile_id": contract["train_tile_id"],
    "train_tile_ids": list(contract["train_tile_ids"]),
    "eval_tile_ids": list(contract["eval_tile_ids"]),
    "variants": list(contract["variants"]),
    "seeds": list(contract["seeds"]),
    "seed_count": int(contract["seed_count"]),
    "policies": ["trained_policy", "first_valid", "seeded_random"],
    "total_timesteps": int(contract["total_timesteps"]),
    "eval_max_steps": int(contract["eval_max_steps"]),
    "max_blocks": int(contract["max_blocks"]),
    "summary_count": len(summaries),
    "mean_reward_by_row_type_variant_eval_tile": nested_means,
    "learned_policy": {
        "mean_reward_by_variant": learned_means,
        "B1_minus_B0_mean_reward": learned_delta,
        "heldout_tile_B1_minus_B0_mean_reward": learned_tile_deltas,
    },
    "baselines": baseline_summary,
    "pilot_result_status": pilot_result_status,
    "claim_boundary": PHASE25_CLAIM_BOUNDARY,
    "remaining_evidence_gaps": list(PHASE25_REMAINING_EVIDENCE_GAPS),
}
```

Set `pilot_result_status` by the learned-policy aggregate delta:

```python
def _pilot_result_status(delta: float | None) -> str:
    if delta is None:
        return "insufficient_B0_B1_learned_policy_rows"
    if delta > 1e-9:
        return "B1_improves_B0"
    if delta < -1e-9:
        return "B1_underperforms_B0"
    return "B1_matches_B0"
```

For each baseline in `("first_valid", "seeded_random")`, store:

```python
{
    "mean_reward_by_variant": {...},
    "B1_minus_B0_mean_reward": value_or_none,
    "heldout_tile_B1_minus_B0_mean_reward": {...},
}
```

- [ ] **Step 5: Implement artifact writer**

Implement `write_phase25_padded_heldout_policy_artifacts(protocol, output_dir)`:

- create output directory;
- write `phase25_padded_heldout_policy_summary.csv` using `SUMMARY_FIELDNAMES`;
- serialize list-valued `selected_block_ids` as semicolon-separated text;
- write full protocol to `phase25_padded_heldout_policy_traces.json`;
- write `protocol["comparison"]` to `phase25_padded_heldout_policy_comparison.json`;
- validate that `protocol["summaries"]` is a list and `protocol["comparison"]` is a mapping.

- [ ] **Step 6: Verify module GREEN**

Run:

```powershell
python -m pytest tests\test_phase25_padded_heldout_policy.py::test_phase25_contract_selects_largest_train_and_distinct_eval_tiles tests\test_phase25_padded_heldout_policy.py::test_phase25_contract_rejects_suitability_variants_and_train_eval_overlap tests\test_phase25_padded_heldout_policy.py::test_phase25_runs_padded_heldout_policy_training_and_comparison tests\test_phase25_padded_heldout_policy.py::test_phase25_writer_outputs_summary_trace_and_comparison -q
```

Expected: PASS.

- [ ] **Step 7: Commit Phase 25 module**

Run:

```powershell
git add src\paper11_geofm\padded_heldout_policy.py tests\test_phase25_padded_heldout_policy.py
git commit -m "feat: add Phase 25 padded held-out policy module"
```

### Task 5: Phase 25 CLI

**Files:**
- Create: `experiments/phase25_padded_heldout_policy/run_phase25_padded_heldout_policy.py`
- Test: `tests/test_phase25_padded_heldout_policy.py`

- [ ] **Step 1: Implement CLI runner**

Create:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.padded_heldout_policy import (
    PHASE25_CLAIM_BOUNDARY,
    run_phase25_padded_heldout_policy,
    write_phase25_padded_heldout_policy_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded padded variable-size held-out-tile B0/B1 "
            "MaskablePPO learned-policy pilot under the deterministic base "
            "planning reward."
        )
    )
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--tile-index-csv", type=Path, required=True)
    parser.add_argument("--variants", default="B0,B1")
    parser.add_argument("--train-tile-id", default=None)
    parser.add_argument("--eval-tile-ids", default=None)
    parser.add_argument("--max-eval-tiles", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=32)
    parser.add_argument("--eval-max-steps", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        protocol = run_phase25_padded_heldout_policy(
            args.phase2_output_dir,
            args.tile_index_csv,
            variants=tuple(
                part.strip() for part in args.variants.split(",") if part.strip()
            ),
            train_tile_id=args.train_tile_id,
            eval_tile_ids=args.eval_tile_ids,
            max_eval_tiles=args.max_eval_tiles,
            total_timesteps=args.total_timesteps,
            eval_max_steps=args.eval_max_steps,
            seeds=args.seeds,
        )
        paths = write_phase25_padded_heldout_policy_artifacts(
            protocol,
            args.output_dir,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    delta = protocol["comparison"]["learned_policy"]["B1_minus_B0_mean_reward"]
    print(f"Train tile: {protocol['train_tile_id']}")
    print(f"Held-out evaluation tiles: {', '.join(protocol['eval_tile_ids'])}")
    print(f"Padded max blocks: {protocol['max_blocks']}")
    print(f"Seeds: {', '.join(str(seed) for seed in protocol['seeds'])}")
    print(f"Variants: {', '.join(protocol['variants'])}")
    print(f"Total timesteps: {protocol['total_timesteps']}")
    print(f"Evaluation max steps: {protocol['eval_max_steps']}")
    print(f"Padded held-out policy status: {protocol['padded_policy_status']}")
    print(f"Summary rows: {protocol['summary_count']}")
    print(f"All evaluations completed: {protocol['all_evaluations_completed']}")
    print(f"B1-B0 held-out learned-policy mean reward delta: {delta}")
    print(f"Pilot result status: {protocol['comparison']['pilot_result_status']}")
    print(f"Summary CSV: {paths['summary_csv']}")
    print(f"Trace JSON: {paths['traces_json']}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Claim boundary: {PHASE25_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify CLI GREEN**

Run:

```powershell
python -m pytest tests\test_phase25_padded_heldout_policy.py::test_phase25_cli_writes_outputs_and_prints_summary -q
```

Expected: PASS.

- [ ] **Step 3: Commit CLI**

Run:

```powershell
git add experiments\phase25_padded_heldout_policy\run_phase25_padded_heldout_policy.py tests\test_phase25_padded_heldout_policy.py
git commit -m "feat: add Phase 25 padded held-out policy CLI"
```

### Task 6: Windows Real Bishan Smoke Run and Colab Command Recipe

**Files:**
- Generated local artifacts under ignored path: `experiments/phase25_padded_heldout_policy/outputs/real_bishan_smoke/`

- [ ] **Step 1: Run short Windows smoke**

Run:

```powershell
python experiments\phase25_padded_heldout_policy\run_phase25_padded_heldout_policy.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --total-timesteps 32 --eval-max-steps 4 --seeds 0 --max-eval-tiles 1 --output-dir experiments\phase25_padded_heldout_policy\outputs\real_bishan_smoke
```

Expected:

- train tile is `tile_r003_c003`;
- held-out evaluation tile is `tile_r002_c003`;
- variants are `B0` and `B1`;
- seed is `0`;
- summary rows are `6`;
- output files are `phase25_padded_heldout_policy_summary.csv`, `phase25_padded_heldout_policy_traces.json`, and `phase25_padded_heldout_policy_comparison.json`;
- claim boundary says Phase 25 does not enable suitability reward, B2/B3, cross-region transfer, or submission-level planning claims.

- [ ] **Step 2: Inspect smoke comparison JSON**

Run:

```powershell
Get-Content -Raw experiments\phase25_padded_heldout_policy\outputs\real_bishan_smoke\phase25_padded_heldout_policy_comparison.json
```

Expected:

- `phase` is `phase25_padded_heldout_policy_comparison`;
- `learned_policy.B1_minus_B0_mean_reward` is present;
- `learned_policy.heldout_tile_B1_minus_B0_mean_reward.tile_r002_c003` is present;
- `pilot_result_status` is one of `B1_improves_B0`, `B1_matches_B0`, or `B1_underperforms_B0`;
- `remaining_evidence_gaps` includes `suitability_reward_validation_before_B2_B3`.

- [ ] **Step 3: Record Colab Pro+ command recipe in docs**

Use this main-run command in `reproducibility/REPRODUCTION_GUIDE.md`:

```bash
python experiments/phase25_padded_heldout_policy/run_phase25_padded_heldout_policy.py --phase2-output-dir experiments/phase11_bishan_dltb_real/outputs/phase2_real --tile-index-csv experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv --variants B0,B1 --total-timesteps 1024 --eval-max-steps 8 --seeds 0,1,2 --max-eval-tiles 3 --output-dir experiments/phase25_padded_heldout_policy/outputs/real_bishan_colab_main
```

State in the guide:

- Windows is for implementation, tests, schema checks, and short smoke runs;
- Colab Pro+ is the main training platform for multi-seed, multi-held-out-tile budgets;
- if the timing probe is stable, increase `--total-timesteps` within `512` to `4096`;
- do not enable B2/B3 or suitability reward in Phase 25.

### Task 7: Documentation and IJAEOG Evidence Notes

**Files:**
- Modify: `README.md`
- Modify: `reproducibility/REPRODUCTION_GUIDE.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `paper/submission/01_ijaeog_submission_readiness.md`
- Modify: `paper/submission/02_draft_titles_highlights_declarations.md`

- [ ] **Step 1: Update README**

Add Phase 25 to the phase summary near Phase 24:

```text
Phase 25 adds a padded variable-size held-out-tile MaskablePPO pilot for B0/B1 under the deterministic base planning reward. It removes the Phase 20/23 flat observation/action shape blocker for held-out Bishan tile evaluation, but it remains a bounded pilot and does not enable suitability reward, B2/B3, cross-region transfer, or submission-level claims.
```

Add the Windows smoke command from Task 6 and list the three expected output files.

- [ ] **Step 2: Update reproduction guide**

Insert a new Phase 25 section after Phase 24. Include:

- Windows smoke command from Task 6;
- expected smoke artifacts;
- Colab Pro+ main-run command from Task 6;
- explicit warning that Phase 25 uses `base_planning_reward` only;
- explicit warning that B2/B3 remain blocked by Phase 10 suitability reward readiness.

- [ ] **Step 3: Update file manifest**

Add rows:

```text
docs/superpowers/specs/2026-06-12-phase25-padded-heldout-policy-design.md	design	Phase 25 design for padded variable-size held-out-tile B0/B1 MaskablePPO learned-policy evidence.
docs/superpowers/plans/2026-06-12-phase25-padded-heldout-policy.md	plan	Implementation plan for the Phase 25 padded held-out-tile B0/B1 learned-policy pilot.
src/paper11_geofm/padded_heldout_policy.py	runtime	Phase 25 padded variable-size tile environment, MaskablePPO held-out evaluation runner, baselines, comparison aggregation, and artifact writer.
experiments/phase25_padded_heldout_policy/run_phase25_padded_heldout_policy.py	experiment	Executable Phase 25 padded held-out-tile B0/B1 learned-policy runner for real Bishan tiled artifacts.
tests/test_phase25_padded_heldout_policy.py	verification	Pytest checks for Phase 25 padded environment masks, held-out tile selection, tiny MaskablePPO run, artifact writing, and CLI output.
```

- [ ] **Step 4: Update submission readiness**

In `paper/submission/01_ijaeog_submission_readiness.md`, add a Phase 25 row or paragraph:

```text
Phase 25 introduces a padded variable-size learned-policy held-out Bishan tile pilot. If the smoke or main run completes, it can support the limited statement that a B0/B1 learned policy can be trained on one Bishan tile and evaluated on distinct held-out Bishan tiles under the deterministic base planning reward. It does not resolve suitability-reward readiness, B2/B3 claims, cross-region transfer, long-budget robustness, or final submission readiness.
```

Keep `submission_ready` as `not_ready`.

- [ ] **Step 5: Update draft language**

In `paper/submission/02_draft_titles_highlights_declarations.md`, add one guarded highlight option:

```text
- A padded variable-size policy contract enables a bounded held-out Bishan tile learned-policy pilot for B0/B1 under a deterministic planning reward.
```

Keep declarations and abstract scaffold clear that suitability reward and cross-region transfer are not claimed.

- [ ] **Step 6: Commit docs and smoke notes**

Run:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv paper\submission\01_ijaeog_submission_readiness.md paper\submission\02_draft_titles_highlights_declarations.md
git commit -m "docs: add Phase 25 held-out policy reproduction notes"
```

### Task 8: Final Verification, Commit, and Push

**Files:**
- Stage only Phase 25 source, runner, tests, docs, spec, and plan files.

- [ ] **Step 1: Run targeted verification**

Run:

```powershell
python -m pytest tests\test_phase25_padded_heldout_policy.py -q
```

Expected: PASS.

- [ ] **Step 2: Run repository verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests -q
git diff --check
git status --short --ignored=matching experiments\phase25_padded_heldout_policy
```

Expected:

- smoke check passes;
- full pytest passes;
- `git diff --check` prints no whitespace errors;
- generated output files under `experiments\phase25_padded_heldout_policy\outputs\` are ignored or intentionally left untracked.

- [ ] **Step 3: Final commit**

If previous task commits already captured all implementation chunks, skip this commit. If there are remaining tracked changes, run:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv paper\submission\01_ijaeog_submission_readiness.md paper\submission\02_draft_titles_highlights_declarations.md docs\superpowers\specs\2026-06-12-phase25-padded-heldout-policy-design.md docs\superpowers\plans\2026-06-12-phase25-padded-heldout-policy.md src\paper11_geofm\padded_heldout_policy.py experiments\phase25_padded_heldout_policy\run_phase25_padded_heldout_policy.py tests\test_phase25_padded_heldout_policy.py
git commit -m "Add Phase 25 padded held-out policy pilot"
```

- [ ] **Step 4: Push**

Run:

```powershell
git push
```

Expected: `main -> main`.

## Plan Self-Review

- Spec coverage: The plan covers the padded environment, MaskablePPO training, distinct held-out tile evaluation, B0/B1 restriction, deterministic base reward, baselines, summary/traces/comparison artifacts, Windows smoke run, Colab Pro+ command recipe, and claim boundaries.
- Placeholder scan: No unresolved placeholders remain in the task steps.
- Type consistency: Function names, output filenames, summary field names, and CLI flags are consistent across tests, implementation tasks, and documentation tasks.
- Scope check: The plan is limited to Phase 25 B0/B1 padded held-out Bishan tile learned-policy evidence. It does not add suitability reward, B2/B3 experiments, cross-region transfer, or final IJAEOG submission claims.
