# Phase 66 Reward-Label Representation Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only Phase 66 audit that explains whether the current base-reward label is redundant with explicit planning features or carries usable GeoFM-derived ranking signal.

**Architecture:** Add one focused analysis module that loads Phase 63/64/65 artifacts, reloads raw tiled matrices through the Phase 63 contract, decomposes deterministic base reward, compares oracle/Phase 63/Phase 65 selected blocks, and computes representation-rank alignment without training any policy. A thin CLI runner writes JSON, CSV, and Markdown artifacts under ignored experiment outputs; a tracked result note is created only after the real audit has run.

**Tech Stack:** Python 3, NumPy, existing `paper11_geofm` tiled input loaders, existing `planning_reward` and Phase 63/64/65 helper patterns, pytest, PowerShell commands using `D:\adk\.venv\Scripts\python.exe`.

---

## File Structure

- Create `src/paper11_geofm/phase66_reward_label_representation_audit.py`
  - Owns Phase 66 claim boundary, reward-component decomposition, selected-block atlas, representation-rank alignment, failure-mode classification, diagnostic gate, artifact writing, and full read-only run orchestration.
- Create `experiments/phase66_reward_label_representation_audit/run_phase66_reward_label_representation_audit.py`
  - Thin CLI wrapper. It accepts Phase 63 comparison/rollout/oracle/history paths, Phase 64 diagnostic paths, Phase 65 comparison/rollout/pairwise/stat paths, and an output directory.
- Create `tests/test_phase66_reward_label_representation_audit.py`
  - Covers deterministic reward decomposition, selected/missed/extra attribution, rank metrics with ties/constant columns, representation grouping, status gates, writers, and CLI parsing.
- Create `paper/phase28_results/32_phase66_reward_label_representation_audit.md`
  - Filled after the real Phase 66 run from generated Markdown plus reproduction command and boundary statement.
- Do not modify `paper/submission/final/*`.

---

### Task 1: Reward Component Decomposition

**Files:**
- Create: `src/paper11_geofm/phase66_reward_label_representation_audit.py`
- Create: `tests/test_phase66_reward_label_representation_audit.py`

- [ ] **Step 1: Write failing tests for deterministic component decomposition**

Create `tests/test_phase66_reward_label_representation_audit.py` with these tests and helpers:

```python
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _reward_columns() -> tuple[str, ...]:
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


def _tiled_input(
    block_ids=("b1", "b2", "b3", "b4"),
    matrix=None,
    columns=None,
    variant_id="D4P8",
    tile_id="tile_eval",
):
    from paper11_geofm.tiled_inputs import TiledVariantInput

    feature_columns = tuple(columns or _reward_columns())
    values = np.asarray(
        matrix
        if matrix is not None
        else [
            [5.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.8, 0.9],
            [4.0, 5.0, 7.0, 0.2, 0.6, 0.0, 0.0, 0.7, 0.8],
            [3.0, 15.0, 20.0, 0.0, 0.1, 0.2, 0.0, 0.2, 0.3],
            [1.0, 25.0, 35.0, 0.0, 0.0, 0.5, 0.4, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    return TiledVariantInput(
        tile_id=tile_id,
        variant_id=variant_id,
        block_ids=tuple(block_ids),
        feature_columns=feature_columns,
        state_matrix=values,
        reward_mode="base_planning_reward",
        state_groups=("synthetic",),
        source_table=Path(f"variant_{variant_id}_features.csv"),
        tile_index_csv=Path("tiles.csv"),
    )


def test_phase66_reward_components_sum_to_base_reward():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        decompose_phase66_base_reward_components,
    )
    from paper11_geofm.planning_reward import (
        compute_base_planning_reward_from_matrix_row,
    )

    tiled = _tiled_input()
    row = decompose_phase66_base_reward_components(
        tiled.feature_columns,
        tiled.state_matrix[0],
    )
    expected = compute_base_planning_reward_from_matrix_row(
        tiled.feature_columns,
        tiled.state_matrix[0],
    )

    assert row["low_slope_farmland_or_orchard_component"] == 0.315
    assert row["current_farmland_or_orchard_component"] == 0.08
    assert row["low_slope_component"] == 0.08
    assert row["area_component"] == 0.1
    assert row["mean_slope_penalty_component"] == -0.0
    assert row["max_slope_penalty_component"] == -0.0
    assert row["built_up_penalty_component"] == -0.0
    assert row["water_penalty_component"] == -0.0
    assert row["total_reward"] == expected


def test_phase66_reward_components_reject_missing_required_columns():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        decompose_phase66_base_reward_components,
    )

    try:
        decompose_phase66_base_reward_components(
            ("explicit_feature_00", "explicit_feature_16"),
            [1.0, 0.9],
        )
    except ValueError as exc:
        assert "explicit feature columns" in str(exc)
        assert "explicit_feature_01" in str(exc)
    else:
        raise AssertionError("Expected missing base-reward columns to fail")


def test_phase66_block_reward_table_ranks_blocks_by_reward_descending():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        build_phase66_block_reward_table,
    )

    table = build_phase66_block_reward_table(_tiled_input())

    assert [row["block_id"] for row in table] == ["b1", "b2", "b3", "b4"]
    assert [row["reward_rank"] for row in table] == [1, 2, 3, 4]
    assert table[0]["total_reward"] > table[1]["total_reward"]
    assert table[0]["variant_id"] == "D4P8"
    assert table[0]["tile_id"] == "tile_eval"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase66_reward_label_representation_audit.py -q --basetemp=.pytest_tmp_phase66_task1_red -p no:cacheprovider
```

Expected: `ModuleNotFoundError` or `ImportError` for `paper11_geofm.phase66_reward_label_representation_audit`.

- [ ] **Step 3: Add Phase 66 constants and component functions**

Create `src/paper11_geofm/phase66_reward_label_representation_audit.py` with this starting content:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
import statistics

import numpy as np

from .planning_reward import (
    BASE_PLANNING_REWARD_REQUIRED_COLUMNS,
    compute_base_planning_reward_from_matrix_row,
)
from .tiled_inputs import load_tiled_variant_input


PHASE66_CLAIM_BOUNDARY = (
    "Phase 66 is a read-only reward-label and representation attribution audit "
    "over existing Phase 63, Phase 64, and Phase 65 artifacts plus raw tiled "
    "feature matrices. It does not train or fine-tune a policy, does not change "
    "the base reward, does not enable suitability reward, does not test B2/B3 "
    "or transfer, does not prove GeoFM advantage or PCA optimality, and does "
    "not justify formal submission-level claims."
)

PHASE66_STATUS_REPRESENTATION_ADDS_SIGNAL = "representation_adds_reward_ranking_signal"
PHASE66_STATUS_REPRESENTATION_REDUNDANT = "representation_signal_redundant_with_explicit_reward"
PHASE66_STATUS_BASE_REWARD_MASKS = "base_reward_target_masks_geofm_signal"
PHASE66_STATUS_INSUFFICIENT = "insufficient"

PHASE66_REWARD_EQUIVALENT_TOLERANCE = 0.02
PHASE66_ALIGNMENT_ADVANTAGE_THRESHOLD = 0.05


PHASE66_COMPONENT_FIELDNAMES = [
    "variant_id",
    "tile_id",
    "block_id",
    "reward_rank",
    "source",
    "seed",
    "action_group",
    "total_reward",
    "low_slope_farmland_or_orchard_component",
    "current_farmland_or_orchard_component",
    "low_slope_component",
    "area_component",
    "mean_slope_penalty_component",
    "max_slope_penalty_component",
    "built_up_penalty_component",
    "water_penalty_component",
    "claim_boundary",
]


def _round_float(value: object, digits: int = 10) -> float:
    rounded = round(float(value), digits)
    compact = round(rounded, 6)
    if abs(rounded - compact) < 5.0e-8:
        return compact
    return rounded


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _column_value(
    column_to_index: Mapping[str, int],
    values: Sequence[float],
    column: str,
) -> float:
    return float(values[int(column_to_index[column])])


def decompose_phase66_base_reward_components(
    feature_columns: Sequence[str],
    values: Sequence[float],
) -> dict[str, float]:
    column_to_index = {str(column): index for index, column in enumerate(feature_columns)}
    missing = [
        column
        for column in BASE_PLANNING_REWARD_REQUIRED_COLUMNS
        if column not in column_to_index
    ]
    if missing:
        raise ValueError(
            "Phase 66 reward decomposition requires explicit feature columns: "
            f"{', '.join(missing)}"
        )
    low_slope_farmland_or_orchard = _clip01(
        _column_value(column_to_index, values, "explicit_feature_16")
    )
    current_farmland_or_orchard = max(
        _clip01(_column_value(column_to_index, values, "explicit_feature_04")),
        _clip01(_column_value(column_to_index, values, "explicit_feature_07")),
    )
    low_slope = _clip01(_column_value(column_to_index, values, "explicit_feature_13"))
    area_score = _clip01(
        _column_value(column_to_index, values, "explicit_feature_00") / 5.0
    )
    mean_slope_score = _clip01(
        _column_value(column_to_index, values, "explicit_feature_01") / 25.0
    )
    max_slope_score = _clip01(
        _column_value(column_to_index, values, "explicit_feature_02") / 35.0
    )
    built_up = _clip01(_column_value(column_to_index, values, "explicit_feature_09"))
    water = _clip01(_column_value(column_to_index, values, "explicit_feature_10"))
    components = {
        "low_slope_farmland_or_orchard_component": 0.35 * low_slope_farmland_or_orchard,
        "current_farmland_or_orchard_component": 0.20 * current_farmland_or_orchard,
        "low_slope_component": 0.10 * low_slope,
        "area_component": 0.10 * area_score,
        "mean_slope_penalty_component": -0.15 * mean_slope_score,
        "max_slope_penalty_component": -0.05 * max_slope_score,
        "built_up_penalty_component": -0.10 * built_up,
        "water_penalty_component": -0.10 * water,
    }
    rounded_components = {
        key: _round_float(value) for key, value in components.items()
    }
    rounded_components["total_reward"] = compute_base_planning_reward_from_matrix_row(
        feature_columns,
        values,
    )
    return rounded_components


def build_phase66_block_reward_table(tiled_input) -> list[dict[str, object]]:
    block_rows: list[dict[str, object]] = []
    for row_index, block_id in enumerate(tiled_input.block_ids):
        components = decompose_phase66_base_reward_components(
            tiled_input.feature_columns,
            tiled_input.state_matrix[row_index],
        )
        block_rows.append(
            {
                "variant_id": str(tiled_input.variant_id),
                "tile_id": str(tiled_input.tile_id),
                "block_id": str(block_id),
                **components,
            }
        )
    ranked_ids = {
        str(row["block_id"]): rank
        for rank, row in enumerate(
            sorted(
                block_rows,
                key=lambda item: (-float(item["total_reward"]), str(item["block_id"])),
            ),
            start=1,
        )
    }
    for row in block_rows:
        row["reward_rank"] = int(ranked_ids[str(row["block_id"])])
        row["claim_boundary"] = PHASE66_CLAIM_BOUNDARY
    return block_rows
```

- [ ] **Step 4: Run tests and verify Task 1 passes**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase66_reward_label_representation_audit.py -q --basetemp=.pytest_tmp_phase66_task1_green -p no:cacheprovider
```

Expected: the three Phase 66 tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src\paper11_geofm\phase66_reward_label_representation_audit.py tests\test_phase66_reward_label_representation_audit.py
git commit -m "feat: add Phase 66 reward component decomposition"
```

Expected: commit succeeds.

---

### Task 2: Selected-Block Atlas And Reward-Equivalent Substitutions

**Files:**
- Modify: `src/paper11_geofm/phase66_reward_label_representation_audit.py`
- Modify: `tests/test_phase66_reward_label_representation_audit.py`

- [ ] **Step 1: Add failing tests for atlas rows**

Append these tests to `tests/test_phase66_reward_label_representation_audit.py`:

```python
def _rollout_row(
    variant_id="D4P8",
    eval_tile_id="tile_eval",
    seed=0,
    selected="b1;b3",
    reward=0.0,
    oracle=0.0,
    gap=0.0,
    gap_fraction=0.0,
):
    return {
        "row_type": "bc_greedy_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": eval_tile_id,
        "eval_tile_rank": 1,
        "seed": seed,
        "phase63_seed_rank": seed + 1,
        "eval_max_steps": 2,
        "n_blocks": 4,
        "n_features": 9,
        "episode_steps": 2,
        "terminated": False,
        "truncated": True,
        "all_actions_valid": True,
        "invalid_action_count": 0,
        "total_contract_reward": reward,
        "oracle_total_reward": oracle,
        "oracle_gap": gap,
        "oracle_gap_fraction": gap_fraction,
        "selected_block_ids": selected,
        "selected_action_indices": "0;2",
        "claim_boundary": "fixture",
    }


def _oracle_row(
    variant_id="D4P8",
    tile_id="tile_eval",
    seed=0,
    selected="b1;b2",
    oracle=0.0,
):
    return {
        "variant_id": variant_id,
        "tile_role": "eval",
        "tile_id": tile_id,
        "seed": seed,
        "eval_max_steps": 2,
        "n_blocks": 4,
        "n_features": 9,
        "episode_steps": 2,
        "terminated": False,
        "total_oracle_reward": oracle,
        "top_k_reward_ceiling": oracle,
        "selected_block_ids": selected,
        "action_indices": "0;1",
        "claim_boundary": "fixture",
    }


def test_phase66_selected_block_atlas_reports_overlap_ranks_and_equivalence():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        build_phase66_selected_block_atlas,
    )

    tiled = _tiled_input(
        block_ids=("b1", "b2", "b3", "b4"),
        matrix=[
            [5.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.8, 0.90],
            [5.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.8, 0.86],
            [5.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.8, 0.84],
            [1.0, 25.0, 35.0, 0.0, 0.0, 0.5, 0.4, 0.1, 0.10],
        ],
    )

    rows = build_phase66_selected_block_atlas(
        phase63_rollout_rows=[_rollout_row(selected="b1;b3")],
        phase65_rollout_rows=[_rollout_row(selected="b1;b2")],
        oracle_rows=[_oracle_row(selected="b1;b2")],
        tiled_inputs={("D4P8", "tile_eval"): tiled},
        reward_tolerance=0.02,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["phase63_oracle_overlap_count"] == 1
    assert row["phase65_oracle_overlap_count"] == 2
    assert row["phase63_oracle_jaccard"] == 0.3333333333
    assert row["phase65_oracle_jaccard"] == 1.0
    assert row["phase63_selected_rank_values"] == "1;3"
    assert row["phase65_selected_rank_values"] == "1;2"
    assert row["phase63_reward_equivalent_substitution"] is True
    assert row["phase65_reward_equivalent_substitution"] is True
    assert row["phase63_extra_selected_block_ids"] == "b3"
    assert row["phase63_missed_oracle_block_ids"] == "b2"


def test_phase66_selected_block_atlas_rejects_missing_rollout_rows():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        build_phase66_selected_block_atlas,
    )

    try:
        build_phase66_selected_block_atlas(
            phase63_rollout_rows=[],
            phase65_rollout_rows=[_rollout_row()],
            oracle_rows=[_oracle_row()],
            tiled_inputs={("D4P8", "tile_eval"): _tiled_input()},
        )
    except ValueError as exc:
        assert "missing Phase 63 rollout row" in str(exc)
    else:
        raise AssertionError("Expected missing Phase 63 row to fail")
```

- [ ] **Step 2: Run atlas tests and verify they fail**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase66_reward_label_representation_audit.py::test_phase66_selected_block_atlas_reports_overlap_ranks_and_equivalence tests\test_phase66_reward_label_representation_audit.py::test_phase66_selected_block_atlas_rejects_missing_rollout_rows -q --basetemp=.pytest_tmp_phase66_task2_red -p no:cacheprovider
```

Expected: imports fail for `build_phase66_selected_block_atlas`.

- [ ] **Step 3: Implement atlas helpers**

Append this code to `src/paper11_geofm/phase66_reward_label_representation_audit.py`:

```python
PHASE66_ATLAS_FIELDNAMES = [
    "variant_id",
    "eval_tile_id",
    "seed",
    "oracle_block_ids",
    "phase63_selected_block_ids",
    "phase65_selected_block_ids",
    "phase63_oracle_overlap_count",
    "phase65_oracle_overlap_count",
    "phase63_oracle_jaccard",
    "phase65_oracle_jaccard",
    "phase63_phase65_jaccard",
    "phase63_selected_rank_values",
    "phase65_selected_rank_values",
    "phase63_missed_oracle_block_ids",
    "phase63_extra_selected_block_ids",
    "phase65_missed_oracle_block_ids",
    "phase65_extra_selected_block_ids",
    "phase63_reward_equivalent_substitution",
    "phase65_reward_equivalent_substitution",
    "claim_boundary",
]


def _split_semicolon_values(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _safe_int(value: object, default: int = 0) -> int:
    if value is None or str(value).strip() == "":
        return int(default)
    return int(float(value))


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def _row_key(row: Mapping[str, object]) -> tuple[str, str, int]:
    return (
        str(row.get("variant_id", "")),
        str(row.get("eval_tile_id", row.get("tile_id", ""))),
        _safe_int(row.get("seed")),
    )


def _index_unique_rows(
    rows: Sequence[Mapping[str, object]],
    label: str,
) -> dict[tuple[str, str, int], Mapping[str, object]]:
    index: dict[tuple[str, str, int], Mapping[str, object]] = {}
    for row in rows:
        key = _row_key(row)
        if key in index:
            raise ValueError(f"Phase 66 found duplicate {label} row for {key}")
        index[key] = row
    return index


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return _round_float(len(left_set & right_set) / len(union))


def _block_rank_and_reward(tiled_input) -> dict[str, dict[str, object]]:
    return {
        str(row["block_id"]): row
        for row in build_phase66_block_reward_table(tiled_input)
    }


def _rank_values(block_ids: Sequence[str], reward_index: Mapping[str, Mapping[str, object]]) -> str:
    return ";".join(str(int(reward_index[str(block_id)]["reward_rank"])) for block_id in block_ids)


def _reward_equivalent_substitution(
    missed_block_ids: Sequence[str],
    extra_block_ids: Sequence[str],
    reward_index: Mapping[str, Mapping[str, object]],
    tolerance: float,
) -> bool:
    if not missed_block_ids and not extra_block_ids:
        return True
    if not missed_block_ids or not extra_block_ids:
        return False
    missed_rewards = [float(reward_index[str(block_id)]["total_reward"]) for block_id in missed_block_ids]
    extra_rewards = [float(reward_index[str(block_id)]["total_reward"]) for block_id in extra_block_ids]
    return bool(abs(statistics.mean(missed_rewards) - statistics.mean(extra_rewards)) <= float(tolerance))


def _missing_block_ids(
    block_ids: Sequence[str],
    reward_index: Mapping[str, Mapping[str, object]],
) -> list[str]:
    return [str(block_id) for block_id in block_ids if str(block_id) not in reward_index]


def build_phase66_selected_block_atlas(
    phase63_rollout_rows: Sequence[Mapping[str, object]],
    phase65_rollout_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
    tiled_inputs: Mapping[tuple[str, str], object],
    reward_tolerance: float = PHASE66_REWARD_EQUIVALENT_TOLERANCE,
) -> list[dict[str, object]]:
    phase63_index = _index_unique_rows(
        [row for row in phase63_rollout_rows if str(row.get("row_type", "")) == "bc_greedy_policy"],
        "Phase 63 rollout",
    )
    phase65_index = _index_unique_rows(
        [row for row in phase65_rollout_rows if str(row.get("row_type", "")) == "bc_greedy_policy"],
        "Phase 65 rollout",
    )
    oracle_index = _index_unique_rows(oracle_rows, "Phase 63 oracle")
    rows: list[dict[str, object]] = []
    for key in sorted(oracle_index):
        variant_id, tile_id, seed = key
        if key not in phase63_index:
            raise ValueError(f"Phase 66 missing Phase 63 rollout row for {key}")
        if key not in phase65_index:
            raise ValueError(f"Phase 66 missing Phase 65 rollout row for {key}")
        tiled = tiled_inputs.get((variant_id, tile_id))
        if tiled is None:
            raise ValueError(f"Phase 66 missing tiled input for {(variant_id, tile_id)}")
        oracle_ids = _split_semicolon_values(oracle_index[key].get("selected_block_ids"))
        phase63_ids = _split_semicolon_values(phase63_index[key].get("selected_block_ids"))
        phase65_ids = _split_semicolon_values(phase65_index[key].get("selected_block_ids"))
        reward_index = _block_rank_and_reward(tiled)
        missing = (
            _missing_block_ids(oracle_ids, reward_index)
            + _missing_block_ids(phase63_ids, reward_index)
            + _missing_block_ids(phase65_ids, reward_index)
        )
        if missing:
            raise ValueError(f"Phase 66 selected block IDs missing from tiled input: {missing[:5]}")
        phase63_missed = [block_id for block_id in oracle_ids if block_id not in set(phase63_ids)]
        phase63_extra = [block_id for block_id in phase63_ids if block_id not in set(oracle_ids)]
        phase65_missed = [block_id for block_id in oracle_ids if block_id not in set(phase65_ids)]
        phase65_extra = [block_id for block_id in phase65_ids if block_id not in set(oracle_ids)]
        rows.append(
            {
                "variant_id": variant_id,
                "eval_tile_id": tile_id,
                "seed": int(seed),
                "oracle_block_ids": ";".join(oracle_ids),
                "phase63_selected_block_ids": ";".join(phase63_ids),
                "phase65_selected_block_ids": ";".join(phase65_ids),
                "phase63_oracle_overlap_count": len(set(phase63_ids) & set(oracle_ids)),
                "phase65_oracle_overlap_count": len(set(phase65_ids) & set(oracle_ids)),
                "phase63_oracle_jaccard": _jaccard(phase63_ids, oracle_ids),
                "phase65_oracle_jaccard": _jaccard(phase65_ids, oracle_ids),
                "phase63_phase65_jaccard": _jaccard(phase63_ids, phase65_ids),
                "phase63_selected_rank_values": _rank_values(phase63_ids, reward_index),
                "phase65_selected_rank_values": _rank_values(phase65_ids, reward_index),
                "phase63_missed_oracle_block_ids": ";".join(phase63_missed),
                "phase63_extra_selected_block_ids": ";".join(phase63_extra),
                "phase65_missed_oracle_block_ids": ";".join(phase65_missed),
                "phase65_extra_selected_block_ids": ";".join(phase65_extra),
                "phase63_reward_equivalent_substitution": _reward_equivalent_substitution(
                    phase63_missed,
                    phase63_extra,
                    reward_index,
                    reward_tolerance,
                ),
                "phase65_reward_equivalent_substitution": _reward_equivalent_substitution(
                    phase65_missed,
                    phase65_extra,
                    reward_index,
                    reward_tolerance,
                ),
                "claim_boundary": PHASE66_CLAIM_BOUNDARY,
            }
        )
    return rows
```

- [ ] **Step 4: Run atlas tests and verify they pass**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase66_reward_label_representation_audit.py -q --basetemp=.pytest_tmp_phase66_task2_green -p no:cacheprovider
```

Expected: all current Phase 66 tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src\paper11_geofm\phase66_reward_label_representation_audit.py tests\test_phase66_reward_label_representation_audit.py
git commit -m "feat: add Phase 66 selected-block atlas"
```

Expected: commit succeeds.

---

### Task 3: Representation-Rank Alignment Metrics

**Files:**
- Modify: `src/paper11_geofm/phase66_reward_label_representation_audit.py`
- Modify: `tests/test_phase66_reward_label_representation_audit.py`

- [ ] **Step 1: Add failing tests for rank alignment and feature grouping**

Append these tests:

```python
def test_phase66_rank_metric_handles_ties_and_constant_columns():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        phase66_spearman_abs,
        phase66_topk_enrichment,
    )

    assert phase66_spearman_abs([1.0, 2.0, 2.0, 4.0], [0.1, 0.2, 0.2, 0.4]) == 1.0
    assert phase66_spearman_abs([1.0, 1.0, 1.0], [0.1, 0.2, 0.3]) == 0.0
    assert phase66_topk_enrichment([0.9, 0.8, 0.1, 0.0], [1.0, 0.7, 0.2, 0.1], top_k=2) == 1.0
    assert phase66_topk_enrichment([0.0, 0.1, 0.8, 0.9], [1.0, 0.7, 0.2, 0.1], top_k=2) == 1.0


def test_phase66_representation_rank_alignment_separates_explicit_and_extra_columns():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        build_phase66_representation_rank_alignment,
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
        "embedding_pca_00",
        "embedding_pca_01",
    )
    matrix = np.asarray(
        [
            [5.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.8, 0.9, 0.9, 0.0],
            [4.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.7, 0.8, 0.8, 0.0],
            [3.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.2, 0.3, 0.3, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.1, 0.1, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    tiled = _tiled_input(columns=columns, matrix=matrix, variant_id="D4P8")

    rows = build_phase66_representation_rank_alignment(
        tiled_inputs={("D4P8", "tile_eval"): tiled},
        eval_max_steps=2,
    )
    by_group = {row["feature_group"]: row for row in rows}

    assert by_group["reward_explicit"]["n_columns"] == 9
    assert by_group["representation_extra"]["n_columns"] == 2
    assert by_group["representation_extra"]["max_abs_spearman"] == 1.0
    assert by_group["representation_extra"]["best_topk_enrichment"] == 1.0
    assert by_group["representation_extra"]["proxy_r2"] > 0.9


def test_phase66_representation_alignment_rejects_geofm_variant_without_extra_columns():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        build_phase66_representation_rank_alignment,
    )

    try:
        build_phase66_representation_rank_alignment(
            tiled_inputs={("D4P8", "tile_eval"): _tiled_input(variant_id="D4P8")},
            eval_max_steps=2,
        )
    except ValueError as exc:
        assert "representation columns" in str(exc)
    else:
        raise AssertionError("Expected D4/D6 without representation columns to fail")
```

- [ ] **Step 2: Run rank-alignment tests and verify they fail**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase66_reward_label_representation_audit.py::test_phase66_rank_metric_handles_ties_and_constant_columns tests\test_phase66_reward_label_representation_audit.py::test_phase66_representation_rank_alignment_separates_explicit_and_extra_columns tests\test_phase66_reward_label_representation_audit.py::test_phase66_representation_alignment_rejects_geofm_variant_without_extra_columns -q --basetemp=.pytest_tmp_phase66_task3_red -p no:cacheprovider
```

Expected: imports fail for rank-alignment helpers.

- [ ] **Step 3: Implement rank alignment**

Append this code:

```python
PHASE66_ALIGNMENT_FIELDNAMES = [
    "variant_id",
    "tile_id",
    "feature_group",
    "n_columns",
    "mean_abs_spearman",
    "max_abs_spearman",
    "best_topk_enrichment",
    "proxy_r2",
    "best_feature_name",
    "claim_boundary",
]


def _rank_average(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def phase66_spearman_abs(feature_values: Sequence[float], reward_values: Sequence[float]) -> float:
    x = np.asarray(feature_values, dtype=np.float64)
    y = np.asarray(reward_values, dtype=np.float64)
    if x.size != y.size:
        raise ValueError("Phase 66 Spearman inputs must have equal length")
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    rx = _rank_average(x)
    ry = _rank_average(y)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return 0.0
    corr = float(np.corrcoef(rx, ry)[0, 1])
    if np.isnan(corr):
        return 0.0
    return _round_float(abs(corr))


def phase66_topk_enrichment(
    feature_values: Sequence[float],
    reward_values: Sequence[float],
    top_k: int,
) -> float:
    x = np.asarray(feature_values, dtype=np.float64)
    y = np.asarray(reward_values, dtype=np.float64)
    if x.size != y.size:
        raise ValueError("Phase 66 top-k enrichment inputs must have equal length")
    k = min(int(top_k), int(x.size))
    if k <= 0:
        return 0.0
    reward_top = set(np.argsort(-y, kind="mergesort")[:k].tolist())
    high_top = set(np.argsort(-x, kind="mergesort")[:k].tolist())
    low_top = set(np.argsort(x, kind="mergesort")[:k].tolist())
    return _round_float(max(len(reward_top & high_top), len(reward_top & low_top)) / k)


def _proxy_r2(matrix: np.ndarray, reward_values: np.ndarray) -> float:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(reward_values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] == 0:
        return 0.0
    keep = np.std(x, axis=0) > 1.0e-12
    if not bool(np.any(keep)) or float(np.std(y)) == 0.0:
        return 0.0
    z = x[:, keep]
    z = (z - np.mean(z, axis=0)) / np.std(z, axis=0)
    design = np.column_stack([np.ones(z.shape[0]), z])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coeffs
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 1.0e-12:
        return 0.0
    residual = float(np.sum((y - predicted) ** 2))
    return _round_float(max(0.0, min(1.0, 1.0 - residual / total)))


def _phase66_feature_groups(feature_columns: Sequence[str]) -> dict[str, list[int]]:
    reward_required = set(BASE_PLANNING_REWARD_REQUIRED_COLUMNS)
    reward_explicit = [
        index for index, column in enumerate(feature_columns) if str(column) in reward_required
    ]
    nonreward_explicit = [
        index
        for index, column in enumerate(feature_columns)
        if str(column).startswith("explicit_feature_") and str(column) not in reward_required
    ]
    representation_extra = [
        index
        for index, column in enumerate(feature_columns)
        if not str(column).startswith("explicit_feature_")
    ]
    return {
        "reward_explicit": reward_explicit,
        "nonreward_explicit": nonreward_explicit,
        "representation_extra": representation_extra,
    }


def _alignment_row(
    tiled_input,
    group_name: str,
    indexes: Sequence[int],
    reward_values: np.ndarray,
    eval_max_steps: int,
) -> dict[str, object]:
    if not indexes:
        return {
            "variant_id": str(tiled_input.variant_id),
            "tile_id": str(tiled_input.tile_id),
            "feature_group": group_name,
            "n_columns": 0,
            "mean_abs_spearman": 0.0,
            "max_abs_spearman": 0.0,
            "best_topk_enrichment": 0.0,
            "proxy_r2": 0.0,
            "best_feature_name": "",
            "claim_boundary": PHASE66_CLAIM_BOUNDARY,
        }
    matrix = np.asarray(tiled_input.state_matrix[:, list(indexes)], dtype=np.float64)
    spearman_values = [
        phase66_spearman_abs(matrix[:, col], reward_values)
        for col in range(matrix.shape[1])
    ]
    enrichment_values = [
        phase66_topk_enrichment(matrix[:, col], reward_values, top_k=eval_max_steps)
        for col in range(matrix.shape[1])
    ]
    best_index = int(np.argmax(spearman_values)) if spearman_values else 0
    return {
        "variant_id": str(tiled_input.variant_id),
        "tile_id": str(tiled_input.tile_id),
        "feature_group": group_name,
        "n_columns": int(len(indexes)),
        "mean_abs_spearman": _round_float(statistics.mean(spearman_values)),
        "max_abs_spearman": _round_float(max(spearman_values)),
        "best_topk_enrichment": _round_float(max(enrichment_values)),
        "proxy_r2": _proxy_r2(matrix, reward_values),
        "best_feature_name": str(tiled_input.feature_columns[int(indexes[best_index])]),
        "claim_boundary": PHASE66_CLAIM_BOUNDARY,
    }


def build_phase66_representation_rank_alignment(
    tiled_inputs: Mapping[tuple[str, str], object],
    eval_max_steps: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(tiled_inputs):
        tiled = tiled_inputs[key]
        reward_values = np.asarray(
            [
                compute_base_planning_reward_from_matrix_row(
                    tiled.feature_columns,
                    tiled.state_matrix[row_index],
                )
                for row_index in range(len(tiled.block_ids))
            ],
            dtype=np.float64,
        )
        groups = _phase66_feature_groups(tiled.feature_columns)
        if str(tiled.variant_id) != "B0" and not groups["representation_extra"]:
            raise ValueError(
                f"Phase 66 cannot separate representation columns for {tiled.variant_id}"
            )
        for group_name, indexes in groups.items():
            rows.append(
                _alignment_row(
                    tiled,
                    group_name,
                    indexes,
                    reward_values,
                    eval_max_steps=eval_max_steps,
                )
            )
    return rows
```

- [ ] **Step 4: Run all Phase 66 tests**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase66_reward_label_representation_audit.py -q --basetemp=.pytest_tmp_phase66_task3_green -p no:cacheprovider
```

Expected: all current Phase 66 tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add src\paper11_geofm\phase66_reward_label_representation_audit.py tests\test_phase66_reward_label_representation_audit.py
git commit -m "feat: add Phase 66 representation rank alignment"
```

Expected: commit succeeds.

---

### Task 4: Reward Attribution Rows, Failure Modes, And Status Gate

**Files:**
- Modify: `src/paper11_geofm/phase66_reward_label_representation_audit.py`
- Modify: `tests/test_phase66_reward_label_representation_audit.py`

- [ ] **Step 1: Add failing tests for attribution rows and gate statuses**

Append these tests:

```python
def test_phase66_failure_modes_cover_reward_equivalent_component_miss_and_standardization_hurt():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        build_phase66_failure_mode_summary,
    )

    atlas_rows = [
        {
            "variant_id": "D4P8",
            "eval_tile_id": "tile_a",
            "seed": 0,
            "phase63_oracle_jaccard": 0.2,
            "phase65_oracle_jaccard": 0.9,
            "phase63_reward_equivalent_substitution": True,
            "phase65_reward_equivalent_substitution": True,
            "phase63_missed_oracle_block_ids": "b2",
            "phase63_extra_selected_block_ids": "b3",
        },
        {
            "variant_id": "D4P16",
            "eval_tile_id": "tile_b",
            "seed": 1,
            "phase63_oracle_jaccard": 0.1,
            "phase65_oracle_jaccard": 0.1,
            "phase63_reward_equivalent_substitution": False,
            "phase65_reward_equivalent_substitution": False,
            "phase63_missed_oracle_block_ids": "b2",
            "phase63_extra_selected_block_ids": "b4",
        },
    ]
    alignment_rows = [
        {"variant_id": "D4P8", "feature_group": "representation_extra", "proxy_r2": 0.10, "max_abs_spearman": 0.20},
        {"variant_id": "D4P8", "feature_group": "reward_explicit", "proxy_r2": 0.90, "max_abs_spearman": 0.95},
        {"variant_id": "D4P16", "feature_group": "representation_extra", "proxy_r2": 0.10, "max_abs_spearman": 0.20},
        {"variant_id": "D4P16", "feature_group": "reward_explicit", "proxy_r2": 0.90, "max_abs_spearman": 0.95},
    ]
    phase65_pairwise_rows = [
        {
            "variant_id": "D4P16",
            "eval_tile_id": "tile_b",
            "seed": 1,
            "standardized_minus_unstandardized_reward": -0.5,
        }
    ]

    rows = build_phase66_failure_mode_summary(
        atlas_rows,
        alignment_rows,
        phase65_pairwise_rows,
    )
    modes = {row["failure_mode"]: row for row in rows}

    assert modes["near_oracle_reward_equivalent"]["case_count"] == 1
    assert modes["misses_explicit_reward_components"]["case_count"] == 1
    assert modes["representation_not_aligned_with_base_reward"]["case_count"] == 2
    assert modes["standardization_hurts_rank_geometry"]["case_count"] == 1


def test_phase66_diagnostic_gate_covers_all_statuses():
    from paper11_geofm.phase66_reward_label_representation_audit import (
        build_phase66_diagnostic_gate,
    )

    strong_alignment = [
        {"variant_id": "B0", "feature_group": "reward_explicit", "proxy_r2": 0.60, "max_abs_spearman": 0.60, "best_topk_enrichment": 0.50},
        {"variant_id": "D4P8", "feature_group": "reward_explicit", "proxy_r2": 0.60, "max_abs_spearman": 0.60, "best_topk_enrichment": 0.50},
        {"variant_id": "D4P8", "feature_group": "representation_extra", "proxy_r2": 0.80, "max_abs_spearman": 0.80, "best_topk_enrichment": 0.75},
    ]
    redundant_alignment = [
        {"variant_id": "B0", "feature_group": "reward_explicit", "proxy_r2": 0.85, "max_abs_spearman": 0.90, "best_topk_enrichment": 1.00},
        {"variant_id": "D4P8", "feature_group": "reward_explicit", "proxy_r2": 0.86, "max_abs_spearman": 0.91, "best_topk_enrichment": 1.00},
        {"variant_id": "D4P8", "feature_group": "representation_extra", "proxy_r2": 0.84, "max_abs_spearman": 0.89, "best_topk_enrichment": 1.00},
    ]
    failure_summary = [
        {"failure_mode": "misses_explicit_reward_components", "case_count": 5},
        {"failure_mode": "representation_not_aligned_with_base_reward", "case_count": 5},
    ]

    assert build_phase66_diagnostic_gate([], strong_alignment, [], {})["phase66_status"] == "representation_adds_reward_ranking_signal"
    assert build_phase66_diagnostic_gate([], redundant_alignment, [], {})["phase66_status"] == "representation_signal_redundant_with_explicit_reward"
    assert build_phase66_diagnostic_gate([], redundant_alignment, failure_summary, {"phase10_status": "not_ready_for_suitability_reward"})["phase66_status"] == "base_reward_target_masks_geofm_signal"
    assert build_phase66_diagnostic_gate(["missing row"], redundant_alignment, failure_summary, {})["phase66_status"] == "insufficient"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase66_reward_label_representation_audit.py::test_phase66_failure_modes_cover_reward_equivalent_component_miss_and_standardization_hurt tests\test_phase66_reward_label_representation_audit.py::test_phase66_diagnostic_gate_covers_all_statuses -q --basetemp=.pytest_tmp_phase66_task4_red -p no:cacheprovider
```

Expected: imports fail for failure summary and gate functions.

- [ ] **Step 3: Implement failure modes and gate**

Append this code:

```python
PHASE66_FAILURE_FIELDNAMES = [
    "failure_mode",
    "case_count",
    "representative_cases",
    "claim_boundary",
]


def _alignment_advantage_summary(
    alignment_rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    b0_explicit = [
        row for row in alignment_rows
        if str(row.get("variant_id")) == "B0" and str(row.get("feature_group")) == "reward_explicit"
    ]
    geofm_rep = [
        row for row in alignment_rows
        if str(row.get("variant_id", "")).startswith(("D4", "D6"))
        and str(row.get("feature_group")) == "representation_extra"
    ]
    geofm_explicit = [
        row for row in alignment_rows
        if str(row.get("variant_id", "")).startswith(("D4", "D6"))
        and str(row.get("feature_group")) == "reward_explicit"
    ]
    b0_r2 = statistics.mean([_safe_float(row.get("proxy_r2")) for row in b0_explicit]) if b0_explicit else 0.0
    rep_r2 = statistics.mean([_safe_float(row.get("proxy_r2")) for row in geofm_rep]) if geofm_rep else 0.0
    explicit_r2 = statistics.mean([_safe_float(row.get("proxy_r2")) for row in geofm_explicit]) if geofm_explicit else 0.0
    rep_topk = statistics.mean([_safe_float(row.get("best_topk_enrichment")) for row in geofm_rep]) if geofm_rep else 0.0
    explicit_topk = statistics.mean([_safe_float(row.get("best_topk_enrichment")) for row in geofm_explicit]) if geofm_explicit else 0.0
    return {
        "b0_explicit_proxy_r2_mean": _round_float(b0_r2),
        "geofm_explicit_proxy_r2_mean": _round_float(explicit_r2),
        "geofm_representation_proxy_r2_mean": _round_float(rep_r2),
        "representation_minus_b0_proxy_r2": _round_float(rep_r2 - b0_r2),
        "representation_minus_explicit_proxy_r2": _round_float(rep_r2 - explicit_r2),
        "representation_minus_explicit_topk": _round_float(rep_topk - explicit_topk),
    }


def _case_id(row: Mapping[str, object]) -> str:
    return f"{row.get('variant_id')}:{row.get('eval_tile_id')}:{row.get('seed')}"


def build_phase66_failure_mode_summary(
    atlas_rows: Sequence[Mapping[str, object]],
    alignment_rows: Sequence[Mapping[str, object]],
    phase65_pairwise_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    modes: dict[str, set[str]] = {
        "near_oracle_reward_equivalent": set(),
        "misses_explicit_reward_components": set(),
        "representation_not_aligned_with_base_reward": set(),
        "standardization_hurts_rank_geometry": set(),
        "tile_specific_instability": set(),
        "seed_instability": set(),
    }
    for row in atlas_rows:
        case = _case_id(row)
        if _safe_float(row.get("phase63_oracle_jaccard")) < 0.5 and bool(row.get("phase63_reward_equivalent_substitution")):
            modes["near_oracle_reward_equivalent"].add(case)
        if not bool(row.get("phase63_reward_equivalent_substitution")) and str(row.get("phase63_missed_oracle_block_ids", "")):
            modes["misses_explicit_reward_components"].add(case)
    by_variant: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in alignment_rows:
        by_variant.setdefault(str(row.get("variant_id")), {})[str(row.get("feature_group"))] = row
    for variant_id, groups in by_variant.items():
        if not variant_id.startswith(("D4", "D6")):
            continue
        rep = groups.get("representation_extra")
        explicit = groups.get("reward_explicit")
        if rep is None or explicit is None:
            continue
        if _safe_float(rep.get("proxy_r2")) + PHASE66_ALIGNMENT_ADVANTAGE_THRESHOLD < _safe_float(explicit.get("proxy_r2")):
            for row in atlas_rows:
                if str(row.get("variant_id")) == variant_id:
                    modes["representation_not_aligned_with_base_reward"].add(_case_id(row))
    for row in phase65_pairwise_rows:
        if _safe_float(row.get("standardized_minus_unstandardized_reward")) < 0.0:
            modes["standardization_hurts_rank_geometry"].add(_case_id(row))
    tile_counts: dict[str, int] = {}
    seed_counts: dict[str, int] = {}
    for row in atlas_rows:
        if _safe_float(row.get("phase63_oracle_jaccard")) < 0.5:
            tile_counts[str(row.get("eval_tile_id"))] = tile_counts.get(str(row.get("eval_tile_id")), 0) + 1
            seed_counts[str(row.get("seed"))] = seed_counts.get(str(row.get("seed")), 0) + 1
    for tile_id, count in tile_counts.items():
        if count >= 2:
            modes["tile_specific_instability"].add(tile_id)
    for seed, count in seed_counts.items():
        if count >= 2:
            modes["seed_instability"].add(seed)
    return [
        {
            "failure_mode": mode,
            "case_count": len(cases),
            "representative_cases": ";".join(sorted(cases)[:5]),
            "claim_boundary": PHASE66_CLAIM_BOUNDARY,
        }
        for mode, cases in modes.items()
    ]


def build_phase66_diagnostic_gate(
    coverage_issues: Sequence[object],
    alignment_rows: Sequence[Mapping[str, object]],
    failure_summary_rows: Sequence[Mapping[str, object]],
    suitability_context: Mapping[str, object],
) -> dict[str, object]:
    if coverage_issues:
        return {
            "phase66_status": PHASE66_STATUS_INSUFFICIENT,
            "coverage_issues": list(coverage_issues),
            "alignment_advantage": {},
            "claim_boundary": PHASE66_CLAIM_BOUNDARY,
        }
    advantage = _alignment_advantage_summary(alignment_rows)
    rep_minus_b0 = float(advantage["representation_minus_b0_proxy_r2"])
    rep_minus_explicit = float(advantage["representation_minus_explicit_proxy_r2"])
    failure_counts = {
        str(row.get("failure_mode")): _safe_int(row.get("case_count"))
        for row in failure_summary_rows
    }
    if rep_minus_b0 >= PHASE66_ALIGNMENT_ADVANTAGE_THRESHOLD and rep_minus_explicit >= PHASE66_ALIGNMENT_ADVANTAGE_THRESHOLD:
        status = PHASE66_STATUS_REPRESENTATION_ADDS_SIGNAL
    elif (
        failure_counts.get("misses_explicit_reward_components", 0) > 0
        and failure_counts.get("representation_not_aligned_with_base_reward", 0) > 0
        and str(suitability_context.get("phase10_status", "")).startswith("not_ready")
    ):
        status = PHASE66_STATUS_BASE_REWARD_MASKS
    else:
        status = PHASE66_STATUS_REPRESENTATION_REDUNDANT
    return {
        "phase66_status": status,
        "coverage_issues": [],
        "alignment_advantage": advantage,
        "failure_mode_counts": failure_counts,
        "suitability_context": dict(suitability_context),
        "claim_boundary": PHASE66_CLAIM_BOUNDARY,
    }
```

- [ ] **Step 4: Run all Phase 66 tests**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase66_reward_label_representation_audit.py -q --basetemp=.pytest_tmp_phase66_task4_green -p no:cacheprovider
```

Expected: all current Phase 66 tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add src\paper11_geofm\phase66_reward_label_representation_audit.py tests\test_phase66_reward_label_representation_audit.py
git commit -m "feat: add Phase 66 diagnostic gate"
```

Expected: commit succeeds.

---

### Task 5: Artifact Writer, CLI, And Full Read-Only Run Orchestration

**Files:**
- Modify: `src/paper11_geofm/phase66_reward_label_representation_audit.py`
- Create: `experiments/phase66_reward_label_representation_audit/run_phase66_reward_label_representation_audit.py`
- Modify: `tests/test_phase66_reward_label_representation_audit.py`

- [ ] **Step 1: Add failing tests for writer, parser, and tiny full run**

Append these tests:

```python
def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_variant_fixture(output_dir: Path, variant_id: str, rows: list[dict[str, object]], columns: tuple[str, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table = output_dir / f"variant_{variant_id}_features.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
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


def test_phase66_writer_outputs_json_csv_and_markdown(tmp_path):
    from paper11_geofm.phase66_reward_label_representation_audit import (
        write_phase66_artifacts,
    )

    analysis = {
        "phase": "phase66_reward_label_representation_audit",
        "reward_component_rows": [{"variant_id": "B0", "tile_id": "tile_eval", "block_id": "b1", "reward_rank": 1, "source": "oracle", "seed": 0, "action_group": "oracle", "total_reward": 0.5, "claim_boundary": "phase66"}],
        "selected_block_atlas_rows": [{"variant_id": "B0", "eval_tile_id": "tile_eval", "seed": 0, "claim_boundary": "phase66"}],
        "representation_rank_alignment_rows": [{"variant_id": "B0", "tile_id": "tile_eval", "feature_group": "reward_explicit", "n_columns": 9, "proxy_r2": 1.0, "claim_boundary": "phase66"}],
        "failure_mode_summary_rows": [{"failure_mode": "near_oracle_reward_equivalent", "case_count": 1, "representative_cases": "B0:tile_eval:0", "claim_boundary": "phase66"}],
        "diagnostic_gate": {"phase66_status": "representation_signal_redundant_with_explicit_reward"},
        "claim_boundary": "phase66",
    }

    paths = write_phase66_artifacts(analysis, tmp_path / "outputs")

    assert paths["component_csv"].name == "phase66_reward_component_attribution.csv"
    assert paths["atlas_csv"].name == "phase66_selected_block_atlas.csv"
    assert paths["alignment_csv"].name == "phase66_representation_rank_alignment.csv"
    assert paths["failure_csv"].name == "phase66_failure_mode_summary.csv"
    assert paths["audit_json"].name == "phase66_reward_label_representation_audit.json"
    assert paths["audit_md"].name == "phase66_reward_label_representation_audit.md"
    saved = json.loads(paths["audit_json"].read_text(encoding="utf-8"))
    assert saved["phase66_status"] == "representation_signal_redundant_with_explicit_reward"
    assert "Phase 66 Reward-Label Representation Audit" in paths["audit_md"].read_text(encoding="utf-8")


def test_phase66_cli_parser_accepts_required_inputs():
    runner_path = (
        ROOT
        / "experiments"
        / "phase66_reward_label_representation_audit"
        / "run_phase66_reward_label_representation_audit.py"
    )
    spec = importlib.util.spec_from_file_location("phase66_runner_args", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--phase63-comparison-json", "phase63.json",
            "--phase63-rollout-csv", "phase63_rollout.csv",
            "--phase63-oracle-summary-csv", "phase63_oracle.csv",
            "--phase64-failure-cases-csv", "phase64_failure.csv",
            "--phase64-feature-effective-rank-csv", "phase64_rank.csv",
            "--phase65-comparison-json", "phase65.json",
            "--phase65-rollout-csv", "phase65_rollout.csv",
            "--phase65-pairwise-delta-csv", "phase65_pairwise.csv",
            "--phase10-reward-readiness-json", "phase10.json",
            "--output-dir", "outputs",
        ]
    )

    assert args.phase63_comparison_json == Path("phase63.json")
    assert args.phase65_pairwise_delta_csv == Path("phase65_pairwise.csv")
    assert args.output_dir == Path("outputs")


def test_phase66_run_wrapper_loads_contract_and_returns_read_only_analysis(tmp_path):
    from paper11_geofm.phase66_reward_label_representation_audit import (
        run_phase66_reward_label_representation_audit,
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
        "embedding_pca_00",
    )
    feature_rows = [
        {**{"block_id": "b1"}, **{column: 0.0 for column in columns}},
        {**{"block_id": "b2"}, **{column: 0.0 for column in columns}},
        {**{"block_id": "b3"}, **{column: 0.0 for column in columns}},
    ]
    feature_rows[0]["explicit_feature_16"] = 0.9
    feature_rows[1]["explicit_feature_16"] = 0.8
    feature_rows[2]["explicit_feature_16"] = 0.1
    feature_rows[0]["embedding_pca_00"] = 0.9
    feature_rows[1]["embedding_pca_00"] = 0.8
    feature_rows[2]["embedding_pca_00"] = 0.1
    phase2 = tmp_path / "phase2"
    _write_variant_fixture(phase2, "D4P8", feature_rows, columns)
    tile_index = _write_csv(
        tmp_path / "tiles.csv",
        [
            {"tile_id": "tile_train", "block_ids": "b1;b2;b3"},
            {"tile_id": "tile_eval", "block_ids": "b1;b2;b3"},
        ],
    )
    comparison = {
        "contract": {
            "tile_index_csv": str(tile_index),
            "variant_source_dirs": {"D4P8": str(phase2)},
            "variants": ["D4P8"],
            "train_tile_id": "tile_train",
            "eval_tile_ids": ["tile_eval"],
            "seeds": [0],
            "eval_max_steps": 2,
        }
    }
    comparison_path = tmp_path / "phase63.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    phase63_rollout = _write_csv(tmp_path / "phase63_rollout.csv", [_rollout_row(selected="b1;b3")])
    phase65_rollout = _write_csv(tmp_path / "phase65_rollout.csv", [_rollout_row(selected="b1;b2")])
    oracle = _write_csv(tmp_path / "oracle.csv", [_oracle_row(selected="b1;b2")])
    pairwise = _write_csv(
        tmp_path / "pairwise.csv",
        [
            {
                "variant_id": "D4P8",
                "eval_tile_id": "tile_eval",
                "seed": 0,
                "standardized_minus_unstandardized_reward": 0.1,
            }
        ],
    )
    phase65_json = tmp_path / "phase65.json"
    phase65_json.write_text(json.dumps({"phase65_status": "fixture"}), encoding="utf-8")
    phase10_json = tmp_path / "phase10.json"
    phase10_json.write_text(json.dumps({"phase10_status": "not_ready_for_suitability_reward"}), encoding="utf-8")

    analysis = run_phase66_reward_label_representation_audit(
        phase63_comparison_json=comparison_path,
        phase63_rollout_csv=phase63_rollout,
        phase63_oracle_summary_csv=oracle,
        phase64_failure_cases_csv=None,
        phase64_feature_effective_rank_csv=None,
        phase65_comparison_json=phase65_json,
        phase65_rollout_csv=phase65_rollout,
        phase65_pairwise_delta_csv=pairwise,
        phase10_reward_readiness_json=phase10_json,
    )

    assert analysis["phase"] == "phase66_reward_label_representation_audit"
    assert len(analysis["selected_block_atlas_rows"]) == 1
    assert len(analysis["representation_rank_alignment_rows"]) == 3
    assert analysis["diagnostic_gate"]["phase66_status"] in {
        "representation_adds_reward_ranking_signal",
        "representation_signal_redundant_with_explicit_reward",
        "base_reward_target_masks_geofm_signal",
        "insufficient",
    }
```

- [ ] **Step 2: Run new tests and verify they fail**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase66_reward_label_representation_audit.py::test_phase66_writer_outputs_json_csv_and_markdown tests\test_phase66_reward_label_representation_audit.py::test_phase66_cli_parser_accepts_required_inputs tests\test_phase66_reward_label_representation_audit.py::test_phase66_run_wrapper_loads_contract_and_returns_read_only_analysis -q --basetemp=.pytest_tmp_phase66_task5_red -p no:cacheprovider
```

Expected: imports fail for writer, runner, or run wrapper.

- [ ] **Step 3: Add loaders, writer, and run wrapper**

Implement these functions in `src/paper11_geofm/phase66_reward_label_representation_audit.py`:

```python
def _load_json_object(path: Path | str | None, label: str) -> dict[str, object]:
    if path is None:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {label}: {json_path}")
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return loaded


def _load_csv_rows(path: Path | str | None, label: str) -> list[dict[str, object]]:
    if path is None:
        return []
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {label}: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def _load_phase66_tiled_inputs(contract: Mapping[str, object]) -> dict[tuple[str, str], object]:
    variant_source_dirs = contract.get("variant_source_dirs")
    if not isinstance(variant_source_dirs, Mapping):
        raise ValueError("Phase 66 contract is missing variant_source_dirs")
    tile_index_csv = contract.get("tile_index_csv")
    if not tile_index_csv:
        raise ValueError("Phase 66 contract is missing tile_index_csv")
    variants = _contract_string_list(contract, "variants")
    eval_tile_ids = _contract_string_list(contract, "eval_tile_ids")
    train_tile_id = str(contract.get("train_tile_id", ""))
    tile_ids = [train_tile_id, *eval_tile_ids] if train_tile_id else eval_tile_ids
    if not variants:
        raise ValueError("Phase 66 contract has no variants")
    if not tile_ids:
        raise ValueError("Phase 66 contract has no train/eval tile IDs")
    tiled_inputs: dict[tuple[str, str], object] = {}
    for variant_id in variants:
        source_dir = variant_source_dirs.get(variant_id)
        if source_dir is None:
            raise ValueError(f"Phase 66 contract has no source for variant {variant_id}")
        for tile_id in tile_ids:
            tiled_inputs[(str(variant_id), str(tile_id))] = load_tiled_variant_input(
                source_dir,
                str(tile_index_csv),
                str(tile_id),
                variant_id=str(variant_id),
            )
    return tiled_inputs


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


def _phase66_markdown(analysis: Mapping[str, object]) -> str:
    gate = dict(analysis.get("diagnostic_gate", {}))
    lines = [
        "# Phase 66 Reward-Label Representation Audit",
        "",
        f"Status: {gate.get('phase66_status', '')}",
        "",
        f"Alignment advantage: {gate.get('alignment_advantage', {})}",
        f"Failure mode counts: {gate.get('failure_mode_counts', {})}",
        "",
        "Claim boundary:",
        str(analysis.get("claim_boundary", PHASE66_CLAIM_BOUNDARY)),
        "",
    ]
    return "\n".join(lines)


def write_phase66_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "component_csv": output_path / "phase66_reward_component_attribution.csv",
        "atlas_csv": output_path / "phase66_selected_block_atlas.csv",
        "alignment_csv": output_path / "phase66_representation_rank_alignment.csv",
        "failure_csv": output_path / "phase66_failure_mode_summary.csv",
        "audit_json": output_path / "phase66_reward_label_representation_audit.json",
        "audit_md": output_path / "phase66_reward_label_representation_audit.md",
    }
    _write_csv_rows(paths["component_csv"], PHASE66_COMPONENT_FIELDNAMES, analysis.get("reward_component_rows", []))
    _write_csv_rows(paths["atlas_csv"], PHASE66_ATLAS_FIELDNAMES, analysis.get("selected_block_atlas_rows", []))
    _write_csv_rows(paths["alignment_csv"], PHASE66_ALIGNMENT_FIELDNAMES, analysis.get("representation_rank_alignment_rows", []))
    _write_csv_rows(paths["failure_csv"], PHASE66_FAILURE_FIELDNAMES, analysis.get("failure_mode_summary_rows", []))
    saved = dict(analysis)
    saved["phase66_status"] = dict(analysis.get("diagnostic_gate", {})).get("phase66_status", PHASE66_STATUS_INSUFFICIENT)
    paths["audit_json"].write_text(
        json.dumps(_json_ready(saved), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["audit_md"].write_text(_phase66_markdown(analysis), encoding="utf-8")
    return paths
```

Then implement `run_phase66_reward_label_representation_audit` with these exact inputs and outputs:

```python
def run_phase66_reward_label_representation_audit(
    phase63_comparison_json: Path | str,
    phase63_rollout_csv: Path | str,
    phase63_oracle_summary_csv: Path | str,
    phase64_failure_cases_csv: Path | str | None,
    phase64_feature_effective_rank_csv: Path | str | None,
    phase65_comparison_json: Path | str,
    phase65_rollout_csv: Path | str,
    phase65_pairwise_delta_csv: Path | str,
    phase10_reward_readiness_json: Path | str | None = None,
) -> dict[str, object]:
    phase63_comparison = _load_json_object(phase63_comparison_json, "Phase 63 comparison JSON")
    phase65_comparison = _load_json_object(phase65_comparison_json, "Phase 65 comparison JSON")
    suitability_context = _load_json_object(phase10_reward_readiness_json, "Phase 10 reward readiness JSON")
    contract = phase63_comparison.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Phase 63 comparison JSON is missing contract metadata")
    phase63_rows = _load_csv_rows(phase63_rollout_csv, "Phase 63 rollout CSV")
    oracle_rows = _load_csv_rows(phase63_oracle_summary_csv, "Phase 63 oracle summary CSV")
    phase64_failure_rows = _load_csv_rows(phase64_failure_cases_csv, "Phase 64 failure cases CSV")
    phase64_rank_rows = _load_csv_rows(phase64_feature_effective_rank_csv, "Phase 64 feature effective rank CSV")
    phase65_rows = _load_csv_rows(phase65_rollout_csv, "Phase 65 rollout CSV")
    phase65_pairwise_rows = _load_csv_rows(phase65_pairwise_delta_csv, "Phase 65 pairwise delta CSV")
    tiled_inputs = _load_phase66_tiled_inputs(contract)
    eval_max_steps = int(contract.get("eval_max_steps", 8))
    eval_tiled_inputs = {
        key: tiled
        for key, tiled in tiled_inputs.items()
        if key[1] in set(_contract_string_list(contract, "eval_tile_ids"))
    }
    atlas_rows = build_phase66_selected_block_atlas(
        phase63_rollout_rows=phase63_rows,
        phase65_rollout_rows=phase65_rows,
        oracle_rows=oracle_rows,
        tiled_inputs=eval_tiled_inputs,
    )
    alignment_rows = build_phase66_representation_rank_alignment(
        tiled_inputs=eval_tiled_inputs,
        eval_max_steps=eval_max_steps,
    )
    failure_rows = build_phase66_failure_mode_summary(
        atlas_rows,
        alignment_rows,
        phase65_pairwise_rows,
    )
    coverage_issues: list[object] = []
    if dict(phase65_comparison).get("phase65_status") == "insufficient":
        coverage_issues.append("Phase 65 status is insufficient")
    gate = build_phase66_diagnostic_gate(
        coverage_issues,
        alignment_rows,
        failure_rows,
        suitability_context,
    )
    component_rows: list[dict[str, object]] = []
    for key, tiled in sorted(eval_tiled_inputs.items()):
        for row in build_phase66_block_reward_table(tiled):
            component_rows.append(
                {
                    **row,
                    "source": "raw_tile",
                    "seed": "",
                    "action_group": "all_blocks",
                }
            )
    return {
        "phase": "phase66_reward_label_representation_audit",
        "phase63_comparison_json": str(Path(phase63_comparison_json)),
        "phase63_rollout_csv": str(Path(phase63_rollout_csv)),
        "phase63_oracle_summary_csv": str(Path(phase63_oracle_summary_csv)),
        "phase64_failure_cases_csv": "" if phase64_failure_cases_csv is None else str(Path(phase64_failure_cases_csv)),
        "phase64_feature_effective_rank_csv": "" if phase64_feature_effective_rank_csv is None else str(Path(phase64_feature_effective_rank_csv)),
        "phase65_comparison_json": str(Path(phase65_comparison_json)),
        "phase65_rollout_csv": str(Path(phase65_rollout_csv)),
        "phase65_pairwise_delta_csv": str(Path(phase65_pairwise_delta_csv)),
        "contract": dict(contract),
        "phase64_failure_case_rows_loaded": len(phase64_failure_rows),
        "phase64_feature_effective_rank_rows_loaded": len(phase64_rank_rows),
        "reward_component_rows": component_rows,
        "selected_block_atlas_rows": atlas_rows,
        "representation_rank_alignment_rows": alignment_rows,
        "failure_mode_summary_rows": failure_rows,
        "diagnostic_gate": gate,
        "claim_boundary": PHASE66_CLAIM_BOUNDARY,
    }
```

- [ ] **Step 4: Add the CLI runner**

Create `experiments/phase66_reward_label_representation_audit/run_phase66_reward_label_representation_audit.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase66_reward_label_representation_audit import (
    run_phase66_reward_label_representation_audit,
    write_phase66_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = run_phase66_reward_label_representation_audit(
            phase63_comparison_json=args.phase63_comparison_json,
            phase63_rollout_csv=args.phase63_rollout_csv,
            phase63_oracle_summary_csv=args.phase63_oracle_summary_csv,
            phase64_failure_cases_csv=args.phase64_failure_cases_csv,
            phase64_feature_effective_rank_csv=args.phase64_feature_effective_rank_csv,
            phase65_comparison_json=args.phase65_comparison_json,
            phase65_rollout_csv=args.phase65_rollout_csv,
            phase65_pairwise_delta_csv=args.phase65_pairwise_delta_csv,
            phase10_reward_readiness_json=args.phase10_reward_readiness_json,
        )
        paths = write_phase66_artifacts(analysis, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    gate = analysis["diagnostic_gate"]
    print(f"Phase 66 status: {gate['phase66_status']}")
    print(f"Audit JSON: {paths['audit_json']}")
    print(f"Atlas CSV: {paths['atlas_csv']}")
    print(f"Alignment CSV: {paths['alignment_csv']}")
    print(f"Failure CSV: {paths['failure_csv']}")
    print(f"Audit Markdown: {paths['audit_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 66 reward-label representation audit."
    )
    parser.add_argument("--phase63-comparison-json", type=Path, required=True)
    parser.add_argument("--phase63-rollout-csv", type=Path, required=True)
    parser.add_argument("--phase63-oracle-summary-csv", type=Path, required=True)
    parser.add_argument("--phase64-failure-cases-csv", type=Path, default=None)
    parser.add_argument("--phase64-feature-effective-rank-csv", type=Path, default=None)
    parser.add_argument("--phase65-comparison-json", type=Path, required=True)
    parser.add_argument("--phase65-rollout-csv", type=Path, required=True)
    parser.add_argument("--phase65-pairwise-delta-csv", type=Path, required=True)
    parser.add_argument("--phase10-reward-readiness-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run all Phase 66 tests**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase66_reward_label_representation_audit.py -q --basetemp=.pytest_tmp_phase66_task5_green -p no:cacheprovider
```

Expected: all Phase 66 tests pass.

- [ ] **Step 6: Commit Task 5**

Run:

```powershell
git add src\paper11_geofm\phase66_reward_label_representation_audit.py experiments\phase66_reward_label_representation_audit\run_phase66_reward_label_representation_audit.py tests\test_phase66_reward_label_representation_audit.py
git commit -m "feat: add Phase 66 audit runner and artifacts"
```

Expected: commit succeeds.

---

### Task 6: Real Phase 66 Run And Result Note

**Files:**
- Create: `paper/phase28_results/32_phase66_reward_label_representation_audit.md`
- Generated ignored outputs under: `experiments/phase66_reward_label_representation_audit/outputs/phase52_full5_seed3/`

- [ ] **Step 1: Run the full Phase 66 audit**

Run from repository root:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase66_reward_label_representation_audit\run_phase66_reward_label_representation_audit.py --phase63-comparison-json experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_set_policy_comparison.json --phase63-rollout-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_bc_rollout_summary.csv --phase63-oracle-summary-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_oracle_summary.csv --phase64-failure-cases-csv experiments\phase64_set_policy_error_diagnosis\outputs\phase52_full5_seed3\phase64_failure_cases.csv --phase64-feature-effective-rank-csv experiments\phase64_set_policy_error_diagnosis\outputs\phase52_full5_seed3\phase64_feature_effective_rank.csv --phase65-comparison-json experiments\phase65_standardized_set_policy_bc_rerun\outputs\phase52_full5_seed3\phase65_set_policy_comparison.json --phase65-rollout-csv experiments\phase65_standardized_set_policy_bc_rerun\outputs\phase52_full5_seed3\phase65_bc_rollout_summary.csv --phase65-pairwise-delta-csv experiments\phase65_standardized_set_policy_bc_rerun\outputs\phase52_full5_seed3\phase65_standardization_pairwise_delta.csv --phase10-reward-readiness-json experiments\phase11_bishan_dltb_real\outputs\phase10_real\phase10_reward_readiness_gate.json --output-dir experiments\phase66_reward_label_representation_audit\outputs\phase52_full5_seed3
```

Expected: exit code `0`, console prints `Phase 66 status:` plus JSON/CSV/Markdown artifact paths and the claim boundary.

- [ ] **Step 2: Inspect generated audit JSON**

Run:

```powershell
Get-Content -Raw experiments\phase66_reward_label_representation_audit\outputs\phase52_full5_seed3\phase66_reward_label_representation_audit.json
```

Expected: JSON contains `phase66_status`, `diagnostic_gate`, `selected_block_atlas_rows`, `representation_rank_alignment_rows`, `failure_mode_summary_rows`, and `claim_boundary`.

- [ ] **Step 3: Create the tracked result note from generated Markdown**

Run:

```powershell
Copy-Item -LiteralPath experiments\phase66_reward_label_representation_audit\outputs\phase52_full5_seed3\phase66_reward_label_representation_audit.md -Destination paper\phase28_results\32_phase66_reward_label_representation_audit.md
Add-Content -LiteralPath paper\phase28_results\32_phase66_reward_label_representation_audit.md -Value @'

## Reproduction

Run Phase 66 from the repository root after Phase 63, Phase 64, and Phase 65 full-run artifacts exist:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase66_reward_label_representation_audit\run_phase66_reward_label_representation_audit.py --phase63-comparison-json experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_set_policy_comparison.json --phase63-rollout-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_bc_rollout_summary.csv --phase63-oracle-summary-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_oracle_summary.csv --phase64-failure-cases-csv experiments\phase64_set_policy_error_diagnosis\outputs\phase52_full5_seed3\phase64_failure_cases.csv --phase64-feature-effective-rank-csv experiments\phase64_set_policy_error_diagnosis\outputs\phase52_full5_seed3\phase64_feature_effective_rank.csv --phase65-comparison-json experiments\phase65_standardized_set_policy_bc_rerun\outputs\phase52_full5_seed3\phase65_set_policy_comparison.json --phase65-rollout-csv experiments\phase65_standardized_set_policy_bc_rerun\outputs\phase52_full5_seed3\phase65_bc_rollout_summary.csv --phase65-pairwise-delta-csv experiments\phase65_standardized_set_policy_bc_rerun\outputs\phase52_full5_seed3\phase65_standardization_pairwise_delta.csv --phase10-reward-readiness-json experiments\phase11_bishan_dltb_real\outputs\phase10_real\phase10_reward_readiness_gate.json --output-dir experiments\phase66_reward_label_representation_audit\outputs\phase52_full5_seed3
```

## Boundary

No formal manuscript files were changed in this phase.
'@
```

- [ ] **Step 4: Verify result note has no unresolved planning markers**

Run:

```powershell
rg -n "<[^>]+>|copy status|temporary marker|replace before run" paper\phase28_results\32_phase66_reward_label_representation_audit.md
```

Expected: no output.

- [ ] **Step 5: Commit Task 6**

Run:

```powershell
git add paper\phase28_results\32_phase66_reward_label_representation_audit.md
git commit -m "docs: record Phase 66 reward-label audit results"
```

Expected: commit succeeds. Generated `experiments/**/outputs/**` files remain ignored unless repository policy changes.

---

### Task 7: Regression Verification And Final Boundary Checks

**Files:**
- No new files unless a failing verification requires a targeted fix.

- [ ] **Step 1: Run targeted Phase 66/65/64/63 tests**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase66_reward_label_representation_audit.py tests\test_phase65_standardized_set_policy_bc_rerun.py tests\test_phase64_set_policy_error_diagnosis.py tests\test_phase63_set_policy_oracle_pretraining.py -q --basetemp=.pytest_tmp_phase66_final -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 2: Run smoke check**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe scripts\smoke_check.py
```

Expected: `Paper11 smoke check passed`.

- [ ] **Step 3: Check whitespace**

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

- [ ] **Step 6: Push completed Phase 66 work**

Run:

```powershell
git push
```

Expected: `main -> main` push succeeds.

---

## Self-Review Checklist

- Spec coverage:
  - Reward-component attribution is covered by Tasks 1 and 5 through component rows for loaded tile matrices.
  - Selected-block atlas is covered by Task 2.
  - Representation-rank alignment is covered by Task 3.
  - Failure-mode classifier and diagnostic gate are covered by Task 4.
  - JSON/CSV/Markdown artifacts, CLI, real run, result note, and final checks are covered by Tasks 5-7.
- Claim boundary:
  - All module, writer, runner, and result-note paths use Phase 66 read-only language.
  - The plan never modifies `paper/submission/final/*`.
- Verification boundary:
  - The final command set includes Phase 66 targeted tests, Phase 65/64/63 regressions, smoke check, `git diff --check`, formal绋?untouched check, and push.
