# Phase 64 Set-Policy Error Diagnosis and Standardization Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a read-only Phase 64 diagnostic layer that explains Phase 63 set-policy errors and decides whether a train-tile-fitted standardization rerun is the next justified experiment.

**Architecture:** Add one focused Phase 64 module that reads Phase 63 artifacts, computes convergence, selected-block overlap, oracle-rank gaps, feature-scale/effective-rank diagnostics, failure cases, and a standardization gate. Add a thin CLI runner and paper-facing evidence note after the real diagnostic run. Keep Phase 63 training behavior and formal manuscript files untouched.

**Tech Stack:** Python standard library, NumPy, existing `paper11_geofm.tiled_inputs`, existing `paper11_geofm.planning_reward`, existing Phase 63 artifact formats, CSV/JSON/Markdown writers, pytest.

---

## File Structure

- Create `src/paper11_geofm/phase64_set_policy_error_diagnosis.py`.
  Owns constants, CSV/JSON loading, semicolon parsing, convergence summaries, overlap metrics, oracle-rank gap metrics, feature-scale/effective-rank diagnostics, failure case extraction, standardization gate logic, artifact writers, and the run wrapper.
- Create `experiments/phase64_set_policy_error_diagnosis/run_phase64_set_policy_error_diagnosis.py`.
  Exposes a CLI that consumes Phase 63 artifacts and writes Phase 64 artifacts.
- Create `tests/test_phase64_set_policy_error_diagnosis.py`.
  Covers parsing, convergence, overlap, oracle-rank gaps, feature scale/effective rank, gate statuses, writer outputs, and CLI behavior on synthetic fixtures.
- Create `paper/phase28_results/30_phase64_set_policy_error_diagnosis.md` after the real run.
  Records Phase 64 diagnostic evidence and the gate decision only.
- Modify `paper/phase28_results/README.md` after the real run.
  Adds the Phase 64 evidence entry and reproduction command.
- Modify `docs/superpowers/phase33_current_progress_handoff.md` after the real run.
  Records the Phase 64 status, generated artifacts, and next entry point.

Do not modify:

- `paper/submission/final/Paper11_formal_conclusion_manuscript.md`
- `paper/submission/final/Paper11_formal_conclusion_manuscript.tex`
- `paper/submission/final/Paper11_formal_conclusion_manuscript.pdf`
- `paper/submission/final/Paper11_submission_metadata_template.md`

---

## Phase 64 Contract

Use the existing Phase 63 full run as the first real input:

- Phase 63 comparison JSON: `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_set_policy_comparison.json`
- Phase 63 rollout CSV: `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_bc_rollout_summary.csv`
- Phase 63 history CSV: `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_bc_training_history.csv`
- Phase 63 oracle summary CSV: `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_oracle_summary.csv`
- Output directory: `experiments/phase64_set_policy_error_diagnosis/outputs/phase52_full5_seed3`

Status values:

- `standardization_route_supported`
- `bc_training_capacity_limited`
- `geofm_features_not_helpful_under_set_policy`
- `diagnostic_inconclusive`

Gate thresholds for the first implementation:

- weak top-1 convergence: mean best top-1 accuracy `< 0.25`;
- weak top-k convergence: mean best top-k hit rate `< 0.50`;
- scale flag: standard deviation ratio `>= 100.0` or mean-scale ratio `>= 10.0`;
- shift flag: max train-to-eval absolute z-shift `>= 3.0`;
- rank flag: effective-rank fraction `<= 0.30` or first-component variance share `>= 0.80`;
- D4 underperformance: Phase 63 `d4_b0_delta_summary.mean_delta <= 0.0` or `d4_d6_delta_summary.mean_delta <= 0.0`.

Claim boundary constant:

```python
PHASE64_CLAIM_BOUNDARY = (
    "Phase 64 is a read-only set-policy error-diagnosis and standardization-gate "
    "phase. It uses Phase 63 base-reward artifacts to diagnose behavior-cloned "
    "set-policy errors and decide whether a train-tile-fitted standardization "
    "rerun is justified. It does not enable suitability reward, does not test "
    "B2/B3, does not test transfer, does not prove GeoFM advantage or PCA "
    "optimality, and does not justify formal submission-level claims."
)
```

---

### Task 1: Parsing and Convergence Diagnostics

**Files:**
- Create: `tests/test_phase64_set_policy_error_diagnosis.py`
- Create: `src/paper11_geofm/phase64_set_policy_error_diagnosis.py`

- [ ] **Step 1: Write failing tests for semicolon parsing and convergence summaries**

Create `tests/test_phase64_set_policy_error_diagnosis.py` with these helpers and tests:

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


def _history_row(
    variant_id="B0",
    seed=0,
    epoch=1,
    loss=1.0,
    top1=0.0,
    topk=0.0,
    train_tile_id="tile_train",
):
    return {
        "variant_id": variant_id,
        "train_tile_id": train_tile_id,
        "seed": seed,
        "epoch": epoch,
        "loss": loss,
        "top1_accuracy": top1,
        "topk_hit_rate": topk,
        "learning_rate": 0.001,
        "hidden_dim": 64,
        "claim_boundary": "phase63",
    }


def test_phase64_splits_semicolon_values_and_summarizes_convergence():
    from paper11_geofm.phase64_set_policy_error_diagnosis import (
        _split_semicolon_values,
        build_phase64_convergence_summary,
    )

    assert _split_semicolon_values(" b2 ; b1;;b3 ") == ["b2", "b1", "b3"]
    assert _split_semicolon_values("") == []

    history_rows = [
        _history_row("B0", 0, 1, loss=4.0, top1=0.0, topk=0.0),
        _history_row("B0", 0, 2, loss=2.0, top1=0.25, topk=0.50),
        _history_row("B0", 0, 3, loss=2.5, top1=0.20, topk=0.75),
        _history_row("D4P8", 1, 1, loss=3.0, top1=0.10, topk=0.25),
        _history_row("D4P8", 1, 2, loss=1.5, top1=0.40, topk=0.50),
    ]

    summary = build_phase64_convergence_summary(history_rows)

    assert len(summary) == 2
    b0 = summary[0]
    assert b0["variant_id"] == "B0"
    assert b0["seed"] == 0
    assert b0["first_epoch"] == 1
    assert b0["final_epoch"] == 3
    assert b0["best_epoch"] == 2
    assert b0["first_loss"] == 4.0
    assert b0["final_loss"] == 2.5
    assert b0["best_loss"] == 2.0
    assert b0["final_top1_accuracy"] == 0.2
    assert b0["best_top1_accuracy"] == 0.25
    assert b0["final_topk_hit_rate"] == 0.75
    assert b0["best_topk_hit_rate"] == 0.75
    assert b0["loss_delta"] == -1.5
```

- [ ] **Step 2: Run the new test to verify RED**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase64_set_policy_error_diagnosis.py -q --basetemp=.pytest_tmp_phase64_task1_red -p no:cacheprovider
```

Expected: fails because `paper11_geofm.phase64_set_policy_error_diagnosis` does not exist.

- [ ] **Step 3: Add the Phase 64 module skeleton and convergence code**

Create `src/paper11_geofm/phase64_set_policy_error_diagnosis.py`:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

from .planning_reward import compute_base_planning_reward_from_matrix_row
from .tiled_inputs import load_tiled_variant_input


PHASE64_CLAIM_BOUNDARY = (
    "Phase 64 is a read-only set-policy error-diagnosis and standardization-gate "
    "phase. It uses Phase 63 base-reward artifacts to diagnose behavior-cloned "
    "set-policy errors and decide whether a train-tile-fitted standardization "
    "rerun is justified. It does not enable suitability reward, does not test "
    "B2/B3, does not test transfer, does not prove GeoFM advantage or PCA "
    "optimality, and does not justify formal submission-level claims."
)

PHASE64_STATUS_STANDARDIZATION = "standardization_route_supported"
PHASE64_STATUS_CAPACITY = "bc_training_capacity_limited"
PHASE64_STATUS_NOT_HELPFUL = "geofm_features_not_helpful_under_set_policy"
PHASE64_STATUS_INCONCLUSIVE = "diagnostic_inconclusive"

PHASE64_WEAK_TOP1_THRESHOLD = 0.25
PHASE64_WEAK_TOPK_THRESHOLD = 0.50
PHASE64_STD_RATIO_THRESHOLD = 100.0
PHASE64_MEAN_SCALE_RATIO_THRESHOLD = 10.0
PHASE64_Z_SHIFT_THRESHOLD = 3.0
PHASE64_EFFECTIVE_RANK_FRACTION_THRESHOLD = 0.30
PHASE64_PC1_SHARE_THRESHOLD = 0.80

PHASE64_CONVERGENCE_FIELDNAMES = [
    "variant_id",
    "train_tile_id",
    "seed",
    "first_epoch",
    "final_epoch",
    "best_epoch",
    "first_loss",
    "final_loss",
    "best_loss",
    "final_top1_accuracy",
    "best_top1_accuracy",
    "final_topk_hit_rate",
    "best_topk_hit_rate",
    "loss_delta",
    "claim_boundary",
]


def _split_semicolon_values(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _round_float(value: object, digits: int = 10) -> float:
    rounded = round(float(value), digits)
    compact = round(rounded, 6)
    if abs(rounded - compact) < 5.0e-8:
        return compact
    return rounded


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def _safe_int(value: object, default: int = 0) -> int:
    if value is None or str(value).strip() == "":
        return int(default)
    return int(float(value))


def _load_csv_rows(path: Path | str, label: str) -> list[dict[str, object]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {label}: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_json_object(path: Path | str, label: str) -> dict[str, object]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {label}: {json_path}")
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return loaded


def _group_key(row: Mapping[str, object], fields: Sequence[str]) -> tuple[object, ...]:
    return tuple(row.get(field, "") for field in fields)


def build_phase64_convergence_summary(
    history_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(history_rows_or_csv, (str, Path)):
        history_rows = _load_csv_rows(history_rows_or_csv, "Phase 63 BC history CSV")
    else:
        history_rows = [dict(row) for row in history_rows_or_csv]

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in history_rows:
        key = _group_key(row, ("variant_id", "train_tile_id", "seed"))
        grouped.setdefault(key, []).append(dict(row))

    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: _safe_int(row.get("epoch")))
        first = rows[0]
        final = rows[-1]
        best = min(rows, key=lambda row: _safe_float(row.get("loss"), math.inf))
        first_loss = _safe_float(first.get("loss"))
        final_loss = _safe_float(final.get("loss"))
        output.append(
            {
                "variant_id": str(key[0]),
                "train_tile_id": str(key[1]),
                "seed": _safe_int(key[2]),
                "first_epoch": _safe_int(first.get("epoch")),
                "final_epoch": _safe_int(final.get("epoch")),
                "best_epoch": _safe_int(best.get("epoch")),
                "first_loss": _round_float(first_loss),
                "final_loss": _round_float(final_loss),
                "best_loss": _round_float(best.get("loss")),
                "final_top1_accuracy": _round_float(final.get("top1_accuracy")),
                "best_top1_accuracy": _round_float(
                    max(_safe_float(row.get("top1_accuracy")) for row in rows)
                ),
                "final_topk_hit_rate": _round_float(final.get("topk_hit_rate")),
                "best_topk_hit_rate": _round_float(
                    max(_safe_float(row.get("topk_hit_rate")) for row in rows)
                ),
                "loss_delta": _round_float(final_loss - first_loss),
                "claim_boundary": PHASE64_CLAIM_BOUNDARY,
            }
        )
    return output
```

- [ ] **Step 4: Run the Task 1 tests to verify GREEN**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase64_set_policy_error_diagnosis.py -q --basetemp=.pytest_tmp_phase64_task1_green -p no:cacheprovider
```

Expected: the Task 1 test passes.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src\paper11_geofm\phase64_set_policy_error_diagnosis.py tests\test_phase64_set_policy_error_diagnosis.py
git commit -m "feat: add Phase 64 convergence diagnostics"
```

---

### Task 2: Selected-Block Overlap and Oracle-Rank Gap

**Files:**
- Modify: `tests/test_phase64_set_policy_error_diagnosis.py`
- Modify: `src/paper11_geofm/phase64_set_policy_error_diagnosis.py`

- [ ] **Step 1: Add failing tests for overlap and oracle-rank gaps**

Append these helpers and tests to `tests/test_phase64_set_policy_error_diagnosis.py`:

```python
def _rollout_row(
    variant_id="B0",
    eval_tile_id="tile_eval",
    seed=0,
    selected="b1;b3",
    reward=1.0,
    oracle=1.5,
    gap=0.5,
    gap_fraction=0.3333333333,
    eval_max_steps=3,
):
    return {
        "row_type": "bc_greedy_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": eval_tile_id,
        "eval_tile_rank": 1,
        "seed": seed,
        "phase63_seed_rank": 1,
        "eval_max_steps": eval_max_steps,
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
        "claim_boundary": "phase63",
    }


def _oracle_row(
    variant_id="B0",
    tile_id="tile_eval",
    seed=0,
    selected="b1;b2;b4",
    action_indices="0;1;3",
    eval_max_steps=3,
    oracle=1.5,
):
    return {
        "variant_id": variant_id,
        "tile_role": "eval",
        "tile_id": tile_id,
        "seed": seed,
        "eval_max_steps": eval_max_steps,
        "n_blocks": 4,
        "n_features": 9,
        "episode_steps": 3,
        "terminated": False,
        "total_oracle_reward": oracle,
        "top_k_reward_ceiling": oracle,
        "selected_block_ids": selected,
        "action_indices": action_indices,
        "claim_boundary": "phase63",
    }


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


def _tiled_input(
    block_ids=("b1", "b2", "b3", "b4"),
    scores=(0.9, 0.8, 0.2, 0.7),
    variant_id="B0",
    tile_id="tile_eval",
):
    from paper11_geofm.tiled_inputs import TiledVariantInput

    columns = _required_feature_columns()
    matrix = np.zeros((len(block_ids), len(columns)), dtype=np.float32)
    score_index = columns.index("explicit_feature_16")
    for row_index, score in enumerate(scores):
        matrix[row_index, score_index] = float(score)
    return TiledVariantInput(
        tile_id=tile_id,
        variant_id=variant_id,
        block_ids=tuple(block_ids),
        feature_columns=columns,
        state_matrix=matrix,
        reward_mode="base_planning_reward",
        state_groups=("explicit_planning_features",),
        source_table=Path(f"variant_{variant_id}_features.csv"),
        tile_index_csv=Path("tiles.csv"),
    )


def test_phase64_rollout_overlap_tracks_prefix_jaccard_and_missed_oracle():
    from paper11_geofm.phase64_set_policy_error_diagnosis import (
        build_phase64_rollout_overlap,
    )

    rows = build_phase64_rollout_overlap(
        [_rollout_row(selected="b1;b3;b3")],
        [_oracle_row(selected="b1;b2;b4")],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["selected_overlap_count"] == 1
    assert row["selected_overlap_fraction"] == 0.3333333333
    assert row["prefix_overlap_count"] == 1
    assert row["jaccard_similarity"] == 0.25
    assert row["duplicate_selection_count"] == 1
    assert row["missed_oracle_block_ids"] == "b2;b4"
    assert row["extra_selected_block_ids"] == "b3"


def test_phase64_oracle_rank_gap_reports_missed_blocks_and_rank_losses():
    from paper11_geofm.phase64_set_policy_error_diagnosis import (
        build_phase64_oracle_rank_gap,
    )

    tiled = _tiled_input()
    rows = build_phase64_oracle_rank_gap(
        [_rollout_row(selected="b1;b3", eval_max_steps=3)],
        {("B0", "tile_eval"): tiled},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["selected_rank_values"] == "1;4"
    assert row["missed_oracle_block_ids"] == "b2;b4"
    assert row["worst_selected_rank"] == 4
    assert row["selected_outside_top_eval_max_steps"] == 1
    assert row["selected_outside_top16"] == 0
    assert row["selected_outside_top32"] == 0
    assert row["reward_loss_from_missed_oracle"] > 0.0
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase64_set_policy_error_diagnosis.py -q --basetemp=.pytest_tmp_phase64_task2_red -p no:cacheprovider
```

Expected: fails because overlap and rank-gap functions are missing.

- [ ] **Step 3: Add overlap and oracle-rank gap code**

Append to `src/paper11_geofm/phase64_set_policy_error_diagnosis.py`:

```python
PHASE64_OVERLAP_FIELDNAMES = [
    "variant_id",
    "train_tile_id",
    "eval_tile_id",
    "seed",
    "eval_max_steps",
    "selected_overlap_count",
    "selected_overlap_fraction",
    "prefix_overlap_count",
    "jaccard_similarity",
    "duplicate_selection_count",
    "invalid_action_count",
    "bc_reward",
    "oracle_total_reward",
    "oracle_gap",
    "oracle_gap_fraction",
    "selected_block_ids",
    "oracle_block_ids",
    "missed_oracle_block_ids",
    "extra_selected_block_ids",
    "claim_boundary",
]

PHASE64_ORACLE_RANK_FIELDNAMES = [
    "variant_id",
    "eval_tile_id",
    "seed",
    "eval_max_steps",
    "selected_rank_values",
    "selected_reward_values",
    "missed_oracle_block_ids",
    "missed_oracle_rewards",
    "reward_loss_from_missed_oracle",
    "worst_selected_rank",
    "selected_outside_top_eval_max_steps",
    "selected_outside_top16",
    "selected_outside_top32",
    "claim_boundary",
]


def _phase64_row_key(row: Mapping[str, object]) -> tuple[str, str, int]:
    return (
        str(row.get("variant_id", "")),
        str(row.get("eval_tile_id", row.get("tile_id", ""))),
        _safe_int(row.get("seed")),
    )


def build_phase64_rollout_overlap(
    rollout_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    oracle_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rollout_rows = (
        _load_csv_rows(rollout_rows_or_csv, "Phase 63 rollout CSV")
        if isinstance(rollout_rows_or_csv, (str, Path))
        else [dict(row) for row in rollout_rows_or_csv]
    )
    oracle_rows = (
        _load_csv_rows(oracle_rows_or_csv, "Phase 63 oracle summary CSV")
        if isinstance(oracle_rows_or_csv, (str, Path))
        else [dict(row) for row in oracle_rows_or_csv]
    )
    oracle_index = {
        (str(row.get("variant_id", "")), str(row.get("tile_id", "")), _safe_int(row.get("seed"))): row
        for row in oracle_rows
    }

    output: list[dict[str, object]] = []
    for rollout in rollout_rows:
        key = _phase64_row_key(rollout)
        oracle = oracle_index.get(key)
        if oracle is None:
            oracle_blocks: list[str] = []
        else:
            oracle_blocks = _split_semicolon_values(oracle.get("selected_block_ids"))
        selected_blocks = _split_semicolon_values(rollout.get("selected_block_ids"))
        selected_unique = set(selected_blocks)
        oracle_set = set(oracle_blocks)
        overlap = selected_unique.intersection(oracle_set)
        union = selected_unique.union(oracle_set)
        prefix_overlap = sum(
            1
            for left, right in zip(selected_blocks, oracle_blocks)
            if left == right
        )
        duplicate_count = len(selected_blocks) - len(selected_unique)
        missed = [block_id for block_id in oracle_blocks if block_id not in selected_unique]
        extra = [block_id for block_id in selected_blocks if block_id not in oracle_set]
        denom = max(len(oracle_blocks), 1)
        output.append(
            {
                "variant_id": key[0],
                "train_tile_id": str(rollout.get("train_tile_id", "")),
                "eval_tile_id": key[1],
                "seed": key[2],
                "eval_max_steps": _safe_int(rollout.get("eval_max_steps")),
                "selected_overlap_count": len(overlap),
                "selected_overlap_fraction": _round_float(len(overlap) / denom),
                "prefix_overlap_count": int(prefix_overlap),
                "jaccard_similarity": _round_float(len(overlap) / max(len(union), 1)),
                "duplicate_selection_count": int(duplicate_count),
                "invalid_action_count": _safe_int(rollout.get("invalid_action_count")),
                "bc_reward": _round_float(rollout.get("total_contract_reward")),
                "oracle_total_reward": _round_float(rollout.get("oracle_total_reward")),
                "oracle_gap": _round_float(rollout.get("oracle_gap")),
                "oracle_gap_fraction": _round_float(rollout.get("oracle_gap_fraction")),
                "selected_block_ids": ";".join(selected_blocks),
                "oracle_block_ids": ";".join(oracle_blocks),
                "missed_oracle_block_ids": ";".join(missed),
                "extra_selected_block_ids": ";".join(extra),
                "claim_boundary": PHASE64_CLAIM_BOUNDARY,
            }
        )
    return output


def _block_reward_ranking(tiled_input) -> list[dict[str, object]]:
    rewards = [
        compute_base_planning_reward_from_matrix_row(
            tiled_input.feature_columns,
            tiled_input.state_matrix[index],
        )
        for index in range(len(tiled_input.block_ids))
    ]
    ranked_indices = sorted(
        range(len(tiled_input.block_ids)),
        key=lambda index: (-rewards[index], str(tiled_input.block_ids[index]), index),
    )
    output: list[dict[str, object]] = []
    for rank, index in enumerate(ranked_indices, start=1):
        output.append(
            {
                "rank": rank,
                "action_index": int(index),
                "block_id": str(tiled_input.block_ids[index]),
                "reward": _round_float(rewards[index]),
            }
        )
    return output


def build_phase64_oracle_rank_gap(
    rollout_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    tiled_inputs: Mapping[tuple[str, str], object],
) -> list[dict[str, object]]:
    rollout_rows = (
        _load_csv_rows(rollout_rows_or_csv, "Phase 63 rollout CSV")
        if isinstance(rollout_rows_or_csv, (str, Path))
        else [dict(row) for row in rollout_rows_or_csv]
    )
    output: list[dict[str, object]] = []
    ranking_cache: dict[tuple[str, str], list[dict[str, object]]] = {}
    for rollout in rollout_rows:
        variant_id, eval_tile_id, seed = _phase64_row_key(rollout)
        input_key = (variant_id, eval_tile_id)
        if input_key not in ranking_cache:
            ranking_cache[input_key] = _block_reward_ranking(tiled_inputs[input_key])
        ranking = ranking_cache[input_key]
        by_block = {str(row["block_id"]): row for row in ranking}
        eval_max_steps = _safe_int(rollout.get("eval_max_steps"))
        oracle_top = [str(row["block_id"]) for row in ranking[:eval_max_steps]]
        selected = _split_semicolon_values(rollout.get("selected_block_ids"))
        selected_set = set(selected)
        selected_rows = [by_block[block_id] for block_id in selected if block_id in by_block]
        selected_ranks = [_safe_int(row["rank"]) for row in selected_rows]
        selected_rewards = [_safe_float(row["reward"]) for row in selected_rows]
        missed = [block_id for block_id in oracle_top if block_id not in selected_set]
        missed_rewards = [_safe_float(by_block[block_id]["reward"]) for block_id in missed]
        selected_non_oracle = [
            row
            for row in selected_rows
            if _safe_int(row["rank"]) > eval_max_steps
        ]
        selected_non_oracle_rewards = [_safe_float(row["reward"]) for row in selected_non_oracle]
        reward_loss = sum(missed_rewards) - sum(selected_non_oracle_rewards[: len(missed_rewards)])
        output.append(
            {
                "variant_id": variant_id,
                "eval_tile_id": eval_tile_id,
                "seed": seed,
                "eval_max_steps": eval_max_steps,
                "selected_rank_values": ";".join(str(rank) for rank in selected_ranks),
                "selected_reward_values": ";".join(str(_round_float(value)) for value in selected_rewards),
                "missed_oracle_block_ids": ";".join(missed),
                "missed_oracle_rewards": ";".join(str(_round_float(value)) for value in missed_rewards),
                "reward_loss_from_missed_oracle": _round_float(max(reward_loss, 0.0)),
                "worst_selected_rank": max(selected_ranks) if selected_ranks else 0,
                "selected_outside_top_eval_max_steps": sum(1 for rank in selected_ranks if rank > eval_max_steps),
                "selected_outside_top16": sum(1 for rank in selected_ranks if rank > 16),
                "selected_outside_top32": sum(1 for rank in selected_ranks if rank > 32),
                "claim_boundary": PHASE64_CLAIM_BOUNDARY,
            }
        )
    return output
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase64_set_policy_error_diagnosis.py -q --basetemp=.pytest_tmp_phase64_task2_green -p no:cacheprovider
```

Expected: all Task 1 and Task 2 tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src\paper11_geofm\phase64_set_policy_error_diagnosis.py tests\test_phase64_set_policy_error_diagnosis.py
git commit -m "feat: add Phase 64 rollout error diagnostics"
```

---

### Task 3: Feature Scale and Effective-Rank Diagnostics

**Files:**
- Modify: `tests/test_phase64_set_policy_error_diagnosis.py`
- Modify: `src/paper11_geofm/phase64_set_policy_error_diagnosis.py`

- [ ] **Step 1: Add failing tests for feature scale and effective rank**

Append these tests:

```python
def _matrix_tiled_input(matrix, variant_id="D4P8", tile_id="tile_train"):
    from paper11_geofm.tiled_inputs import TiledVariantInput

    feature_columns = tuple(f"feature_{index:02d}" for index in range(np.asarray(matrix).shape[1]))
    block_ids = tuple(f"b{index}" for index in range(np.asarray(matrix).shape[0]))
    return TiledVariantInput(
        tile_id=tile_id,
        variant_id=variant_id,
        block_ids=block_ids,
        feature_columns=feature_columns,
        state_matrix=np.asarray(matrix, dtype=np.float32),
        reward_mode="base_planning_reward",
        state_groups=("synthetic",),
        source_table=Path(f"variant_{variant_id}_features.csv"),
        tile_index_csv=Path("tiles.csv"),
    )


def test_phase64_feature_diagnostics_detect_scale_shift_and_low_rank():
    from paper11_geofm.phase64_set_policy_error_diagnosis import (
        build_phase64_feature_diagnostics,
    )

    train = _matrix_tiled_input(
        [
            [1.0, 10.0, 0.0],
            [2.0, 20.0, 0.0],
            [3.0, 30.0, 0.0],
            [4.0, 40.0, 0.0],
        ],
        variant_id="D4P8",
        tile_id="tile_train",
    )
    eval_tile = _matrix_tiled_input(
        [
            [11.0, 100.0, 0.0],
            [12.0, 120.0, 0.0],
        ],
        variant_id="D4P8",
        tile_id="tile_eval",
    )

    diagnostics = build_phase64_feature_diagnostics(
        [("train", train), ("eval", eval_tile)],
        {"D4P8": "tile_train"},
    )

    feature_rows = diagnostics["feature_scale_rows"]
    rank_rows = diagnostics["feature_effective_rank_rows"]
    assert len(feature_rows) == 6
    train_feature0 = [
        row for row in feature_rows
        if row["tile_role"] == "train" and row["feature_name"] == "feature_00"
    ][0]
    assert train_feature0["mean"] == 2.5
    assert train_feature0["zero_variance"] is False
    eval_feature0 = [
        row for row in feature_rows
        if row["tile_role"] == "eval" and row["feature_name"] == "feature_00"
    ][0]
    assert eval_feature0["eval_mean_z_shift"] > 3.0

    eval_rank = [row for row in rank_rows if row["tile_role"] == "eval"][0]
    assert eval_rank["zero_variance_feature_count"] == 1
    assert eval_rank["rank_flag"] is True
    assert eval_rank["shift_flag"] is True
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase64_set_policy_error_diagnosis.py -q --basetemp=.pytest_tmp_phase64_task3_red -p no:cacheprovider
```

Expected: fails because feature diagnostic functions are missing.

- [ ] **Step 3: Add feature-scale and effective-rank code**

Append to `src/paper11_geofm/phase64_set_policy_error_diagnosis.py`:

```python
PHASE64_FEATURE_SCALE_FIELDNAMES = [
    "variant_id",
    "tile_role",
    "tile_id",
    "feature_index",
    "feature_name",
    "mean",
    "std",
    "min",
    "max",
    "median",
    "p1",
    "p99",
    "train_mean",
    "train_std",
    "eval_mean_z_shift",
    "zero_variance",
    "claim_boundary",
]

PHASE64_EFFECTIVE_RANK_FIELDNAMES = [
    "variant_id",
    "tile_role",
    "tile_id",
    "n_blocks",
    "n_features",
    "zero_variance_feature_count",
    "std_ratio",
    "mean_scale_ratio",
    "max_train_eval_abs_z_shift",
    "effective_rank",
    "effective_rank_fraction",
    "pc1_variance_share",
    "pc3_variance_share",
    "scale_flag",
    "shift_flag",
    "rank_flag",
    "claim_boundary",
]


def _feature_distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": _round_float(np.mean(array)),
        "std": _round_float(np.std(array, ddof=0)),
        "min": _round_float(np.min(array)),
        "max": _round_float(np.max(array)),
        "median": _round_float(np.median(array)),
        "p1": _round_float(np.percentile(array, 1)),
        "p99": _round_float(np.percentile(array, 99)),
    }


def _effective_rank_stats(matrix: np.ndarray) -> dict[str, float]:
    values = np.asarray(matrix, dtype=float)
    if values.size == 0:
        return {
            "effective_rank": 0.0,
            "effective_rank_fraction": 0.0,
            "pc1_variance_share": 0.0,
            "pc3_variance_share": 0.0,
        }
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    positive = singular_values[singular_values > 1.0e-12]
    if positive.size == 0:
        effective_rank = 0.0
    else:
        probabilities = positive / np.sum(positive)
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        effective_rank = float(np.exp(entropy))
    squared = singular_values ** 2
    variance_total = float(np.sum(squared))
    pc1_share = float(squared[0] / variance_total) if variance_total > 0.0 else 0.0
    pc3_share = float(np.sum(squared[:3]) / variance_total) if variance_total > 0.0 else 0.0
    max_rank = max(1, min(values.shape))
    return {
        "effective_rank": _round_float(effective_rank),
        "effective_rank_fraction": _round_float(effective_rank / max_rank),
        "pc1_variance_share": _round_float(pc1_share),
        "pc3_variance_share": _round_float(pc3_share),
    }


def _train_feature_reference(
    tiled_inputs: Sequence[tuple[str, object]],
    train_tile_ids: Mapping[str, str],
) -> dict[str, dict[str, np.ndarray]]:
    reference: dict[str, dict[str, np.ndarray]] = {}
    for role, tiled_input in tiled_inputs:
        variant_id = str(tiled_input.variant_id)
        if str(role) != "train":
            continue
        if str(tiled_input.tile_id) != str(train_tile_ids.get(variant_id, tiled_input.tile_id)):
            continue
        matrix = np.asarray(tiled_input.state_matrix, dtype=float)
        reference[variant_id] = {
            "mean": np.mean(matrix, axis=0),
            "std": np.std(matrix, axis=0, ddof=0),
        }
    return reference


def build_phase64_feature_diagnostics(
    tiled_inputs: Sequence[tuple[str, object]],
    train_tile_ids: Mapping[str, str],
) -> dict[str, list[dict[str, object]]]:
    train_reference = _train_feature_reference(tiled_inputs, train_tile_ids)
    feature_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    for tile_role, tiled_input in tiled_inputs:
        variant_id = str(tiled_input.variant_id)
        matrix = np.asarray(tiled_input.state_matrix, dtype=float)
        train_stats = train_reference.get(variant_id)
        if train_stats is None:
            train_mean = np.mean(matrix, axis=0)
            train_std = np.std(matrix, axis=0, ddof=0)
        else:
            train_mean = train_stats["mean"]
            train_std = train_stats["std"]
        safe_train_std = np.where(train_std > 1.0e-12, train_std, np.nan)
        z_shift_values: list[float] = []
        std_values: list[float] = []
        mean_scale_values: list[float] = []
        zero_variance_count = 0
        for feature_index, feature_name in enumerate(tiled_input.feature_columns):
            values = matrix[:, feature_index]
            dist = _feature_distribution(values)
            feature_std = float(dist["std"])
            if feature_std <= 1.0e-12:
                zero_variance_count += 1
            else:
                std_values.append(feature_std)
            train_std_value = float(train_std[feature_index])
            z_shift = 0.0
            if train_std_value > 1.0e-12:
                z_shift = (float(dist["mean"]) - float(train_mean[feature_index])) / train_std_value
                z_shift_values.append(abs(z_shift))
            median_non_zero_std = max(float(np.nanmedian(safe_train_std)), 1.0e-12)
            mean_scale_values.append(abs(float(dist["mean"])) / median_non_zero_std)
            feature_rows.append(
                {
                    "variant_id": variant_id,
                    "tile_role": str(tile_role),
                    "tile_id": str(tiled_input.tile_id),
                    "feature_index": int(feature_index),
                    "feature_name": str(feature_name),
                    **dist,
                    "train_mean": _round_float(train_mean[feature_index]),
                    "train_std": _round_float(train_std_value),
                    "eval_mean_z_shift": _round_float(z_shift),
                    "zero_variance": bool(feature_std <= 1.0e-12),
                    "claim_boundary": PHASE64_CLAIM_BOUNDARY,
                }
            )
        non_zero_std = [value for value in std_values if value > 1.0e-12]
        std_ratio = max(non_zero_std) / min(non_zero_std) if non_zero_std else 0.0
        mean_scale_ratio = max(mean_scale_values) if mean_scale_values else 0.0
        max_z_shift = max(z_shift_values) if z_shift_values else 0.0
        rank_stats = _effective_rank_stats(matrix)
        scale_flag = (
            std_ratio >= PHASE64_STD_RATIO_THRESHOLD
            or mean_scale_ratio >= PHASE64_MEAN_SCALE_RATIO_THRESHOLD
        )
        shift_flag = max_z_shift >= PHASE64_Z_SHIFT_THRESHOLD
        rank_flag = (
            rank_stats["effective_rank_fraction"] <= PHASE64_EFFECTIVE_RANK_FRACTION_THRESHOLD
            or rank_stats["pc1_variance_share"] >= PHASE64_PC1_SHARE_THRESHOLD
        )
        rank_rows.append(
            {
                "variant_id": variant_id,
                "tile_role": str(tile_role),
                "tile_id": str(tiled_input.tile_id),
                "n_blocks": int(matrix.shape[0]),
                "n_features": int(matrix.shape[1]),
                "zero_variance_feature_count": int(zero_variance_count),
                "std_ratio": _round_float(std_ratio),
                "mean_scale_ratio": _round_float(mean_scale_ratio),
                "max_train_eval_abs_z_shift": _round_float(max_z_shift),
                **rank_stats,
                "scale_flag": bool(scale_flag),
                "shift_flag": bool(shift_flag),
                "rank_flag": bool(rank_flag),
                "claim_boundary": PHASE64_CLAIM_BOUNDARY,
            }
        )
    return {
        "feature_scale_rows": feature_rows,
        "feature_effective_rank_rows": rank_rows,
    }
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase64_set_policy_error_diagnosis.py -q --basetemp=.pytest_tmp_phase64_task3_green -p no:cacheprovider
```

Expected: all current Phase 64 tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add src\paper11_geofm\phase64_set_policy_error_diagnosis.py tests\test_phase64_set_policy_error_diagnosis.py
git commit -m "feat: add Phase 64 feature diagnostics"
```

---

### Task 4: Standardization Gate, Failure Cases, and Artifact Writer

**Files:**
- Modify: `tests/test_phase64_set_policy_error_diagnosis.py`
- Modify: `src/paper11_geofm/phase64_set_policy_error_diagnosis.py`

- [ ] **Step 1: Add failing tests for gate statuses and artifact writing**

Append these tests:

```python
def _comparison(
    d4_b0_mean=-0.1,
    d4_d6_mean=-0.05,
    missing=None,
    duplicate=None,
    unexpected=None,
):
    return {
        "coverage_issues": {
            "missing_rollout_rows": [] if missing is None else missing,
            "duplicate_rollout_rows": [] if duplicate is None else duplicate,
            "unexpected_rollout_rows": [] if unexpected is None else unexpected,
        },
        "d4_b0_delta_summary": {"mean_delta": d4_b0_mean, "positive_count": 0, "total_count": 4},
        "d4_d6_delta_summary": {"mean_delta": d4_d6_mean, "positive_count": 0, "total_count": 4},
        "oracle_gap_fraction_summary": {"mean_delta": 0.08, "positive_count": 4, "total_count": 4},
        "d4_b0_delta_rows": [
            {
                "left_variant_id": "D4P8",
                "right_variant_id": "B0",
                "eval_tile_id": "tile_eval",
                "seed": 0,
                "left_minus_right_reward": -0.2,
            }
        ],
        "d4_d6_delta_rows": [],
    }


def test_phase64_standardization_gate_reports_supported_capacity_not_helpful_and_inconclusive():
    from paper11_geofm.phase64_set_policy_error_diagnosis import (
        build_phase64_standardization_gate,
    )

    strong_convergence = [
        {"variant_id": "B0", "best_top1_accuracy": 0.6, "best_topk_hit_rate": 0.9},
        {"variant_id": "D4P8", "best_top1_accuracy": 0.5, "best_topk_hit_rate": 0.8},
    ]
    weak_convergence = [
        {"variant_id": "B0", "best_top1_accuracy": 0.1, "best_topk_hit_rate": 0.2},
        {"variant_id": "D4P8", "best_top1_accuracy": 0.1, "best_topk_hit_rate": 0.2},
    ]
    flagged_rank = [
        {
            "variant_id": "D4P8",
            "scale_flag": True,
            "shift_flag": False,
            "rank_flag": False,
            "tile_role": "eval",
        }
    ]
    clean_rank = [
        {
            "variant_id": "D4P8",
            "scale_flag": False,
            "shift_flag": False,
            "rank_flag": False,
            "tile_role": "eval",
        }
    ]

    supported = build_phase64_standardization_gate(
        _comparison(),
        strong_convergence,
        flagged_rank,
    )
    assert supported["phase64_status"] == "standardization_route_supported"
    assert supported["recommend_standardized_rerun"] is True

    capacity = build_phase64_standardization_gate(
        _comparison(),
        weak_convergence,
        flagged_rank,
    )
    assert capacity["phase64_status"] == "bc_training_capacity_limited"

    not_helpful = build_phase64_standardization_gate(
        _comparison(),
        strong_convergence,
        clean_rank,
    )
    assert not_helpful["phase64_status"] == "geofm_features_not_helpful_under_set_policy"

    inconclusive = build_phase64_standardization_gate(
        _comparison(missing=["B0:tile_eval:0"]),
        strong_convergence,
        flagged_rank,
    )
    assert inconclusive["phase64_status"] == "diagnostic_inconclusive"


def test_phase64_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase64_set_policy_error_diagnosis import (
        build_phase64_failure_cases,
        build_phase64_standardization_gate,
        write_phase64_artifacts,
    )

    convergence = [
        {
            "variant_id": "D4P8",
            "train_tile_id": "tile_train",
            "seed": 0,
            "best_top1_accuracy": 0.5,
            "best_topk_hit_rate": 0.75,
            "final_loss": 1.0,
            "claim_boundary": "phase64",
        }
    ]
    overlap = [
        {
            "variant_id": "D4P8",
            "eval_tile_id": "tile_eval",
            "seed": 0,
            "oracle_gap_fraction": 0.4,
            "selected_overlap_fraction": 0.25,
            "missed_oracle_block_ids": "b2",
            "selected_block_ids": "b1",
        }
    ]
    rank_gap = [
        {
            "variant_id": "D4P8",
            "eval_tile_id": "tile_eval",
            "seed": 0,
            "reward_loss_from_missed_oracle": 0.2,
            "worst_selected_rank": 4,
        }
    ]
    effective_rank = [
        {
            "variant_id": "D4P8",
            "tile_role": "eval",
            "tile_id": "tile_eval",
            "scale_flag": True,
            "shift_flag": False,
            "rank_flag": False,
        }
    ]
    gate = build_phase64_standardization_gate(_comparison(), convergence, effective_rank)
    failure_cases = build_phase64_failure_cases(
        _comparison(),
        overlap,
        rank_gap,
        convergence,
        effective_rank,
        limit=3,
    )
    analysis = {
        "phase": "phase64_set_policy_error_diagnosis",
        "convergence_rows": convergence,
        "overlap_rows": overlap,
        "oracle_rank_gap_rows": rank_gap,
        "feature_scale_rows": [],
        "feature_effective_rank_rows": effective_rank,
        "failure_case_rows": failure_cases,
        "standardization_gate": gate,
        "claim_boundary": "phase64",
    }

    paths = write_phase64_artifacts(analysis, tmp_path / "outputs")

    assert paths["convergence_csv"].name == "phase64_convergence_summary.csv"
    assert paths["overlap_csv"].name == "phase64_rollout_overlap.csv"
    assert paths["oracle_rank_csv"].name == "phase64_oracle_rank_gap.csv"
    assert paths["gate_json"].name == "phase64_standardization_gate.json"
    saved = json.loads(paths["gate_json"].read_text(encoding="utf-8"))
    assert saved["phase64_status"] == "standardization_route_supported"
    markdown = paths["diagnosis_md"].read_text(encoding="utf-8")
    assert "Phase 64 Set-Policy Error Diagnosis" in markdown
    assert "standardization_route_supported" in markdown
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase64_set_policy_error_diagnosis.py -q --basetemp=.pytest_tmp_phase64_task4_red -p no:cacheprovider
```

Expected: fails because gate, failure-case, and writer functions are missing.

- [ ] **Step 3: Add gate, failure-case, writer, and Markdown code**

Append to `src/paper11_geofm/phase64_set_policy_error_diagnosis.py`:

```python
PHASE64_FAILURE_CASE_FIELDNAMES = [
    "case_type",
    "variant_id",
    "eval_tile_id",
    "seed",
    "bc_reward",
    "oracle_total_reward",
    "oracle_gap",
    "oracle_gap_fraction",
    "selected_overlap_fraction",
    "worst_selected_rank",
    "reward_loss_from_missed_oracle",
    "selected_block_ids",
    "missed_oracle_block_ids",
    "training_best_top1_accuracy",
    "training_best_topk_hit_rate",
    "feature_flags",
    "claim_boundary",
]


def _mean_numeric(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [
        _safe_float(row.get(field))
        for row in rows
        if str(row.get(field, "")).strip() != ""
    ]
    return statistics.mean(values) if values else 0.0


def _coverage_incomplete(comparison: Mapping[str, object]) -> bool:
    coverage = comparison.get("coverage_issues", {})
    if not isinstance(coverage, Mapping):
        return True
    return bool(
        coverage.get("missing_rollout_rows")
        or coverage.get("duplicate_rollout_rows")
        or coverage.get("unexpected_rollout_rows")
    )


def _d4_underperforms(comparison: Mapping[str, object]) -> bool:
    d4_b0 = comparison.get("d4_b0_delta_summary", {})
    d4_d6 = comparison.get("d4_d6_delta_summary", {})
    return (
        _safe_float(d4_b0.get("mean_delta"), 0.0) <= 0.0
        or _safe_float(d4_d6.get("mean_delta"), 0.0) <= 0.0
    )


def _geofm_feature_flags(feature_effective_rank_rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    flags = {"scale_flag_count": 0, "shift_flag_count": 0, "rank_flag_count": 0}
    for row in feature_effective_rank_rows:
        variant_id = str(row.get("variant_id", ""))
        if not (variant_id.startswith("D4") or variant_id.startswith("D6")):
            continue
        if str(row.get("tile_role", "")) not in {"train", "eval"}:
            continue
        if bool(row.get("scale_flag")):
            flags["scale_flag_count"] += 1
        if bool(row.get("shift_flag")):
            flags["shift_flag_count"] += 1
        if bool(row.get("rank_flag")):
            flags["rank_flag_count"] += 1
    return flags


def build_phase64_standardization_gate(
    phase63_comparison: Mapping[str, object],
    convergence_rows: Sequence[Mapping[str, object]],
    feature_effective_rank_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    flags = _geofm_feature_flags(feature_effective_rank_rows)
    mean_best_top1 = _mean_numeric(convergence_rows, "best_top1_accuracy")
    mean_best_topk = _mean_numeric(convergence_rows, "best_topk_hit_rate")
    capacity_limited = (
        mean_best_top1 < PHASE64_WEAK_TOP1_THRESHOLD
        and mean_best_topk < PHASE64_WEAK_TOPK_THRESHOLD
    )
    d4_underperformance = _d4_underperforms(phase63_comparison)
    feature_flagged = any(value > 0 for value in flags.values())
    if _coverage_incomplete(phase63_comparison):
        status = PHASE64_STATUS_INCONCLUSIVE
        recommendation = False
        reason = "Phase 63 coverage is incomplete."
    elif capacity_limited:
        status = PHASE64_STATUS_CAPACITY
        recommendation = False
        reason = "Behavior cloning convergence is weak across variants."
    elif d4_underperformance and feature_flagged:
        status = PHASE64_STATUS_STANDARDIZATION
        recommendation = True
        reason = "D4/D6 underperformance coincides with feature scale, shift, or rank flags."
    elif d4_underperformance and not feature_flagged and mean_best_topk >= PHASE64_WEAK_TOPK_THRESHOLD:
        status = PHASE64_STATUS_NOT_HELPFUL
        recommendation = False
        reason = "D4 remains behind without scale, shift, or rank flags under adequate convergence."
    else:
        status = PHASE64_STATUS_INCONCLUSIVE
        recommendation = False
        reason = "Diagnostics do not isolate one next experiment."
    return {
        "phase": "phase64_standardization_gate",
        "phase64_status": status,
        "recommend_standardized_rerun": bool(recommendation),
        "reason": reason,
        "mean_best_top1_accuracy": _round_float(mean_best_top1),
        "mean_best_topk_hit_rate": _round_float(mean_best_topk),
        "d4_underperformance": bool(d4_underperformance),
        **flags,
        "claim_boundary": PHASE64_CLAIM_BOUNDARY,
    }


def _index_by_variant_tile_seed(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str, int], Mapping[str, object]]:
    return {
        (
            str(row.get("variant_id", "")),
            str(row.get("eval_tile_id", row.get("tile_id", ""))),
            _safe_int(row.get("seed")),
        ): row
        for row in rows
    }


def _training_index(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, int], Mapping[str, object]]:
    return {
        (str(row.get("variant_id", "")), _safe_int(row.get("seed"))): row
        for row in rows
    }


def _feature_flag_index(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], str]:
    output: dict[tuple[str, str], str] = {}
    for row in rows:
        flags = []
        if bool(row.get("scale_flag")):
            flags.append("scale")
        if bool(row.get("shift_flag")):
            flags.append("shift")
        if bool(row.get("rank_flag")):
            flags.append("rank")
        output[(str(row.get("variant_id", "")), str(row.get("tile_id", "")))] = ";".join(flags)
    return output


def build_phase64_failure_cases(
    phase63_comparison: Mapping[str, object],
    overlap_rows: Sequence[Mapping[str, object]],
    oracle_rank_rows: Sequence[Mapping[str, object]],
    convergence_rows: Sequence[Mapping[str, object]],
    feature_effective_rank_rows: Sequence[Mapping[str, object]],
    limit: int = 12,
) -> list[dict[str, object]]:
    rank_index = _index_by_variant_tile_seed(oracle_rank_rows)
    train_index = _training_index(convergence_rows)
    flag_index = _feature_flag_index(feature_effective_rank_rows)
    candidates: list[tuple[float, str, Mapping[str, object]]] = []
    for row in overlap_rows:
        candidates.append((_safe_float(row.get("oracle_gap_fraction")), "highest_oracle_gap", row))
        if str(row.get("variant_id", "")).startswith("D4"):
            candidates.append((_safe_float(row.get("oracle_gap_fraction")), "d4_high_oracle_gap", row))
        if _safe_float(row.get("selected_overlap_fraction"), 1.0) < 0.5:
            candidates.append((1.0 - _safe_float(row.get("selected_overlap_fraction")), "weak_selected_overlap", row))
    for delta_row in phase63_comparison.get("d4_b0_delta_rows", []):
        if _safe_float(delta_row.get("left_minus_right_reward")) < 0.0:
            candidates.append(
                (
                    abs(_safe_float(delta_row.get("left_minus_right_reward"))),
                    "d4_loses_to_b0",
                    {
                        "variant_id": delta_row.get("left_variant_id", ""),
                        "eval_tile_id": delta_row.get("eval_tile_id", ""),
                        "seed": delta_row.get("seed", 0),
                    },
                )
            )
    for delta_row in phase63_comparison.get("d4_d6_delta_rows", []):
        if _safe_float(delta_row.get("left_minus_right_reward")) < 0.0:
            candidates.append(
                (
                    abs(_safe_float(delta_row.get("left_minus_right_reward"))),
                    "d4_loses_to_d6",
                    {
                        "variant_id": delta_row.get("left_variant_id", ""),
                        "eval_tile_id": delta_row.get("eval_tile_id", ""),
                        "seed": delta_row.get("seed", 0),
                    },
                )
            )
    output: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, int]] = set()
    overlap_index = _index_by_variant_tile_seed(overlap_rows)
    for _score, case_type, base_row in sorted(candidates, key=lambda item: (-item[0], item[1])):
        variant_id = str(base_row.get("variant_id", ""))
        eval_tile_id = str(base_row.get("eval_tile_id", ""))
        seed = _safe_int(base_row.get("seed"))
        seen_key = (case_type, variant_id, eval_tile_id, seed)
        if seen_key in seen:
            continue
        seen.add(seen_key)
        key = (variant_id, eval_tile_id, seed)
        overlap = overlap_index.get(key, base_row)
        rank = rank_index.get(key, {})
        training = train_index.get((variant_id, seed), {})
        feature_flags = flag_index.get((variant_id, eval_tile_id), "")
        output.append(
            {
                "case_type": case_type,
                "variant_id": variant_id,
                "eval_tile_id": eval_tile_id,
                "seed": seed,
                "bc_reward": _round_float(overlap.get("bc_reward", 0.0)),
                "oracle_total_reward": _round_float(overlap.get("oracle_total_reward", 0.0)),
                "oracle_gap": _round_float(overlap.get("oracle_gap", 0.0)),
                "oracle_gap_fraction": _round_float(overlap.get("oracle_gap_fraction", 0.0)),
                "selected_overlap_fraction": _round_float(overlap.get("selected_overlap_fraction", 0.0)),
                "worst_selected_rank": _safe_int(rank.get("worst_selected_rank")),
                "reward_loss_from_missed_oracle": _round_float(rank.get("reward_loss_from_missed_oracle", 0.0)),
                "selected_block_ids": str(overlap.get("selected_block_ids", "")),
                "missed_oracle_block_ids": str(overlap.get("missed_oracle_block_ids", "")),
                "training_best_top1_accuracy": _round_float(training.get("best_top1_accuracy", 0.0)),
                "training_best_topk_hit_rate": _round_float(training.get("best_topk_hit_rate", 0.0)),
                "feature_flags": feature_flags,
                "claim_boundary": PHASE64_CLAIM_BOUNDARY,
            }
        )
        if len(output) >= int(limit):
            break
    return output


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
    if isinstance(value, Path):
        return str(value)
    return value


def _phase64_markdown(analysis: Mapping[str, object]) -> str:
    gate = dict(analysis.get("standardization_gate", {}))
    lines = [
        "# Phase 64 Set-Policy Error Diagnosis",
        "",
        f"Status: {gate.get('phase64_status', '')}",
        "",
        f"Recommendation: standardized rerun = {gate.get('recommend_standardized_rerun', False)}",
        "",
        f"Reason: {gate.get('reason', '')}",
        "",
        "Gate evidence:",
        f"- mean best top-1 accuracy: {gate.get('mean_best_top1_accuracy', '')}",
        f"- mean best top-k hit rate: {gate.get('mean_best_topk_hit_rate', '')}",
        f"- D4 underperformance: {gate.get('d4_underperformance', '')}",
        f"- scale flag count: {gate.get('scale_flag_count', '')}",
        f"- shift flag count: {gate.get('shift_flag_count', '')}",
        f"- rank flag count: {gate.get('rank_flag_count', '')}",
        "",
        "Failure case rows:",
        f"- {len(analysis.get('failure_case_rows', []))}",
        "",
        "Claim boundary:",
        str(analysis.get("claim_boundary", PHASE64_CLAIM_BOUNDARY)),
        "",
    ]
    return "\n".join(lines)


def write_phase64_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "convergence_csv": output_path / "phase64_convergence_summary.csv",
        "overlap_csv": output_path / "phase64_rollout_overlap.csv",
        "oracle_rank_csv": output_path / "phase64_oracle_rank_gap.csv",
        "feature_scale_csv": output_path / "phase64_feature_scale_summary.csv",
        "effective_rank_csv": output_path / "phase64_feature_effective_rank.csv",
        "failure_cases_csv": output_path / "phase64_failure_cases.csv",
        "gate_json": output_path / "phase64_standardization_gate.json",
        "diagnosis_md": output_path / "phase64_set_policy_error_diagnosis.md",
    }
    _write_csv_rows(paths["convergence_csv"], PHASE64_CONVERGENCE_FIELDNAMES, analysis.get("convergence_rows", []))
    _write_csv_rows(paths["overlap_csv"], PHASE64_OVERLAP_FIELDNAMES, analysis.get("overlap_rows", []))
    _write_csv_rows(paths["oracle_rank_csv"], PHASE64_ORACLE_RANK_FIELDNAMES, analysis.get("oracle_rank_gap_rows", []))
    _write_csv_rows(paths["feature_scale_csv"], PHASE64_FEATURE_SCALE_FIELDNAMES, analysis.get("feature_scale_rows", []))
    _write_csv_rows(paths["effective_rank_csv"], PHASE64_EFFECTIVE_RANK_FIELDNAMES, analysis.get("feature_effective_rank_rows", []))
    _write_csv_rows(paths["failure_cases_csv"], PHASE64_FAILURE_CASE_FIELDNAMES, analysis.get("failure_case_rows", []))
    paths["gate_json"].write_text(
        json.dumps(_json_ready(analysis.get("standardization_gate", {})), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["diagnosis_md"].write_text(_phase64_markdown(analysis), encoding="utf-8")
    return paths
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase64_set_policy_error_diagnosis.py -q --basetemp=.pytest_tmp_phase64_task4_green -p no:cacheprovider
```

Expected: all current Phase 64 tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add src\paper11_geofm\phase64_set_policy_error_diagnosis.py tests\test_phase64_set_policy_error_diagnosis.py
git commit -m "feat: add Phase 64 standardization gate"
```

---

### Task 5: Run Wrapper and CLI

**Files:**
- Modify: `tests/test_phase64_set_policy_error_diagnosis.py`
- Modify: `src/paper11_geofm/phase64_set_policy_error_diagnosis.py`
- Create: `experiments/phase64_set_policy_error_diagnosis/run_phase64_set_policy_error_diagnosis.py`

- [ ] **Step 1: Add failing tests for run wrapper and CLI parser**

Append this synthetic fixture helper and tests:

```python
def _write_variant_fixture(output_dir: Path, variant_id: str, rows: list[dict[str, float]], columns: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table = output_dir / f"variant_{variant_id}_features.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["block_id", *columns])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    manifest = {
        "variants": {
            variant_id: {
                "ready": True,
                "feature_table": table.name,
                "required_columns": columns,
                "reward": "base_planning_reward",
                "state_groups": ["synthetic"],
            }
        }
    }
    (output_dir / "experiment_variants.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_phase64_run_wrapper_loads_phase63_artifacts_and_writes_analysis(tmp_path):
    from paper11_geofm.phase64_set_policy_error_diagnosis import (
        run_phase64_set_policy_error_diagnosis,
    )

    columns = _required_feature_columns()
    feature_rows = [
        {**{"block_id": "b1"}, **{column: 0.0 for column in columns}},
        {**{"block_id": "b2"}, **{column: 0.0 for column in columns}},
        {**{"block_id": "b3"}, **{column: 0.0 for column in columns}},
    ]
    feature_rows[0]["explicit_feature_16"] = 0.9
    feature_rows[1]["explicit_feature_16"] = 0.8
    feature_rows[2]["explicit_feature_16"] = 0.2
    phase2 = tmp_path / "phase2"
    _write_variant_fixture(phase2, "B0", feature_rows, list(columns))
    tile_index = _write_csv(
        tmp_path / "tiles.csv",
        [
            {"tile_id": "tile_train", "block_ids": "b1;b2;b3"},
            {"tile_id": "tile_eval", "block_ids": "b1;b2;b3"},
        ],
    )
    comparison = _comparison(d4_b0_mean=0.1, d4_d6_mean=0.1)
    comparison["contract"] = {
        "phase2_output_dir": str(phase2),
        "phase8_output_dir": str(tmp_path / "phase8"),
        "phase61_output_dir": str(tmp_path / "phase61"),
        "tile_index_csv": str(tile_index),
        "variant_source_dirs": {"B0": str(phase2)},
        "variants": ["B0"],
        "train_tile_id": "tile_train",
        "eval_tile_ids": ["tile_eval"],
        "seeds": [0],
        "eval_max_steps": 2,
    }
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    rollout_csv = _write_csv(tmp_path / "rollout.csv", [_rollout_row(selected="b1;b3", eval_max_steps=2)])
    history_csv = _write_csv(
        tmp_path / "history.csv",
        [
            _history_row("B0", 0, 1, loss=2.0, top1=0.0, topk=0.5),
            _history_row("B0", 0, 2, loss=1.0, top1=0.5, topk=1.0),
        ],
    )
    oracle_csv = _write_csv(tmp_path / "oracle.csv", [_oracle_row(selected="b1;b2", eval_max_steps=2)])

    analysis = run_phase64_set_policy_error_diagnosis(
        phase63_comparison_json=comparison_path,
        phase63_rollout_csv=rollout_csv,
        phase63_history_csv=history_csv,
        phase63_oracle_summary_csv=oracle_csv,
    )

    assert analysis["phase"] == "phase64_set_policy_error_diagnosis"
    assert len(analysis["convergence_rows"]) == 1
    assert len(analysis["overlap_rows"]) == 1
    assert len(analysis["oracle_rank_gap_rows"]) == 1
    assert analysis["standardization_gate"]["phase64_status"] in {
        "diagnostic_inconclusive",
        "geofm_features_not_helpful_under_set_policy",
        "standardization_route_supported",
    }


def test_phase64_cli_parser_accepts_required_inputs():
    runner_path = (
        ROOT
        / "experiments"
        / "phase64_set_policy_error_diagnosis"
        / "run_phase64_set_policy_error_diagnosis.py"
    )
    spec = importlib.util.spec_from_file_location("phase64_runner_args", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--phase63-comparison-json",
            "comparison.json",
            "--phase63-rollout-csv",
            "rollout.csv",
            "--phase63-history-csv",
            "history.csv",
            "--phase63-oracle-summary-csv",
            "oracle.csv",
            "--output-dir",
            "outputs",
        ]
    )

    assert args.phase63_comparison_json == Path("comparison.json")
    assert args.output_dir == Path("outputs")
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase64_set_policy_error_diagnosis.py -q --basetemp=.pytest_tmp_phase64_task5_red -p no:cacheprovider
```

Expected: fails because the wrapper or runner is missing.

- [ ] **Step 3: Add the Phase 64 run wrapper**

Append to `src/paper11_geofm/phase64_set_policy_error_diagnosis.py`:

```python
def _contract_tiled_inputs(contract: Mapping[str, object]) -> list[tuple[str, object]]:
    variant_source_dirs = contract.get("variant_source_dirs", {})
    if not isinstance(variant_source_dirs, Mapping):
        raise ValueError("Phase 63 contract is missing variant_source_dirs")
    variants = [str(value) for value in contract.get("variants", [])]
    train_tile_id = str(contract.get("train_tile_id", ""))
    eval_tile_ids = [str(value) for value in contract.get("eval_tile_ids", [])]
    tile_index_csv = str(contract.get("tile_index_csv", ""))
    tiled_inputs: list[tuple[str, object]] = []
    for variant_id in variants:
        source_dir = variant_source_dirs.get(variant_id)
        if source_dir is None:
            raise ValueError(f"Phase 63 contract has no source for variant {variant_id}")
        tiled_inputs.append(
            (
                "train",
                load_tiled_variant_input(
                    source_dir,
                    tile_index_csv,
                    train_tile_id,
                    variant_id=variant_id,
                ),
            )
        )
        for eval_tile_id in eval_tile_ids:
            tiled_inputs.append(
                (
                    "eval",
                    load_tiled_variant_input(
                        source_dir,
                        tile_index_csv,
                        eval_tile_id,
                        variant_id=variant_id,
                    ),
                )
            )
    return tiled_inputs


def _tiled_input_index(tiled_inputs: Sequence[tuple[str, object]]) -> dict[tuple[str, str], object]:
    return {
        (str(tiled_input.variant_id), str(tiled_input.tile_id)): tiled_input
        for _role, tiled_input in tiled_inputs
    }


def run_phase64_set_policy_error_diagnosis(
    phase63_comparison_json: Path | str,
    phase63_rollout_csv: Path | str,
    phase63_history_csv: Path | str,
    phase63_oracle_summary_csv: Path | str,
) -> dict[str, object]:
    comparison = _load_json_object(phase63_comparison_json, "Phase 63 comparison JSON")
    contract = comparison.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Phase 63 comparison JSON is missing contract metadata")
    train_tile_id = str(contract.get("train_tile_id", ""))
    variants = [str(value) for value in contract.get("variants", [])]
    train_tile_ids = {variant_id: train_tile_id for variant_id in variants}
    tiled_inputs = _contract_tiled_inputs(contract)
    tiled_index = _tiled_input_index(tiled_inputs)
    convergence_rows = build_phase64_convergence_summary(phase63_history_csv)
    overlap_rows = build_phase64_rollout_overlap(phase63_rollout_csv, phase63_oracle_summary_csv)
    oracle_rank_gap_rows = build_phase64_oracle_rank_gap(phase63_rollout_csv, tiled_index)
    feature_diagnostics = build_phase64_feature_diagnostics(tiled_inputs, train_tile_ids)
    standardization_gate = build_phase64_standardization_gate(
        comparison,
        convergence_rows,
        feature_diagnostics["feature_effective_rank_rows"],
    )
    failure_case_rows = build_phase64_failure_cases(
        comparison,
        overlap_rows,
        oracle_rank_gap_rows,
        convergence_rows,
        feature_diagnostics["feature_effective_rank_rows"],
    )
    return {
        "phase": "phase64_set_policy_error_diagnosis",
        "phase63_comparison_json": str(Path(phase63_comparison_json)),
        "phase63_rollout_csv": str(Path(phase63_rollout_csv)),
        "phase63_history_csv": str(Path(phase63_history_csv)),
        "phase63_oracle_summary_csv": str(Path(phase63_oracle_summary_csv)),
        "contract": dict(contract),
        "phase63_comparison": comparison,
        "convergence_rows": convergence_rows,
        "overlap_rows": overlap_rows,
        "oracle_rank_gap_rows": oracle_rank_gap_rows,
        "feature_scale_rows": feature_diagnostics["feature_scale_rows"],
        "feature_effective_rank_rows": feature_diagnostics["feature_effective_rank_rows"],
        "failure_case_rows": failure_case_rows,
        "standardization_gate": standardization_gate,
        "claim_boundary": PHASE64_CLAIM_BOUNDARY,
    }
```

- [ ] **Step 4: Add the CLI runner**

Create `experiments/phase64_set_policy_error_diagnosis/run_phase64_set_policy_error_diagnosis.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase64_set_policy_error_diagnosis import (
    run_phase64_set_policy_error_diagnosis,
    write_phase64_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = run_phase64_set_policy_error_diagnosis(
            phase63_comparison_json=args.phase63_comparison_json,
            phase63_rollout_csv=args.phase63_rollout_csv,
            phase63_history_csv=args.phase63_history_csv,
            phase63_oracle_summary_csv=args.phase63_oracle_summary_csv,
        )
        paths = write_phase64_artifacts(analysis, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    gate = analysis["standardization_gate"]
    print(f"Phase 64 status: {gate['phase64_status']}")
    print(f"Standardized rerun recommended: {gate['recommend_standardized_rerun']}")
    print(f"Gate JSON: {paths['gate_json']}")
    print(f"Diagnosis Markdown: {paths['diagnosis_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 64 set-policy error diagnosis."
    )
    parser.add_argument("--phase63-comparison-json", type=Path, required=True)
    parser.add_argument("--phase63-rollout-csv", type=Path, required=True)
    parser.add_argument("--phase63-history-csv", type=Path, required=True)
    parser.add_argument("--phase63-oracle-summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase64_set_policy_error_diagnosis.py -q --basetemp=.pytest_tmp_phase64_task5_green -p no:cacheprovider
```

Expected: all Phase 64 tests pass.

- [ ] **Step 6: Commit Task 5**

Run:

```powershell
git add src\paper11_geofm\phase64_set_policy_error_diagnosis.py tests\test_phase64_set_policy_error_diagnosis.py experiments\phase64_set_policy_error_diagnosis\run_phase64_set_policy_error_diagnosis.py
git commit -m "feat: add Phase 64 diagnosis runner"
```

---

### Task 6: Real Phase 64 Run and Evidence Record

**Files:**
- Create: `paper/phase28_results/30_phase64_set_policy_error_diagnosis.md`
- Modify: `paper/phase28_results/README.md`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [ ] **Step 1: Run targeted tests before the real diagnostic run**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase64_set_policy_error_diagnosis.py tests\test_phase63_set_policy_oracle_pretraining.py -q --basetemp=.pytest_tmp_phase64_pre_real -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the real Phase 64 diagnostic**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase64_set_policy_error_diagnosis\run_phase64_set_policy_error_diagnosis.py --phase63-comparison-json experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_set_policy_comparison.json --phase63-rollout-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_bc_rollout_summary.csv --phase63-history-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_bc_training_history.csv --phase63-oracle-summary-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_oracle_summary.csv --output-dir experiments\phase64_set_policy_error_diagnosis\outputs\phase52_full5_seed3
```

Expected generated artifacts:

- `experiments/phase64_set_policy_error_diagnosis/outputs/phase52_full5_seed3/phase64_convergence_summary.csv`
- `experiments/phase64_set_policy_error_diagnosis/outputs/phase52_full5_seed3/phase64_rollout_overlap.csv`
- `experiments/phase64_set_policy_error_diagnosis/outputs/phase52_full5_seed3/phase64_oracle_rank_gap.csv`
- `experiments/phase64_set_policy_error_diagnosis/outputs/phase52_full5_seed3/phase64_feature_scale_summary.csv`
- `experiments/phase64_set_policy_error_diagnosis/outputs/phase52_full5_seed3/phase64_feature_effective_rank.csv`
- `experiments/phase64_set_policy_error_diagnosis/outputs/phase52_full5_seed3/phase64_failure_cases.csv`
- `experiments/phase64_set_policy_error_diagnosis/outputs/phase52_full5_seed3/phase64_standardization_gate.json`
- `experiments/phase64_set_policy_error_diagnosis/outputs/phase52_full5_seed3/phase64_set_policy_error_diagnosis.md`

- [ ] **Step 3: Inspect the gate decision and key numbers**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -c "import json; p='experiments/phase64_set_policy_error_diagnosis/outputs/phase52_full5_seed3/phase64_standardization_gate.json'; d=json.load(open(p, encoding='utf-8')); print(d['phase64_status']); print(d['recommend_standardized_rerun']); print(d['reason']); print({k:d[k] for k in ['mean_best_top1_accuracy','mean_best_topk_hit_rate','d4_underperformance','scale_flag_count','shift_flag_count','rank_flag_count']})"
```

Expected: prints one Phase 64 status, the standardized-rerun boolean, the reason string, and the gate evidence dictionary.

- [ ] **Step 4: Create the Phase 64 evidence document**

Run:

```powershell
Copy-Item -LiteralPath experiments\phase64_set_policy_error_diagnosis\outputs\phase52_full5_seed3\phase64_set_policy_error_diagnosis.md -Destination paper\phase28_results\30_phase64_set_policy_error_diagnosis.md
```

Append this reproduction and boundary block to `paper/phase28_results/30_phase64_set_policy_error_diagnosis.md`:

````markdown
## Reproduction

Run Phase 64 from the repository root after Phase 63 full-run artifacts exist:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase64_set_policy_error_diagnosis\run_phase64_set_policy_error_diagnosis.py --phase63-comparison-json experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_set_policy_comparison.json --phase63-rollout-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_bc_rollout_summary.csv --phase63-history-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_bc_training_history.csv --phase63-oracle-summary-csv experiments\phase63_set_policy_oracle_pretraining\outputs\phase52_full5_seed3\phase63_oracle_summary.csv --output-dir experiments\phase64_set_policy_error_diagnosis\outputs\phase52_full5_seed3
```

## Boundary

Phase 64 is diagnostic evidence under the existing deterministic base-planning
reward. It does not run a new policy training phase, does not enable
suitability reward, does not test B2/B3, does not test cross-region transfer,
does not prove GeoFM advantage, does not prove PCA optimality, and does not
justify formal submission-level planning-performance claims. No formal
manuscript files were changed in this phase.
````

- [ ] **Step 5: Update `paper/phase28_results/README.md`**

Add this bullet after the Phase 63 entry:

```markdown
- `30_phase64_set_policy_error_diagnosis.md`: set-policy error diagnosis and
  standardization gate using Phase 63 behavior-cloned rollout, oracle summary,
  training-history, and feature-scale artifacts to decide whether standardized
  set-policy BC is the next justified algorithm experiment.
```

Add this paragraph near the recent phase history:

```markdown
Phase 64 keeps the work in algorithm/experiment mode. It does not retrain a
policy. It reads Phase 63 set-policy artifacts, diagnoses selected-block
overlap, oracle-rank gaps, training convergence, and feature scale/effective
rank, then records a standardization gate decision in
`30_phase64_set_policy_error_diagnosis.md`.
```

- [ ] **Step 6: Update the handoff document**

In `docs/superpowers/phase33_current_progress_handoff.md`, add a Phase 64 block with the values printed in Step 3:

```markdown
## Phase 64 Set-Policy Error Diagnosis and Standardization Gate

- Branch: `main`
- Formal manuscript files changed: no
- Implementation module: `src/paper11_geofm/phase64_set_policy_error_diagnosis.py`
- Runner: `experiments/phase64_set_policy_error_diagnosis/run_phase64_set_policy_error_diagnosis.py`
- Evidence document: `paper/phase28_results/30_phase64_set_policy_error_diagnosis.md`
- Generated output directory: `experiments/phase64_set_policy_error_diagnosis/outputs/phase52_full5_seed3`
- Phase 64 status: copy the first line printed by Step 3
- Standardized rerun recommended: copy the second line printed by Step 3
- Gate reason: copy the third line printed by Step 3
- Gate evidence: copy the dictionary printed by Step 3
- Claim boundary: base-reward diagnostic evidence only; no new policy training, no suitability reward, no B2/B3, no transfer, no GeoFM-advantage claim, no PCA-optimality claim, no formal submission-level performance claim.
```

- [ ] **Step 7: Commit Task 6**

Run:

```powershell
git add paper\phase28_results\30_phase64_set_policy_error_diagnosis.md paper\phase28_results\README.md docs\superpowers\phase33_current_progress_handoff.md
git commit -m "docs: record Phase 64 set policy diagnosis"
```

---

### Task 7: Final Verification and Push

**Files:**
- No planned file edits.

- [ ] **Step 1: Run targeted Phase 64 and regression tests**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase64_set_policy_error_diagnosis.py tests\test_phase63_set_policy_oracle_pretraining.py tests\test_phase62_d4_d6_matched_ppo.py -q --basetemp=.pytest_tmp_phase64_final -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 2: Run repository smoke check**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe scripts\smoke_check.py
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

- [ ] **Step 6: Push saved Phase 64 work**

Run:

```powershell
git push
```

Expected: `origin/main` receives the Phase 64 spec, implementation, diagnostics, evidence, and handoff commits.

---

## Execution Notes

- Keep Phase 64 read-only against Phase 63 artifacts. Do not rerun Phase 63 training inside Phase 64.
- Keep generated outputs under `experiments/phase64_set_policy_error_diagnosis/outputs/`.
- Do not edit formal submission files in this phase.
- If the gate returns `standardization_route_supported`, the next phase should be a separate spec or plan for train-tile-fitted standardized set-policy BC.
- If the gate returns `bc_training_capacity_limited`, the next phase should improve the supervised set-policy training setup before representation conclusions.
- If the gate returns `geofm_features_not_helpful_under_set_policy`, the next phase should avoid a GeoFM advantage claim under the current base-reward protocol.
