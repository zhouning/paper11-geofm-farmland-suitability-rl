# Phase 16 Tiled Baseline Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tiled non-learning baseline protocol that runs short deterministic masked rollouts across real Phase 13 tiles.

**Architecture:** Add `paper11_geofm.tiled_baseline_protocol` to load a representation-only Phase 2 variant once, iterate Phase 13 tile rows, and run `first_valid` and `seeded_random` policies in the existing Phase 4 action-mask environment. Add a CLI, tests, documentation, and real Bishan verification.

**Tech Stack:** Python, NumPy through existing input matrices, CSV/JSON artifacts, pytest.

---

## File Structure

- Create `src/paper11_geofm/tiled_baseline_protocol.py`: tile parsing, deterministic policy selection, per-tile rollout, aggregate report, CSV/JSON writer.
- Create `experiments/phase16_tiled_baseline_protocol/run_phase16_tiled_baselines.py`: CLI runner.
- Create `tests/test_phase16_tiled_baseline_protocol.py`: tests for default policies, seeded reproducibility, max-tile cap, reward-variant rejection, writer output, and CLI output.
- Modify `README.md`: add Phase 16 command and key entry point.
- Modify `reproducibility/REPRODUCTION_GUIDE.md`: add Phase 16 section after Phase 15 and update runtime code list.
- Modify `reproducibility/FILE_MANIFEST.tsv`: add Phase 16 design, plan, module, CLI, and tests.

## Task 1: Baseline Protocol Contract Test

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase16_tiled_baseline_protocol.py`:

```python
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id, suitability=0.75):
    row = {"block_id": block_id, "suitability_proxy": suitability}
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(dim)
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = float(idx)
    return row


def _write_ready_phase2_outputs(output_dir):
    from paper11_geofm.artifacts import write_phase2_artifacts

    return write_phase2_artifacts(
        [
            _complete_phase2_feature_row("b1", 0.25),
            _complete_phase2_feature_row("b2", 0.50),
            _complete_phase2_feature_row("b3", 0.75),
            _complete_phase2_feature_row("b4", 1.00),
        ],
        output_dir,
        {
            "metadata_source": "test",
            "base_year_requested": 2020,
            "base_year_used": 2020,
            "years": [2020],
            "grid_shape": [2, 2],
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
                "n_blocks": 3,
                "block_ids": "b1;b3;b4",
            }
        )
        writer.writerow(
            {
                "tile_id": "tile_r000_c001",
                "tile_row": 0,
                "tile_col": 1,
                "n_blocks": 1,
                "block_ids": "b2",
            }
        )
    return path
```

Add:

```python
def test_phase16_runs_default_policies_for_all_tiles(tmp_path):
    from paper11_geofm.tiled_baseline_protocol import (
        PHASE16_CLAIM_BOUNDARY,
        run_phase16_tiled_baseline_protocol,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    protocol = run_phase16_tiled_baseline_protocol(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variant_id="B1",
        max_steps=2,
        seed=0,
    )

    assert protocol["tile_count"] == 2
    assert protocol["policy_ids"] == ["first_valid", "seeded_random"]
    assert protocol["summary_count"] == 4
    assert protocol["total_blocks"] == 4
    assert protocol["max_observation_shape"] == 246
    assert protocol["all_rollouts_completed"] is True
    assert protocol["summaries"][0]["policy_id"] == "first_valid"
    assert protocol["summaries"][0]["tile_id"] == "tile_r000_c000"
    assert protocol["summaries"][0]["episode_steps"] == 2
    assert protocol["summaries"][0]["selected_block_ids"] == ["b1", "b3"]
    assert protocol["summaries"][0]["total_contract_reward"] == 0.0
    assert protocol["claim_boundary"] == PHASE16_CLAIM_BOUNDARY
```

- [ ] **Step 2: Run and confirm failure**

Run:

```powershell
python -m pytest tests\test_phase16_tiled_baseline_protocol.py::test_phase16_runs_default_policies_for_all_tiles -q
```

Expected: fail with `ModuleNotFoundError: No module named 'paper11_geofm.tiled_baseline_protocol'`.

## Task 2: Protocol Module Implementation

- [ ] **Step 1: Create `src/paper11_geofm/tiled_baseline_protocol.py`**

Implement:

- `PHASE16_CLAIM_BOUNDARY`;
- `SUMMARY_FIELDNAMES`;
- `run_phase16_tiled_baseline_protocol(phase2_output_dir, tile_index_csv, variant_id="B1", policy_ids=("first_valid", "seeded_random"), max_steps=4, seed=0, max_tiles=None)`;
- `write_phase16_tiled_baseline_artifacts(protocol, output_dir)`.

Required behavior:

- load the requested Phase 2 variant once with `load_variant_input`;
- reject `base_plus_suitability_reward` variants;
- parse Phase 13 tile rows from CSV;
- optionally cap rows by `max_tiles`;
- support only `first_valid` and `seeded_random`;
- derive a deterministic seeded random generator from `seed`, `policy_id`, `variant_id`, and `tile_id`;
- run each policy on each tile in a `Phase4InputContractEnv` with `max_steps`;
- return summaries and traces with the Phase 16 claim boundary.

- [ ] **Step 2: Run first test**

Run:

```powershell
python -m pytest tests\test_phase16_tiled_baseline_protocol.py::test_phase16_runs_default_policies_for_all_tiles -q
```

Expected: pass.

## Task 3: Writer, Reproducibility, Cap, and CLI Tests

- [ ] **Step 1: Add tests**

Add tests for:

- `seeded_random` reproducibility and seed sensitivity;
- `max_tiles=1` returns two summaries when two policies are used;
- B3 is rejected by default;
- writer creates `phase16_tiled_baseline_summary.csv` and `phase16_tiled_baseline_traces.json`;
- CLI prints `Tiles processed: 2`, `Summary rows: 4`, `All completed: True`, and output paths.

- [ ] **Step 2: Create CLI after CLI test fails**

Create `experiments/phase16_tiled_baseline_protocol/run_phase16_tiled_baselines.py` with flags:

- `--phase2-output-dir`;
- `--tile-index-csv`;
- `--variant` default `B1`;
- `--policies` default `first_valid,seeded_random`;
- `--max-steps` default `4`;
- `--seed` default `0`;
- `--max-tiles`;
- `--output-dir`.

Return `1` for `FileNotFoundError` or `ValueError`.

- [ ] **Step 3: Run focused tests**

Run:

```powershell
python -m pytest tests\test_phase16_tiled_baseline_protocol.py -q
```

Expected: all Phase 16 tests pass.

## Task 4: Docs, Real Run, Verification, Integration

- [ ] **Step 1: Update docs and manifest**

Add Phase 16 to README, reproduction guide, and file manifest.

- [ ] **Step 2: Run real Phase 16**

Run:

```powershell
python experiments\phase16_tiled_baseline_protocol\run_phase16_tiled_baselines.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variant B1 --policies first_valid,seeded_random --max-steps 4 --seed 0 --output-dir experiments\phase16_tiled_baseline_protocol\outputs\real_bishan_b1
```

Expected:

- tiles processed: `54`;
- policies: `2`;
- summary rows: `108`;
- total blocks: `64984`;
- max observation shape: `180957`;
- all completed: `True`;
- total contract reward remains `0.0` for B1 summaries.

- [ ] **Step 3: Full verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests -q
git diff --check
```

- [ ] **Step 4: Commit, merge, push**

Commit with message `Add Phase 16 tiled baseline protocol`, push feature branch, fast-forward merge to `main`, rerun real Phase 16 and full verification on `main`, push `main`, and delete the local feature branch.

---

## Self-Review

- Spec coverage: covers batch tiled baseline protocol, deterministic policies, max-step cap, reward rejection, writer, CLI, docs, real run, and integration.
- Placeholder scan: no placeholder steps remain.
- Type consistency: output names, function names, and CLI flags match the design.
