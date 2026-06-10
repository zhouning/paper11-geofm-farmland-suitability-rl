# Phase 15 Tiled Batch Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a batch smoke runner that verifies every Phase 13 tile can run a one-step B1 tile-level input contract.

**Architecture:** Add `paper11_geofm.tiled_batch_smoke` to load a Phase 2 variant once, iterate Phase 13 tile rows, run one Phase 4 environment step per tile, and write CSV/JSON summaries. Add a CLI, tests, and docs.

**Tech Stack:** Python, NumPy through existing input matrices, CSV/JSON artifacts, pytest.

---

## File Structure

- Create `src/paper11_geofm/tiled_batch_smoke.py`: batch tile reader, one-step runner, aggregate report, artifact writer.
- Create `experiments/phase15_tiled_batch_smoke/run_phase15_tiled_batch_smoke.py`: CLI runner.
- Create `tests/test_phase15_tiled_batch_smoke.py`: tests for batch summaries, tile cap, reward rejection, writer, and CLI.
- Modify `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`, and `reproducibility/FILE_MANIFEST.tsv`.

## Task 1: Batch Runner Test

- [ ] **Step 1: Write failing tests**

Create `tests/test_phase15_tiled_batch_smoke.py` with helpers similar to Phase 14 but with two tile rows:

```python
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id):
    row = {"block_id": block_id, "suitability_proxy": 0.75}
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(dim)
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = float(idx)
    return row


def _write_ready_phase2_outputs(output_dir):
    from paper11_geofm.artifacts import write_phase2_artifacts

    return write_phase2_artifacts(
        [_complete_phase2_feature_row(block_id) for block_id in ["b1", "b2", "b3", "b4"]],
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
def test_phase15_runs_batch_smoke_for_all_tiles(tmp_path):
    from paper11_geofm.tiled_batch_smoke import (
        PHASE15_CLAIM_BOUNDARY,
        run_phase15_tiled_batch_smoke,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    report = run_phase15_tiled_batch_smoke(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variant_id="B1",
    )

    assert report["tile_count"] == 2
    assert report["total_blocks"] == 4
    assert report["block_count_summary"]["max"] == 3
    assert report["max_observation_shape"] == 246
    assert report["all_tile_smokes_passed"] is True
    assert report["rows"][0]["tile_id"] == "tile_r000_c000"
    assert report["rows"][0]["selected_block_id"] == "b1"
    assert report["rows"][0]["step_reward"] == 0.0
    assert report["claim_boundary"] == PHASE15_CLAIM_BOUNDARY
```

- [ ] **Step 2: Run and confirm failure**

Run:

```powershell
python -m pytest tests\test_phase15_tiled_batch_smoke.py::test_phase15_runs_batch_smoke_for_all_tiles -q
```

Expected: fail with `ModuleNotFoundError: No module named 'paper11_geofm.tiled_batch_smoke'`.

## Task 2: Batch Module Implementation

- [ ] **Step 1: Create `src/paper11_geofm/tiled_batch_smoke.py`**

Implement:

- `PHASE15_CLAIM_BOUNDARY`;
- `run_phase15_tiled_batch_smoke(phase2_output_dir, tile_index_csv, variant_id="B1", max_tiles=None)`;
- `write_phase15_tiled_batch_smoke(report, output_dir)`.

Rules:

- load `VariantInput` once with `load_variant_input`;
- reject `base_plus_suitability_reward`;
- parse all tile rows from Phase 13 CSV;
- optionally cap by `max_tiles`;
- for each tile, build `TiledVariantInput`, run `Phase4InputContractEnv`, reset, take first valid action;
- write one CSV row per tile and one JSON aggregate report;
- report `all_tile_smokes_passed = true` only if every tile status is `passed`.

- [ ] **Step 2: Run the first test**

Run:

```powershell
python -m pytest tests\test_phase15_tiled_batch_smoke.py::test_phase15_runs_batch_smoke_for_all_tiles -q
```

Expected: pass.

## Task 3: Writer, Cap, Rejection, and CLI

- [ ] **Step 1: Add tests**

Add tests for:

- `max_tiles=1` returns one row;
- B3 is rejected by default;
- writer creates `phase15_tiled_batch_smoke_summary.csv` and `phase15_tiled_batch_smoke_report.json`;
- CLI writes outputs and prints `Tiles processed: 2`, `All passed: True`, and output paths.

- [ ] **Step 2: Create CLI after CLI test fails**

Create `experiments/phase15_tiled_batch_smoke/run_phase15_tiled_batch_smoke.py` with flags:

- `--phase2-output-dir`;
- `--tile-index-csv`;
- `--variant` default `B1`;
- `--output-dir`;
- `--max-tiles`.

Return `1` for `FileNotFoundError` or `ValueError`.

- [ ] **Step 3: Run focused tests**

Run:

```powershell
python -m pytest tests\test_phase15_tiled_batch_smoke.py -q
```

Expected: all Phase 15 tests pass.

## Task 4: Docs, Real Run, Verification, Integration

- [ ] **Step 1: Update docs and manifest**

Add Phase 15 to README, reproduction guide, and file manifest.

- [ ] **Step 2: Run real Phase 15**

Run:

```powershell
python experiments\phase15_tiled_batch_smoke\run_phase15_tiled_batch_smoke.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variant B1 --output-dir experiments\phase15_tiled_batch_smoke\outputs\real_bishan_all_tiles
```

Expected:

- tiles processed: `54`;
- total blocks: `64984`;
- max observation shape: `180957`;
- all passed: `True`.

- [ ] **Step 3: Full verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests -q
git diff --check
```

- [ ] **Step 4: Commit, merge, push**

Commit with message `Add Phase 15 tiled batch smoke`, push feature branch, fast-forward merge to `main`, rerun real Phase 15 and full verification on `main`, push `main`, and delete the local feature branch.

---

## Self-Review

- Spec coverage: covers batch runner, reward rejection, max tile cap, writer, CLI, docs, real run, and integration.
- Placeholder scan: no placeholder steps remain.
- Type consistency: output names, function names, and CLI flags match the design.
