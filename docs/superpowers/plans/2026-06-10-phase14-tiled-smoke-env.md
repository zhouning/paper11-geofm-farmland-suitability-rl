# Phase 14 Tiled Smoke Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tile-level one-step smoke runner that loads a Phase 13 tile subset into the existing Phase 4 environment contract.

**Architecture:** Add `paper11_geofm.tiled_inputs` to read Phase 13 tile IDs, subset Phase 2 variant inputs, reject reward variants by default, run one Phase 4 env step, and write a JSON summary. Add a CLI plus tests and docs.

**Tech Stack:** Python, NumPy via existing Phase 3/4 modules, CSV/JSON artifacts, pytest.

---

## File Structure

- Create `src/paper11_geofm/tiled_inputs.py`: tile index parser, tiled variant loader, one-step smoke runner, JSON writer.
- Create `experiments/phase14_tiled_smoke_env/run_phase14_tiled_smoke.py`: CLI runner.
- Create `tests/test_phase14_tiled_smoke.py`: tests for tile subset loading, reward-variant rejection, writer, and CLI.
- Modify `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`, and `reproducibility/FILE_MANIFEST.tsv`.

## Task 1: Tiled Loader Test

- [ ] **Step 1: Write failing tests**

Create `tests/test_phase14_tiled_smoke.py`:

```python
import csv
import importlib.util
import json
import sys
from pathlib import Path


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
        [
            _complete_phase2_feature_row("b1"),
            _complete_phase2_feature_row("b2"),
            _complete_phase2_feature_row("b3"),
            _complete_phase2_feature_row("b4"),
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
            fieldnames=[
                "tile_id",
                "tile_row",
                "tile_col",
                "n_blocks",
                "min_grid_row",
                "max_grid_row",
                "min_grid_col",
                "max_grid_col",
                "block_ids",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "tile_id": "tile_r000_c000",
                "tile_row": 0,
                "tile_col": 0,
                "n_blocks": 3,
                "min_grid_row": 0,
                "max_grid_row": 3,
                "min_grid_col": 0,
                "max_grid_col": 3,
                "block_ids": "b1;b3;b4",
            }
        )
    return path
```

Add:

```python
def test_phase14_loads_tiled_b1_variant_subset_in_tile_order(tmp_path):
    from paper11_geofm.tiled_inputs import (
        PHASE14_CLAIM_BOUNDARY,
        load_tiled_variant_input,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    loaded = load_tiled_variant_input(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        "tile_r000_c000",
        "B1",
    )

    assert loaded.tile_id == "tile_r000_c000"
    assert loaded.variant_id == "B1"
    assert loaded.block_ids == ("b1", "b3", "b4")
    assert loaded.state_matrix.shape == (3, 81)
    assert loaded.reward_mode == "base_planning_reward"
    assert loaded.claim_boundary == PHASE14_CLAIM_BOUNDARY
```

- [ ] **Step 2: Run and confirm failure**

Run:

```powershell
python -m pytest tests\test_phase14_tiled_smoke.py::test_phase14_loads_tiled_b1_variant_subset_in_tile_order -q
```

Expected: fail with `ModuleNotFoundError: No module named 'paper11_geofm.tiled_inputs'`.

## Task 2: Tiled Input Implementation

- [ ] **Step 1: Create `src/paper11_geofm/tiled_inputs.py`**

Implement:

- `PHASE14_CLAIM_BOUNDARY`;
- `TiledVariantInput` dataclass with Phase 4-compatible attributes plus `tile_id`, `tile_index_csv`, and `claim_boundary`;
- `load_tiled_variant_input(phase2_output_dir, tile_index_csv, tile_id, variant_id="B1", allow_suitability_reward_contract=False)`;
- `run_phase14_tiled_smoke(...)`;
- `write_phase14_tiled_smoke_summary(summary, output_dir)`.

Rules:

- read `phase13_tile_index.csv` and parse `block_ids` as semicolon-separated IDs;
- use existing `load_variant_input(...)`;
- subset the variant matrix in tile order;
- reject `base_plus_suitability_reward` variants unless `allow_suitability_reward_contract=True`;
- use `Phase4InputContractEnv` for one reset/step;
- write `phase14_tiled_smoke_summary.json`.

- [ ] **Step 2: Run loader test**

Run:

```powershell
python -m pytest tests\test_phase14_tiled_smoke.py::test_phase14_loads_tiled_b1_variant_subset_in_tile_order -q
```

Expected: pass.

## Task 3: Smoke, Writer, and CLI Tests

- [ ] **Step 1: Add tests**

Add tests that:

- `run_phase14_tiled_smoke(...)` returns B1 summary with observation shape `246` and step reward `0.0`;
- B3 is rejected by default with a `suitability reward variants are disabled` error;
- writer creates `phase14_tiled_smoke_summary.json`;
- CLI prints `Tile: tile_r000_c000`, `Variant: B1`, `Rows: 3`, and the output path.

- [ ] **Step 2: Create CLI after CLI test fails**

Create `experiments/phase14_tiled_smoke_env/run_phase14_tiled_smoke.py` with flags:

- `--phase2-output-dir`;
- `--tile-index-csv`;
- `--tile-id`;
- `--variant` default `B1`;
- `--output-dir`;
- `--max-steps`.

Return `1` for `FileNotFoundError` or `ValueError`.

- [ ] **Step 3: Run focused tests**

Run:

```powershell
python -m pytest tests\test_phase14_tiled_smoke.py -q
```

Expected: all Phase 14 tests pass.

## Task 4: Docs, Real Run, Verification, Integration

- [ ] **Step 1: Update docs and manifest**

Add Phase 14 to README, reproduction guide, and file manifest.

- [ ] **Step 2: Run real Phase 14**

Run:

```powershell
python experiments\phase14_tiled_smoke_env\run_phase14_tiled_smoke.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --tile-id tile_r003_c003 --variant B1 --output-dir experiments\phase14_tiled_smoke_env\outputs\real_bishan_largest_tile
```

Expected:

- tile: `tile_r003_c003`;
- variant: `B1`;
- rows: `2234`;
- features: `81`;
- observation shape: `180957`;
- step reward: `0.0`.

- [ ] **Step 3: Full verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests -q
git diff --check
```

- [ ] **Step 4: Commit, merge, push**

Commit with message `Add Phase 14 tiled smoke env`, push feature branch, fast-forward merge to `main`, rerun real Phase 14 and full verification on `main`, push `main`, and delete the local feature branch.

---

## Self-Review

- Spec coverage: covers tiled loader, env smoke, reward-variant rejection, CLI, docs, real run, and integration.
- Placeholder scan: no placeholder steps remain.
- Type consistency: function names, paths, output file names, and CLI flags match the design.
