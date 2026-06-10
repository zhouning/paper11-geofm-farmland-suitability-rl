# Phase 13 Tiled Real Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Phase 13 tiled contract builder that turns real Bishan DLTB block mappings into tractable tile-level episode metadata.

**Architecture:** Add `paper11_geofm.tiled_contract` to read Phase 11 mapping CSV plus Phase 2 variant manifest, group blocks into configurable grid tiles, compute per-tile and per-variant observation dimensions, and write CSV/JSON artifacts. Add a CLI runner and reviewer documentation.

**Tech Stack:** Python standard library, CSV/JSON artifacts, pytest.

---

## File Structure

- Create `src/paper11_geofm/tiled_contract.py`: tile grouping, manifest feature counts, threshold decisions, CSV/JSON writer.
- Create `experiments/phase13_tiled_real_contract/run_phase13_tiled_real_contract.py`: CLI runner.
- Create `tests/test_phase13_tiled_contract.py`: synthetic mapping/manifest tests for tiling, threshold gates, writer output, and CLI.
- Modify `README.md`: add Phase 13 command and key entry point.
- Modify `reproducibility/REPRODUCTION_GUIDE.md`: add Phase 13 after Phase 12 and renumber later sections.
- Modify `reproducibility/FILE_MANIFEST.tsv`: add Phase 13 design, plan, module, CLI, and tests.

## Task 1: Tile Builder Contract Test

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase13_tiled_contract.py`:

```python
import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_mapping(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"block_id": "b1", "row": 0, "col": 0, "weight": 1.0},
        {"block_id": "b2", "row": 0, "col": 1, "weight": 1.0},
        {"block_id": "b3", "row": 3, "col": 0, "weight": 1.0},
        {"block_id": "b4", "row": 5, "col": 5, "weight": 1.0},
        {"block_id": "b5", "row": 6, "col": 5, "weight": 1.0},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["block_id", "row", "col", "weight"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_manifest(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    variants = {
        "B0": {
            "ready": True,
            "missing": [],
            "required_columns": ["explicit_feature_00", "explicit_feature_01"],
            "reward": "base_planning_reward",
            "feature_table": "variant_B0_features.csv",
            "state_groups": ["explicit_planning_features"],
        },
        "B1": {
            "ready": True,
            "missing": [],
            "required_columns": [
                "explicit_feature_00",
                "explicit_feature_01",
                "embedding_mean_00",
            ],
            "reward": "base_planning_reward",
            "feature_table": "variant_B1_features.csv",
            "state_groups": ["explicit_planning_features", "geofm_embedding"],
        },
        "B2": {
            "ready": True,
            "missing": [],
            "required_columns": [
                "explicit_feature_00",
                "explicit_feature_01",
                "suitability_proxy",
            ],
            "reward": "base_plus_suitability_reward",
            "feature_table": "variant_B2_features.csv",
            "state_groups": ["explicit_planning_features", "suitability_proxy"],
        },
        "B3": {
            "ready": True,
            "missing": [],
            "required_columns": [
                "explicit_feature_00",
                "explicit_feature_01",
                "embedding_mean_00",
                "suitability_proxy",
            ],
            "reward": "base_plus_suitability_reward",
            "feature_table": "variant_B3_features.csv",
            "state_groups": [
                "explicit_planning_features",
                "geofm_embedding",
                "suitability_proxy",
            ],
        },
    }
    path.write_text(json.dumps({"variants": variants}, indent=2), encoding="utf-8")
    return path
```

Add:

```python
def test_phase13_builds_tile_index_and_contract_summary(tmp_path):
    from paper11_geofm.tiled_contract import (
        PHASE13_CLAIM_BOUNDARY,
        build_phase13_tiled_contract,
    )

    report = build_phase13_tiled_contract(
        _write_mapping(tmp_path / "mapping.csv"),
        _write_manifest(tmp_path / "experiment_variants.json"),
        tile_rows=4,
        tile_cols=4,
        observation_threshold=20,
    )

    assert report["total_blocks"] == 5
    assert report["tile_count"] == 2
    assert report["block_count_summary"]["max"] == 3
    assert report["variants"]["B3"]["n_features"] == 4
    assert report["variants"]["B3"]["max_tile_observation_dimension"] == 15
    assert report["all_tiles_within_observation_threshold"] is True
    assert report["tiled_contract_ready"] is True
    assert report["tiles"][0]["tile_id"] == "tile_r000_c000"
    assert report["tiles"][0]["block_ids"] == ["b1", "b2", "b3"]
    assert report["claim_boundary"] == PHASE13_CLAIM_BOUNDARY
```

- [ ] **Step 2: Run and confirm failure**

Run:

```powershell
python -m pytest tests\test_phase13_tiled_contract.py::test_phase13_builds_tile_index_and_contract_summary -q
```

Expected: fail with `ModuleNotFoundError: No module named 'paper11_geofm.tiled_contract'`.

## Task 2: Tile Builder Implementation

- [ ] **Step 1: Create `src/paper11_geofm/tiled_contract.py`**

Implement:

```python
from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PHASE13_CLAIM_BOUNDARY = (
    "Phase 13 builds tiled real-data contract metadata only; it does not "
    "train, tune, evaluate, or compare a DRL policy and does not enable "
    "suitability reward."
)
DEFAULT_TILE_ROWS = 8
DEFAULT_TILE_COLS = 8
DEFAULT_OBSERVATION_THRESHOLD = 1_000_000
REQUIRED_VARIANTS = ("B0", "B1", "B2", "B3")
```

Required public functions:

- `build_phase13_tiled_contract(mapping_csv, variant_manifest_path, tile_rows=8, tile_cols=8, observation_threshold=1_000_000)`;
- `write_phase13_tiled_contract(report, output_dir)`.

Required behavior:

- validate positive `tile_rows`, `tile_cols`, and `observation_threshold`;
- require mapping CSV fields `block_id`, `row`, `col`;
- compute `tile_row = floor(row / tile_rows)` and `tile_col = floor(col / tile_cols)`;
- produce stable `tile_id = tile_r{tile_row:03d}_c{tile_col:03d}`;
- sort tiles by `tile_row`, then `tile_col`;
- sort block IDs within each tile by input order;
- compute min/max/mean block counts;
- read B0/B1/B2/B3 feature counts from `required_columns`;
- compute per-variant max tile observation dimensions;
- set `all_tiles_within_observation_threshold` and `tiled_contract_ready`;
- write CSV and JSON artifacts.

- [ ] **Step 2: Run the builder test**

Run:

```powershell
python -m pytest tests\test_phase13_tiled_contract.py::test_phase13_builds_tile_index_and_contract_summary -q
```

Expected: pass.

## Task 3: Writer, Threshold, and CLI Tests

- [ ] **Step 1: Add tests**

Add:

```python
def test_phase13_threshold_blocks_contract_when_tile_observation_is_too_large(tmp_path):
    from paper11_geofm.tiled_contract import build_phase13_tiled_contract

    report = build_phase13_tiled_contract(
        _write_mapping(tmp_path / "mapping.csv"),
        _write_manifest(tmp_path / "experiment_variants.json"),
        tile_rows=4,
        tile_cols=4,
        observation_threshold=14,
    )

    assert report["all_tiles_within_observation_threshold"] is False
    assert report["tiled_contract_ready"] is False
    assert "increase_tile_partitioning" in report["recommendation"]
```

```python
def test_phase13_writer_outputs_tile_csv_and_json(tmp_path):
    from paper11_geofm.tiled_contract import (
        build_phase13_tiled_contract,
        write_phase13_tiled_contract,
    )

    report = build_phase13_tiled_contract(
        _write_mapping(tmp_path / "mapping.csv"),
        _write_manifest(tmp_path / "experiment_variants.json"),
        tile_rows=4,
        tile_cols=4,
    )
    paths = write_phase13_tiled_contract(report, tmp_path / "outputs")

    assert paths["tile_index"].name == "phase13_tile_index.csv"
    assert paths["summary"].name == "phase13_tiled_real_contract.json"
    with paths["tile_index"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert rows[0]["tile_id"] == "tile_r000_c000"
    assert rows[0]["block_ids"] == "b1;b2;b3"
    assert summary["tile_count"] == 2
```

```python
def test_phase13_cli_writes_tiled_contract_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase13_tiled_real_contract"
        / "run_phase13_tiled_real_contract.py"
    )
    spec = importlib.util.spec_from_file_location("phase13_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--mapping-csv",
            str(_write_mapping(tmp_path / "mapping.csv")),
            "--variant-manifest",
            str(_write_manifest(tmp_path / "experiment_variants.json")),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--tile-rows",
            "4",
            "--tile-cols",
            "4",
            "--observation-threshold",
            "20",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Tiles: 2" in stdout
    assert "Tiled contract ready: True" in stdout
    assert "phase13_tile_index.csv" in stdout
```

- [ ] **Step 2: Run CLI test before CLI exists**

Run:

```powershell
python -m pytest tests\test_phase13_tiled_contract.py::test_phase13_cli_writes_tiled_contract_outputs -q
```

Expected: fail because the CLI file does not exist.

- [ ] **Step 3: Create CLI**

Create `experiments/phase13_tiled_real_contract/run_phase13_tiled_real_contract.py` with flags:

- `--mapping-csv`;
- `--variant-manifest`;
- `--output-dir`;
- `--tile-rows`;
- `--tile-cols`;
- `--observation-threshold`.

Print blocks, tiles, max blocks per tile, max B3 observation dimension, readiness, output paths, and claim boundary. Return `1` for `FileNotFoundError` or `ValueError`.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests\test_phase13_tiled_contract.py -q
```

Expected: all Phase 13 tests pass.

## Task 4: Documentation and Manifest

- [ ] **Step 1: Update README**

Add `experiments/phase13_tiled_real_contract/` to the layout, add a Phase 13 command after Phase 12, and add the runner to key entry points.

- [ ] **Step 2: Update reproduction guide**

Add Phase 13 after Phase 12 with real Bishan expected values and renumber later sections.

- [ ] **Step 3: Update file manifest**

Add rows for Phase 13 design, plan, module, CLI, and tests.

## Task 5: Real Run, Verification, Commit, Merge

- [ ] **Step 1: Run real Phase 13**

Run:

```powershell
python experiments\phase13_tiled_real_contract\run_phase13_tiled_real_contract.py --mapping-csv experiments\phase11_bishan_dltb_real\outputs\adapter\block_pixel_mapping.csv --variant-manifest experiments\phase11_bishan_dltb_real\outputs\phase2_real\experiment_variants.json --output-dir experiments\phase13_tiled_real_contract\outputs\real_bishan --tile-rows 8 --tile-cols 8
```

Expected:

- total blocks: `64984`;
- tiles: `54`;
- max blocks per tile: `2234`;
- B3 max tile observation dimension: `183191`;
- tiled contract ready: `True`.

- [ ] **Step 2: Run full verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests -q
git diff --check
```

Expected: smoke check passes, all tests pass, and diff check reports no whitespace errors.

- [ ] **Step 3: Commit implementation**

Run:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv src\paper11_geofm\tiled_contract.py experiments\phase13_tiled_real_contract\run_phase13_tiled_real_contract.py tests\test_phase13_tiled_contract.py docs\superpowers\plans\2026-06-10-phase13-tiled-real-contract.md
git commit -m "Add Phase 13 tiled real contract"
```

- [ ] **Step 4: Integrate**

Push the feature branch, fast-forward merge it to `main`, rerun real Phase 13 plus full verification on `main`, push `main`, and delete the local feature branch after `main` is synchronized with `origin/main`.

---

## Self-Review

- Spec coverage: covers tile builder, writer, CLI, docs, manifest, real run, and integration.
- Placeholder scan: no placeholder steps remain.
- Type consistency: filenames, function names, output keys, and CLI flags match the Phase 13 design.
