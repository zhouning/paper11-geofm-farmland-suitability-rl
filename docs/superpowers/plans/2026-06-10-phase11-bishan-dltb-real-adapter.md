# Phase 11 Bishan DLTB Real-Data Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real Bishan DLTB adapter that converts `DLTB_with_slope.gpkg` into Phase 2-compatible mapping and attributes CSV files.

**Architecture:** Add `paper11_geofm.dltb_adapter` to read a DLTB GeoPackage with geopandas, assign polygon centroids to the Bishan AlphaEarth grid from metadata, derive 17 explicit planning features and weak labels, and write CSV/JSON artifacts. Add a CLI under `experiments/phase11_bishan_dltb_real/` and document the local real-data workflow through Phase 2, Phase 9, and Phase 10.

**Tech Stack:** Python, pandas, geopandas, shapely, CSV/JSON artifacts, pytest.

---

## File Structure

- Create `src/paper11_geofm/dltb_adapter.py`: Phase 11 claim boundary, metadata reader, GeoPackage reader, centroid-to-grid assignment, feature/label derivation, artifact writer.
- Create `experiments/phase11_bishan_dltb_real/run_phase11_bishan_dltb_adapter.py`: CLI runner.
- Create `tests/test_phase11_bishan_dltb_adapter.py`: synthetic GeoPackage tests for assignment, feature derivation, writer, and CLI.
- Modify `README.md`: add Phase 11 local real-data command path.
- Modify `reproducibility/REPRODUCTION_GUIDE.md`: add Phase 11 real-data workflow and expected local result.
- Modify `reproducibility/DATA_MANIFEST.md`: document the external Bishan DLTB source path and why it is not committed.
- Modify `reproducibility/FILE_MANIFEST.tsv`: add Phase 11 design, plan, module, CLI, and tests.

## Task 1: Adapter Contract Test

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase11_bishan_dltb_adapter.py`:

```python
import csv
import importlib.util
import json
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _metadata_path(tmp_path: Path) -> Path:
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "bbox": [0.0, 0.0, 2.0, 2.0],
                "grid_shape": [2, 2],
                "scale_m": 500,
                "embedding_dim": 64,
            }
        ),
        encoding="utf-8",
    )
    return path


def _tiny_dltb_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_dltb.gpkg"
    gdf = gpd.GeoDataFrame(
        [
            {
                "BSM": 1,
                "DLBM": "011",
                "DLMC": "水田",
                "TBMJ": 10000.0,
                "category": "Farmland",
                "slope_mean": 5.0,
                "slope_max": 8.0,
                "slope_pixel_count": 4,
                "geometry": Polygon(
                    [(0.1, 1.6), (0.3, 1.6), (0.3, 1.8), (0.1, 1.8)]
                ),
            },
            {
                "BSM": 2,
                "DLBM": "031",
                "DLMC": "有林地",
                "TBMJ": 20000.0,
                "category": "Forest",
                "slope_mean": 18.0,
                "slope_max": 25.0,
                "slope_pixel_count": 7,
                "geometry": Polygon(
                    [(1.5, 0.1), (1.7, 0.1), (1.7, 0.3), (1.5, 0.3)]
                ),
            },
            {
                "BSM": 3,
                "DLBM": "023",
                "DLMC": "其他园地",
                "TBMJ": 15000.0,
                "category": "Orchard",
                "slope_mean": 4.0,
                "slope_max": 6.0,
                "slope_pixel_count": 3,
                "geometry": Polygon(
                    [(0.6, 0.6), (0.8, 0.6), (0.8, 0.8), (0.6, 0.8)]
                ),
            },
        ],
        crs="EPSG:4326",
    )
    gdf.to_file(path, layer="DLTB", driver="GPKG")
    return path


def test_phase11_builds_mapping_attributes_and_summary(tmp_path):
    from paper11_geofm.dltb_adapter import (
        PHASE11_CLAIM_BOUNDARY,
        build_bishan_dltb_phase2_inputs,
    )

    payload = build_bishan_dltb_phase2_inputs(
        _tiny_dltb_path(tmp_path),
        _metadata_path(tmp_path),
    )

    mapping_rows = payload["mapping_rows"]
    attribute_rows = payload["attribute_rows"]
    summary = payload["summary"]

    assert len(mapping_rows) == 3
    assert mapping_rows[0] == {
        "block_id": "dltb_1",
        "row": 0,
        "col": 0,
        "weight": 1.0,
    }
    assert mapping_rows[1]["row"] == 1
    assert mapping_rows[1]["col"] == 1
    assert attribute_rows[0]["current_farmland_label"] == 1
    assert attribute_rows[0]["low_slope_farmland_label"] == 1
    assert attribute_rows[1]["current_farmland_label"] == 0
    assert attribute_rows[1]["explicit_feature_15"] == 1.0
    assert attribute_rows[2]["farmland_or_orchard_label"] == 1
    assert all(
        f"explicit_feature_{idx:02d}" in attribute_rows[0]
        for idx in range(17)
    )
    assert summary["rows_exported"] == 3
    assert summary["label_positive_counts"]["current_farmland_label"] == 1
    assert summary["claim_boundary"] == PHASE11_CLAIM_BOUNDARY
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```powershell
python -m pytest tests\test_phase11_bishan_dltb_adapter.py::test_phase11_builds_mapping_attributes_and_summary -q
```

Expected result: fail with `ModuleNotFoundError: No module named 'paper11_geofm.dltb_adapter'`.

## Task 2: Adapter Builder Implementation

- [ ] **Step 1: Create `src/paper11_geofm/dltb_adapter.py`**

Implement:

```python
from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


PHASE11_CLAIM_BOUNDARY = (
    "Phase 11 builds real Bishan DLTB-derived Phase 2 inputs; "
    "centroid-to-grid assignment is an alignment adapter, not final "
    "parcel-accurate GeoFM evidence, and this phase does not train or "
    "evaluate a DRL policy."
)
```

Required public functions:

- `build_bishan_dltb_phase2_inputs(dltb_path, metadata_path, max_blocks=None)`;
- `write_bishan_dltb_phase2_inputs(payload, output_dir)`.

Required helper behavior:

- `_read_metadata(path)` validates `bbox` and `grid_shape`;
- `_load_dltb(dltb_path, bbox)` imports `geopandas` inside the helper and calls `geopandas.read_file(dltb_path, bbox=tuple(bbox))`;
- `_assign_row_col(centroid_x, centroid_y, bbox, grid_shape)` uses north-to-south row indexing and clips to valid grid bounds;
- `_build_mapping_row(block_id, row, col)` returns `block_id,row,col,weight`;
- `_build_attribute_row(record)` returns 17 explicit features, three weak labels, split, source fields, area, and slope fields;
- `_split_for_block(block_id)` deterministically assigns `train`, `validation`, or `test` from a stable checksum;
- summary records bbox, grid shape, DLTB path, metadata path, rows read, rows exported, category counts, label positive counts, slope summary, and claim boundary.

- [ ] **Step 2: Run the focused test**

Run:

```powershell
python -m pytest tests\test_phase11_bishan_dltb_adapter.py::test_phase11_builds_mapping_attributes_and_summary -q
```

Expected result: pass.

## Task 3: Writer and Max-Block Tests

- [ ] **Step 1: Add writer and max-block tests**

Append:

```python
def test_phase11_max_blocks_caps_rows_deterministically(tmp_path):
    from paper11_geofm.dltb_adapter import build_bishan_dltb_phase2_inputs

    payload = build_bishan_dltb_phase2_inputs(
        _tiny_dltb_path(tmp_path),
        _metadata_path(tmp_path),
        max_blocks=2,
    )

    assert [row["block_id"] for row in payload["mapping_rows"]] == [
        "dltb_1",
        "dltb_2",
    ]
    assert payload["summary"]["rows_exported"] == 2


def test_phase11_writes_phase2_input_csvs_and_summary(tmp_path):
    from paper11_geofm.dltb_adapter import (
        build_bishan_dltb_phase2_inputs,
        write_bishan_dltb_phase2_inputs,
    )

    payload = build_bishan_dltb_phase2_inputs(
        _tiny_dltb_path(tmp_path),
        _metadata_path(tmp_path),
    )

    paths = write_bishan_dltb_phase2_inputs(payload, tmp_path / "outputs")

    assert paths["mapping_csv"].name == "block_pixel_mapping.csv"
    assert paths["attributes_csv"].name == "block_attributes.csv"
    assert paths["summary"].name == "phase11_bishan_dltb_adapter_summary.json"
    with paths["mapping_csv"].open("r", encoding="utf-8", newline="") as handle:
        mapping_records = list(csv.DictReader(handle))
    with paths["attributes_csv"].open("r", encoding="utf-8", newline="") as handle:
        attribute_records = list(csv.DictReader(handle))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert mapping_records[0]["block_id"] == "dltb_1"
    assert attribute_records[0]["current_farmland_label"] == "1"
    assert summary["rows_exported"] == 3
```

- [ ] **Step 2: Run the test file**

Run:

```powershell
python -m pytest tests\test_phase11_bishan_dltb_adapter.py -q
```

Expected result: all current Phase 11 tests pass.

## Task 4: CLI Runner

- [ ] **Step 1: Add CLI test**

Append:

```python
def test_phase11_cli_writes_adapter_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase11_bishan_dltb_real"
        / "run_phase11_bishan_dltb_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("phase11_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--dltb-path",
            str(_tiny_dltb_path(tmp_path)),
            "--metadata-path",
            str(_metadata_path(tmp_path)),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "Rows exported: 3" in stdout
    assert "block_pixel_mapping.csv" in stdout
    assert "block_attributes.csv" in stdout
    assert "Claim boundary: Phase 11 builds real Bishan DLTB-derived" in stdout
```

- [ ] **Step 2: Run the CLI test and confirm it fails before the file exists**

Run:

```powershell
python -m pytest tests\test_phase11_bishan_dltb_adapter.py::test_phase11_cli_writes_adapter_outputs -q
```

Expected result: fail because `experiments/phase11_bishan_dltb_real/run_phase11_bishan_dltb_adapter.py` does not exist.

- [ ] **Step 3: Create the CLI**

Create `experiments/phase11_bishan_dltb_real/run_phase11_bishan_dltb_adapter.py` with flags:

- `--dltb-path`;
- `--metadata-path`;
- `--output-dir`;
- `--max-blocks`.

The CLI should call the builder and writer, print rows exported, category counts, label positive counts, output paths, and `PHASE11_CLAIM_BOUNDARY`, and return `1` for `FileNotFoundError`, `ValueError`, or `ImportError`.

- [ ] **Step 4: Run the CLI test**

Run:

```powershell
python -m pytest tests\test_phase11_bishan_dltb_adapter.py::test_phase11_cli_writes_adapter_outputs -q
```

Expected result: pass.

## Task 5: Documentation and Manifest

- [ ] **Step 1: Update README**

Add `experiments/phase11_bishan_dltb_real/` to the layout and add a Phase 11 local real-data command block.

- [ ] **Step 2: Update reproduction guide**

Add a Phase 11 section after Phase 10 with the adapter command and the Phase 2/9/10 real-data chain.

- [ ] **Step 3: Update data manifest**

Document `D:\test\dem_slope_analysis\output\DLTB_with_slope.gpkg` as a local external real-data source that is not committed due to size and provenance.

- [ ] **Step 4: Update file manifest**

Add rows for the Phase 11 design, plan, module, CLI, and test file.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests\test_phase11_bishan_dltb_adapter.py -q
```

Expected result: all Phase 11 tests pass.

## Task 6: Real Bishan Workflow, Verification, Commit, Merge

- [ ] **Step 1: Run the real local adapter**

Run:

```powershell
python experiments\phase11_bishan_dltb_real\run_phase11_bishan_dltb_adapter.py --dltb-path D:\test\dem_slope_analysis\output\DLTB_with_slope.gpkg --output-dir experiments\phase11_bishan_dltb_real\outputs\adapter
```

Expected result: adapter writes mapping CSV, attributes CSV, and summary JSON with tens of thousands of exported real DLTB blocks.

- [ ] **Step 2: Run real Phase 2, Phase 9, and Phase 10**

Run:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv experiments\phase11_bishan_dltb_real\outputs\adapter\block_pixel_mapping.csv --attributes-csv experiments\phase11_bishan_dltb_real\outputs\adapter\block_attributes.csv --output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real
python experiments\phase9_proxy_validation\run_phase9_proxy_validation.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase11_bishan_dltb_real\outputs\phase9_real --label-columns current_farmland_label,low_slope_farmland_label,farmland_or_orchard_label
python experiments\phase10_reward_readiness\run_phase10_reward_readiness.py --phase9-report experiments\phase11_bishan_dltb_real\outputs\phase9_real\phase9_proxy_validation_report.json --output-dir experiments\phase11_bishan_dltb_real\outputs\phase10_real --required-labels current_farmland_label,low_slope_farmland_label,farmland_or_orchard_label
```

Expected result: all commands complete and Phase 10 reports whether real DLTB-derived weak labels permit later suitability-reward smoke experiments.

- [ ] **Step 3: Run full verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests
git diff --check
```

Expected result: smoke check passes, all tests pass, and diff check reports no whitespace errors.

- [ ] **Step 4: Commit implementation**

Run:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\DATA_MANIFEST.md reproducibility\FILE_MANIFEST.tsv src\paper11_geofm\dltb_adapter.py experiments\phase11_bishan_dltb_real\run_phase11_bishan_dltb_adapter.py tests\test_phase11_bishan_dltb_adapter.py docs\superpowers\plans\2026-06-10-phase11-bishan-dltb-real-adapter.md
git commit -m "Add Phase 11 Bishan DLTB real-data adapter"
```

- [ ] **Step 5: Integrate**

Push the feature branch, fast-forward merge it to `main`, rerun the real local workflow plus full verification on `main`, push `main`, and delete the local feature branch after `main` is synchronized with `origin/main`.

---

## Self-Review

- Spec coverage: covers real DLTB input, metadata-grid alignment, output CSV contracts, explicit features, weak labels, summary artifact, CLI, docs, real workflow, and verification.
- Scope check: the plan does not copy large DLTB data into Git, does not claim parcel-accurate overlap, does not train a policy, and does not treat current farmland as stable or high-standard farmland.
- Type consistency: function names, artifact filenames, CLI flags, and claim boundary match the Phase 11 design spec.

