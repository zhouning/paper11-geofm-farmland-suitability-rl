# Phase 72A Temporal Label Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an audited Bishan-Dongxing annual ESRI LULC label package that aligns 2017-2024 product labels with temporally truncated AlphaEarth histories and emits stable prediction samples for the Phase 72B information-gain screen.

**Architecture:** Keep network acquisition separate from local scientific assembly. A tracked region contract defines source provenance, bounding boxes, years, and expected grids; a fetch script writes external labels and hashes; focused library modules validate independence and shape, assemble farmland-persistence samples without future leakage, and create a deterministic manual-review frame. The local runner consumes files only and returns `phase72a_label_inputs_ready` or an explicit blocker without synthesizing proxy labels.

**Tech Stack:** Python 3.11, NumPy, pandas, Earth Engine Python API, pytest, CSV/JSON/NPZ artifacts, existing Paper11 CLI and artifact conventions.

---

## Scope

This plan implements Phase 72A only. It does not train prediction models, fit
PCA, create GeoFM-STaR, optimize planning decisions, alter rewards, run PPO, or
revise `paper/submission/final/*`.

Master design:

`docs/superpowers/specs/2026-07-10-phase72-geofm-star-future-stability-planning-design.md`

## Files

- Create `experiments/phase72a_temporal_label_package/phase72a_regions.json`:
  tracked source and region contract without machine-specific asset paths.
- Create `src/paper11_geofm/phase72a_label_sources.py`: contract parser, source
  independence checks, annual file validation, and hashes.
- Create `src/paper11_geofm/phase72a_temporal_samples.py`: one- and two-year
  farmland-persistence sample assembly.
- Create `src/paper11_geofm/phase72a_review_frame.py`: deterministic blank
  manual/high-resolution review sample.
- Create `src/paper11_geofm/phase72a_temporal_label_package.py`: package gate,
  orchestration, and writer.
- Create `experiments/phase72a_temporal_label_package/fetch_phase72a_esri_lulc.py`:
  optional network acquisition with an injectable extractor.
- Create `experiments/phase72a_temporal_label_package/run_phase72a_temporal_label_package.py`:
  local-only runner.
- Create `tests/test_phase72a_temporal_label_package.py`: unit, integration,
  writer, and CLI tests.
- Create `paper/phase28_results/38_phase72a_temporal_label_package.md`: measured
  real-run note.
- Modify `paper/phase28_results/README.md` and
  `docs/superpowers/phase33_current_progress_handoff.md` after the real run.

Generated data remains ignored below
`experiments/phase72a_temporal_label_package/outputs/`.

## Contracts

### Region Contract

Create `phase72a_regions.json` with this content:

```json
{
  "source": {
    "source_id": "esri_global_lulc_10m_ts",
    "collection": "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS",
    "label_role": "independent_annual_product_label",
    "independent_from_dltb_slope_reward_geofm": true,
    "crop_class_code": 5,
    "scale_m": 500
  },
  "regions": [
    {
      "region_id": "bishan",
      "bbox": [106.02, 29.38, 106.33, 29.68],
      "years": [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
      "grid_shape": [67, 70],
      "embedding_dim": 64,
      "embedding_pattern": "bishan_emb_{year}.npy",
      "label_pattern": "bishan_lulc_{year}.npy"
    },
    {
      "region_id": "dongxing",
      "bbox": [104.97937034, 29.44405085, 105.42096421, 29.85040656],
      "years": [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
      "grid_shape": [91, 99],
      "embedding_dim": 64,
      "embedding_pattern": "dongxing_emb_{year}.npy",
      "label_pattern": "dongxing_lulc_{year}.npy"
    }
  ]
}
```

### Sample Index

`phase72a_temporal_sample_index.csv` contains:

```text
sample_index,region_id,unit_id,row,col,spatial_block_id,origin_year,
history_start_year,history_end_year,history_length,current_lulc_class,
target_year_1y,y_1y,target_year_2y,y_2y,y_continuous_2y,
label_source_id,label_source_role,label_confidence,claim_boundary
```

Unavailable two-year outcomes use blank CSV values and NPZ sentinel `-1`.

### Tensor Package

`phase72a_temporal_samples.npz` contains:

```text
embedding_history: float32 [n_samples, 8, 64]
history_mask: bool [n_samples, 8]
origin_year: int16 [n_samples]
current_lulc_class: int16 [n_samples]
y_1y: int8 [n_samples]
y_2y: int8 [n_samples], -1 when unavailable
y_continuous_2y: int8 [n_samples], -1 when unavailable
row: int16 [n_samples]
col: int16 [n_samples]
region_index: int8 [n_samples]
```

No embedding after `origin_year` may occur in a sample history.

### Status

- `label_inputs_not_ready`: a source, independence, year, file, hash, shape, or
  cohort check failed.
- `phase72a_label_inputs_ready`: both regions passed and nonempty one- and
  two-year cohorts were written.

---

### Task 1: Region Contract and Source Independence

**Files:**
- Create: `experiments/phase72a_temporal_label_package/phase72a_regions.json`
- Create: `src/paper11_geofm/phase72a_label_sources.py`
- Create: `tests/test_phase72a_temporal_label_package.py`

- [ ] **Step 1: Write failing contract tests**

Create the test file with imports, a `_region_config` fixture writer, and these
tests:

```python
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _region_config(path: Path, *, independent: bool = True) -> Path:
    payload = {
        "source": {
            "source_id": "esri_global_lulc_10m_ts",
            "collection": "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS",
            "label_role": "independent_annual_product_label",
            "independent_from_dltb_slope_reward_geofm": independent,
            "crop_class_code": 5,
            "scale_m": 500,
        },
        "regions": [{
            "region_id": "alpha",
            "bbox": [100.0, 20.0, 101.0, 21.0],
            "years": [2017, 2018, 2019, 2020],
            "grid_shape": [2, 3],
            "embedding_dim": 2,
            "embedding_pattern": "alpha_emb_{year}.npy",
            "label_pattern": "alpha_lulc_{year}.npy",
        }],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase72a_region_contract_loads_independent_source(tmp_path):
    from paper11_geofm.phase72a_label_sources import load_phase72a_region_contract

    contract = load_phase72a_region_contract(_region_config(tmp_path / "regions.json"))
    assert contract.source_id == "esri_global_lulc_10m_ts"
    assert contract.crop_class_code == 5
    assert [region.region_id for region in contract.regions] == ["alpha"]
    assert contract.regions[0].grid_shape == (2, 3)


def test_phase72a_region_contract_rejects_nonindependent_source(tmp_path):
    from paper11_geofm.phase72a_label_sources import load_phase72a_region_contract

    try:
        load_phase72a_region_contract(
            _region_config(tmp_path / "regions.json", independent=False)
        )
    except ValueError as exc:
        assert "independent" in str(exc).lower()
    else:
        raise AssertionError("Expected non-independent labels to be rejected")
```

- [ ] **Step 2: Verify RED**

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase72a_temporal_label_package.py -q --basetemp=D:\tmp\paper11_phase72a_task1_red -p no:cacheprovider
```

Expected: FAIL with `ModuleNotFoundError` for `phase72a_label_sources`.

- [ ] **Step 3: Implement the contract loader**

Create `phase72a_label_sources.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

PHASE72A_CLAIM_BOUNDARY = (
    "Phase 72A validates and aligns independent annual product labels with "
    "temporally truncated AlphaEarth histories. It does not train a prediction "
    "model, alter rewards, run planning, prove GeoFM value, or revise the formal manuscript."
)


@dataclass(frozen=True)
class Phase72ARegionSpec:
    region_id: str
    bbox: tuple[float, float, float, float]
    years: tuple[int, ...]
    grid_shape: tuple[int, int]
    embedding_dim: int
    embedding_pattern: str
    label_pattern: str


@dataclass(frozen=True)
class Phase72ARegionContract:
    source_id: str
    collection: str
    label_role: str
    crop_class_code: int
    scale_m: int
    regions: tuple[Phase72ARegionSpec, ...]


def load_phase72a_region_contract(path: Path | str) -> Phase72ARegionContract:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    source = payload.get("source", {})
    if source.get("independent_from_dltb_slope_reward_geofm") is not True:
        raise ValueError("Phase 72A label source must be independent")
    regions = []
    seen = set()
    for raw in payload.get("regions", []):
        region_id = str(raw["region_id"]).strip().lower()
        if not region_id or region_id in seen:
            raise ValueError(f"Phase 72A region_id must be nonblank and unique: {region_id}")
        seen.add(region_id)
        years = tuple(sorted({int(year) for year in raw["years"]}))
        if len(years) < 3:
            raise ValueError(f"Phase 72A region requires at least three years: {region_id}")
        regions.append(Phase72ARegionSpec(
            region_id=region_id,
            bbox=tuple(float(value) for value in raw["bbox"]),
            years=years,
            grid_shape=tuple(int(value) for value in raw["grid_shape"]),
            embedding_dim=int(raw["embedding_dim"]),
            embedding_pattern=str(raw["embedding_pattern"]),
            label_pattern=str(raw["label_pattern"]),
        ))
    if not regions:
        raise ValueError("Phase 72A region contract has no regions")
    return Phase72ARegionContract(
        source_id=str(source["source_id"]),
        collection=str(source["collection"]),
        label_role=str(source["label_role"]),
        crop_class_code=int(source["crop_class_code"]),
        scale_m=int(source["scale_m"]),
        regions=tuple(regions),
    )
```

Add the tracked JSON from the Contracts section.

- [ ] **Step 4: Verify GREEN**

Run the Task 1 command again. Expected: `2 passed`.

- [ ] **Step 5: Commit**

```powershell
git add experiments/phase72a_temporal_label_package/phase72a_regions.json src/paper11_geofm/phase72a_label_sources.py tests/test_phase72a_temporal_label_package.py
git commit -m "feat: add Phase 72A region contract"
```

---

### Task 2: Annual Asset Validation and Hash Manifest

**Files:**
- Modify: `src/paper11_geofm/phase72a_label_sources.py`
- Modify: `tests/test_phase72a_temporal_label_package.py`

- [ ] **Step 1: Write failing validation tests**

Append a fixture that writes four `2 x 3 x 2` embeddings and four `2 x 3`
labels, then assert complete files return `region_label_inputs_ready`, eight
manifest rows, and 64-character hashes. Add a second test where the 2020 label
shape is `2 x 2` and assert `label_inputs_not_ready` with a shape error.

```python
def _asset_dirs(tmp_path: Path, *, bad_label_shape: bool = False):
    embedding_dir = tmp_path / "embeddings"
    label_dir = tmp_path / "labels"
    embedding_dir.mkdir()
    label_dir.mkdir()
    for year in (2017, 2018, 2019, 2020):
        np.save(embedding_dir / f"alpha_emb_{year}.npy",
                np.full((2, 3, 2), float(year), dtype=np.float32))
        shape = (2, 2) if bad_label_shape and year == 2020 else (2, 3)
        np.save(label_dir / f"alpha_lulc_{year}.npy",
                np.full(shape, 5 if year < 2020 else 7, dtype=np.int32))
    return embedding_dir, label_dir


def test_phase72a_asset_audit_hashes_complete_aligned_years(tmp_path):
    from paper11_geofm.phase72a_label_sources import (
        audit_phase72a_region_assets,
        load_phase72a_region_contract,
    )

    contract = load_phase72a_region_contract(_region_config(tmp_path / "regions.json"))
    embedding_dir, label_dir = _asset_dirs(tmp_path)
    audit = audit_phase72a_region_assets(
        contract,
        contract.regions[0],
        embedding_dir=embedding_dir,
        label_dir=label_dir,
    )
    assert audit["status"] == "region_label_inputs_ready"
    assert audit["years_ready"] == [2017, 2018, 2019, 2020]
    assert len(audit["file_manifest_rows"]) == 8
    assert all(len(row["sha256"]) == 64 for row in audit["file_manifest_rows"])


def test_phase72a_asset_audit_blocks_shape_mismatch(tmp_path):
    from paper11_geofm.phase72a_label_sources import (
        audit_phase72a_region_assets,
        load_phase72a_region_contract,
    )

    contract = load_phase72a_region_contract(_region_config(tmp_path / "regions.json"))
    embedding_dir, label_dir = _asset_dirs(tmp_path, bad_label_shape=True)
    audit = audit_phase72a_region_assets(
        contract,
        contract.regions[0],
        embedding_dir=embedding_dir,
        label_dir=label_dir,
    )
    assert audit["status"] == "label_inputs_not_ready"
    assert "shape" in " ".join(audit["errors"]).lower()
```

- [ ] **Step 2: Verify RED**

Run the full test file with basetemp `D:\tmp\paper11_phase72a_task2_red`.
Expected: missing `audit_phase72a_region_assets`.

- [ ] **Step 3: Implement validation**

Add `_sha256(path)` using 1 MiB chunks and:

```python
import hashlib
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_phase72a_region_assets(
    contract: Phase72ARegionContract,
    region: Phase72ARegionSpec,
    *,
    embedding_dir: Path | str,
    label_dir: Path | str,
) -> dict[str, object]:
    embedding_dir = Path(embedding_dir)
    label_dir = Path(label_dir)
    errors: list[str] = []
    rows: list[dict[str, object]] = []
    years_ready: list[int] = []
    for year in region.years:
        year_ok = True
        assets = {
            "embedding": embedding_dir / region.embedding_pattern.format(year=year),
            "label": label_dir / region.label_pattern.format(year=year),
        }
        for asset_type, path in assets.items():
            if not path.exists():
                errors.append(
                    f"missing {asset_type} for {region.region_id} {year}: {path}"
                )
                year_ok = False
                continue
            array = np.load(path, mmap_mode="r")
            expected_shape = (
                (*region.grid_shape, region.embedding_dim)
                if asset_type == "embedding"
                else region.grid_shape
            )
            if tuple(array.shape) != tuple(expected_shape):
                errors.append(
                    f"{asset_type} shape mismatch for {region.region_id} {year}: "
                    f"expected {expected_shape}, got {tuple(array.shape)}"
                )
                year_ok = False
            rows.append(
                {
                    "region_id": region.region_id,
                    "year": int(year),
                    "asset_type": asset_type,
                    "source_id": (
                        contract.source_id
                        if asset_type == "label"
                        else "alphaearth_annual"
                    ),
                    "path": str(path),
                    "shape": "x".join(str(value) for value in array.shape),
                    "dtype": str(array.dtype),
                    "sha256": _sha256(path),
                    "independent_label": asset_type == "label",
                }
            )
        if year_ok:
            years_ready.append(int(year))
    return {
        "region_id": region.region_id,
        "status": (
            "region_label_inputs_ready" if not errors else "label_inputs_not_ready"
        ),
        "years_ready": years_ready,
        "errors": errors,
        "file_manifest_rows": rows,
        "claim_boundary": PHASE72A_CLAIM_BOUNDARY,
    }
```

- [ ] **Step 4: Verify GREEN**

Expected cumulative result: `4 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src/paper11_geofm/phase72a_label_sources.py tests/test_phase72a_temporal_label_package.py
git commit -m "feat: validate Phase 72A annual assets"
```

---

### Task 3: Leakage-Free Persistence Samples

**Files:**
- Create: `src/paper11_geofm/phase72a_temporal_samples.py`
- Modify: `tests/test_phase72a_temporal_label_package.py`

- [ ] **Step 1: Write failing temporal tests**

Use a `1 x 2` four-year fixture. Assert that a 2017 sample contains only the
2017 embedding, its history mask is `[True, False, False, False]`, and its
one-year/two-year/continuous labels match the 2018/2019 product labels. Assert a
2019 sample has a valid 2020 one-year label but blank CSV and `-1` tensor values
for the unavailable two-year target.

```python
def test_phase72a_samples_are_temporally_truncated_and_build_endpoints():
    from paper11_geofm.phase72a_label_sources import Phase72ARegionSpec
    from paper11_geofm.phase72a_temporal_samples import build_phase72a_temporal_samples

    region = Phase72ARegionSpec(
        "alpha", (100.0, 20.0, 101.0, 21.0), (2017, 2018, 2019, 2020),
        (1, 2), 2, "alpha_emb_{year}.npy", "alpha_lulc_{year}.npy"
    )
    embeddings = {
        year: np.full((1, 2, 2), float(year), dtype=np.float32)
        for year in region.years
    }
    labels = {
        2017: np.array([[5, 7]], dtype=np.int32),
        2018: np.array([[5, 5]], dtype=np.int32),
        2019: np.array([[7, 5]], dtype=np.int32),
        2020: np.array([[5, 7]], dtype=np.int32),
    }
    samples = build_phase72a_temporal_samples(
        region,
        embeddings=embeddings,
        labels=labels,
        crop_class_code=5,
        source_id="esri_global_lulc_10m_ts",
        source_role="independent_annual_product_label",
        max_history_years=4,
        spatial_block_size=2,
    )
    first = next(
        row for row in samples["sample_rows"]
        if row["unit_id"] == "r0000_c0000" and row["origin_year"] == 2017
    )
    index = int(first["sample_index"])
    assert (first["y_1y"], first["y_2y"], first["y_continuous_2y"]) == (1, 0, 0)
    assert samples["tensors"]["history_mask"][index].tolist() == [True, False, False, False]
    assert samples["tensors"]["embedding_history"][index, 0].tolist() == [2017.0, 2017.0]
    assert float(samples["tensors"]["embedding_history"][index, 1:].sum()) == 0.0


def test_phase72a_samples_mark_unavailable_two_year_target():
    from paper11_geofm.phase72a_label_sources import Phase72ARegionSpec
    from paper11_geofm.phase72a_temporal_samples import build_phase72a_temporal_samples

    region = Phase72ARegionSpec(
        "alpha", (100.0, 20.0, 101.0, 21.0), (2018, 2019, 2020),
        (1, 1), 1, "alpha_emb_{year}.npy", "alpha_lulc_{year}.npy"
    )
    samples = build_phase72a_temporal_samples(
        region,
        embeddings={
            year: np.array([[[float(year)]]], dtype=np.float32)
            for year in region.years
        },
        labels={year: np.array([[5]], dtype=np.int32) for year in region.years},
        crop_class_code=5,
        source_id="esri_global_lulc_10m_ts",
        source_role="independent_annual_product_label",
        max_history_years=3,
        spatial_block_size=1,
    )
    latest = next(
        row for row in samples["sample_rows"] if row["origin_year"] == 2019
    )
    index = int(latest["sample_index"])
    assert latest["y_1y"] == 1
    assert latest["y_2y"] == ""
    assert samples["tensors"]["y_2y"][index] == -1
```

- [ ] **Step 2: Verify RED**

Run with basetemp `D:\tmp\paper11_phase72a_task3_red`. Expected: missing
`phase72a_temporal_samples`.

- [ ] **Step 3: Implement sample assembly**

Create:

```python
from __future__ import annotations

from typing import Mapping

import numpy as np

from .phase72a_label_sources import PHASE72A_CLAIM_BOUNDARY, Phase72ARegionSpec


def build_phase72a_temporal_samples(
    region: Phase72ARegionSpec,
    *,
    embeddings: Mapping[int, np.ndarray],
    labels: Mapping[int, np.ndarray],
    crop_class_code: int,
    source_id: str,
    source_role: str,
    max_history_years: int,
    spatial_block_size: int,
) -> dict[str, object]:
    if max_history_years < len(region.years):
        raise ValueError("max_history_years must cover the contract history")
    rows = []
    histories = []
    masks = []
    y1_values = []
    y2_values = []
    continuous_values = []
    row_values = []
    col_values = []
    origin_values = []
    years = list(region.years)
    for origin_offset, origin_year in enumerate(years[:-1]):
        for grid_row, grid_col in np.argwhere(
            np.asarray(labels[origin_year]) == int(crop_class_code)
        ):
            history_years = years[: origin_offset + 1]
            history = np.zeros(
                (max_history_years, region.embedding_dim), dtype=np.float32
            )
            mask = np.zeros(max_history_years, dtype=bool)
            for history_offset, year in enumerate(history_years):
                history[history_offset] = embeddings[year][grid_row, grid_col]
                mask[history_offset] = True
            y1 = int(labels[origin_year + 1][grid_row, grid_col] == crop_class_code)
            has_2y = origin_offset + 2 < len(years)
            y2 = (
                int(labels[years[origin_offset + 2]][grid_row, grid_col] == crop_class_code)
                if has_2y else -1
            )
            continuous = int(y1 == 1 and y2 == 1) if has_2y else -1
            sample_index = len(rows)
            rows.append({
                "sample_index": sample_index,
                "region_id": region.region_id,
                "unit_id": f"r{int(grid_row):04d}_c{int(grid_col):04d}",
                "row": int(grid_row), "col": int(grid_col),
                "spatial_block_id": (
                    f"{region.region_id}_br{int(grid_row)//spatial_block_size:03d}_"
                    f"bc{int(grid_col)//spatial_block_size:03d}"
                ),
                "origin_year": int(origin_year),
                "history_start_year": int(history_years[0]),
                "history_end_year": int(origin_year),
                "history_length": len(history_years),
                "current_lulc_class": int(crop_class_code),
                "target_year_1y": int(origin_year + 1), "y_1y": y1,
                "target_year_2y": int(years[origin_offset + 2]) if has_2y else "",
                "y_2y": y2 if has_2y else "",
                "y_continuous_2y": continuous if has_2y else "",
                "label_source_id": source_id, "label_source_role": source_role,
                "label_confidence": "product_label",
                "claim_boundary": PHASE72A_CLAIM_BOUNDARY,
            })
            histories.append(history); masks.append(mask)
            y1_values.append(y1); y2_values.append(y2)
            continuous_values.append(continuous)
            row_values.append(int(grid_row)); col_values.append(int(grid_col))
            origin_values.append(int(origin_year))
    if not rows:
        raise ValueError(f"Phase 72A region has no farmland samples: {region.region_id}")
    tensors = {
        "embedding_history": np.stack(histories).astype(np.float32),
        "history_mask": np.stack(masks).astype(bool),
        "origin_year": np.asarray(origin_values, dtype=np.int16),
        "current_lulc_class": np.full(len(rows), crop_class_code, dtype=np.int16),
        "y_1y": np.asarray(y1_values, dtype=np.int8),
        "y_2y": np.asarray(y2_values, dtype=np.int8),
        "y_continuous_2y": np.asarray(continuous_values, dtype=np.int8),
        "row": np.asarray(row_values, dtype=np.int16),
        "col": np.asarray(col_values, dtype=np.int16),
    }
    return {"sample_rows": rows, "tensors": tensors}
```

- [ ] **Step 4: Verify GREEN**

Expected cumulative result: `6 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src/paper11_geofm/phase72a_temporal_samples.py tests/test_phase72a_temporal_label_package.py
git commit -m "feat: assemble Phase 72A temporal samples"
```

---

### Task 4: Review Frame, Package Gate, and Writer

**Files:**
- Create: `src/paper11_geofm/phase72a_review_frame.py`
- Create: `src/paper11_geofm/phase72a_temporal_label_package.py`
- Modify: `tests/test_phase72a_temporal_label_package.py`

- [ ] **Step 1: Write failing integration tests**

Build a fixture package and assert:

```python
assert package["phase72a_status"] == "phase72a_label_inputs_ready"
assert package["row_counts"]["sample_rows"] > 0
assert set(paths) == {
    "manifest_csv", "audit_csv", "sample_index_csv", "sample_tensors_npz",
    "review_frame_csv", "summary_csv", "package_json", "package_md"
}
```

Read the review CSV with `keep_default_na=False` and assert `review_label`,
`review_source`, `review_date`, and `review_confidence` exist and stay blank.
Delete one annual label in a second test and assert status
`label_inputs_not_ready` and no sample rows.

```python
def test_phase72a_package_writes_outputs_and_blank_review_labels(tmp_path):
    from paper11_geofm.phase72a_temporal_label_package import (
        build_phase72a_temporal_label_package,
        write_phase72a_temporal_label_package_artifacts,
    )

    embedding_dir, label_dir = _asset_dirs(tmp_path)
    package = build_phase72a_temporal_label_package(
        region_config=_region_config(tmp_path / "regions.json"),
        embedding_dirs={"alpha": embedding_dir},
        label_dirs={"alpha": label_dir},
        manual_review_per_stratum=2,
        spatial_block_size=2,
    )
    paths = write_phase72a_temporal_label_package_artifacts(
        package, tmp_path / "outputs"
    )
    assert package["phase72a_status"] == "phase72a_label_inputs_ready"
    assert package["row_counts"]["sample_rows"] > 0
    assert set(paths) == {
        "manifest_csv", "audit_csv", "sample_index_csv", "sample_tensors_npz",
        "review_frame_csv", "summary_csv", "package_json", "package_md"
    }
    review = pd.read_csv(paths["review_frame_csv"], keep_default_na=False)
    assert {"review_label", "review_source", "review_date", "review_confidence"}.issubset(review.columns)
    assert review["review_label"].eq("").all()
    tensors = np.load(paths["sample_tensors_npz"])
    assert tensors["embedding_history"].shape[0] == package["row_counts"]["sample_rows"]


def test_phase72a_package_blocks_samples_when_an_asset_is_missing(tmp_path):
    from paper11_geofm.phase72a_temporal_label_package import (
        build_phase72a_temporal_label_package,
    )

    embedding_dir, label_dir = _asset_dirs(tmp_path)
    (label_dir / "alpha_lulc_2020.npy").unlink()
    package = build_phase72a_temporal_label_package(
        region_config=_region_config(tmp_path / "regions.json"),
        embedding_dirs={"alpha": embedding_dir},
        label_dirs={"alpha": label_dir},
    )
    assert package["phase72a_status"] == "label_inputs_not_ready"
    assert package["sample_rows"] == []
```

- [ ] **Step 2: Verify RED**

Run with basetemp `D:\tmp\paper11_phase72a_task4_red`. Expected: missing review
and package modules.

- [ ] **Step 3: Implement deterministic review sampling**

Create:

```python
from __future__ import annotations

import hashlib


REVIEW_FIELDS = (
    "review_label", "review_source", "review_source_id",
    "review_date", "review_confidence", "reviewer_note",
)


def build_phase72a_review_frame(
    sample_rows: list[dict[str, object]],
    *,
    per_stratum: int = 20,
    seed: int = 72,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in sample_rows:
        transition = "persistent_crop" if int(row["y_1y"]) == 1 else "crop_conversion"
        key = (str(row["region_id"]), int(row["origin_year"]), transition)
        candidate = dict(row)
        candidate["transition_type"] = transition
        grouped.setdefault(key, []).append(candidate)
    output = []
    for key in sorted(grouped):
        ordered = sorted(
            grouped[key],
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['region_id']}:{row['unit_id']}:{row['origin_year']}".encode("utf-8")
            ).hexdigest(),
        )[: int(per_stratum)]
        for row in ordered:
            review = {
                field: row[field]
                for field in (
                    "sample_index", "region_id", "unit_id", "row", "col",
                    "spatial_block_id", "origin_year", "target_year_1y",
                    "transition_type", "label_source_id",
                )
            }
            review.update({field: "" for field in REVIEW_FIELDS})
            output.append(review)
    return output
```

- [ ] **Step 4: Implement package builder and writer**

Create:

```python
from __future__ import annotations

from collections.abc import Mapping
import csv
import json
from pathlib import Path

import numpy as np

from .phase72a_label_sources import (
    PHASE72A_CLAIM_BOUNDARY,
    audit_phase72a_region_assets,
    load_phase72a_region_contract,
)
from .phase72a_review_frame import build_phase72a_review_frame
from .phase72a_temporal_samples import build_phase72a_temporal_samples


def build_phase72a_temporal_label_package(
    *,
    region_config: Path | str,
    embedding_dirs: Mapping[str, Path | str],
    label_dirs: Mapping[str, Path | str],
    manual_review_per_stratum: int = 20,
    spatial_block_size: int = 8,
) -> dict[str, object]:
    contract = load_phase72a_region_contract(region_config)
    audits = []
    manifest_rows = []
    for region in contract.regions:
        if region.region_id not in embedding_dirs or region.region_id not in label_dirs:
            audits.append({
                "region_id": region.region_id,
                "status": "label_inputs_not_ready",
                "years_ready": [],
                "errors": [f"missing directory mapping for {region.region_id}"],
                "file_manifest_rows": [],
                "claim_boundary": PHASE72A_CLAIM_BOUNDARY,
            })
            continue
        audit = audit_phase72a_region_assets(
            contract, region,
            embedding_dir=embedding_dirs[region.region_id],
            label_dir=label_dirs[region.region_id],
        )
        audits.append(audit)
        manifest_rows.extend(audit["file_manifest_rows"])
    if any(audit["status"] != "region_label_inputs_ready" for audit in audits):
        return {
            "phase": "phase72a_temporal_label_package",
            "phase72a_status": "label_inputs_not_ready",
            "region_audits": audits,
            "manifest_rows": manifest_rows,
            "sample_rows": [], "review_rows": [], "tensors": {},
            "row_counts": {"regions": len(audits), "sample_rows": 0},
            "recommended_next_step": "Resolve Phase 72A label blockers before model work.",
            "claim_boundary": PHASE72A_CLAIM_BOUNDARY,
        }
    sample_rows = []
    tensor_parts: dict[str, list[np.ndarray]] = {}
    region_index_parts = []
    max_history = max(len(region.years) for region in contract.regions)
    for region_index, region in enumerate(contract.regions):
        embeddings = {
            year: np.load(Path(embedding_dirs[region.region_id]) / region.embedding_pattern.format(year=year))
            for year in region.years
        }
        labels = {
            year: np.load(Path(label_dirs[region.region_id]) / region.label_pattern.format(year=year))
            for year in region.years
        }
        built = build_phase72a_temporal_samples(
            region, embeddings=embeddings, labels=labels,
            crop_class_code=contract.crop_class_code,
            source_id=contract.source_id, source_role=contract.label_role,
            max_history_years=max_history, spatial_block_size=spatial_block_size,
        )
        offset = len(sample_rows)
        for row in built["sample_rows"]:
            adjusted = dict(row); adjusted["sample_index"] = int(row["sample_index"]) + offset
            sample_rows.append(adjusted)
        for key, value in built["tensors"].items():
            tensor_parts.setdefault(key, []).append(value)
        region_index_parts.append(
            np.full(len(built["sample_rows"]), region_index, dtype=np.int8)
        )
    tensors = {key: np.concatenate(parts, axis=0) for key, parts in tensor_parts.items()}
    tensors["region_index"] = np.concatenate(region_index_parts, axis=0)
    review_rows = build_phase72a_review_frame(
        sample_rows, per_stratum=manual_review_per_stratum
    )
    return {
        "phase": "phase72a_temporal_label_package",
        "phase72a_status": "phase72a_label_inputs_ready",
        "region_audits": audits, "manifest_rows": manifest_rows,
        "sample_rows": sample_rows, "review_rows": review_rows, "tensors": tensors,
        "row_counts": {
            "regions": len(audits), "manifest_rows": len(manifest_rows),
            "sample_rows": len(sample_rows), "review_rows": len(review_rows),
        },
        "recommended_next_step": "Design Phase 72B after checking class-support summaries.",
        "claim_boundary": PHASE72A_CLAIM_BOUNDARY,
    }


def write_phase72a_temporal_label_package_artifacts(
    package: Mapping[str, object], output_dir: Path | str
) -> dict[str, Path]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest_csv": output / "phase72a_label_manifest.csv",
        "audit_csv": output / "phase72a_region_label_audit.csv",
        "sample_index_csv": output / "phase72a_temporal_sample_index.csv",
        "sample_tensors_npz": output / "phase72a_temporal_samples.npz",
        "review_frame_csv": output / "phase72a_manual_review_frame.csv",
        "summary_csv": output / "phase72a_package_summary.csv",
        "package_json": output / "phase72a_temporal_label_package.json",
        "package_md": output / "phase72a_temporal_label_package.md",
    }
    # Use a local _write_rows helper that derives stable fieldnames from the
    # contract constants and emits headers for empty blocked tables.
    _write_rows(paths["manifest_csv"], package["manifest_rows"], MANIFEST_FIELDS)
    _write_rows(paths["audit_csv"], _audit_rows(package["region_audits"]), AUDIT_FIELDS)
    _write_rows(paths["sample_index_csv"], package["sample_rows"], SAMPLE_FIELDS)
    _write_rows(paths["review_frame_csv"], package["review_rows"], REVIEW_FRAME_FIELDS)
    _write_rows(paths["summary_csv"], _summary_rows(package), SUMMARY_FIELDS)
    np.savez_compressed(paths["sample_tensors_npz"], **package["tensors"])
    preview = {key: value for key, value in package.items() if key not in {"sample_rows", "review_rows", "tensors"}}
    paths["package_json"].write_text(json.dumps(preview, indent=2), encoding="utf-8")
    paths["package_md"].write_text(_render_markdown(package), encoding="utf-8")
    return paths
```

Require directories for every region, audit all files before loading full
arrays, stop assembly on any failed region, concatenate sample indexes with
correct offsets, add `region_index`, and build the blank review frame.

Write exactly:

```text
phase72a_label_manifest.csv
phase72a_region_label_audit.csv
phase72a_temporal_sample_index.csv
phase72a_temporal_samples.npz
phase72a_manual_review_frame.csv
phase72a_package_summary.csv
phase72a_temporal_label_package.json
phase72a_temporal_label_package.md
```

JSON contains counts and audits, not full samples/tensors. Markdown prints all
blockers. Summary includes counts and class rates by region/horizon.

Define `MANIFEST_FIELDS`, `AUDIT_FIELDS`, `SAMPLE_FIELDS`,
`REVIEW_FRAME_FIELDS`, and `SUMMARY_FIELDS` as tuples matching the Contracts
section. `_audit_rows` joins `years_ready` and errors with `|` for CSV. The
writer must write an empty NPZ for blocked status and still emit every stable
artifact filename.

- [ ] **Step 5: Verify GREEN**

Expected cumulative result: `8 passed`.

- [ ] **Step 6: Commit**

```powershell
git add src/paper11_geofm/phase72a_review_frame.py src/paper11_geofm/phase72a_temporal_label_package.py tests/test_phase72a_temporal_label_package.py
git commit -m "feat: write Phase 72A label package"
```

---

### Task 5: Network Fetcher and Local Runner

**Files:**
- Create: `experiments/phase72a_temporal_label_package/fetch_phase72a_esri_lulc.py`
- Create: `experiments/phase72a_temporal_label_package/run_phase72a_temporal_label_package.py`
- Modify: `tests/test_phase72a_temporal_label_package.py`

- [ ] **Step 1: Write failing fetcher and CLI tests**

Inject a fake extractor returning `2 x 3 int32` arrays and assert four labels and
a complete manifest are written. Invoke the local runner with:

```text
--embedding-dir alpha=<fixture path>
--label-dir alpha=<fixture path>
```

Assert exit code zero, ready status in stdout, and package JSON exists.

```python
def test_phase72a_fetcher_uses_injected_extractor(tmp_path):
    from experiments.phase72a_temporal_label_package.fetch_phase72a_esri_lulc import (
        fetch_phase72a_labels,
    )

    def fake_extractor(*, bbox, year, scale, collection):
        assert bbox == (100.0, 20.0, 101.0, 21.0)
        assert scale == 500
        assert "ESRI_Global-LULC" in collection
        return np.full((2, 3), 5 if year < 2020 else 7, dtype=np.int32)

    manifest = fetch_phase72a_labels(
        region_config=_region_config(tmp_path / "regions.json"),
        output_dir=tmp_path / "labels",
        regions=("alpha",),
        years=(2017, 2018, 2019, 2020),
        extractor=fake_extractor,
    )
    assert manifest["status"] == "complete"
    assert manifest["n_records"] == 4
    assert manifest["n_failures"] == 0
    assert (tmp_path / "labels" / "alpha_lulc_2020.npy").exists()


def test_phase72a_local_runner_succeeds_on_fixture(tmp_path):
    embedding_dir, label_dir = _asset_dirs(tmp_path)
    script = (
        ROOT / "experiments" / "phase72a_temporal_label_package"
        / "run_phase72a_temporal_label_package.py"
    )
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--region-config", str(_region_config(tmp_path / "regions.json")),
            "--embedding-dir", f"alpha={embedding_dir}",
            "--label-dir", f"alpha={label_dir}",
            "--output-dir", str(tmp_path / "outputs"),
            "--manual-review-per-stratum", "2",
            "--spatial-block-size", "2",
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "phase72a_label_inputs_ready" in result.stdout
    assert (tmp_path / "outputs" / "phase72a_temporal_label_package.json").exists()
```

- [ ] **Step 2: Verify RED**

Run with basetemp `D:\tmp\paper11_phase72a_task5_red`. Expected: both scripts
missing.

- [ ] **Step 3: Implement the fetcher**

Expose:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase72a_label_sources import (
    _sha256,
    load_phase72a_region_contract,
)


def _parse_years(raw: str) -> tuple[int, ...]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        elif part:
            values.append(int(part))
    return tuple(sorted(set(values)))


def _default_extractor(*, bbox, year, scale, collection):
    import ee
    region = ee.Geometry.Rectangle(list(bbox))
    image = (
        ee.ImageCollection(collection)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(region)
        .select(["b1"])
        .mosaic()
        .clip(region)
        .setDefaultProjection(ee.Projection("EPSG:4326").atScale(int(scale)))
    )
    result = image.sampleRectangle(region=region, defaultValue=0).getInfo()
    values = result.get("properties", {}).get("b1")
    if values is None:
        raise RuntimeError(f"ESRI LULC returned no b1 values for {year}")
    return np.asarray(values, dtype=np.int32)


def initialize_earth_engine(
    *, project: str | None = None, authenticate: bool = False
) -> None:
    import ee

    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception as exc:
        if not authenticate:
            raise RuntimeError(
                "Google Earth Engine is not initialized; authenticate first or use --authenticate"
            ) from exc
        ee.Authenticate()
        ee.Initialize(project=project) if project else ee.Initialize()


def fetch_phase72a_labels(
    *,
    region_config: Path | str,
    output_dir: Path | str,
    regions: tuple[str, ...],
    years: tuple[int, ...],
    extractor=None,
    overwrite: bool = False,
) -> dict[str, object]:
    contract = load_phase72a_region_contract(region_config)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    requested_regions = set(regions)
    requested_years = set(int(year) for year in years)
    extractor = extractor or _default_extractor
    records = []; failures = []
    for region in contract.regions:
        if region.region_id not in requested_regions:
            continue
        for year in region.years:
            if year not in requested_years:
                continue
            path = output / region.label_pattern.format(year=year)
            try:
                if path.exists() and not overwrite:
                    array = np.load(path)
                    status = "cached"
                else:
                    array = np.asarray(
                        extractor(
                            bbox=region.bbox, year=year, scale=contract.scale_m,
                            collection=contract.collection,
                        ),
                        dtype=np.int32,
                    )
                    if tuple(array.shape) != region.grid_shape:
                        raise ValueError(
                            f"label shape mismatch: expected {region.grid_shape}, got {tuple(array.shape)}"
                        )
                    np.save(path, array); status = "fetched"
                if tuple(array.shape) != region.grid_shape:
                    raise ValueError(
                        f"cached label shape mismatch: expected {region.grid_shape}, got {tuple(array.shape)}"
                    )
                records.append({
                    "region_id": region.region_id, "year": year,
                    "bbox": list(region.bbox), "scale_m": contract.scale_m,
                    "collection": contract.collection, "path": str(path),
                    "shape": list(array.shape), "sha256": _sha256(path),
                    "status": status,
                })
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append({
                    "region_id": region.region_id, "year": year,
                    "reason": str(exc),
                })
    manifest = {
        "status": "complete" if records and not failures else "partial" if records else "failed",
        "source_id": contract.source_id, "collection": contract.collection,
        "n_records": len(records), "n_failures": len(failures),
        "records": records, "failures": failures,
    }
    (output / "phase72a_lulc_fetch_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
```

Default extraction initializes Earth Engine, mosaics the configured ESRI annual
collection, clips the bbox, samples at contract scale, and returns an `int32`
2-D array. Validate expected grid shape before save. Write
`phase72a_lulc_fetch_manifest.json` with collection, bbox, year, scale, shape,
SHA256, and failures. Never create a fallback label.

CLI arguments are `--region-config`, `--output-dir`, `--regions`, `--years`,
`--overwrite`, `--project`, and `--authenticate`. Years support `2017-2024` and
comma-separated values. `main()` calls `initialize_earth_engine()` once before
using the default extractor; injected test extractors never require Earth Engine.

`main()` loads the parser, normalizes comma-separated region names, calls the
public function, prints status/counts/manifest path, and exits `0` only when
manifest status is `complete`; partial or failed network fetch exits `1`.

- [ ] **Step 4: Implement the local runner**

Parse repeated `--embedding-dir region=path` and `--label-dir region=path`
arguments. Add `--output-dir`, `--manual-review-per-stratum`, and
`--spatial-block-size`. Print status, counts, paths, next action, and claim
boundary. Return `0` for a completed blocked audit; malformed input returns `1`.

Use this mapping parser and main structure:

```python
def _parse_region_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected region=path, got {value}")
        region, raw_path = value.split("=", 1)
        region = region.strip().lower()
        if not region or region in result:
            raise ValueError(f"region mapping must be nonblank and unique: {region}")
        result[region] = Path(raw_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Paper11 Phase 72A label package")
    parser.add_argument("--region-config", type=Path, required=True)
    parser.add_argument("--embedding-dir", action="append", default=[], required=True)
    parser.add_argument("--label-dir", action="append", default=[], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manual-review-per-stratum", type=int, default=20)
    parser.add_argument("--spatial-block-size", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        package = build_phase72a_temporal_label_package(
            region_config=args.region_config,
            embedding_dirs=_parse_region_paths(args.embedding_dir),
            label_dirs=_parse_region_paths(args.label_dir),
            manual_review_per_stratum=args.manual_review_per_stratum,
            spatial_block_size=args.spatial_block_size,
        )
        paths = write_phase72a_temporal_label_package_artifacts(package, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr); return 1
    print(f"Phase 72A temporal label status: {package['phase72a_status']}")
    print(f"Row counts: {package['row_counts']}")
    for key, path in paths.items(): print(f"{key}: {path}")
    print(f"Recommended next step: {package['recommended_next_step']}")
    print(f"Claim boundary: {package['claim_boundary']}")
    return 0
```

- [ ] **Step 5: Verify GREEN**

Expected cumulative result: `10 passed`.

- [ ] **Step 6: Commit**

```powershell
git add experiments/phase72a_temporal_label_package/fetch_phase72a_esri_lulc.py experiments/phase72a_temporal_label_package/run_phase72a_temporal_label_package.py tests/test_phase72a_temporal_label_package.py
git commit -m "feat: add Phase 72A label runners"
```

---

### Task 6: Real Acquisition, Gate Evaluation, and Result Note

**Files:**
- Generate: `experiments/phase72a_temporal_label_package/outputs/esri_labels/*`
- Generate: `experiments/phase72a_temporal_label_package/outputs/bishan_dongxing_esri_2017_2024/*`
- Create: `paper/phase28_results/38_phase72a_temporal_label_package.md`
- Modify: `paper/phase28_results/README.md`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [ ] **Step 1: Verify tests before network use**

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase72a_temporal_label_package.py -q --basetemp=D:\tmp\paper11_phase72a_pre_real -p no:cacheprovider
```

Expected: `10 passed`.

- [ ] **Step 2: Fetch Dongxing labels**

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase72a_temporal_label_package\fetch_phase72a_esri_lulc.py --region-config experiments\phase72a_temporal_label_package\phase72a_regions.json --regions dongxing --years 2017-2024 --output-dir experiments\phase72a_temporal_label_package\outputs\esri_labels
```

Success evidence: status `complete`, 8 records, 0 failures, `91 x 99` shapes,
and eight hashes. If authentication, network, source, or shape blocks this, keep
the manifest and record `label_inputs_not_ready`. Do not resample, pad, or use
DLTB proxy labels without a new reviewed design.

- [ ] **Step 3: Run the dual-region local package**

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase72a_temporal_label_package\run_phase72a_temporal_label_package.py --region-config experiments\phase72a_temporal_label_package\phase72a_regions.json --embedding-dir bishan=data\bishan_alphaearth_sample --label-dir bishan=D:\test\paper58-geofm-world-model-rl\data\independent_change_labels\labels --embedding-dir dongxing=D:\test\dongxing_alphaearth --label-dir dongxing=experiments\phase72a_temporal_label_package\outputs\esri_labels --manual-review-per-stratum 20 --spatial-block-size 8 --output-dir experiments\phase72a_temporal_label_package\outputs\bishan_dongxing_esri_2017_2024
```

Ready evidence requires both regions, 32 annual manifest rows, nonzero samples,
both one-year classes in usable data, and nonzero two-year eligible rows.
Single-class region-year folds remain flagged and cannot train Phase 72B.

- [ ] **Step 4: Inspect measured artifacts**

```powershell
Get-Content -Raw experiments\phase72a_temporal_label_package\outputs\bishan_dongxing_esri_2017_2024\phase72a_temporal_label_package.json
Import-Csv experiments\phase72a_temporal_label_package\outputs\bishan_dongxing_esri_2017_2024\phase72a_package_summary.csv | Format-List
Import-Csv experiments\phase72a_temporal_label_package\outputs\bishan_dongxing_esri_2017_2024\phase72a_region_label_audit.csv | Format-Table -AutoSize
```

Confirm actual counts, rates, years, hashes, and blockers. CLI exit code alone
does not prove readiness.

- [ ] **Step 5: Write measured documentation**

Create `38_phase72a_temporal_label_package.md` with actual status, collection,
scale, region shapes/years, counts, class rates, blockers, exact commands, claim
boundary, and a Phase 72B recommendation only when ready. Add one README index
line and append the same measured state to the handoff. Do not modify the formal
manuscript.

- [ ] **Step 6: Verify and commit the measured result**

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase72a_temporal_label_package.py -q --basetemp=D:\tmp\paper11_phase72a_real_verify -p no:cacheprovider
D:\adk\.venv\Scripts\python.exe scripts\smoke_check.py
git diff --check
git diff --name-only HEAD -- paper/submission/final
```

Expected: tests and smoke pass, diff check is clean, and the final command emits
nothing.

```powershell
git add src/paper11_geofm/phase72a_*.py experiments/phase72a_temporal_label_package/*.py experiments/phase72a_temporal_label_package/phase72a_regions.json tests/test_phase72a_temporal_label_package.py paper/phase28_results/38_phase72a_temporal_label_package.md paper/phase28_results/README.md docs/superpowers/phase33_current_progress_handoff.md
git commit -m "docs: record Phase 72A temporal label result"
```

---

### Task 7: Final Verification and Phase 72B Gate

**Files:**
- Verify only unless measured documentation needs correction.

- [ ] **Step 1: Run adjacent regressions**

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase72a_temporal_label_package.py tests\test_phase68_external_independent_label_package.py tests\test_phase40_independent_label_gate.py tests\test_phase39_independent_label_audit.py -q --basetemp=D:\tmp\paper11_phase72a_adjacent -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 2: Run repository checks**

```powershell
D:\adk\.venv\Scripts\python.exe scripts\smoke_check.py
git diff --check
git status --short --branch
git log -6 --oneline --decorate
```

Expected: smoke passes, diff check is clean, and intended changes are committed.

- [ ] **Step 3: Enforce the transition gate**

Design Phase 72B only when the real package reports:

```text
phase72a_status == phase72a_label_inputs_ready
both regions passed
one-year samples > 0
two-year eligible rows > 0
both one-year classes occur
no independence, shape, year, or hash blocker
```

Otherwise remain in Phase 72A and resolve the measured label blocker without
starting model training.

## Self-Review

- Phase 72A master-spec coverage is complete: acquisition, provenance,
  independence, stable samples, review frame, and gate.
- Phase 72B-F are excluded and require separate specifications.
- Public names, statuses, sample fields, tensors, and CLI mapping syntax are
  consistent across tasks.
- Proxy/static DLTB labels are never outcomes; fixtures inject arrays only for
  tests; product labels remain distinct from manual truth and policy outcomes.
- Real result values must be measured; no success count is predeclared beyond
  input dimensions already verified locally.
