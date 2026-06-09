# Phase 2 Block GeoFM Feature Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight, deterministic Phase 2 feature-assembly pipeline that converts block-to-pixel mappings and AlphaEarth embeddings into B0/B1/B2/B3-ready block-level feature tables.

**Architecture:** Extend the existing `src/paper11_geofm` package with focused modules for mapping validation, block feature aggregation, and schema/readiness metadata. Add `experiments/phase2_block_geofm_features/run_phase2.py` as the executable entry point; its default path will derive a reproducible block mapping from the included Bishan sample so reviewers can run it without GIS dependencies or external data.

**Tech Stack:** Python standard library, NumPy, pytest, Git.

---

### Task 1: Plan Registration

**Files:**
- Create: `docs/superpowers/plans/2026-06-09-phase2-block-geofm-feature-assembly.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `scripts/smoke_check.py`
- Modify: `tests/test_repository_layout.py`

- [ ] **Step 1: Save this implementation plan**

Create `docs/superpowers/plans/2026-06-09-phase2-block-geofm-feature-assembly.md` with the exact plan content.

- [ ] **Step 2: Register the plan in the manifest**

Add this line to `reproducibility/FILE_MANIFEST.tsv` after the Phase 2 spec line:

```text
docs/superpowers/plans/2026-06-09-phase2-block-geofm-feature-assembly.md	plan	Implementation plan for the Phase 2 block-level GeoFM feature assembly pipeline.
```

- [ ] **Step 3: Add the plan to repository layout checks**

Add this path to `REQUIRED_PATHS` in `scripts/smoke_check.py` and to `required_paths` in `tests/test_repository_layout.py`:

```python
"docs/superpowers/plans/2026-06-09-phase2-block-geofm-feature-assembly.md",
```

- [ ] **Step 4: Verify plan-only registration**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests\test_repository_layout.py tests\test_smoke_check.py
```

Expected: both commands exit 0.

- [ ] **Step 5: Commit the plan checkpoint**

Run:

```powershell
git add docs\superpowers\plans\2026-06-09-phase2-block-geofm-feature-assembly.md reproducibility\FILE_MANIFEST.tsv scripts\smoke_check.py tests\test_repository_layout.py
git commit -m "Add Phase 2 block GeoFM feature assembly plan"
```

### Task 2: Mapping Validation Tests and Module

**Files:**
- Create: `tests/test_phase2_block_geofm.py`
- Create: `src/paper11_geofm/block_mapping.py`

- [ ] **Step 1: Write failing mapping tests**

Create `tests/test_phase2_block_geofm.py` with tests for mapping validation:

```python
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_validate_block_pixel_mapping_rejects_out_of_range_pixels():
    from paper11_geofm.block_mapping import validate_block_pixel_mapping

    rows = [
        {"block_id": "b0", "row": 0, "col": 0},
        {"block_id": "b1", "row": 67, "col": 0},
    ]

    with pytest.raises(ValueError, match="outside grid_shape"):
        validate_block_pixel_mapping(rows, (67, 70))


def test_validate_block_pixel_mapping_defaults_weights_and_preserves_order():
    from paper11_geofm.block_mapping import validate_block_pixel_mapping

    rows = [
        {"block_id": "b0", "row": 0, "col": 0},
        {"block_id": "b0", "row": 0, "col": 1, "weight": 2.5},
        {"block_id": "b1", "row": 1, "col": 0},
    ]

    mapping = validate_block_pixel_mapping(rows, (67, 70))

    assert [entry["block_id"] for entry in mapping] == ["b0", "b0", "b1"]
    assert [entry["weight"] for entry in mapping] == [1.0, 2.5, 1.0]
    assert all(isinstance(entry["row"], int) for entry in mapping)
    assert all(isinstance(entry["col"], int) for entry in mapping)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest tests\test_phase2_block_geofm.py::test_validate_block_pixel_mapping_rejects_out_of_range_pixels tests\test_phase2_block_geofm.py::test_validate_block_pixel_mapping_defaults_weights_and_preserves_order -v
```

Expected: FAIL because `paper11_geofm.block_mapping` does not exist.

- [ ] **Step 3: Implement mapping validation**

Create `src/paper11_geofm/block_mapping.py`:

```python
from __future__ import annotations

from collections.abc import Iterable, Mapping


BlockPixel = dict[str, str | int | float]


def validate_block_pixel_mapping(
    rows: Iterable[Mapping[str, object]],
    grid_shape: tuple[int, int],
) -> list[BlockPixel]:
    rows_count, cols_count = grid_shape
    if rows_count <= 0 or cols_count <= 0:
        raise ValueError(f"grid_shape must be positive, got {grid_shape}")

    validated: list[BlockPixel] = []
    for index, row in enumerate(rows):
        try:
            block_id = str(row["block_id"])
            pixel_row = int(row["row"])
            pixel_col = int(row["col"])
        except KeyError as exc:
            raise ValueError(f"mapping row {index} missing required column {exc.args[0]}") from exc

        weight = float(row.get("weight", 1.0))
        if not block_id:
            raise ValueError(f"mapping row {index} has empty block_id")
        if pixel_row < 0 or pixel_row >= rows_count or pixel_col < 0 or pixel_col >= cols_count:
            raise ValueError(
                f"mapping row {index} points outside grid_shape {grid_shape}: "
                f"row={pixel_row}, col={pixel_col}"
            )
        if weight <= 0:
            raise ValueError(f"mapping row {index} has non-positive weight {weight}")

        validated.append(
            {
                "block_id": block_id,
                "row": pixel_row,
                "col": pixel_col,
                "weight": weight,
            }
        )

    if not validated:
        raise ValueError("block pixel mapping must contain at least one row")
    return validated
```

- [ ] **Step 4: Run the mapping tests to verify they pass**

Run:

```powershell
python -m pytest tests\test_phase2_block_geofm.py::test_validate_block_pixel_mapping_rejects_out_of_range_pixels tests\test_phase2_block_geofm.py::test_validate_block_pixel_mapping_defaults_weights_and_preserves_order -v
```

Expected: PASS.

### Task 3: Block Feature Aggregation

**Files:**
- Modify: `tests/test_phase2_block_geofm.py`
- Create: `src/paper11_geofm/block_features.py`

- [ ] **Step 1: Write failing block feature tests**

Append tests that aggregate a small in-memory embedding grid and verify weighted means, bounds, dispersion, and 64 embedding columns:

```python
import numpy as np


def _tiny_embedding_grid():
    grid = np.zeros((2, 2, 64), dtype=np.float64)
    grid[0, 0, :] = 1.0
    grid[0, 1, :] = 3.0
    grid[1, 0, :] = 10.0
    grid[1, 1, :] = 20.0
    return grid


def test_compute_block_geofm_features_uses_weighted_pixel_means():
    from paper11_geofm.block_features import compute_block_geofm_features
    from paper11_geofm.block_mapping import validate_block_pixel_mapping

    base_embedding = _tiny_embedding_grid()
    mapping = validate_block_pixel_mapping(
        [
            {"block_id": "b0", "row": 0, "col": 0, "weight": 1.0},
            {"block_id": "b0", "row": 0, "col": 1, "weight": 3.0},
            {"block_id": "b1", "row": 1, "col": 0},
        ],
        (2, 2),
    )

    rows = compute_block_geofm_features(base_embedding, mapping)

    assert [row["block_id"] for row in rows] == ["b0", "b1"]
    assert rows[0]["pixel_count"] == 2
    assert rows[0]["pixel_weight_sum"] == 4.0
    assert rows[0]["row_min"] == 0
    assert rows[0]["row_max"] == 0
    assert rows[0]["col_min"] == 0
    assert rows[0]["col_max"] == 1
    assert rows[0]["embedding_mean_00"] == 2.5
    assert rows[0]["embedding_mean_63"] == 2.5
    assert rows[1]["embedding_mean_00"] == 10.0
    assert "embedding_std_mean" in rows[0]
    assert "temporal_stability" in rows[0]


def test_compute_block_geofm_features_uses_annual_temporal_stability():
    from paper11_geofm.block_features import compute_block_geofm_features
    from paper11_geofm.block_mapping import validate_block_pixel_mapping

    base_embedding = _tiny_embedding_grid()
    annual_embeddings = {
        2020: base_embedding,
        2021: base_embedding + 1.0,
    }
    mapping = validate_block_pixel_mapping(
        [{"block_id": "b0", "row": 0, "col": 0}],
        (2, 2),
    )

    rows = compute_block_geofm_features(base_embedding, mapping, annual_embeddings)

    assert 0.0 < rows[0]["temporal_stability"] < 1.0
```

- [ ] **Step 2: Run the block feature tests to verify they fail**

Run:

```powershell
python -m pytest tests\test_phase2_block_geofm.py::test_compute_block_geofm_features_uses_weighted_pixel_means tests\test_phase2_block_geofm.py::test_compute_block_geofm_features_uses_annual_temporal_stability -v
```

Expected: FAIL because `paper11_geofm.block_features` does not exist.

- [ ] **Step 3: Implement block feature aggregation**

Create `src/paper11_geofm/block_features.py`:

```python
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from .block_mapping import BlockPixel


def compute_block_geofm_features(
    base_embedding: np.ndarray,
    mapping: Sequence[BlockPixel],
    annual_embeddings: Mapping[int, np.ndarray] | None = None,
) -> list[dict[str, float | int | str]]:
    _validate_embedding_grid("base_embedding", base_embedding)
    if annual_embeddings is not None:
        for year, embedding in annual_embeddings.items():
            _validate_embedding_grid(f"annual_embeddings[{year}]", embedding)
            if embedding.shape != base_embedding.shape:
                raise ValueError(
                    f"annual_embeddings[{year}] shape {embedding.shape} "
                    f"must match base embedding shape {base_embedding.shape}"
                )

    grouped: dict[str, list[BlockPixel]] = defaultdict(list)
    for entry in mapping:
        grouped[str(entry["block_id"])].append(entry)

    rows: list[dict[str, float | int | str]] = []
    for block_id in sorted(grouped):
        entries = grouped[block_id]
        pixel_rows = np.array([int(entry["row"]) for entry in entries], dtype=np.int64)
        pixel_cols = np.array([int(entry["col"]) for entry in entries], dtype=np.int64)
        weights = np.array([float(entry["weight"]) for entry in entries], dtype=np.float64)
        pixels = np.asarray(base_embedding[pixel_rows, pixel_cols], dtype=np.float64)
        weight_sum = float(weights.sum())
        mean_embedding = np.average(pixels, axis=0, weights=weights)
        centered = pixels - mean_embedding
        weighted_variance = np.average(centered * centered, axis=0, weights=weights)

        row: dict[str, float | int | str] = {
            "block_id": block_id,
            "pixel_count": int(len(entries)),
            "pixel_weight_sum": weight_sum,
            "row_min": int(pixel_rows.min()),
            "row_max": int(pixel_rows.max()),
            "col_min": int(pixel_cols.min()),
            "col_max": int(pixel_cols.max()),
            "embedding_std_mean": float(np.sqrt(weighted_variance).mean()),
            "temporal_stability": _compute_temporal_stability(entries, annual_embeddings),
        }
        for dim, value in enumerate(mean_embedding):
            row[f"embedding_mean_{dim:02d}"] = float(value)
        rows.append(row)

    return rows


def _validate_embedding_grid(name: str, embedding: np.ndarray) -> None:
    if embedding.ndim != 3 or embedding.shape[-1] != 64:
        raise ValueError(f"{name} must have shape [rows, cols, 64], got {embedding.shape}")


def _compute_temporal_stability(
    entries: Sequence[BlockPixel],
    annual_embeddings: Mapping[int, np.ndarray] | None,
) -> float:
    if not annual_embeddings:
        return 1.0

    pixel_rows = np.array([int(entry["row"]) for entry in entries], dtype=np.int64)
    pixel_cols = np.array([int(entry["col"]) for entry in entries], dtype=np.int64)
    weights = np.array([float(entry["weight"]) for entry in entries], dtype=np.float64)
    year_means = []
    for year in sorted(annual_embeddings):
        pixels = np.asarray(annual_embeddings[year][pixel_rows, pixel_cols], dtype=np.float64)
        year_means.append(np.average(pixels, axis=0, weights=weights))

    temporal_variation = float(np.vstack(year_means).std(axis=0).mean())
    return float(1.0 / (1.0 + temporal_variation))
```

- [ ] **Step 4: Run the block feature tests to verify they pass**

Run:

```powershell
python -m pytest tests\test_phase2_block_geofm.py::test_compute_block_geofm_features_uses_weighted_pixel_means tests\test_phase2_block_geofm.py::test_compute_block_geofm_features_uses_annual_temporal_stability -v
```

Expected: PASS.

### Task 4: Schema, Optional Features, Suitability, and Artifacts

**Files:**
- Modify: `tests/test_phase2_block_geofm.py`
- Create: `src/paper11_geofm/block_schema.py`
- Modify: `src/paper11_geofm/artifacts.py`

- [ ] **Step 1: Write failing schema and artifact tests**

Append tests for joining explicit features, readiness metadata, bounded suitability, and Phase 2 artifact writing:

```python
import csv
import json


def test_attach_optional_block_attributes_preserves_explicit_features_and_reports_readiness():
    from paper11_geofm.block_features import attach_optional_block_attributes
    from paper11_geofm.block_schema import summarize_phase2_readiness

    rows = [
        {
            "block_id": "b0",
            "pixel_count": 1,
            "pixel_weight_sum": 1.0,
            "embedding_mean_00": 1.0,
            "embedding_mean_63": 1.0,
            "suitability_proxy": 0.5,
        }
    ]
    attributes = [
        {
            "block_id": "b0",
            "explicit_feature_00": 7.0,
            "explicit_feature_16": 9.0,
            "stable_farmland_label": 1,
            "split": "train",
        }
    ]

    joined = attach_optional_block_attributes(rows, attributes)
    readiness = summarize_phase2_readiness(joined)

    assert joined[0]["explicit_feature_00"] == 7.0
    assert joined[0]["explicit_feature_16"] == 9.0
    assert joined[0]["stable_farmland_label"] == 1
    assert joined[0]["split"] == "train"
    assert readiness["B0"]["ready"] is False
    assert readiness["B1"]["ready"] is False
    assert readiness["B2"]["ready"] is False
    assert readiness["B3"]["ready"] is False
    assert "explicit_features_incomplete" in readiness["B0"]["missing"]


def test_phase2_artifacts_are_written_with_readiness_and_claim_boundary(tmp_path):
    from paper11_geofm.artifacts import write_phase2_artifacts

    row = {
        "block_id": "b0",
        "pixel_count": 1,
        "pixel_weight_sum": 1.0,
        "row_min": 0,
        "row_max": 0,
        "col_min": 0,
        "col_max": 0,
        "embedding_std_mean": 0.0,
        "temporal_stability": 1.0,
        "suitability_proxy": 0.75,
    }
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(dim)

    paths = write_phase2_artifacts(
        [row],
        tmp_path,
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

    with paths["block_table"].open("r", encoding="utf-8", newline="") as handle:
        record = next(csv.DictReader(handle))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

    assert record["block_id"] == "b0"
    assert paths["block_table"].name == "block_geofm_features.csv"
    assert summary["n_blocks"] == 1
    assert summary["block_table"] == "block_geofm_features.csv"
    assert summary["feature_readiness"]["B1"]["ready"] is False
    assert "does not directly measure soil" in summary["claim_boundary"].lower()
```

- [ ] **Step 2: Run the schema and artifact tests to verify they fail**

Run:

```powershell
python -m pytest tests\test_phase2_block_geofm.py::test_attach_optional_block_attributes_preserves_explicit_features_and_reports_readiness tests\test_phase2_block_geofm.py::test_phase2_artifacts_are_written_with_readiness_and_claim_boundary -v
```

Expected: FAIL because `attach_optional_block_attributes`, `block_schema`, or `write_phase2_artifacts` do not exist.

- [ ] **Step 3: Implement optional attribute joining**

Add to `src/paper11_geofm/block_features.py`:

```python
def attach_optional_block_attributes(
    rows: Sequence[Mapping[str, object]],
    attributes: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    if not attributes:
        return [dict(row) for row in rows]

    by_block = {str(row["block_id"]): dict(row) for row in attributes}
    joined: list[dict[str, object]] = []
    for row in rows:
        block_id = str(row["block_id"])
        output = dict(row)
        extra = by_block.get(block_id, {})
        for key, value in extra.items():
            if key != "block_id":
                output[key] = value
        joined.append(output)
    return joined
```

- [ ] **Step 4: Implement schema readiness metadata**

Create `src/paper11_geofm/block_schema.py`:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence


EMBEDDING_COLUMNS = [f"embedding_mean_{idx:02d}" for idx in range(64)]
EXPLICIT_FEATURE_COLUMNS = [f"explicit_feature_{idx:02d}" for idx in range(17)]


def summarize_phase2_readiness(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    columns = {key for row in rows for key in row}
    has_embedding = all(column in columns for column in EMBEDDING_COLUMNS)
    has_suitability = "suitability_proxy" in columns
    has_explicit = all(column in columns for column in EXPLICIT_FEATURE_COLUMNS)

    return {
        "B0": _readiness(
            ready=has_explicit,
            missing=_missing(has_explicit, has_embedding, has_suitability),
        ),
        "B1": _readiness(
            ready=has_explicit and has_embedding,
            missing=_missing(
                has_explicit,
                has_embedding,
                has_suitability,
                require_embedding=True,
            ),
        ),
        "B2": _readiness(
            ready=has_explicit and has_suitability,
            missing=_missing(
                has_explicit,
                has_embedding,
                has_suitability,
                require_suitability=True,
            ),
        ),
        "B3": _readiness(
            ready=has_explicit and has_embedding and has_suitability,
            missing=_missing(
                has_explicit,
                has_embedding,
                has_suitability,
                require_embedding=True,
                require_suitability=True,
            ),
        ),
    }


def _readiness(ready: bool, missing: list[str]) -> dict[str, object]:
    return {"ready": ready, "missing": missing}


def _missing(
    has_explicit: bool,
    has_embedding: bool,
    has_suitability: bool,
    require_embedding: bool = False,
    require_suitability: bool = False,
) -> list[str]:
    missing: list[str] = []
    if not has_explicit:
        missing.append("explicit_features_incomplete")
    if require_embedding and not has_embedding:
        missing.append("geofm_embedding_columns_missing")
    if require_suitability and not has_suitability:
        missing.append("suitability_proxy_missing")
    return missing
```

- [ ] **Step 5: Implement Phase 2 artifact writing**

Add to `src/paper11_geofm/artifacts.py`:

```python
from .block_schema import summarize_phase2_readiness

BLOCK_BASE_COLUMNS = [
    "block_id",
    "pixel_count",
    "pixel_weight_sum",
    "row_min",
    "row_max",
    "col_min",
    "col_max",
]


def write_phase2_artifacts(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    summary: Mapping[str, object],
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    block_table = output_dir / "block_geofm_features.csv"
    summary_path = output_dir / "summary.json"
    fieldnames = _phase2_fieldnames(rows)

    with block_table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    suitability = np.array(
        [float(row["suitability_proxy"]) for row in rows if "suitability_proxy" in row],
        dtype=np.float64,
    )
    output_summary = dict(summary)
    output_summary.update(
        {
            "n_blocks": len(rows),
            "block_table": block_table.name,
            "feature_readiness": summarize_phase2_readiness(rows),
            "claim_boundary": CLAIM_BOUNDARY,
            "suitability_min": float(suitability.min()) if suitability.size else None,
            "suitability_max": float(suitability.max()) if suitability.size else None,
            "suitability_mean": float(suitability.mean()) if suitability.size else None,
        }
    )
    summary_path.write_text(
        json.dumps(output_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"block_table": block_table, "summary": summary_path}


def _phase2_fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    known = (
        BLOCK_BASE_COLUMNS
        + EMBEDDING_COLUMNS
        + ["embedding_std_mean", "temporal_stability", "suitability_proxy"]
    )
    extras = sorted({key for row in rows for key in row if key not in known})
    return [field for field in known if any(field in row for row in rows)] + extras
```

- [ ] **Step 6: Run the schema and artifact tests to verify they pass**

Run:

```powershell
python -m pytest tests\test_phase2_block_geofm.py::test_attach_optional_block_attributes_preserves_explicit_features_and_reports_readiness tests\test_phase2_block_geofm.py::test_phase2_artifacts_are_written_with_readiness_and_claim_boundary -v
```

Expected: PASS.

### Task 5: Experiment Runner and Generated Lightweight Mapping

**Files:**
- Modify: `tests/test_phase2_block_geofm.py`
- Create: `experiments/phase2_block_geofm_features/run_phase2.py`

- [ ] **Step 1: Write failing Phase 2 runner test**

Append a test that imports `run_phase2.py`, executes it with a temp output directory, and inspects both artifacts:

```python
import importlib.util


def test_phase2_runner_writes_block_feature_artifacts(tmp_path):
    runner_path = (
        ROOT / "experiments" / "phase2_block_geofm_features" / "run_phase2.py"
    )
    spec = importlib.util.spec_from_file_location("phase2_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(["--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "block_geofm_features.csv").exists()
    assert (tmp_path / "summary.json").exists()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_blocks"] == 25
    assert summary["mapping_mode"] == "generated_grid"
    assert summary["feature_readiness"]["B3"]["ready"] is False
```

- [ ] **Step 2: Run the runner test to verify it fails**

Run:

```powershell
python -m pytest tests\test_phase2_block_geofm.py::test_phase2_runner_writes_block_feature_artifacts -v
```

Expected: FAIL because `experiments/phase2_block_geofm_features/run_phase2.py` does not exist.

- [ ] **Step 3: Implement Phase 2 runner**

Create `experiments/phase2_block_geofm_features/run_phase2.py` with CLI options:

```text
--sample-dir
--base-year
--row-bins
--col-bins
--output-dir
```

Use `make_grid_region_labels` from Phase 1 only to derive a deterministic lightweight block-to-pixel mapping. Convert each region label into `block_id = "grid_block_<id>"`, validate the mapping, aggregate with `compute_block_geofm_features`, score with `add_suitability_proxy`, and write artifacts with `write_phase2_artifacts`.

The implementation should include:

```python
def build_generated_grid_mapping(grid_shape: tuple[int, int], row_bins: int, col_bins: int) -> list[dict[str, object]]:
    labels = make_grid_region_labels(grid_shape, row_bins, col_bins)
    rows = []
    for pixel_row in range(grid_shape[0]):
        for pixel_col in range(grid_shape[1]):
            rows.append(
                {
                    "block_id": f"grid_block_{int(labels[pixel_row, pixel_col]):02d}",
                    "row": pixel_row,
                    "col": pixel_col,
                    "weight": 1.0,
                }
            )
    return rows
```

Default output directory must be `experiments/phase2_block_geofm_features/outputs/`.

- [ ] **Step 4: Run the runner test to verify it passes**

Run:

```powershell
python -m pytest tests\test_phase2_block_geofm.py::test_phase2_runner_writes_block_feature_artifacts -v
```

Expected: PASS.

### Task 6: Docs, Manifest, Smoke Check, and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `reproducibility/REPRODUCTION_GUIDE.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `scripts/smoke_check.py`
- Modify: `tests/test_repository_layout.py`

- [ ] **Step 1: Update reviewer documentation**

Add the Phase 2 runner to the quick-start sections in `README.md` and `reproducibility/REPRODUCTION_GUIDE.md`:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py
```

State that the default Phase 2 path uses a generated grid-derived mapping from the included Bishan sample and is a feature-assembly smoke test, not real block-level DRL evidence.

- [ ] **Step 2: Register Phase 2 implementation files**

Add these lines to `reproducibility/FILE_MANIFEST.tsv`:

```text
src/paper11_geofm/block_mapping.py	runtime	Block-to-pixel mapping validation utilities for Phase 2 block-level feature assembly.
src/paper11_geofm/block_features.py	runtime	Block-level GeoFM embedding aggregation and optional attribute joining utilities.
src/paper11_geofm/block_schema.py	runtime	Phase 2 block-feature schema and B0/B1/B2/B3 readiness metadata utilities.
experiments/phase2_block_geofm_features/run_phase2.py	experiment	Executable Phase 2 block-level GeoFM feature assembly runner.
tests/test_phase2_block_geofm.py	verification	Pytest checks for Phase 2 mapping validation, block aggregation, schema readiness, artifact writing, and runner behavior.
```

- [ ] **Step 3: Extend repository layout checks**

Add the new Phase 2 files to `REQUIRED_PATHS` in `scripts/smoke_check.py` and to `required_paths` in `tests/test_repository_layout.py`.

- [ ] **Step 4: Run full verification**

Run:

```powershell
python experiments\phase1_bishan_baseline\run_phase1.py
python experiments\phase2_block_geofm_features\run_phase2.py
python scripts\smoke_check.py
python -m pytest tests
git diff --check
```

Expected: all Python commands exit 0; pytest includes the new Phase 2 tests; `git diff --check` reports no whitespace errors.

### Task 7: Commit and Push Phase 2 Implementation

**Files:**
- All implementation, tests, docs, and manifest files from Tasks 2-6.

- [ ] **Step 1: Inspect working tree**

Run:

```powershell
git status --short --branch
git diff --stat
```

Expected: only intended Phase 2 implementation, docs, and verification files are changed.

- [ ] **Step 2: Commit implementation**

Run:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv scripts\smoke_check.py tests\test_repository_layout.py tests\test_phase2_block_geofm.py src\paper11_geofm\block_mapping.py src\paper11_geofm\block_features.py src\paper11_geofm\block_schema.py src\paper11_geofm\artifacts.py experiments\phase2_block_geofm_features\run_phase2.py
git commit -m "Add Phase 2 block GeoFM feature assembly pipeline"
```

- [ ] **Step 3: Push main**

Run:

```powershell
git push
```

- [ ] **Step 4: Report verification evidence**

Report the exact commands that passed, the commit SHA, the pushed branch, and the Phase 2 artifact paths.
