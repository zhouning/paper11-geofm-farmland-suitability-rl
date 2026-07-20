# Phase 72B GeoFM Information-Gain Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a locked, leakage-free Bishan-Dongxing screen that tests whether temporal AlphaEarth features improve one-year farmland-conversion prediction beyond strong public-GIS and LULC-history baselines and strict representation controls.

**Architecture:** Keep public terrain acquisition, deterministic base-feature assembly, split/protocol freezing, partition-local control materialization, model selection, and confirmation evaluation in separate modules. `prepare` creates the split registry before any data-dependent control and writes label-separated development and confirmation packages plus a hashed protocol; `fit-freeze` constructs train/validation controls inside each declared axis, uses development outcomes only, and writes hashed selected model bundles; `confirm` verifies every hash, constructs test controls inside each declared test partition, and evaluates the 2023 and zero-shot outcomes once. All primary comparisons use matched feature rows, model search budgets, spatial-block uncertainty, and predefined practical gates.

**Tech Stack:** Python 3.11+, NumPy, pandas, SciPy, scikit-learn, joblib, Earth Engine Python API, pytest, CSV/JSON/NPZ artifacts.

---

## Scope

This plan implements Phase 72B only. It does not implement GeoFM-STaR neural
models, alter planning rewards, run PPO, optimize spatial plans, use DLTB as the
primary comparator, or modify `paper/submission/final/*`.

Approved design:

`docs/superpowers/specs/2026-07-10-phase72b-geofm-information-gain-screen-design.md`

Required Phase 72A input status:

```text
phase72a_status == phase72a_label_inputs_ready
regions == {bishan, dongxing}
sample_rows == 31627
```

## Files

- Create `experiments/phase72b_geofm_information_gain_screen/phase72b_protocol.json`:
  tracked model, feature, split, control, metric, and gate contract.
- Create `src/paper11_geofm/phase72b_protocol.py`: contract parsing and canonical
  hashed-JSON helpers.
- Create `src/paper11_geofm/phase72b_terrain.py`: terrain contract, audit, and
  local package helpers.
- Create `src/paper11_geofm/phase72b_explicit_features.py`: public-GIS and
  leakage-free LULC-history features.
- Create `src/paper11_geofm/phase72b_geofm_features.py`: temporal GeoFM summaries,
  data-independent random projection, and partition-local strict controls.
- Create `src/paper11_geofm/phase72b_splits.py`: pooled, buffered-spatial, and
  zero-shot split registry and leakage audit.
- Create `src/paper11_geofm/phase72b_metrics.py`: calibration metrics,
  block-bootstrap deltas, and the Phase 72B gate.
- Create `src/paper11_geofm/phase72b_models.py`: model search, calibration,
  bundles, and fit-freeze manifest.
- Create `src/paper11_geofm/phase72b_information_gain_screen.py`: prepare and
  confirmation orchestration plus stable artifact writers.
- Create `experiments/phase72b_geofm_information_gain_screen/fetch_phase72b_terrain.py`.
- Create `experiments/phase72b_geofm_information_gain_screen/run_phase72b_information_gain_screen.py`.
- Create `tests/test_phase72b_geofm_information_gain_screen.py`.
- Create `paper/phase28_results/39_phase72b_geofm_information_gain_screen.md`
  only after the measured confirmation run.
- Modify `paper/phase28_results/README.md` and
  `docs/superpowers/phase33_current_progress_handoff.md` only after the measured
  result is inspected.

Generated arrays, feature packages, model bundles, and confirmation outputs
remain ignored below
`experiments/phase72b_geofm_information_gain_screen/outputs/`.

## Stable Naming

Prepared output directory:

```text
phase72b_terrain_manifest.csv
phase72b_feature_manifest.csv
phase72b_feature_registry.json
phase72b_feature_rows.csv
phase72b_feature_matrices.npz
phase72b_development_targets.npz
phase72b_confirmation_targets.npz
phase72b_split_registry.json
phase72b_row_alignment_audit.csv
phase72b_leakage_audit.json
phase72b_frozen_protocol.json
phase72b_frozen_protocol.sha256
```

`phase72b_feature_manifest.csv` records the frozen base matrices. The fit and
confirmation control manifests use the common fields `axis_id`, `partition_id`,
`control_id`, `seed`, `index_sha256`, `matrix_sha256`, and
`cross_partition_count`. Prepared output contains base matrices only;
fit-freeze and confirmation write separate manifests without rewriting the
frozen prepared manifest.

Fit-freeze output directory:

```text
bundles/*.joblib
phase72b_validation_metrics.csv
phase72b_fit_control_manifest.csv
phase72b_selected_models.json
phase72b_selected_models.sha256
```

Confirmation output directory:

```text
phase72b_metrics.csv
phase72b_predictions.csv
phase72b_calibration.csv
phase72b_bootstrap_deltas.csv
phase72b_control_comparison.csv
phase72b_confirmation_control_manifest.csv
phase72b_transfer_summary.csv
phase72b_information_gain_screen.json
phase72b_information_gain_screen.md
```

---

### Task 1: Protocol Contract and Hash Refusal

**Files:**
- Create: `experiments/phase72b_geofm_information_gain_screen/phase72b_protocol.json`
- Create: `src/paper11_geofm/phase72b_protocol.py`
- Create: `tests/test_phase72b_geofm_information_gain_screen.py`

- [ ] **Step 1: Write failing protocol and hash tests**

Create the test file with repository imports and these tests:

```python
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _protocol_payload() -> dict:
    return {
        "phase": "phase72b_geofm_information_gain_screen",
        "seed": 72,
        "terrain": {
            "source_id": "copernicus_dem_glo30",
            "collection": "COPERNICUS/DEM/GLO30",
            "band": "DEM",
            "scale_m": 500,
            "feature_names": [
                "elevation_mean", "elevation_std", "elevation_min",
                "elevation_max", "slope_mean", "slope_std",
                "slope_max", "local_relief",
            ],
        },
        "years": {"train": [2017, 2018, 2019, 2020, 2021], "validation": [2022], "test": [2023]},
        "controls": {
            "seeds": [72, 73, 74, 75, 76],
            "random_projection_dim": 320,
            "partition_local": True,
            "learned_transform_fit_scope": "training_rows_only",
            "reuse_phase8_d4_tables": False,
        },
        "spatial": {"block_size": 8, "folds": 5, "buffer_rings": 1},
        "bootstrap": {"iterations": 2000, "seed": 72},
        "models": {
            "logistic_c": [0.01, 0.1, 1.0, 10.0],
            "logistic_class_weight": ["none", "balanced"],
            "hgb_learning_rate": [0.03, 0.08],
            "hgb_max_leaf_nodes": [15, 31],
            "hgb_min_samples_leaf": [20, 50],
            "hgb_max_iter": 200,
            "hgb_l2_regularization": [0.0, 1.0],
        },
        "calibration": {"methods": ["none", "sigmoid", "isotonic"], "ece_bins": 10},
        "budgets": [0.10, 0.20],
        "variants": [
            "explicit_static", "explicit_history", "geofm_current_only",
            "geofm_temporal_mean_only", "explicit_plus_geofm_current",
            "explicit_plus_geofm_temporal_full",
            "explicit_plus_temporal_order_shuffle",
            "explicit_plus_spatial_shuffle", "explicit_plus_random_projection",
        ],
        "gates": {
            "ap_vs_explicit": 0.015, "brier_vs_explicit": 0.005,
            "ece_vs_explicit": 0.010, "ap_vs_control": 0.005,
            "brier_vs_control": 0.002, "transfer_ap_gain": 0.005,
            "transfer_brier_gain": 0.002, "transfer_ap_harm": 0.005,
            "transfer_brier_harm": 0.002,
        },
    }


def _write_protocol(path: Path) -> Path:
    path.write_text(json.dumps(_protocol_payload()), encoding="utf-8")
    return path


def test_phase72b_protocol_loads_frozen_thresholds(tmp_path):
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol

    protocol = load_phase72b_protocol(_write_protocol(tmp_path / "protocol.json"))
    assert protocol.seed == 72
    assert protocol.terrain_features[-1] == "local_relief"
    assert protocol.train_years == (2017, 2018, 2019, 2020, 2021)
    assert protocol.gates["ap_vs_explicit"] == 0.015
    assert protocol.control_partition_local is True
    assert protocol.learned_transform_fit_scope == "training_rows_only"
    assert protocol.reuse_phase8_d4_tables is False


def test_phase72b_hashed_json_rejects_modified_payload(tmp_path):
    from paper11_geofm.phase72b_protocol import load_hashed_json, write_hashed_json

    json_path, hash_path = write_hashed_json(tmp_path / "frozen.json", {"status": "frozen", "seed": 72})
    assert load_hashed_json(json_path, hash_path)["status"] == "frozen"
    json_path.write_text('{"status":"changed","seed":72}', encoding="utf-8")
    try:
        load_hashed_json(json_path, hash_path)
    except ValueError as exc:
        assert "hash" in str(exc).lower()
    else:
        raise AssertionError("Expected a modified frozen payload to be rejected")
```

- [ ] **Step 2: Verify RED**

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase72b_geofm_information_gain_screen.py -q --basetemp=D:\tmp\paper11_phase72b_task1_red -p no:cacheprovider
```

Expected: `ModuleNotFoundError` for `phase72b_protocol`.

- [ ] **Step 3: Implement the protocol module and tracked JSON**

Create `phase72b_protocol.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping


PHASE72B_CLAIM_BOUNDARY = (
    "Phase 72B is a leakage-free low-cost information-gain screen using "
    "independent annual product labels. It does not implement GeoFM-STaR, "
    "alter planning rewards, run planning, or revise the formal manuscript."
)


@dataclass(frozen=True)
class Phase72BProtocol:
    seed: int
    terrain_source_id: str
    terrain_collection: str
    terrain_band: str
    terrain_scale_m: int
    terrain_features: tuple[str, ...]
    train_years: tuple[int, ...]
    validation_years: tuple[int, ...]
    test_years: tuple[int, ...]
    control_seeds: tuple[int, ...]
    random_projection_dim: int
    control_partition_local: bool
    learned_transform_fit_scope: str
    reuse_phase8_d4_tables: bool
    spatial_block_size: int
    spatial_folds: int
    buffer_rings: int
    bootstrap_iterations: int
    bootstrap_seed: int
    gates: dict[str, float]
    raw: dict[str, object]


def canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_hashed_json(path: Path | str, payload: Mapping[str, object]) -> tuple[Path, Path]:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = json.loads(json.dumps(dict(payload)))
    json_path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    hash_path = json_path.with_suffix(".sha256")
    hash_path.write_text(canonical_json_sha256(normalized) + "\n", encoding="ascii")
    return json_path, hash_path


def load_hashed_json(json_path: Path | str, hash_path: Path | str | None = None) -> dict[str, object]:
    source = Path(json_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    digest_path = Path(hash_path) if hash_path is not None else source.with_suffix(".sha256")
    expected = digest_path.read_text(encoding="ascii").strip().lower()
    actual = canonical_json_sha256(payload)
    if expected != actual:
        raise ValueError(f"Phase 72B frozen JSON hash mismatch: expected {expected}, got {actual}")
    return payload


def load_phase72b_protocol(path: Path | str) -> Phase72BProtocol:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("phase") != "phase72b_geofm_information_gain_screen":
        raise ValueError("Invalid Phase 72B protocol phase")
    terrain = dict(payload["terrain"]); years = dict(payload["years"])
    controls = dict(payload["controls"]); spatial = dict(payload["spatial"])
    bootstrap = dict(payload["bootstrap"]); gates = {str(k): float(v) for k, v in dict(payload["gates"]).items()}
    if sorted(years["train"] + years["validation"] + years["test"]) != list(range(2017, 2024)):
        raise ValueError("Phase 72B years must partition origins 2017-2023")
    if tuple(controls["seeds"]) != (72, 73, 74, 75, 76):
        raise ValueError("Phase 72B control seeds are frozen")
    if controls.get("partition_local") is not True:
        raise ValueError("Phase 72B controls must be partition-local")
    if controls.get("learned_transform_fit_scope") != "training_rows_only":
        raise ValueError("Phase 72B learned transforms must fit training rows only")
    if controls.get("reuse_phase8_d4_tables") is not False:
        raise ValueError("Phase 72B must not reuse transductive Phase 8 D4 tables")
    return Phase72BProtocol(
        seed=int(payload["seed"]), terrain_source_id=str(terrain["source_id"]),
        terrain_collection=str(terrain["collection"]), terrain_band=str(terrain["band"]),
        terrain_scale_m=int(terrain["scale_m"]), terrain_features=tuple(terrain["feature_names"]),
        train_years=tuple(int(v) for v in years["train"]),
        validation_years=tuple(int(v) for v in years["validation"]),
        test_years=tuple(int(v) for v in years["test"]),
        control_seeds=tuple(int(v) for v in controls["seeds"]),
        random_projection_dim=int(controls["random_projection_dim"]),
        control_partition_local=bool(controls["partition_local"]),
        learned_transform_fit_scope=str(controls["learned_transform_fit_scope"]),
        reuse_phase8_d4_tables=bool(controls["reuse_phase8_d4_tables"]),
        spatial_block_size=int(spatial["block_size"]), spatial_folds=int(spatial["folds"]),
        buffer_rings=int(spatial["buffer_rings"]), bootstrap_iterations=int(bootstrap["iterations"]),
        bootstrap_seed=int(bootstrap["seed"]), gates=gates, raw=payload,
    )
```

Create the tracked protocol with the test payload plus these exact sections:

```json
"models": {
  "logistic_c": [0.01, 0.1, 1.0, 10.0],
  "logistic_class_weight": ["none", "balanced"],
  "hgb_learning_rate": [0.03, 0.08],
  "hgb_max_leaf_nodes": [15, 31],
  "hgb_min_samples_leaf": [20, 50],
  "hgb_max_iter": 200,
  "hgb_l2_regularization": [0.0, 1.0]
},
"calibration": {"methods": ["none", "sigmoid", "isotonic"], "ece_bins": 10},
"budgets": [0.10, 0.20],
"variants": [
  "explicit_static", "explicit_history", "geofm_current_only",
  "geofm_temporal_mean_only", "explicit_plus_geofm_current",
  "explicit_plus_geofm_temporal_full",
  "explicit_plus_temporal_order_shuffle",
  "explicit_plus_spatial_shuffle", "explicit_plus_random_projection"
]
```

- [ ] **Step 4: Verify GREEN**

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```powershell
git add experiments\phase72b_geofm_information_gain_screen\phase72b_protocol.json src\paper11_geofm\phase72b_protocol.py tests\test_phase72b_geofm_information_gain_screen.py
git commit -m "feat: add Phase 72B frozen protocol"
```

---

### Task 2: Copernicus Terrain Acquisition and Audit

**Files:**
- Create: `src/paper11_geofm/phase72b_terrain.py`
- Create: `experiments/phase72b_geofm_information_gain_screen/fetch_phase72b_terrain.py`
- Modify: `tests/test_phase72b_geofm_information_gain_screen.py`

- [ ] **Step 1: Write failing terrain tests**

Append tests that use the Phase 72A region contract and an injected extractor:

```python
def _phase72a_regions(path: Path) -> Path:
    payload = {
        "source": {"source_id": "esri", "collection": "esri", "label_role": "independent_annual_product_label", "independent_from_dltb_slope_reward_geofm": True, "crop_class_code": 5, "scale_m": 500},
        "regions": [{"region_id": "alpha", "bbox": [100, 20, 101, 21], "years": [2017, 2018, 2019], "grid_shape": [2, 3], "embedding_dim": 2, "embedding_pattern": "alpha_emb_{year}.npy", "label_pattern": "alpha_lulc_{year}.npy"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8"); return path


def test_phase72b_terrain_fetch_and_audit_use_exact_grid(tmp_path):
    from paper11_geofm.phase72a_label_sources import load_phase72a_region_contract
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol
    from paper11_geofm.phase72b_terrain import audit_phase72b_terrain_assets, fetch_phase72b_terrain

    def extractor(*, bbox, shape, scale_m, collection, band):
        assert bbox == (100.0, 20.0, 101.0, 21.0); assert shape == (2, 3)
        base = np.arange(6, dtype=np.float32).reshape(2, 3)
        return {
            "elevation_mean": base, "elevation_std": base + 1,
            "elevation_min": base - 1, "elevation_max": base + 2,
            "slope_mean": base + 3, "slope_std": base + 4,
            "slope_max": base + 5, "local_relief": np.full((2, 3), 3, np.float32),
        }

    protocol = load_phase72b_protocol(_write_protocol(tmp_path / "protocol.json"))
    regions = load_phase72a_region_contract(_phase72a_regions(tmp_path / "regions.json"))
    manifest = fetch_phase72b_terrain(protocol, regions, output_dir=tmp_path / "terrain", extractor=extractor)
    audit = audit_phase72b_terrain_assets(protocol, regions, tmp_path / "terrain")
    assert manifest["status"] == "complete"
    assert audit["status"] == "terrain_inputs_ready"
    assert audit["rows"][0]["shape"] == "2x3"
    assert len(audit["rows"][0]["sha256"]) == 64


def test_phase72b_terrain_audit_blocks_wrong_shape(tmp_path):
    from paper11_geofm.phase72a_label_sources import load_phase72a_region_contract
    from paper11_geofm.phase72b_protocol import load_phase72b_protocol
    from paper11_geofm.phase72b_terrain import audit_phase72b_terrain_assets

    protocol = load_phase72b_protocol(_write_protocol(tmp_path / "protocol.json"))
    regions = load_phase72a_region_contract(_phase72a_regions(tmp_path / "regions.json"))
    terrain = tmp_path / "terrain"; terrain.mkdir()
    np.savez(terrain / "alpha_terrain.npz", **{name: np.zeros((2, 2), np.float32) for name in protocol.terrain_features})
    audit = audit_phase72b_terrain_assets(protocol, regions, terrain)
    assert audit["status"] == "phase72b_inputs_not_ready"
    assert "shape" in " ".join(audit["errors"]).lower()
```

- [ ] **Step 2: Verify RED**

Expected: missing `phase72b_terrain`.

- [ ] **Step 3: Implement terrain local functions**

Create `phase72b_terrain.py`:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .phase72a_label_sources import Phase72ARegionContract
from .phase72b_protocol import PHASE72B_CLAIM_BOUNDARY, Phase72BProtocol


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def fetch_phase72b_terrain(protocol: Phase72BProtocol, regions: Phase72ARegionContract, *, output_dir: Path | str, extractor: Callable[..., dict[str, np.ndarray]]) -> dict[str, object]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    records = []; failures = []
    for region in regions.regions:
        path = output / f"{region.region_id}_terrain.npz"
        try:
            arrays = extractor(bbox=region.bbox, shape=region.grid_shape, scale_m=protocol.terrain_scale_m, collection=protocol.terrain_collection, band=protocol.terrain_band)
            missing = [name for name in protocol.terrain_features if name not in arrays]
            if missing: raise ValueError(f"missing terrain features: {missing}")
            normalized = {}
            for name in protocol.terrain_features:
                value = np.asarray(arrays[name], dtype=np.float32)
                if tuple(value.shape) != region.grid_shape:
                    raise ValueError(f"terrain shape mismatch for {region.region_id} {name}: expected {region.grid_shape}, got {tuple(value.shape)}")
                if not np.isfinite(value).all(): raise ValueError(f"non-finite terrain values: {region.region_id} {name}")
                normalized[name] = value
            np.savez_compressed(path, **normalized)
            records.append({"region_id": region.region_id, "path": str(path), "shape": "x".join(map(str, region.grid_shape)), "sha256": _file_sha256(path)})
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append({"region_id": region.region_id, "reason": str(exc)})
    manifest = {"status": "complete" if records and not failures else "partial" if records else "failed", "records": records, "failures": failures, "claim_boundary": PHASE72B_CLAIM_BOUNDARY}
    (output / "phase72b_terrain_fetch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def audit_phase72b_terrain_assets(protocol: Phase72BProtocol, regions: Phase72ARegionContract, terrain_dir: Path | str) -> dict[str, object]:
    root = Path(terrain_dir); rows = []; errors = []
    for region in regions.regions:
        path = root / f"{region.region_id}_terrain.npz"
        if not path.exists(): errors.append(f"missing terrain file for {region.region_id}: {path}"); continue
        try:
            with np.load(path) as package:
                for name in protocol.terrain_features:
                    if name not in package: errors.append(f"missing terrain feature {region.region_id} {name}"); continue
                    if tuple(package[name].shape) != region.grid_shape: errors.append(f"terrain shape mismatch {region.region_id} {name}")
            rows.append({"region_id": region.region_id, "path": str(path), "shape": "x".join(map(str, region.grid_shape)), "sha256": _file_sha256(path)})
        except (OSError, ValueError) as exc: errors.append(f"unreadable terrain package {region.region_id}: {exc}")
    return {"status": "terrain_inputs_ready" if not errors and len(rows) == len(regions.regions) else "phase72b_inputs_not_ready", "rows": rows, "errors": errors, "claim_boundary": PHASE72B_CLAIM_BOUNDARY}
```

- [ ] **Step 4: Implement the Earth Engine fetch CLI**

The script imports Phase 72A/72B contracts, exposes an injectable
`fetch_phase72b_terrain`, and uses this default extractor:

```python
def _default_extractor(*, bbox, shape, scale_m, collection, band):
    import ee
    region = ee.Geometry.Rectangle(list(bbox))
    elevation = ee.ImageCollection(collection).filterBounds(region).select([band]).mosaic().clip(region)
    slope = ee.Terrain.slope(elevation)
    reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True).combine(ee.Reducer.minMax(), sharedInputs=True)
    elevation_stats = elevation.reduceResolution(reducer=reducer, maxPixels=4096).reproject(ee.Projection("EPSG:4326").atScale(int(scale_m)))
    slope_stats = slope.reduceResolution(reducer=reducer, maxPixels=4096).reproject(ee.Projection("EPSG:4326").atScale(int(scale_m)))
    image = ee.Image.cat([
        elevation_stats.select([f"{band}_mean"]).rename("elevation_mean"),
        elevation_stats.select([f"{band}_stdDev"]).rename("elevation_std"),
        elevation_stats.select([f"{band}_min"]).rename("elevation_min"),
        elevation_stats.select([f"{band}_max"]).rename("elevation_max"),
        slope_stats.select(["slope_mean"]).rename("slope_mean"),
        slope_stats.select(["slope_stdDev"]).rename("slope_std"),
        slope_stats.select(["slope_max"]).rename("slope_max"),
        elevation_stats.select([f"{band}_max"]).subtract(elevation_stats.select([f"{band}_min"])).rename("local_relief"),
    ]).setDefaultProjection(ee.Projection("EPSG:4326").atScale(int(scale_m)))
    result = image.sampleRectangle(region=region, defaultValue=0).getInfo().get("properties", {})
    arrays = {name: np.asarray(result[name], dtype=np.float32) for name in result}
    if any(tuple(value.shape) != tuple(shape) for value in arrays.values()):
        raise ValueError(f"Earth Engine terrain shape mismatch: expected {shape}, got {[value.shape for value in arrays.values()]}")
    return arrays
```

CLI arguments:

```text
--phase72a-region-config
--phase72b-protocol
--output-dir
--project
--authenticate
```

Initialize Earth Engine once, print status/counts/manifest, and return nonzero
unless both regions are complete. Never create a fallback terrain package.

- [ ] **Step 5: Verify GREEN and commit**

Expected cumulative result: `4 passed`.

```powershell
git add src\paper11_geofm\phase72b_terrain.py experiments\phase72b_geofm_information_gain_screen\fetch_phase72b_terrain.py tests\test_phase72b_geofm_information_gain_screen.py
git commit -m "feat: add Phase 72B terrain intake"
```

---

### Task 3: Leakage-Free Explicit GIS and LULC Features

**Files:**
- Create: `src/paper11_geofm/phase72b_explicit_features.py`
- Modify: `tests/test_phase72b_geofm_information_gain_screen.py`

- [ ] **Step 1: Write failing hand-computable feature tests**

Use a `3 x 3` grid with 2017-2020 labels and one sample at the center. Assert
terrain lookup, center coordinates, history counts, transition count, and
clipped neighborhood proportions. Build the origin-2018 feature twice while
changing only 2019/2020 labels and assert exact equality.

```python
def test_phase72b_explicit_features_use_only_history_through_origin():
    from paper11_geofm.phase72a_label_sources import Phase72ARegionSpec
    from paper11_geofm.phase72b_explicit_features import build_phase72b_explicit_features

    region = Phase72ARegionSpec("alpha", (100, 20, 101, 21), (2017, 2018, 2019, 2020), (3, 3), 2, "e{year}.npy", "l{year}.npy")
    rows = [{"sample_index": 0, "region_id": "alpha", "row": 1, "col": 1, "origin_year": 2018, "history_length": 2}]
    labels = {
        2017: np.array([[1, 5, 1], [5, 7, 5], [1, 5, 1]], np.int16),
        2018: np.array([[5, 5, 5], [5, 5, 7], [1, 5, 1]], np.int16),
        2019: np.full((3, 3), 7, np.int16), 2020: np.full((3, 3), 1, np.int16),
    }
    terrain = {name: np.full((3, 3), i, np.float32) for i, name in enumerate(("elevation_mean", "elevation_std", "elevation_min", "elevation_max", "slope_mean", "slope_std", "slope_max", "local_relief"))}
    first = build_phase72b_explicit_features(rows, regions={"alpha": region}, labels={"alpha": labels}, terrain={"alpha": terrain}, crop_class_code=5)
    changed = {**labels, 2019: np.zeros((3, 3), np.int16), 2020: np.zeros((3, 3), np.int16)}
    second = build_phase72b_explicit_features(rows, regions={"alpha": region}, labels={"alpha": changed}, terrain={"alpha": terrain}, crop_class_code=5)
    assert np.array_equal(first["explicit_history"], second["explicit_history"])
    registry = first["registry"]
    values = dict(zip(registry["explicit_history"], first["explicit_history"][0]))
    assert values["terrain_local_relief"] == 7.0
    assert values["cell_crop_transition_count"] == 1.0
    assert values["cell_history_count_lulc_07"] == 1.0
    assert values["neighbor3_current_crop_fraction"] == 6 / 9
```

Add a second test for a corner cell to prove windows use only available cells
and never wrap across array edges.

- [ ] **Step 2: Verify RED**

Expected: missing `phase72b_explicit_features`.

- [ ] **Step 3: Implement explicit features**

Create these constants and public function:

```python
ESRI_CLASS_CODES = (1, 2, 4, 5, 7, 8, 9, 10, 11)
TERRAIN_FEATURES = (
    "elevation_mean", "elevation_std", "elevation_min", "elevation_max",
    "slope_mean", "slope_std", "slope_max", "local_relief",
)


def build_phase72b_explicit_features(sample_rows, *, regions, labels, terrain, crop_class_code=5):
    static_names = [f"terrain_{name}" for name in TERRAIN_FEATURES] + [
        "cell_longitude", "cell_latitude", "cell_row_normalized",
        "cell_col_normalized", "region_index", "origin_year", "history_length",
    ]
    history_names = static_names + [
        "previous_lulc_class", "previous_crop_flag", "cell_historical_crop_fraction",
        "cell_crop_transition_count", "cell_years_since_last_non_crop",
        *[f"cell_history_count_lulc_{code:02d}" for code in ESRI_CLASS_CODES],
        "cell_history_count_lulc_unknown",
        *[f"neighbor3_current_fraction_lulc_{code:02d}" for code in ESRI_CLASS_CODES],
        "neighbor3_current_fraction_lulc_unknown", "neighbor3_current_crop_fraction",
        *[f"neighbor5_current_fraction_lulc_{code:02d}" for code in ESRI_CLASS_CODES],
        "neighbor5_current_fraction_lulc_unknown", "neighbor5_current_crop_fraction",
        "neighbor3_historical_crop_mean", "neighbor3_historical_crop_trend",
        "neighbor5_historical_crop_mean", "neighbor5_historical_crop_trend",
    ]
    region_order = {name: index for index, name in enumerate(sorted(regions))}
    static_rows = []; history_rows = []
    for row in sample_rows:
        region_id = str(row["region_id"]); spec = regions[region_id]
        grid_row = int(row["row"]); grid_col = int(row["col"]); origin = int(row["origin_year"])
        years = [year for year in spec.years if year <= origin]
        cell_history = [int(labels[region_id][year][grid_row, grid_col]) for year in years]
        lon = spec.bbox[0] + (grid_col + 0.5) / spec.grid_shape[1] * (spec.bbox[2] - spec.bbox[0])
        lat = spec.bbox[3] - (grid_row + 0.5) / spec.grid_shape[0] * (spec.bbox[3] - spec.bbox[1])
        static = [float(terrain[region_id][name][grid_row, grid_col]) for name in TERRAIN_FEATURES] + [
            lon, lat, grid_row / max(1, spec.grid_shape[0] - 1), grid_col / max(1, spec.grid_shape[1] - 1),
            float(region_order[region_id]), float(origin), float(len(years)),
        ]
        previous = cell_history[-2] if len(cell_history) >= 2 else -1
        transitions = sum(int(a != b) for a, b in zip(cell_history[:-1], cell_history[1:]))
        non_crop_indexes = [i for i, value in enumerate(cell_history) if value != crop_class_code]
        since_non_crop = len(cell_history) - 1 - non_crop_indexes[-1] if non_crop_indexes else len(cell_history)
        counts = [float(sum(value == code for value in cell_history)) for code in ESRI_CLASS_CODES]
        counts.append(float(sum(value not in ESRI_CLASS_CODES for value in cell_history)))
        current = labels[region_id][origin]
        neighbor_values = []
        history_neighbor = []
        for radius in (1, 2):
            window = _window(current, grid_row, grid_col, radius)
            neighbor_values.extend([float(np.mean(window == code)) for code in ESRI_CLASS_CODES])
            neighbor_values.append(float(np.mean(~np.isin(window, ESRI_CLASS_CODES))))
            neighbor_values.append(float(np.mean(window == crop_class_code)))
            annual = [float(np.mean(_window(labels[region_id][year], grid_row, grid_col, radius) == crop_class_code)) for year in years]
            history_neighbor.extend([float(np.mean(annual)), _linear_trend(annual)])
        history = static + [float(previous), float(previous == crop_class_code), float(np.mean(np.asarray(cell_history) == crop_class_code)), float(transitions), float(since_non_crop), *counts, *neighbor_values, *history_neighbor]
        static_rows.append(static); history_rows.append(history)
    return {
        "explicit_static": np.asarray(static_rows, np.float32),
        "explicit_history": np.asarray(history_rows, np.float32),
        "registry": {"explicit_static": static_names, "explicit_history": history_names},
    }
```

Implement `_window` with clipped row/column slices and `_linear_trend` using a
centered least-squares slope that returns zero for one value. Validate sample
indexes are contiguous, every referenced year exists, and all output values are
finite.

- [ ] **Step 4: Verify GREEN and commit**

Expected cumulative result: `6 passed`.

```powershell
git add src\paper11_geofm\phase72b_explicit_features.py tests\test_phase72b_geofm_information_gain_screen.py
git commit -m "feat: assemble Phase 72B explicit features"
```

---

### Task 4: GeoFM Temporal Base Features and Partition-Local Controls

**Files:**
- Create: `src/paper11_geofm/phase72b_geofm_features.py`
- Modify: `tests/test_phase72b_geofm_information_gain_screen.py`

- [ ] **Step 1: Write failing temporal and control tests**

Use four short histories with two regions/origin strata. Assert:

```python
def test_phase72b_geofm_temporal_summary_and_controls_are_deterministic():
    from paper11_geofm.phase72b_geofm_features import build_phase72b_geofm_features, build_phase72b_control_features

    histories = np.zeros((4, 4, 2), np.float32)
    masks = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0]], bool)
    histories[0, :3] = [[1, 2], [2, 4], [4, 8]]
    histories[1, :3] = [[3, 1], [4, 2], [5, 4]]
    histories[2, :2] = [[1, 1], [2, 2]]; histories[3, :2] = [[7, 7], [8, 8]]
    rows = [
        {"sample_index": 0, "region_id": "a", "origin_year": 2019},
        {"sample_index": 1, "region_id": "a", "origin_year": 2019},
        {"sample_index": 2, "region_id": "b", "origin_year": 2018},
        {"sample_index": 3, "region_id": "b", "origin_year": 2018},
    ]
    partitions = ["pooled:train", "pooled:train", "pooled:validation", "pooled:validation"]
    features = build_phase72b_geofm_features(histories, masks)
    assert features["geofm_current"][0].tolist() == [4.0, 8.0]
    assert features["geofm_temporal_full"].shape == (4, 10)
    first = build_phase72b_control_features("spatial_shuffle", histories, masks, rows, partition_ids=partitions, seed=72, output_dim=10)
    second = build_phase72b_control_features("spatial_shuffle", histories, masks, rows, partition_ids=partitions, seed=72, output_dim=10)
    assert np.array_equal(first["matrix"], second["matrix"])
    assert first["manifest"]["cross_partition_count"] == 0
    for target, source in enumerate(first["manifest"]["source_index_by_target"]):
        assert partitions[target] == partitions[source]


def test_phase72b_temporal_shuffle_keeps_current_embedding():
    from paper11_geofm.phase72b_geofm_features import build_phase72b_control_features, build_phase72b_geofm_features
    histories = np.array([[[1.0], [2.0], [3.0], [4.0]]], np.float32); masks = np.ones((1, 4), bool)
    control = build_phase72b_control_features("temporal_order_shuffle", histories, masks, [{"sample_index": 0, "region_id": "a", "origin_year": 2020}], partition_ids=["pooled:test"], seed=72, output_dim=5)
    original = build_phase72b_geofm_features(histories, masks)["geofm_temporal_full"]
    assert control["matrix"][0, 0] == 4.0
    assert not np.array_equal(control["matrix"], original)


def test_phase72b_spatial_shuffle_cannot_cross_split_partition():
    from paper11_geofm.phase72b_geofm_features import build_phase72b_control_features
    histories = np.arange(16, dtype=np.float32).reshape(4, 2, 2)
    masks = np.ones((4, 2), bool)
    rows = [{"sample_index": i, "region_id": "a", "origin_year": 2018} for i in range(4)]
    partitions = ["axis:train", "axis:train", "axis:test", "axis:test"]
    result = build_phase72b_control_features("spatial_shuffle", histories, masks, rows, partition_ids=partitions, seed=72, output_dim=10)
    sources = result["manifest"]["source_index_by_target"]
    assert all(partitions[target] == partitions[source] for target, source in enumerate(sources))
    assert result["manifest"]["cross_partition_count"] == 0


def test_phase72b_random_projection_is_data_independent_and_orthonormal():
    from paper11_geofm.phase72b_geofm_features import build_phase72b_random_projection
    first = build_phase72b_random_projection(input_dim=8, output_dim=3, seed=72)
    second = build_phase72b_random_projection(input_dim=8, output_dim=3, seed=72)
    assert np.array_equal(first, second)
    assert np.allclose(first.T @ first, np.eye(3), atol=1e-6)


def test_phase72b_controls_require_nonblank_partition_ids():
    from paper11_geofm.phase72b_geofm_features import build_phase72b_control_features
    histories = np.ones((1, 2, 2), np.float32)
    masks = np.ones((1, 2), bool)
    rows = [{"sample_index": 0, "region_id": "a", "origin_year": 2018}]
    with pytest.raises(ValueError, match="partition"):
        build_phase72b_control_features("spatial_shuffle", histories, masks, rows, partition_ids=[""], seed=72, output_dim=10)
```

Extend the refusal test to assert that omitting `partition_ids` or providing a
learned-transform fit scope other than `training_rows_only` raises `ValueError`
before any control matrix is returned.

- [ ] **Step 2: Verify RED**

Expected: missing `phase72b_geofm_features`.

- [ ] **Step 3: Implement temporal features and controls**

Create:

```python
def build_phase72b_geofm_features(embedding_history: np.ndarray, history_mask: np.ndarray) -> dict[str, np.ndarray]:
    history = np.asarray(embedding_history, np.float32); mask = np.asarray(history_mask, bool)
    if history.ndim != 3 or mask.shape != history.shape[:2]: raise ValueError("Invalid Phase 72B embedding history")
    current = []; means = []; stds = []; deltas = []; trends = []
    for values, valid in zip(history, mask):
        observed = values[valid]
        if len(observed) == 0: raise ValueError("Phase 72B history cannot be empty")
        current.append(observed[-1]); means.append(observed.mean(axis=0)); stds.append(observed.std(axis=0))
        deltas.append(observed[-1] - observed[0]); trends.append(_vector_trend(observed))
    current = np.asarray(current, np.float32); mean = np.asarray(means, np.float32)
    full = np.concatenate([current, mean, np.asarray(stds, np.float32), np.asarray(deltas, np.float32), np.asarray(trends, np.float32)], axis=1)
    return {"geofm_current": current, "geofm_temporal_mean": mean, "geofm_temporal_full": full}


def build_phase72b_random_projection(*, input_dim, output_dim, seed):
    if int(output_dim) <= 0 or int(output_dim) > int(input_dim):
        raise ValueError("Phase 72B random projection requires 0 < output_dim <= input_dim")
    rng = np.random.default_rng(int(seed))
    matrix = rng.normal(size=(int(input_dim), int(output_dim)))
    q, _ = np.linalg.qr(matrix, mode="reduced")
    return np.asarray(q[:, : int(output_dim)], np.float32)


def build_phase72b_control_features(control_id, embedding_history, history_mask, sample_rows, *, partition_ids=None, seed, output_dim, learned_transform_fit_scope="training_rows_only"):
    history = np.asarray(embedding_history, np.float32).copy(); mask = np.asarray(history_mask, bool)
    if partition_ids is None:
        raise ValueError("Phase 72B controls require partition IDs")
    partitions = np.asarray([str(value) for value in partition_ids], dtype=object)
    if len(history) != len(sample_rows) or len(history) != len(partitions):
        raise ValueError("Phase 72B controls require aligned histories, rows, and partitions")
    if any(not value.strip() for value in partitions):
        raise ValueError("Phase 72B controls require nonblank partition IDs")
    if learned_transform_fit_scope != "training_rows_only":
        raise ValueError("Phase 72B learned transforms must fit training rows only")
    rng = np.random.default_rng(int(seed))
    source_by_target = list(range(len(history)))
    data_dependent = control_id in {"temporal_order_shuffle", "spatial_shuffle"}
    if control_id == "temporal_order_shuffle":
        for row_index in range(len(history)):
            valid_indexes = np.flatnonzero(mask[row_index])
            earlier_positions = valid_indexes[:-1]
            if len(earlier_positions) > 1:
                original = history[row_index, earlier_positions].copy()
                permutation = rng.permutation(len(earlier_positions))
                if np.array_equal(permutation, np.arange(len(earlier_positions))):
                    permutation = np.roll(permutation, 1)
                history[row_index, earlier_positions] = original[permutation]
        matrix = build_phase72b_geofm_features(history, mask)["geofm_temporal_full"]
    elif control_id == "spatial_shuffle":
        groups = {}
        for index, row in enumerate(sample_rows):
            key = (str(partitions[index]), str(row["region_id"]), int(row["origin_year"]))
            groups.setdefault(key, []).append(index)
        shuffled = history.copy(); shuffled_mask = mask.copy()
        for indexes in groups.values():
            source = np.asarray(indexes); permuted = rng.permutation(source)
            if len(source) > 1 and np.array_equal(source, permuted):
                permuted = np.roll(permuted, 1)
            shuffled[source] = history[permuted]
            shuffled_mask[source] = mask[permuted]
            for target, source_index in zip(source.tolist(), permuted.tolist()):
                source_by_target[target] = source_index
        matrix = build_phase72b_geofm_features(shuffled, shuffled_mask)["geofm_temporal_full"]
    elif control_id == "random_projection":
        flattened = (history * mask[..., None]).reshape(len(history), -1)
        projection = build_phase72b_random_projection(input_dim=flattened.shape[1], output_dim=int(output_dim), seed=int(seed))
        matrix = np.asarray(flattened @ projection, np.float32)
    else:
        raise ValueError(f"Unknown Phase 72B control: {control_id}")
    cross_partition_count = sum(
        partitions[target] != partitions[source]
        for target, source in enumerate(source_by_target)
    )
    if cross_partition_count:
        raise ValueError("Phase 72B control crossed a split partition")
    return {
        "matrix": np.asarray(matrix, np.float32),
        "manifest": {
            "control_id": str(control_id), "seed": int(seed),
            "partition_ids": sorted(set(partitions.tolist())),
            "data_dependent": bool(data_dependent),
            "learned_transform_fit_scope": learned_transform_fit_scope,
            "source_index_by_target": source_by_target,
            "cross_partition_count": int(cross_partition_count),
        },
    }
```

The temporal permutation uses only the earlier observed value rows and preserves
their target positions; it never reads label values. Spatial permutations are
grouped by `partition_id x region_id x origin_year` and therefore cannot move a
history across train, validation, test, spatial-fold, source-region, or target-
region boundaries. Assert `output_dim` equals the full temporal dimension for
temporal/spatial controls under the tracked protocol. Do not import, read, or
reuse Phase 8 D4 feature tables.

- [ ] **Step 4: Verify GREEN and commit**

Expected: every Phase 72B test through Task 4 passes.

```powershell
git add src\paper11_geofm\phase72b_geofm_features.py tests\test_phase72b_geofm_information_gain_screen.py
git commit -m "feat: add partition-local Phase 72B controls"
```

---

### Task 5: Split Registry, Leakage Audit, and Prepare Package

**Files:**
- Create: `src/paper11_geofm/phase72b_splits.py`
- Create: `src/paper11_geofm/phase72b_information_gain_screen.py`
- Modify: `tests/test_phase72b_geofm_information_gain_screen.py`

- [ ] **Step 1: Write failing split and prepare tests**

Append a split fixture containing both regions, origins 2017-2023, and multiple
8-cell blocks. Assert pooled years, transfer isolation, and spatial-buffer
exclusion. Add a prepare integration fixture built from a tiny Phase 72A CSV/NPZ
and assert 2023 targets occur only in `phase72b_confirmation_targets.npz`.

```python
def test_phase72b_splits_lock_years_regions_and_spatial_buffers():
    from paper11_geofm.phase72b_splits import build_phase72b_split_registry
    rows = []
    for region in ("bishan", "dongxing"):
        for year in range(2017, 2024):
            for br, bc in ((0, 0), (0, 2), (2, 0), (2, 2), (4, 4)):
                rows.append({"sample_index": len(rows), "region_id": region, "origin_year": year, "spatial_block_id": f"{region}_br{br:03d}_bc{bc:03d}"})
    registry = build_phase72b_split_registry(rows, train_years=(2017, 2018, 2019, 2020, 2021), validation_year=2022, test_year=2023, folds=5, buffer_rings=1)
    pooled = registry["pooled_temporal"]
    assert {rows[i]["origin_year"] for i in pooled["train"]} == {2017, 2018, 2019, 2020, 2021}
    transfer = registry["bishan_to_dongxing"]
    assert {rows[i]["region_id"] for i in transfer["train"]} == {"bishan"}
    assert {rows[i]["region_id"] for i in transfer["test"]} == {"dongxing"}
    spatial = registry["spatial_bishan_fold0"]
    assert not set(spatial["train_block_ids"]) & set(spatial["test_block_ids"])
    assert not set(spatial["train_block_ids"]) & set(spatial["buffer_block_ids"])
```

Prepare test required assertions:

```python
assert set(np.load(paths["development_targets_npz"])["origin_year"]) <= set(range(2017, 2023))
assert set(np.load(paths["confirmation_targets_npz"])["origin_year"]) == {2023}
assert load_hashed_json(paths["protocol_json"], paths["protocol_hash"])["status"] == "phase72b_protocol_frozen"
assert package["leakage_audit"]["status"] == "leakage_audit_passed"
assert package["control_materialization_status"] == "deferred_until_axis_partitions_frozen"
with np.load(paths["feature_matrices_npz"]) as matrices:
    assert not any("shuffle" in name or "random_projection" in name for name in matrices.files)
assert package["frozen_protocol"]["split_before_controls"] is True
```

- [ ] **Step 2: Verify RED**

Expected: missing split and prepare APIs.

- [ ] **Step 3: Implement split registry and leakage audit**

Create `phase72b_splits.py` with stable SHA256 fold assignment, block parsing
using `^(?P<region>.+)_br(?P<br>\d+)_bc(?P<bc>\d+)$`, Chebyshev-distance buffer
rings, and these axes:

```text
pooled_temporal
bishan_to_dongxing
dongxing_to_bishan
spatial_bishan_fold0 ... spatial_bishan_fold4
spatial_dongxing_fold0 ... spatial_dongxing_fold4
```

Each axis stores train/validation/test integer indexes, region/year summaries,
train/test/buffer block IDs, and class counts. `audit_phase72b_splits` must
reject index overlap, wrong years, wrong transfer regions, test blocks in
training, buffer blocks in training, missing indexes, and missing train or
validation outcome classes.

- [ ] **Step 4: Implement prepare package orchestration**

In `phase72b_information_gain_screen.py`, implement:

```python
def prepare_phase72b_information_gain_screen(
    *, protocol_path, phase72a_region_config, phase72a_package_dir,
    embedding_dirs, label_dirs, terrain_dir,
) -> dict[str, object]:
```

The function must:

1. Load and validate both contracts and Phase 72A package status.
2. Re-audit terrain and all Phase 72A manifest paths/hashes.
3. Load sample rows and tensors and require contiguous sample indexes.
4. Build the split registry and leakage audit before calling any control API.
5. Build explicit and GeoFM base matrices only; do not materialize temporal-
   shuffle, spatial-shuffle, or random-projection matrices in `prepare`.
6. Write no model metrics and no control matrix.
7. Return separate development targets (`origin_year <= 2022`) and confirmation
   targets (`origin_year == 2023`).
8. Build a frozen protocol payload containing the tracked protocol, source file
   hashes, matrix shapes/dtypes/hashes, feature registry hash, split registry
   hash, leakage status, `split_before_controls=true`, and
   `control_materialization_status=deferred_until_axis_partitions_frozen`.

Implement `write_phase72b_prepared_artifacts(package, output_dir)` with every
stable prepared filename listed above. `phase72b_feature_rows.csv` contains
sample index, region, unit, row, col, spatial block, origin year, and no outcome
columns. NPZ matrices contain `explicit_static`, `explicit_history`,
`geofm_current`, `geofm_temporal_mean`, `geofm_temporal_full`,
`embedding_history`, and `history_mask`.

`phase72b_feature_manifest.csv` contains only these base matrices at prepare
time. Its control-related fields are blank and its package-level record states
that control manifests will be written by fit-freeze and confirmation after
axis partition selection. `audit_phase72b_splits` must fail if the frozen
protocol does not require partition-local controls or allows Phase 8 D4 reuse.

- [ ] **Step 5: Verify GREEN and commit**

Expected: every Phase 72B test through Task 5 passes.

```powershell
git add src\paper11_geofm\phase72b_splits.py src\paper11_geofm\phase72b_information_gain_screen.py tests\test_phase72b_geofm_information_gain_screen.py
git commit -m "feat: prepare Phase 72B frozen features"
```

---

### Task 6: Metrics, Block Bootstrap, and Gate

**Files:**
- Create: `src/paper11_geofm/phase72b_metrics.py`
- Modify: `tests/test_phase72b_geofm_information_gain_screen.py`

- [ ] **Step 1: Write failing metric and gate tests**

Use hand-computable labels/probabilities:

```python
def test_phase72b_metrics_match_hand_computable_brier_and_ece():
    from paper11_geofm.phase72b_metrics import phase72b_metrics
    result = phase72b_metrics(np.array([0, 1, 1, 0]), np.array([0.1, 0.8, 0.6, 0.2]), threshold=0.5, budgets=(0.10, 0.20), ece_bins=2)
    assert result["brier"] == 0.0625
    assert result["ece"] == 0.225
    assert result["balanced_accuracy"] == 1.0


def test_phase72b_block_bootstrap_uses_paired_blocks():
    from paper11_geofm.phase72b_metrics import paired_block_bootstrap
    rows = [{"region_id": "a", "spatial_block_id": "a0"}, {"region_id": "a", "spatial_block_id": "a0"}, {"region_id": "a", "spatial_block_id": "a1"}, {"region_id": "a", "spatial_block_id": "a1"}]
    y = np.array([0, 1, 0, 1]); explicit = np.array([0.4, 0.6, 0.4, 0.6]); geofm = np.array([0.1, 0.9, 0.2, 0.8])
    result = paired_block_bootstrap(y, explicit, geofm, rows, iterations=100, seed=72)
    assert result["ap_delta_mean"] >= 0
    assert result["brier_delta_mean"] > 0
    assert result["n_clusters"] == 2
```

Add gate fixtures for all four statuses. A supported fixture must satisfy two
practical deltas, a favorable bootstrap interval, all three controls, both
transfer directions, and spatial direction. A mixed fixture passes pooled and
controls but fails one transfer. A not-supported fixture misses pooled AP/Brier
thresholds. An input-not-ready fixture has leakage failure.

- [ ] **Step 2: Verify RED**

Expected: missing `phase72b_metrics`.

- [ ] **Step 3: Implement metrics and bootstrap**

Create:

```python
def expected_calibration_error(y_true, probability, bins=10):
    y = np.asarray(y_true, int); p = np.asarray(probability, float)
    order = np.argsort(p, kind="mergesort"); groups = np.array_split(order, min(int(bins), len(y)))
    return float(sum(len(group) / len(y) * abs(float(y[group].mean()) - float(p[group].mean())) for group in groups if len(group)))


def phase72b_metrics(y_true, probability, *, threshold, budgets, ece_bins):
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
    y = np.asarray(y_true, int); p = np.asarray(probability, float); predicted = (p >= float(threshold)).astype(int)
    result = {
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, ece_bins),
        "roc_auc": float(roc_auc_score(y, p)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
    }
    for budget in budgets:
        k = max(1, int(np.ceil(len(y) * float(budget)))); selected = np.argsort(-p, kind="mergesort")[:k]
        result[f"capture_at_{int(100*budget)}pct"] = float(y[selected].sum() / max(1, y.sum()))
        result[f"precision_at_{int(100*budget)}pct"] = float(y[selected].mean())
    return result
```

`paired_block_bootstrap` samples spatial blocks with replacement, includes all
rows in selected blocks, stratifies by region when both regions occur, skips
replicates lacking both outcome classes, and returns mean and 2.5/97.5
percentiles for favorable AP delta (`GeoFM - explicit`) and favorable Brier
delta (`explicit - GeoFM`). Require at least 80% valid replicates or raise.

- [ ] **Step 4: Implement the frozen gate**

```python
def build_phase72b_gate(*, pooled_delta, pooled_bootstrap, control_rows, transfer_rows, spatial_rows, leakage_ok, gates):
    if not leakage_ok:
        return {"phase72b_status": "phase72b_inputs_not_ready", "reasons": ["leakage audit failed"]}
    practical = sum([
        pooled_delta["ap_delta"] >= gates["ap_vs_explicit"],
        pooled_delta["brier_delta"] >= gates["brier_vs_explicit"],
        pooled_delta["ece_delta"] >= gates["ece_vs_explicit"],
    ]) >= 2
    statistical = pooled_bootstrap["ap_delta_ci_low"] > 0 or pooled_bootstrap["brier_delta_ci_low"] > 0
    controls = bool(control_rows) and all(row["ap_delta"] >= gates["ap_vs_control"] and row["brier_delta"] >= gates["brier_vs_control"] for row in control_rows)
    if not (practical and statistical and controls):
        return {"phase72b_status": "geofm_information_not_supported", "reasons": ["pooled practical/statistical/control gate failed"]}
    transfer = len(transfer_rows) == 2 and all(
        (row["ap_delta"] >= gates["transfer_ap_gain"] or row["brier_delta"] >= gates["transfer_brier_gain"])
        and row["ap_delta"] >= -gates["transfer_ap_harm"]
        and row["brier_delta"] >= -gates["transfer_brier_harm"]
        for row in transfer_rows
    )
    spatial = bool(spatial_rows) and all(row["ap_delta"] >= 0 or row["brier_delta"] >= 0 for row in spatial_rows)
    return {"phase72b_status": "geofm_information_supported" if transfer and spatial else "geofm_information_mixed", "reasons": [] if transfer and spatial else ["spatial or transfer heterogeneity"]}
```

Include the Phase 72B claim boundary and all evaluated booleans/deltas in the
returned gate.

- [ ] **Step 5: Verify GREEN and commit**

Expected: every Phase 72B test through Task 6 passes.

```powershell
git add src\paper11_geofm\phase72b_metrics.py tests\test_phase72b_geofm_information_gain_screen.py
git commit -m "feat: add Phase 72B metrics and gate"
```

---

### Task 7: Model Search, Calibration, and Fit-Freeze Manifest

**Files:**
- Create: `src/paper11_geofm/phase72b_models.py`
- Modify: `src/paper11_geofm/phase72b_information_gain_screen.py`
- Modify: `tests/test_phase72b_geofm_information_gain_screen.py`

- [ ] **Step 1: Write failing model and freeze tests**

Create a deterministic synthetic binary dataset where one GeoFM column adds
signal beyond explicit features. Assert logistic/HGB search returns a fitted
bundle, calibration selection uses validation only, and prediction is finite.

```python
def test_phase72b_model_selection_returns_frozen_bundle():
    from paper11_geofm.phase72b_models import fit_select_phase72b_model, predict_phase72b_bundle
    rng = np.random.default_rng(72); x = rng.normal(size=(120, 3)); y = (x[:, 0] + 0.8 * x[:, 2] > 0).astype(int)
    bundle, rows = fit_select_phase72b_model(x[:80], y[:80], x[80:], y[80:], variant_id="fixture", axis_id="pooled_temporal", protocol=_protocol_payload())
    probability = predict_phase72b_bundle(bundle, x[80:])
    assert bundle["variant_id"] == "fixture"
    assert bundle["calibration_method"] in {"none", "sigmoid", "isotonic"}
    assert len(rows) > 1
    assert np.isfinite(probability).all()
    assert np.allclose(bundle["scaler"].mean_, x[:80].mean(axis=0)) if bundle["model_family"] == "logistic" else True
```

Add a second logistic-only fixture with training features centered at zero and
validation features shifted by `+100`. Assert the frozen scaler mean equals the
training mean, not the combined mean. Add a control-materialization fixture
that fits one axis and asserts every control manifest row has the expected
`axis_id`, exactly one `partition_id`, the frozen seed, a 64-character index
hash, and `cross_partition_count == 0`.

Add an integration test that writes a tiny prepared package, runs
`fit_freeze_phase72b_models`, and verifies:

```python
assert selected["status"] == "phase72b_models_frozen"
assert set(selected["axes"]) >= {"pooled_temporal", "bishan_to_dongxing", "dongxing_to_bishan"}
assert load_hashed_json(paths["selected_models_json"], paths["selected_models_hash"])["status"] == "phase72b_models_frozen"
assert all(len(record["bundle_sha256"]) == 64 for record in selected["bundle_records"])
```

Modify one joblib byte and assert model loading refuses the changed bundle.

- [ ] **Step 2: Verify RED**

Expected: missing `phase72b_models` and fit-freeze API.

- [ ] **Step 3: Implement model bundles and calibration**

Implement candidate expansion directly from the tracked protocol. Logistic
uses `StandardScaler` fitted on training rows and `LogisticRegression` with
`max_iter=2000`, `solver="lbfgs"`, and seed 72. HGB uses the frozen grid and
seed 72 without scaling.

Implement calibration objects as dictionaries:

```python
{"method": "none"}
{"method": "sigmoid", "model": LogisticRegression(C=1e6, max_iter=2000).fit(logit_p[:, None], y_val)}
{"method": "isotonic", "model": IsotonicRegression(out_of_bounds="clip").fit(p_val, y_val)}
```

Clip probabilities to `[1e-6, 1-1e-6]` before logits. Select calibrator by
validation Brier then ECE. Select model candidate by calibrated validation AP,
then Brier, then ECE, then a stable lexical candidate ID. Freeze the best-F1
validation threshold and the validation probability quantiles for 10% and 20%
risk budgets.

Bundle fields:

```text
variant_id, axis_id, model_family, candidate_id, estimator_params,
feature_count, scaler, estimator, calibration_method, calibrator,
f1_threshold, budget_thresholds, validation_metrics, train_index_sha256,
validation_index_sha256, claim_boundary
```

- [ ] **Step 4: Implement `fit_freeze_phase72b_models`**

The orchestration must verify the frozen protocol hash, load development
targets only, load raw embedding histories, and fit:

- full validation search for pooled temporal and both transfer axes;
- five control seeds for each control, freezing the strongest validation seed;
- buffered spatial bundles using each variant's pooled selected candidate
  configuration and fold-specific training/validation rows.

For every axis, construct three explicit partition IDs using the exact strings
`{axis_id}:train`, `{axis_id}:validation`, and `{axis_id}:test`. During
fit-freeze, call `build_phase72b_control_features` separately on the declared
training indexes and validation indexes; never concatenate those rows for a
shuffle. Random-projection matrices are generated from the frozen seed and
input/output dimensions only. Test controls are not materialized or inspected
during fit-freeze. Write fit-freeze control-manifest rows with source index
hashes, matrix hashes, and zero cross-partition counts, and include their hash
in the selected-model manifest.

Write each selected bundle with joblib, record SHA256, and write
`phase72b_validation_metrics.csv` plus hashed selected-model JSON. The selected
manifest includes the frozen protocol SHA256 and refuses any missing required
axis/variant/control.

- [ ] **Step 5: Verify GREEN and commit**

Expected: every Phase 72B test through Task 7 passes.

```powershell
git add src\paper11_geofm\phase72b_models.py src\paper11_geofm\phase72b_information_gain_screen.py tests\test_phase72b_geofm_information_gain_screen.py
git commit -m "feat: freeze Phase 72B model selection"
```

---

### Task 8: Confirmation Evaluation, Artifact Writer, and CLI

**Files:**
- Modify: `src/paper11_geofm/phase72b_information_gain_screen.py`
- Create: `experiments/phase72b_geofm_information_gain_screen/run_phase72b_information_gain_screen.py`
- Modify: `tests/test_phase72b_geofm_information_gain_screen.py`

- [ ] **Step 1: Write failing confirmation and CLI tests**

Build a tiny prepared/frozen fixture and assert confirmation writes every stable
artifact and never changes the selected manifest. Add a subprocess test for all
three modes.

```python
def test_phase72b_confirmation_writes_stable_outputs(tmp_path):
    from paper11_geofm.phase72b_information_gain_screen import confirm_phase72b_information_gain_screen, write_phase72b_confirmation_artifacts
    result = confirm_phase72b_information_gain_screen(prepared_dir=tmp_path / "prepared", frozen_dir=tmp_path / "frozen")
    paths = write_phase72b_confirmation_artifacts(result, tmp_path / "confirm")
    assert set(paths) == {
        "metrics_csv", "predictions_csv", "calibration_csv",
        "bootstrap_csv", "control_csv", "control_manifest_csv", "transfer_csv",
        "screen_json", "screen_md",
    }
    assert result["phase72b_status"] in {
        "phase72b_inputs_not_ready", "geofm_information_not_supported",
        "geofm_information_mixed", "geofm_information_supported",
    }
```

Hash-refusal CLI test: change `phase72b_selected_models.json` after its hash is
written, run `confirm`, assert return code 1 and `hash mismatch` in stderr.
Add a second refusal test that changes a fit-control manifest partition ID or
sets `cross_partition_count` to `1` while leaving the selected manifest
unchanged; confirmation must return `phase72b_inputs_not_ready` before reading
confirmation outcomes or writing model metrics.

- [ ] **Step 2: Verify RED**

Expected: missing confirmation and CLI behavior.

- [ ] **Step 3: Implement confirmation**

`confirm_phase72b_information_gain_screen` must:

1. Verify frozen protocol, selected manifest, and every bundle hash.
2. Load confirmation targets only after verification.
3. For each declared axis and frozen control seed, materialize the test control
   from raw embedding history using exactly one `{axis_id}:test` partition ID;
   reject any nonzero cross-partition count and write
   `phase72b_confirmation_control_manifest.csv`.
4. Evaluate all required bundles on their declared test indexes.
5. Write row-level probabilities with sample, axis, variant, control seed,
   outcome, calibrated probability, threshold, block, region, and origin year.
6. Compute core/secondary metrics, calibration-bin rows, paired block bootstrap,
   control deltas, transfer summaries, and spatial-fold deltas.
7. Call the frozen gate with no alternative metrics or thresholds.
8. Return exact source hashes, counts, invalid folds, blockers, status, next
   action, and claim boundary.

The confirmation result includes both fit and confirmation control-manifest
hashes. A missing row, unexpected partition ID, changed frozen seed, changed
index hash, or nonzero `cross_partition_count` returns
`phase72b_inputs_not_ready` before model metrics are accepted.

The primary comparison is
`explicit_plus_geofm_temporal_full - explicit_history`. Control comparisons are
against the strongest frozen seed for each control family. Missing controls or
axes return `phase72b_inputs_not_ready`, not a partial positive gate.

- [ ] **Step 4: Implement stable writers and runner**

The runner supports:

```text
--mode prepare
--mode fit-freeze
--mode confirm
```

Prepare arguments:

```text
--protocol
--phase72a-region-config
--phase72a-package-dir
--embedding-dir region=path (repeated)
--label-dir region=path (repeated)
--terrain-dir
--output-dir
```

Fit-freeze arguments:

```text
--prepared-dir
--output-dir
```

Confirm arguments:

```text
--prepared-dir
--frozen-dir
--output-dir
```

Use the Phase 72A `region=path` parser pattern. Malformed input, hash mismatch,
or failed mandatory audit exits 1. A scientifically valid negative or mixed
confirmation exits 0 because the experiment completed. Print status, row
counts, hashes, artifact paths, blockers, next action, and claim boundary.

- [ ] **Step 5: Verify GREEN and commit**

Expected: every Phase 72B test through Task 8 passes.

```powershell
git add src\paper11_geofm\phase72b_information_gain_screen.py experiments\phase72b_geofm_information_gain_screen\run_phase72b_information_gain_screen.py tests\test_phase72b_geofm_information_gain_screen.py
git commit -m "feat: confirm Phase 72B information gain"
```

---

### Task 9: Real Terrain, Prepare, Fit-Freeze, Confirm, and Result Note

**Files:**
- Generate ignored outputs below
  `experiments/phase72b_geofm_information_gain_screen/outputs/`.
- Create: `paper/phase28_results/39_phase72b_geofm_information_gain_screen.md`
- Modify: `paper/phase28_results/README.md`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [ ] **Step 1: Verify implementation before network use**

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase72b_geofm_information_gain_screen.py -q --basetemp=D:\tmp\paper11_phase72b_pre_real -p no:cacheprovider
D:\adk\.venv\Scripts\python.exe scripts\smoke_check.py
```

Expected: all Phase 72B tests and smoke check pass.

- [ ] **Step 2: Fetch and inspect real terrain**

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase72b_geofm_information_gain_screen\fetch_phase72b_terrain.py --phase72a-region-config experiments\phase72a_temporal_label_package\phase72a_regions.json --phase72b-protocol experiments\phase72b_geofm_information_gain_screen\phase72b_protocol.json --output-dir experiments\phase72b_geofm_information_gain_screen\outputs\terrain
```

Require two records, zero failures, Bishan `67 x 70`, Dongxing `91 x 99`, all
eight finite features, and 64-character hashes. If Earth Engine aggregation
returns another shape, record `phase72b_inputs_not_ready`; do not crop or pad.

- [ ] **Step 3: Run `prepare` and inspect frozen protocol**

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase72b_geofm_information_gain_screen\run_phase72b_information_gain_screen.py --mode prepare --protocol experiments\phase72b_geofm_information_gain_screen\phase72b_protocol.json --phase72a-region-config experiments\phase72a_temporal_label_package\phase72a_regions.json --phase72a-package-dir experiments\phase72a_temporal_label_package\outputs\bishan_dongxing_esri_2017_2024 --embedding-dir bishan=data\bishan_alphaearth_sample --label-dir bishan=D:\test\paper58-geofm-world-model-rl\data\independent_change_labels\labels --embedding-dir dongxing=D:\test\dongxing_alphaearth --label-dir dongxing=experiments\phase72a_temporal_label_package\outputs\esri_labels --terrain-dir experiments\phase72b_geofm_information_gain_screen\outputs\terrain --output-dir experiments\phase72b_geofm_information_gain_screen\outputs\prepared
```

Inspect matrix shapes, feature names, development/confirmation target year
separation, every split axis, class support, leakage audit, and protocol hash.
Require `split_before_controls=true`, no prepared control matrix, partition-
local control enforcement, training-only learned transformations, and explicit
refusal to reuse Phase 8 D4 tables. Do not continue if any blocker occurs.

- [ ] **Step 4: Run `fit-freeze` without confirmation labels**

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase72b_geofm_information_gain_screen\run_phase72b_information_gain_screen.py --mode fit-freeze --prepared-dir experiments\phase72b_geofm_information_gain_screen\outputs\prepared --output-dir experiments\phase72b_geofm_information_gain_screen\outputs\frozen
```

Inspect required axes/variants, selected model families, hyperparameters,
calibration methods, five-seed controls, bundle hashes, and selected-manifest
hash. Inspect `phase72b_fit_control_manifest.csv`: every row must contain one
train or validation partition, the expected axis and frozen seed, 64-character
index/matrix hashes, and `cross_partition_count=0`. Record the manifest and
selected-model hashes before confirmation.

- [ ] **Step 5: Run confirmation exactly once**

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase72b_geofm_information_gain_screen\run_phase72b_information_gain_screen.py --mode confirm --prepared-dir experiments\phase72b_geofm_information_gain_screen\outputs\prepared --frozen-dir experiments\phase72b_geofm_information_gain_screen\outputs\frozen --output-dir experiments\phase72b_geofm_information_gain_screen\outputs\confirmation
```

Inspect rather than trusting the exit code:

```powershell
Get-Content -Raw experiments\phase72b_geofm_information_gain_screen\outputs\confirmation\phase72b_information_gain_screen.json
Import-Csv experiments\phase72b_geofm_information_gain_screen\outputs\confirmation\phase72b_metrics.csv | Format-Table -AutoSize
Import-Csv experiments\phase72b_geofm_information_gain_screen\outputs\confirmation\phase72b_control_comparison.csv | Format-Table -AutoSize
Import-Csv experiments\phase72b_geofm_information_gain_screen\outputs\confirmation\phase72b_transfer_summary.csv | Format-Table -AutoSize
```

Confirm exact pooled deltas, confidence intervals, control margins, both
transfer directions, spatial folds, invalid-fold coverage, and final status.
Inspect `phase72b_confirmation_control_manifest.csv` and require only declared
test partitions, the frozen control seeds, matching index hashes, and zero
cross-partition counts.

- [ ] **Step 6: Write measured documentation**

Create `39_phase72b_geofm_information_gain_screen.md` with:

- exact terrain source, shapes, and hashes;
- frozen protocol and selected-model hashes;
- development and confirmation row counts;
- selected model/calibrator for the explicit and primary GeoFM variants;
- pooled AP/Brier/ECE and practical deltas;
- bootstrap intervals;
- each control margin and five-seed range;
- fit and confirmation control-manifest hashes, partition IDs, and proof that
  no control crossed a temporal, buffered-spatial, or region boundary;
- both zero-shot directions and buffered spatial results;
- status, blockers, and exact next-phase decision;
- exact reproduction commands and claim boundary.

Add one README index line and append the same measured state to the handoff.
Do not modify the formal manuscript. Do not start any next-stage GeoFM
algorithm design unless the status is exactly `geofm_information_supported`.

- [ ] **Step 7: Verify and commit the measured result**

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase72b_geofm_information_gain_screen.py tests\test_phase72a_temporal_label_package.py tests\test_phase68_external_independent_label_package.py tests\test_phase40_independent_label_gate.py tests\test_phase39_independent_label_audit.py -q --basetemp=D:\tmp\paper11_phase72b_adjacent -p no:cacheprovider
D:\adk\.venv\Scripts\python.exe -m pytest -q --basetemp=D:\tmp\paper11_phase72b_full -p no:cacheprovider
D:\adk\.venv\Scripts\python.exe scripts\smoke_check.py
git diff --check
git diff --name-only HEAD -- paper\submission\final
```

Expected: all tests pass, smoke passes, diff check is clean, and the formal
manuscript command emits nothing.

```powershell
git add experiments\phase72b_geofm_information_gain_screen\*.py experiments\phase72b_geofm_information_gain_screen\phase72b_protocol.json src\paper11_geofm\phase72b_*.py tests\test_phase72b_geofm_information_gain_screen.py paper\phase28_results\39_phase72b_geofm_information_gain_screen.md paper\phase28_results\README.md docs\superpowers\phase33_current_progress_handoff.md
git commit -m "docs: record Phase 72B information-gain result"
```

---

### Task 10: Final Transition Gate

**Files:**
- Verify only unless measured documentation requires a factual correction.

- [ ] **Step 1: Re-read the frozen hashes and final gate**

Verify that the protocol hash recorded before fit-freeze equals the confirmation
protocol hash, the selected-model hash recorded before confirmation equals the
confirmation hash, and every bundle hash still matches.

- [ ] **Step 2: Enforce the next-phase decision**

```text
geofm_information_supported -> a separate reviewed next-stage GeoFM algorithm design may begin; it must include a genuinely spatially coupled planner before any planning-performance claim
geofm_information_mixed -> only the frozen heterogeneity audit may begin
geofm_information_not_supported -> stop the GeoFM-specific planning claim; only generic low-dimensional optimization or a negative-result route may continue
phase72b_inputs_not_ready -> remain in Phase 72B and resolve the measured input/audit blocker
```

No other status permits next-stage GeoFM algorithm development, and no status
automatically enables a suitability reward.

- [ ] **Step 3: Record final repository state**

```powershell
git status --short --branch
git log -8 --oneline --decorate
git rev-parse HEAD
git rev-parse origin/main
```

Record whether the branch is ahead, whether outputs are ignored and present,
and whether `paper/submission/final/*` remains unchanged.

## Plan Self-Review

- The plan covers every approved design requirement: common public terrain,
  strong LULC history, current/mean/full GeoFM, three strict controls with five
  seeds, pooled temporal, buffered spatial, bidirectional transfer, calibration,
  practical thresholds, block bootstrap, freeze hashes, and stable artifacts.
- Development and confirmation labels are stored separately; fit-freeze does
  not receive the confirmation target path.
- The split registry is frozen before data-dependent controls. Fit-freeze
  constructs train and validation controls separately, confirmation constructs
  test controls separately, and every control manifest requires zero cross-
  partition exchanges.
- Random projection is data-independent; any future learned projection must fit
  training rows only. Phase 8 D4 tables are explicitly excluded.
- Fit and confirmation control manifests are separate immutable artifacts whose
  hashes are included in the selected-model and final-result evidence chains.
- Model/calibrator refitting after 2022 is prohibited; confirmation loads frozen
  bundles.
- Spatial folds reuse pooled selected candidate configurations but retrain only
  on fold-allowed 2017-2021 rows and recalibrate only on fold-allowed 2022 rows.
- DLTB, deep temporal neural models, planning, and formal manuscript revision
  are excluded.
- The final transition gate permits a separately reviewed spatially coupled
  planning design only after `geofm_information_supported`; mixed, negative,
  and input-not-ready statuses cannot advance the GeoFM planning claim.
- All code behavior follows red-green-refactor TDD; real values are measured,
  never predeclared.
