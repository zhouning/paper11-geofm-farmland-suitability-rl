# Phase 61 D6 GeoFM Projection Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a read-only Phase 61 D6 projection-control generator and geometry audit that prepares GeoFM-derived same-dimension controls for later matched PPO training.

**Architecture:** Add one focused Phase 61 module for D6 feature generation, geometry/similarity auditing, artifact writing, and status assignment. Add a thin experiment runner. Record the real Bishan audit result without modifying formal manuscript files or training policies.

**Tech Stack:** Python standard library, `numpy`, CSV/JSON artifact writers, pytest, existing Paper11 manifest and result-note conventions.

---

## File Structure

- Create `src/paper11_geofm/phase61_d6_geofm_projection_controls.py`.
  This module owns D6 constants, CSV loading, row-alignment checks, projection generation, manifest generation, geometry/similarity auditing, status rules, and artifact writers.
- Create `experiments/phase61_d6_geofm_projection_controls/run_phase61_d6_geofm_projection_controls.py`.
  This CLI builds D6 feature tables and writes the Phase 61 geometry audit from existing B0/B1/D4 CSVs.
- Create `tests/test_phase61_d6_geofm_projection_controls.py`.
  Tests cover D6 feature generation, row-alignment failure, writer outputs, status rules, and CLI behavior.
- Create `paper/phase28_results/27_phase61_d6_geofm_projection_controls.md` after the real run.
  This records Phase 61 feature/geometry evidence only.
- Modify `paper/phase28_results/README.md` and `docs/superpowers/phase33_current_progress_handoff.md` after the real run.
  These records point to Phase 61 outputs and preserve claim boundaries.

---

### Task 1: Add D6 Generation and Geometry Status Logic

**Files:**
- Create: `tests/test_phase61_d6_geofm_projection_controls.py`
- Create: `src/paper11_geofm/phase61_d6_geofm_projection_controls.py`

- [ ] **Step 1: Write the failing generation/status tests**

Create `tests/test_phase61_d6_geofm_projection_controls.py` with fixture helpers and three tests:

```python
import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _row(block_id, explicit_00, embedding_values=None, pca_values=None):
    row = {"block_id": block_id, "explicit_feature_00": explicit_00}
    if embedding_values is not None:
        for index, value in enumerate(embedding_values):
            row[f"embedding_mean_{index:02d}"] = value
    if pca_values is not None:
        for index, value in enumerate(pca_values):
            row[f"embedding_pca_{index:02d}"] = value
    return row


def _b0_rows():
    return [
        _row("b1", 1.0),
        _row("b2", 2.0),
        _row("b3", 3.0),
        _row("b4", 4.0),
        _row("b5", 5.0),
    ]


def _b1_rows():
    return [
        _row("b1", 1.0, [1.0, 0.0, 0.0]),
        _row("b2", 2.0, [0.0, 1.0, 0.0]),
        _row("b3", 3.0, [0.0, 0.0, 1.0]),
        _row("b4", 4.0, [1.0, 1.0, 0.0]),
        _row("b5", 5.0, [0.0, 1.0, 1.0]),
    ]


def _d4p2_rows():
    return [
        _row("b1", 1.0, pca_values=[0.0, 0.4]),
        _row("b2", 2.0, pca_values=[0.8, 0.1]),
        _row("b3", 3.0, pca_values=[-0.4, -0.3]),
        _row("b4", 4.0, pca_values=[0.5, 0.5]),
        _row("b5", 5.0, pca_values=[-0.9, -0.7]),
    ]


def _d4p3_rows():
    return [
        _row("b1", 1.0, pca_values=[0.0, 0.4, 0.2]),
        _row("b2", 2.0, pca_values=[0.8, 0.1, -0.1]),
        _row("b3", 3.0, pca_values=[-0.4, -0.3, 0.3]),
        _row("b4", 4.0, pca_values=[0.5, 0.5, -0.2]),
        _row("b5", 5.0, pca_values=[-0.9, -0.7, -0.2]),
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_phase61_builds_deterministic_d6_projection_controls():
    from paper11_geofm.phase61_d6_geofm_projection_controls import (
        build_phase61_d6_projection_controls,
    )

    protocol = build_phase61_d6_projection_controls(
        b0_rows_or_csv=_b0_rows(),
        b1_rows_or_csv=_b1_rows(),
        d4p8_rows_or_csv=_d4p2_rows(),
        d4p16_rows_or_csv=_d4p3_rows(),
        dimensions=(2, 3),
        seed=61,
    )

    assert protocol["phase"] == "phase61_d6_projection_control_features"
    assert protocol["phase61_d6_projection_status"] == "d6_projection_controls_ready_for_training"
    assert protocol["variant_ids"] == ["D6R2", "D6P2", "D6R3", "D6P3"]
    assert set(protocol["variant_tables"]) == {"D6R2", "D6P2", "D6R3", "D6P3"}
    assert protocol["summary"]["D6R2"]["projection_type"] == "random_orthonormal_raw_b1_projection"
    assert protocol["summary"]["D6P3"]["projection_type"] == "pca_raw_b1_projection"
    assert protocol["geometry_rows"][0]["row_count"] == 5
    assert all(row["explicit_feature_00"] == float(index + 1) for index, row in enumerate(protocol["variant_tables"]["D6P2"]))
    assert "projection_01" in protocol["variant_tables"]["D6R2"][0]
    assert "projection_02" in protocol["variant_tables"]["D6P3"][0]


def test_phase61_rejects_misaligned_block_ids():
    from paper11_geofm.phase61_d6_geofm_projection_controls import (
        build_phase61_d6_projection_controls,
    )

    b1_rows = _b1_rows()
    b1_rows[1] = {**b1_rows[1], "block_id": "different"}

    try:
        build_phase61_d6_projection_controls(
            _b0_rows(), b1_rows, _d4p2_rows(), _d4p3_rows(), dimensions=(2, 3)
        )
    except ValueError as exc:
        assert "aligned block IDs" in str(exc)
    else:
        raise AssertionError("expected row-alignment failure")


def test_phase61_status_blocks_zero_variance_projection():
    from paper11_geofm.phase61_d6_geofm_projection_controls import (
        build_phase61_d6_projection_controls,
    )

    b1_rows = [_row(row["block_id"], row["explicit_feature_00"], [1.0, 1.0, 1.0]) for row in _b0_rows()]
    analysis = build_phase61_d6_projection_controls(
        _b0_rows(), b1_rows, _d4p2_rows(), _d4p3_rows(), dimensions=(2, 3)
    )

    assert analysis["phase61_d6_projection_status"] == "d6_projection_controls_blocked"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests\test_phase61_d6_geofm_projection_controls.py -q --basetemp=.pytest_tmp_phase61_red -p no:cacheprovider
```

Expected result: FAIL with `ModuleNotFoundError` for `paper11_geofm.phase61_d6_geofm_projection_controls`.

- [ ] **Step 3: Implement minimal generation/status logic**

Create `src/paper11_geofm/phase61_d6_geofm_projection_controls.py` with:

- constants `PHASE61_CLAIM_BOUNDARY`, `PHASE61_VARIANT_PREFIXES`, and CSV field helpers;
- `_load_rows()`, `_require_aligned_block_ids()`, `_available_explicit_columns()`, `_numeric_columns()`, `_matrix_for_columns()` helpers modeled on Phase 59/57;
- `build_phase61_d6_projection_controls(...)` public function;
- projection helpers:

```python
def _centered(matrix: np.ndarray) -> np.ndarray:
    return matrix - np.mean(matrix, axis=0, keepdims=True)


def _random_orthonormal_projection(centered: np.ndarray, dimension: int, rng: np.random.Generator) -> np.ndarray:
    random_matrix = rng.standard_normal(size=(centered.shape[1], int(dimension)))
    q, _ = np.linalg.qr(random_matrix)
    return centered @ q[:, : int(dimension)]


def _pca_projection(centered: np.ndarray, dimension: int) -> np.ndarray:
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: int(dimension)].T
    if components.shape[1] < int(dimension):
        padding = np.zeros((centered.shape[0], int(dimension) - components.shape[1]))
        return np.hstack([centered @ components, padding])
    return centered @ components
```

Use variant IDs `D6R{dimension}` and `D6P{dimension}` so tests can use small `2/3` dimensions and the real run can use `8/16`.

Build rows as:

```python
row = {"block_id": source_row["block_id"]}
for explicit column: copy float from B0
for projection column: row[f"projection_{index:02d}"] = float(projected[row_index, index])
```

Create geometry rows with fields:

- `variant_id`
- `projection_type`
- `row_count`
- `projection_dimension`
- `total_centered_variance`
- `raw_variance_retention`
- `effective_rank`
- `positive_variance_column_count`
- `d4_reference_variant_id`
- `d4_mean_abs_column_correlation`
- `claim_boundary`

Status is ready when all variants have expected row count, expected projection dimension, and `total_centered_variance > 0`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase61_d6_geofm_projection_controls.py -q --basetemp=.pytest_tmp_phase61_green -p no:cacheprovider
```

Expected result: `3 passed`.

- [ ] **Step 5: Commit generation/status logic**

Run:

```powershell
git add src\paper11_geofm\phase61_d6_geofm_projection_controls.py tests\test_phase61_d6_geofm_projection_controls.py
git commit -m "feat: add Phase 61 D6 projection status logic"
```

---

### Task 2: Add Writers and CLI

**Files:**
- Modify: `src/paper11_geofm/phase61_d6_geofm_projection_controls.py`
- Modify: `tests/test_phase61_d6_geofm_projection_controls.py`
- Create: `experiments/phase61_d6_geofm_projection_controls/run_phase61_d6_geofm_projection_controls.py`

- [ ] **Step 1: Add failing writer and CLI tests**

Append tests that:

- write all four D6 variant CSVs;
- write `experiment_variants.json`;
- write `phase61_d6_projection_feature_summary.json`;
- write `phase61_d6_projection_geometry.json`;
- write `phase61_d6_projection_geometry.csv`;
- write `phase61_d6_projection_similarity.csv`;
- write `phase61_d6_projection_controls.md`;
- import the CLI runner and execute it against temporary fixture CSVs.

Test assertions should include:

```python
assert paths["manifest"].name == "experiment_variants.json"
assert paths["feature_summary"].name == "phase61_d6_projection_feature_summary.json"
assert paths["geometry_json"].name == "phase61_d6_projection_geometry.json"
assert paths["readiness_md"].name == "phase61_d6_projection_controls.md"
assert (tmp_path / "outputs" / "variant_D6R2_features.csv").exists()
assert saved["phase61_d6_projection_status"] == "d6_projection_controls_ready_for_training"
assert "does not train PPO policies" in readiness_text
assert "Phase 61 D6 projection status: d6_projection_controls_ready_for_training" in stdout
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests\test_phase61_d6_geofm_projection_controls.py -q --basetemp=.pytest_tmp_phase61_writer_red -p no:cacheprovider
```

Expected result: FAIL because writer/CLI functions do not exist.

- [ ] **Step 3: Implement writer**

Add `write_phase61_d6_projection_control_artifacts(protocol, output_dir)` that writes feature CSVs, manifest JSON, feature summary JSON, geometry JSON, geometry CSV, similarity CSV, and readiness Markdown.

Manifest entries must follow the existing convention:

```python
{
  "description": "Explicit planning features plus D6R8 GeoFM projection controls.",
  "state_groups": ["explicit_planning_features", "phase61_d6r8_geofm_projection"],
  "reward": "base_planning_reward",
  "required_columns": explicit_columns + projection_columns,
  "ready": True,
  "missing": [],
  "feature_table": "variant_D6R8_features.csv",
  "row_count": len(rows),
}
```

Markdown should include status, variant summary, geometry rows, similarity rows, output paths, and claim boundary.

- [ ] **Step 4: Implement CLI**

Create `experiments/phase61_d6_geofm_projection_controls/run_phase61_d6_geofm_projection_controls.py` with argparse flags:

```text
--b0-features-csv
--b1-features-csv
--d4p8-features-csv
--d4p16-features-csv
--output-dir
--dimensions default "8,16"
--seed default 61
```

The runner calls `build_phase61_d6_projection_controls(...)`, writes artifacts, prints the status and key output paths, and returns `1` for `OSError`, `RuntimeError`, or `ValueError`.

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase61_d6_geofm_projection_controls.py -q --basetemp=.pytest_tmp_phase61_writer_green -p no:cacheprovider
```

Expected result: all Phase 61 tests pass.

- [ ] **Step 6: Commit writer and CLI**

Run:

```powershell
git add src\paper11_geofm\phase61_d6_geofm_projection_controls.py tests\test_phase61_d6_geofm_projection_controls.py experiments\phase61_d6_geofm_projection_controls\run_phase61_d6_geofm_projection_controls.py
git commit -m "feat: add Phase 61 D6 projection runner"
```

---

### Task 3: Run Real Phase 61 and Record Evidence

**Files:**
- Create ignored outputs under `experiments/phase61_d6_geofm_projection_controls/outputs/phase52_full5_seed3/`
- Create: `paper/phase28_results/27_phase61_d6_geofm_projection_controls.md`
- Modify: `paper/phase28_results/README.md`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [ ] **Step 1: Run real Phase 61**

Run:

```powershell
python experiments\phase61_d6_geofm_projection_controls\run_phase61_d6_geofm_projection_controls.py --b0-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\variant_B0_features.csv --b1-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\variant_B1_features.csv --d4p8-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P8_features.csv --d4p16-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P16_features.csv --output-dir experiments\phase61_d6_geofm_projection_controls\outputs\phase52_full5_seed3 --dimensions 8,16 --seed 61
```

Expected result: `Phase 61 D6 projection status: d6_projection_controls_ready_for_training` unless a real geometry/lineage issue is found.

- [ ] **Step 2: Inspect real outputs**

Read:

```powershell
Get-Content -Raw experiments\phase61_d6_geofm_projection_controls\outputs\phase52_full5_seed3\phase61_d6_projection_geometry.json
Get-Content -Raw experiments\phase61_d6_geofm_projection_controls\outputs\phase52_full5_seed3\phase61_d6_projection_geometry.csv
Get-Content -Raw experiments\phase61_d6_geofm_projection_controls\outputs\phase52_full5_seed3\phase61_d6_projection_similarity.csv
```

Record the actual status, row count, D6P retention ratios, D6R retention ratios, and D4 similarity metrics.

- [ ] **Step 3: Create Phase 61 result note**

Create `paper/phase28_results/27_phase61_d6_geofm_projection_controls.md` with:

- real Phase 61 status;
- generated D6 variants and dimensions;
- row lineage result;
- geometry/similarity table using real values;
- interpretation that D6 controls are ready or blocked for later matched training;
- explicit statement that Phase 61 does not train PPO and does not revise formal manuscript files.

- [ ] **Step 4: Update README and handoff**

Add one README bullet for `27_phase61_d6_geofm_projection_controls.md`.

Append a handoff section with real command, status, real geometry values, output paths, and next-step recommendation.

- [ ] **Step 5: Commit real evidence docs**

Run:

```powershell
git add paper\phase28_results\27_phase61_d6_geofm_projection_controls.md paper\phase28_results\README.md docs\superpowers\phase33_current_progress_handoff.md
git commit -m "docs: record Phase 61 D6 projection evidence"
```

---

### Task 4: Verification and Save

**Files:**
- All Phase 61 implementation, test, runner, and documentation files.

- [ ] **Step 1: Run targeted tests**

Run:

```powershell
python -m pytest tests\test_phase61_d6_geofm_projection_controls.py tests\test_phase60_information_optimization_attribution.py tests\test_phase59_matched_dimension_controls.py -q --basetemp=.pytest_tmp_phase61_verify -p no:cacheprovider
```

Expected result: all selected tests pass.

- [ ] **Step 2: Run smoke check**

Run:

```powershell
python scripts\smoke_check.py
```

Expected result: `Paper11 smoke check passed.`

- [ ] **Step 3: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected result: no output and exit code `0`.

- [ ] **Step 4: Review final git state**

Run:

```powershell
git status --short --branch
git log --oneline -8
```

Expected result: branch is `main`, local branch is ahead by Phase 61 commits unless pushed, and no unstaged source/documentation edits remain.

- [ ] **Step 5: Push after final verification**

Run:

```powershell
git push origin main
```

Expected result: `main` synchronizes with `origin/main`.