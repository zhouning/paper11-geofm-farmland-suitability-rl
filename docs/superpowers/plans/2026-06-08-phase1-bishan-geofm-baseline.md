# Phase 1 Bishan GeoFM Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, lightweight Phase 1 experiment that converts the included Bishan AlphaEarth sample into region-level GeoFM features, bounded suitability proxy scores, and JSON/CSV artifacts.

**Architecture:** Add a focused `src/paper11_geofm` package with separate modules for sample loading, deterministic region labels, feature aggregation, suitability scoring, and artifact writing. Add `experiments/phase1_bishan_baseline/run_phase1.py` as the executable entry point and cover it with pytest tests using the included sample data and temporary output directories.

**Tech Stack:** Python standard library, NumPy, pytest, Git.

---

### Task 1: Plan Registration

**Files:**
- Create: `docs/superpowers/plans/2026-06-08-phase1-bishan-geofm-baseline.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`

- [ ] **Step 1: Save this implementation plan**

Create this file with the exact plan content.

- [ ] **Step 2: Register the plan in the manifest**

Add this line to `reproducibility/FILE_MANIFEST.tsv`:

```text
docs/superpowers/plans/2026-06-08-phase1-bishan-geofm-baseline.md	plan	Implementation plan for the Phase 1 Bishan GeoFM baseline.
```

- [ ] **Step 3: Verify manifest formatting**

Run:

```powershell
python scripts\smoke_check.py
```

Expected: smoke check exits 0 and reports the Bishan sample years and embedding shape.

### Task 2: Sample Loading Tests and Module

**Files:**
- Create: `tests/test_phase1_geofm.py`
- Create: `src/paper11_geofm/__init__.py`
- Create: `src/paper11_geofm/sample_data.py`

- [ ] **Step 1: Write the failing sample-loader tests**

Add tests that import `paper11_geofm.sample_data`, load metadata from `data/bishan_alphaearth_sample`, and assert years `2017-2024`, embedding dimension `64`, and base-year embedding shape `(67, 70, 64)`.

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
python -m pytest tests\test_phase1_geofm.py::test_load_bishan_metadata_and_base_embedding -v
```

Expected: FAIL because `paper11_geofm` does not exist.

- [ ] **Step 3: Implement sample loading**

Create `sample_data.py` with:

```python
def load_metadata(sample_dir: Path) -> dict: ...
def load_embedding(sample_dir: Path, year: int, mmap: bool = True) -> np.ndarray: ...
def load_annual_embeddings(sample_dir: Path, years: Iterable[int]) -> dict[int, np.ndarray]: ...
```

Validation must check that metadata years are 2017-2024, embedding dimension is 64, and loaded arrays have final dimension 64.

- [ ] **Step 4: Run the test to verify it passes**

Run:

```powershell
python -m pytest tests\test_phase1_geofm.py::test_load_bishan_metadata_and_base_embedding -v
```

Expected: PASS.

### Task 3: Deterministic Region and Feature Aggregation

**Files:**
- Modify: `tests/test_phase1_geofm.py`
- Create: `src/paper11_geofm/regions.py`
- Create: `src/paper11_geofm/features.py`

- [ ] **Step 1: Write failing region and feature tests**

Add tests that create grid labels for shape `(67, 70)` with `5 x 5` bins, assert the labels cover the full grid, and assert region feature rows include `embedding_mean_00` through `embedding_mean_63`.

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest tests\test_phase1_geofm.py::test_grid_regions_cover_full_embedding tests\test_phase1_geofm.py::test_region_features_have_expected_schema -v
```

Expected: FAIL because region and feature modules do not exist.

- [ ] **Step 3: Implement deterministic grid regions**

Create `regions.py` with:

```python
def make_grid_region_labels(grid_shape: tuple[int, int], n_row_bins: int = 5, n_col_bins: int = 5) -> np.ndarray: ...
def iter_region_bounds(labels: np.ndarray) -> list[dict[str, int]]: ...
```

Region IDs must be contiguous integers starting at 0 and every pixel must receive exactly one ID.

- [ ] **Step 4: Implement feature aggregation**

Create `features.py` with:

```python
def compute_region_features(base_embedding: np.ndarray, labels: np.ndarray, annual_embeddings: Mapping[int, np.ndarray] | None = None) -> list[dict[str, float | int]]: ...
```

Rows must include bounds, pixel count, 64 embedding mean columns, `embedding_std_mean`, and `temporal_stability`.

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```powershell
python -m pytest tests\test_phase1_geofm.py::test_grid_regions_cover_full_embedding tests\test_phase1_geofm.py::test_region_features_have_expected_schema -v
```

Expected: PASS.

### Task 4: Suitability Proxy and Artifact Writer

**Files:**
- Modify: `tests/test_phase1_geofm.py`
- Create: `src/paper11_geofm/suitability.py`
- Create: `src/paper11_geofm/artifacts.py`

- [ ] **Step 1: Write failing suitability and artifact tests**

Add tests that compute bounded `suitability_proxy` values in `[0, 1]`, write `region_features.csv` and `summary.json` to a temp directory, and assert the summary contains a claim boundary that rejects direct soil, fertility, or irrigation measurement.

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest tests\test_phase1_geofm.py::test_suitability_proxy_is_bounded tests\test_phase1_geofm.py::test_artifacts_are_written_with_claim_boundary -v
```

Expected: FAIL because suitability and artifact modules do not exist.

- [ ] **Step 3: Implement suitability scoring**

Create `suitability.py` with:

```python
def add_suitability_proxy(rows: list[dict[str, float | int]]) -> list[dict[str, float | int]]: ...
```

Use cosine similarity to the mean region embedding centroid plus an inverse dispersion term, then min-max scale the combined score into `[0, 1]`.

- [ ] **Step 4: Implement artifact writing**

Create `artifacts.py` with:

```python
CLAIM_BOUNDARY = "..."
def write_phase1_artifacts(rows: Sequence[Mapping[str, object]], output_dir: Path, summary: Mapping[str, object]) -> dict[str, Path]: ...
```

Write deterministic CSV column order and JSON with `claim_boundary`, `suitability_min`, `suitability_max`, and `suitability_mean`.

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```powershell
python -m pytest tests\test_phase1_geofm.py::test_suitability_proxy_is_bounded tests\test_phase1_geofm.py::test_artifacts_are_written_with_claim_boundary -v
```

Expected: PASS.

### Task 5: Experiment Runner, Docs, and Manifest

**Files:**
- Modify: `tests/test_phase1_geofm.py`
- Create: `experiments/phase1_bishan_baseline/run_phase1.py`
- Modify: `README.md`
- Modify: `reproducibility/REPRODUCTION_GUIDE.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `scripts/smoke_check.py`
- Modify: `tests/test_repository_layout.py`

- [ ] **Step 1: Write failing runner test**

Add a test that imports `run_phase1`, calls `run_phase1.main(["--output-dir", tmp_path])`, and asserts the command returns 0 and writes both artifacts.

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
python -m pytest tests\test_phase1_geofm.py::test_phase1_runner_writes_artifacts -v
```

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement the runner**

Create `run_phase1.py` with CLI options:

```text
--sample-dir
--base-year
--row-bins
--col-bins
--output-dir
```

Default output directory must be `experiments/phase1_bishan_baseline/outputs/`.

- [ ] **Step 4: Update docs and smoke checks**

Document the command in README and the reproduction guide. Add the new package, experiment, tests, and generated artifact policy to the file manifest. Add smoke-check coverage for the runner path.

- [ ] **Step 5: Run the runner and tests**

Run:

```powershell
python experiments\phase1_bishan_baseline\run_phase1.py
python scripts\smoke_check.py
python -m pytest tests
```

Expected: all commands exit 0.

### Task 6: Commit and Push

**Files:**
- All implementation, tests, docs, and manifest files from Tasks 1-5.

- [ ] **Step 1: Inspect working tree**

Run:

```powershell
git status --short --branch
git diff --check
```

Expected: only intended Phase 1 files changed and no whitespace errors.

- [ ] **Step 2: Commit**

Run:

```powershell
git add .
git commit -m "Add Phase 1 Bishan GeoFM baseline"
```

- [ ] **Step 3: Push feature branch**

Run:

```powershell
git push -u origin paper11-phase1-bishan-baseline
```

- [ ] **Step 4: Report verification evidence**

Report the exact commands that passed, the commit SHA, branch name, and artifact paths.
