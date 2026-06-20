# Phase 31 Case Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only Phase 31 diagnostic that ranks informative tile-seed cases from Phase 30 and summarizes their selected blocks, reward components, and tile geometry for follow-up spatial inspection.

**Architecture:** Reuse the existing Phase 30 summary and trace artifacts as the primary evidence source, then join them with Phase 11 block feature rows, Phase 11 block-pixel mappings, and the Phase 13 tile index. The module should not run PPO training or alter reward logic; it should only compute case rankings, per-case block summaries, and tile geometry summaries, then write CSV/JSON/Markdown artifacts plus a small CLI wrapper.

**Tech Stack:** Python, csv/json/pathlib, existing Paper11 utilities, pytest.

---

### Task 1: Add failing Phase 31 behavior tests

**Files:**
- Create: `tests/test_phase31_case_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
def test_phase31_builds_ranked_case_tables(tmp_path):
    from paper11_geofm.phase31_case_diagnostics import (
        PHASE31_CASE_DIAGNOSTICS_CLAIM_BOUNDARY,
        build_phase31_case_diagnostics,
    )

    paths = _write_phase31_fixture_inputs(tmp_path)
    analysis = build_phase31_case_diagnostics(
        summary_csv=paths["summary_csv"],
        traces_json=paths["traces_json"],
        phase2_features_csv=paths["phase2_features_csv"],
        tile_index_csv=paths["tile_index_csv"],
        block_mapping_csv=paths["block_mapping_csv"],
        top_k=2,
    )

    assert analysis["phase"] == "phase31_case_diagnostics"
    assert analysis["phase31_case_diagnostic_status"] == "case_diagnostics_ready"
    assert analysis["claim_boundary"] == PHASE31_CASE_DIAGNOSTICS_CLAIM_BOUNDARY
    assert [row["case_id"] for row in analysis["ranked_case_rows"]] == [
        "tile_good|1|N1ZR|B1",
        "tile_bad|2|N1ZR|B1",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests\test_phase31_case_diagnostics.py::test_phase31_builds_ranked_case_tables -q --basetemp=.pytest_tmp_phase31_red -p no:cacheprovider`

Expected: `ModuleNotFoundError` for `paper11_geofm.phase31_case_diagnostics`.

- [ ] **Step 3: Write minimal implementation**

Create the Phase 31 module with:

```python
def build_phase31_case_diagnostics(...):
    ...

def write_phase31_case_diagnostics_artifacts(...):
    ...
```

The first green target is enough logic to:
- load summary/traces/features/tile-index/mapping inputs;
- rank candidate cases from a chosen focal delta (default `N1ZR` vs `B1`);
- summarize selected blocks and tile geometry for the selected cases.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests\test_phase31_case_diagnostics.py::test_phase31_builds_ranked_case_tables -q --basetemp=.pytest_tmp_phase31_green1 -p no:cacheprovider`

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_phase31_case_diagnostics.py src/paper11_geofm/phase31_case_diagnostics.py
git commit -m "feat: add Phase 31 case diagnostics core analysis"
```

### Task 2: Add artifact-writer and CLI coverage

**Files:**
- Modify: `tests/test_phase31_case_diagnostics.py`
- Create: `experiments/phase31_case_diagnostics/run_phase31_case_diagnostics.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_phase31_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase31_case_diagnostics import (
        build_phase31_case_diagnostics,
        write_phase31_case_diagnostics_artifacts,
    )
    ...
    assert paths["ranked_cases_csv"].name == "phase31_ranked_cases.csv"
    assert paths["selected_blocks_csv"].name == "phase31_selected_blocks.csv"
    assert paths["tile_geometry_csv"].name == "phase31_tile_geometry.csv"
    assert paths["diagnosis_json"].name == "phase31_case_diagnostics.json"
    assert paths["diagnosis_md"].name == "phase31_case_diagnostics.md"

def test_phase31_cli_writes_outputs(tmp_path):
    ...
    assert result.returncode == 0
    assert "Phase 31 case diagnostic status: case_diagnostics_ready" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests\test_phase31_case_diagnostics.py::test_phase31_writer_outputs_csv_json_and_markdown tests\test_phase31_case_diagnostics.py::test_phase31_cli_writes_outputs -q --basetemp=.pytest_tmp_phase31_red2 -p no:cacheprovider`

Expected: missing writer and/or CLI failures.

- [ ] **Step 3: Write minimal implementation**

Implement:
- CSV/JSON/Markdown writers in `src/paper11_geofm/phase31_case_diagnostics.py`;
- CLI wrapper in `experiments/phase31_case_diagnostics/run_phase31_case_diagnostics.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests\test_phase31_case_diagnostics.py::test_phase31_writer_outputs_csv_json_and_markdown tests\test_phase31_case_diagnostics.py::test_phase31_cli_writes_outputs -q --basetemp=.pytest_tmp_phase31_green2 -p no:cacheprovider`

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_phase31_case_diagnostics.py experiments/phase31_case_diagnostics/run_phase31_case_diagnostics.py src/paper11_geofm/phase31_case_diagnostics.py
git commit -m "feat: add Phase 31 case diagnostics artifacts and cli"
```

### Task 3: Add the reviewer-facing result note

**Files:**
- Create: `paper/phase28_results/05_phase31_case_diagnostics.md`
- Modify: `paper/phase28_results/README.md`

- [ ] **Step 1: Write the failing repository-level expectation**

```python
def test_repository_layout_includes_phase31_outputs():
    ...
```

Or, if repository-layout coverage is too broad, add a focused assertion in:

```python
def test_phase31_markdown_mentions_case_diagnostics(tmp_path):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests\test_phase31_case_diagnostics.py::test_phase31_markdown_mentions_case_diagnostics -q --basetemp=.pytest_tmp_phase31_red3 -p no:cacheprovider`

Expected: missing Markdown file/reference failure.

- [ ] **Step 3: Write minimal implementation**

Add:
- a bounded Phase 31 result note describing the new case-diagnostic role;
- a README entry and reproduction command for the read-only Phase 31 runner.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests\test_phase31_case_diagnostics.py::test_phase31_markdown_mentions_case_diagnostics -q --basetemp=.pytest_tmp_phase31_green3 -p no:cacheprovider`

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add paper/phase28_results/05_phase31_case_diagnostics.md paper/phase28_results/README.md tests/test_phase31_case_diagnostics.py
git commit -m "docs: record Phase 31 case diagnostics boundary"
```

### Task 4: Full verification

**Files:**
- Verify only: `tests/test_phase31_case_diagnostics.py`
- Verify only: `src/paper11_geofm/phase31_case_diagnostics.py`
- Verify only: `experiments/phase31_case_diagnostics/run_phase31_case_diagnostics.py`
- Verify only: `paper/phase28_results/05_phase31_case_diagnostics.md`

- [ ] **Step 1: Run targeted Phase 31 tests**

Run: `python -m pytest tests\test_phase31_case_diagnostics.py -q --basetemp=.pytest_tmp_phase31_final -p no:cacheprovider`

Expected: all Phase 31 tests pass.

- [ ] **Step 2: Run adjacent regression tests**

Run: `python -m pytest tests\test_phase28_compression_diagnosis.py tests\test_phase29_representation_scale_diagnosis.py tests\test_phase30_normalized_b1_ablation.py -q --basetemp=.pytest_tmp_phase31_regression -p no:cacheprovider`

Expected: all adjacent diagnostic tests pass.

- [ ] **Step 3: Run repository smoke verification**

Run: `python scripts\smoke_check.py`

Expected: `Paper11 smoke check passed.`

- [ ] **Step 4: Run diff hygiene check**

Run: `git diff --check`

Expected: no output.

- [ ] **Step 5: Run real read-only Phase 31 diagnostic**

Run: `python experiments\phase31_case_diagnostics\run_phase31_case_diagnostics.py --summary-csv experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental\phase30_normalized_b1_summary.csv --traces-json experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental\phase30_normalized_b1_traces.json --phase2-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\block_geofm_features.csv --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --block-mapping-csv experiments\phase11_bishan_dltb_real\outputs\adapter\block_pixel_mapping.csv --output-dir experiments\phase31_case_diagnostics\outputs\real_bishan_4096 --top-k 6`

Expected: CLI exits `0`, prints `case_diagnostics_ready`, and writes Phase 31 artifacts.
