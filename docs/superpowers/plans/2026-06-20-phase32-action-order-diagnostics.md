# Phase 32 Action-Order Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only Phase 32 diagnostic that explains Phase 31 failure cases by comparing focal/comparator action order, cumulative reward trajectories, and selected-block composition within the local tile pool.

**Architecture:** Reuse Phase 31 ranked cases as the case selector, then join Phase 30 N1ZR traces, Phase 28 B1 traces, Phase 30 summary rows, Phase 2 block features, and the Phase 13 tile index. The module should compute per-case step alignment, shared-block order displacement, early-step cumulative reward gaps, and tile-pool composition contrasts, then write CSV/JSON/Markdown artifacts plus a small CLI runner.

**Tech Stack:** Python, csv/json/pathlib, existing Paper11 utilities, pytest.

---

### Task 1: Add failing Phase 32 core-analysis tests

**Files:**
- Create: `tests/test_phase32_action_order_diagnostics.py`

- [x] **Step 1: Write the failing test**

```python
def test_phase32_builds_step_alignment_and_tile_pool_diagnostics(tmp_path):
    from paper11_geofm.phase32_action_order_diagnostics import (
        PHASE32_ACTION_ORDER_CLAIM_BOUNDARY,
        build_phase32_action_order_diagnostics,
    )

    paths = _write_phase32_fixture_inputs(tmp_path)
    analysis = build_phase32_action_order_diagnostics(
        ranked_cases_csv=paths["ranked_cases_csv"],
        focal_traces_json=paths["focal_traces_json"],
        comparator_traces_json=paths["comparator_traces_json"],
        phase2_features_csv=paths["phase2_features_csv"],
        tile_index_csv=paths["tile_index_csv"],
        top_k=2,
    )

    assert analysis["phase"] == "phase32_action_order_diagnostics"
    assert analysis["phase32_action_order_status"] == "action_order_diagnostics_ready"
    assert analysis["claim_boundary"] == PHASE32_ACTION_ORDER_CLAIM_BOUNDARY
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests\test_phase32_action_order_diagnostics.py::test_phase32_builds_step_alignment_and_tile_pool_diagnostics -q --basetemp=.pytest_tmp_phase32_red -p no:cacheprovider`

Expected: `ModuleNotFoundError` for `paper11_geofm.phase32_action_order_diagnostics`.

- [x] **Step 3: Write minimal implementation**

Create the Phase 32 module with:

```python
def build_phase32_action_order_diagnostics(...):
    ...

def write_phase32_action_order_diagnostics_artifacts(...):
    ...
```

The first green target is enough logic to:
- read ranked cases and retain the top-k rows;
- align focal/comparator traces by case;
- compute per-case cumulative reward gaps and shared-block order displacement;
- summarize selected blocks against the local tile block pool.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests\test_phase32_action_order_diagnostics.py::test_phase32_builds_step_alignment_and_tile_pool_diagnostics -q --basetemp=.pytest_tmp_phase32_green1 -p no:cacheprovider`

Expected: `1 passed`.

- [x] **Step 5: Commit**

```bash
git add tests/test_phase32_action_order_diagnostics.py src/paper11_geofm/phase32_action_order_diagnostics.py
git commit -m "feat: add Phase 32 action-order diagnostics core analysis"
```

### Task 2: Add writer and CLI coverage

**Files:**
- Modify: `tests/test_phase32_action_order_diagnostics.py`
- Create: `experiments/phase32_action_order_diagnostics/run_phase32_action_order_diagnostics.py`

- [x] **Step 1: Write the failing tests**

```python
def test_phase32_writer_outputs_csv_json_and_markdown(tmp_path):
    ...
    assert paths["step_alignment_csv"].name == "phase32_step_alignment.csv"
    assert paths["case_summary_csv"].name == "phase32_case_summary.csv"
    assert paths["tile_pool_csv"].name == "phase32_tile_pool_composition.csv"
    assert paths["diagnosis_json"].name == "phase32_action_order_diagnostics.json"
    assert paths["diagnosis_md"].name == "phase32_action_order_diagnostics.md"

def test_phase32_cli_writes_outputs(tmp_path):
    ...
    assert result.returncode == 0
    assert "Phase 32 action-order status: action_order_diagnostics_ready" in result.stdout
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests\test_phase32_action_order_diagnostics.py::test_phase32_writer_outputs_csv_json_and_markdown tests\test_phase32_action_order_diagnostics.py::test_phase32_cli_writes_outputs -q --basetemp=.pytest_tmp_phase32_red2 -p no:cacheprovider`

Expected: missing writer and/or CLI failures.

- [x] **Step 3: Write minimal implementation**

Implement:
- artifact writers in `src/paper11_geofm/phase32_action_order_diagnostics.py`;
- CLI wrapper in `experiments/phase32_action_order_diagnostics/run_phase32_action_order_diagnostics.py`.

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests\test_phase32_action_order_diagnostics.py -q --basetemp=.pytest_tmp_phase32_green2 -p no:cacheprovider`

Expected: all Phase 32 tests pass.

- [x] **Step 5: Commit**

```bash
git add tests/test_phase32_action_order_diagnostics.py experiments/phase32_action_order_diagnostics/run_phase32_action_order_diagnostics.py src/paper11_geofm/phase32_action_order_diagnostics.py
git commit -m "feat: add Phase 32 action-order diagnostics cli"
```

### Task 3: Add Phase 32 result note and README entry

**Files:**
- Create: `paper/phase28_results/06_phase32_action_order_diagnostics.md`
- Modify: `paper/phase28_results/README.md`

- [x] **Step 1: Write the failing documentation expectation**

```python
def test_phase32_markdown_mentions_action_order(tmp_path):
    ...
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests\test_phase32_action_order_diagnostics.py::test_phase32_markdown_mentions_action_order -q --basetemp=.pytest_tmp_phase32_red3 -p no:cacheprovider`

Expected: missing Markdown file/reference failure.

- [x] **Step 3: Write minimal implementation**

Add:
- a bounded result note describing Phase 32 as a read-only explanation phase;
- a README entry and reproduction command for the Phase 32 runner.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests\test_phase32_action_order_diagnostics.py::test_phase32_markdown_mentions_action_order -q --basetemp=.pytest_tmp_phase32_green3 -p no:cacheprovider`

Expected: `1 passed`.

- [x] **Step 5: Commit**

```bash
git add paper/phase28_results/06_phase32_action_order_diagnostics.md paper/phase28_results/README.md tests/test_phase32_action_order_diagnostics.py
git commit -m "docs: record Phase 32 action-order diagnostics"
```

### Task 4: Full verification

**Files:**
- Verify only: `tests/test_phase32_action_order_diagnostics.py`
- Verify only: `src/paper11_geofm/phase32_action_order_diagnostics.py`
- Verify only: `experiments/phase32_action_order_diagnostics/run_phase32_action_order_diagnostics.py`
- Verify only: `paper/phase28_results/06_phase32_action_order_diagnostics.md`

- [x] **Step 1: Run targeted Phase 32 tests**

Run: `python -m pytest tests\test_phase32_action_order_diagnostics.py -q --basetemp=.pytest_tmp_phase32_final -p no:cacheprovider`

Expected: all Phase 32 tests pass.

- [x] **Step 2: Run adjacent regression tests**

Run: `python -m pytest tests\test_phase31_case_diagnostics.py tests\test_phase28_compression_diagnosis.py tests\test_phase29_representation_scale_diagnosis.py tests\test_phase30_normalized_b1_ablation.py -q --basetemp=.pytest_tmp_phase32_regression -p no:cacheprovider`

Expected: all adjacent tests pass.

- [x] **Step 3: Run repository smoke verification**

Run: `python scripts\smoke_check.py`

Expected: `Paper11 smoke check passed.`

- [x] **Step 4: Run diff hygiene check**

Run: `git diff --check`

Expected: no error output, only possible CRLF warnings.

- [x] **Step 5: Run the real read-only Phase 32 diagnostic**

Run: `python experiments\phase32_action_order_diagnostics\run_phase32_action_order_diagnostics.py --ranked-cases-csv experiments\phase31_case_diagnostics\outputs\real_bishan_4096\phase31_ranked_cases.csv --focal-traces-json experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental\phase30_normalized_b1_traces.json --comparator-traces-json experiments\phase28_representation_controls\outputs\real_bishan_4096\phase28_representation_control_traces.json --phase2-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\block_geofm_features.csv --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --output-dir experiments\phase32_action_order_diagnostics\outputs\real_bishan_4096 --top-k 6`

Expected: CLI exits `0`, prints `action_order_diagnostics_ready`, and writes Phase 32 artifacts.
