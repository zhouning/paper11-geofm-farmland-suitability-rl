# Phase 49 Compressed Route Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only robustness audit for the Phase 48 compressed GeoFM route.

**Architecture:** Consume Phase 48 delta rows, compute pooled and per-comparison statistical summaries, write JSON/CSV/Markdown artifacts, and expose an analyze-only CLI.

**Tech Stack:** Python standard library, pytest.

---

### Task 1: Tests

**Files:**
- Create: `tests/test_phase49_compressed_route_robustness.py`

- [x] **Step 1: Write failing tests**

Cover robust status, fragile status, artifact writing, and CLI behavior.

- [x] **Step 2: Verify red**

Run: `python -m pytest tests\test_phase49_compressed_route_robustness.py -q --basetemp=.pytest_tmp_phase49_red -p no:cacheprovider`

Expected: failures from missing module and runner.

### Task 2: Implementation

**Files:**
- Create: `src/paper11_geofm/phase49_compressed_route_robustness.py`
- Create: `experiments/phase49_compressed_route_robustness/run_phase49_compressed_route_robustness.py`

- [x] **Step 1: Implement sign-test, bootstrap, leave-one summaries, status rules, and writers**

- [x] **Step 2: Run focused tests**

Run: `python -m pytest tests\test_phase49_compressed_route_robustness.py -q --basetemp=.pytest_tmp_phase49_green -p no:cacheprovider`

Expected: `4 passed`.

### Task 3: Real Audit And Docs

**Files:**
- Create: `paper/phase28_results/18_phase49_compressed_route_robustness.md`
- Modify: `README.md`
- Modify: `paper/phase28_results/README.md`
- Modify: `paper/submission/04_formal_conclusion_manuscript.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [x] **Step 1: Run real Phase 49 audit**

Run: `python experiments\phase49_compressed_route_robustness\run_phase49_compressed_route_robustness.py --phase48-delta-csv experiments\phase48_compressed_geofm_rescue\outputs\real_bishan_4096\phase48_compressed_geofm_rescue_delta_table.csv --output-dir experiments\phase49_compressed_route_robustness\outputs\real_bishan_4096 --bootstrap-iterations 10000 --random-seed 49`

Expected: `compressed_route_statistically_robust`.

- [x] **Step 2: Update manuscript and package docs**

Record the Phase 49 robustness result without expanding suitability-reward or transfer claims.

### Task 4: Verification

- [ ] **Step 1: Run focused tests, submission preflight, smoke check, manifest check, and diff check**
