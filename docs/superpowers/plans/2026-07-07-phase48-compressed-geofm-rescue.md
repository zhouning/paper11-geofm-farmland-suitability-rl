# Phase 48 Compressed GeoFM Rescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only Phase 48 audit that tests `D4P8` and `D4P16` as compressed GeoFM candidate routes.

**Architecture:** The new module consumes existing summary CSV rows, computes tile-seed deltas for compressed candidates against `B0`, `B1`, `D2`, and `D3`, applies explicit status rules, and writes JSON/CSV/Markdown artifacts. The runner is analyze-only and performs no training.

**Tech Stack:** Python standard library, existing `padded_heldout_policy.SUMMARY_FIELDNAMES`, pytest.

---

### Task 1: Lock the Phase 48 API with tests

**Files:**
- Create: `tests/test_phase48_compressed_geofm_rescue.py`

- [x] **Step 1: Write failing tests**

Tests cover supported, partial, not-supported, insufficient, artifact writing, and CLI analyze-only behavior.

- [x] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests\test_phase48_compressed_geofm_rescue.py -q --basetemp=.pytest_tmp_phase48_red -p no:cacheprovider`

Expected: failures from missing `paper11_geofm.phase48_compressed_geofm_rescue` and missing runner.

### Task 2: Implement the audit module and runner

**Files:**
- Create: `src/paper11_geofm/phase48_compressed_geofm_rescue.py`
- Create: `experiments/phase48_compressed_geofm_rescue/run_phase48_compressed_geofm_rescue.py`

- [x] **Step 1: Implement summary loading, coverage checks, delta summaries, status rules, and writers**

The module must keep the claim boundary explicit and exclude `source_rows` from the comparison JSON.

- [x] **Step 2: Implement analyze-only CLI**

Run: `python experiments\phase48_compressed_geofm_rescue\run_phase48_compressed_geofm_rescue.py --existing-summary-csv <summary.csv> --output-dir <output_dir>`

Expected: status and artifact paths printed to stdout.

- [x] **Step 3: Run focused tests**

Run: `python -m pytest tests\test_phase48_compressed_geofm_rescue.py -q --basetemp=.pytest_tmp_phase48_green -p no:cacheprovider`

Expected: `6 passed`.

### Task 3: Generate real Phase 48 artifacts and update docs

**Files:**
- Create: `paper/phase28_results/17_phase48_compressed_geofm_rescue.md`
- Modify: `paper/phase28_results/README.md`
- Modify: `README.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [x] **Step 1: Run real analyze-only audit**

Run: `python experiments\phase48_compressed_geofm_rescue\run_phase48_compressed_geofm_rescue.py --existing-summary-csv experiments\phase28_representation_controls\outputs\real_bishan_4096\phase28_representation_control_summary.csv --output-dir experiments\phase48_compressed_geofm_rescue\outputs\real_bishan_4096`

Expected: `compressed_geofm_route_supported`.

- [x] **Step 2: Update result and repository documentation**

Record that compressed GeoFM is supported as a candidate base-reward representation route while raw B1 and suitability reward remain unsupported.

### Task 4: Verify

**Files:**
- All touched files.

- [x] **Step 1: Run focused and gate tests**

Run Phase 48 tests, Phase 40/41 tests, submission preflight if manuscript or submission files change, smoke check, manifest check, and `git diff --check`.
