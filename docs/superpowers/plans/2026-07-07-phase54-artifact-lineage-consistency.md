# Phase 54 Artifact Lineage Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only artifact-lineage audit proving that the formal Phase 52/53 compressed GeoFM evidence values are internally reproducible from one authoritative artifact chain.

**Architecture:** The Phase 54 module consumes the authoritative Phase 48 delta CSV, Phase 50 cluster CSV, Phase 51 JSON, and Phase 53 JSON. It recomputes cluster rows, signed-rank values, and cluster-mean support values using existing Phase 50/51/53 modules, compares values within tolerance, and writes JSON/CSV/Markdown audit artifacts.

**Tech Stack:** Python standard library, pytest, existing `paper11_geofm` Phase 50/51/53 modules, existing Markdown manuscript pipeline.

---

### Task 1: Tests

**Files:**
- Create: `tests/test_phase54_artifact_lineage_consistency.py`

- [x] Write a failing test that builds fixture Phase 48 delta rows, authoritative Phase 50 cluster rows, Phase 51 JSON values, and Phase 53 JSON values, then expects `artifact_lineage_consistent`.
- [x] Write a failing mismatch test that perturbs one authoritative cluster value and expects `artifact_lineage_inconsistent`.
- [x] Write a failing CLI test that verifies JSON, CSV, and Markdown outputs.
- [x] Run `python -m pytest tests\test_phase54_artifact_lineage_consistency.py -q --basetemp=.pytest_tmp_phase54_red -p no:cacheprovider` and confirm failure because the module/runner do not exist.

### Task 2: Module And CLI

**Files:**
- Create: `src/paper11_geofm/phase54_artifact_lineage_consistency.py`
- Create: `experiments/phase54_artifact_lineage_consistency/run_phase54_artifact_lineage_consistency.py`

- [x] Implement `build_phase54_artifact_lineage_consistency(...)` to load the four authoritative inputs, recompute Phase 50/51/53 values, and produce check rows.
- [x] Implement a writer that emits `phase54_artifact_lineage_consistency.json`, `phase54_artifact_lineage_checks.csv`, and `phase54_artifact_lineage_consistency.md`.
- [x] Implement the analyze-only CLI with explicit input paths and output directory.
- [x] Run `python -m pytest tests\test_phase54_artifact_lineage_consistency.py -q --basetemp=.pytest_tmp_phase54_green -p no:cacheprovider` and confirm the focused tests pass.

### Task 3: Real Audit

**Files:**
- Generated only under ignored `experiments/phase54_artifact_lineage_consistency/outputs/phase52_full5_seed3/`

- [x] Run the real Phase 54 CLI over the authoritative Phase 52/53 artifacts.
- [x] Confirm status is `artifact_lineage_consistent`.
- [x] Record key values: recomputed cluster count, mean cluster delta, Phase 51 signed-rank p, and Phase 53 sign-flip p/bootstrap CI.

### Task 4: Documentation And Manuscript Package

**Files:**
- Create: `paper/phase28_results/23_phase54_artifact_lineage_consistency.md`
- Modify: `paper/phase28_results/README.md`
- Modify: `README.md`
- Modify: `paper/submission/04_formal_conclusion_manuscript.md`
- Modify: `paper/submission/final/Paper11_formal_conclusion_manuscript.md`
- Modify: `paper/submission/final/Paper11_formal_conclusion_manuscript.docx`
- Modify: `paper/submission/final/Paper11_cover_letter_and_declarations.md`
- Modify: `paper/submission/final/Paper11_cover_letter_and_declarations.docx`
- Modify: `paper/submission/final/README.md`
- Modify: `paper/submission/final/Paper11_phase46_submission_bundle.zip`
- Modify: `paper/submission/final/Paper11_phase46_submission_contents_sha256.txt`
- Modify: `paper/submission/final/Paper11_phase46_submission_bundle_sha256.txt`
- Modify: `scripts/paper11_submission_preflight.py`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [x] Add reviewer-facing Phase 54 result documentation.
- [x] Add one bounded manuscript sentence that Phase 54 verifies the formal Phase 52/53 artifact lineage.
- [x] Rebuild DOCX exports, bundle, checksums, and preflight JSON.
- [x] Keep wording bounded to artifact lineage consistency, not suitability reward or transfer.

### Task 5: Verification And Delivery

**Files:**
- Modify plan checklist after successful steps.

- [x] Run `python -m pytest tests\test_phase54_artifact_lineage_consistency.py tests\test_phase53_cluster_mean_support.py tests\test_phase51_cluster_magnitude_support.py tests\test_phase50_cluster_level_robustness.py tests\test_phase49_compressed_route_robustness.py tests\test_phase48_compressed_geofm_rescue.py tests\test_paper11_submission_preflight.py -q --basetemp=.pytest_tmp_phase54_final -p no:cacheprovider`.
- [x] Run `python scripts\paper11_submission_preflight.py --root . --json-out paper\submission\final\Paper11_phase47_submission_preflight.json`.
- [x] Run `python scripts\smoke_check.py`.
- [x] Run manifest path check and NUL byte check.
- [x] Run `git diff --check`.
- [ ] Commit and push to `origin/main`.
