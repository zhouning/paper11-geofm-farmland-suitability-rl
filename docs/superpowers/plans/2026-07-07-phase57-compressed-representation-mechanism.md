# Phase 57 Compressed Representation Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only compressed-representation mechanism audit and update the formal manuscript with the resulting evidence.

**Architecture:** Implement a focused Phase 57 module that reads existing feature and delta artifacts, computes representation geometry and diagnostic reward-gain associations, writes JSON/CSV/Markdown outputs, and exposes a CLI runner. Update the manuscript only after the real audit has run.

**Tech Stack:** Python standard library, NumPy, pytest, Pandoc, pdflatex.

---

### Task 1: Define Phase 57 behavior with tests

**Files:**
- Create: `tests/test_phase57_compressed_representation_mechanism.py`

- [ ] Write tests for aligned feature geometry, variance retention, effective rank, status selection, artifact writing, and CLI execution.
- [ ] Run `python -m pytest tests/test_phase57_compressed_representation_mechanism.py -q` and verify the tests fail because the Phase 57 module does not exist.

### Task 2: Implement Phase 57 module and CLI

**Files:**
- Create: `src/paper11_geofm/phase57_compressed_representation_mechanism.py`
- Create: `experiments/phase57_compressed_representation_mechanism/run_phase57_compressed_representation_mechanism.py`

- [ ] Implement CSV loading, block alignment, numeric matrix extraction, covariance eigenvalue geometry, reward-gain summary, tile-level diagnostics, JSON/CSV/Markdown writers, and CLI argument parsing.
- [ ] Run the Phase 57 test file and verify it passes.

### Task 3: Run real Phase 57 audit

**Files:**
- Create output files under `experiments/phase57_compressed_representation_mechanism/outputs/phase52_full5_seed3/`

- [ ] Run the CLI on the real B1/D4P8/D4P16 feature tables and expanded Phase 52 delta table.
- [ ] Inspect the JSON/Markdown outputs and record the status and key values.

### Task 4: Update manuscript and reproducibility records

**Files:**
- Modify: `paper/submission/final/Paper11_formal_conclusion_manuscript.md`
- Modify: `paper/submission/final/Paper11_formal_conclusion_manuscript.tex`
- Modify: `paper/submission/final/Paper11_formal_conclusion_manuscript.pdf`
- Modify: `paper/phase28_results/README.md`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`

- [ ] Add Phase 57 mechanism evidence to Methods, Results, and Discussion without expanding the claim beyond a diagnostic mechanism audit.
- [ ] Fix existing wording bugs in the formal manuscript.
- [ ] Regenerate LaTeX and PDF from Markdown.

### Task 5: Verify and commit

**Files:** all changed files

- [ ] Run the Phase 57 test file.
- [ ] Run relevant existing phase tests for Phase 48/53/54.
- [ ] Run `git diff --check`.
- [ ] Run `pdflatex` twice and inspect the log for warnings.
- [ ] Extract PDF text and confirm expected sections and no Markdown heading literals.
- [ ] Commit and push the completed Phase 57 work.