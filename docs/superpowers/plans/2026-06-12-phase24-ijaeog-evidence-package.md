# Phase 24 IJAEOG Evidence Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible IJAEOG evidence package that summarizes Phase 22/23 pilot outputs and locks claim-readiness boundaries.

**Architecture:** Add a small parser/aggregator module that reads Phase 22 CSV, Phase 23 CSV, and Phase 23 comparison JSON. The module writes a compact evidence CSV, JSON summary, and Markdown claim-readiness note for submission materials.

**Tech Stack:** Python standard library, pytest, CSV/JSON/Markdown artifact writing.

---

### Task 1: Failing Phase 24 Tests

**Files:**
- Create: `tests/test_phase24_ijaeog_evidence_package.py`

- [ ] **Step 1: Write aggregation tests**

Create tiny Phase 22 and Phase 23 fixture files. Assert:

```python
package = build_phase24_ijaeog_evidence_package(...)
assert package["phase"] == "phase24_ijaeog_evidence_package"
assert package["phase22"]["summary_rows"] == 4
assert package["phase23"]["summary_rows"] == 4
assert package["phase23"]["B1_minus_B0_mean_reward"] == 0.42
assert package["claim_readiness"]["submission_ready"]["status"] == "not_ready"
```

- [ ] **Step 2: Write writer and CLI tests**

Assert that the writer creates:

```text
phase24_ijaeog_evidence_table.csv
phase24_ijaeog_evidence_summary.json
phase24_ijaeog_claim_readiness.md
```

Add a CLI test that prints Phase 22 rows, Phase 23 rows, B1-B0 delta, submission readiness, and claim boundary.

- [ ] **Step 3: Verify RED**

Run:

```powershell
python -m pytest tests\test_phase24_ijaeog_evidence_package.py -q
```

Expected: FAIL because the Phase 24 module and runner do not exist.

### Task 2: Phase 24 Module

**Files:**
- Create: `src/paper11_geofm/ijaeog_evidence_package.py`
- Test: `tests/test_phase24_ijaeog_evidence_package.py`

- [ ] **Step 1: Implement constants and CSV/JSON readers**

Create:

```python
PHASE24_CLAIM_BOUNDARY = "Phase 24 is a synthesis and claim-readiness package ..."
def build_phase24_ijaeog_evidence_package(phase22_summary_csv, phase23_summary_csv, phase23_comparison_json): ...
```

- [ ] **Step 2: Implement aggregation**

Aggregate row counts, variants, policies, tiles, seeds, and mean reward by variant/policy. Preserve Phase 23 `B1_minus_B0_mean_reward`.

- [ ] **Step 3: Implement claim readiness**

Return statuses:

```python
{
  "same_tile_b0_b1_training_pilot": {"status": "pilot_supported"},
  "multi_tile_scorer_interface": {"status": "pilot_supported"},
  "suitability_reward": {"status": "not_ready"},
  "transfer": {"status": "not_ready"},
  "submission_ready": {"status": "not_ready"},
}
```

- [ ] **Step 4: Implement writer**

Write evidence CSV, summary JSON, and Markdown.

- [ ] **Step 5: Verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase24_ijaeog_evidence_package.py -q
```

Expected: PASS.

### Task 3: Phase 24 CLI and Real Run

**Files:**
- Create: `experiments/phase24_ijaeog_evidence_package/run_phase24_ijaeog_evidence_package.py`

- [ ] **Step 1: Add CLI parser**

Flags:

```text
--phase22-summary-csv
--phase23-summary-csv
--phase23-comparison-json
--output-dir
```

- [ ] **Step 2: Run real package**

Run:

```powershell
python experiments\phase24_ijaeog_evidence_package\run_phase24_ijaeog_evidence_package.py --phase22-summary-csv experiments\phase22_multi_tile_scorer_eval\outputs\real_bishan_pilot\phase22_multi_tile_scorer_eval_summary.csv --phase23-summary-csv experiments\phase23_multi_seed_training\outputs\real_bishan_pilot\phase23_multi_seed_training_summary.csv --phase23-comparison-json experiments\phase23_multi_seed_training\outputs\real_bishan_pilot\phase23_multi_seed_training_comparison.json --output-dir experiments\phase24_ijaeog_evidence_package\outputs\real_bishan
```

Expected: evidence table, JSON, and Markdown are written; submission readiness remains `not_ready`.

### Task 4: Documentation and Commit

**Files:**
- Modify: `README.md`
- Modify: `reproducibility/REPRODUCTION_GUIDE.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `paper/submission/01_ijaeog_submission_readiness.md`
- Modify: `paper/submission/02_draft_titles_highlights_declarations.md`

- [ ] **Step 1: Update docs**

Add Phase 24 command and real Bishan expected results. State that Phase 24 is a synthesis artifact, not a new performance experiment.

- [ ] **Step 2: Verify**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests -q
git diff --check
```

- [ ] **Step 3: Commit and push**

Run:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv paper\submission\01_ijaeog_submission_readiness.md paper\submission\02_draft_titles_highlights_declarations.md docs\superpowers\specs\2026-06-12-phase24-ijaeog-evidence-package-design.md docs\superpowers\plans\2026-06-12-phase24-ijaeog-evidence-package.md src\paper11_geofm\ijaeog_evidence_package.py experiments\phase24_ijaeog_evidence_package\run_phase24_ijaeog_evidence_package.py tests\test_phase24_ijaeog_evidence_package.py
git commit -m "Add Phase 24 IJAEOG evidence package"
git push
```
