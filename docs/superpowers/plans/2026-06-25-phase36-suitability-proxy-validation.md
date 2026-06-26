# Phase 36 Suitability-Proxy Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only Phase 36 diagnostic package that tests whether GeoFM-derived feature families add weak-label suitability signal before B2/B3 reward work.

**Architecture:** Add one focused analysis module, one CLI runner, one unit/CLI test file, and result documentation. The module reads existing CSV feature tables, builds feature-family matrices, runs spatial held-out logistic-regression validation, writes CSV/JSON/Markdown artifacts, and returns a conservative status.

**Tech Stack:** Python standard library (`argparse`, `csv`, `json`, `pathlib`), NumPy, scikit-learn, pytest, existing Paper11 feature-table conventions.

---

## Files

- Create: `src/paper11_geofm/phase36_suitability_proxy_validation.py`
- Create: `experiments/phase36_suitability_proxy_validation/run_phase36_suitability_proxy_validation.py`
- Create: `tests/test_phase36_suitability_proxy_validation.py`
- Create: `paper/phase28_results/10_phase36_suitability_proxy_validation.md`
- Modify: `README.md`
- Modify: `paper/phase28_results/README.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

## Task 1: Add failing Phase 36 module tests

- [ ] **Step 1: Write tests**

Create `tests/test_phase36_suitability_proxy_validation.py` with tests that:

- build tiny aligned B0/B1/B2/D2/D3/D4P8/D4P16 feature tables;
- verify Phase 36 reports label summaries and model rows;
- verify leakage risk is flagged for DLTB-derived labels;
- verify writer artifacts exist;
- verify CLI invocation writes outputs.

- [ ] **Step 2: Run red test**

Run:

```powershell
python -m pytest tests\test_phase36_suitability_proxy_validation.py -q --basetemp=.pytest_tmp_phase36_red -p no:cacheprovider
```

Expected: fail because `paper11_geofm.phase36_suitability_proxy_validation`
does not exist.

## Task 2: Implement Phase 36 analyzer

- [ ] **Step 1: Create module**

Implement `src/paper11_geofm/phase36_suitability_proxy_validation.py` with:

- constants for claim boundary, default labels, fieldnames, family definitions;
- CSV readers keyed by `block_id`;
- feature-family builders for explicit, raw GeoFM, suitability proxy, Phase 8
  controls, and optional normalized controls;
- split handling using `split` column when available;
- logistic-regression validation with standard scaling;
- conservative status rule;
- CSV/JSON/Markdown artifact writers.

- [ ] **Step 2: Run green test**

Run:

```powershell
python -m pytest tests\test_phase36_suitability_proxy_validation.py -q --basetemp=.pytest_tmp_phase36_green1 -p no:cacheprovider
```

Expected: all Phase 36 tests pass.

## Task 3: Add CLI runner

- [ ] **Step 1: Create runner**

Create `experiments/phase36_suitability_proxy_validation/run_phase36_suitability_proxy_validation.py` with arguments:

- `--phase2-output-dir`
- `--phase8-output-dir`
- `--normalized-controls-dir` optional
- `--output-dir`
- `--label-columns`
- `--min-delta`

The runner prints status, artifact paths, usable labels, and claim boundary.

- [ ] **Step 2: Run CLI test**

Run:

```powershell
python -m pytest tests\test_phase36_suitability_proxy_validation.py::test_phase36_cli_writes_outputs -q --basetemp=.pytest_tmp_phase36_cli -p no:cacheprovider
```

Expected: pass.

## Task 4: Run real Bishan Phase 36

- [ ] **Step 1: Execute real diagnostic**

Run from repository root:

```powershell
python experiments\phase36_suitability_proxy_validation\run_phase36_suitability_proxy_validation.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --normalized-controls-dir experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental\derived_normalized_controls --output-dir experiments\phase36_suitability_proxy_validation\outputs\real_bishan --label-columns current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label
```

Expected: exits `0`, writes Phase 36 artifacts, and prints a conservative status.

- [ ] **Step 2: Inspect artifacts**

Read:

```powershell
Get-Content -Raw experiments\phase36_suitability_proxy_validation\outputs\real_bishan\phase36_suitability_proxy_validation.md
Get-Content -Raw experiments\phase36_suitability_proxy_validation\outputs\real_bishan\phase36_suitability_proxy_validation.json
```

Expected: Markdown and JSON summarize labels, feature-family performance, status, and leakage boundary.

## Task 5: Update docs

- [ ] **Step 1: Add result note**

Write `paper/phase28_results/10_phase36_suitability_proxy_validation.md` from the real output. It must state whether Phase 36 supports a bounded B2/B3 reward smoke or keeps suitability reward blocked.

- [ ] **Step 2: Update index docs**

Update `README.md`, `paper/phase28_results/README.md`, `reproducibility/FILE_MANIFEST.tsv`, and `docs/superpowers/phase33_current_progress_handoff.md`.

## Task 6: Verify and commit

- [ ] **Step 1: Run targeted tests**

Run:

```powershell
python -m pytest tests\test_phase36_suitability_proxy_validation.py tests\test_phase9_proxy_validation.py tests\test_phase10_reward_readiness.py -q --basetemp=.pytest_tmp_phase36_final -p no:cacheprovider
```

Expected: all targeted tests pass.

- [ ] **Step 2: Run smoke check**

Run:

```powershell
python scripts\smoke_check.py
```

Expected: smoke check passes.

- [ ] **Step 3: Review diff and commit**

Run:

```powershell
git status --short --branch
git diff -- README.md paper/phase28_results/README.md paper/phase28_results/10_phase36_suitability_proxy_validation.md reproducibility/FILE_MANIFEST.tsv docs/superpowers/phase33_current_progress_handoff.md src/paper11_geofm/phase36_suitability_proxy_validation.py experiments/phase36_suitability_proxy_validation/run_phase36_suitability_proxy_validation.py tests/test_phase36_suitability_proxy_validation.py
git add README.md paper/phase28_results/README.md paper/phase28_results/10_phase36_suitability_proxy_validation.md reproducibility/FILE_MANIFEST.tsv docs/superpowers/phase33_current_progress_handoff.md docs/superpowers/specs/2026-06-25-phase36-suitability-proxy-validation-design.md docs/superpowers/plans/2026-06-25-phase36-suitability-proxy-validation.md src/paper11_geofm/phase36_suitability_proxy_validation.py experiments/phase36_suitability_proxy_validation/run_phase36_suitability_proxy_validation.py tests/test_phase36_suitability_proxy_validation.py
git commit -m "feat: add Phase 36 suitability proxy validation"
```

Expected: commit succeeds.
