# Phase 23 Multi-Seed B0/B1 Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded multi-seed B0/B1 same-tile MaskablePPO training pilot and aggregate comparison report.

**Architecture:** Reuse Phase 20 training/evaluation exactly, loop it across seeds, add seed-rank metadata, and compute conservative aggregate diagnostics. Keep the output separate from Phase 22 cross-tile scorer evaluation so the manuscript can distinguish trained-policy evidence from interface-pilot evidence.

**Tech Stack:** Python, pytest, NumPy, CSV/JSON writers, existing Phase 20 MaskablePPO runner.

---

### Task 1: Failing Phase 23 Tests

**Files:**
- Create: `tests/test_phase23_multi_seed_training.py`

- [ ] **Step 1: Write contract tests**

Add tests that import `build_phase23_multi_seed_training_contract`, write a two-tile Phase 13 fixture, and assert:

```python
assert contract["phase"] == "phase23_multi_seed_training"
assert contract["train_tile_id"] == "tile_r000_c001"
assert contract["eval_tile_id"] == "tile_r000_c001"
assert contract["seeds"] == [0, 1]
assert contract["seed_ranks"] == {"0": 1, "1": 2}
assert contract["variants"] == ["B0", "B1"]
```

Also test that B3 is rejected and distinct eval tiles remain blocked.

- [ ] **Step 2: Write run, writer, and CLI tests**

Use a small Phase 2 fixture and `pytest.importorskip` for `stable_baselines3` and `sb3_contrib`. Assert that two seeds, two variants, and three policies produce 12 rows:

```python
protocol = run_phase23_multi_seed_training(..., variants=("B0", "B1"), seeds=(0, 1))
assert protocol["summary_count"] == 12
assert protocol["comparison"]["learned_policy"]["B1_minus_B0_mean_reward"] is not None
```

Add writer assertions for:

```text
phase23_multi_seed_training_summary.csv
phase23_multi_seed_training_traces.json
phase23_multi_seed_training_comparison.json
```

Add a CLI test that verifies train tile, seeds, variants, row count, comparison JSON path, and claim boundary in stdout.

- [ ] **Step 3: Verify RED**

Run:

```powershell
python -m pytest tests\test_phase23_multi_seed_training.py -q
```

Expected: FAIL because the Phase 23 module and runner do not exist.

### Task 2: Phase 23 Module

**Files:**
- Create: `src/paper11_geofm/multi_seed_training.py`
- Test: `tests/test_phase23_multi_seed_training.py`

- [ ] **Step 1: Implement contract**

Create:

```python
PHASE23_CLAIM_BOUNDARY = "Phase 23 is a bounded multi-seed same-tile B0/B1 MaskablePPO training pilot ..."

def build_phase23_multi_seed_training_contract(...):
    seeds = _normalize_seeds(seeds)
    phase20_contract = build_phase20_bounded_training_contract(...)
    return {...}
```

Use Phase 20 tile selection and B0/B1 validation so distinct learned-policy evaluation remains blocked.

- [ ] **Step 2: Implement seed loop**

Create:

```python
def run_phase23_multi_seed_training(...):
    for seed_rank, seed in enumerate(contract["seeds"], start=1):
        phase20 = run_phase20_bounded_tiled_training(..., seed=seed)
        for row in phase20["summaries"]:
            row = _phase23_summary(row, seed_rank)
```

Store traces as `traces[row_type][variant_id][str(seed)]`.

- [ ] **Step 3: Implement comparison aggregation**

Compute mean reward by variant for `trained_policy`, `first_valid`, and `seeded_random`. Compute `B1_minus_B0_mean_reward` for trained policy when both variants are present. Include `remaining_evidence_gaps`.

- [ ] **Step 4: Implement artifact writer**

Write CSV, traces JSON, and comparison JSON.

- [ ] **Step 5: Verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase23_multi_seed_training.py -q
```

Expected: PASS.

### Task 3: Phase 23 CLI

**Files:**
- Create: `experiments/phase23_multi_seed_training/run_phase23_multi_seed_training.py`
- Test: `tests/test_phase23_multi_seed_training.py`

- [ ] **Step 1: Add CLI parser**

Add flags:

```text
--phase2-output-dir
--tile-index-csv
--variants
--train-tile-id
--eval-tile-id
--total-timesteps
--eval-max-steps
--seeds
--output-dir
```

- [ ] **Step 2: Print reviewer-facing summary**

Print train tile, evaluation tile, seeds, variants, total timesteps, summary rows, learned-policy B1-B0 mean reward delta, artifact paths, and claim boundary.

- [ ] **Step 3: Verify CLI test**

Run:

```powershell
python -m pytest tests\test_phase23_multi_seed_training.py::test_phase23_cli_writes_outputs_and_prints_summary -q
```

Expected: PASS.

### Task 4: Real Run and Docs

**Files:**
- Modify: `README.md`
- Modify: `reproducibility/REPRODUCTION_GUIDE.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `paper/submission/01_ijaeog_submission_readiness.md`
- Modify: `paper/submission/02_draft_titles_highlights_declarations.md`

- [ ] **Step 1: Run real Phase 23 pilot**

Run:

```powershell
python experiments\phase23_multi_seed_training\run_phase23_multi_seed_training.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --total-timesteps 8 --eval-max-steps 4 --seeds 0,1,2 --output-dir experiments\phase23_multi_seed_training\outputs\real_bishan_pilot
```

Expected: 18 summary rows and comparison JSON.

- [ ] **Step 2: Update docs**

Add Phase 23 to README, reproduction guide, file manifest, and submission materials. State that Phase 23 adds multi-seed learned-policy evidence but still does not establish transfer or suitability-reward claims.

### Task 5: Verification and Commit

**Files:**
- Stage only source, tests, docs, runner, spec, and plan files.

- [ ] **Step 1: Run verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests -q
git diff --check
git status --short --ignored=matching experiments\phase23_multi_seed_training
```

- [ ] **Step 2: Commit and push**

Run:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv paper\submission\01_ijaeog_submission_readiness.md paper\submission\02_draft_titles_highlights_declarations.md docs\superpowers\specs\2026-06-12-phase23-multi-seed-b0-b1-training-design.md docs\superpowers\plans\2026-06-12-phase23-multi-seed-b0-b1-training.md src\paper11_geofm\multi_seed_training.py experiments\phase23_multi_seed_training\run_phase23_multi_seed_training.py tests\test_phase23_multi_seed_training.py
git commit -m "Add Phase 23 multi-seed training evidence"
git push
```
