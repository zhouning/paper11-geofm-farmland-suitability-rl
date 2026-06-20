# Phase 33 Budget Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bounded experiment-first Phase 33 follow-up that reruns the normalized-B1 and compressed-control variants at a higher training budget and compares them with the existing Phase 30 4096-step result.

**Architecture:** Reuse the existing Phase 30 training and analysis implementation for the new-budget run, then add a small Phase 33 analyzer that compares two Phase 30 comparison JSON files by budget, variant, comparator gap, and tile-seed stability. The Phase 33 runner has a run-and-analyze mode for the higher-budget experiment and an analyze-only mode for comparing already generated Phase 30 outputs.

**Tech Stack:** Python, csv/json/pathlib, existing Paper11 Phase 30 utilities, pytest.

---

### Task 1: Add Phase 33 analysis tests

**Files:**
- Create: `tests/test_phase33_budget_robustness.py`

- [x] **Step 1: Write the failing tests**

Create fixture helpers that write two Phase 30 comparison JSON files with `learned_policy.focal_deltas`, `delta_rows`, `train_timesteps`, and `eval_max_steps`. Add tests:

```python
def test_phase33_builds_budget_transition_and_tile_seed_stability(tmp_path):
    from paper11_geofm.phase33_budget_robustness import (
        PHASE33_BUDGET_ROBUSTNESS_CLAIM_BOUNDARY,
        build_phase33_budget_robustness,
    )

    lower = _write_phase30_comparison(
        tmp_path / "lower" / "phase30_normalized_b1_comparison.json",
        timesteps=4096,
        deltas={
            ("N1Z", "D4P16", "tile_a", 0): -0.3,
            ("N1Z", "D4P16", "tile_a", 1): -0.1,
            ("N1ZR", "D4P16", "tile_a", 0): -0.5,
            ("N1ZR", "D4P16", "tile_a", 1): -0.2,
        },
    )
    higher = _write_phase30_comparison(
        tmp_path / "higher" / "phase30_normalized_b1_comparison.json",
        timesteps=8192,
        deltas={
            ("N1Z", "D4P16", "tile_a", 0): -0.1,
            ("N1Z", "D4P16", "tile_a", 1): 0.2,
            ("N1ZR", "D4P16", "tile_a", 0): -0.4,
            ("N1ZR", "D4P16", "tile_a", 1): -0.1,
        },
    )

    analysis = build_phase33_budget_robustness([lower, higher])

    assert analysis["phase"] == "phase33_budget_robustness"
    assert analysis["claim_boundary"] == PHASE33_BUDGET_ROBUSTNESS_CLAIM_BOUNDARY
    assert analysis["phase33_budget_status"] == "budget_improves_but_not_closed"
    assert analysis["budget_transition_rows"][1]["train_timesteps"] == 8192
    assert analysis["focal_gap_transition_rows"][0]["mean_delta_change_from_previous"] == 0.25
    assert analysis["tile_seed_stability_counts"]["flip_to_positive"] == 1
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests\test_phase33_budget_robustness.py::test_phase33_builds_budget_transition_and_tile_seed_stability -q --basetemp=.pytest_tmp_phase33_red1 -p no:cacheprovider`

Expected: `ModuleNotFoundError` for `paper11_geofm.phase33_budget_robustness`.

- [x] **Step 3: Write minimal implementation**

Create `src/paper11_geofm/phase33_budget_robustness.py` with:

```python
PHASE33_BUDGET_ROBUSTNESS_CLAIM_BOUNDARY = "..."
def build_phase33_budget_robustness(phase30_comparison_json_paths): ...
```

Implement JSON loading, budget ordering, focal gap transition rows, tile-seed stability rows, stability counts, and a conservative status function.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests\test_phase33_budget_robustness.py::test_phase33_builds_budget_transition_and_tile_seed_stability -q --basetemp=.pytest_tmp_phase33_green1 -p no:cacheprovider`

Expected: `1 passed`.

- [x] **Step 5: Commit**

Do not commit until all Phase 33 tasks and real-run verification are complete in this session.

### Task 2: Add writer and CLI coverage

**Files:**
- Modify: `tests/test_phase33_budget_robustness.py`
- Modify: `src/paper11_geofm/phase33_budget_robustness.py`
- Create: `experiments/phase33_budget_robustness/run_phase33_budget_robustness.py`

- [x] **Step 1: Write the failing tests**

Add:

```python
def test_phase33_writer_outputs_csv_json_and_markdown(tmp_path): ...
def test_phase33_cli_analyze_only_writes_outputs(tmp_path): ...
def test_phase33_cli_rejects_single_input(tmp_path): ...
```

Expected artifact names:

```text
phase33_budget_transition.csv
phase33_focal_gap_transition.csv
phase33_tile_seed_stability.csv
phase33_budget_robustness.json
phase33_budget_robustness.md
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests\test_phase33_budget_robustness.py -q --basetemp=.pytest_tmp_phase33_red2 -p no:cacheprovider`

Expected: writer/CLI missing failures.

- [x] **Step 3: Write minimal implementation**

Add `write_phase33_budget_robustness_artifacts(...)` and a CLI runner with:

- `--mode analyze-only`;
- repeated `--phase30-comparison-json`;
- `--output-dir`;
- printed status and artifact paths.

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests\test_phase33_budget_robustness.py -q --basetemp=.pytest_tmp_phase33_green2 -p no:cacheprovider`

Expected: all Phase 33 unit tests pass.

- [x] **Step 5: Commit**

Do not commit until all Phase 33 tasks and real-run verification are complete in this session.

### Task 3: Add run-and-analyze experiment mode

**Files:**
- Modify: `experiments/phase33_budget_robustness/run_phase33_budget_robustness.py`
- Modify: `tests/test_phase33_budget_robustness.py`

- [x] **Step 1: Write the failing CLI test**

Add a monkeypatched test that verifies run-and-analyze passes through to `run_phase30_normalized_b1_ablation` with the requested higher budget and then compares the existing and generated comparison JSONs.

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests\test_phase33_budget_robustness.py::test_phase33_cli_run_and_analyze_reuses_phase30_training -q --basetemp=.pytest_tmp_phase33_red3 -p no:cacheprovider`

Expected: missing run-and-analyze arguments or behavior.

- [x] **Step 3: Write minimal implementation**

The runner should support:

```text
--mode run-and-analyze
--baseline-phase30-comparison-json <existing 4096 comparison>
--phase2-output-dir <dir>
--phase8-output-dir <dir>
--tile-index-csv <csv>
--variants B1,N1Z,N1ZR,D4P8,D4P16
--total-timesteps 8192
--eval-max-steps 8
--seeds 0,1,2
--max-eval-tiles 3
--output-dir <dir>
```

It should call Phase 30, write Phase 30 artifacts to `output_dir/phase30_high_budget`, then run Phase 33 analysis over the baseline comparison and the generated high-budget comparison.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests\test_phase33_budget_robustness.py -q --basetemp=.pytest_tmp_phase33_green3 -p no:cacheprovider`

Expected: all Phase 33 tests pass.

- [x] **Step 5: Commit**

Do not commit until all Phase 33 tasks and real-run verification are complete in this session.

### Task 4: Real bounded experiment and minimal result note

**Files:**
- Create: `paper/phase28_results/07_phase33_budget_robustness.md`
- Modify: `paper/phase28_results/README.md`

- [x] **Step 1: Run real Phase 33**

Run:

```powershell
python experiments\phase33_budget_robustness\run_phase33_budget_robustness.py --mode run-and-analyze --baseline-phase30-comparison-json experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental\phase30_normalized_b1_comparison.json --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B1,N1Z,N1ZR,D4P8,D4P16 --total-timesteps 8192 --eval-max-steps 8 --seeds 0,1,2 --max-eval-tiles 3 --output-dir experiments\phase33_budget_robustness\outputs\real_bishan_8192
```

Expected: exits `0`, prints Phase 33 status, writes Phase 30 high-budget artifacts under `phase30_high_budget/`, and writes Phase 33 analysis artifacts.

- [x] **Step 2: Write minimal result note**

Record only the experiment setup, current status, top transition numbers, and claim boundary in `paper/phase28_results/07_phase33_budget_robustness.md`.

- [x] **Step 3: Add README entry**

Add a one-paragraph Phase 33 reproduction command and expected artifact list to `paper/phase28_results/README.md`.

- [ ] **Step 4: Full verification**

Run:

```powershell
python -m pytest tests\test_phase33_budget_robustness.py tests\test_phase32_action_order_diagnostics.py tests\test_phase30_normalized_b1_ablation.py -q --basetemp=.pytest_tmp_phase33_final -p no:cacheprovider
python scripts\smoke_check.py
git diff --check
```

Expected: tests pass, smoke passes, no diff hygiene errors beyond existing CRLF warnings.

- [ ] **Step 5: Commit**

Stage only tracked source/test/docs/runner/plan files. Do not add ignored real outputs. Commit:

```bash
git commit -m "feat: add Phase 33 budget robustness experiment"
```
