# Phase 62 D4/D6 Matched PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a bounded Phase 62 matched PPO evaluation comparing D4P8/D4P16 against D6R8/D6R16 under the existing Bishan base-reward held-out protocol.

**Architecture:** Add a focused Phase 62 module that routes D4 variants to Phase 8 features and D6 variants to Phase 61 features, reuses existing padded held-out PPO helpers, and owns D4-vs-D6 analysis/status/writers. Add a thin CLI runner supporting `run-and-analyze` and `analyze-only`. Record real evidence without modifying formal manuscript files.

**Tech Stack:** Python standard library, existing `paper11_geofm.padded_heldout_policy`, existing MaskablePPO training helper from Phase 28, CSV/JSON writers, pytest.

---

## File Structure

- Create `src/paper11_geofm/phase62_d4_d6_matched_ppo.py`.
  Owns constants, contract routing, run/evaluate wrapper, analyze-only logic, D4-vs-D6 delta summaries, cluster/signed-rank summaries, status rules, and artifact writers.
- Create `experiments/phase62_d4_d6_matched_ppo/run_phase62_d4_d6_matched_ppo.py`.
  Exposes `run-and-analyze` and `analyze-only` modes.
- Create `tests/test_phase62_d4_d6_matched_ppo.py`.
  Covers contract routing, analysis status rules, missing coverage, writer outputs, and CLI analyze-only.
- Create `paper/phase28_results/28_phase62_d4_d6_matched_ppo.md` after the real run.
  Records Phase 62 learned-policy evidence only.
- Modify `paper/phase28_results/README.md` and `docs/superpowers/phase33_current_progress_handoff.md` after the real run.

---

### Task 1: Add Contract and Analysis Status Logic

**Files:**
- Create: `tests/test_phase62_d4_d6_matched_ppo.py`
- Create: `src/paper11_geofm/phase62_d4_d6_matched_ppo.py`

- [ ] **Step 1: Write failing tests**

Create tests with helper `_summary_row(variant_id, reward, tile_id="tile_a", seed=0)` matching the existing Phase 25/28 summary schema. Include:

```python
def test_phase62_contract_routes_d4_and_d6_variants(tmp_path):
    from paper11_geofm.phase62_d4_d6_matched_ppo import (
        build_phase62_d4_d6_contract,
    )
    tile_index = _write_csv(tmp_path / "tiles.csv", [
        {"tile_id": "tile_train", "block_ids": "b1;b2;b3;b4"},
        {"tile_id": "tile_a", "block_ids": "b1;b2"},
    ])
    contract = build_phase62_d4_d6_contract(
        phase8_output_dir=tmp_path / "phase8",
        phase61_output_dir=tmp_path / "phase61",
        tile_index_csv=tile_index,
        train_tile_id="tile_train",
        eval_tile_ids="tile_a",
        variants="D4P8,D4P16,D6R8,D6R16",
        seeds="0",
    )
    assert contract["variants"] == ["D4P8", "D4P16", "D6R8", "D6R16"]
    assert contract["variant_source_dirs"]["D4P8"].endswith("phase8")
    assert contract["variant_source_dirs"]["D6R16"].endswith("phase61")
```

Add three analysis tests using complete two-tile/two-seed synthetic rows:

```python
analysis = build_phase62_d4_d6_analysis(rows, metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]})
assert analysis["phase62_d4_d6_status"] == "d4_pca_advantage_over_d6_supported"
assert analysis["matched_deltas"]["D4P8_minus_D6R8"]["mean_delta"] == 0.25
assert analysis["pooled_primary_delta"]["positive_count"] == 8
```

Also cover `d6_random_projection_advantage`, `d4_d6_not_distinguishable`, and `insufficient` when `D6R16` rows are removed.

- [ ] **Step 2: Run tests to verify RED**

```powershell
python -m pytest tests\test_phase62_d4_d6_matched_ppo.py -q --basetemp=.pytest_tmp_phase62_red -p no:cacheprovider
```

Expected: fails with module not found.

- [ ] **Step 3: Implement minimal contract and analysis**

In `src/paper11_geofm/phase62_d4_d6_matched_ppo.py` implement:

- constants:
  - `PHASE62_CLAIM_BOUNDARY`
  - `PHASE62_PRIMARY_VARIANTS = ("D4P8", "D4P16", "D6R8", "D6R16")`
  - `PHASE62_OPTIONAL_VARIANTS = ("D6P8", "D6P16")`
  - `PHASE62_PRIMARY_COMPARISONS = (("D4P8", "D6R8"), ("D4P16", "D6R16"))`
- `build_phase62_d4_d6_contract(...)` using `_select_train_eval_tiles`, `_normalize_seeds`, and source routing: D4 variants to `phase8_output_dir`, D6 variants to `phase61_output_dir`.
- `build_phase62_d4_d6_analysis(summary_rows_or_csv, metadata=None, bootstrap_iterations=5000, random_seed=62)` that:
  - filters `row_type == "trained_policy"`;
  - validates coverage for requested variants/eval tiles/seeds;
  - builds delta rows for all configured D4-vs-D6 comparisons present;
  - computes per-comparison summaries, pooled primary summary, cluster rows, sign-test, and signed-rank summary;
  - assigns status using the spec rules.

Reuse small helper patterns from Phase 59 for metadata parsing, coverage issues, sign-test, signed-rank, and rounding.

- [ ] **Step 4: Run tests to verify GREEN**

```powershell
python -m pytest tests\test_phase62_d4_d6_matched_ppo.py -q --basetemp=.pytest_tmp_phase62_green -p no:cacheprovider
```

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src\paper11_geofm\phase62_d4_d6_matched_ppo.py tests\test_phase62_d4_d6_matched_ppo.py
git commit -m "feat: add Phase 62 D4 D6 analysis logic"
```

---

### Task 2: Add Training Wrapper, Writers, and CLI

**Files:**
- Modify: `src/paper11_geofm/phase62_d4_d6_matched_ppo.py`
- Modify: `tests/test_phase62_d4_d6_matched_ppo.py`
- Create: `experiments/phase62_d4_d6_matched_ppo/run_phase62_d4_d6_matched_ppo.py`

- [ ] **Step 1: Add failing writer/CLI tests**

Append tests asserting:

```python
paths = write_phase62_d4_d6_artifacts({**analysis, "summaries": rows, "traces": {}}, tmp_path / "outputs")
assert paths["summary_csv"].name == "phase62_d4_d6_matched_ppo_summary.csv"
assert paths["delta_csv"].name == "phase62_d4_d6_delta_table.csv"
assert paths["cluster_csv"].name == "phase62_d4_d6_cluster_summary.csv"
assert paths["comparison_json"].name == "phase62_d4_d6_matched_ppo.json"
assert paths["readiness_md"].name == "phase62_d4_d6_matched_ppo.md"
```

Add CLI analyze-only test importing the runner and checking stdout includes:

```text
Phase 62 D4/D6 status: d4_pca_advantage_over_d6_supported
phase62_d4_d6_matched_ppo.json
```

- [ ] **Step 2: Run tests to verify RED**

```powershell
python -m pytest tests\test_phase62_d4_d6_matched_ppo.py -q --basetemp=.pytest_tmp_phase62_writer_red -p no:cacheprovider
```

Expected: writer and CLI missing.

- [ ] **Step 3: Implement writer**

Add `write_phase62_d4_d6_artifacts(analysis, output_dir)` writing:

- `phase62_d4_d6_matched_ppo_summary.csv` using `SUMMARY_FIELDNAMES`;
- `phase62_d4_d6_matched_ppo_traces.json`;
- `phase62_d4_d6_delta_table.csv`;
- `phase62_d4_d6_cluster_summary.csv`;
- `phase62_d4_d6_matched_ppo.json`, excluding raw `summaries` and `traces`;
- `phase62_d4_d6_matched_ppo.md` with status, comparison deltas, pooled delta, cluster summary, signed-rank, and claim boundary.

- [ ] **Step 4: Implement training wrapper**

Add `run_phase62_d4_d6_evaluation(...)` that follows Phase 59:

- build contract;
- for each variant and seed, train with `_train_maskable_ppo_model`;
- evaluate trained, first_valid, and seeded_random policies on each eval tile;
- append optional `existing_summary_csv` rows before analysis;
- return contract + analysis + summaries + traces + dependency metadata.

Use `_load_phase62_tiled_variant_input(contract, tile_id, variant_id)` with `load_tiled_variant_input` and contract source routing.

- [ ] **Step 5: Implement CLI runner**

Create `experiments/phase62_d4_d6_matched_ppo/run_phase62_d4_d6_matched_ppo.py` with flags:

```text
--mode choices run-and-analyze,analyze-only
--phase8-output-dir
--phase61-output-dir
--tile-index-csv
--existing-summary-csv
--output-dir
--variants default D4P8,D4P16,D6R8,D6R16
--train-tile-id
--eval-tile-ids
--max-eval-tiles default 5
--total-timesteps default 4096
--eval-max-steps default 8
--seeds default 0,1,2
--bootstrap-iterations default 5000
--seed default 62
```

Validation:

- `analyze-only` requires `--existing-summary-csv`;
- `run-and-analyze` requires `--phase8-output-dir`, `--phase61-output-dir`, and `--tile-index-csv`.

- [ ] **Step 6: Run tests to verify GREEN**

```powershell
python -m pytest tests\test_phase62_d4_d6_matched_ppo.py -q --basetemp=.pytest_tmp_phase62_writer_green -p no:cacheprovider
```

Expected: all Phase 62 tests pass.

- [ ] **Step 7: Commit Task 2**

```powershell
git add src\paper11_geofm\phase62_d4_d6_matched_ppo.py tests\test_phase62_d4_d6_matched_ppo.py experiments\phase62_d4_d6_matched_ppo\run_phase62_d4_d6_matched_ppo.py
git commit -m "feat: add Phase 62 D4 D6 runner"
```

---

### Task 3: Run Real Phase 62 and Record Evidence

**Files:**
- Generated ignored outputs under `experiments/phase62_d4_d6_matched_ppo/outputs/phase52_full5_seed3/`
- Create: `paper/phase28_results/28_phase62_d4_d6_matched_ppo.md`
- Modify: `paper/phase28_results/README.md`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [ ] **Step 1: Run real Phase 62 primary comparison**

```powershell
python experiments\phase62_d4_d6_matched_ppo\run_phase62_d4_d6_matched_ppo.py --mode run-and-analyze --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --phase61-output-dir experiments\phase61_d6_geofm_projection_controls\outputs\phase52_full5_seed3 --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants D4P8,D4P16,D6R8,D6R16 --eval-tile-ids tile_r002_c003,tile_r005_c004,tile_r005_c003,tile_r000_c004,tile_r001_c004 --total-timesteps 4096 --eval-max-steps 8 --seeds 0,1,2 --bootstrap-iterations 5000 --seed 62 --output-dir experiments\phase62_d4_d6_matched_ppo\outputs\phase52_full5_seed3
```

Expected runtime may be nontrivial because this trains `4 variants x 3 seeds`.

- [ ] **Step 2: Inspect real outputs**

Read:

```powershell
Get-Content -Raw experiments\phase62_d4_d6_matched_ppo\outputs\phase52_full5_seed3\phase62_d4_d6_matched_ppo.json
Get-Content -Raw experiments\phase62_d4_d6_matched_ppo\outputs\phase52_full5_seed3\phase62_d4_d6_delta_table.csv
Get-Content -Raw experiments\phase62_d4_d6_matched_ppo\outputs\phase52_full5_seed3\phase62_d4_d6_cluster_summary.csv
```

Record real status, D4P8-D6R8 mean, D4P16-D6R16 mean, pooled primary mean,
positive count, cluster summaries, and p-values.

- [ ] **Step 3: Create result note**

Create `paper/phase28_results/28_phase62_d4_d6_matched_ppo.md` with:

- real Phase 62 status;
- command;
- primary comparison table;
- pooled and cluster evidence;
- interpretation against Phase 60/61;
- explicit boundary that Phase 62 does not enable suitability reward, B2/B3,
  transfer, independent suitability, or formal manuscript edits.

- [ ] **Step 4: Update README and handoff**

Add README bullet and append handoff section with real command and results.

- [ ] **Step 5: Commit real evidence docs**

```powershell
git add paper\phase28_results\28_phase62_d4_d6_matched_ppo.md paper\phase28_results\README.md docs\superpowers\phase33_current_progress_handoff.md
git commit -m "docs: record Phase 62 D4 D6 evidence"
```

---

### Task 4: Final Verification and Push

- [ ] **Step 1: Run targeted tests**

```powershell
python -m pytest tests\test_phase62_d4_d6_matched_ppo.py tests\test_phase61_d6_geofm_projection_controls.py tests\test_phase59_matched_dimension_controls.py -q --basetemp=.pytest_tmp_phase62_verify -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 2: Run smoke check**

```powershell
python scripts\smoke_check.py
```

Expected: `Paper11 smoke check passed.`

- [ ] **Step 3: Run whitespace check**

```powershell
git diff --check
```

Expected: no output and exit code `0`.

- [ ] **Step 4: Review final git state**

```powershell
git status --short --branch
git log --oneline -10
```

Expected: branch is `main`, no unstaged source/docs edits remain, and local branch is ahead unless already pushed.

- [ ] **Step 5: Push**

```powershell
git push origin main
```

Expected: `main` synchronizes with `origin/main`.