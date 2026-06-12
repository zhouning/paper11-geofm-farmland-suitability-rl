# Phase 22 Multi-Tile Seed Scorer Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded Phase 22 multi-tile, multi-seed evaluator for the Phase 21 per-block scorer.

**Architecture:** Reuse Phase 21's standardized ridge-linear scorer and deterministic rollout helpers, but wrap them in a Phase 22 contract that selects one train tile and several distinct evaluation tiles. Write one summary row per variant, evaluation tile, seed, and policy, with an explicit pilot-only claim boundary.

**Tech Stack:** Python, NumPy, Gymnasium environment wrapper from `Phase4InputContractEnv`, pytest, CSV/JSON artifact writers.

---

### Task 1: Failing Phase 22 Tests

**Files:**
- Create: `tests/test_phase22_multi_tile_scorer_eval.py`

- [ ] **Step 1: Write the failing test fixture and contract tests**

Create a fixture that writes five Phase 2-ready rows and a three-tile Phase 13 index:

```python
def _write_tile_index(path: Path) -> Path:
    writer.writerow({"tile_id": "tile_r000_c001", "block_ids": "b1;b3;b5"})
    writer.writerow({"tile_id": "tile_r000_c002", "block_ids": "b2;b4"})
    writer.writerow({"tile_id": "tile_r000_c000", "block_ids": "b6"})
```

Add tests that import `build_phase22_multi_tile_scorer_eval_contract`, expect the largest tile as train tile, the next largest distinct tiles as evaluation tiles, integer seeds `[0, 1]`, B0/B1-only variant normalization, and `eval_tile_ranks` keyed by tile ID.

- [ ] **Step 2: Write failing rollout, writer, and CLI tests**

Add tests for:

```python
protocol = run_phase22_multi_tile_scorer_eval(..., variants=("B0", "B1"), max_eval_tiles=2, seeds=(0, 1))
assert len(protocol["summaries"]) == 24
assert all("eval_tile_rank" in row for row in protocol["summaries"])
```

Add writer assertions for `phase22_multi_tile_scorer_eval_summary.csv` and `phase22_multi_tile_scorer_eval_traces.json`. Add a CLI test that imports `experiments/phase22_multi_tile_scorer_eval/run_phase22_multi_tile_scorer_eval.py` and verifies printed train tile, evaluation tiles, seeds, variants, row count, artifact paths, and claim boundary.

- [ ] **Step 3: Verify RED**

Run:

```powershell
python -m pytest tests\test_phase22_multi_tile_scorer_eval.py -q
```

Expected: FAIL because `paper11_geofm.multi_tile_scorer_eval` and the Phase 22 runner do not exist yet.

### Task 2: Phase 22 Core Module

**Files:**
- Create: `src/paper11_geofm/multi_tile_scorer_eval.py`
- Test: `tests/test_phase22_multi_tile_scorer_eval.py`

- [ ] **Step 1: Implement contract and normalization helpers**

Create:

```python
PHASE22_CLAIM_BOUNDARY = "Phase 22 is a bounded multi-tile, multi-seed per-block scorer evaluation pilot ..."

def build_phase22_multi_tile_scorer_eval_contract(...):
    normalized_variants = _normalize_variants(variants)
    seeds = _normalize_seeds(seeds)
    selected = _select_train_eval_tiles(Path(tile_index_csv), train_tile_id, eval_tile_ids, max_eval_tiles)
    return {...}
```

Use default variants `("B0", "B1")`, default seeds `(0, 1)`, and default `max_eval_tiles=2`. Reject non-positive `eval_max_steps`, negative `ridge_alpha`, empty seeds, missing train tiles, missing eval tiles, same train/eval tiles, and B2/B3 variants.

- [ ] **Step 2: Implement multi-tile rollout aggregation**

Create:

```python
def run_phase22_multi_tile_scorer_eval(...):
    contract = build_phase22_multi_tile_scorer_eval_contract(...)
    for variant_id in contract["variants"]:
        train_tiled = load_tiled_variant_input(...)
        scorer, metadata = _fit_ridge_block_scorer(...)
        for eval_tile_id in contract["eval_tile_ids"]:
            for seed in contract["seeds"]:
                learned_summary, learned_steps = _evaluate_learned_scorer(...)
                baseline_summary, baseline_steps = _evaluate_baseline_policy(...)
```

Add `eval_tile_rank`, replace the Phase 21 claim boundary with the Phase 22 claim boundary on every row, and store traces as `traces[row_type][variant_id][eval_tile_id][str(seed)]`.

- [ ] **Step 3: Implement artifact writer**

Create:

```python
SUMMARY_FIELDNAMES = ["row_type", "variant_id", "train_tile_id", "eval_tile_id", "eval_tile_rank", ...]
def write_phase22_multi_tile_scorer_eval_artifacts(protocol, output_dir):
    ...
```

Write CSV and JSON with names `phase22_multi_tile_scorer_eval_summary.csv` and `phase22_multi_tile_scorer_eval_traces.json`.

- [ ] **Step 4: Verify GREEN for Phase 22 tests**

Run:

```powershell
python -m pytest tests\test_phase22_multi_tile_scorer_eval.py -q
```

Expected: PASS.

### Task 3: Phase 22 CLI Runner

**Files:**
- Create: `experiments/phase22_multi_tile_scorer_eval/run_phase22_multi_tile_scorer_eval.py`
- Test: `tests/test_phase22_multi_tile_scorer_eval.py`

- [ ] **Step 1: Implement CLI parser**

Add flags:

```text
--phase2-output-dir
--tile-index-csv
--variants
--train-tile-id
--eval-tile-ids
--max-eval-tiles
--seeds
--ridge-alpha
--eval-max-steps
--output-dir
```

Parse comma-separated variants, evaluation tile IDs, and seeds.

- [ ] **Step 2: Print reviewer-facing summary**

Print train tile, evaluation tiles, seeds, variants, ridge alpha, evaluation max steps, summary row count, completion flag, artifact paths, and claim boundary.

- [ ] **Step 3: Verify CLI test**

Run:

```powershell
python -m pytest tests\test_phase22_multi_tile_scorer_eval.py::test_phase22_cli_writes_outputs_and_prints_summary -q
```

Expected: PASS.

### Task 4: Documentation Updates

**Files:**
- Modify: `README.md`
- Modify: `reproducibility/REPRODUCTION_GUIDE.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `paper/submission/01_ijaeog_submission_readiness.md`
- Modify: `paper/submission/02_draft_titles_highlights_declarations.md`

- [ ] **Step 1: Add README Phase 22 entry**

Add the runner path, command, expected real Bishan behavior, and guarded claim language after Phase 21.

- [ ] **Step 2: Add reproduction guide Phase 24**

Insert a Phase 22 section before the design-inspection section, then renumber later sections. State expected artifact names and that the protocol is pilot evidence only.

- [ ] **Step 3: Add file manifest rows**

Add rows for the Phase 22 module, runner, tests, and superpowers design/plan documents.

- [ ] **Step 4: Update submission readiness text**

Update the readiness table and safe-current-claim paragraph to include Phase 22 as multi-tile, multi-seed scorer pilot evidence, while keeping planning-performance and suitability-reward claims not ready.

### Task 5: Real Run and Verification

**Files:**
- Generated only: `experiments/phase22_multi_tile_scorer_eval/outputs/real_bishan_pilot/`

- [ ] **Step 1: Run real Phase 22 pilot**

Run:

```powershell
python experiments\phase22_multi_tile_scorer_eval\run_phase22_multi_tile_scorer_eval.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --ridge-alpha 1e-6 --eval-max-steps 4 --seeds 0,1 --max-eval-tiles 2 --output-dir experiments\phase22_multi_tile_scorer_eval\outputs\real_bishan_pilot
```

Expected: summary rows `24`, train tile `tile_r003_c003`, two distinct evaluation tiles, all evaluations completed `True`.

- [ ] **Step 2: Run repository verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests -q
git diff --check
git ls-files --others --exclude-standard
git status --short --ignored=matching experiments\phase22_multi_tile_scorer_eval
```

Expected: smoke check passes, pytest passes, whitespace check exits 0, generated outputs remain ignored.

### Task 6: Commit and Push

**Files:**
- Stage only source, tests, docs, runner, and superpowers spec/plan files.

- [ ] **Step 1: Inspect diff and status**

Run:

```powershell
git status --short
git diff --stat
```

- [ ] **Step 2: Commit**

Run:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv paper\submission\01_ijaeog_submission_readiness.md paper\submission\02_draft_titles_highlights_declarations.md docs\superpowers\specs\2026-06-11-phase22-multi-tile-seed-scorer-eval-design.md docs\superpowers\plans\2026-06-12-phase22-multi-tile-seed-scorer-eval.md src\paper11_geofm\multi_tile_scorer_eval.py experiments\phase22_multi_tile_scorer_eval\run_phase22_multi_tile_scorer_eval.py tests\test_phase22_multi_tile_scorer_eval.py
git commit -m "Add Phase 22 multi-tile scorer evaluation"
```

- [ ] **Step 3: Push**

Run:

```powershell
git push
```

Expected: `main` pushed to `origin/main`.
