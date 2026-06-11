# Phase 21 Cross-Tile Per-Block Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded cross-tile B0/B1 learned block-scorer pilot that trains on one tile and evaluates on a distinct tile without depending on flat observation shape.

**Architecture:** Add a focused `paper11_geofm.cross_tile_block_scorer` module that reuses Phase 13 tile metadata, Phase 14 tiled inputs, Phase 4 evaluation rollouts, and Phase 19 base reward scoring. The learned pilot is a standardized ridge-linear per-block scorer trained on train-tile block features and base rewards, then evaluated on a distinct tile by scoring valid blocks independently. The phase writes CSV/JSON pilot artifacts with strict claim boundaries and keeps B2/B3 suitability reward disabled.

**Tech Stack:** Python standard library, NumPy, Gymnasium environment contract already in the repo, pytest.

---

## File Structure

- Create `src/paper11_geofm/cross_tile_block_scorer.py`: Phase 21 constants, tile selection, B0/B1 validation, ridge-linear block scorer, cross-tile learned rollout, deterministic baseline rollouts, artifact writers, and model metadata.
- Create `experiments/phase21_cross_tile_block_scorer/run_phase21_cross_tile_block_scorer.py`: CLI runner.
- Create `tests/test_phase21_cross_tile_block_scorer.py`: tests for contract summary, variant rejection, ridge scorer behavior, cross-tile rollout, writer, and CLI.
- Modify `README.md`: add Phase 21 command and key entry points.
- Modify `reproducibility/REPRODUCTION_GUIDE.md`: add Phase 21 reproduction section.
- Modify `reproducibility/FILE_MANIFEST.tsv`: add Phase 21 spec, plan, module, CLI, and tests.
- Modify `paper/submission/01_ijaeog_submission_readiness.md`: record Phase 21 as cross-tile pilot evidence, not final performance evidence.
- Modify `paper/submission/02_draft_titles_highlights_declarations.md`: update guarded highlights and blocked-claim wording.

## Task 1: Contract and Guard Tests

- [ ] **Step 1: Write failing tests**

Create `tests/test_phase21_cross_tile_block_scorer.py` with fixtures that write ready Phase 2 B0/B1/B2/B3 feature tables and a two-tile index. Add tests for:

- default train tile is largest and evaluation tile is the next largest distinct tile;
- requested variants are restricted to B0/B1;
- B3 is rejected with a B0/B1 base-reward message.

- [ ] **Step 2: Run RED**

Run:

```powershell
python -m pytest tests\test_phase21_cross_tile_block_scorer.py -q
```

Expected: fail because `paper11_geofm.cross_tile_block_scorer` does not exist.

## Task 2: Core Contract Module

- [ ] **Step 1: Implement contract helpers**

Create `src/paper11_geofm/cross_tile_block_scorer.py` with:

- `PHASE21_CLAIM_BOUNDARY`;
- `SUMMARY_FIELDNAMES`;
- `build_phase21_cross_tile_scorer_contract(...)`;
- `_select_train_eval_tiles(...)`;
- `_normalize_variants(...)`;
- `_read_tile_rows(...)`.

The default train tile must be the largest tile. The default evaluation tile
must be the largest distinct tile. Explicit train/evaluation IDs must be known
and distinct.

- [ ] **Step 2: Run contract tests**

Run:

```powershell
python -m pytest tests\test_phase21_cross_tile_block_scorer.py -q
```

Expected: contract and guard tests pass; scorer/evaluation tests still fail
until Task 3.

## Task 3: Ridge Scorer and Cross-Tile Evaluation

- [ ] **Step 1: Add failing scorer/evaluation tests**

Extend `tests/test_phase21_cross_tile_block_scorer.py` with tests that:

- fit a ridge scorer on a three-block train tile;
- evaluate on a one-block distinct tile with a different flat observation shape;
- produce one `learned_block_scorer`, one `first_valid`, and one
  `seeded_random` row per B0/B1 variant;
- preserve the cross-tile claim boundary and selected block IDs.

- [ ] **Step 2: Implement scorer and rollouts**

Add:

- `run_phase21_cross_tile_block_scorer(...)`;
- `_fit_ridge_block_scorer(...)`;
- `_base_reward_targets(...)`;
- `_evaluate_learned_scorer(...)`;
- `_evaluate_baseline_policy(...)`;
- `_valid_actions(...)`;
- `_rng_for(...)`;
- `_round_float(...)`.

The scorer should standardize train features, solve a ridge-linear system with
an intercept column, and score evaluation blocks using train statistics. It
must raise a clear `ValueError` if train and evaluation per-block feature
dimensions differ.

- [ ] **Step 3: Run focused tests**

Run:

```powershell
python -m pytest tests\test_phase21_cross_tile_block_scorer.py -q
```

Expected: all Phase 21 tests pass except writer/CLI tests if they have not yet
been added.

## Task 4: Writer, CLI, and Docs

- [ ] **Step 1: Add writer and CLI tests**

Test that `write_phase21_cross_tile_block_scorer_artifacts(...)` writes:

- `phase21_cross_tile_block_scorer_summary.csv`;
- `phase21_cross_tile_block_scorer_traces.json`.

Test that the CLI prints train tile, evaluation tile, variants, summary rows,
artifact paths, and claim boundary.

- [ ] **Step 2: Implement writer and CLI**

Create the runner with flags:

- `--phase2-output-dir`;
- `--tile-index-csv`;
- `--variants`;
- `--train-tile-id`;
- `--eval-tile-id`;
- `--ridge-alpha`;
- `--eval-max-steps`;
- `--seed`;
- `--output-dir`.

- [ ] **Step 3: Update docs and manifest**

Document the Phase 21 command and pilot-only boundary in README, reproduction
guide, file manifest, and submission readiness materials.

## Task 5: Real Run and Verification

- [ ] **Step 1: Run real Phase 21 pilot**

Run:

```powershell
python experiments\phase21_cross_tile_block_scorer\run_phase21_cross_tile_block_scorer.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --ridge-alpha 1e-6 --eval-max-steps 4 --seed 0 --output-dir experiments\phase21_cross_tile_block_scorer\outputs\real_bishan_pilot
```

Expected: the runner writes both artifacts and reports completed B0/B1
cross-tile learned-scorer and baseline rows. Numerical rewards are pilot
diagnostics only.

- [ ] **Step 2: Run full verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests -q
git diff --check
```

Expected: smoke check passes, tests pass, and diff check reports no whitespace
errors.

## Self-Review

- Spec coverage: plan covers Phase 21 design requirements, outputs, guardrails,
  docs, real run, and verification.
- Placeholder scan: no `TBD` or incomplete implementation steps remain.
- Type consistency: function names, artifact filenames, CLI flags, and
  claim-boundary fields are consistent across tasks.
