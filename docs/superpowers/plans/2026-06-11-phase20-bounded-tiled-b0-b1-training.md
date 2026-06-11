# Phase 20 Bounded Same-Tile B0/B1 Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded real-data same-tile MaskablePPO training/evaluation pilot for B0/B1 tiled Paper11 variants.

**Architecture:** Build a focused `paper11_geofm.bounded_tiled_training` module that reuses Phase 4 environments and Phase 13 tile metadata. The module selects the largest train tile by default, evaluates the learned policy and deterministic baselines on the same tile, rejects distinct learned-policy evaluation tiles under the current flat observation design, and writes CSV/JSON artifacts with a strict pilot-only claim boundary.

**Tech Stack:** Python standard library, NumPy, Gymnasium, stable-baselines3/sb3-contrib MaskablePPO, pytest.

---

## File Structure

- Create `src/paper11_geofm/bounded_tiled_training.py`: Phase 20 constants, tile selection, B0/B1 validation, MaskablePPO training, deterministic evaluation rollout, baseline rollout, artifact writers, dependency metadata.
- Create `experiments/phase20_bounded_tiled_training/run_phase20_bounded_tiled_training.py`: CLI runner.
- Create `tests/test_phase20_bounded_tiled_training.py`: tests for contract summary, variant rejection, tiny training run, writer, and CLI.
- Modify `README.md`: add Phase 20 command and key entry point.
- Modify `reproducibility/REPRODUCTION_GUIDE.md`: add Phase 20 reproduction section.
- Modify `reproducibility/FILE_MANIFEST.tsv`: add Phase 20 spec, plan, module, CLI, and tests.
- Modify `paper/submission/01_ijaeog_submission_readiness.md`: note that Phase 20 is a bounded same-tile pilot, not final submission evidence.

## Task 1: Contract and Guard Tests

- [ ] **Step 1: Write failing tests**

Create `tests/test_phase20_bounded_tiled_training.py` with fixtures that write ready Phase 2 B0/B1/B2/B3 feature tables and a two-tile index. Add tests for:

- default train tile is largest and evaluation tile is the same tile;
- a distinct explicit evaluation tile is rejected with the variable-shape
  learned-policy blocker;
- requested variants are restricted to B0/B1;
- B3 is rejected with a suitability-reward message.

- [ ] **Step 2: Run RED**

Run:

```powershell
python -m pytest tests\test_phase20_bounded_tiled_training.py -q
```

Expected: fail because `paper11_geofm.bounded_tiled_training` does not exist.

## Task 2: Core Module

- [ ] **Step 1: Implement contract helpers**

Create `src/paper11_geofm/bounded_tiled_training.py` with:

- `PHASE20_CLAIM_BOUNDARY`;
- `SUMMARY_FIELDNAMES`;
- `build_phase20_bounded_training_contract(...)`;
- `_select_train_eval_tiles(...)`;
- `_normalize_variants(...)`;
- `_read_tile_rows(...)`.

- [ ] **Step 2: Run contract tests**

Run:

```powershell
python -m pytest tests\test_phase20_bounded_tiled_training.py -q
```

Expected: contract and guard tests pass; training tests still fail until Task 3.

## Task 3: Tiny Training and Evaluation

- [ ] **Step 1: Add failing tiny-training test**

Add a test guarded by `pytest.importorskip("stable_baselines3")` and
`pytest.importorskip("sb3_contrib")` that runs B0/B1 on a tiny tile fixture with
`total_timesteps=8` and `eval_max_steps=2`.

- [ ] **Step 2: Implement training/evaluation**

Add:

- `run_phase20_bounded_tiled_training(...)`;
- `_train_maskableppo(...)`;
- `_evaluate_trained_policy(...)`;
- `_evaluate_baseline_policy(...)`;
- `_valid_actions(...)`;
- `_round_float(...)`;
- dependency metadata helpers.

The implementation must train and evaluate only B0/B1 by default, use same-tile
learned-policy evaluation, and write pilot diagnostics only. Cross-tile learned
evaluation must remain blocked until a later variable-size, padded, or
per-block policy design exists.

- [ ] **Step 3: Run focused tests**

Run:

```powershell
python -m pytest tests\test_phase20_bounded_tiled_training.py -q
```

Expected: all Phase 20 tests pass.

## Task 4: Writer, CLI, and Docs

- [ ] **Step 1: Add writer and CLI tests**

Test that `write_phase20_bounded_tiled_training_artifacts(...)` writes:

- `phase20_bounded_tiled_training_summary.csv`;
- `phase20_bounded_tiled_training_traces.json`.

Test that the CLI prints train tile, evaluation tile, variants, summary rows,
artifact paths, and claim boundary.

- [ ] **Step 2: Implement writer and CLI**

Create the runner with flags:

- `--phase2-output-dir`;
- `--tile-index-csv`;
- `--variants`;
- `--train-tile-id`;
- `--eval-tile-id`;
- `--total-timesteps`;
- `--eval-max-steps`;
- `--seed`;
- `--output-dir`.

- [ ] **Step 3: Update docs and manifest**

Document the Phase 20 command and pilot-only boundary in README, reproduction
guide, file manifest, and submission readiness audit.

## Task 5: Real Run and Verification

- [ ] **Step 1: Run real Phase 20 pilot**

Run:

```powershell
python experiments\phase20_bounded_tiled_training\run_phase20_bounded_tiled_training.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --total-timesteps 8 --eval-max-steps 4 --seed 0 --output-dir experiments\phase20_bounded_tiled_training\outputs\real_bishan_pilot
```

Expected: the runner writes both artifacts and reports completed B0/B1
same-tile trained-policy and baseline rows. It should also print the
`blocked_variable_observation_shape` cross-tile learned-policy status.
Numerical rewards are pilot diagnostics only.

- [ ] **Step 2: Run full verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests -q
git diff --check
```

Expected: smoke check passes, tests pass, and diff check reports no whitespace errors.

## Self-Review

- Spec coverage: plan covers Phase 20 design requirements, outputs, guardrails, docs, real run, and verification.
- Placeholder scan: no `TBD` or incomplete implementation steps remain.
- Type consistency: function names, artifact filenames, CLI flags, and claim-boundary fields are consistent across tasks.
