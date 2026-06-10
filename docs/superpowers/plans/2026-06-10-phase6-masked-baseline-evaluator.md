# Phase 6 Masked Baseline Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic non-learning masked baseline evaluator that runs `first_valid` and `seeded_random` action selectors across ready B0/B1/B2/B3 variants and writes comparable CSV/JSON artifacts.

**Architecture:** Add `paper11_geofm.baseline_eval` beside the Phase 5 rollout module, reusing `make_phase4_smoke_env()` for environment construction. Add a CLI under `experiments/phase6_masked_baselines/` and keep Phase 6 artifacts under ignored experiment `outputs/` paths.

**Tech Stack:** Python, NumPy, csv/json standard library, Gymnasium-compatible Phase 4 env, pytest.

---

## File Structure

- Create `src/paper11_geofm/baseline_eval.py`: Phase 6 claim boundary, baseline evaluator, action selectors, artifact writer.
- Create `experiments/phase6_masked_baselines/run_phase6_baselines.py`: CLI runner.
- Create `tests/test_phase6_baseline_eval.py`: evaluator, policy, artifact, validation, and CLI tests.
- Modify `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`, and `reproducibility/FILE_MANIFEST.tsv`.

## Task 1: Evaluator Contract

- [ ] Write `tests/test_phase6_baseline_eval.py` with fixture helpers mirroring Phase 5 and a failing `test_phase6_runs_default_policies_for_all_ready_variants`.
- [ ] Run `python -m pytest tests\test_phase6_baseline_eval.py::test_phase6_runs_default_policies_for_all_ready_variants -q`; expect `ModuleNotFoundError`.
- [ ] Create `src/paper11_geofm/baseline_eval.py` with `PHASE6_CLAIM_BOUNDARY`, `run_phase6_baseline_evaluator()`, and internal first-valid/random rollout support.
- [ ] Re-run the focused test; expect pass.

Expected behavior:

- default policies are `first_valid` and `seeded_random`;
- default variants are B0/B1/B2/B3;
- summary row count is `8`;
- first-valid B3 selects fixture blocks in order;
- B0/B1 total contract rewards are `0.0`;
- B2/B3 total contract rewards are positive.

## Task 2: Seed and Validation Behavior

- [ ] Add tests for same-seed determinism, different-seed variation, `max_steps`, and unknown policy errors.
- [ ] Run those focused tests; expect failures only where implementation is missing behavior.
- [ ] Patch `baseline_eval.py` minimally.
- [ ] Re-run focused tests; expect pass.

Expected behavior:

- same seed gives identical `seeded_random` selected-block sequence;
- different seed changes the `seeded_random` selected-block sequence for the fixture;
- `max_steps=2` limits each rollout to two steps;
- unknown policy raises `ValueError` containing `Unknown Phase 6 policy`.

## Task 3: Artifact Writer

- [ ] Add `test_phase6_baseline_artifacts_are_written`.
- [ ] Implement `write_phase6_baseline_artifacts()` writing `phase6_baseline_summary.csv` and `phase6_baseline_traces.json`.
- [ ] Run the artifact test; expect pass.

Expected CSV columns:

```text
policy_id,variant_id,seed,n_blocks,n_features,observation_shape,action_space_n,reward_mode,max_steps,episode_steps,terminated,truncated,valid_action_rate,total_contract_reward,selected_block_ids,claim_boundary
```

## Task 4: CLI Runner

- [ ] Add a failing import-based CLI test for `experiments/phase6_masked_baselines/run_phase6_baselines.py`.
- [ ] Run it; expect missing file failure.
- [ ] Implement CLI with `--phase2-output-dir`, `--output-dir`, `--variants`, `--policies`, `--max-steps`, and `--seed`.
- [ ] Re-run CLI test; expect pass.

Expected CLI output includes:

```text
Policy first_valid / Variant B0: steps=4 features=17 total_contract_reward=0.000000
Policy seeded_random / Variant B3:
Summary CSV:
Trace JSON:
Claim boundary: Phase 6 is a non-learning masked baseline evaluator
```

## Task 5: Documentation, Verification, Commit, Merge

- [ ] Update README with a Phase 6 command after Phase 5 using `experiments\phase6_masked_baselines\outputs\phase2_fixture` and `experiments\phase6_masked_baselines\outputs\phase6_baselines`.
- [ ] Update reproduction guide with expected Phase 6 policies, artifacts, seed determinism, and no-training boundary.
- [ ] Update file manifest rows for design, plan, module, CLI, and test.
- [ ] Run:

```powershell
python -m pytest tests\test_phase6_baseline_eval.py -q
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase6_masked_baselines\outputs\phase2_fixture
python experiments\phase6_masked_baselines\run_phase6_baselines.py --phase2-output-dir experiments\phase6_masked_baselines\outputs\phase2_fixture --output-dir experiments\phase6_masked_baselines\outputs\phase6_baselines --variants B0,B1,B2,B3 --policies first_valid,seeded_random --seed 0
python scripts\smoke_check.py
python -m pytest tests
git diff --check
```

- [ ] Commit implementation:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv src\paper11_geofm\baseline_eval.py experiments\phase6_masked_baselines\run_phase6_baselines.py tests\test_phase6_baseline_eval.py
git commit -m "Add Phase 6 masked baseline evaluator"
```

- [ ] Push feature branch, fast-forward merge to `main`, re-run main verification, push `main`, and delete the local feature branch.

---

## Self-Review

- Spec coverage: covers evaluator API, policies, seed behavior, artifacts, CLI, docs, verification, and merge.
- Red-flag scan: no unspecified implementation gaps remain.
- Type consistency: function names, filenames, policy IDs, artifact names, and CLI flags match the Phase 6 design spec.
