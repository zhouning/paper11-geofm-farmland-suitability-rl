# Phase 4 DRL Smoke Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal Gymnasium-compatible smoke environment that consumes Phase 3 variant inputs and verifies observation/action/mask/reward wiring without training or evaluating a policy.

**Architecture:** Add a focused `paper11_geofm.drl_smoke_env` module that wraps `VariantInput` into a deterministic Gymnasium contract-smoke environment. Add a CLI under `experiments/phase4_drl_smoke_env/` that runs one reset/step cycle and prints the contract summary.

**Tech Stack:** Python, NumPy, Gymnasium, pytest, existing Phase 3 `load_variant_input()`.

---

## File Structure

- Create `src/paper11_geofm/drl_smoke_env.py`: `Phase4InputContractEnv`, `make_phase4_smoke_env()`, and claim boundary constant.
- Create `experiments/phase4_drl_smoke_env/run_phase4_smoke.py`: one-step smoke runner.
- Create `tests/test_phase4_drl_smoke_env.py`: env and CLI tests.
- Modify `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`, and `reproducibility/FILE_MANIFEST.tsv`.

## Task 1: Environment Contract

- [ ] Write failing test `test_phase4_env_wraps_b3_variant_as_gym_contract`.
- [ ] Run `python -m pytest tests\test_phase4_drl_smoke_env.py::test_phase4_env_wraps_b3_variant_as_gym_contract -q`; expect module missing.
- [ ] Implement `src/paper11_geofm/drl_smoke_env.py` with `Phase4InputContractEnv`.
- [ ] Re-run the focused test; expect pass.

Expected behavior:

- B3 fixture with 2 blocks and 82 features has observation shape `(2 * 82 + 3,)`.
- `action_space.n == 2`.
- `action_masks()` starts as `[True, True]`.
- after `step(0)`, mask is `[False, True]`, reward is selected row's `suitability_proxy`, and observation remains `float32`.

## Task 2: Validation and Reward Modes

- [ ] Add tests for B1 zero contract reward and out-of-range action failure.
- [ ] Run `python -m pytest tests\test_phase4_drl_smoke_env.py -q`; expect failures only if implementation lacks validation.
- [ ] Patch env validation as needed.
- [ ] Re-run focused tests; expect pass.

Expected behavior:

- B1 uses `base_planning_reward` and returns `0.0` contract reward.
- invalid actions raise `ValueError` with `out of range`.
- selecting an already-selected action raises `ValueError` with `already selected`.

## Task 3: CLI Smoke Runner

- [ ] Add failing CLI test `test_run_phase4_smoke_cli_prints_one_step_summary`.
- [ ] Run the CLI test; expect missing file failure.
- [ ] Create `experiments/phase4_drl_smoke_env/run_phase4_smoke.py`.
- [ ] Re-run CLI test; expect pass.

Expected CLI output includes:

- `Variant: B3`
- `Observation shape: 331`
- `Action space: Discrete(4)` for the included 4-row fixture path
- `Initial valid actions: 4`
- `Selected block: sample_block_00`
- `Step reward:`
- `Claim boundary: Phase 4 is a DRL input-contract smoke environment`

## Task 4: Documentation

- [ ] Update README with Phase 4 one-step smoke command after Phase 3.
- [ ] Update reproduction guide with expected Phase 4 output and explicit no-training boundary.
- [ ] Update file manifest with spec, plan, env module, CLI, and test rows.
- [ ] Run `python -m pytest tests\test_phase4_drl_smoke_env.py -q`; expect pass.

## Task 5: Verification and Commit

- [ ] Run Phase 2 fixture output:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir .pytest_tmp\phase4_drl_smoke_fixture
```

- [ ] Run Phase 4 smoke CLI:

```powershell
python experiments\phase4_drl_smoke_env\run_phase4_smoke.py --phase2-output-dir .pytest_tmp\phase4_drl_smoke_fixture --variant B3
```

- [ ] Run repository verification:

```powershell
python scripts\smoke_check.py
python -m pytest tests
git diff --check
```

- [ ] Stage and commit:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv src\paper11_geofm\drl_smoke_env.py experiments\phase4_drl_smoke_env\run_phase4_smoke.py tests\test_phase4_drl_smoke_env.py docs\superpowers\plans\2026-06-09-phase4-drl-smoke-env.md
git commit -m "Add Phase 4 DRL smoke environment"
```

## Task 6: Merge and Push

- [ ] Push feature branch.
- [ ] Fast-forward merge to `main`.
- [ ] Re-run smoke check, full pytest, and diff check on `main`.
- [ ] Push `main`.
- [ ] Delete the local feature branch after merge.

---

## Self-Review

- Spec coverage: plan covers environment, validation, CLI, documentation, verification, and merge.
- Placeholder scan: no `TBD` or incomplete steps.
- Type consistency: module names, function names, and CLI paths match the design spec.
