# Phase 7 MaskablePPO Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal `sb3-contrib` `MaskablePPO` compatibility smoke check for the Phase 4 Gymnasium input-contract environment.

**Architecture:** Add `paper11_geofm.maskableppo_smoke` to build the Phase 4 env, verify mask support, run a tiny CPU-only `MaskablePPO.learn()` call, run one masked `predict()`, and write a JSON smoke summary. Add a CLI under `experiments/phase7_maskableppo_smoke/`.

**Tech Stack:** Python, Gymnasium, NumPy, Stable-Baselines3, sb3-contrib MaskablePPO, pytest.

---

## File Structure

- Create `src/paper11_geofm/maskableppo_smoke.py`: Phase 7 claim boundary, dependency imports, smoke runner, artifact writer.
- Create `experiments/phase7_maskableppo_smoke/run_phase7_maskableppo_smoke.py`: CLI runner.
- Create `tests/test_phase7_maskableppo_smoke.py`: env mask support, smoke summary, artifact, and CLI tests.
- Modify `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`, and `reproducibility/FILE_MANIFEST.tsv`.

## Task 1: MaskablePPO Smoke Contract

- [ ] Write `tests/test_phase7_maskableppo_smoke.py` with Phase 2 fixture helpers and a failing `test_phase7_maskableppo_smoke_runs_tiny_learn_and_predict`.
- [ ] Run `python -m pytest tests\test_phase7_maskableppo_smoke.py::test_phase7_maskableppo_smoke_runs_tiny_learn_and_predict -q`; expect `ModuleNotFoundError`.
- [ ] Create `src/paper11_geofm/maskableppo_smoke.py` with `PHASE7_CLAIM_BOUNDARY` and `run_phase7_maskableppo_smoke()`.
- [ ] Re-run the focused test; expect pass.

Expected behavior:

- dependency checks use real `stable_baselines3` and `sb3_contrib` imports;
- B3 fixture env reports mask support;
- smoke uses CPU;
- `learn_timesteps` equals requested timesteps;
- predicted action is inside the initial action mask;
- summary includes claim boundary and dependency availability metadata.

## Task 2: Artifact Writer

- [ ] Add `test_phase7_maskableppo_artifact_is_written`.
- [ ] Implement `write_phase7_maskableppo_artifact()`.
- [ ] Run the artifact test; expect pass.

Expected artifact:

```text
phase7_maskableppo_smoke.json
```

## Task 3: CLI Runner

- [ ] Add an import-based CLI test for `experiments/phase7_maskableppo_smoke/run_phase7_maskableppo_smoke.py`.
- [ ] Run it; expect missing file failure.
- [ ] Implement CLI with `--phase2-output-dir`, `--output-dir`, `--variant`, `--total-timesteps`, and `--seed`.
- [ ] Re-run CLI test; expect pass.

Expected CLI output includes:

```text
Variant: B3
Observation shape: 331
Action space: Discrete(4)
Masking supported: True
Predicted action valid: True
Artifact:
Claim boundary: Phase 7 is a MaskablePPO compatibility smoke check
```

## Task 4: Documentation and Manifest

- [ ] Update README with a Phase 7 command after Phase 6 using `experiments\phase7_maskableppo_smoke\outputs\phase2_fixture` and `experiments\phase7_maskableppo_smoke\outputs\phase7_smoke`.
- [ ] Update reproduction guide with expected Phase 7 output, dependency note, and no-policy-performance boundary.
- [ ] Update file manifest rows for design, plan, module, CLI, and test.
- [ ] Run `python -m pytest tests\test_phase7_maskableppo_smoke.py -q`; expect pass or dependency skip only if SB3 dependencies are unavailable.

## Task 5: Verification, Commit, Merge

- [ ] Run:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase7_maskableppo_smoke\outputs\phase2_fixture
python experiments\phase7_maskableppo_smoke\run_phase7_maskableppo_smoke.py --phase2-output-dir experiments\phase7_maskableppo_smoke\outputs\phase2_fixture --output-dir experiments\phase7_maskableppo_smoke\outputs\phase7_smoke --variant B3 --total-timesteps 8 --seed 0
python scripts\smoke_check.py
python -m pytest tests
git diff --check
```

- [ ] Commit implementation:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv src\paper11_geofm\maskableppo_smoke.py experiments\phase7_maskableppo_smoke\run_phase7_maskableppo_smoke.py tests\test_phase7_maskableppo_smoke.py
git commit -m "Add Phase 7 MaskablePPO smoke check"
```

- [ ] Push feature branch, fast-forward merge to `main`, re-run main verification, push `main`, and delete the local feature branch.

---

## Self-Review

- Spec coverage: covers dependency handling, mask support, tiny learn, masked predict, artifact writing, CLI, docs, verification, and merge.
- Red-flag scan: no unspecified implementation gaps remain.
- Type consistency: function names, artifact filename, CLI flags, and claim boundary match the Phase 7 design spec.
