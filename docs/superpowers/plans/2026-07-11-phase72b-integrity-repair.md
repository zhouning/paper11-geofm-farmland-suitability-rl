# Phase 72B Integrity Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Phase 72B provenance, prepared-artifact, spatial-coverage, CLI-exit, and confirmation-receipt gaps before the measured negative result is committed.

**Architecture:** Add deterministic Phase 72A source-to-derived validation during `prepare`, then write a hashed manifest covering every prepared artifact. Load that manifest through one shared verifier in both `fit-freeze` and `confirm`, bind its hash to fit progress and the selected-model manifest, require complete spatial coverage, and make confirmation output exclusive with a hashed receipt. Archive the pre-repair ignored outputs, rebuild from audited sources, refit from a clean frozen directory, and run one official receipt-bound confirmation without changing the tracked scientific protocol or thresholds.

**Tech Stack:** Python 3.11+, NumPy, pandas, scikit-learn, joblib, pytest, hashed JSON/CSV/NPZ artifacts, PowerShell.

---

### Task 1: Phase 72A Derived-Input Provenance

**Files:**
- Modify: `tests/test_phase72b_geofm_information_gain_screen.py`
- Modify: `src/paper11_geofm/phase72b_information_gain_screen.py`

- [ ] **Step 1: Write failing CSV-tamper test**

Create a test that builds `_phase72b_prepare_fixture`, flips one `y_1y` value in `phase72a_temporal_sample_index.csv`, calls `prepare_phase72b_information_gain_screen`, and expects `ValueError` containing `Phase 72A derived sample mismatch`.

- [ ] **Step 2: Verify RED**

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase72b_geofm_information_gain_screen.py::test_phase72b_prepare_rejects_tampered_phase72a_sample_csv -q --basetemp=D:\tmp\paper11_phase72b_integrity_task1_csv_red -p no:cacheprovider
```

Expected: FAIL because the current prepare path accepts the changed CSV.

- [ ] **Step 3: Write failing tensor-tamper test**

Create a second test that changes one value in `phase72a_temporal_samples.npz` and expects `ValueError` containing `Phase 72A derived tensor mismatch`.

- [ ] **Step 4: Verify RED**

Run the tensor test alone and confirm it fails because prepare accepts the changed tensor.

- [ ] **Step 5: Implement deterministic provenance verification**

In `phase72b_information_gain_screen.py`, rebuild Phase 72A with `build_phase72a_temporal_label_package` using the supplied region config, embedding directories, and label directories. Require:

```python
rebuilt["phase72a_status"] == "phase72a_label_inputs_ready"
set(region.region_id for region in contract.regions) == {"bishan", "dongxing"}
phase72a_package["row_counts"]["sample_rows"] == len(sample_rows)
rebuilt["manifest_rows"] == phase72a_package["manifest_rows"]
```

Normalize CSV scalar types through the declared Phase 72A sample fields, compare every rebuilt sample row with the loaded CSV row, and compare every rebuilt tensor with the loaded NPZ array using exact shape, dtype, and value equality. Raise the test-specific mismatch messages before any Phase 72B features or targets are built.

- [ ] **Step 6: Verify GREEN**

Run both new tests and the existing prepare test. Expected: all pass.

- [ ] **Step 7: Commit**

```powershell
git add src\paper11_geofm\phase72b_information_gain_screen.py tests\test_phase72b_geofm_information_gain_screen.py docs\superpowers\plans\2026-07-11-phase72b-integrity-repair.md
git commit -m "fix: verify Phase 72B source provenance"
```

---

### Task 2: Hashed Prepared-Artifact Manifest

**Files:**
- Create: `src/paper11_geofm/phase72b_prepared.py`
- Modify: `src/paper11_geofm/phase72b_information_gain_screen.py`
- Modify: `src/paper11_geofm/phase72b_models.py`
- Modify: `tests/test_phase72b_geofm_information_gain_screen.py`

- [ ] **Step 1: Write failing matrix, split, row, and leakage mutation tests**

Using a written fixture package, mutate one artifact at a time:

```text
phase72b_feature_matrices.npz
phase72b_split_registry.json
phase72b_feature_rows.csv
phase72b_leakage_audit.json
```

Assert both `fit_freeze_phase72b_models` and `confirm_phase72b_information_gain_screen` reject the mutation with `prepared artifact hash mismatch` before fitting or label evaluation.

- [ ] **Step 2: Verify RED**

Run the four focused mutation tests. Expected: matrix mutation is accepted by fit-freeze and row/audit mutations are accepted by confirmation.

- [ ] **Step 3: Implement prepared-manifest writer**

After `write_phase72b_prepared_artifacts` writes its current files, compute byte-level SHA256 values for this fixed set:

```python
PREPARED_ARTIFACT_NAMES = (
    "phase72b_terrain_manifest.csv",
    "phase72b_feature_manifest.csv",
    "phase72b_feature_registry.json",
    "phase72b_feature_rows.csv",
    "phase72b_feature_matrices.npz",
    "phase72b_development_targets.npz",
    "phase72b_confirmation_targets.npz",
    "phase72b_split_registry.json",
    "phase72b_row_alignment_audit.csv",
    "phase72b_leakage_audit.json",
    "phase72b_frozen_protocol.json",
    "phase72b_frozen_protocol.sha256",
)
```

Write `phase72b_prepared_artifacts.json` plus `phase72b_prepared_artifacts.sha256` with status `phase72b_prepared_artifacts_frozen`, the frozen protocol hash, and `{name, sha256}` records.

- [ ] **Step 4: Implement one shared verifier**

Create `phase72b_prepared.py` with `load_verified_phase72b_prepared(prepared_dir)`. It must load the hashed manifest, require the exact artifact-name set, verify every byte hash, load the hashed frozen protocol, verify the manifest/protocol relationship, verify matrix/split/registry semantic hashes already recorded by the protocol, require contiguous and in-range feature rows, require every matrix row count to equal feature-row count, and recompute `audit_phase72b_splits` from feature rows and the split registry. The stored leakage audit must equal the recomputed audit.

- [ ] **Step 5: Route fit and confirm through the verifier**

Replace the separate prepared reads in `phase72b_models.py` and confirmation with the shared loader. Keep development and confirmation targets unopened until their existing phase-specific boundary.

- [ ] **Step 6: Verify GREEN**

Run the four mutation tests plus the existing fit/confirm tests. Expected: all pass.

- [ ] **Step 7: Commit**

```powershell
git add src\paper11_geofm\phase72b_prepared.py src\paper11_geofm\phase72b_information_gain_screen.py src\paper11_geofm\phase72b_models.py tests\test_phase72b_geofm_information_gain_screen.py
git commit -m "fix: freeze Phase 72B prepared artifacts"
```

---

### Task 3: Bind Fits to the Prepared Manifest

**Files:**
- Modify: `src/paper11_geofm/phase72b_models.py`
- Modify: `src/paper11_geofm/phase72b_information_gain_screen.py`
- Modify: `tests/test_phase72b_geofm_information_gain_screen.py`

- [ ] **Step 1: Write failing binding tests**

Assert a new fit-progress file and selected-model manifest contain `prepared_artifacts_sha256`. Change only the prepared-manifest hash pair after fitting and assert both resume and confirmation reject the mismatch.

- [ ] **Step 2: Verify RED**

Run the binding tests. Expected: the field is absent and changed prepared manifests are not bound to the selected models.

- [ ] **Step 3: Implement progress and selected-manifest binding**

Pass the verified prepared-manifest digest into `_load_fit_progress`. New progress must store it. Existing progress with a different or missing digest must be rejected. Add the digest to `phase72b_selected_models.json`; confirmation must require equality among the prepared manifest, fit selection, and protocol relationship before loading bundles or confirmation labels.

- [ ] **Step 4: Verify GREEN**

Run focused binding tests and the complete Phase 72B test file. Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add src\paper11_geofm\phase72b_models.py src\paper11_geofm\phase72b_information_gain_screen.py tests\test_phase72b_geofm_information_gain_screen.py
git commit -m "fix: bind Phase 72B fits to prepared inputs"
```

---

### Task 4: Complete Spatial Coverage, CLI Failure, and Exclusive Receipt

**Files:**
- Modify: `src/paper11_geofm/phase72b_information_gain_screen.py`
- Modify: `experiments/phase72b_geofm_information_gain_screen/run_phase72b_information_gain_screen.py`
- Modify: `tests/test_phase72b_geofm_information_gain_screen.py`

- [ ] **Step 1: Write failing incomplete-spatial test**

Build a valid fixture, remove one expected spatial primary group or make its confirmation outcomes single-class, and assert the result is `phase72b_inputs_not_ready` with a spatial-coverage blocker.

- [ ] **Step 2: Verify RED**

Expected: current confirmation silently appends the axis to `invalid_spatial_axes` and evaluates the remaining folds.

- [ ] **Step 3: Implement complete coverage requirement**

After group construction, require the set of spatial axes with valid explicit/primary groups and two classes to equal `expected_spatial_axes`. Missing or invalid axes become blockers and force `phase72b_inputs_not_ready`; no partial positive or mixed gate is allowed.

- [ ] **Step 4: Write failing CLI status test**

Patch or construct a confirmation fixture returning `phase72b_inputs_not_ready` without throwing. Run the CLI and assert return code `1` and the status in stdout.

- [ ] **Step 5: Implement CLI status exit**

After artifacts are written, return `1` when `result["phase72b_status"] == "phase72b_inputs_not_ready"`; scientifically valid negative, mixed, and supported outcomes remain exit `0`.

- [ ] **Step 6: Write failing exclusive-output test**

Write confirmation artifacts once, assert a receipt exists, then call the writer again on the same directory and expect `FileExistsError` without changing any artifact.

- [ ] **Step 7: Implement exclusive directory and receipt**

Require the output directory not to exist before writing. After all eight stable artifacts are written, hash them and write hashed `phase72b_confirmation_receipt.json` containing the protocol hash, selected-model hash, prepared-artifact hash, final status, and artifact hashes. Return receipt paths from the writer and update the stable-output test.

- [ ] **Step 8: Verify GREEN and commit**

Run the focused tests and the complete Phase 72B test file, then commit:

```powershell
git add src\paper11_geofm\phase72b_information_gain_screen.py experiments\phase72b_geofm_information_gain_screen\run_phase72b_information_gain_screen.py tests\test_phase72b_geofm_information_gain_screen.py
git commit -m "fix: enforce Phase 72B confirmation integrity"
```

---

### Task 5: Regenerate Official Phase 72B Evidence

**Files:**
- Archive ignored directories only; do not delete them.
- Regenerate ignored outputs below `experiments/phase72b_geofm_information_gain_screen/outputs/`.

- [ ] **Step 1: Verify pre-real tests**

Run the full Phase 72B test file and smoke check. Stop on any failure.

- [ ] **Step 2: Archive pre-repair outputs**

Resolve and verify the source and destination paths remain inside the Phase 72B output directory. Rename `prepared`, `frozen`, and `confirmation` to timestamped `*_pre_integrity_repair` directories. Preserve `terrain` because its hashes were independently verified.

- [ ] **Step 3: Re-run prepare from audited sources**

Use the tracked Task 9 prepare command, pointing the Phase 72A package to the parent checkout's ignored real package if necessary. Verify the protocol hash remains the tracked scientific hash and inspect the new prepared-manifest hash.

- [ ] **Step 4: Refit from a clean frozen directory**

Run `--mode fit-freeze` with the regenerated prepared package. Do not copy legacy progress or bundles. Record the new selected-model hash and prepared-manifest binding.

- [ ] **Step 5: Run one official confirmation**

Run `--mode confirm` into a nonexistent `outputs/confirmation` directory. Verify the receipt, every artifact hash, complete 10-axis spatial coverage, zero blockers, and the final status. Compare every metric with the archived pre-repair result and explain any difference.

---

### Task 6: Update Evidence, Verify, and Complete Branch

**Files:**
- Modify: `paper/phase28_results/39_phase72b_geofm_information_gain_screen.md`
- Modify: `paper/phase28_results/README.md` only if status wording changes.
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [ ] **Step 1: Update measured documentation**

Record the prepared-manifest hash, new selected-model hash, confirmation receipt hash, regeneration reason, exact official metrics, branch ahead/behind state, HEAD, and `origin/main`. Preserve the formal-manuscript exclusion and negative claim boundary unless the clean rerun changes the gate.

- [ ] **Step 2: Run adjacent and full verification**

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase72b_geofm_information_gain_screen.py tests\test_phase72a_temporal_label_package.py tests\test_phase68_external_independent_label_package.py tests\test_phase40_independent_label_gate.py tests\test_phase39_independent_label_audit.py -q --basetemp=D:\tmp\paper11_phase72b_integrity_adjacent -p no:cacheprovider
D:\adk\.venv\Scripts\python.exe -m pytest -q --basetemp=D:\tmp\paper11_phase72b_integrity_full -p no:cacheprovider
D:\adk\.venv\Scripts\python.exe scripts\smoke_check.py
git diff --check
git diff --name-only HEAD -- paper\submission\final
```

Expected: all tests and smoke pass, diff check is clean, and the formal-manuscript command is empty.

- [ ] **Step 3: Request independent review**

Have a reviewer cross-check every documented measurement, all prepared/selected/receipt hashes, mutation tests, spatial coverage, CLI exit behavior, and formal-manuscript zero-diff.

- [ ] **Step 4: Commit measured repair result**

```powershell
git add paper\phase28_results\39_phase72b_geofm_information_gain_screen.md paper\phase28_results\README.md docs\superpowers\phase33_current_progress_handoff.md
git commit -m "docs: record integrity-verified Phase 72B result"
```

## Plan Self-Review

- The plan covers every reviewer finding: Phase 72A derived provenance, all prepared artifacts, fit binding, incomplete spatial coverage, CLI exit semantics, exclusive confirmation receipt, Task 10 branch state, clean regeneration, and final review.
- The tracked Phase 72B scientific protocol and thresholds remain unchanged; integrity metadata is added outside the protocol payload.
- The regeneration uses a clean frozen directory, so no legacy bundle can bypass the new prepared-manifest binding.
- Archived ignored outputs remain available for before/after comparison.
- No step changes `paper/submission/final/*` or permits Phase 72C after a negative gate.
