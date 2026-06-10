# Phase 8 Ablation-Control Feature Tables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic D2/D3/D4P8/D4P16 ablation-control feature tables from ready Phase 2 outputs without training or evaluating a DRL policy.

**Architecture:** Add `paper11_geofm.ablation_controls` to load existing Phase 2 B0/B1 variant inputs, derive random, shuffled, and PCA-compressed controls, and write Phase 3-compatible artifacts. Add a CLI under `experiments/phase8_ablation_controls/` and document a reviewer command path under ignored `outputs/`.

**Tech Stack:** Python, NumPy, CSV/JSON artifacts, existing Phase 3 `load_variant_input()`, pytest.

---

## File Structure

- Create `src/paper11_geofm/ablation_controls.py`: Phase 8 claim boundary, table builder, deterministic control generators, manifest builder, artifact writer.
- Create `experiments/phase8_ablation_controls/run_phase8_ablation_controls.py`: CLI runner.
- Create `tests/test_phase8_ablation_controls.py`: build contract, determinism, artifact, loader compatibility, and CLI tests.
- Modify `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`, and `reproducibility/FILE_MANIFEST.tsv`.

## Task 1: Build Contract

- [ ] Write `tests/test_phase8_ablation_controls.py` with Phase 2 fixture helpers and a failing `test_phase8_builds_expected_ablation_control_tables`.
- [ ] Run `python -m pytest tests\test_phase8_ablation_controls.py::test_phase8_builds_expected_ablation_control_tables -q`; expect `ModuleNotFoundError`.
- [ ] Create `src/paper11_geofm/ablation_controls.py` with `PHASE8_CLAIM_BOUNDARY` and `build_phase8_ablation_controls()`.
- [ ] Re-run the focused test; expect pass.

Expected first test assertions:

```python
assert protocol["phase"] == "phase8_ablation_control_features"
assert protocol["variant_ids"] == ["D2", "D3", "D4P8", "D4P16"]
assert protocol["seed"] == 0
assert protocol["source_variants"]["B0"]["n_features"] == 17
assert protocol["source_variants"]["B1"]["n_features"] == 81
assert protocol["summary"]["D2"]["n_features"] == 81
assert protocol["summary"]["D3"]["n_features"] == 81
assert protocol["summary"]["D4P8"]["n_features"] == 25
assert protocol["summary"]["D4P16"]["n_features"] == 33
assert protocol["manifest"]["variants"]["D2"]["ready"] is True
assert protocol["manifest"]["variants"]["D4P16"]["row_count"] == 4
assert protocol["claim_boundary"] == PHASE8_CLAIM_BOUNDARY
```

Implementation requirements:

- Load B0 and B1 through `load_variant_input(phase2_output_dir, "B0")` and `load_variant_input(phase2_output_dir, "B1")`.
- Reject mismatched block IDs with `ValueError("Phase 8 requires aligned B0 and B1 block IDs")`.
- Build table rows as `list[dict[str, object]]` keyed by `block_id`.
- Preserve canonical `embedding_mean_00` through `embedding_mean_63` column names for D2 and D3.
- Use `embedding_pca_00` through `embedding_pca_07` for D4P8 and `embedding_pca_00` through `embedding_pca_15` for D4P16.
- Use `base_planning_reward` for all Phase 8 controls.

## Task 2: Deterministic Control Semantics

- [ ] Add `test_phase8_random_control_is_reproducible_and_seed_sensitive`.
- [ ] Add `test_phase8_shuffled_control_uses_non_identity_permutation`.
- [ ] Add `test_phase8_pca_controls_emit_requested_dimensions_with_padding`.
- [ ] Patch `ablation_controls.py` until these tests pass.
- [ ] Run `python -m pytest tests\test_phase8_ablation_controls.py -q`; expect pass for Task 1 and Task 2 tests.

Expected deterministic checks:

- `build_phase8_ablation_controls(tmp_path, seed=0)` equals a repeated seed-0 call for D2 embedding values.
- seed `1` changes at least one D2 embedding value.
- D3 summary includes a `shuffle_permutation` list of length `4`.
- For the 4-block fixture, D3 permutation is not `[0, 1, 2, 3]`.
- D4P8 required columns include exactly 8 PCA columns.
- D4P16 required columns include exactly 16 PCA columns.
- D4P16 zero-pads trailing components when requested component count exceeds available centered-matrix rank.

Implementation requirements:

- Use `np.random.default_rng(seed)` for D2 and D3.
- For D2, generate standard-normal random values and rescale by source B1 embedding column means and safe standard deviations.
- For D3, use a deterministic permutation and rotate by one position if the permutation is identity and row count is greater than one.
- For D4, use deterministic NumPy SVD on centered source B1 embedding columns; zero-pad to the requested dimension.

## Task 3: Artifact Writer and Loader Compatibility

- [ ] Add `test_phase8_ablation_artifacts_are_written_and_loadable`.
- [ ] Implement `write_phase8_ablation_artifacts()`.
- [ ] Re-run the artifact test; expect pass.

Expected artifact assertions:

```python
paths = write_phase8_ablation_artifacts(protocol, output_dir)
assert paths["manifest"].name == "experiment_variants.json"
assert paths["summary"].name == "phase8_ablation_control_summary.json"
assert paths["variant_tables"]["D2"].name == "variant_D2_features.csv"
assert paths["variant_tables"]["D4P16"].name == "variant_D4P16_features.csv"
loaded_d2 = load_variant_input(output_dir, "D2")
loaded_d4p16 = load_variant_input(output_dir, "D4P16")
assert loaded_d2.state_matrix.shape == (4, 81)
assert loaded_d4p16.state_matrix.shape == (4, 33)
```

Implementation requirements:

- Create output directories with `mkdir(parents=True, exist_ok=True)`.
- Write CSV columns in the exact order `block_id` plus each variant's `required_columns`.
- Write `experiment_variants.json` with `indent=2` and `sort_keys=True`.
- Write `phase8_ablation_control_summary.json` without embedding row payloads, so the summary stays small and reviewable.
- Return `{"manifest": Path, "summary": Path, "variant_tables": dict[str, Path]}`.

## Task 4: CLI Runner

- [ ] Add an import-based CLI test for `experiments/phase8_ablation_controls/run_phase8_ablation_controls.py`.
- [ ] Run it; expect missing file failure.
- [ ] Implement CLI with `--phase2-output-dir`, `--output-dir`, `--seed`, and `--pca-dimensions`.
- [ ] Re-run the CLI test; expect pass.

Expected CLI output includes:

```text
Generated variants: D2,D3,D4P8,D4P16
D2 features: 81
D3 features: 81
D4P8 features: 25
D4P16 features: 33
Manifest:
Summary:
Claim boundary: Phase 8 builds diagnostic ablation-control feature tables
```

Implementation requirements:

- Parse `--pca-dimensions 8,16` into `tuple[int, ...]`.
- Catch `FileNotFoundError` and `ValueError`, print `Error: ...` to stderr, and return exit code `1`.
- Do not import or run Stable-Baselines3, sb3-contrib, or MaskablePPO.

## Task 5: Documentation and Manifest

- [ ] Update README with a Phase 8 command after Phase 7 using `experiments\phase8_ablation_controls\outputs\phase2_fixture` and `experiments\phase8_ablation_controls\outputs\phase8_controls`.
- [ ] Update reproduction guide with expected Phase 8 generated variants, artifact files, deterministic seed behavior, and no-training/no-policy-performance boundary.
- [ ] Update file manifest rows for design, plan, module, CLI, and test.
- [ ] Run `python -m pytest tests\test_phase8_ablation_controls.py -q`; expect pass.

README command block:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase8_ablation_controls\outputs\phase2_fixture
python experiments\phase8_ablation_controls\run_phase8_ablation_controls.py --phase2-output-dir experiments\phase8_ablation_controls\outputs\phase2_fixture --output-dir experiments\phase8_ablation_controls\outputs\phase8_controls --seed 0 --pca-dimensions 8,16
```

## Task 6: Verification, Commit, Merge

- [ ] Run:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase8_ablation_controls\outputs\phase2_fixture
python experiments\phase8_ablation_controls\run_phase8_ablation_controls.py --phase2-output-dir experiments\phase8_ablation_controls\outputs\phase2_fixture --output-dir experiments\phase8_ablation_controls\outputs\phase8_controls --seed 0 --pca-dimensions 8,16
python scripts\smoke_check.py
python -m pytest tests
git diff --check
```

- [ ] Commit implementation:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv src\paper11_geofm\ablation_controls.py experiments\phase8_ablation_controls\run_phase8_ablation_controls.py tests\test_phase8_ablation_controls.py docs\superpowers\plans\2026-06-10-phase8-ablation-control-features.md
git commit -m "Add Phase 8 ablation control feature tables"
```

- [ ] Push feature branch, fast-forward merge to `main`, re-run main verification, push `main`, and delete the local feature branch.

---

## Self-Review

- Spec coverage: covers D2, D3, D4P8, D4P16 generation, determinism, Phase 3-compatible manifest writing, CLI, docs, verification, and merge.
- Red-flag scan: no training, reward reporting, planning metrics, transfer metrics, policy comparisons, or AlphaEarth direct-measurement claims are included.
- Type consistency: function names, artifact filenames, variant IDs, CLI flags, and claim boundary match the Phase 8 design spec.
