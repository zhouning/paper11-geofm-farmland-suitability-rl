# Phase 30 Normalized-B1 Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bounded Phase 30 normalized-B1 ablation package that generates normalized B1 control tables, runs the held-out padded Bishan base-reward evaluation, and reports whether normalization improves raw B1 under the current protocol.

**Architecture:** Add one focused Phase 30 module that owns both normalized-control generation and the new evaluation/analysis path. Reuse the existing Phase 25 padded environment helpers and the Phase 28 control-evaluation structure, but keep Phase 30 logic isolated so Phase 28 remains a frozen historical benchmark.

**Tech Stack:** Python standard library (`argparse`, `csv`, `json`, `pathlib`), NumPy, pytest, and the existing Paper11 Phase 25/28 runtime helpers.

---

## File Structure

- Create `src/paper11_geofm/phase30_normalized_b1_ablation.py`: normalized-control builders, evaluation contract, analysis, artifact writers, and Markdown renderer.
- Create `experiments/phase30_normalized_b1_ablation/run_phase30_normalized_b1_ablation.py`: CLI entry point with `run-and-analyze` and `analyze-only`.
- Create `tests/test_phase30_normalized_b1_ablation.py`: fixture feature tables, normalized-control checks, analysis/status assertions, writer assertions, and CLI checks.
- Modify `README.md`: add Phase 30 runner and result link.
- Modify `paper/phase26_results/02_next_experiment_matrix.md`: record that Phase 30 is the concrete next representation-branch experiment.
- Modify `paper/phase28_results/README.md`: index the Phase 30 follow-up.
- Create `paper/phase28_results/04_phase30_normalized_b1_ablation.md`: reviewer-facing interpretation of real Phase 30 results.
- Modify `reproducibility/REPRODUCTION_GUIDE.md`: add Phase 30 reproduction commands and artifact list.
- Modify `reproducibility/FILE_MANIFEST.tsv`: add Phase 30 spec, plan, module, runner, result, and test rows.

## Tasks

- [ ] Write failing tests for normalized-control generation, analysis, writer, and CLI behavior.
- [ ] Run the new Phase 30 tests and confirm they fail because the module and runner do not exist.
- [ ] Implement normalized B1 control generation for `N1Z` and `N1ZR`.
- [ ] Implement Phase 30 held-out evaluation and summary analysis.
- [ ] Add the optional incremental path that merges an existing Phase 28
  control summary and trains only `N1Z` and `N1ZR`.
- [ ] Implement the Phase 30 CLI runner.
- [ ] Run the Phase 30 tests and the relevant Phase 25/28 regression tests.
- [ ] Run the real Bishan Phase 30 experiment and inspect the comparison JSON and Markdown.
- [ ] Update README, reproduction guide, file manifest, and the result package.
- [ ] Run smoke check, targeted pytest, and `git diff --check`.
- [ ] Commit and push the completed Phase 30 package.

## Test Expectations

The first test must import:

```python
from paper11_geofm.phase30_normalized_b1_ablation import (
    PHASE30_CLAIM_BOUNDARY,
    build_phase30_normalized_b1_controls,
    build_phase30_normalized_b1_analysis,
    write_phase30_normalized_b1_artifacts,
    write_phase30_normalized_b1_controls,
)
```

It must assert:

- `N1Z` and `N1ZR` manifests are emitted;
- both normalized variants preserve `block_id` order and the original explicit
  feature columns;
- `N1Z` columns are centered/scaled as true z-scores;
- `N1ZR` rows have unit row L2 norm after z-score normalization;
- Phase 30 analysis emits focal deltas for normalized variants against `B1`;
- Markdown includes the claim boundary and does not claim submission-level
  success.

The CLI test must import the runner by file path, call `main([...])`, and
check that exit code `0` prints the Phase 30 status and comparison JSON path.

An additional run-path test should verify that, when
`existing_control_summary_csv` is supplied, the Phase 30 runner trains only the
normalized variants and merges the existing control rows into the final
analysis.

## Verification Commands

```powershell
python -m pytest tests\test_phase30_normalized_b1_ablation.py -q --basetemp=.pytest_tmp_phase30 -p no:cacheprovider
python -m pytest tests\test_phase25_padded_heldout_policy.py tests\test_phase28_representation_controls.py tests\test_phase29_representation_scale_diagnosis.py -q --basetemp=.pytest_tmp_phase30_regression -p no:cacheprovider
python scripts\smoke_check.py
git diff --check
git status --short --branch
```

## Self-Review

- Spec coverage: the plan covers derived normalized inputs, Phase 30 training/evaluation, analysis, CLI, docs, and verification.
- Placeholder scan: no TODO or TBD placeholders remain.
- Type consistency: the normalized variant IDs, artifact names, and public
  function names match the design.
- Scope check: the plan stays within the representation branch and does not add
  reward integration, transfer expansion, or B2/B3 work.
