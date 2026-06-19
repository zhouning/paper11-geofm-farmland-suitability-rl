# Phase 29 Representation-Scale Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only Phase 29 diagnostic package for B1 representation scale, row norms, normalization profiles, tile-level scale variation, and PCA redundancy.

**Architecture:** Add one focused analysis module, one CLI runner, one pytest file, and documentation updates. The module will read existing CSV artifacts, compute deterministic NumPy summaries, write CSV/JSON/Markdown artifacts, and keep all scientific claims descriptive.

**Tech Stack:** Python standard library (`argparse`, `csv`, `json`, `pathlib`, `statistics`), NumPy, pytest, existing `paper11_geofm.block_schema` column definitions.

---

## File Structure

- Create `tests/test_phase29_representation_scale_diagnosis.py`: fixture feature tables, tile index, analysis assertions, writer assertions, and CLI test.
- Create `src/paper11_geofm/phase29_representation_scale_diagnosis.py`: CSV readers, matrix builders, scale summaries, normalization profiles, PCA diagnostics, artifact writer, Markdown renderer.
- Create `experiments/phase29_representation_scale_diagnosis/run_phase29_representation_scale_diagnosis.py`: command-line entry point.
- Modify `README.md`: add Phase 29 command and result link.
- Modify `reproducibility/REPRODUCTION_GUIDE.md`: add Phase 29 deterministic reproduction step and executable list entry.
- Modify `reproducibility/FILE_MANIFEST.tsv`: add Phase 29 docs, module, runner, result, and test rows.
- Modify `paper/phase28_results/README.md`: index the Phase 29 follow-up.
- Create `paper/phase28_results/03_phase29_representation_scale_diagnosis.md`: reviewer-facing interpretation of real Phase 29 results.

## Tasks

- [ ] Write failing tests for Phase 29 analysis, writer, and CLI.
- [ ] Run the new tests and confirm they fail because the module/runner do not exist.
- [ ] Implement the Phase 29 analysis module with deterministic summaries.
- [ ] Implement the Phase 29 CLI runner.
- [ ] Run the Phase 29 tests and Phase 28 compression regression tests.
- [ ] Run Phase 29 on real Bishan artifacts and inspect the JSON/Markdown output.
- [ ] Update README, reproduction guide, file manifest, and result package.
- [ ] Run smoke check, targeted pytest, and `git diff --check`.
- [ ] Commit and push the completed Phase 29 package.

## Test Expectations

The first test must import:

```python
from paper11_geofm.phase29_representation_scale_diagnosis import (
    PHASE29_REPRESENTATION_SCALE_CLAIM_BOUNDARY,
    build_phase29_representation_scale_diagnosis,
    write_phase29_representation_scale_diagnosis_artifacts,
)
```

It must assert:

- `analysis["phase"] == "phase29_representation_scale_diagnosis"`;
- the claim boundary is present;
- B1, D4P8, and D4P16 global rows are emitted;
- B1 raw and normalized row-norm profiles are emitted;
- tile-level summaries are emitted from the tile index;
- PCA diagnostics include top-8 and top-16 variance ratios;
- Markdown includes the causal boundary.

The CLI test must import the runner by file path, call `main([...])`, and check
that exit code `0` prints the status and JSON artifact path.

## Verification Commands

```powershell
python -m pytest tests\test_phase29_representation_scale_diagnosis.py -q --basetemp=.pytest_tmp_phase29 -p no:cacheprovider
python -m pytest tests\test_phase28_compression_diagnosis.py tests\test_phase28_representation_controls.py -q --basetemp=.pytest_tmp_phase29_regression -p no:cacheprovider
python scripts\smoke_check.py
git diff --check
git status --short --branch
```

## Self-Review

- Spec coverage: the plan covers Phase 29 inputs, deterministic diagnostics,
  outputs, CLI, real-data run, documentation, and verification.
- Placeholder scan: no TODO/TBD placeholders remain.
- Type consistency: artifact names and function names match the design.
- Scope check: the plan remains read-only and does not add PPO training or new
  scientific claims.
