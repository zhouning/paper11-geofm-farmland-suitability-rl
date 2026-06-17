# Phase 26 Current Progress Handoff

Last updated: 2026-06-17.

## Repository State

- Repository: `D:\test\paper11-geofm-farmland-suitability-rl`
- Branch: `main`
- Remote: `origin/main`
- Current pushed head: `3e0a147 fix: validate Phase 26 tile-seed coverage`
- Status at handoff creation: `main...origin/main`
- Feature worktree `phase26-main-empirical-experiment` was merged and removed.

## Completed Work

Phase 26 was implemented, reviewed, merged to `main`, and pushed.

Key commits:

- `9cc043a test: add Phase 26 main empirical analysis tests`
- `c9c9e9c feat: add Phase 26 main empirical analysis module`
- `04c31d1 feat: complete Phase 26 empirical artifact writer`
- `4cd1552 feat: add Phase 26 main empirical CLI`
- `5773af4 test: harden Phase 26 CLI validation`
- `2b3fe91 test: cover Phase 26 run-and-analyze explicit flags`
- `6428693 docs: add Phase 26 main empirical analysis guidance`
- `3e0a147 fix: validate Phase 26 tile-seed coverage`

New Phase 26 files:

- `src/paper11_geofm/phase26_main_experiment.py`
- `experiments/phase26_main_experiment/run_phase26_main_experiment.py`
- `tests/test_phase26_main_experiment.py`

Updated documentation:

- `README.md`
- `reproducibility/REPRODUCTION_GUIDE.md`
- `reproducibility/FILE_MANIFEST.tsv`
- `paper/submission/01_ijaeog_submission_readiness.md`
- `paper/submission/02_draft_titles_highlights_declarations.md`

## What Phase 26 Solves

Phase 26 turns Phase 25 padded held-out B0/B1 outputs into the first
manuscript-facing main empirical analysis package for Paper11.

It reads Phase 25 artifacts:

- `phase25_padded_heldout_policy_summary.csv`
- `phase25_padded_heldout_policy_comparison.json`

It writes:

- `phase26_main_summary.csv`
- `phase26_tile_seed_delta_table.csv`
- `phase26_main_comparison.json`
- `phase26_claim_readiness.md`

It reports B1-B0 learned-policy deltas by held-out Bishan tile and random seed,
computes conservative claim status, and records remaining evidence gaps.

## Final Review Fixes

Final review identified two important correctness issues, both fixed:

1. Coverage validation now detects missing expected tile-seed pairs, unexpected
   tile-seed pairs, and duplicate trained-policy variant rows.
2. Any coverage issue forces `phase26_claim_status = insufficient` instead of
   allowing a misleading positive claim.

Additional cleanup:

- Phase 25 runner/writer imports are lazy in the Phase 26 CLI, so analyze-only
  import does not load training-side dependencies.
- Submission highlight wording is guarded: Phase 26 is ready to report main
  deltas once real main-run artifacts exist; it does not claim those real
  results are already available.

## Verification

Verified after merge on `main`:

```powershell
python -m pytest tests\test_phase26_main_experiment.py -q --basetemp=.pytest_tmp_main_phase26
```

Result:

```text
13 passed
```

```powershell
python scripts\smoke_check.py
```

Result:

```text
Paper11 smoke check passed.
```

```powershell
python -m pytest tests -q --basetemp=.pytest_tmp_main_all
```

Result:

```text
161 passed
```

```powershell
git diff --check
```

Result: clean.

## Timing Probe Status

The Phase 26 Windows timing probe was not run in the isolated implementation
worktree because real generated Phase 11/13 outputs were absent there:

- `experiments\phase11_bishan_dltb_real\outputs\phase2_real`
- `experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv`

This is documented in `reproducibility/REPRODUCTION_GUIDE.md`.

## Important Claim Boundaries

Phase 26 remains restricted to:

- B0/B1 only;
- deterministic `base_planning_reward`;
- held-out Bishan tiles from Phase 25 outputs;
- analysis of Phase 25 outputs, not a new reward or representation family.

Do not claim:

- suitability reward benefit;
- B2/B3 superiority;
- cross-region transfer;
- final submission-level planning performance;
- positive multi-tile empirical results until real Phase 26 main-run artifacts
  support that statement.

## Next Experimental Step

Run the real main empirical training on Colab Pro+ or another stronger training
platform, then analyze the outputs with Phase 26.

Recommended Colab Pro+ Phase 25 main run:

```bash
python experiments/phase25_padded_heldout_policy/run_phase25_padded_heldout_policy.py --phase2-output-dir experiments/phase11_bishan_dltb_real/outputs/phase2_real --tile-index-csv experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv --variants B0,B1 --total-timesteps 1024 --eval-max-steps 8 --seeds 0,1,2 --max-eval-tiles 3 --output-dir experiments/phase26_main_experiment/outputs/colab_main/phase25_run
```

If runtime allows, repeat with:

```bash
--total-timesteps 4096
```

Then run Phase 26 analysis-only:

```powershell
python experiments\phase26_main_experiment\run_phase26_main_experiment.py --mode analyze-only --phase25-output-dir experiments\phase26_main_experiment\outputs\colab_main\phase25_run --output-dir experiments\phase26_main_experiment\outputs\colab_main\phase26_analysis
```

## Suggested Initial Commands Next Session

```powershell
cd D:\test\paper11-geofm-farmland-suitability-rl
git status --short --branch --untracked-files=no
git log --oneline --max-count=8
Get-Content docs\superpowers\phase26_current_progress_handoff.md
python -m pytest tests\test_phase26_main_experiment.py -q --basetemp=.pytest_tmp_resume_phase26
```

After that, continue with the real Phase 25/26 main-run execution workflow.
