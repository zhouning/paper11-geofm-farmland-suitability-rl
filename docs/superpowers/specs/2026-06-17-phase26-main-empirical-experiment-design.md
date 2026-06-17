# Phase 26 Main Empirical Experiment Design

## Goal

Turn the Phase 25 padded held-out policy smoke pilot into the first Paper11
main empirical result package.

Phase 26 should answer one bounded empirical question:

```text
Under a deterministic base planning reward, does the GeoFM-enhanced B1 learned
policy outperform the explicit-feature B0 learned policy across multiple
held-out Bishan tiles and multiple random seeds?
```

## Motivation

Phase 25 closes the flat observation/action shape blocker by adding a padded
variable-size MaskablePPO contract. The verified Windows smoke result trains
on `tile_r003_c003`, evaluates on held-out `tile_r002_c003`, and reports a
positive B1-B0 learned-policy reward delta under a short budget.

That smoke result is useful but not yet manuscript-grade empirical evidence.
Paper11 now needs a main empirical package that broadens Phase 25 across
multiple held-out tiles and seeds, records stability, and produces tables that
can be used in the Results section without overclaiming.

## Claim Target

Phase 26 may support this bounded claim if results are positive and stable:

```text
Within real Bishan held-out tiles and under a deterministic base planning
reward, the GeoFM-enhanced B1 representation yields a higher padded
MaskablePPO learned-policy reward than the explicit-feature B0 representation
under the same train/evaluation protocol.
```

Phase 26 must not claim:

- suitability-reward benefit;
- B2/B3 full-model superiority;
- cross-region transfer beyond Bishan held-out tiles;
- final IJAEOG submission-level planning performance;
- calibrated agronomic suitability.

## Experiment Scope

Phase 26 is restricted to:

- variants: `B0` and `B1`;
- reward: deterministic `base_planning_reward`;
- train/evaluation unit: Phase 13 real Bishan tiles;
- policy: existing Phase 25 padded MaskablePPO runner with `MlpPolicy`;
- train tile: default largest tile, currently `tile_r003_c003`;
- held-out evaluation tiles: default largest distinct `3` tiles;
- seeds: default `0,1,2`;
- baselines: Phase 25 `first_valid` and `seeded_random`;
- local Windows run: timing probe and short smoke only;
- main run: Colab Pro+ or another stronger training platform.

Do not enable B2/B3 or suitability reward. Phase 10 still reports:

```text
status: not_ready_for_suitability_reward
recommendation: do_not_enable_suitability_reward
```

## Recommended Run Protocol

### Windows Timing Probe

Use Windows for a short runtime and artifact check:

```powershell
python experiments\phase25_padded_heldout_policy\run_phase25_padded_heldout_policy.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1 --total-timesteps 128 --eval-max-steps 4 --seeds 0 --max-eval-tiles 1 --output-dir experiments\phase26_main_experiment\outputs\windows_timing_probe\phase25_run
```

Expected purpose:

- confirm the current machine can still import the RL stack;
- estimate runtime per variant/seed/tile;
- verify output schemas before the main run;
- avoid treating the timing probe as the main result.

### Colab Pro+ Main Run

Use Colab Pro+ for the main empirical run:

```bash
python experiments/phase25_padded_heldout_policy/run_phase25_padded_heldout_policy.py --phase2-output-dir experiments/phase11_bishan_dltb_real/outputs/phase2_real --tile-index-csv experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv --variants B0,B1 --total-timesteps 1024 --eval-max-steps 8 --seeds 0,1,2 --max-eval-tiles 3 --output-dir experiments/phase26_main_experiment/outputs/colab_main/phase25_run
```

If the timing probe is stable and Colab runtime allows, repeat with:

```bash
--total-timesteps 4096
```

The Phase 26 analyzer should treat the selected main run directory as an input
and should record the training budget in every output artifact.

## Phase 26 Analysis Module

Create a new analysis module:

```text
src/paper11_geofm/phase26_main_experiment.py
```

Responsibilities:

1. Read Phase 25 `phase25_padded_heldout_policy_summary.csv`.
2. Read Phase 25 `phase25_padded_heldout_policy_comparison.json`.
3. Validate that the input is B0/B1 only and uses `base_planning_reward`.
4. Aggregate learned-policy reward by variant, seed, and held-out tile.
5. Compute B1-B0 deltas by seed and tile.
6. Compute mean, standard deviation, minimum, maximum, and positive-count
   diagnostics for learned-policy deltas.
7. Preserve baseline comparisons for `first_valid` and `seeded_random`.
8. Produce a conservative claim-readiness status.

## Phase 26 Runner

Create a new runner:

```text
experiments/phase26_main_experiment/run_phase26_main_experiment.py
```

The runner should support two modes:

1. `analyze-only`: read an existing Phase 25 output directory and write Phase
   26 analysis artifacts.
2. `run-and-analyze`: call the existing Phase 25 runner first, then analyze
   the produced outputs.

Default mode should be `analyze-only` so Colab-produced outputs can be copied
back and analyzed without rerunning training.

Minimum CLI arguments:

```text
--phase25-output-dir
--output-dir
--mode analyze-only|run-and-analyze
--phase2-output-dir
--tile-index-csv
--variants
--total-timesteps
--eval-max-steps
--seeds
--max-eval-tiles
```

Only `--phase25-output-dir` and `--output-dir` are required in `analyze-only`
mode. The Phase 25 run arguments are required in `run-and-analyze` mode.

## Outputs

Phase 26 writes:

```text
phase26_main_summary.csv
phase26_tile_seed_delta_table.csv
phase26_main_comparison.json
phase26_claim_readiness.md
```

### `phase26_main_summary.csv`

One row per row type, variant, and evaluation tile:

- `row_type`;
- `variant_id`;
- `eval_tile_id`;
- `seed_count`;
- `mean_total_contract_reward`;
- `std_total_contract_reward`;
- `min_total_contract_reward`;
- `max_total_contract_reward`;
- `train_timesteps`;
- `eval_max_steps`;
- `claim_boundary`.

### `phase26_tile_seed_delta_table.csv`

One row per held-out tile and seed for learned-policy B1-B0 deltas:

- `eval_tile_id`;
- `seed`;
- `b0_reward`;
- `b1_reward`;
- `b1_minus_b0_reward`;
- `b1_improves_b0`;
- `train_timesteps`;
- `eval_max_steps`.

### `phase26_main_comparison.json`

Include:

- source Phase 25 artifact paths;
- variants, seeds, held-out tiles, train tile, training timesteps, evaluation
  max steps;
- learned-policy mean reward by variant;
- learned-policy B1-B0 mean delta;
- learned-policy B1-B0 standard deviation;
- positive tile-seed count and total tile-seed count;
- per-tile mean deltas;
- per-seed mean deltas;
- baseline summaries;
- `phase26_claim_status`;
- remaining evidence gaps.

### `phase26_claim_readiness.md`

Write a short reviewer-facing note with:

- empirical setup;
- main learned-policy result;
- stability diagnostics;
- whether the result supports, weakly supports, or fails to support the bounded
  B1-over-B0 claim;
- explicit remaining gaps.

## Claim Status Rules

Let `delta_mean` be the learned-policy mean B1-B0 reward delta across all
held-out tile and seed combinations.

Let `positive_count` be the number of tile-seed combinations where B1-B0 is
strictly positive.

Let `total_count` be the number of valid tile-seed combinations.

Use these statuses:

- `pilot_supported`: `delta_mean > 0` and `positive_count / total_count >= 0.6`;
- `mixed`: `delta_mean > 0` but `positive_count / total_count < 0.6`;
- `not_supported`: `delta_mean <= 0`;
- `insufficient`: missing B0/B1 rows, no held-out tiles, no seeds, or invalid
  input artifacts.

The threshold is deliberately conservative and transparent. Phase 26 should
report the raw counts so readers can judge stability.

## Testing Requirements

Add tests for:

1. Reading a tiny Phase 25 summary/comparison fixture.
2. Computing B1-B0 tile-seed deltas, including zero and negative cases.
3. Claim status rules for `pilot_supported`, `mixed`, `not_supported`, and
   `insufficient`.
4. Writing all four Phase 26 artifacts.
5. CLI `analyze-only` mode.
6. CLI validation that `run-and-analyze` requires Phase 25 run inputs.

Use synthetic fixtures for tests. Do not run long RL training in Phase 26 unit
tests; Phase 25 already covers the padded MaskablePPO integration.

## Documentation Updates

Update:

- `README.md`;
- `reproducibility/REPRODUCTION_GUIDE.md`;
- `reproducibility/FILE_MANIFEST.tsv`;
- `paper/submission/01_ijaeog_submission_readiness.md`;
- `paper/submission/02_draft_titles_highlights_declarations.md`.

The docs should distinguish:

- Windows timing probe;
- Colab Pro+ main run;
- analysis-only reproduction;
- claim status and remaining evidence gaps.

## Success Criteria

Phase 26 is successful when:

1. The analyzer can ingest Phase 25 outputs and produce all Phase 26 artifacts.
2. The tile-seed delta table exposes whether B1 improvement is stable.
3. The comparison JSON reports a conservative claim status.
4. The documentation gives exact commands for timing probe, Colab main run, and
   analysis-only reproduction.
5. Tests pass without requiring long training.

## Evidence Boundaries

Phase 26 can strengthen Paper11 from a smoke-level held-out learned-policy pilot
to a multi-seed, multi-held-out-tile Bishan empirical result under
`base_planning_reward`.

It still cannot support:

- suitability reward claims;
- B2/B3 full-model claims;
- cross-region transfer;
- calibrated agronomic suitability;
- final submission claims without figures, ablations, uncertainty, and a
  manuscript-ready results package.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: Phase 26 reuses Phase 25 for training and adds analysis;
  it does not introduce a new reward or representation family.
- Scope check: Phase 26 is a single implementation plan: analysis package plus
  optional run-and-analyze wrapper.
- Ambiguity check: run protocol, outputs, claim rules, tests, and evidence
  boundaries are explicit.
