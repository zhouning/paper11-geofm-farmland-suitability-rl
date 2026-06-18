# Phase 27 B0/B1 Stability Diagnosis Design

## Goal

Diagnose why the current Phase 26 padded held-out Bishan B0/B1 learned-policy
evidence does not support a positive GeoFM-enhanced B1-over-B0 claim.

Phase 27 should answer one bounded diagnostic question:

```text
Across the existing 1024-step and 4096-step Phase 26 result sets, are B1-B0
learned-policy outcomes stable by tile and seed, or are they dominated by
budget sensitivity and tile-seed flips?
```

## Motivation

Phase 26 converted padded held-out Phase 25 outputs into manuscript-facing
tables, but both available main result sets are negative on average:

- 1024 steps: B1-B0 learned-policy mean delta `-0.4329022862`, positive
  tile-seed count `4 / 9`, claim status `not_supported`;
- 4096 steps: B1-B0 learned-policy mean delta `-0.1318712688`, positive
  tile-seed count `3 / 9`, claim status `not_supported`.

The mean improves with a larger budget, but the positive count worsens. That
means the next work should diagnose stability before launching more training or
writing a positive performance claim.

## Claim Target

Phase 27 may support this bounded diagnosis:

```text
The current B0/B1 learned-policy evidence is unstable across budgets and
tile-seed pairs, so Phase 26 should remain negative evidence and the next
experimental step should focus on representation controls or repeated-budget
stability rather than manuscript-level B1 superiority.
```

Phase 27 must not claim:

- that GeoFM improves planning decisions;
- that longer training is sufficient;
- suitability-reward readiness;
- B2/B3 superiority;
- cross-region transfer;
- final IJAEOG submission readiness.

## Experiment Scope

Phase 27 is a read-only diagnostic package. It does not run RL training and
does not modify Phase 25 or Phase 26 protocols.

Inputs are the existing Phase 26 analysis artifacts:

```text
experiments/phase26_main_experiment/outputs/macos_main/phase26_analysis/phase26_main_comparison.json
experiments/phase26_main_experiment/outputs/macos_main_4096/phase26_analysis/phase26_main_comparison.json
```

The analyzer must use the `tile_seed_delta_rows` embedded in each Phase 26
comparison JSON. It should pair rows by `(eval_tile_id, seed)` and compare the
learned-policy B1-B0 deltas across budgets.

## Diagnostic Classifications

For each paired tile-seed row:

- `stable_positive`: B1-B0 is positive in both budgets;
- `stable_negative`: B1-B0 is non-positive in both budgets;
- `flip_to_positive`: B1-B0 is non-positive at the lower budget and positive
  at the higher budget;
- `flip_to_negative`: B1-B0 is positive at the lower budget and non-positive
  at the higher budget.

The diagnostic summary should also count:

- stable-positive pairs;
- stable-negative pairs;
- flip-to-positive pairs;
- flip-to-negative pairs;
- incomplete pairs;
- lower-budget and higher-budget mean deltas;
- mean change from lower to higher budget;
- positive-count change.

## Outputs

Phase 27 writes:

```text
phase27_budget_transition_table.csv
phase27_tile_seed_stability.csv
phase27_diagnostic_summary.json
phase27_diagnostic_readiness.md
```

### `phase27_budget_transition_table.csv`

One row per budget input:

- `budget_label`;
- `train_timesteps`;
- `eval_max_steps`;
- `b1_minus_b0_mean_reward`;
- `positive_tile_seed_count`;
- `total_tile_seed_count`;
- `positive_fraction`;
- `phase26_claim_status`;
- `mean_delta_change_from_previous`;
- `positive_count_change_from_previous`;
- `claim_boundary`.

### `phase27_tile_seed_stability.csv`

One row per paired tile-seed:

- `eval_tile_id`;
- `seed`;
- `lower_budget_label`;
- `higher_budget_label`;
- `lower_train_timesteps`;
- `higher_train_timesteps`;
- `lower_b1_minus_b0_reward`;
- `higher_b1_minus_b0_reward`;
- `delta_change`;
- `lower_b1_improves_b0`;
- `higher_b1_improves_b0`;
- `stability_class`;
- `diagnostic_note`.

### `phase27_diagnostic_summary.json`

Include:

- source comparison JSON paths;
- ordered budgets;
- budget summary records;
- tile-seed stability counts;
- per-tile transition summaries;
- per-seed transition summaries;
- `phase27_diagnostic_status`;
- recommendation;
- remaining evidence gaps;
- claim boundary.

### `phase27_diagnostic_readiness.md`

Write a short reviewer-facing note with:

- diagnostic setup;
- budget-level result;
- tile-seed stability result;
- interpretation;
- claim boundary;
- recommended next experiment.

## Diagnostic Status Rules

Use these statuses:

- `budget_not_explanatory`: both budgets have `not_supported` Phase 26 status,
  higher-budget mean delta is still non-positive, and higher-budget positive
  fraction is below `0.6`;
- `budget_promising_unstable`: higher-budget mean delta becomes positive but
  higher-budget positive fraction remains below `0.6`;
- `budget_promising_stable`: higher-budget mean delta is positive and
  higher-budget positive fraction is at least `0.6`;
- `insufficient`: missing budgets, unpaired tile-seed rows, or invalid input.

The current expected macOS diagnosis is `budget_not_explanatory`.

## Testing Requirements

Add tests for:

1. Reading two tiny Phase 26 comparison fixtures.
2. Building a budget transition table with mean and positive-count changes.
3. Classifying tile-seed pairs as stable positive, stable negative,
   flip-to-positive, and flip-to-negative.
4. Producing `budget_not_explanatory` on negative higher-budget evidence.
5. Reporting `insufficient` when tile-seed coverage differs across budgets.
6. Writing all four Phase 27 artifacts.
7. CLI behavior for valid inputs and missing/invalid inputs.

Use synthetic fixtures. Do not run RL training in Phase 27 tests.

## Documentation Updates

Update:

- `README.md`;
- `reproducibility/REPRODUCTION_GUIDE.md`;
- `reproducibility/FILE_MANIFEST.tsv`;
- `paper/submission/01_ijaeog_submission_readiness.md`;
- `paper/phase26_results/02_next_experiment_matrix.md`;
- a new `paper/phase27_results/` package.

The docs should state that Phase 27 is diagnostic evidence only and that the
current B1-over-B0 learned-policy claim remains unsupported.

## Success Criteria

Phase 27 is successful when:

1. The analyzer ingests two Phase 26 comparison JSONs without rerunning
   training.
2. The budget transition table exposes whether the larger budget improved the
   mean and positive-count stability.
3. The tile-seed stability table exposes all sign flips.
4. The diagnostic summary assigns a conservative status and recommendation.
5. Tests pass without requiring GPU, internet, or long training.
6. Documentation keeps the evidence boundary explicit.

## Evidence Boundaries

Phase 27 can guide the next experimental decision. It cannot turn the current
negative Phase 26 evidence into a positive manuscript claim.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: Phase 27 consumes Phase 26 outputs only and does not
  introduce new training, rewards, or variants.
- Scope check: the work is a single implementation plan: diagnostic analyzer,
  CLI runner, tests, and documentation.
- Ambiguity check: input files, outputs, classification rules, status rules,
  tests, and claim boundaries are explicit.
