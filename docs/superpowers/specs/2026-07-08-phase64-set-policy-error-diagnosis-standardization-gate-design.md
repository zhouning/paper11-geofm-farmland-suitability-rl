# Phase 64 Set-Policy Error Diagnosis and Standardization Gate Design

## Purpose

Phase 64 follows Phase 63's set-policy oracle-pretraining result. Phase 63
showed that the task-aware set-policy architecture strongly improves over the
flattened padded PPO baseline, but it did not distinguish GeoFM-derived D4
features from B0 or D6 controls. B0 remained slightly ahead of D4 and D6 in
mean behavior-cloned rollout reward.

The purpose of Phase 64 is to diagnose that remaining error before running a
new training phase. The phase should explain whether the observed D4/D6
shortfall is more consistent with feature scale or conditioning, behavior
cloning capacity, rollout ranking mistakes, or genuinely weak GeoFM-derived
state for the current base-reward task.

Phase 64 is algorithm and experiment infrastructure work. It does not revise
formal submission files.

## Scientific Question

Why does B0 remain slightly ahead of D4 and D6 under the Phase 63 set-policy
behavior-cloned rollout, and is a train-tile-fitted feature standardization
rerun justified as the next experiment?

The expected output is a diagnostic decision, not a manuscript claim.

## Why Phase 64 Is Needed

Phase 63 produced three important facts:

- the set-policy architecture delta versus flattened PPO is large and complete:
  mean `4.4387176072`, positive `75 / 75`;
- the oracle gap remains positive but moderate: mean gap fraction
  `0.0882844088`;
- D4 does not beat B0 or D6 under the same set-policy route:
  D4/B0 mean delta `-0.0677835004`, D4/D6 mean delta `-0.0479468867`.

Those facts support the architecture route but do not yet support a GeoFM
advantage claim. A direct PPO fine-tuning run would be expensive and premature
because it would not explain why the supervised set-policy still misses oracle
blocks. A direct standardization rerun would be faster, but without diagnosis it
would be hard to defend scientifically. Phase 64 therefore adds an explicit
error analysis and a standardization readiness gate.

## Inputs

Phase 64 should treat Phase 63 artifacts as the primary input:

- `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_set_policy_comparison.json`
- `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_bc_rollout_summary.csv`
- `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_bc_training_history.csv`
- `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_set_policy_delta_table.csv`
- `experiments/phase63_set_policy_oracle_pretraining/outputs/phase52_full5_seed3/phase63_oracle_summary.csv`

For feature-scale diagnostics, Phase 64 should reload the same tiled variant
inputs used by Phase 63 through the existing tiled input loaders and Phase 63
contract metadata. The first diagnostic run should use the same variants,
single train tile, five eval tiles, three seeds, and `eval_max_steps=8` as the
Phase 63 full run:

- variants: `B0,D4P8,D4P16,D6R8,D6R16`;
- train tile: `tile_r003_c003`;
- eval tiles: `tile_r002_c003,tile_r005_c004,tile_r005_c003,tile_r000_c004,tile_r001_c004`;
- seeds: `0,1,2`.

## Non-Goals

Phase 64 should not:

- run PPO fine-tuning;
- introduce the suitability reward;
- add B2/B3 or transfer experiments;
- change the Phase 63 trained policy behavior;
- edit `paper/submission/final/*`;
- claim GeoFM advantage, PCA optimality, independent agronomic suitability, or
  submission-level planning performance.

## Diagnostic Modules

### 1. Training Convergence Diagnostics

Aggregate `phase63_bc_training_history.csv` by variant, seed, and train tile.
Report first epoch, final epoch, best epoch, final loss, best loss, final top-1
accuracy, best top-1 accuracy, final top-k hit rate, and best top-k hit rate.

This module should identify whether D4 or D6 variants are less learnable under
the existing supervised setup. It should also flag broad behavior-cloning
capacity limits if all variants plateau with weak top-1 or top-k behavior.

### 2. Rollout Selected-Block Overlap

Compare each greedy rollout's selected block IDs with the corresponding oracle
selected block IDs from `phase63_oracle_summary.csv`.

Metrics should include:

- exact selected-set overlap count and fraction;
- prefix overlap count for the first `eval_max_steps` actions;
- Jaccard similarity between selected and oracle block sets;
- number of duplicate or invalid rollout selections, expected to remain zero;
- selected reward total versus oracle reward total.

This module distinguishes harmless alternate high-reward choices from true
missed-oracle failures.

### 3. Oracle-Rank Gap and Missed-Oracle Blocks

Rebuild the per-block base-reward ranking for each variant and tile. For each
rollout row, compute:

- oracle rank of every selected block;
- reward contribution of selected blocks;
- missed oracle block IDs and rewards;
- reward loss attributable to missed oracle blocks;
- worst selected block rank within the rollout;
- number of selected blocks outside the oracle top `eval_max_steps`, top `16`,
  and top `32`.

Because Phase 63's oracle reward is identical across variants, this diagnostic
should focus on policy ranking behavior rather than reward-definition changes.

### 4. Feature Scale and Effective-Rank Audit

Reload each variant's train and eval tile feature matrices and compute
train-tile-fitted scale diagnostics without using eval statistics to define any
transform.

Per variant and tile, report:

- feature count and block count;
- per-feature mean, standard deviation, min, max, median, p1, and p99;
- zero-variance feature count;
- maximum feature standard deviation divided by minimum non-zero standard
  deviation;
- maximum absolute mean divided by median non-zero standard deviation;
- train-to-eval z-shift using train-tile mean and standard deviation;
- effective rank from singular values of the centered feature matrix;
- share of variance explained by the first component and first three
  components.

This audit should identify whether D4 or D6 are disadvantaged by scale,
conditioning, or rank concentration before any standardization experiment is
scheduled.

### 5. Failure Case Table

Build a compact table of the most informative failures:

- highest oracle gap fraction rows;
- D4 rows where D4 loses most to B0;
- D4 rows where D4 loses most to paired D6;
- rows with weak selected-block overlap despite moderate total reward;
- rows with feature-scale or train-to-eval shift flags.

Each failure row should include variant, tile, seed, BC reward, oracle reward,
oracle gap, selected block IDs, missed oracle block IDs, overlap metrics, and
the relevant training endpoint metrics.

### 6. Standardization Gate

Produce a standardization decision file that recommends whether the next phase
should run a train-tile-fitted standardized set-policy BC rerun.

The gate should use only Phase 63 artifacts and Phase 64 diagnostics. It should
not run the standardized training itself.

Suggested gate logic:

- return `diagnostic_inconclusive` if Phase 63 coverage is incomplete or the
  required artifacts are missing;
- return `bc_training_capacity_limited` if all variants show weak convergence
  and the error pattern is not variant-specific;
- return `standardization_route_supported` if D4/D6 underperformance coincides
  with scale, conditioning, or train-to-eval shift flags while the oracle ceiling
  and selected-block reward task remain otherwise comparable;
- return `geofm_features_not_helpful_under_set_policy` if convergence is
  adequate, selected-block errors are interpretable, scale diagnostics are not
  concerning, and D4 still does not match B0 or D6;
- return `diagnostic_inconclusive` if the metrics disagree in a way that cannot
  support a single next experiment.

The gate should include enough numeric evidence for the decision, including
which flags fired and which did not.

## Proposed Files

Implementation files for the later implementation plan:

- `src/paper11_geofm/phase64_set_policy_error_diagnosis.py`
- `experiments/phase64_set_policy_error_diagnosis/run_phase64_set_policy_error_diagnosis.py`
- `tests/test_phase64_set_policy_error_diagnosis.py`
- `paper/phase28_results/30_phase64_set_policy_error_diagnosis.md`

Generated artifacts should remain under ignored experiment output directories:

- `phase64_convergence_summary.csv`
- `phase64_rollout_overlap.csv`
- `phase64_oracle_rank_gap.csv`
- `phase64_feature_scale_summary.csv`
- `phase64_feature_effective_rank.csv`
- `phase64_failure_cases.csv`
- `phase64_standardization_gate.json`
- `phase64_set_policy_error_diagnosis.md`

## Testing Requirements

Unit tests should cover:

- parsing semicolon-separated selected block IDs and action indices;
- selected-block overlap with exact matches, partial matches, and disjoint
  selections;
- oracle-rank gap calculations under ties with deterministic tie-breaking;
- feature-scale summaries with zero-variance features and zero singular values;
- effective-rank calculations on low-rank and full-rank synthetic matrices;
- standardization gate statuses for supported, capacity-limited, not-helpful,
  and inconclusive cases;
- CLI artifact writing on tiny synthetic Phase 63-style fixtures.

Verification should include:

- targeted Phase 64 unit tests;
- the relevant Phase 63 regression tests;
- `git diff --check`;
- a check that `paper/submission/final/*` remains unchanged.

Use the stable project Python environment for torch-adjacent verification:
`D:\adk\.venv\Scripts\python.exe`.

## Claim Boundary

Phase 64 may report diagnostic evidence about why Phase 63's set-policy
behavior-cloned rollout misses oracle blocks and whether a standardized rerun is
scientifically justified. It may not claim that GeoFM improves farmland
suitability planning, that PCA is optimal, that B2/B3 or transfer behavior is
known, or that the current manuscript is submission-ready.

The manuscript remains downstream of algorithm and experiment evidence.
