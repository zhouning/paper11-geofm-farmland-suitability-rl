# Phase 65 Standardized Set-Policy BC Rerun Design

## Purpose

Phase 65 follows the Phase 64 standardization gate. Phase 64 showed that Phase
63 behavior cloning learned the train-tile oracle well, but D4/D6
underperformance coincided with feature scale and effective-rank flags. The
next experiment should therefore isolate one variable: train-tile-fitted feature
standardization before the Phase 63 set-policy behavior-cloning workflow.

Phase 65 is algorithm and experiment evidence work. It does not revise formal
submission files and it does not make manuscript-level claims.

## Scientific Question

Does train-tile-fitted feature standardization remove the conditioning
bottleneck identified in Phase 64 and improve D4/D6 set-policy behavior-cloned
rollout performance relative to B0 and paired D6 controls?

The expected output is a controlled experimental decision. A positive result
would justify continuing the standardized set-policy route. A negative or
inconclusive result would indicate that scale conditioning alone does not
explain the remaining GeoFM shortfall under the current base reward.

## Why Phase 65 Is Needed

Phase 63 showed that the set-policy architecture strongly improves over the
flattened padded PPO baseline, but it did not distinguish GeoFM-derived
variants from B0:

- mean BC reward by variant: B0 `4.9556965601`, D4P8 `4.8935972062`, D4P16
  `4.8822289094`, D6R8 `4.9472654273`, D6R16 `4.9244544652`;
- D4/B0 mean delta: `-0.0677835004`;
- D4/D6 mean delta: `-0.0479468867`;
- oracle gap fraction mean: `0.0882844088`.

Phase 64 then found that behavior-cloning convergence was not the limiting
factor:

- mean best top-1 accuracy: `0.9916666667`;
- mean best top-k hit rate: `1.0`;
- D4 underperformance: `true`;
- D4/D6 scale flag count: `24`;
- D4/D6 rank flag count: `24`;
- standardization gate: `standardization_route_supported`.

The clean next step is not to change reward, architecture, or model capacity.
The clean next step is to rerun the same set-policy BC experiment with a
train-tile-fitted standardization transform and compare paired rows against the
unstandardized Phase 63 outputs.

## Scope

The first Phase 65 implementation should keep the Phase 63 full-run protocol:

- reward: existing deterministic `base_planning_reward`;
- variants: `B0,D4P8,D4P16,D6R8,D6R16`;
- train tile: `tile_r003_c003`;
- eval tiles:
  `tile_r002_c003,tile_r005_c004,tile_r005_c003,tile_r000_c004,tile_r001_c004`;
- seeds: `0,1,2`;
- `eval_max_steps`: `8`;
- `bc_epochs`: `80`;
- `learning_rate`: `0.001`;
- `hidden_dim`: `64`;
- `top_k`: `3`;
- source outputs: the same Phase 2, Phase 8, Phase 61, Phase 13, Phase 52, and
  Phase 62 artifacts used by Phase 63.

The only intentional experimental change is feature standardization for model
inputs. Oracle targets and rollout rewards must remain computed from the raw
unstandardized feature matrices so Phase 65 stays comparable to Phase 63.

## Non-Goals

Phase 65 should not:

- introduce the suitability reward;
- add B2/B3, D2/D3, D5, or transfer experiments;
- change the set-policy scorer architecture;
- change model capacity, optimizer, learning rate, epoch count, seed set, or
  tile protocol in the default run;
- run PPO fine-tuning;
- edit `paper/submission/final/*`;
- claim GeoFM advantage, PCA optimality, independent agronomic suitability, or
  submission readiness.

## Proposed Components

### 1. Train-Tile-Fitted Standardizer

Add a small standardization utility that fits per-feature statistics on the
train tile only for each variant:

- `mean`: feature-wise arithmetic mean on the train tile matrix;
- `std`: feature-wise population standard deviation on the train tile matrix;
- `safe_std`: `std` where `std > 1e-12`, otherwise `1.0`;
- transformed model-input matrix: `(matrix - mean) / safe_std`.

The transform must be fitted separately per variant. Eval tile statistics must
never be used to fit the transform. The transform should preserve tile ID,
variant ID, block IDs, feature columns, reward mode, state groups, and source
metadata while replacing only the state matrix.

The standardized matrix is for policy inputs only. The raw tiled input must
remain available for oracle action construction and reward calculation. This
prevents standardization from silently changing the base-reward task.

The design intentionally uses z-score standardization first. Robust scaling and
min-max scaling are deferred because Phase 65 is meant to isolate whether the
Phase 64 scale/rank warning explains the Phase 63 shortfall.

### 2. Standardized Set-Policy BC Runner

Reuse the Phase 63 oracle, set-policy scorer, training loop shape, greedy
rollout contract, and analysis logic where possible. The runner should:

1. build the same Phase 63 contract;
2. load each variant's raw train tile;
3. fit the standardizer on that raw train tile;
4. build oracle action targets from the raw train tile;
5. train the Phase 63 set-policy scorer using standardized train-tile features
   as model inputs and raw-oracle action targets as labels;
6. load each raw eval tile for the same variant;
7. apply the train-tile-fitted standardizer to create standardized eval model
   inputs;
8. roll out the trained policy greedily using standardized eval model inputs
   for logits and raw eval tile features for rewards;
9. analyze standardized rollout rows using the existing Phase 63 analysis
   function.

Phase 65 should not call Phase 63 helper functions in a way that computes
oracle rankings or rewards from standardized feature values. If a helper assumes
one matrix for both model input and reward, Phase 65 should wrap or duplicate
the minimal needed logic so the policy sees standardized features while the
task remains the raw Phase 63 base-reward task.

### 3. Paired Standardization Comparison

Add a paired comparison between Phase 65 standardized rows and Phase 63
unstandardized rows using `(variant_id, eval_tile_id, seed)` as the key.

Metrics should include:

- standardized BC reward;
- unstandardized BC reward;
- standardized minus unstandardized reward;
- standardized oracle gap fraction;
- unstandardized oracle gap fraction;
- standardized minus unstandardized oracle gap fraction;
- paired D4/B0 reward delta after standardization;
- paired D4/D6 reward delta after standardization;
- whether D4 improves over its own Phase 63 row;
- whether D4 improves over B0 after standardization;
- whether D4 improves over paired D6 after standardization;
- coverage issues for missing, duplicate, or unexpected rows.

The comparison should preserve row-level evidence so the result is auditable and
not only a pooled mean.

### 4. Status Gate

The Phase 65 status should be based on complete paired coverage and conservative
thresholds:

- `standardization_improves_geofm_set_policy`: complete coverage, D4 mean
  standardized-minus-unstandardized reward is positive, D4/B0 mean delta is
  positive, and D4/D6 mean delta is positive.
- `standardization_improves_all_variants_no_geofm_advantage`: complete
  coverage, standardized-minus-unstandardized reward is positive overall, but
  D4 still does not beat B0 or paired D6.
- `standardization_not_helpful`: complete coverage, standardized-minus-
  unstandardized reward is not positive overall and D4 remains behind B0 or
  D6.
- `standardization_hurts_or_inconclusive`: complete coverage but metrics point
  in conflicting directions, or standardization degrades mean reward while
  improving one secondary comparison.
- `insufficient`: missing artifacts, missing paired rows, duplicate rows, or
  invalid coverage.

The gate should include numeric summaries rather than only a label:

- mean reward by variant after standardization;
- mean standardized-minus-unstandardized reward by variant;
- overall standardized-minus-unstandardized reward summary;
- D4/B0 delta summary after standardization;
- D4/D6 delta summary after standardization;
- oracle gap fraction summary after standardization;
- coverage issue counts.

### 5. Artifacts

Suggested implementation files for the later implementation plan:

- `src/paper11_geofm/phase65_standardized_set_policy_bc_rerun.py`
- `experiments/phase65_standardized_set_policy_bc_rerun/run_phase65_standardized_set_policy_bc_rerun.py`
- `tests/test_phase65_standardized_set_policy_bc_rerun.py`
- `paper/phase28_results/31_phase65_standardized_set_policy_bc_rerun.md`

Generated artifacts should remain under ignored experiment output directories:

- `phase65_standardization_stats.json`
- `phase65_bc_training_history.csv`
- `phase65_bc_rollout_summary.csv`
- `phase65_set_policy_comparison.json`
- `phase65_standardization_pairwise_delta.csv`
- `phase65_standardized_set_policy_bc_rerun.md`

The implementation should not write into `paper/submission/final/*`.

## Error Handling

The implementation should fail clearly when:

- Phase 63 comparison or rollout artifacts are missing for paired comparison;
- required variant source directories are missing from the contract;
- the train tile has no blocks or no feature columns;
- a standardized train/eval tile changes feature count, block ID order, or
  feature column order unexpectedly;
- a standardized-model-input row cannot be aligned with the corresponding raw
  reward row by action index or block ID;
- paired comparison finds missing or duplicate `(variant_id, eval_tile_id,
  seed)` rows.

Zero-variance feature columns should not fail the run. They should use
`safe_std = 1.0` and be counted in `phase65_standardization_stats.json`.

## Testing Requirements

Unit tests should cover:

- fitting standardization on the train tile only;
- applying train-fitted statistics to eval tiles without using eval means or
  standard deviations;
- zero-variance feature handling through `safe_std`;
- preservation of tile metadata and block ordering after standardization;
- raw-feature oracle targets and raw-feature rollout rewards are unchanged by
  model-input standardization;
- paired standardized versus unstandardized delta calculation;
- Phase 65 status outcomes for improvement, all-variant improvement without
  GeoFM advantage, not-helpful, inconclusive, and insufficient cases;
- writer outputs JSON, CSV, and Markdown artifacts on tiny synthetic fixtures;
- CLI parser accepts the Phase 63-compatible inputs and prior Phase 63 artifact
  paths.

Verification for the later implementation should include:

- targeted Phase 65 unit tests;
- Phase 63 and Phase 64 regression tests;
- `D:\adk\.venv\Scripts\python.exe scripts\smoke_check.py`;
- `git diff --check`;
- a check that `paper/submission/final/*` remains unchanged.

## Claim Boundary

Phase 65 may report whether train-tile-fitted feature standardization improves
base-reward set-policy behavior-cloned rollout performance under the Phase 63
protocol. It may not claim that GeoFM improves farmland suitability planning,
that PCA is optimal, that B2/B3 or transfer behavior is known, or that the
current manuscript is ready for submission.

The manuscript remains downstream of algorithm and experiment evidence.
