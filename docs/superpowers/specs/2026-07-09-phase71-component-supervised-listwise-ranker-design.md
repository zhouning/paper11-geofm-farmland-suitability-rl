# Phase 71 Component-Supervised Listwise Ranker Design

## Goal

Build Phase 71 as a new algorithm/model experiment after Phase 70 showed that train-tile-fitted standardization is insufficient. Phase 71 tests whether a component-supervised learning-to-rank objective can improve base-reward block selection over the Phase 63/70 behavior-cloned set-policy route while keeping GeoFM-specific claims secondary.

## Background

Phase 63 established that the set-policy architecture improves over the flattened PPO route, but it did not distinguish GeoFM-derived variants from B0. Phase 64 identified feature-scale and rank flags and justified a standardized rerun. Phase 70 executed that rerun with a dual-matrix protocol and returned `standardization_not_sufficient`: overall Phase70 minus Phase63 mean delta was negative, and D4 remained behind B0 and D6 on mean standardized deltas.

The key constraint is that the current `base_planning_reward` is computed entirely from explicit planning features. B0 directly exposes those inputs, while GeoFM-derived variants are being evaluated against a target that is largely explicit-feature aligned. Therefore Phase 71 must not use GeoFM-specific superiority as the primary success criterion. The main question becomes whether a better supervised ranking objective can learn the current decision task more effectively and more stably than trajectory-level behavior cloning.

## Claim Boundary

Phase 71 is an algorithm/model experiment under the existing Bishan base-reward protocol. It does not alter the reward, enable B2/B3, introduce suitability reward, validate independent agronomic suitability, prove GeoFM superiority, prove PCA optimality, test transfer, or justify formal submission-level claims. Formal manuscript files under `paper/submission/final/*` remain out of scope.

## Recommended Approach

Use a component-supervised listwise ranker.

Instead of training only from deterministic top-k oracle trajectories, Phase 71 will train from all available blocks in the training tiles. Each block receives deterministic supervision from the existing base-reward total and its explicit component decomposition. The ranker then scores every block in a held-out tile and selects the top `eval_max_steps` blocks greedily. This directly optimizes the selection problem while preserving the existing reward definition.

## Experimental Protocol

Phase 71 should use the same five evaluation tiles, three seeds, variants, and `eval_max_steps=8` as Phase 63/70 unless a fixture test uses smaller synthetic data. The full real run uses variants:

- `B0`
- `D4P8`
- `D4P16`
- `D6R8`
- `D6R16`

The real held-out tiles remain:

- `tile_r002_c003`
- `tile_r005_c004`
- `tile_r005_c003`
- `tile_r000_c004`
- `tile_r001_c004`

Training uses a leave-one-eval-tile protocol over these selected tiles plus the Phase 63 train tile when available. For each target eval tile, the model trains on all blocks from the other available training tiles for the same variant. The held-out tile is never used to fit standardization, model weights, or calibration parameters for that fold.

## Model

The initial Phase 71 model should be intentionally modest:

- Input: variant feature matrix for one block.
- Preprocessing: train-fold-fitted z-score standardization, using safe scale replacement for zero-variance features.
- Backbone: small multilayer perceptron producing a scalar ranking score and component-contribution predictions per block.
- Loss: ListNet-style listwise cross-entropy between predicted scores and reward-derived target probabilities within each training tile, plus a small component MSE term against deterministic reward-component contributions.
- Tie handling: deterministic tie-break by block ID and original action index for oracle-equivalent ordering.

The implementation should keep B0, D4, and D6 on separate per-variant models in the first pass. A shared multi-variant model is out of scope for Phase 71 because it would add confounding before the per-variant ranker baseline is established.

## Supervision

The primary ranking target is the existing base-planning reward total computed from `compute_base_planning_reward_from_matrix_row`. Phase 71 also uses deterministic reward-component contributions as auxiliary supervised targets. Component decomposition should mirror the current reward formula:

- low-slope farmland or orchard contribution
- current farmland or orchard contribution
- low-slope contribution
- area contribution
- mean-slope penalty
- max-slope penalty
- built-up penalty
- water penalty

The component targets regularize the model toward the known reward structure and the component rows are diagnostic. They do not define a new reward.

## Evaluation

For each variant, eval tile, and seed/fold, Phase 71 will output:

- selected block IDs and action indices
- total selected base reward
- oracle total reward for the same tile and budget
- oracle gap and oracle gap fraction
- top-k overlap with the deterministic oracle
- worst selected oracle rank
- compared Phase 63 reward when the matching row exists
- compared Phase 70 reward when the matching row exists

Aggregate analysis will report:

- mean Phase71 reward by variant
- Phase71 minus Phase63 summary
- Phase71 minus Phase70 summary
- D4 versus B0 summary
- D4 versus D6 summary
- coverage issues for missing or duplicate rows

## Status Model

Phase 71 should classify outcomes conservatively:

- `ranker_improves_decision_route`: Phase 71 improves mean reward over both Phase 63 and Phase 70 with adequate positive-row coverage, regardless of GeoFM-specific ordering.
- `ranker_improves_but_target_masks_geofm`: Phase 71 improves the decision route, but D4 does not beat B0 and D6, consistent with the explicit-feature target masking GeoFM-specific value.
- `ranker_supports_geofm_followup`: Phase 71 improves the decision route and D4 also beats B0 and D6 on mean paired deltas with adequate positive-row coverage.
- `ranker_not_sufficient`: Phase 71 does not improve over the Phase 63/70 references.
- `ranker_incomplete`: expected variant/tile/seed coverage is missing or duplicated.

The likely scientifically honest success state is `ranker_improves_but_target_masks_geofm`. That would still be useful because it establishes a stronger algorithm baseline while preserving the target limitation identified in Phase 66 and Phase 70.

## Artifacts

Phase 71 should write generated artifacts under `experiments/phase71_component_supervised_ranker/outputs/phase52_full5_seed3/`:

- `phase71_ranker_training_history.csv`
- `phase71_ranker_rollout_summary.csv`
- `phase71_ranker_oracle_summary.csv`
- `phase71_ranker_component_diagnostics.csv`
- `phase71_ranker_delta_table.csv`
- `phase71_component_supervised_ranker.json`
- `phase71_component_supervised_ranker.md`

The repository-tracked result note should be:

- `paper/phase28_results/37_phase71_component_supervised_ranker.md`

The result note should record the status, main deltas, reproduction command, and claim boundary. It should not revise formal manuscript files.

## Testing Strategy

Use TDD for implementation. Required coverage:

- reward component decomposition sums to the existing base reward within rounding tolerance
- train-fold standardization uses only training rows and preserves held-out reward matrices
- pairwise/listwise examples order blocks by reward without leaking held-out rows
- greedy rollout scores selected blocks with the original reward matrix
- comparison builder distinguishes decision-route improvement, GeoFM follow-up support, target-masked improvement, insufficiency, and incomplete coverage
- writer emits JSON, CSV, and Markdown artifacts with stable filenames
- CLI runner succeeds on a small fixture using real `load_tiled_variant_input` paths and a synthetic tile index

## Non-Goals

Phase 71 will not train PPO, alter the deterministic base reward, introduce external labels, enable suitability reward, create B2/B3 variants, merge all variants into one shared model, tune many architectures, or change the formal submission manuscript. Any manuscript changes come only after algorithm and experiment evidence is stronger.