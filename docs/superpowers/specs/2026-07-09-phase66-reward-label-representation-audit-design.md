# Phase 66 Reward-Label Representation Audit Design

## Purpose

Phase 66 follows the Phase 65 standardized set-policy rerun. Phase 65 showed
that train-tile-fitted z-score standardization does not recover a stable D4/D6
advantage under the Phase 63 set-policy behavior-cloning route. B0 improves on
average after standardization, D4P8 improves in many row-level cases, but D4P16
falls sharply and D4 remains behind B0 and paired D6 controls overall.

The next step should not be another training tweak. Phase 66 should explain
whether the current base-reward target itself is masking, duplicating, or
failing to use GeoFM-derived information. It is a read-only attribution audit
over existing Phase 63, Phase 64, and Phase 65 artifacts plus the underlying
tiled feature matrices.

Phase 66 is algorithm and experiment evidence work. It does not revise formal
submission files and it does not make manuscript-level claims.

## Scientific Question

Under the current deterministic `base_planning_reward`, do GeoFM-derived D4/D6
representations provide block-ranking information beyond the nine explicit
planning features that directly define the reward, or is the current set-policy
target mostly redundant with explicit-feature ranking?

The expected output is a diagnostic decision about the next algorithm route:
continue representation modeling, redesign the reward/label target, or treat
the current base-reward protocol as unsuitable for demonstrating GeoFM
advantage.

## Why Phase 66 Is Needed

Phase 63 established that set-policy behavior cloning is a strong architecture
route, but not a GeoFM advantage:

- architecture delta versus flattened PPO: mean `4.4387176072`, positive
  `75 / 75`;
- D4/B0 mean delta: `-0.0677835004`;
- D4/D6 mean delta: `-0.0479468867`;
- oracle gap fraction mean: `0.0882844088`.

Phase 64 diagnosed a plausible feature-conditioning issue:

- mean best top-1 accuracy: `0.9916666667`;
- mean best top-k hit rate: `1.0`;
- D4/D6 scale flag count: `24`;
- D4/D6 rank flag count: `24`;
- recommendation: standardized rerun.

Phase 65 then tested that route and found it insufficient:

- status: `standardization_not_helpful`;
- overall standardized-minus-unstandardized mean: `-0.071192921`;
- D4 standardized-minus-unstandardized mean: `-0.1435525318`;
- D4/B0 delta after standardization: `-0.3287026297`;
- D4/D6 delta after standardization: `-0.0983863509`;
- oracle gap fraction after standardization: `0.102231062`.

The base reward is important here. It is computed only from these explicit
planning columns:

- `explicit_feature_00`;
- `explicit_feature_01`;
- `explicit_feature_02`;
- `explicit_feature_04`;
- `explicit_feature_07`;
- `explicit_feature_09`;
- `explicit_feature_10`;
- `explicit_feature_13`;
- `explicit_feature_16`.

GeoFM-derived columns do not directly enter the reward. Therefore, a policy can
look strong under the current base reward by ranking explicit features well,
without using GeoFM information in a scientifically meaningful way. Phase 66
should quantify that issue before any new reward, B2/B3, transfer, or
manuscript route is reopened.

## Scope

Phase 66 should be read-only and reproducible from existing artifacts:

- Phase 63 comparison JSON, rollout CSV, oracle summary CSV, and training
  history CSV;
- Phase 64 overlap, oracle-rank gap, feature diagnostics, and failure-case
  outputs;
- Phase 65 comparison JSON, rollout CSV, pairwise delta CSV, and
  standardization stats JSON;
- Phase 63 contract metadata for loading raw tiled variant inputs;
- underlying B0, D4P8, D4P16, D6R8, and D6R16 tiled feature matrices.

The default audit should use the same full Phase 63/65 protocol:

- variants: `B0,D4P8,D4P16,D6R8,D6R16`;
- train tile: `tile_r003_c003`;
- eval tiles:
  `tile_r002_c003,tile_r005_c004,tile_r005_c003,tile_r000_c004,tile_r001_c004`;
- seeds: `0,1,2`;
- `eval_max_steps`: `8`.

## Non-Goals

Phase 66 should not:

- train or fine-tune any policy;
- change the base reward;
- enable suitability reward;
- add B2/B3, D2/D3, D5, or transfer experiments;
- modify the Phase 63 or Phase 65 rollout behavior;
- edit `paper/submission/final/*`;
- claim GeoFM advantage, PCA optimality, independent agronomic suitability, or
  submission readiness.

## Proposed Diagnostic Modules

### 1. Reward-Component Attribution

For every relevant variant/tile/seed row, decompose the base reward into its
component contributions for:

- oracle top-`eval_max_steps` blocks;
- Phase 63 unstandardized BC selected blocks;
- Phase 65 standardized-input BC selected blocks;
- missed oracle blocks;
- extra selected non-oracle blocks.

The output should show whether losses come from low-slope farmland/orchard,
current farmland/orchard, low-slope score, area score, slope penalties,
built-up penalty, water penalty, or combinations of those components.

This module should also report component deltas:

- selected minus oracle;
- missed oracle minus extra selected;
- Phase 65 selected minus Phase 63 selected;
- D4 selected minus B0 selected on matched tile-seed pairs;
- D4 selected minus paired D6 selected on matched tile-seed pairs.

### 2. Selected-Block Atlas

Build a compact atlas of action-level overlap and reward-equivalent
substitution patterns. For each matched row, report:

- selected block IDs for oracle, Phase 63, and Phase 65;
- overlap count and Jaccard overlap between policy and oracle;
- overlap between B0, D4, and D6 selected sets;
- reward rank of each selected block;
- whether non-overlap blocks are reward-equivalent within a small tolerance;
- whether a policy missed high-reward blocks or simply selected alternative
  high-reward blocks.

This module should prevent over-interpreting low selected-block overlap when
reward totals are close, and it should identify cases where D4/D6 failures are
true rank failures rather than harmless substitutions.

### 3. Representation-Rank Alignment

For each variant and tile, quantify how well representation columns align with
raw base-reward ranking and explicit reward components without training a
policy.

Suggested metrics:

- Spearman correlation between each feature dimension and per-block raw reward;
- rank-AUC or top-k enrichment for predicting oracle top-`eval_max_steps`
  blocks from individual dimensions;
- linear ridge or ordinary least-squares proxy fit from representation columns
  to raw reward, evaluated within tile as a diagnostic only;
- comparison of explicit-only B0 features versus D4/D6 extra representation
  columns;
- effective rank and variance concentration carried forward from Phase 64 where
  useful.

This module should answer whether D4/D6 have independent ranking signal under
the current base reward, or whether their apparent information is redundant
with the explicit planning columns already available in B0.

### 4. Failure-Mode Classifier

Classify each variant/tile/seed row into interpretable failure modes:

- `near_oracle_reward_equivalent`: low block overlap but small reward loss;
- `misses_explicit_reward_components`: missed blocks have better explicit
  reward components than selected extras;
- `representation_not_aligned_with_base_reward`: D4/D6 dimensions do not rank
  oracle blocks better than explicit features;
- `standardization_hurts_rank_geometry`: Phase 65 worsens rank/gap despite
  improved scale conditioning;
- `tile_specific_instability`: failures concentrate in specific eval tiles;
- `seed_instability`: failures vary mainly by seed.

The classifier should produce counts and representative cases rather than a
black-box label.

### 5. Diagnostic Gate

Phase 66 should reduce the audit into one conservative status:

- `representation_adds_reward_ranking_signal`: D4/D6 extra representation
  columns show stronger alignment with raw base-reward ranking than B0 explicit
  features alone, and failures are mostly optimization or action-selection
  issues rather than absence of ranking signal.
- `representation_signal_redundant_with_explicit_reward`: D4/D6 alignment is no
  stronger than B0 explicit-feature alignment, and selected-block differences
  mostly track the same explicit reward components.
- `base_reward_target_masks_geofm_signal`: base reward is dominated by explicit
  planning components, D4/D6 do not improve ranking under that target, and the
  available suitability-reward gates remain too weak to justify changing the
  reward yet.
- `insufficient`: required artifacts are missing, coverage is incomplete, or
  diagnostics conflict without supporting a single next experiment.

The status should include numeric evidence, not only a label.

## Artifacts

Suggested implementation files for the later implementation plan:

- `src/paper11_geofm/phase66_reward_label_representation_audit.py`
- `experiments/phase66_reward_label_representation_audit/run_phase66_reward_label_representation_audit.py`
- `tests/test_phase66_reward_label_representation_audit.py`
- `paper/phase28_results/32_phase66_reward_label_representation_audit.md`

Generated artifacts should remain under ignored experiment output directories:

- `phase66_reward_component_attribution.csv`;
- `phase66_selected_block_atlas.csv`;
- `phase66_representation_rank_alignment.csv`;
- `phase66_failure_mode_summary.csv`;
- `phase66_reward_label_representation_audit.json`;
- `phase66_reward_label_representation_audit.md`.

## Error Handling

The implementation should fail clearly when:

- Phase 63, Phase 64, or Phase 65 required artifacts are missing;
- the Phase 63 contract metadata is absent or incomplete;
- selected block IDs cannot be matched to loaded tiled inputs;
- a requested variant/tile/seed row is missing or duplicated;
- required base-reward columns are absent from a feature matrix;
- D4/D6 representation columns cannot be separated from explicit planning
  columns.

Missing optional Phase 64 or Phase 65 diagnostic rows may downgrade status to
`insufficient`, but should not silently produce partial positive claims.

## Testing Requirements

Unit tests should cover:

- reward-component decomposition matches `compute_base_planning_reward`;
- selected/missed/extra block attribution on a tiny synthetic tile;
- reward-equivalent substitution classification;
- Spearman or rank-alignment calculations with ties and constant columns;
- top-k enrichment on a synthetic representation matrix;
- failure-mode classifier outputs for controlled cases;
- diagnostic gate statuses for all four status labels;
- writer outputs JSON, CSV, and Markdown artifacts;
- CLI parser accepts Phase 63, Phase 64, Phase 65 artifact paths.

Verification for the later implementation should include:

- targeted Phase 66 unit tests;
- Phase 65, Phase 64, and Phase 63 regression tests;
- `D:\adk\.venv\Scripts\python.exe scripts\smoke_check.py`;
- `git diff --check`;
- a check that `paper/submission/final/*` remains unchanged.

## Claim Boundary

Phase 66 may report whether the current base-reward target is aligned with,
redundant with, or masking GeoFM-derived representation signal under the
existing Bishan set-policy protocol. It may not claim that GeoFM improves
farmland suitability planning, that PCA is optimal, that suitability reward is
ready, that B2/B3 or transfer behavior is known, or that the current manuscript
is ready for submission.

The manuscript remains downstream of algorithm and experiment evidence.
