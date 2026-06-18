# Phase 28 Representation-Control Evaluation Design

## Goal

Add a bounded Phase 28 evaluation package that tests whether the current B1
GeoFM representation behaves differently from random, shuffled, and
PCA-compressed representation controls under the same padded held-out policy
protocol used by Phase 25 and analyzed by Phase 26.

Phase 28 should answer one diagnostic question:

```text
When B1 is compared against D2 random-64, D3 shuffled-GeoFM, and D4
PCA-compressed GeoFM controls under the same base-reward held-out-tile policy
protocol, is there evidence that the raw GeoFM embedding contributes signal
beyond dimensionality, spatial misalignment, or low-rank compression effects?
```

## Motivation

Phase 26 and Phase 27 show that the current B1-over-B0 learned-policy evidence
is negative and unstable:

- 1024 steps: B1-B0 learned-policy mean delta `-0.4329022862`, positive
  tile-seed count `4 / 9`, claim status `not_supported`;
- 4096 steps: B1-B0 learned-policy mean delta `-0.1318712688`, positive
  tile-seed count `3 / 9`, claim status `not_supported`;
- Phase 27 status: `budget_not_explanatory`.

This blocks any positive claim that GeoFM improves planning decisions. The next
necessary experiment is therefore not a reward extension or transfer claim. It
is a representation-control evaluation that separates raw GeoFM signal from
extra input capacity, spatial alignment artifacts, and compressed alternatives.

## Claim Target

Phase 28 may support only this bounded diagnosis:

```text
The current B1 representation is distinguishable or not distinguishable from
random, shuffled, and PCA-compressed controls under the same Bishan padded
held-out base-reward protocol.
```

Phase 28 must not claim:

- that GeoFM improves planning decisions unless B1 also beats B0 and the
  controls under stable tile-seed evidence;
- that suitability reward is ready;
- that B2 or B3 improves outcomes;
- cross-region transfer;
- final submission readiness.

## Recommended Approach

Use a small generalization of the existing padded held-out evaluator rather
than duplicating policy-training logic.

The current `run_phase25_padded_heldout_policy()` path already provides the
right environment, masked action protocol, baseline policies, CSV writer, trace
writer, and comparison skeleton. Its current `_normalize_variants()` restricts
the API to `B0` and `B1`. Phase 28 should introduce a Phase 28-specific
evaluation entry point that reuses the same environment and internal
evaluation helpers while accepting base-reward variants:

```text
B0, B1, D2, D3, D4P8, D4P16
```

The Phase 25 public boundary should remain historically accurate as a B0/B1
pilot. Phase 28 should have its own claim boundary, runner, outputs, and docs.
The implementation may share lower-level helpers with Phase 25, but public
Phase 28 artifacts must be clearly labeled as representation-control
diagnostics.

## Inputs

Phase 28 consumes:

1. a Phase 2-ready output directory containing `B0` and `B1`;
2. a Phase 8 output directory containing `D2`, `D3`, `D4P8`, and `D4P16`;
3. the Phase 13/14 tile index CSV used by the padded held-out protocol;
4. explicit train/evaluation tile settings or the existing largest-train,
   largest-distinct-eval tile selection;
5. explicit seeds, total timesteps, and evaluation max steps.

The runner should support two modes:

- `run-and-analyze`: run B0/B1/D controls and write Phase 28 analysis;
- `analyze-only`: analyze an existing Phase 28 summary CSV.

## Feature Source Contract

The evaluator must load each variant from the directory where that variant is
declared:

| Variant | Source |
|---|---|
| B0 | Phase 2 output directory |
| B1 | Phase 2 output directory |
| D2 | Phase 8 output directory |
| D3 | Phase 8 output directory |
| D4P8 | Phase 8 output directory |
| D4P16 | Phase 8 output directory |

All variants must use `base_planning_reward`. Any
`base_plus_suitability_reward` variant must be rejected. Phase 28 must verify
that the selected tile block IDs are present in each requested variant.

## Diagnostic Variants

| ID | State | Reward | Diagnostic role |
|---|---|---|---|
| B0 | explicit planning features | base reward | GIS-only baseline |
| B1 | explicit planning features + raw GeoFM 64d | base reward | target representation |
| D2 | explicit planning features + deterministic random 64d | base reward | dimensionality control |
| D3 | explicit planning features + shuffled GeoFM 64d | base reward | spatial-alignment control |
| D4P8 | explicit planning features + PCA-8 GeoFM | base reward | compact representation control |
| D4P16 | explicit planning features + PCA-16 GeoFM | base reward | wider compact representation control |

Default Phase 28 variants should be:

```text
B0,B1,D2,D3,D4P8,D4P16
```

The implementation should also allow a subset for smoke tests, as long as B1
and at least one comparator are present for analysis.

## Evaluation Protocol

For each requested variant and seed:

1. load the train tile as a padded `Phase25PaddedTileEnv`;
2. train a `MaskablePPO` policy with the requested total timesteps;
3. evaluate the deterministic learned policy on each held-out tile;
4. evaluate `first_valid` and `seeded_random` baselines on the same tiles;
5. record one summary row per policy, variant, held-out tile, and seed;
6. store learned-policy traces and baseline traces for audit.

The protocol should keep Phase 25's conservative settings unless the caller
explicitly supplies alternatives:

```text
device=cpu
n_steps=4
batch_size=4
n_epochs=1
gamma=0.99
```

## Analysis Metrics

Phase 28 should compute learned-policy diagnostics from the summary rows:

- mean reward by variant;
- B1 minus each comparator mean reward;
- per tile-seed B1 minus comparator delta;
- positive tile-seed count and fraction for each comparator;
- best comparator by learned-policy mean reward;
- whether B1 beats B0, D2, D3, D4P8, and D4P16;
- baseline policy summaries for audit only.

Primary comparator deltas:

```text
B1_minus_B0
B1_minus_D2
B1_minus_D3
B1_minus_D4P8
B1_minus_D4P16
```

## Diagnostic Status Rules

Use these statuses:

- `representation_signal_supported`: B1 has positive mean deltas against B0,
  D2, and D3, and B1 has positive tile-seed fractions of at least `0.6`
  against D2 and D3;
- `representation_signal_control_limited`: B1 beats D2 or D3 but does not beat
  B0, or positive tile-seed fractions are below `0.6`;
- `representation_signal_not_distinguishable`: B1 does not beat D2 and does
  not beat D3 on mean learned-policy reward;
- `compression_matches_raw`: D4P8 or D4P16 matches or beats B1 within a
  configured tolerance while B1 does not clearly beat both D2 and D3;
- `insufficient`: required rows, variants, tile-seed pairs, or comparators are
  missing.

The default tolerance for a D4 compression match should be `1e-9` unless an
explicit CLI argument overrides it.

## Outputs

Write Phase 28 artifacts under a requested output directory:

```text
phase28_representation_control_summary.csv
phase28_representation_control_traces.json
phase28_representation_control_comparison.json
phase28_tile_seed_delta_table.csv
phase28_control_readiness.md
```

### `phase28_representation_control_summary.csv`

Use the Phase 25 summary schema where possible, adding no columns unless
needed. Existing downstream readers should still be able to inspect:

- `row_type`;
- `variant_id`;
- `train_tile_id`;
- `eval_tile_id`;
- `seed`;
- `train_timesteps`;
- `eval_max_steps`;
- `n_features`;
- `total_contract_reward`;
- `selected_block_ids`;
- `claim_boundary`.

### `phase28_representation_control_comparison.json`

Include:

- source Phase 2 and Phase 8 directories;
- tile index CSV;
- requested variants and seeds;
- train/eval tile IDs;
- mean reward by policy and variant;
- learned-policy comparator deltas;
- tile-seed delta rows;
- positive counts and fractions;
- diagnostic status;
- claim boundary;
- remaining evidence gaps.

### `phase28_tile_seed_delta_table.csv`

One row per comparator, held-out tile, and seed:

- `comparator_variant_id`;
- `eval_tile_id`;
- `seed`;
- `b1_reward`;
- `comparator_reward`;
- `b1_minus_comparator_reward`;
- `b1_improves_comparator`;
- `train_timesteps`;
- `eval_max_steps`;
- `claim_boundary`.

### `phase28_control_readiness.md`

Write a short reviewer-facing note with:

- setup;
- variants evaluated;
- mean comparator deltas;
- tile-seed positive counts;
- diagnostic status;
- safe manuscript wording;
- unsafe manuscript wording;
- next recommended experiment.

## Claim Boundary

Use an explicit Phase 28 claim boundary:

```text
Phase 28 is a representation-control diagnostic over B0/B1/D2/D3/D4 base-reward
padded held-out Bishan policy runs; it does not enable suitability reward, does
not test B2/B3, does not test cross-region transfer, and does not support
submission-level planning-performance claims.
```

## Non-Goals

Do not:

- enable `suitability_proxy` as a reward term;
- run B2 or B3;
- add held-out regions outside the current tile-index contract;
- modify legacy county or parcel environments;
- claim AlphaEarth measures soil, fertility, irrigation, or farmer behavior;
- treat D2/D3/D4 controls as final scientific evidence without stable B1-vs-B0
  support.

## Public API

Create a new module:

```text
src/paper11_geofm/phase28_representation_controls.py
```

with:

```python
PHASE28_CLAIM_BOUNDARY = (
    "Phase 28 is a representation-control diagnostic over B0/B1/D2/D3/D4 "
    "base-reward padded held-out Bishan policy runs; it does not enable "
    "suitability reward, does not test B2/B3, does not test cross-region "
    "transfer, and does not support submission-level planning-performance "
    "claims."
)

def build_phase28_representation_control_contract(...):
    ...

def run_phase28_representation_control_evaluation(...):
    ...

def build_phase28_representation_control_analysis(...):
    ...

def write_phase28_representation_control_artifacts(...):
    ...
```

The implementation may refactor Phase 25 helper functions only where required
to avoid duplicating training and evaluation logic. Refactors must preserve
Phase 25 tests and public B0/B1 behavior.

## CLI

Create:

```text
experiments/phase28_representation_controls/run_phase28_representation_controls.py
```

The command accepts:

- `--mode`, either `run-and-analyze` or `analyze-only`;
- `--phase2-output-dir`;
- `--phase8-output-dir`;
- `--tile-index-csv`;
- `--output-dir`;
- `--existing-summary-csv` for analyze-only mode;
- `--variants`, default `B0,B1,D2,D3,D4P8,D4P16`;
- `--train-tile-id`;
- `--eval-tile-ids`;
- `--max-eval-tiles`;
- `--total-timesteps`;
- `--eval-max-steps`;
- `--seeds`;
- `--compression-match-tolerance`, default `1e-9`.

`run-and-analyze` should require explicit training settings, matching the
Phase 26 safety pattern. It should not silently launch a long run with defaults.

## Test Strategy

Use TDD. Add tests that:

1. build tiny Phase 2 B0/B1 fixtures and Phase 8 D-control fixtures;
2. verify the Phase 28 contract accepts B0/B1/D2/D3/D4P8/D4P16 and rejects
   unsupported or suitability-reward variants;
3. verify variant source routing loads B0/B1 from Phase 2 and D variants from
   Phase 8;
4. verify a small patched/fake training run produces summary rows for all
   requested variants, policies, tiles, and seeds;
5. verify comparator deltas are computed for B1 against B0, D2, D3, D4P8, and
   D4P16;
6. verify diagnostic status rules for supported, control-limited,
   not-distinguishable, compression-match, and insufficient cases;
7. verify writer outputs all CSV/JSON/Markdown artifacts;
8. verify CLI validation blocks missing explicit training settings in
   `run-and-analyze` mode;
9. verify Phase 25 and Phase 26 tests still pass after any helper refactor.

Tests must be offline, CPU-only, and use alternate pytest basetemp paths on
Windows if the repository default `.pytest_tmp` is locked.

## Documentation Updates

Update:

- `README.md`;
- `reproducibility/REPRODUCTION_GUIDE.md`;
- `reproducibility/FILE_MANIFEST.tsv`;
- `paper/phase26_results/02_next_experiment_matrix.md`;
- a new `paper/phase28_results/` package after real Phase 28 output exists.

Documentation must state that Phase 28 is diagnostic and that current Phase 26
and Phase 27 evidence still blocks a positive B1-over-B0 manuscript claim.

## Success Criteria

Phase 28 is successful when:

1. D2/D3/D4 controls can be evaluated under the same padded held-out policy
   protocol as B0/B1.
2. B1-vs-control deltas are reported by mean reward and by tile-seed pair.
3. The diagnostic status is conservative and reproducible.
4. Phase 25 B0/B1 behavior remains backward compatible.
5. Tests pass without GPU, internet, or long training.
6. Documentation keeps the claim boundary explicit.

## Evidence Boundaries

Even a favorable B1-vs-control result would only show representation-control
separation under the current Bishan base-reward pilot. It would not by itself
establish a final paper claim. A positive manuscript claim still requires
stable B1-vs-B0 evidence, validated suitability-reward evidence before B2/B3,
and held-out-region transfer evidence.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: Phase 28 evaluates only base-reward B0/B1/D controls and
  does not introduce B2/B3, suitability reward, or transfer.
- Scope check: the work is one implementation plan: contract, evaluator,
  analyzer, writer, CLI, tests, and documentation.
- Ambiguity check: inputs, variant routing, outputs, diagnostic status rules,
  tests, and claim boundaries are explicit.
