# Phase 23 Multi-Seed B0/B1 Training Design

## Goal

Add a bounded multi-seed B0/B1 MaskablePPO training evaluation pilot that
responds to the IJAEOG reviewer concern that the current learned-policy evidence
is single-seed and mostly interface-readiness evidence.

## Problem

Phase 20 shows that B0/B1 MaskablePPO can train and evaluate on one real tile,
but only for one seed. Phase 21 and Phase 22 broaden cross-tile evaluation with
a ridge-linear per-block scorer, but they do not run PPO training and therefore
cannot support a learned-policy performance claim. The next safest experiment
is to repeat the Phase 20 same-tile training protocol across several seeds and
aggregate B0/B1 learned-policy and baseline diagnostics.

Phase 23 remains a bounded pilot. It does not solve the variable tile-size
policy blocker, does not enable suitability reward, does not run B2/B3, and
does not support transfer or final GeoFM-superiority claims.

## Inputs

Phase 23 consumes:

- Phase 2 real feature output directory;
- Phase 13 tile index CSV;
- variants, restricted to B0/B1;
- train tile ID, defaulting to the largest tile;
- integer seeds, defaulting to `0,1,2`;
- total MaskablePPO timesteps;
- evaluation max steps;
- output directory.

## Protocol

For each requested seed:

1. Run the existing Phase 20 bounded same-tile training protocol for B0/B1 on
   the same selected train tile.
2. Collect trained-policy, first-valid, and seeded-random summary rows.
3. Add `phase23_seed_rank` and preserve the original seed.
4. Aggregate learned-policy B0/B1 comparison diagnostics across seeds.
5. Write summary, traces, and an aggregate comparison report.

The selected train and evaluation tile are intentionally the same because the
current flat observation/action spaces remain tile-size-specific.

## Outputs

Phase 23 writes:

- `phase23_multi_seed_training_summary.csv`;
- `phase23_multi_seed_training_traces.json`;
- `phase23_multi_seed_training_comparison.json`.

Each summary row includes the Phase 20 fields plus `phase23_seed_rank`.

The comparison JSON records:

- seed count;
- variants;
- policies;
- summary row count;
- learned-policy mean reward by variant;
- B1 minus B0 learned-policy reward difference;
- baseline mean rewards by variant and policy;
- claim boundary and remaining evidence gaps.

## Claim Boundary

Phase 23 is a bounded multi-seed same-tile B0/B1 MaskablePPO training pilot
under the deterministic base planning reward. It strengthens learned-policy
execution evidence relative to Phase 20, but it does not prove GeoFM
superiority, suitability-reward benefit, cross-region transfer, or
submission-level planning performance.

## Real Bishan Expected Result

For current real Bishan artifacts, the default run should:

- select `tile_r003_c003` as the largest train/evaluation tile;
- run B0 and B1 across seeds `0,1,2`;
- train each bounded MaskablePPO pilot for the requested short timestep budget;
- write 18 summary rows for two variants, three seeds, and three policies;
- write an aggregate comparison JSON;
- keep the remaining evidence gaps explicit.

Numerical rewards are still pilot diagnostics until longer training budgets,
cross-tile or variable-size learned-policy evaluation, suitability-reward
validation, and held-out-region transfer experiments are complete.

## Implementation Units

- `src/paper11_geofm/multi_seed_training.py`: Phase 23 contract, seed loop,
  Phase 20 reuse, artifact writing, and aggregate comparison.
- `experiments/phase23_multi_seed_training/run_phase23_multi_seed_training.py`:
  CLI runner.
- `tests/test_phase23_multi_seed_training.py`: tests for seed normalization,
  B0/B1 restriction, aggregation shape, writer, and CLI output.
- Documentation updates in `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`,
  `reproducibility/FILE_MANIFEST.tsv`, and submission-readiness materials.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: Phase 23 extends Phase 20 and does not claim cross-tile
  learned-policy evaluation.
- Scope check: this phase is limited to B0/B1 multi-seed same-tile training
  pilots.
- Ambiguity check: defaults, outputs, aggregation fields, and claim boundary
  are explicit.
