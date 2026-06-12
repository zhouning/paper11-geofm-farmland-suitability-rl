# Phase 25 Padded Held-Out Policy Design

## Goal

Create the first learned-policy held-out tile experiment for Paper11 by replacing
the current tile-size-specific flat observation protocol with a padded
variable-size MaskablePPO environment.

## Motivation

Phase 20 and Phase 23 can train B0/B1 MaskablePPO policies only on the same
tile because the flat observation and action spaces depend on tile block count.
Phase 21 and Phase 22 verify a cross-tile per-block scorer interface, but that
interface is not PPO training. Phase 25 should close this evidence gap by
training a learned policy on one or more tiles and evaluating it on distinct
held-out tiles under a shared padded observation and action contract.

## Claim Target

Phase 25 may support this bounded claim if the results are positive:

```text
Under a deterministic base planning reward and a padded variable-size tile
contract, the GeoFM-enhanced B1 representation can be evaluated as a learned
policy on held-out Bishan tiles and compared against the explicit-feature B0
baseline.
```

Phase 25 must not claim suitability-reward benefit, B2/B3 superiority,
cross-region transfer beyond held-out Bishan tiles, or final submission-level
planning performance.

## Experiment Scope

Phase 25 is restricted to:

- variants: `B0` and `B1`;
- reward: deterministic `base_planning_reward`;
- train/evaluation unit: Phase 13 real Bishan tiles;
- policy: `sb3-contrib` MaskablePPO with `MlpPolicy`;
- observation contract: fixed padded block-feature matrix plus mask/global
  features;
- action contract: fixed `Discrete(max_blocks)` with invalid padded and
  already-selected blocks masked out;
- baselines: `first_valid` and `seeded_random`;
- statistics: per-seed and aggregate B1-B0 learned-policy reward delta,
  per-tile reward, action validity, selected block IDs, and runtime metadata.

## Padded Environment Contract

For each variant, Phase 25 selects a train/evaluation tile set and computes:

```text
max_blocks = max(n_blocks across selected train and evaluation tiles)
n_features = feature count for the variant
```

The observation is a flat vector:

```text
padded_state_matrix.reshape(-1)
+ selected_mask
+ valid_block_mask
+ global_features
```

Where:

- `padded_state_matrix` has shape `(max_blocks, n_features)`;
- rows after the tile's real block count are zeros;
- `selected_mask` has length `max_blocks`;
- `valid_block_mask` has length `max_blocks` and is false for padded rows;
- `global_features` includes budget remaining, step fraction, valid-action
  fraction, real block fraction, and real block count normalized by
  `max_blocks`.

The action space is:

```text
Discrete(max_blocks)
```

The action mask is true only for real, unselected block rows.

## Tile Selection

Default real Bishan pilot:

- train tile: largest tile, currently `tile_r003_c003`;
- held-out evaluation tiles: largest distinct tiles, default count `3`;
- seeds: `0,1,2`;
- training budget for local smoke: `32` to `128` timesteps;
- training budget for main Colab run: `512` to `4096` timesteps, adjusted after
  a timing probe.

The runner should also accept explicit train/evaluation tile IDs so later
experiments can test other spatial splits without code changes.

## Outputs

Phase 25 writes:

- `phase25_padded_heldout_policy_summary.csv`;
- `phase25_padded_heldout_policy_traces.json`;
- `phase25_padded_heldout_policy_comparison.json`.

The comparison JSON should include:

- train tile IDs and held-out tile IDs;
- variants, seeds, training timesteps, evaluation max steps, and `max_blocks`;
- mean reward by row type, variant, and evaluation tile;
- B1-B0 learned-policy mean reward delta;
- B1-B0 held-out tile delta by tile;
- baseline deltas for `first_valid` and `seeded_random`;
- claim boundary and remaining evidence gaps.

## Windows Hardware Assessment

Checked on 2026-06-12 in the local Windows workspace:

- OS: Windows 11 Home China, 64-bit;
- CPU: Intel Core Ultra 9 185H, 16 cores, 22 logical processors;
- RAM: 31.43 GiB total, about 14.32 GiB available during the check;
- workspace disk: `D:` has about 88.82 GiB free;
- GPU: Intel Arc Graphics; no NVIDIA CUDA device detected;
- Python: `C:\Python314\python.exe`;
- RL stack: `gymnasium 1.2.3`, `stable-baselines3 2.7.1`,
  `sb3-contrib 2.7.1`, `torch 2.10.0+xpu`;
- observed import behavior: direct `torch` import took about 83 seconds once,
  while `stable_baselines3` and `sb3_contrib` imports took about 6 seconds in
  later checks.

This machine is suitable for development, unit tests, smoke checks, and short
Phase 25 pilot runs. It is not ideal for the main long-budget multi-seed
training run because there is no CUDA GPU, the system Python is 3.14 rather than
a project-local virtual environment, and the Torch XPU stack has high import
latency. The main result run should be prepared so it can execute on Google
Colab Pro+ with a CUDA runtime.

## Recommended Compute Split

Use Windows for:

- TDD and implementation;
- tiny synthetic tests;
- real Bishan smoke runs with `total_timesteps <= 128`;
- artifact/schema validation.

Use Google Colab Pro+ for:

- main Phase 25 multi-seed runs;
- longer training budgets;
- timing and stability probes;
- regenerated final artifacts for manuscript tables.

Use macOS only if it has substantially more RAM or a working MPS PyTorch stack;
otherwise it is a secondary CPU development option rather than the main
training platform.

## Success Criteria

Phase 25 is successful when:

1. A padded environment can reset, mask actions, step, and reject invalid padded
   or already-selected actions.
2. MaskablePPO can train on the padded B0/B1 environment and evaluate on
   distinct held-out tiles without observation/action shape errors.
3. The runner writes summary, trace, and comparison artifacts.
4. A real Bishan smoke run completes on Windows.
5. The comparison artifact is explicit about whether B1 improves, matches, or
   underperforms B0 under the pilot budget.

## Evidence Boundaries

Phase 25 can strengthen the learned-policy evidence from same-tile to held-out
tile evaluation within Bishan. It still cannot support:

- suitability-reward claims;
- B2/B3 full-model claims;
- cross-region transfer beyond Bishan tiles;
- final IJAEOG submission claims without longer runs, ablations, spatial maps,
  and calibrated uncertainty.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: B2/B3 and suitability reward are excluded because Phase 10
  currently blocks suitability reward.
- Scope check: Phase 25 is limited to B0/B1 padded held-out policy evidence.
- Ambiguity check: environment contract, default tile selection, outputs,
  compute split, and claim boundary are explicit.
