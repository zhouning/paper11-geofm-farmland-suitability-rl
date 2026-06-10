# Phase 16 Tiled Baseline Protocol Design

## Goal

Build a non-learning tiled baseline protocol over the real Phase 13 tile index. Phase 16 runs short, deterministic masked rollouts for simple policies on each real tile using a representation-only variant.

## Problem

Phase 15 verifies that every real tile can run one input-contract step. That is still weaker than a reproducible tiled baseline protocol: reviewers need to see that the tiled environment contract can support deterministic multi-step action selection across all real tiles without training a policy or enabling suitability reward.

## Inputs

Phase 16 consumes:

- Phase 2 output directory with ready variant CSVs;
- Phase 13 `phase13_tile_index.csv`;
- variant ID, default `B1`;
- policy IDs, default `first_valid,seeded_random`;
- maximum steps per tile, default `4`;
- seed, default `0`;
- optional maximum number of tiles for quick checks.

## Outputs

The runner writes:

- `phase16_tiled_baseline_summary.csv`;
- `phase16_tiled_baseline_traces.json`.

The summary CSV contains one row per policy and tile:

- `policy_id`;
- `variant_id`;
- `tile_id`;
- `seed`;
- `n_blocks`;
- `n_features`;
- `observation_shape`;
- `action_space_n`;
- `max_steps`;
- `episode_steps`;
- `terminated`;
- `truncated`;
- `valid_action_rate`;
- `total_contract_reward`;
- `selected_block_ids`;
- `claim_boundary`.

The JSON trace contains the same summaries plus per-step action, selected block, reward, valid-action counts, and termination flags.

## Behavior

Phase 16 loads the requested Phase 2 variant once, rejects suitability-reward variants by default, reads all tile rows from the Phase 13 tile index, then runs each requested non-learning policy on each tile.

Supported policies:

- `first_valid`: always selects the first currently valid action;
- `seeded_random`: selects uniformly from valid actions using a deterministic seed derived from the base seed, policy, variant, and tile ID.

The default `max_steps=4` prevents the protocol from being mistaken for full planning evaluation. It checks multi-step contract behavior, mask updates, and reproducibility only.

## Claim Boundary

Phase 16 is a tiled non-learning baseline protocol. It does not train, tune, evaluate, or compare a DRL policy. It does not enable suitability reward. It does not report planning performance.

## Implementation Units

- `src/paper11_geofm/tiled_baseline_protocol.py`: tile reader, deterministic policy selection, rollout runner, artifact writer.
- `experiments/phase16_tiled_baseline_protocol/run_phase16_tiled_baselines.py`: CLI runner.
- `tests/test_phase16_tiled_baseline_protocol.py`: tests for all-tile summaries, seed reproducibility, max-tile cap, reward-variant rejection, writer output, and CLI output.
- Documentation updates in `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`, and `reproducibility/FILE_MANIFEST.tsv`.

## Real Bishan Expected Result

For the current real Bishan Phase 13 tile index with B1, policies `first_valid,seeded_random`, seed `0`, and `max_steps=4`:

- tiles processed: `54`;
- policies processed: `2`;
- summary rows: `108`;
- total blocks represented: `64984`;
- maximum observation shape: `180957`;
- every summary row has `episode_steps = 4`;
- every `total_contract_reward` is `0.0` because B1 uses `base_planning_reward`.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: Phase 16 reuses Phase 13 tile metadata and Phase 4 action-mask environment behavior.
- Scope check: this phase is limited to non-learning baseline protocol artifacts and does not train.
- Ambiguity check: default variant, policies, max steps, outputs, and claim boundary are explicit.
