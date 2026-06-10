# Phase 12 Real DLTB Scale Audit Design

## Goal

Build an executable audit gate for the real Bishan DLTB chain produced by Phase 11. The gate reads Phase 11, Phase 2, Phase 9, and Phase 10 JSON artifacts and reports which real-data experiments are currently defensible.

## Problem

Phase 11 proves that Bishan DLTB polygons can be converted into Phase 2-compatible feature tables. That is not the same as proving that a flat DRL environment is trainable at real scale, or that the suitability proxy can be used as a reward.

For the current Bishan run, Phase 2 creates 64,984 block rows. The flat Phase 4 observation size is `n_blocks * n_features + 3`, which reaches 5,328,691 for B3. Phase 10 also rejects the suitability reward because `low_slope_farmland_label` fails the alignment gate. A reviewer-facing repository needs a machine-checkable way to say:

- real DLTB feature tables are ready;
- short representation-only contract checks are allowed;
- suitability reward experiments are not allowed;
- flat full-scale MaskablePPO training is not ready and needs a tiled or hierarchical environment first.

## Inputs

Phase 12 consumes existing generated artifacts:

- Phase 11 adapter summary: `phase11_bishan_dltb_adapter_summary.json`;
- Phase 2 output directory containing `summary.json` and `experiment_variants.json`;
- Phase 9 proxy report: `phase9_proxy_validation_report.json`;
- Phase 10 reward-readiness gate: `phase10_reward_readiness_gate.json`.

The runner should not read the large real DLTB GeoPackage and should not read the full Phase 2 feature CSVs. It derives scale from summaries and manifests.

## Outputs

The runner writes:

- `phase12_real_dltb_scale_audit.json`;
- a concise console summary.

The JSON report includes:

- real DLTB row counts from Phase 11;
- Phase 2 block count and feature group readiness;
- per-variant row count, feature count, reward mode, observation dimension, estimated float32 observation size in MiB, and readiness flags;
- Phase 9 label interpretations;
- Phase 10 reward status and recommendation;
- decision flags:
  - `real_feature_tables_ready`;
  - `representation_only_smoke_allowed`;
  - `suitability_reward_allowed`;
  - `flat_full_scale_training_ready`;
  - `requires_tiled_or_hierarchical_env`;
- a recommendation string and claim boundary.

## Decision Rules

`real_feature_tables_ready` is true when Phase 11 exported at least one row, Phase 2 `n_blocks` matches the Phase 11 exported count, and all B0/B1/B2/B3 variants are ready with matching row counts.

`representation_only_smoke_allowed` is true when B0 and B1 are ready. These checks may inspect inputs or run short masked contract rollouts with explicit max-step caps. They must not claim policy performance.

`suitability_reward_allowed` is true only when Phase 10 status is `ready_for_suitability_reward` and the recommendation is not `do_not_enable_suitability_reward`.

`flat_full_scale_training_ready` is false when any of these conditions holds:

- suitability reward is not allowed;
- the maximum flat observation dimension exceeds a configurable threshold;
- any B0/B1/B2/B3 variant is not ready.

The default flat observation threshold is 1,000,000 float32 values. For the current Bishan B3 table, the expected maximum observation dimension is 5,328,691, so flat full-scale training is not ready.

`requires_tiled_or_hierarchical_env` is true when real feature tables are ready but flat full-scale training is not ready because the observation dimension exceeds the threshold.

## Claim Boundary

Phase 12 is an audit and gate over existing artifacts. It does not train, tune, evaluate, or compare a DRL policy. It does not convert weak-label alignment into agronomic validity. It does not override the Phase 10 reward gate.

## Implementation Units

- `src/paper11_geofm/real_scale_audit.py`: pure audit builder and JSON writer.
- `experiments/phase12_real_scale_audit/run_phase12_real_scale_audit.py`: CLI wrapper.
- `tests/test_phase12_real_scale_audit.py`: synthetic artifact tests for decisions, writer output, threshold behavior, and CLI output.
- Documentation updates in `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`, and `reproducibility/FILE_MANIFEST.tsv`.

## Real Bishan Expected Result

For the current local Bishan run:

- `real_feature_tables_ready = true`;
- `representation_only_smoke_allowed = true`;
- `suitability_reward_allowed = false`;
- `flat_full_scale_training_ready = false`;
- `requires_tiled_or_hierarchical_env = true`;
- recommendation: keep representation-only real-data analysis, keep suitability reward disabled, and design a tiled or hierarchical environment before any full-scale DRL training claim.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: the decision rules use only artifact fields already emitted by Phases 11, 2, 9, and 10.
- Scope check: the phase is limited to audit/gating and does not add training.
- Ambiguity check: reward and flat-training decisions have explicit Boolean rules.
