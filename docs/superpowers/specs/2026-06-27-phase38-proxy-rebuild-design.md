# Phase 38 Proxy-Rebuild Design

## Goal

Phase 38 rebuilds the Paper11 suitability proxy as an experiment-first
algorithm branch before any B2/B3 suitability-reward work. It should turn the
current Phase 36/37 negative boundary into a testable proxy-learning pipeline,
while keeping reward integration blocked unless the rebuilt proxy clears a
conservative validation gate.

## One-Sentence Argument

Because the current centroid-style `suitability_proxy` is not supported by
Phase 36 and Phase 37, Phase 38 first builds a supervised or semi-supervised
proxy-rebuild framework using leakage-aware labels and spatial held-out
validation, then decides whether a bounded B2/B3 reward smoke is justified.

## Scope

Phase 38 is an algorithm and diagnostic experiment. It may train lightweight
proxy models and write rebuilt proxy scores, but it does not run PPO, does not
alter the planning reward, does not enable B2/B3 by default, and does not
support final planning-performance claims.

The first implementation should use existing local labels to validate the
pipeline and leave the input contract open for stronger independent labels.
Local DLTB/slope-derived labels remain leakage-risk labels and are not treated
as agronomic ground truth.

## Inputs

Required inputs:

- Phase 2 real block feature table: `block_geofm_features.csv`.
- Phase 2 variant tables for B0/B1/B2/B3 when available.
- Phase 8 control feature tables for D2/D3/D4P8/D4P16 when available.
- Phase 30 normalized controls for N1Z/N1ZR when available.
- One or more label columns.

Initial local labels:

- `current_farmland_label`
- `farmland_or_orchard_label`
- `low_slope_farmland_label`

Future independent labels should use the same interface. Candidate sources
include high-standard farmland labels, irrigation or water-proximity proxies,
soil/yield/productivity proxies, retention or persistence labels, and external
agronomic suitability labels.

## Label Boundary

Every label must be classified before modeling:

- `explicit_label_leakage_risk`: labels derived from DLTB, slope, land-use
  class, or explicit planning features.
- `candidate_independent_proxy`: labels not directly encoded in explicit
  planning features but still proxy-level rather than ground truth.
- `independent_validation_label`: labels with defensible external provenance
  suitable for stronger proxy-readiness claims.

Phase 38 may use leakage-risk labels to debug the pipeline, compare models,
and inspect failure modes. It must not use leakage-risk labels alone to unlock
B2/B3 or suitability reward.

## Feature Families

Evaluate the same family structure as Phase 36 where inputs exist:

- `explicit_only`
- `raw_geofm_only`
- `explicit_plus_raw_geofm`
- `suitability_proxy_only`
- `explicit_plus_suitability_proxy`
- `explicit_plus_random_geofm`
- `explicit_plus_shuffled_geofm`
- `explicit_plus_pca8_geofm`
- `explicit_plus_pca16_geofm`
- `explicit_plus_normalized_geofm_zscore`
- `explicit_plus_normalized_geofm_zscore_row_l2`

Phase 38 should additionally write rebuilt proxy scores for selected model and
feature-family combinations. These scores are diagnostic artifacts until the
status gate supports a bounded reward smoke.

## Model Families

The first implementation should prioritize stable, auditable models:

- `logistic_elastic_net`: primary linear baseline with standard scaling and
  class balancing.
- `random_forest`: non-linear tabular control for feature interactions.
- `hist_gradient_boosting`: optional non-linear tabular model when available
  in the installed scikit-learn environment.

The model API must be extensible, but Phase 38 should not add neural models in
the first pass. A heavier model would be premature until the label boundary is
stronger.

## Validation Design

Use spatial held-out validation as the default:

- If a `split` column exists, use train rows for model fitting and test,
  validation, eval, or evaluation rows for held-out scoring.
- If no split column exists, use a deterministic block-id modulo split and
  mark the split source explicitly.
- Require positive and negative labels in both train and evaluation subsets.
- Report ROC AUC, average precision, balanced accuracy, accuracy, train/eval
  counts, positive rates, calibration bins, and top model diagnostics.

The evaluation must compare rebuilt GeoFM-derived models against explicit-only
and diagnostic controls. High explicit-only scores on leakage-risk labels are
expected and should lower claim strength, not raise it.

## Status Rule

Phase 38 should emit one conservative aggregate status:

- `proxy_rebuild_supported_for_bounded_b2_b3_smoke`: at least one
  GeoFM-derived model family improves over explicit-only and random/shuffled
  controls by the configured ROC AUC and average-precision margins on at least
  one `candidate_independent_proxy` or `independent_validation_label`.
- `proxy_rebuild_diagnostic_only`: models run successfully, but support comes
  only from leakage-risk labels or does not clear control margins.
- `proxy_rebuild_inputs_insufficient`: required features, labels, class
  variation, or split coverage are missing.

Even the supported status only permits a bounded B2/B3 reward smoke. It does
not permit final suitability, agronomic validity, cross-region transfer, or
planning-performance claims.

## Outputs

The runner should write:

- `phase38_label_summary.csv`
- `phase38_model_summary.csv`
- `phase38_rebuilt_proxy_scores.csv`
- `phase38_proxy_rebuild.json`
- `phase38_proxy_rebuild.md`

The JSON artifact should include source paths, label classifications, model
families, feature families, metric tables, status, interpretation, and claim
boundary.

## Implementation Shape

Add one focused module:

```text
src/paper11_geofm/phase38_proxy_rebuild.py
```

The module should expose:

- `build_phase38_proxy_rebuild(...)`
- `write_phase38_proxy_rebuild_artifacts(...)`

Add one runner:

```text
experiments/phase38_proxy_rebuild/run_phase38_proxy_rebuild.py
```

Add tests:

```text
tests/test_phase38_proxy_rebuild.py
```

The implementation should follow the Phase 36/37 pattern: pure in-memory
analysis builder, explicit artifact writer, CLI runner, deterministic fixtures,
and no reward or PPO side effects.

## Verification

The implementation must use test-first development. Tests should cover:

- label classification and leakage boundaries;
- spatial split handling and insufficient-label cases;
- linear and non-linear model evaluation over synthetic fixtures;
- status reduction for supported, diagnostic-only, and insufficient inputs;
- rebuilt proxy score writing;
- CLI execution against a synthetic fixture.

Focused verification after implementation:

```powershell
python -m pytest tests\test_phase38_proxy_rebuild.py tests\test_phase36_suitability_proxy_validation.py -q --basetemp=.pytest_tmp_phase38_final -p no:cacheprovider
python scripts\smoke_check.py
```

## Documentation Updates

After implementation and real-run verification, update:

- `README.md`
- `paper/phase28_results/README.md`
- a new `paper/phase28_results/12_phase38_proxy_rebuild.md`
- `reproducibility/FILE_MANIFEST.tsv`
- `docs/superpowers/phase33_current_progress_handoff.md`

The handoff must state whether Phase 38 remains diagnostic-only or supports a
bounded B2/B3 smoke, and it must preserve the Phase 36/37 boundary unless the
new status rule is satisfied with non-leakage labels.

## Claim Boundary

Phase 38 may support this guarded statement:

> A rebuilt suitability proxy can be evaluated for spatial held-out alignment
> with leakage-aware labels and diagnostic representation controls before
> suitability reward is considered.

Phase 38 may not support:

> GeoFM directly measures soil quality, irrigation, fertility, productivity,
> final planning quality, or cross-region transfer.
