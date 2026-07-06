# Phase 41 GeoFM Suitability Prior Design

## Purpose

Phase 41 tests a revised, stricter hypothesis for making GeoFM useful in
Paper11:

```text
GeoFM should not be injected as raw 64-dimensional policy state. It should be
used only if an independent label can calibrate it into a low-dimensional,
control-robust suitability prior that is safe to pass to later B2/B3 reward or
action-prior experiments.
```

Phase 41 is not a PPO, B2, or B3 experiment. It is an admission gate between
the Phase 40 independent-label gate and any later reward modification.

## Background

The completed Paper11 evidence rejects the original positive B1 framing under
the tested protocol:

- Phase 26 reports a B1-B0 learned-policy mean delta of `-0.1318712688`, with
  only `3 / 9` held-out tile-seed pairs favoring B1.
- Phase 28 reports `compression_matches_raw`; D4P8 and D4P16 exceed raw B1 at
  4096 steps.
- Phase 33 reports `budget_not_explanatory`; modestly higher training budget
  does not rescue normalized B1.
- Phase 40 reports `independent_label_inputs_missing` for the current no-label
  real run.

These results do not prove that GeoFM information is universally useless. They
show that raw GeoFM concatenation is not a defensible route for the present
Paper11 design. Phase 41 therefore changes the role of GeoFM from direct policy
state to an independently calibrated suitability prior.

## Revised Hypothesis

The old hypothesis was:

```text
Raw GeoFM-enhanced B1 state improves learned farmland layout policy performance.
```

The Phase 41 hypothesis is:

```text
GeoFM can support farmland layout optimization only if it first predicts an
independent, non-leakage suitability label under spatial holdout and
representation-control tests. If that gate passes, the calibrated GeoFM prior
may be passed to later bounded B2/B3 experiments.
```

## Scope

### In Scope

- Reuse the Phase 40 label registry contract and gate result.
- Build a Phase 41 readout pipeline for independent suitability labels.
- Compare explicit-only, GeoFM-only, explicit-plus-GeoFM, compressed-GeoFM, and
  shuffled-control feature families.
- Use spatially aware held-out splits so block-level leakage does not inflate
  label-readout metrics.
- Calibrate the accepted GeoFM readout into a low-dimensional suitability prior.
- Produce reviewer-facing CSV, JSON, and Markdown artifacts.
- Write a per-block prior file only when the Phase 41 gate passes.
- Add focused tests for pass, blocked, missing-label, control-failure, and
  artifact-writing paths.
- Update submission and reproducibility documentation without claiming B2/B3
  readiness.

### Out of Scope

- Creating, scraping, or downloading external labels.
- Treating DLTB-derived, slope-derived, source-metadata, or GeoFM-derived
  labels as independent labels.
- Running PPO or changing the planning reward.
- Claiming that suitability reward is validated.
- Claiming cross-region transfer.
- Reframing the current conclusion manuscript as a positive GeoFM-performance
  manuscript before Phase 41 and later B2/B3 gates pass.

## Inputs

### Required

- Phase 2 real feature table:

```text
experiments/phase11_bishan_dltb_real/outputs/phase2_real/block_geofm_features.csv
```

- Independent-label registry accepted by Phase 40.

### Optional

- CLI thresholds for minimum metric deltas, calibration quality, and spatial
  split coverage.
- Output run name.
- Feature-family selection for ablations.

If no independent label registry is supplied, Phase 41 must stop with
`phase41_independent_label_inputs_missing`. It must not fabricate labels or
reuse weak DLTB/slope labels as substitutes.

## Feature Families

Phase 41 evaluates these families against each accepted independent label:

| Family | Definition | Role |
|---|---|---|
| `explicit_only` | Existing explicit planning features only. | Baseline. |
| `geofm_raw_only` | Raw 64-dimensional GeoFM embedding only. | Tests GeoFM signal without explicit leakage. |
| `geofm_pca_only` | PCA-compressed and standardized GeoFM embedding. | Tests whether compressed geometry is more useful than raw embedding. |
| `explicit_plus_geofm_pca` | Explicit features plus compressed GeoFM prior features. | Candidate readout family. |
| `geofm_shuffled_control` | GeoFM rows shuffled within the training/evaluation contract. | Control for accidental split or target leakage. |
| `geofm_random_control` | Random features with matched dimensionality. | Control for optimization and dimensionality effects. |

Raw 64-dimensional GeoFM should not be exported as the final prior. It is an
ablation input only.

## Readout Models

Use lightweight, reproducible scikit-learn models already compatible with the
repository:

- standardized penalized logistic regression for binary labels;
- optional histogram gradient boosting for non-linear sensitivity checks;
- isotonic or Platt calibration only on training-fold validation data.

The default reported model should be the simpler calibrated logistic readout.
The non-linear model may support robustness interpretation, but it should not
be the only passing evidence.

## Spatial Evaluation Contract

Phase 41 must avoid random row-level splits as the primary evidence. Preferred
split order:

1. existing tile or split column if available;
2. block-derived spatial tile groups if available;
3. deterministic spatial group assignment from coordinates if tile metadata
   exists;
4. blocked fallback split with a clear artifact warning.

Each accepted label must report:

- valid label count;
- positive rate;
- number of spatial folds or held-out groups;
- train/evaluation coverage;
- metric mean and per-fold values.

## Metrics

For each feature family and label, report:

- ROC AUC;
- average precision;
- balanced accuracy at a training-selected threshold;
- Brier score;
- expected calibration error if implemented without new dependencies;
- positive-fold count for GeoFM improvement over baseline and controls.

Metric deltas must always name the comparator. Bare GeoFM scores are not
claimable evidence.

## Gate Rules

### Label Admission

Phase 41 may evaluate a label only when Phase 40 classifies it as
`label_gate_passed`.

Labels classified as diagnostic-only, blocked, missing, or unknown remain
excluded from Phase 41 passing evidence.

### GeoFM Prior Support

For a label to support a GeoFM prior, all must be true:

- the label passes Phase 40;
- `geofm_pca_only` or `explicit_plus_geofm_pca` improves over `explicit_only`
  by at least `min_auc_delta` or `min_ap_delta`;
- the improvement is positive in at least `min_positive_fold_fraction` of
  spatial folds;
- shuffled and random controls do not meet the same passing rule;
- calibrated Brier score is not worse than `explicit_only` by more than
  `max_brier_regression`;
- the positive label definition, provenance, and allowed evaluation role are
  recorded in the artifact.

Suggested defaults:

| Parameter | Default |
|---|---:|
| `min_auc_delta` | `0.03` |
| `min_ap_delta` | `0.03` |
| `min_positive_fold_fraction` | `0.67` |
| `max_brier_regression` | `0.02` |
| `n_pca_components` | `8` |

### Gate-Level Status

`geofm_suitability_prior_supported` when at least one independent label passes
the GeoFM prior support rules.

`geofm_suitability_prior_not_supported` when independent labels are available
but GeoFM does not clear baseline, control, or calibration thresholds.

`geofm_suitability_prior_control_failed` when GeoFM appears to improve but a
shuffled or random control also passes.

`phase41_independent_label_inputs_missing` when no Phase 40-passed label is
available.

## Outputs

Write artifacts under:

```text
experiments/phase41_geofm_suitability_prior/outputs/<run_name>
```

Expected files:

- `phase41_geofm_prior_summary.csv`
- `phase41_geofm_prior_metrics.csv`
- `phase41_geofm_prior.json`
- `phase41_geofm_prior.md`

Only when the gate status is `geofm_suitability_prior_supported`, also write:

- `block_geofm_suitability_prior.csv`

The prior file should contain:

- stable block identifier;
- accepted label name;
- calibrated suitability prior;
- calibration uncertainty or fold-derived confidence where available;
- feature family used to generate the prior;
- model and threshold metadata.

## Code Architecture

Add:

```text
src/paper11_geofm/phase41_geofm_suitability_prior.py
experiments/phase41_geofm_suitability_prior/run_phase41_geofm_suitability_prior.py
tests/test_phase41_geofm_suitability_prior.py
paper/phase28_results/15_phase41_geofm_suitability_prior.md
```

The module should expose small functions:

- `load_phase41_inputs(...)`
- `select_phase40_passed_labels(...)`
- `build_feature_families(...)`
- `make_spatial_splits(...)`
- `fit_readout_model(...)`
- `evaluate_feature_family(...)`
- `summarize_phase41_gate(...)`
- `build_block_prior(...)`
- `write_phase41_artifacts(...)`
- `run_phase41_geofm_suitability_prior(...)`

The CLI should remain a thin wrapper.

## Error Handling

- Missing feature table is a hard error.
- Missing or empty registry is not a hard error; it yields
  `phase41_independent_label_inputs_missing`.
- A registry with no Phase 40-passed labels yields
  `phase41_independent_label_inputs_missing`.
- Non-binary labels are blocked unless a deterministic positive definition is
  supplied in the registry.
- Missing explicit or GeoFM columns should produce a clear blocked status for
  the affected feature family.
- A failed control check must block the prior, not merely lower confidence.

## Tests

Add tests for:

1. no independent registry yields `phase41_independent_label_inputs_missing`;
2. registry with only DLTB/slope/source-derived labels yields
   `phase41_independent_label_inputs_missing`;
3. valid independent label with GeoFM signal and weak controls yields
   `geofm_suitability_prior_supported`;
4. valid independent label with no GeoFM improvement yields
   `geofm_suitability_prior_not_supported`;
5. shuffled control passing yields `geofm_suitability_prior_control_failed`;
6. PCA feature family is standardized and dimension-limited;
7. spatial splits are deterministic;
8. prior file is written only when the gate passes;
9. JSON, CSV, and Markdown artifacts are deterministic and include claim
   boundaries.

## Documentation Updates

Update:

- `README.md`
- `paper/phase28_results/README.md`
- `paper/submission/01_ijaeog_submission_readiness.md`
- `paper/submission/03_conclusion_manuscript_draft.md`
- `reproducibility/FILE_MANIFEST.tsv`
- `docs/superpowers/phase33_current_progress_handoff.md`

The documentation must preserve this boundary:

```text
Phase 41 can support a calibrated GeoFM suitability prior only after an
independent label passes Phase 40 and GeoFM clears baseline, representation
control, and calibration checks. It does not by itself validate B2/B3 reward or
prove planning-policy improvement.
```

## Success Criteria

Implementation is complete when:

- focused Phase 41 tests pass;
- smoke check still passes;
- a no-registry run produces `phase41_independent_label_inputs_missing`;
- a synthetic independent-label fixture can produce deterministic supported,
  not-supported, and control-failed statuses;
- artifacts clearly state whether a prior may be passed to future B2/B3 work;
- the manuscript remains a conclusion-level negative-results paper unless a
  real independent label and later B2/B3 experiments support a different
  conclusion.

## Next Step After Phase 41

If Phase 41 reports `geofm_suitability_prior_supported` on a real independent
label, the next phase should test a bounded low-dimensional prior interface:

- `B1P`: B0 plus calibrated GeoFM prior, with no raw 64-dimensional embedding;
- `B2S`: a small-weight suitability reward sweep using the calibrated prior;
- `B3G`: a gated action-prior experiment where GeoFM can influence ranking but
  cannot override planning constraints.

If Phase 41 remains blocked or not supported, Paper11 should keep the current
conclusion manuscript and should not restart B2/B3.
