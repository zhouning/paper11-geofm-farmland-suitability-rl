# Phase 39 Independent Label Audit Design

## Goal

Phase 39 creates an executable independent-label audit before any new B2/B3
suitability-reward or planning-performance experiment. It answers one narrow
question: do the current or user-supplied Paper11 block labels include a
defensible non-DLTB, non-slope, non-explicit-feature validation target that can
justify rerunning Phase 38 and eventually testing a bounded B2/B3 smoke?

## One-Sentence Argument

Because Phase 36 and Phase 38 show that all currently available real labels are
explicit leakage risks, Phase 39 should turn label provenance and readiness into
a reproducible gate rather than relying on manual judgment.

## Scope

Phase 39 is a data-readiness and evidence-boundary experiment. It scans label
columns, classifies their provenance, checks train/evaluation usability, writes
a reviewer-facing audit, and emits one conservative readiness status.

Phase 39 does not train PPO, does not alter reward logic, does not rebuild the
suitability proxy itself, does not run B2/B3, and does not prove agronomic
validity. A positive Phase 39 status can only authorize a Phase 38 rerun with
the accepted label set.

## Inputs

Required input:

- Phase 2 real block feature table:
  `experiments/phase11_bishan_dltb_real/outputs/phase2_real/block_geofm_features.csv`

Optional inputs:

- one or more external label CSV files keyed by `block_id`
- a label registry CSV or JSON describing candidate columns and provenance
- the existing Phase 38 diagnosis JSON, used only as prior context for known
  leakage labels

The first real Bishan run should work without optional external files. In that
case the expected result is that no independent labels are ready.

## Label Registry

Every candidate label must be assigned one provenance class:

- `explicit_label_leakage_risk`: derived from DLTB class, slope, explicit
  planning features, current farmland masks, orchard/farmland class merges, or
  any label already encoded by the explicit planning state.
- `source_field_leakage_risk`: derived from raw source fields such as
  `source_dlbm`, `source_dlmc`, `source_category`, or similar land-use source
  attributes that are not safe independent agronomic labels.
- `candidate_independent_proxy`: plausibly external or not directly encoded by
  explicit planning features, but still a proxy rather than ground truth.
- `independent_validation_label`: externally sourced, documented, and suitable
  for a stronger suitability-proxy validation claim.
- `unclassified`: present but lacking enough provenance information. This
  class must not unlock downstream experiments.

Built-in defaults should classify the current real Bishan labels as
`explicit_label_leakage_risk`:

- `current_farmland_label`
- `farmland_or_orchard_label`
- `low_slope_farmland_label`

Built-in defaults should also classify source land-use descriptor columns as
`source_field_leakage_risk`, not as labels:

- `source_bsm`
- `source_category`
- `source_dlbm`
- `source_dlmc`

External labels should require explicit registry metadata before they can be
classified as `candidate_independent_proxy` or `independent_validation_label`.

## Readiness Checks

For each candidate label, Phase 39 should report:

- column source: base Phase 2 table or external CSV path
- availability in the joined block table
- valid numeric/binary label count
- positive count, negative count, and positive rate
- train/evaluation counts using the existing `split` column when available
- class variation in both train and evaluation subsets
- provenance class
- whether a registry entry exists
- whether block joins dropped rows
- whether the label is allowed for Phase 38 rerun
- a short reason string for the decision

The first implementation should keep the label contract binary. Continuous
labels can be supported later only after an explicit thresholding rule is
specified and tested.

## Status Rule

Phase 39 should emit exactly one aggregate status:

- `independent_labels_ready_for_phase38_rerun`: at least one label is usable,
  has train/evaluation class variation, and is classified as
  `candidate_independent_proxy` or `independent_validation_label`.
- `candidate_proxy_labels_need_review`: at least one label is usable and
  plausibly non-explicit, but it is `unclassified` or lacks enough registry
  metadata to support a Phase 38 rerun.
- `independent_label_inputs_missing`: no usable non-leakage label exists.
- `independent_label_inputs_insufficient`: candidate labels exist but fail
  usability checks, such as missing joins, single-class labels, or missing
  evaluation split coverage.

For the current real Bishan table without external labels, the expected status
is `independent_label_inputs_missing`.

## Outputs

The runner should write:

- `phase39_label_inventory.csv`
- `phase39_label_readiness.csv`
- `phase39_independent_label_audit.json`
- `phase39_independent_label_audit.md`
- `phase39_label_registry_template.csv`

The JSON artifact should include source paths, registry settings, candidate
label list, row counts, per-label readiness rows, aggregate status,
interpretation, and claim boundary.

The registry template should be generated even when no external labels are
provided. It should make the next data-acquisition step concrete by showing the
columns required for future labels:

- `label_column`
- `source_path`
- `provenance_class`
- `description`
- `external_source_name`
- `independence_rationale`
- `allowed_for_phase38_rerun`

## Implementation Shape

Add one focused module:

```text
src/paper11_geofm/phase39_independent_label_audit.py
```

The module should expose:

- `build_phase39_independent_label_audit(...)`
- `write_phase39_independent_label_audit_artifacts(...)`

Add one runner:

```text
experiments/phase39_independent_label_audit/run_phase39_independent_label_audit.py
```

Add tests:

```text
tests/test_phase39_independent_label_audit.py
```

The implementation should follow the Phase 36/38 pattern: pure in-memory
analysis builder, explicit artifact writer, thin CLI runner, deterministic
fixtures, and no reward or PPO side effects.

## Error Handling

The builder should raise clear `ValueError` messages for:

- missing Phase 2 block feature CSV
- external label file without `block_id`
- duplicate external rows for the same `block_id`
- registry entries with unsupported provenance classes
- requested labels that do not exist in the joined table

Non-blocking label problems should be represented in readiness rows rather than
crashing the whole audit. Examples include single-class labels, missing split
coverage, or labels classified as leakage risks.

## Verification

The implementation must use test-first development. Tests should cover:

- current Bishan-like labels classified as explicit leakage risks
- source descriptor fields classified as source-field leakage risks
- external candidate labels joined by `block_id`
- independent candidate labels clearing the Phase 39 gate
- unclassified labels producing `candidate_proxy_labels_need_review`
- single-class or split-missing labels producing insufficient status
- registry parsing and unsupported provenance validation
- artifact writing and CLI execution against synthetic fixtures

Focused verification after implementation:

```powershell
python -m pytest tests\test_phase39_independent_label_audit.py tests\test_phase38_proxy_rebuild.py -q --basetemp=.pytest_tmp_phase39_final -p no:cacheprovider
python scripts\smoke_check.py
```

## Documentation Updates

After implementation and real-run verification, update:

- `README.md`
- `paper/phase28_results/README.md`
- a new `paper/phase28_results/13_phase39_independent_label_audit.md`
- `reproducibility/FILE_MANIFEST.tsv`
- `docs/superpowers/phase33_current_progress_handoff.md`

The handoff must record the real Phase 39 status and explicitly state whether
Phase 38 can be rerun with any non-leakage label. If the real run has only the
current Phase 2 labels, it must state that B2/B3 suitability reward remains
blocked.

## Claim Boundary

Phase 39 may support this guarded statement:

> Independent-label readiness can be audited reproducibly before Paper11
> rebuilds or rewards a suitability proxy.

Phase 39 may not support:

> GeoFM measures soil quality, irrigation, fertility, productivity, final
> planning performance, B2/B3 superiority, or cross-region transfer.
