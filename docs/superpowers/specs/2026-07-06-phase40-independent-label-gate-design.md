# Phase 40 Independent-Label Gate Design

## Purpose

Phase 40 turns the Phase 39 conclusion, `independent_label_inputs_missing`,
into an executable admission gate for future suitability-proxy work. It is
designed to answer one reviewer-critical question:

```text
Does Paper11 have at least one defensible independent, non-leakage label that
can support a Phase 38 proxy-rebuild rerun before any B2/B3 suitability reward
experiment?
```

Phase 40 is not a policy-training experiment. It does not enable B2/B3 by
itself, does not change rewards, and does not claim suitability-reward or
planning-performance improvement.

## Background

The current Paper11 evidence chain is blocked at the suitability branch:

- Phase 36 reports `proxy_signal_not_supported`.
- Phase 37 reports `decision_alignment_not_supported`.
- Phase 38 reports `proxy_rebuild_diagnostic_only`.
- Phase 39 reports `independent_label_inputs_missing`.

The available default columns in the real Bishan Phase 2 table are DLTB,
slope, or source-code derived. They are useful diagnostics, but they are not
independent agronomic or planning-outcome labels. Phase 40 therefore defines a
strict registry-driven gate for any future label source.

## Scope

### In Scope

- Add a Phase 40 Python module that validates independent-label readiness.
- Add a Phase 40 CLI runner under `experiments/`.
- Support CSV and JSON label registries, following Phase 39 registry patterns.
- Join registry entries to the real Phase 2 feature table by label column name.
- Compute label-level and gate-level readiness diagnostics.
- Produce CSV, JSON, and Markdown artifacts.
- Add focused pytest coverage for pass, blocked, missing, and registry parsing
  paths.
- Update README, file manifest, phase-results documentation, and submission
  readiness text to reflect the new gate.

### Out of Scope

- Creating or downloading external labels.
- Inferring independent labels from DLTB, slope, source class, or GeoFM
  embeddings.
- Running Phase 38 proxy rebuild with a newly accepted label.
- Running PPO, B2, B3, reward integration, or transfer experiments.
- Claiming agronomic validity, suitability-reward readiness, or policy
  superiority.

## Inputs

### Required

- Phase 2 real feature table directory:

```text
experiments/phase11_bishan_dltb_real/outputs/phase2_real
```

The module reads `block_geofm_features.csv` from this directory.

### Optional

- Label registry file, CSV or JSON.
- If omitted, Phase 40 should report `independent_label_inputs_missing`.
- If supplied but empty, Phase 40 should also report
  `independent_label_inputs_missing`.

## Registry Contract

Each registry entry describes one candidate label column. Required fields:

| Field | Meaning |
|---|---|
| `label_column` | Column name expected in `block_geofm_features.csv`. |
| `label_source` | Human-readable source description. |
| `source_type` | Controlled source category. |
| `independence_level` | Claimed independence from DLTB/slope/source-code fields. |
| `allowed_eval_roles` | Split roles where the label may be evaluated. |

Recommended optional fields:

| Field | Meaning |
|---|---|
| `provenance_note` | Short explanation of how the label was produced. |
| `license_or_access` | Sharing or access note for the label source. |
| `expected_positive_definition` | What value should be treated as positive. |

Accepted `source_type` values:

- `external_field_survey`
- `external_agronomic`
- `external_soil`
- `external_irrigation`
- `external_yield`
- `external_high_standard_farmland`
- `external_retention_or_policy`
- `remote_sensing_independent_product`
- `diagnostic_internal`
- `dltb_derived`
- `slope_derived`
- `source_metadata`
- `geofm_derived`
- `unknown`

Accepted `independence_level` values:

- `independent`
- `partially_independent`
- `diagnostic_only`
- `leakage_risk`
- `unknown`

## Gate Rules

Phase 40 evaluates each registry entry, then assigns a single gate status.

### Label-Level Status

`label_gate_passed` when all are true:

- `label_column` exists in the Phase 2 feature table.
- valid label count is at least `min_valid_count`.
- missing rate is at most `max_missing_rate`.
- positive rate is between `min_positive_rate` and `max_positive_rate`.
- train and evaluation split coverage both meet `min_split_valid_count`.
- `source_type` is an accepted independent external category or
  `remote_sensing_independent_product`.
- `independence_level` is `independent` or `partially_independent`.
- `allowed_eval_roles` includes at least one evaluation role present in the
  feature table split column.

`label_gate_diagnostic_only` when the label exists and is computable but fails
because source provenance is internal, partially unsafe, or evaluation use is
not allowed.

`label_gate_blocked` when the label exists but fails count, missingness,
balance, split-coverage, or parsing requirements.

`label_missing` when the registry names a column not present in the feature
table.

### Gate-Level Status

`independent_label_gate_passed` when at least one label reaches
`label_gate_passed`.

`independent_label_gate_diagnostic_only` when no label passes, but at least one
label is computable as diagnostic-only evidence.

`independent_label_gate_blocked` when a non-empty registry exists but every
candidate is missing or blocked.

`independent_label_inputs_missing` when no registry is supplied, the supplied
registry has no candidate rows, or all registry entries fail to parse.

## Default Thresholds

Use conservative defaults that can be overridden by CLI arguments:

| Parameter | Default |
|---|---:|
| `min_valid_count` | `100` |
| `max_missing_rate` | `0.20` |
| `min_positive_rate` | `0.02` |
| `max_positive_rate` | `0.98` |
| `min_split_valid_count` | `20` |

These thresholds are readiness gates only. Passing them does not establish
agronomic validity.

## Outputs

Write artifacts under:

```text
experiments/phase40_independent_label_gate/outputs/<run_name>
```

Expected files:

- `phase40_label_gate_summary.csv`
- `phase40_independent_label_gate.json`
- `phase40_independent_label_gate.md`

The JSON should include:

- gate status;
- row counts;
- threshold settings;
- registry path and row count;
- per-label diagnostics;
- claim boundary;
- recommended next step.

The Markdown should be reviewer-facing and explicitly state whether Phase 38
may be rerun with a stronger label.

## Code Architecture

Add:

```text
src/paper11_geofm/phase40_independent_label_gate.py
experiments/phase40_independent_label_gate/run_phase40_independent_label_gate.py
tests/test_phase40_independent_label_gate.py
paper/phase28_results/14_phase40_independent_label_gate.md
```

The module should expose small, testable functions:

- `load_feature_table(path)`
- `load_label_registry(path)`
- `evaluate_label_candidate(features, registry_row, thresholds)`
- `summarize_gate(label_results)`
- `write_phase40_artifacts(result, output_dir)`
- `run_phase40_independent_label_gate(...)`

The CLI should be a thin wrapper around the module.

## Error Handling

- Missing feature table is a hard error.
- Missing registry is not a hard error; it produces
  `independent_label_inputs_missing`.
- Invalid registry rows should be captured as blocked diagnostics rather than
  crashing the full run when other rows are usable.
- Unsupported registry extension should be a hard error.
- Non-binary labels should be handled only if they can be mapped through
  `expected_positive_definition`; otherwise they are blocked.

## Tests

Add tests for:

1. no registry supplied returns `independent_label_inputs_missing`;
2. empty registry returns `independent_label_inputs_missing`;
3. independent CSV registry with a valid binary label returns
   `independent_label_gate_passed`;
4. JSON registry is parsed with the same semantics as CSV;
5. DLTB-derived or slope-derived label returns diagnostic-only or blocked;
6. missing label column returns `label_missing` and gate blocked;
7. severe class imbalance blocks a label;
8. train/evaluation split coverage below threshold blocks a label;
9. artifact writer creates CSV, JSON, and Markdown outputs.

## Documentation Updates

Update:

- `README.md`
- `paper/phase28_results/README.md`
- `paper/submission/01_ijaeog_submission_readiness.md`
- `paper/submission/02_draft_titles_highlights_declarations.md`
- `reproducibility/FILE_MANIFEST.tsv`
- `docs/superpowers/phase33_current_progress_handoff.md`

The documentation must preserve the claim boundary:

```text
Phase 40 validates independent-label readiness. It does not enable B2/B3,
does not run PPO, does not alter rewards, and does not prove suitability or
planning-performance improvement.
```

## Success Criteria

Implementation is complete when:

- focused Phase 40 tests pass;
- smoke check still passes;
- Phase 40 can run without a registry and produces
  `independent_label_inputs_missing`;
- Phase 40 can run on a test registry and produces deterministic pass/blocked
  artifacts;
- docs state the new gate and do not overclaim suitability reward readiness.

## Next Step After Phase 40

If Phase 40 passes with a defensible independent label, the next phase should
rerun Phase 38 proxy rebuild using that label and then test whether Phase
33/34/35 selected block sets are ordered by the rebuilt proxy. Only after that
should Paper11 consider a bounded B2/B3 reward smoke.

If Phase 40 remains blocked, the manuscript should be framed as a reproducible
GeoFM-planning diagnostic platform with explicit negative evidence rather than
a positive suitability-reward paper.
