# Phase 68 External Independent Label Package Design

## Purpose

Phase 68 follows the Phase 67 candidate reward/label target audit. Phase 67
found that existing Paper11 artifacts do not contain a candidate diagnostic
target that can support reward redesign: no target passed the gate, and the
strongest GeoFM-minus-explicit proxy signal remained far below the diagnostic
threshold.

The next step should not be another policy-training run and should not be a
reward rewrite. Phase 68 should build a reproducible external independent label
package and preflight gate so that Paper11 can accept, validate, and document a
future non-leakage label before rerunning Phase 40, Phase 41, or any reward
redesign work.

Phase 68 is algorithm and experiment-infrastructure work. It does not revise
formal submission files and it does not make manuscript-level claims.

## Scientific Question

Can Paper11 define a strict, reproducible admission contract for external
independent labels so that future soil, irrigation, yield, field-survey,
high-standard-farmland, or policy-outcome data can be checked before it affects
GeoFM suitability-prior or reward experiments?

The expected output is a conservative data-readiness decision: either an
external label package is ready for Phase 40 rerun, the package is missing or
invalid, or the independent-label route remains blocked.

## Why Phase 68 Is Needed

The current blocker is no longer model capacity alone. Prior phases show that
the evidence path is gated by label independence:

- Phase 40 reports missing independent label inputs unless a valid registry is
  supplied.
- Phase 41 cannot build `block_geofm_suitability_prior.csv` without a Phase
  40-passed independent label.
- Phase 42 found no local Phase 40-passing label source; only DLTB/slope-derived
  diagnostic labels and unrelated local labels were available.
- Phase 67 found zero candidate reward/label targets that can support reward
  redesign from current artifacts.

Therefore Phase 68 should convert the vague instruction "obtain independent
labels" into a concrete, testable data package. This keeps the project aligned
with the user's principle: improve the algorithm/model/experiment evidence
first, and only revise the manuscript after the evidence is stable.

## Scope

Phase 68 should produce a read-only package and preflight audit around external
labels. It should use the current real Bishan Phase 2 block table as the join
universe and should stay compatible with the existing Phase 39 and Phase 40
interfaces.

Included:

- generate a block-level external label CSV template;
- generate a Phase 40-compatible label registry template;
- generate a short README for external data providers;
- audit supplied external label CSVs and registry files before Phase 40;
- report schema, join, label-balance, missingness, split-coverage, and
  independence-readiness status;
- write JSON, CSV, and Markdown artifacts under the Phase 68 output directory;
- record a conservative result note under `paper/phase28_results` after the
  real run.

Excluded:

- training PPO, behavior cloning, supervised label models, or reward models;
- changing the reward function or enabling B2/B3;
- modifying Phase 40/41 decisions to force a pass;
- deriving labels from DLTB, slope, source metadata, current explicit planning
  features, GeoFM embeddings, or model predictions;
- editing `paper/submission/final/*`.

## Accepted Input Forms

Phase 68 should support two input modes.

The first mode is template-only mode with no external data. This mode generates
templates and returns `external_label_package_ready`. It is useful immediately
because the current workspace does not contain a passing independent label
source. If validation mode is requested but the external CSV or registry is
missing, the status should instead be `external_label_inputs_missing`.

The second mode is a supplied external label package:

- one or more block-level CSV files with `block_id` plus one or more label
  columns;
- a registry CSV or JSON file describing each label column;
- optional source-note metadata used only for audit reporting.

Spatial joins from source geometries are out of scope for Phase 68
implementation. If the data provider has geometry-only data, the Phase 68 README
should instruct them to provide a block-level joined CSV for this phase, using
Paper11 `block_id` as the stable join key.

## Registry Contract

The Phase 68 registry should be compatible with Phase 40 fields and may add
audit-only metadata. Required Phase 40-compatible fields:

- `label_column`
- `label_source`
- `source_type`
- `independence_level`
- `allowed_eval_roles`
- `provenance_note`
- `license_or_access`
- `expected_positive_definition`

Recommended audit-only fields:

- `source_owner`
- `collection_date_or_period`
- `spatial_join_method`
- `original_unit`
- `label_scale`
- `missing_value_policy`
- `known_overlap_with_dltb_slope_or_source_metadata`
- `contact_or_access_note`

Phase 68 should not expand Phase 40's accepted source types in a way that makes
the gate easier to pass. It should classify labels as ready for Phase 40 only
when their `source_type` and `independence_level` are already compatible with
Phase 40.

## Accepted Source Classes

Labels can be treated as independent candidates only when their registry
declares a source class compatible with Phase 40 independent sources, such as:

- `external_field_survey`
- `external_agronomic`
- `external_soil`
- `external_irrigation`
- `external_yield`
- `external_high_standard_farmland`
- `external_retention_or_policy`
- `remote_sensing_independent_product`

The following should remain diagnostic-only or blocked:

- `diagnostic_internal`
- `dltb_derived`
- `slope_derived`
- `source_metadata`
- `geofm_derived`
- `unknown`

Phase 68 should emit explicit warnings when a supplied label appears to overlap
with DLTB, slope, source category, current explicit features, or GeoFM-derived
outputs.

## Preflight Gate

For each registered label, Phase 68 should evaluate:

- required file and registry fields are present;
- `block_id` is present, non-empty, and unique in each external CSV;
- every external `block_id` either joins to the Phase 2 block table or is
  reported as out-of-universe;
- every Phase 2 block has either a parsed label value or an explicit missing
  status;
- the label can be parsed as binary using the registry positive definition;
- valid label count, missing count, missing rate, positive count, negative
  count, and positive rate are computed;
- train and evaluation split coverage contain enough valid labels;
- both train and evaluation splits contain positive and negative examples;
- source type and independence level are compatible with Phase 40;
- diagnostic-only labels are blocked from reward redesign even when they are
  numerically usable.

The default quantitative thresholds should mirror Phase 40:

- `min_valid_count`: `100`
- `max_missing_rate`: `0.20`
- `min_positive_rate`: `0.02`
- `max_positive_rate`: `0.98`
- `min_split_valid_count`: `20`

Phase 68 may be stricter than Phase 40 in its recommendation, but it must not
be more permissive.

## Status Model

Phase 68 should return one top-level status:

- `external_label_package_ready`: templates and documentation were generated,
  but no external label was supplied.
- `external_label_inputs_missing`: required external CSV or registry inputs are
  absent when validation mode was requested.
- `external_label_inputs_invalid`: files were supplied but failed schema, join,
  parsing, balance, missingness, split, or provenance checks.
- `phase40_ready_to_rerun_with_external_label`: at least one supplied label is
  independent, parseable, balanced enough, joined to the Phase 2 universe, and
  compatible with Phase 40 rerun.
- `independent_label_route_blocked`: all supplied labels are diagnostic-only,
  leakage-risk, or otherwise unsuitable for Phase 40/41 progression.

If multiple conditions apply, the status should prefer the most conservative
scientific interpretation. For example, a package with one diagnostic usable
label and no valid independent label should return
`independent_label_route_blocked`, not a ready status.

## Architecture

Add one focused module:

`src/paper11_geofm/phase68_external_independent_label_package.py`

Responsibilities:

- own the Phase 68 claim boundary;
- load the Phase 2 block table and split column;
- generate template rows from the Phase 2 block IDs;
- load and normalize external label CSVs;
- load and normalize registry CSV or JSON;
- validate schema and join behavior;
- compute per-label preflight metrics;
- classify labels against Phase 40-compatible source rules;
- write artifacts.

Add one thin runner:

`experiments/phase68_external_independent_label_package/run_phase68_external_independent_label_package.py`

Responsibilities:

- parse CLI arguments;
- call the module;
- print status, artifact paths, and claim boundary;
- support template-only mode and validation mode.

Add focused tests:

`tests/test_phase68_external_independent_label_package.py`

Responsibilities:

- verify template generation;
- verify missing input status;
- verify duplicate and blank `block_id` errors;
- verify unjoined external rows are reported;
- verify diagnostic DLTB/slope/source labels cannot pass;
- verify a valid independent label becomes ready for Phase 40 rerun;
- verify artifact writers produce JSON, CSV, and Markdown outputs.

## Data Flow

1. The runner receives a Phase 2 output directory and an output directory.
2. The module loads `block_geofm_features.csv` and extracts `block_id` plus
   `split`.
3. If no external labels are supplied, the module writes label and registry
   templates plus README and returns `external_label_package_ready` in
   template-only mode or `external_label_inputs_missing` in validation mode.
4. If external labels are supplied, the module validates each file, joins labels
   by `block_id`, and audits each registry row.
5. The module writes per-label preflight rows, package summary rows, diagnosis
   JSON, and diagnosis Markdown.
6. A real run result note records only the conservative status, reproduction
   command, and claim boundary.

## Error Handling

Hard errors should be reserved for malformed inputs that cannot be interpreted:

- missing Phase 2 block table;
- missing `block_id` or `split` in Phase 2;
- missing `block_id` in an external CSV;
- duplicate or blank external `block_id`;
- malformed registry JSON;
- unsupported registry extension;
- registry rows with blank `label_column`.

Recoverable validation failures should be represented in output rows:

- label column missing from supplied CSVs;
- label values not parseable as binary;
- high missingness;
- no positive or no negative examples;
- insufficient train/evaluation coverage;
- source class is diagnostic-only;
- independence rationale is absent or too weak;
- external block IDs do not join to Phase 2.

## Artifacts

The Phase 68 run should write:

- `phase68_external_label_template.csv`
- `phase68_label_registry_template.csv`
- `phase68_external_label_package_readme.md`
- `phase68_label_preflight.csv`
- `phase68_package_summary.csv`
- `phase68_external_independent_label_package.json`
- `phase68_external_independent_label_package.md`

Generated experiment outputs remain under ignored experiment output paths. The
paper-facing summary should be a concise result note:

`paper/phase28_results/34_phase68_external_independent_label_package.md`

## Testing And Verification

Targeted tests:

- template-only run produces the expected template and README artifacts;
- validation mode with missing files reports missing inputs;
- duplicate or blank external `block_id` raises a clear error;
- diagnostic source classes are blocked from Phase 40-ready status;
- a valid independent label with train/eval positive and negative coverage
  returns `phase40_ready_to_rerun_with_external_label`;
- artifact writers produce stable CSV, JSON, and Markdown files.

Regression checks:

- Phase 39 independent-label audit tests;
- Phase 40 independent-label gate tests;
- Phase 67 candidate target audit tests;
- smoke check;
- `git diff --check`;
- formal manuscript diff check under `paper/submission/final`.

## Claim Boundary

Phase 68 may claim that Paper11 now has a reproducible external-label package
and preflight gate. It may claim whether supplied external label inputs are
missing, invalid, blocked, or ready for Phase 40 rerun.

Phase 68 must not claim that GeoFM suitability reward is ready, that B2/B3 can
run, that a suitability prior is supported, that policy performance improved,
or that the formal manuscript is ready for submission. Those claims require a
Phase 40-passed independent label followed by the appropriate downstream
experiments.
