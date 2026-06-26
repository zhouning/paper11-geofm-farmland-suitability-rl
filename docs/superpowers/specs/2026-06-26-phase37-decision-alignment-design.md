# Phase 37 Decision-Alignment Audit Design

## Goal

Phase 37 tests whether the Phase 33 normalized-B1 selected blocks are aligned
with existing suitability-proxy and weak environmental diagnostics before any
new suitability reward or B2/B3 planning experiment is considered.

## One-Sentence Argument

Because Phase 36 found that the current suitability proxy is not reward-ready,
Phase 37 remains read-only and asks whether the already-generated Phase 33,
Phase 34, and Phase 35 decisions are at least directionally consistent with the
available proxy, slope, and weak farmland indicators.

## Scope

Phase 37 consumes existing diagnostic outputs and writes only diagnostic
artifacts. It does not run PPO, train a new proxy, alter reward definitions,
enable B2/B3, change Phase 2 feature tables, or support planning-performance
claims.

Inputs:

- Phase 34 case summary CSV:
  `experiments/phase34_case_map_diagnostics/outputs/real_bishan_5120_phase33_9run/phase34_case_map_cases.csv`.
- Phase 34 selected-block CSV:
  `experiments/phase34_case_map_diagnostics/outputs/real_bishan_5120_phase33_9run/phase34_case_map_blocks.csv`.
- Phase 35 action-overlap case CSV:
  `experiments/phase35_phase33_action_overlap_diagnostics/outputs/real_bishan_5120_phase33_9run/phase35_action_overlap_cases.csv`.
- Optional Phase 36 diagnosis JSON:
  `experiments/phase36_suitability_proxy_validation/outputs/real_bishan/phase36_suitability_proxy_validation.json`.

Outputs:

- `phase37_decision_alignment_cases.csv`
- `phase37_decision_alignment_summary.csv`
- `phase37_decision_alignment.json`
- `phase37_decision_alignment.md`

The local real-run output directory should be:

```text
experiments/phase37_decision_alignment/outputs/real_bishan_5120_phase33_9run
```

## Metrics

Phase 37 operates at the same case grain as Phase 34 and Phase 35:

```text
case_id = eval_tile_id|seed|variant_id|comparator_variant_id
```

For each case, it reports:

- Phase 33 role: positive, failure, or neutral case.
- Summary reward gap from Phase 35.
- Spatial pattern from Phase 34.
- Action-overlap pattern and selected-block Jaccard from Phase 35.
- Variant-minus-comparator gaps for:
  - base planning reward;
  - suitability proxy;
  - low-slope farmland label;
  - current farmland label;
  - slope mean;
  - slope max.

The Phase 34 case table already provides case-level gaps for base reward,
suitability proxy, and low-slope farmland label. The selected-block CSV is used
to compute current-farmland, slope-mean, and slope-max gaps by reconstructing
the variant and comparator selected sets from `selection_role`, `variant_step`,
and `comparator_step`.

Shared selected blocks are included in both the variant and comparator set when
computing per-set means. Variant-only and comparator-only blocks are used only
for selected-set reconstruction, not as unmatched samples.

## Summary Reduction

The summary artifact groups cases by:

- `case_role`
- `eval_tile_id`
- `variant_id`
- `comparator_variant_id`
- `spatial_pattern`
- `action_overlap_pattern`

For each group, it reports case count, positive-gap counts, and mean gaps for
all alignment metrics. The row-level `proxy_alignment_pattern` is descriptive;
the summary grouping stays anchored to the Phase 34 `spatial_pattern` and Phase
35 `action_overlap_pattern`.

The aggregate status is conservative:

- `decision_alignment_supported_for_proxy_rebuild`: at least one Phase 33
  positive-case group has positive mean suitability-proxy or low-slope label
  alignment, and no Phase 33 failure-case group shows the same positive
  status-gate signal.
- `decision_alignment_not_supported`: inputs are complete, but positive and
  failure cases do not separate in the available proxy or weak environmental
  metrics.
- `decision_alignment_inputs_insufficient`: required CSVs are missing,
  unreadable, or contain no joinable cases.

This status can justify a proxy-rebuild or external-label acquisition branch
only when the conservative subgroup gate is supported. It cannot justify
suitability reward use.

## Leakage And Claim Boundary

Phase 37 inherits the Phase 36 boundary. The available weak labels are
DLTB/slope-derived and include explicit-feature leakage risk. Current-farmland
and low-slope metrics are therefore descriptive diagnostics only. They are not
independent agronomic validation labels.

Phase 37 may support this guarded statement:

> Existing Phase 33 decision differences can be audited for directional
> consistency with available weak environmental and proxy diagnostics.

Phase 37 may not support:

> GeoFM directly measures farmland suitability, soil quality, irrigation
> access, productivity, or final planning performance.

Phase 37 must not emit any status that says reward use is ready.

## Implementation Shape

Add one focused module:

```text
src/paper11_geofm/phase37_decision_alignment.py
```

The module should expose:

- `build_phase37_decision_alignment(...)`
- `write_phase37_decision_alignment_artifacts(...)`

Add one runner:

```text
experiments/phase37_decision_alignment/run_phase37_decision_alignment.py
```

Add tests:

```text
tests/test_phase37_decision_alignment.py
```

The implementation should follow the Phase 34/35/36 pattern: read CSV/JSON
inputs, build a pure in-memory analysis dictionary, write CSV/JSON/Markdown
artifacts, and keep all claim boundaries explicit in every output row.

## Verification

The implementation must use test-first development. The core tests should
cover:

- successful case/block/action join;
- selected-set reconstruction with shared blocks included in both sets;
- conservative status reduction for supported, unsupported, and insufficient
  inputs;
- artifact writing for CSV, JSON, and Markdown;
- CLI execution against a synthetic fixture.

The real-run verification should include:

```powershell
python -m pytest tests\test_phase37_decision_alignment.py tests\test_phase36_suitability_proxy_validation.py -q --basetemp=.pytest_tmp_phase37_final -p no:cacheprovider
python scripts\smoke_check.py
```

## Documentation Updates

After implementation and real-run verification, update:

- `README.md`
- `paper/phase28_results/README.md`
- `paper/phase28_results/11_phase37_decision_alignment.md`
- `reproducibility/FILE_MANIFEST.tsv`
- `docs/superpowers/phase33_current_progress_handoff.md`

The handoff must state that Phase 37 is diagnostic-only and that Phase 36 still
blocks B2/B3 suitability reward.
