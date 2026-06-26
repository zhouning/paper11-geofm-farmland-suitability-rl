# Phase 37 Decision-Alignment Audit

## One-Sentence Argument

Phase 37 audits whether completed Phase 33 normalized-B1 decisions separate
from matched comparator decisions in available proxy, slope, and weak farmland
diagnostics, and the current real run does not provide conservative
decision-alignment support while Phase 36 continues to block B2/B3 suitability
reward use.

## Current Experiment Snapshot

Inputs:

- Phase 34 case map cases:
  `experiments/phase34_case_map_diagnostics/outputs/real_bishan_5120_phase33_9run/phase34_case_map_cases.csv`
- Phase 34 selected blocks:
  `experiments/phase34_case_map_diagnostics/outputs/real_bishan_5120_phase33_9run/phase34_case_map_blocks.csv`
- Phase 35 action-overlap cases:
  `experiments/phase35_phase33_action_overlap_diagnostics/outputs/real_bishan_5120_phase33_9run/phase35_action_overlap_cases.csv`
- Phase 36 diagnosis:
  `experiments/phase36_suitability_proxy_validation/outputs/real_bishan/phase36_suitability_proxy_validation.json`

Local ignored outputs:

```text
experiments/phase37_decision_alignment/outputs/real_bishan_5120_phase33_9run
```

Generated artifacts:

```text
phase37_decision_alignment_cases.csv
phase37_decision_alignment_summary.csv
phase37_decision_alignment.json
phase37_decision_alignment.md
```

## Main Result

The current Phase 37 status is:

```text
decision_alignment_not_supported
```

Phase 36 remains:

```text
proxy_signal_not_supported
```

Real row counts from `phase37_decision_alignment.json`:

- Phase 34 case rows: `54`
- Phase 34 selected-block rows: `857`
- Phase 35 case rows: `54`
- Phase 37 joined case rows: `54`
- Phase 37 summary rows: `37`

## Interpretation

Phase 37 did not find conservative decision-alignment support under the failure-subgroup gate. Across all
joined cases, the mean summary reward gap is `-0.4109646286`, the mean
suitability-proxy gap is `-0.0066127380`, and the mean low-slope farmland-label
gap is `-0.0486111111`. Positive Phase 33 cases have positive mean summary
reward gap (`0.5822555613`) and `22 / 24` proxy-or-label alignment cases, but
failure cases also include `17 / 30` proxy-or-label alignment cases while
remaining strongly negative on mean summary reward gap (`-1.2055407806`).

This makes the result diagnostic-only and not sufficient for a proxy-rebuild
success claim. It does not make B2/B3 or suitability reward ready.

## Claim Boundary

Phase 37 is diagnostic only. It does not run policy training, alter rewards,
enable suitability reward, test B2/B3, prove GeoFM agronomic validity, or
support final planning-performance claims.
