# Phase 26 Results Package

This folder records the current Phase 26 evidence for Paper11. It translates
the executable Phase 25 padded held-out policy outputs and the Phase 26
analysis package into reviewer-facing interpretation, and it defines the next
experiment matrix needed before manuscript-level claims can be made.

## Files

- `01_phase26_result_interpretation.md`: what the current Phase 26 outputs
  show, what they do not show, and how they should be cited in the manuscript
  workflow.
- `02_next_experiment_matrix.md`: the next Paper11 experiments required to
  resolve the remaining suitability, transfer, ablation, and robustness gaps.
- `03_phase26_budget_comparison.md`: a direct comparison of the 1024-step and
  4096-step Phase 26 result sets.

## Reproduction Link

The Phase 26 analysis package is driven from the Phase 25 padded held-out
policy outputs under:

```text
experiments/phase26_main_experiment/outputs/macos_main_4096/phase25_run/
experiments/phase26_main_experiment/outputs/macos_main_4096/phase26_analysis/
```

The current reviewed artifacts are based on the `4096`-step macOS result set.

## Claim Boundary

Phase 26 currently shows that B0/B1 padded held-out Bishan tile learned-policy
outputs can be analyzed into tile-seed delta tables and claim-readiness
artifacts. It does not yet demonstrate stable B1 superiority over B0, does not
enable suitability reward, does not test B2/B3, and does not support
cross-region transfer or final manuscript claims.
