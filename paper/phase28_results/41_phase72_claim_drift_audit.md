# Phase 72 Claim-Drift Audit

Status: `claim_drift_requires_narrowing`

## Purpose

This read-only audit compares four GeoFM-compressed-state sentences in the
formal manuscript with the later Phase 60, 62, 69, 71, 72B, and exhaustion
evidence. It does not edit `paper/submission/final/*`, train Phase 72C, alter
rewards, or promote blocked transfer, suitability, future-planning, PCA, or
GeoFM-specific claims.

## Current Decision

The formal manuscript can retain a bounded Bishan base-reward compressed-state
result, but its broad wording that “GeoFM information improved” should be
reviewed before any future manuscript revision. Phase 59 and Phase 62 prevent
attributing the improvement uniquely to GeoFM information or to PCA. Phase 69
and Phase 71 keep the explicit-target and suitability limitations active.
Phase 72B blocks future-aware and cross-region claims.

## Claim Matrix

| Claim | Status | Current evidence | Action |
| --- | --- | --- | --- |
| Real Bishan planning workflow | `supported` | Real-unit, tiled, padded planning protocol is reproducible. | Retain with scope. |
| Bounded low-dimensional compressed route | `bounded_supported` | Phase 60 preserves the earlier compressed-route result under the Bishan base reward. | Retain bounded wording. |
| GeoFM-specific compressed information | `blocked` | Phase 59 does not support matched-dimension GeoFM advantage; Phase 62 favors D6 random projections. | Do not claim. |
| PCA optimality | `blocked` | D4P8/D4P16 do not beat matched D6 controls. | Do not claim. |
| Suitability reward or agronomic value | `blocked` | Phase 69 and Phase 71 keep suitability and independent agronomic claims blocked. | Do not claim. |
| Cross-region transfer | `blocked` | Phase 72B fails both zero-shot transfer directions. | Do not claim. |
| Future-aware prediction/planning | `blocked` | Phase 72B stops before Phase 72C; planning outcomes are not evaluated. | Do not claim. |
| Current formal wording | `needs_narrowing` | Later evidence requires separating low-dimensional optimization benefit from GeoFM-specific information. | Review before any future revision. |

## Required Boundary

This audit does not authorize a formal manuscript change. The repository keeps
the submitted files unchanged and records the claim drift as a future revision
input only.

## Generated Artifacts

Ignored real-run artifacts are under:

```text
experiments/phase72_claim_drift_audit/outputs/real_formal_manuscript
```

Artifacts:

```text
phase72_claim_drift_claims.csv
phase72_claim_drift_audit.json
phase72_claim_drift_audit.md
```

## Verification

```text
claim-drift plus Phase 72A/exhaustion tests: 19 passed
full repository: 558 passed, 84 existing sklearn warnings
smoke check: passed
git diff --check: passed
paper/submission/final/*: unchanged
```
