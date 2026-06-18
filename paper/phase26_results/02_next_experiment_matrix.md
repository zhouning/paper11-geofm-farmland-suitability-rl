# Phase 26 Next Experiment Matrix

## Purpose

This document turns the current Phase 26 result boundary into the next
executable Paper11 experiments. The goal is to move from a manuscript-facing
analysis package to evidence that can actually support a paper claim.

## Evidence Ladder

The next work should follow this order:

1. **Stability diagnosis:** explain why the learned-policy B1-B0 mean delta is
   currently negative on the 4096-step result set. Phase 27 now completes this
   read-only diagnostic step and reports `budget_not_explanatory`.
2. **Representation diagnostics:** separate GeoFM signal from extra input
   capacity and seed noise.
3. **Proxy validation:** re-check whether `suitability_proxy` is ready for
   reward use.
4. **Reward integration:** test whether suitability-aware reward improves
   layout realism without breaking planning constraints.
5. **Transfer and robustness:** test held-out-region behavior, ablations, and
   uncertainty.

## Current Evidence Gap

The current Phase 26 package is not enough for the main manuscript claim
because:

- learned-policy mean B1-B0 delta is negative at `4096` steps;
- only `3 / 9` tile-seed pairs favor B1;
- the 1024-step result is also negative on average, so budget alone has not
  resolved the claim gap;
- suitability reward remains blocked;
- transfer evidence remains Bishan-only;
- spatial case maps and uncertainty are still missing;
- the ablation package is incomplete.

## Required Data Before the Next Phase

| Data item | Required for | Minimum acceptable form |
|---|---|---|
| Phase 25/26 outputs | learned-policy diagnosis | tile-seed comparison CSV/JSON artifacts |
| Explicit block features | B0 baseline and comparison | existing block feature tables |
| GeoFM block features | B1 representation diagnostics | 64-dimensional block embeddings |
| Weak farmland labels | proxy validation | stable farmland or related weak labels |
| Held-out region sample | transfer claim | at least one non-Bishan region |
| Spatial case maps | interpretability | representative map panels |
| Ablation runs | semantic vs dimensionality separation | random/shuffled/PCA controls |

## Main Experimental Conditions

| ID | State | Reward | Purpose | Minimum evidence |
|---|---|---|---|---|
| B0 | explicit planning features | base reward | GIS-only baseline | planning metrics and action validity |
| B1 | explicit features + raw AlphaEarth 64d | base reward | representation-only gain | B1 vs B0 under the same protocol |
| B2 | explicit features + `suitability_proxy` | base + suitability reward | distilled proxy gain | suitability and planning comparison |
| B3 | explicit features + raw AlphaEarth 64d + `suitability_proxy` | base + suitability reward | full Paper11 model | B3 vs B0/B1/B2 |

The central Paper11 claim should not be made until B1/B2/B3 are compared
against B0 using the same planning environment and reporting protocol.

## Diagnostic Ablations

| ID | Variant | Purpose | Expected interpretation |
|---|---|---|---|
| D1 | AlphaEarth 64d only | tests whether explicit planning features remain necessary | poor validity would support the boundary that GeoFM enriches but does not replace GIS constraints |
| D2 | explicit features + random 64d | controls for extra dimensionality | B1 should beat D2 if GeoFM semantics matter |
| D3 | explicit features + shuffled AlphaEarth 64d | controls for spatial alignment | B1 should beat D3 if block alignment matters |
| D4 | explicit features + PCA-8 or PCA-16 GeoFM | tests compression | similar performance would justify smaller state vectors |
| D5 | no slope term | tests explicit constraint necessity | degradation would prevent overclaiming GeoFM as a slope substitute |
| D6 | no contiguity term | tests spatial-structure necessity | degradation would support retaining planning logic |

## Proxy Validation Experiments

Before using `suitability_proxy` as a reward component, validate it as a weak
remote-sensing proxy:

| Check | Question | Acceptable evidence |
|---|---|---|
| Stable farmland contrast | Are stable farmland blocks assigned higher proxy scores? | distribution shift or AUC/F1 with weak labels |
| Slope contrast | Does the proxy avoid rewarding high-slope farmland after explicit slope is included? | score distribution by slope quantile |
| Land-use contrast | Does the proxy separate retained farmland from non-farmland land cover? | class-wise score summaries |
| Spatial inspection | Are high and low proxy regions plausible on the map? | map panels and short qualitative notes |
| Calibration | Are score bins meaningful? | reliability curve or monotonic trend against weak labels |

If these checks are unavailable, the manuscript should state that the score is
a latent representation proxy and should not frame it as validated farmland
suitability.

## Transfer Design

The most important Paper11 evidence is transfer. The minimum transfer design
should include:

| Split | Role | Requirement |
|---|---|---|
| Train region | policy fitting | one or more regions with full explicit features and GeoFM features |
| Validation region | model selection | held-out Bishan tile or same-county subset |
| Test region | main transfer evidence | different township, county, or terrain context |
| OOD region | optional stress test | different terrain, urban pressure, or ecological background |

Report:

```text
transfer_drop = in_region_score - held_out_score
```

The claim should focus on whether GeoFM reduces the transfer drop relative to
explicit-feature-only policies.

## Metrics

| Metric family | Metrics |
|---|---|---|
| Planning quality | total reward, slope change, contiguity improvement, baimu-fang count, baimu-fang area, valid action rate |
| Suitability quality | mean final `suitability_proxy`, suitability-weighted farmland area, low-suitability farmland area |
| Action behavior | action concentration by proxy quantile, action mask violations, budget efficiency |
| Transfer | in-region score, held-out score, transfer drop, ranking stability |
| Boundary checks | no-slope degradation, no-contiguity degradation, random/shuffled feature controls |

## Minimum Viable Next Experiment

The minimum next implementation should be:

1. Compare B1 against random, shuffled, and PCA-compressed GeoFM controls.
2. Run B0/B1 stability checks across additional seeds or intermediate/repeated
   budgets only if compute is available.
3. Validate or further block `suitability_proxy` for reward use.
4. Add at least one spatial case map for a representative held-out tile.

This work can be completed before any new manuscript claim. It would create the
diagnostic evidence needed to decide whether Paper11 should stay focused on a
GeoFM-enhanced representation paper or escalate to a full suitability-reward
paper.

## Manuscript Claim Gate

| Claim | Gate before use in manuscript |
|---|---|
| GeoFM improves representation | B1 outperforms B0 and D2 |
| Suitability reward improves realism | B2 or B3 improves suitability metrics without harming slope or contiguity |
| GeoFM improves transfer | B1 or B3 has smaller transfer drop than B0 |
| GeoFM replaces explicit planning features | do not make this claim; D1 is expected to fail key constraints |
| AlphaEarth measures soil, fertility, or irrigation | do not make this claim |

## Recommended Next Step

Phase 27 has now run the Phase 26 follow-on diagnosis package comparing the
current 1024 and 4096 step result sets. It reports:

- diagnostic status: `budget_not_explanatory`;
- mean delta change from 1024 to 4096 steps: `0.3010310174`;
- positive tile-seed count change: `-1`;
- stability counts: stable-positive `1`, stable-negative `3`,
  flip-to-positive `2`, flip-to-negative `3`, incomplete `0`.

The next implementation target should therefore be:

1. run the implemented Phase 28 representation-control package on
   `B0,B1,D2,D3,D4P8,D4P16` under the same padded held-out base-reward
   protocol;
2. require full B0/B1/D2/D3/D4P8/D4P16 coverage before any
   `representation_signal_supported` interpretation;
3. run a repeated or intermediate-budget B0/B1 stability sweep only if compute
   is available after the control package is analyzed;
4. update suitability-proxy validation before any reward integration.
