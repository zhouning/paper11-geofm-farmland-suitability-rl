# Next Experiment Matrix

## Purpose

This document turns the Phase 1 representation baseline into the next executable Paper11 experiments. The goal is to move from repository-level representation verification to evidence for the Paper11 scientific claims.

## Evidence Ladder

The next work should follow this order:

1. **Representation linkage:** attach real planning blocks or parcels to AlphaEarth-derived features.
2. **Proxy validation:** compare `suitability_proxy` against weak labels or external proxies.
3. **Policy integration:** add GeoFM features to the current block-level DRL state.
4. **Reward integration:** test whether suitability-aware reward terms improve spatial realism.
5. **Ablation and transfer:** separate semantic signal from dimensionality and test held-out regions.

## Required Data Before Phase 2

| Data item | Required for | Minimum acceptable form |
|---|---|---|
| Block or parcel polygons | Replace Phase 1 grid regions with planning units | GeoPackage, shapefile, or tabular block-to-pixel mapping |
| Explicit block features | B0 baseline and policy input | Existing 17-dimensional block features or documented replacement |
| DEM/slope features | Constraint and reward validity | Block-level mean slope or parcel slope distribution |
| Planning masks | Valid action set | Boolean action mask per block |
| Baseline reward components | B0 comparison | Slope, contiguity, baimu-fang, invalid-action components |
| Weak farmland labels | Proxy validation | Stable cropland, retained farmland, or high-standard farmland labels |
| Held-out region sample | Transfer claim | At least one held-out township, county, or physiographic region |

## Main Experimental Conditions

| ID | State | Reward | Purpose | Minimum evidence |
|---|---|---|---|---|
| B0 | explicit planning features | base reward | GIS-only DRL baseline | Planning metrics and action validity |
| B1 | explicit features + raw AlphaEarth 64d | base reward | Representation-only gain | B1 vs B0 under same reward |
| B2 | explicit features + `suitability_proxy` | base + suitability reward | Distilled proxy gain | B2 vs B0 on suitability and planning metrics |
| B3 | explicit features + raw AlphaEarth 64d + `suitability_proxy` | base + suitability reward | Full Paper11 model | B3 vs B0, B1, and B2 |

The central Paper11 claim should not be made until B1/B2/B3 are compared against B0 using the same planning environment and reporting protocol.

## Diagnostic Ablations

| ID | Variant | Purpose | Expected interpretation |
|---|---|---|---|
| D1 | AlphaEarth 64d only | Tests whether explicit planning features remain necessary | Poor planning validity would support the boundary that GeoFM enriches but does not replace GIS constraints |
| D2 | explicit features + random 64d | Controls for extra dimensionality | B1 should outperform D2 if GeoFM semantics matter |
| D3 | explicit features + shuffled AlphaEarth 64d | Controls for spatial alignment | Performance drop would support spatial semantic value |
| D4 | explicit features + PCA-8 or PCA-16 GeoFM | Tests compression | Similar performance would justify smaller state vectors |
| D5 | no slope term | Tests explicit constraint necessity | Degradation would prevent overclaiming GeoFM as a slope substitute |
| D6 | no contiguity term | Tests spatial-structure necessity | Degradation would support retaining planning logic |

## Proxy Validation Experiments

Before using `suitability_proxy` as a reward component, validate it as a weak remote-sensing proxy:

| Check | Question | Acceptable evidence |
|---|---|---|
| Stable farmland contrast | Are stable farmland blocks assigned higher proxy scores? | Distribution shift or AUC/F1 with weak labels |
| Slope contrast | Does the proxy avoid rewarding high-slope farmland after explicit slope is included? | Score distribution by slope quantile |
| Land-use contrast | Does the proxy separate retained farmland from non-farmland land cover? | Class-wise score summaries |
| Spatial inspection | Are high and low proxy regions plausible on the map? | Map panels and short qualitative notes |
| Calibration | Are score bins meaningful? | Reliability curve or monotonic trend against weak labels |

If these checks are unavailable, the manuscript should state that the score is a latent representation proxy and should not frame it as validated farmland suitability.

## Transfer Design

The most important Paper11 evidence is transfer. The minimum transfer design should include:

| Split | Role | Requirement |
|---|---|---|
| Train region | Policy fitting | One or more regions with full explicit features and GeoFM features |
| Validation region | Model selection | Held-out township or block subset from the same county |
| Test region | Main transfer evidence | Different township, county, or terrain context |
| OOD region | Optional stress test | Different terrain, urban pressure, or ecological background |

Report:

```text
transfer_drop = in_region_score - held_out_score
```

The claim should focus on whether GeoFM reduces the transfer drop relative to explicit-feature-only policies.

## Metrics

| Metric family | Metrics |
|---|---|
| Planning quality | total reward, slope change, contiguity improvement, baimu-fang count, baimu-fang area, valid action rate |
| Suitability quality | mean final `suitability_proxy`, suitability-weighted farmland area, low-suitability farmland area |
| Action behavior | action concentration by proxy quantile, action mask violations, budget efficiency |
| Transfer | in-region score, held-out score, transfer drop, ranking stability |
| Boundary checks | no-slope degradation, no-contiguity degradation, random/shuffled feature controls |

## Phase 2 Minimum Viable Experiment

The minimum next implementation should be:

1. Replace deterministic Phase 1 grid regions with real block or parcel regions.
2. Aggregate AlphaEarth 64-dimensional embeddings per block.
3. Join GeoFM block features with explicit block features.
4. Export a block-level feature table with B0, B1, B2, and B3-ready columns.
5. Validate the `suitability_proxy` against at least one weak label or explicit constraint.

This phase can be completed before full DRL training. It would create the table needed for clean policy integration and ablation design.

## Manuscript Claim Gate

| Claim | Gate before use in manuscript |
|---|---|
| GeoFM improves representation | B1 outperforms B0 and D2 |
| Suitability reward improves realism | B2 or B3 improves suitability metrics without harming slope or contiguity |
| GeoFM improves transfer | B1 or B3 has smaller transfer drop than B0 |
| GeoFM replaces explicit planning features | Do not make this claim; D1 is expected to fail key constraints |
| AlphaEarth measures soil, fertility, or irrigation | Do not make this claim |

## Recommended Next Step

Build a Phase 2 block-feature assembly pipeline. The output should be a block-level table that joins explicit planning features, AlphaEarth 64-dimensional block embeddings, `suitability_proxy`, weak labels, and region split identifiers.
