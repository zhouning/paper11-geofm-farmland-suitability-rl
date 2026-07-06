# Evidence-Gated GeoFM Representation for Farmland Layout Optimization: A Conclusion Manuscript Draft

Manuscript status: formal conclusion draft. The completed evidence rejects the current positive Paper11 hypothesis under the tested protocol.

Current decision: do not submit this as a manuscript claiming GeoFM superiority,
suitability-reward improvement, B2/B3 readiness, or cross-region transfer. The
current evidence directly rejects those claims for the present Paper11 design:
raw B1 does not stably outperform B0, compressed controls exceed raw B1, and
suitability-reward work remains blocked by the independent-label gate.

## Title

Experimental rejection of unsupported GeoFM superiority in reinforcement-learning farmland layout optimization

## Short Title

Unsupported GeoFM superiority in farmland planning

## Keywords

Geospatial foundation model; AlphaEarth embeddings; farmland layout
optimization; reinforcement learning; suitability proxy; representation
control; evidence gate; land-use planning

## Highlights

- A reproducible workflow links AlphaEarth embeddings to real Bishan DLTB planning blocks.
- Tiled and padded policy interfaces make large real planning instances testable.
- Multi-tile B0/B1 learned-policy evidence does not support raw GeoFM superiority.
- Representation controls show that compressed GeoFM controls can exceed raw B1.
- Suitability-reward work is blocked until an independent non-leakage label gate passes.
- The current conclusion rejects the positive GeoFM-superiority and B2/B3-readiness version of Paper11 under the tested protocol.

## Abstract

Farmland spatial layout optimization requires decisions over many heterogeneous
planning units, but direct variables for soil quality, irrigation, productivity,
and long-term suitability are often unavailable at operational planning scale.
Geospatial foundation-model (GeoFM) embeddings offer a possible source of
latent land-surface information, but their usefulness for reinforcement-learning
planning should not be assumed without representation controls, reward
validation, and leakage-aware suitability labels. This study builds a
reproducible evidence-gated workflow that aggregates AlphaEarth embeddings and
explicit planning features to real Bishan DLTB land-use blocks, constructs
tiled and padded planning interfaces, and evaluates whether GeoFM-enhanced
representations support learned farmland layout decisions under a deterministic
base planning reward. The current multi-tile, multi-seed held-out evidence does
not support a positive raw-GeoFM claim: at 4096 training steps, the learned
B1-B0 mean reward delta is `-0.1318712688`, with only `3 / 9` held-out
tile-seed pairs favoring B1. Representation-control experiments also report
`compression_matches_raw`, with PCA-compressed controls exceeding raw B1 in the
4096-step run. A normalized-B1 branch partially improves raw B1 at 4096 steps,
but a broader 5120-step matched robustness check reports
`budget_not_explanatory`. Suitability-proxy validation remains more restrictive:
available weak labels are DLTB/slope-derived leakage risks, the scalar
`suitability_proxy` is not supported as a reward term, and the Phase 40
independent-label gate reports `independent_label_inputs_missing` with
`64,984` feature rows and `0` registry rows. These results directly reject the
current positive Paper11 hypothesis under the tested protocol. The defensible
conclusion is negative and decision-level: B2/B3 suitability-reward experiments
should not proceed until an independent non-leakage label registry passes the
gate and a leakage-aware proxy rebuild clears control checks.

## 1. Introduction

Farmland layout optimization is not only a geometric consolidation problem. In
practical land-use planning, decisions about whether to retain, exchange, or
consolidate farmland depend on explicit spatial constraints as well as latent
environmental conditions that are difficult to observe comprehensively at the
planning-unit scale. Slope, contiguity, block area, adjacency, and fragmentation
can often be derived from GIS data, but variables related to irrigation,
soil quality, moisture, and productivity may be incomplete, unavailable, or
restricted.

Geospatial foundation-model embeddings provide a tempting route around this
data gap. Multi-sensor satellite embedding products such as AlphaEarth can
encode broad land-surface context in compact feature vectors, and such vectors
may contain information relevant to farmland suitability. However, a remote-
sensing embedding is not a direct measurement of agronomic suitability. If a
planning model improves after adding GeoFM embeddings, the improvement may come
from meaningful environmental signal, from representation scale, from redundant
compression, from easier optimization geometry, or from leakage through labels
or features.

Paper11 was designed to test this distinction rather than assume it. The
workflow links AlphaEarth embeddings to real Bishan DLTB land-use polygons,
constructs explicit-only and GeoFM-enhanced planning representations, and
evaluates learned policies under a deterministic base planning reward. It also
implements random, shuffled, compressed, normalized, suitability-proxy, and
independent-label gates so that each stronger manuscript claim must pass an
explicit evidence threshold.

The current evidence rejects the original positive framing that GeoFM features improve farmland layout optimization in the tested Paper11 design. The strongest current contribution is a conclusion rather than a promise: under the completed evidence gates, the raw GeoFM representation is not a supported improvement, and suitability reward must not be used. This paper therefore frames the current Paper11 result as an evidence-gated negative finding.

## 2. Data and Planning Units

The workflow uses real Bishan DLTB land-use polygons as planning units. The
Phase 11 adapter exports `64,984` DLTB polygons into Phase 2-compatible block
feature tables. The feature pipeline aggregates AlphaEarth annual embeddings
to block-level representations and joins explicit planning attributes such as
slope- and land-use-derived variables where available.

The repository includes lightweight sample AlphaEarth arrays for smoke tests,
while the full external Bishan DLTB-with-slope GeoPackage is documented as a
local external input and is not redistributed in ordinary Git. Generated real
outputs under `experiments/**/outputs/` are ignored artifacts. The repository
therefore separates lightweight reviewer reproducibility from full local data
reconstruction.

## 3. Method

### 3.1 Representation Families

The core representation families are:

- `B0`: explicit planning features with deterministic `base_planning_reward`;
- `B1`: explicit planning features plus raw 64-dimensional AlphaEarth or GeoFM embeddings;
- `D2`: explicit planning features plus random 64-dimensional controls;
- `D3`: explicit planning features plus shuffled AlphaEarth embeddings;
- `D4P8` and `D4P16`: explicit planning features plus PCA-compressed AlphaEarth embeddings;
- `N1Z` and `N1ZR`: normalized B1 variants used to test representation-scale effects;
- `B2` and `B3`: suitability-reward variants reserved for later use and not enabled in the current evidence.

This design separates three questions. First, can the planning environment and
policy interface operate on real planning units? Second, does adding raw GeoFM
information improve learned-policy decisions under the same deterministic base
reward? Third, is there validated suitability evidence strong enough to justify
adding a suitability reward? The current study answers the first question
positively and the second and third questions negatively under present inputs.

### 3.2 Planning Reward and Policy Interface

The current learned-policy experiments use a deterministic `base_planning_reward`.
This reward encodes explicit planning logic such as slope, contiguity,
baimu-fang-style consolidation, action validity, and related planning
constraints. It deliberately excludes suitability reward because the suitability
proxy has not passed the evidence gates required for reward integration.

Early flat observation/action interfaces were tile-size-specific and could not
support learned-policy evaluation across variable-size held-out tiles. Later
phases introduced tiled contracts and padded variable-size policy evaluation,
allowing bounded same-reward comparisons across held-out Bishan tiles.

### 3.3 Evidence Gates

The evidence-gated design prevents unsupported claims from entering the
manuscript. Phase 26 converts padded held-out policy outputs into
manuscript-facing B0/B1 summaries. Phase 28 adds representation controls.
Phase 33 tests whether a modestly higher budget rescues normalized-B1 behavior.
Phase 36 evaluates suitability-proxy signal against available weak labels.
Phase 38 rebuilds suitability proxies under leakage-aware controls.
Phase 39 audits available label-like columns. Phase 40 adds the hard
independent-label go/no-go gate before any Phase 38 rerun or B2/B3 reward
smoke.

## 4. Experiments

### 4.1 Held-Out B0/B1 Learned-Policy Analysis

The main B0/B1 held-out analysis trains on `tile_r003_c003` and evaluates on
held-out Bishan tiles `tile_r002_c003`, `tile_r005_c004`, and
`tile_r005_c003` across seeds `0`, `1`, and `2`. At 4096 training steps and
evaluation horizon `8`, Phase 26 reports a learned-policy B1-B0 mean reward
delta of `-0.1318712688`. Only `3 / 9` tile-seed pairs favor B1.

This result does not support a stable claim that raw GeoFM-enhanced B1
outperforms explicit-feature B0 under the deterministic base planning reward.

### 4.2 Representation-Control Results

Phase 28 compares B1 against B0, D2, D3, D4P8, and D4P16 under the same padded
held-out protocol at 1024 and 4096 training steps. Both runs report
`compression_matches_raw` rather than `representation_signal_supported`.

At 4096 steps, the B1-comparator mean deltas are:

| Comparison | B1 minus comparator mean delta | Positive tile-seed count |
|---|---:|---:|
| B1 - B0 | `-0.1318712688` | `3 / 9` |
| B1 - D2 | `0.0744591656` | `4 / 9` |
| B1 - D3 | `-0.1094750135` | `3 / 9` |
| B1 - D4P8 | `-0.3768518347` | `2 / 9` |
| B1 - D4P16 | `-0.6411940236` | `2 / 9` |

The compressed controls are central to the interpretation. D4P8 and D4P16
exceed raw B1 at 4096 steps, so the current evidence is more consistent with
representation compression or optimization effects than with a robust semantic
advantage of raw 64-dimensional GeoFM embeddings.

### 4.3 Normalization and Budget Robustness

Phase 30 tests whether normalizing B1 embeddings repairs raw-B1 underperformance.
At 4096 steps, normalized variants improve raw B1 and recover the mean B0 gap:
`N1Z` achieves mean learned-policy reward `0.6515323140`, compared with
`0.4825072170` for B0 and `0.3506359482` for raw B1. However, normalized
variants do not consistently exceed compressed controls.

Phase 33 extends this question with bounded 5120-step matched pilots across
three held-out tiles and three seeds. The full aggregate reports
`budget_not_explanatory`. At 5120 steps, tracked focal gaps remain negative on
mean, including `N1Z - D4P16 = -0.7597285327`, `N1Z - D4P8 = -0.4953863438`,
`N1ZR - D4P16 = -0.6658915329`, and `N1ZR - D4P8 = -0.4015493440`.

This result prevents the manuscript from claiming that a modestly higher
training budget or simple normalization resolves the representation-control
problem.

### 4.4 Suitability-Proxy and Independent-Label Gates

Phase 36 evaluates whether GeoFM-derived feature families add weak-label
suitability signal beyond explicit planning features. The available labels are
`current_farmland_label`, `farmland_or_orchard_label`, and
`low_slope_farmland_label`, with `64,984` valid labels each. All three labels
are DLTB/slope-derived and flagged as `explicit_label_leakage_risk`.

Because explicit planning features encode the weak-label logic, `explicit_only`
models reach ROC AUC, average precision, and balanced accuracy of `1.0` for all
three labels. This perfect performance is a leakage warning, not suitability
validation. GeoFM-only checks are weak: for example, raw GeoFM-only features
reach ROC AUC `0.6490064144` for `low_slope_farmland_label`, but the scalar
`suitability_proxy` is near random for that label with ROC AUC `0.4979564572`.

Phase 38 rebuilds proxy classifiers but remains
`proxy_rebuild_diagnostic_only` because evaluated labels are leakage risks or
GeoFM-derived proxies do not clear control thresholds. Phase 39 reports
`independent_label_inputs_missing` after auditing the real Phase 2 table. Phase
40 then formalizes the decision as a hard gate: without a registered
independent non-leakage label, Phase 38 cannot be rerun as a stronger proxy
validation, and B2/B3 suitability-reward experiments should not proceed. The
current Phase 40 no-registry run reads `64,984` feature rows, `0` registry
rows, and reports `independent_label_inputs_missing`.

## 5. Discussion

The current Paper11 evidence settles the current scientific claim negatively. The project began
with the question of whether GeoFM embeddings could improve farmland layout
optimization by adding latent environmental context. The current bounded
evidence does not support that positive claim. Raw B1 does not stably
outperform B0 across held-out Bishan tiles and seeds, compressed controls can
exceed raw B1, and normalized B1 does not survive broader robustness checks as
a general rescue.

This negative result is informative rather than incidental. It shows that
remote-sensing foundation-model embeddings cannot be treated as decision-ready
suitability variables merely because they are semantically rich. In the current
planning environment, representation scale, compression, spatial alignment, and
reward definition all shape learned-policy behavior. Without controls, a
positive result could be misattributed to GeoFM semantics.

The suitability branch is even more constrained. The current weak labels are
derived from DLTB, slope, or source metadata, so they cannot validate an
independent suitability reward. Phase 40 therefore performs an important
methodological function: it prevents the workflow from converting weak,
leakage-prone labels into B2/B3 reward claims. This gate is the main practical
outcome of the current revision.

The negative conclusion remains bounded. It is Bishan-only, uses bounded training budgets,
does not provide cross-region transfer evidence, does not enable B2/B3, and
does not validate agronomic suitability with independent field, soil,
irrigation, yield, or high-standard-farmland labels. Phase 41 defines a future
route for GeoFM as a calibrated suitability prior, but the current real run
cannot produce that prior because Phase 40 has no accepted independent label
registry. These limitations are not minor additions to a positive story. They
define the boundary of the current paper.

## 6. Conclusion

The current Paper11 repository establishes a reproducible workflow for testing
GeoFM-enhanced farmland layout optimization on real planning units, and the
completed evidence rejects the current positive Paper11 hypothesis. Under the
current Bishan held-out protocol, raw GeoFM B1 is not a stable positive
learned-policy signal; compressed and normalized controls reveal unresolved
representation effects; and suitability-reward work must remain blocked until
an independent non-leakage label registry passes Phase 40 and a leakage-aware
proxy rebuild clears subsequent controls.

## Claim-Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| The repository implements a reproducible GeoFM-enhanced farmland planning workflow. | Phase 1-25 pipeline, real Bishan DLTB adapter, tiled contracts, padded held-out policy runner, smoke tests. | Supported |
| Raw GeoFM B1 improves learned-policy planning decisions. | Phase 26 B1-B0 mean delta `-0.1318712688`, `3 / 9` positive tile-seed pairs. | Not supported |
| Raw B1 carries a stable representation advantage over controls. | Phase 28 reports `compression_matches_raw`; D4P8 and D4P16 exceed B1 at 4096 steps. | Not supported |
| Normalization or more budget resolves the representation problem. | Phase 30 partially improves B1; Phase 33 full bounded aggregate reports `budget_not_explanatory`. | Not supported |
| Suitability reward is ready for B2/B3. | Phase 36 `proxy_signal_not_supported`; Phase 38 `proxy_rebuild_diagnostic_only`; Phase 40 `independent_label_inputs_missing`. | Not supported |
| The current paper can be submitted as a positive performance manuscript. | Performance, suitability, B2/B3, and transfer claims remain unsupported. | Not supported |
| The current paper can make a conclusion-level negative claim. | B1 underperforms or fails controls, normalized/budget checks do not rescue the claim, and suitability reward is blocked by Phase 40. | Supported |

## Data Availability Draft

The repository includes lightweight Bishan AlphaEarth sample arrays, code,
tests, reproduction guides, and file manifests for reviewer smoke checks. Full
real Bishan DLTB-with-slope inputs are external local data and are not
redistributed in ordinary Git. Large derived arrays, full generated outputs,
and trained artifacts should be deposited in an external archive before any
submission that relies on them.

## Code Availability Draft

All code for the reviewer-facing workflow is maintained in the Paper11
repository:

```text
https://github.com/zhouning/paper11-geofm-farmland-suitability-rl
```

The final submitted version should cite a release tag or immutable commit hash.

## Author Input Needed Before Journal Submission

- Final author list, affiliations, and corresponding-author details.
- Funding statement.
- Author contributions.
- Final data access statement for the external DLTB-with-slope GeoPackage.
- Reference list and citation formatting for the selected journal.
- Decision on whether to submit this as a negative-results research article now or hold the paper until independent labels and B2/B3 results can support a different conclusion.
