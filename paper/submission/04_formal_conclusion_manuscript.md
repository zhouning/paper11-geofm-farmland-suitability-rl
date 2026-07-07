# Evidence-Gated GeoFM Representation for Farmland Layout Optimization

## Title

Compressed GeoFM representations improve held-out farmland layout optimization under evidence gates

## Short Title

Compressed GeoFM planning

## Keywords

Geospatial foundation model; AlphaEarth embeddings; farmland layout
optimization; reinforcement learning; suitability proxy; representation
control; evidence gate; land-use planning

## Highlights

- A reproducible workflow links AlphaEarth embeddings to real Bishan DLTB planning blocks.
- Tiled and padded policy interfaces make large real planning instances testable.
- Multi-tile B0/B1 evidence rejects raw 64-dimensional GeoFM direct injection.
- Phase 48 shows that compressed GeoFM routes D4P8 and D4P16 outperform B0, raw B1, random D2, and shuffled D3 on mean held-out reward.
- Phase 52 expands this compressed-route evidence to five held-out tiles and three seeds.
- Phase 53 supports the expanded cluster mean with exact sign-flip, bootstrap, and leave-one influence checks.
- Phase 54 verifies that the formal Phase 52/53 values come from one authoritative artifact chain.
- Suitability-reward work remains blocked until an independent non-leakage label gate passes.
- A local Phase 42 label-source audit finds no usable external suitability label for the real Bishan run.
- The evidence supports a bounded positive compressed-representation conclusion, not a raw-GeoFM or suitability-reward claim.
## Abstract

Farmland spatial layout optimization requires decisions over many heterogeneous
planning units, but direct variables for soil quality, irrigation,
productivity, and long-term suitability are often unavailable at operational
planning scale. Geospatial foundation-model (GeoFM) embeddings offer a possible
source of latent land-surface information, but their usefulness for
reinforcement-learning planning depends on representation design, reward
validation, and leakage-aware labels. This study builds a reproducible
evidence-gated workflow that aggregates AlphaEarth embeddings and explicit
planning features to real Bishan DLTB land-use blocks, constructs tiled and
padded planning interfaces, and evaluates GeoFM-enhanced representations under
a deterministic base planning reward. Raw 64-dimensional GeoFM direct injection
is not supported: at 4096 training steps, B1 has a learned-policy B1-B0 mean
reward delta of `-0.1318712688`, with only `3 / 9` held-out tile-seed pairs
favoring B1. Phase 48 changes the broader conclusion by re-evaluating the
PCA-compressed GeoFM variants as candidate state routes. D4P8 and D4P16 exceed
B0, raw B1, random D2, and shuffled D3 on mean learned-policy reward; the
pooled compressed-control delta is `0.4673011499`, with `48 / 72` positive
comparisons. Phase 49 further reports `compressed_route_statistically_robust`,
with one-sided sign-test p `0.0031549137` and bootstrap CI95
`[0.2827829983, 0.6639974489]`. Phase 50 aggregates the same evidence to
tile-seed clusters and reports directional support (`7 / 9` positive clusters,
p `0.08984375`), so the sign-only cluster test is underpowered. Phase 51 then
uses an exact signed-rank test over the cluster mean deltas and supports the
magnitude-sensitive cluster effect (positive rank sum `40 / 45`, p
`0.01953125`). Phase 52 expands the six-variant replication to five held-out
tiles and three seeds; it again reports `compressed_geofm_route_supported`, all
eight compressed-versus-control mean deltas positive, pooled delta
`0.2921767818` with `74 / 120` positive comparisons, row-level sign-test p
`0.0066881634`, cluster sign-only p `0.1508789062`, and cluster signed-rank p
`0.0206298828`. Phase 53 adds direct cluster-mean support: exact one-sided
sign-flip mean p `0.0196838379`, bootstrap CI95 `[0.0570820445, 0.5823557658]`,
and positive leave-one cluster, tile, and seed means. Phase 54 reports
`artifact_lineage_consistent`, confirming that the formal Phase 52/53 values
are reproducible from one authoritative delta-to-cluster-to-statistics artifact
chain. D4P16 reaches mean reward `0.9918299718` in the original three-tile Phase
48 audit and `0.5819662325` in the expanded Phase 52 audit.
Normalized-B1 and budget
checks do not displace the compressed route: Phase 30 improves raw B1 but
remains below D4P8/D4P16, and Phase 33 reports `budget_not_explanatory` for the
normalized branch. Suitability-reward evidence remains blocked: weak labels are
DLTB/slope-derived leakage risks, the scalar `suitability_proxy` is not
supported as a reward term, and Phase 40/41 report missing independent label
inputs. These results support a bounded positive conclusion for compressed
GeoFM state representations under the current Bishan base-reward held-out
protocol, while raw direct injection, B2/B3 suitability reward, and
cross-region transfer claims remain unsupported.
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

The current evidence refines rather than rejects the original positive framing. Raw GeoFM direct injection is not a supported improvement, and suitability reward must not be used without independent labels. However, Phase 48 shows that compressed GeoFM state routes are supported under the current base-reward held-out protocol. This paper therefore frames Paper11 as a bounded positive compressed-representation result with explicit evidence gates around raw B1, suitability reward, and transfer claims.

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
policy interface operate on real planning units? Second, does GeoFM information
improve learned-policy decisions under the same deterministic base reward, and
which state representation makes that information usable? Third, is there
validated suitability evidence strong enough to justify adding a suitability
reward? The current study answers the first question positively, the second
positively only for compressed GeoFM state routes, and the third negatively
under present inputs.
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
Phase 48 re-evaluates D4P8 and D4P16 as compressed GeoFM candidate routes under
the same held-out base-reward protocol. Phase 49-54 then test statistical
robustness, tile-seed clustering, signed-rank cluster magnitude support, an
expanded five-tile, three-seed replication, cluster-mean influence support, and
artifact-lineage consistency for the same compressed route. Phase 36 evaluates suitability-proxy
signal against available weak labels. Phase 38 rebuilds suitability proxies
under leakage-aware controls. Phase 39 audits available label-like columns.
Phase 40 adds the hard independent-label go/no-go gate before any Phase 38
rerun or B2/B3 reward smoke. Phase 41 tests a separate suitability-prior route
in which GeoFM can enter reward design only after independent-label calibration.
Phase 42 audits local candidate label sources to distinguish true external
labels from diagnostic DLTB/slope weak labels and unrelated labels from other
projects.
## 4. Experiments

### 4.1 Held-Out B0/B1 Learned-Policy Analysis

The main B0/B1 held-out analysis trains on `tile_r003_c003` and evaluates on
held-out Bishan tiles `tile_r002_c003`, `tile_r005_c004`, and
`tile_r005_c003` across seeds `0`, `1`, and `2`. At 4096 training steps and
evaluation horizon `8`, Phase 26 reports a learned-policy B1-B0 mean reward
delta of `-0.1318712688`. Only `3 / 9` tile-seed pairs favor B1.

This result does not support a stable claim that raw GeoFM-enhanced B1
outperforms explicit-feature B0 under the deterministic base planning reward.

### 4.2 Raw and Compressed Representation-Control Results

Phase 28 first compares B1 against B0, D2, D3, D4P8, and D4P16 under the same
padded held-out protocol at 1024 and 4096 training steps. Both runs report
`compression_matches_raw` rather than `representation_signal_supported` for raw
B1. At 4096 steps, the B1-comparator mean deltas are:

| Comparison | B1 minus comparator mean delta | Positive tile-seed count |
|---|---:|---:|
| B1 - B0 | `-0.1318712688` | `3 / 9` |
| B1 - D2 | `0.0744591656` | `4 / 9` |
| B1 - D3 | `-0.1094750135` | `3 / 9` |
| B1 - D4P8 | `-0.3768518347` | `2 / 9` |
| B1 - D4P16 | `-0.6411940236` | `2 / 9` |

This result rejects raw 64-dimensional GeoFM direct injection as the effective
state route. It does not reject GeoFM information itself, because the compressed
GeoFM variants were the strongest variants in the same held-out summary set.

Phase 48 therefore re-evaluates D4P8 and D4P16 as compressed GeoFM candidate
routes. The audit is read-only over the frozen 4096-step Phase 28 summary CSV
and compares each compressed candidate against B0, raw B1, random D2, and
shuffled D3 on the same tile-seed pairs.

| Comparison | Compressed minus comparator mean delta | Positive tile-seed count |
|---|---:|---:|
| D4P8 - B0 | `0.2449805659` | `4 / 9` |
| D4P8 - B1 | `0.3768518347` | `7 / 9` |
| D4P8 - D2 | `0.4513110003` | `7 / 9` |
| D4P8 - D3 | `0.2673768211` | `6 / 9` |
| D4P16 - B0 | `0.5093227548` | `5 / 9` |
| D4P16 - B1 | `0.6411940236` | `7 / 9` |
| D4P16 - D2 | `0.7156531892` | `5 / 9` |
| D4P16 - D3 | `0.5317190100` | `7 / 9` |

Phase 48 reports `compressed_geofm_route_supported`. The pooled
compressed-control delta is `0.4673011499`, with `48 / 72` positive
comparisons. The supported representation claim is therefore compressed and
bounded: D4P8 and D4P16 improve mean learned-policy reward under the current
Bishan base-reward held-out protocol, while raw B1 remains unsupported.
Phase 49 strengthens this result with a pooled one-sided sign-test p of
`0.0031549137`, bootstrap CI95 `[0.2827829983, 0.6639974489]`, and positive
leave-one-tile and leave-one-seed sensitivity checks. Phase 50 then aggregates
to `9` tile-seed clusters and reports directional support (`7 / 9` positive,
p `0.08984375`), which narrows sign-test wording. Phase 51 uses exact
signed-rank evidence over cluster magnitudes and supports the cluster-level
effect with p `0.01953125`.

Phase 52 expands this replication to five held-out tiles and three seeds while
keeping the same six variants, `4096` training steps, and evaluation horizon
`8`. The expanded run again reports `compressed_geofm_route_supported`: all
eight compressed-versus-control mean deltas remain positive, and the pooled
compressed-control delta is `0.2921767818` with `74 / 120` positive row-level
comparisons.

| Expanded Phase 52 comparison | Compressed minus comparator mean delta | Positive tile-seed count |
|---|---:|---:|
| D4P8 - B0 | `0.2896842037` | `9 / 15` |
| D4P8 - B1 | `0.2050431914` | `9 / 15` |
| D4P8 - D2 | `0.2506866266` | `10 / 15` |
| D4P8 - D3 | `0.1973780839` | `10 / 15` |
| D4P16 - B0 | `0.4026417146` | `8 / 15` |
| D4P16 - B1 | `0.3180007023` | `10 / 15` |
| D4P16 - D2 | `0.3636441375` | `7 / 15` |
| D4P16 - D3 | `0.3103355948` | `11 / 15` |

The expanded Phase 49-style robustness check remains
`compressed_route_statistically_robust`, with pooled row-level sign-test p
`0.0066881634`, bootstrap CI95 `[0.1623326461, 0.4323997354]`, and positive
leave-one-tile and leave-one-seed means. The expanded tile-seed sign-only test
remains directional (`10 / 15` positive clusters, p `0.1508789062`), but the
exact signed-rank test over cluster magnitudes remains significant (positive
rank sum `96 / 120`, p `0.0206298828`).

Phase 53 audits whether the expanded cluster mean is driven only by one or two
large positive clusters. The status is `cluster_mean_support`: the cluster mean
delta remains `0.2921767818`, the exact one-sided sign-flip mean p is
`0.0196838379`, the bootstrap CI95 is `[0.0570820445, 0.5823557658]`, and the
minimum leave-one-cluster, leave-one-tile, and leave-one-seed means are
`0.2060081575`, `0.0954244478`, and `0.2083797951`, respectively.

Phase 54 then verifies the artifact lineage for the formal Phase 52/53 evidence
chain. It reports `artifact_lineage_consistent`: recomputing Phase 50 cluster
means from the authoritative Phase 48 delta table and recomputing Phase 51 and
Phase 53 statistics from the authoritative cluster CSV reproduces the values
used in this manuscript.

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

### 4.4 Suitability-Proxy, Prior, and Independent-Label Gates

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

Phase 41 tests a separate suitability route for GeoFM: after the compressed
state route has been evaluated under the base reward, any suitability-prior
export still requires independent-label calibration and must pass
explicit-baseline, shuffled-control, random-control, fold-stability, and
calibration checks. The current real no-registry run reports
`phase41_independent_label_inputs_missing`, reads `64,984` feature rows, finds
`0` Phase-40-passed labels, and produces no
`block_geofm_suitability_prior.csv`.

Phase 42 audits local candidate label sources after Phase 41. DLTB/slope labels
can be registered only as diagnostic leakage-risk labels: Phase 40 reports
`independent_label_gate_diagnostic_only`, and Phase 41 still reports
`phase41_independent_label_inputs_missing` with the same diagnostic registry.
No local external soil, irrigation, yield, productivity, high-standard-farmland,
field-survey, or policy-outcome label was found that can pass Phase 40.

## 5. Discussion

The current Paper11 evidence supports a bounded positive representation claim.
The project began with the question of whether GeoFM embeddings could improve
farmland layout optimization by adding latent environmental context. The answer
is conditional. Raw 64-dimensional B1 direct injection does not stably
outperform B0 across held-out Bishan tiles and seeds. However, Phase 48 shows
that compressed GeoFM state routes D4P8 and D4P16 outperform B0, raw B1,
random D2, and shuffled D3 on mean learned-policy reward under the same
base-reward held-out protocol. Phase 49 shows row-level statistical robustness within the current Bishan
protocol, while Phase 50 shows directional support after conservative tile-seed
cluster aggregation; Phase 51 adds exact signed-rank cluster magnitude support.
Phase 52 then expands the same six-variant protocol to five held-out tiles and
three seeds and preserves the same conclusion: all compressed-versus-control
mean deltas remain positive, row-level evidence is statistically robust, and
cluster magnitude remains significant even though cluster signs alone remain
underpowered. Phase 53 adds that the expanded cluster mean itself is supported
by exact sign-flip, bootstrap, and leave-one influence checks, so the positive
compressed-route conclusion is not driven only by one favorable cluster, tile,
or seed. Phase 54 verifies that these formal Phase 52/53 values are internally
reproducible from one authoritative artifact chain, preventing mixed-output
lineage from supporting the manuscript claim.

This distinction matters scientifically. A negative raw-B1 result could have
been misread as evidence that GeoFM information is irrelevant to farmland
layout optimization. The compressed-route result instead points to
representation geometry as the limiting factor. In the current planning
environment, GeoFM information appears useful when it enters the policy state
through compact PCA-compressed inputs, but not when raw low-variance embedding
dimensions are appended directly to explicit planning features.

The result remains bounded. The Phase 48 audit is read-only over the existing
4096-step Bishan held-out summary rows, and Phase 52 expands the same protocol
within Bishan rather than testing a new region. These analyses do not prove
that PCA is intrinsically optimal, that the effect transfers across regions, or
that the policy has learned agronomic suitability semantics. The positive claim
is about state representation under a deterministic base planning reward, not
about a validated suitability reward.

The suitability branch is still more constrained than a simple missing
experiment. The current weak labels are derived from DLTB, slope, or source
metadata, so they cannot validate an independent suitability reward. Phase 40
prevents the workflow from converting weak, leakage-prone labels into B2/B3
reward claims. Phase 41 defines a stricter future route in which GeoFM must be
converted into a calibrated low-dimensional prior before it can influence
suitability reward. Phase 42 then tests whether the required labels are already
available locally and concludes that they are not: DLTB/slope labels remain
diagnostic-only, and candidate labels from Paper10 and Paper58 are not Paper11
Bishan suitability labels.

The practical implication is that Paper11 should proceed with two separate
claim levels. The base-reward representation claim is now supported for
compressed GeoFM state routes in Bishan. The suitability-reward claim remains
blocked until external non-leakage labels pass the independent-label gate and a
leakage-aware GeoFM prior clears control and calibration checks.
## 6. Conclusion

The current Paper11 repository establishes a reproducible workflow for testing
GeoFM-enhanced farmland layout optimization on real planning units. The
completed evidence supports a compressed GeoFM representation route under the
Bishan base-reward held-out protocol. Raw B1 direct injection remains
unsupported, but D4P8 and D4P16 improve over B0, raw B1, random D2, and
shuffled D3 on mean learned-policy reward, Phase 49 robustness checks keep
the pooled effect positive, and Phase 50 keeps the cluster-level direction
positive with sign-only p `0.08984375`; Phase 51 supports the cluster
magnitude effect with signed-rank p `0.01953125`. Phase 52 expands the same six-variant
test to five held-out tiles and three seeds, again supporting the compressed
route with pooled delta `0.2921767818`, `74 / 120` positive row-level
comparisons, row-level sign-test p `0.0066881634`, and cluster signed-rank p
`0.0206298828`. Phase 53 further supports the expanded cluster mean with exact
sign-flip p `0.0196838379`, bootstrap CI95 `[0.0570820445, 0.5823557658]`, and
positive leave-one cluster, tile, and seed means. Phase 54 verifies the formal
artifact lineage as `artifact_lineage_consistent`. The appropriate current
conclusion is therefore not that GeoFM fails, but that GeoFM must be represented through a controlled compressed state
route before it improves the learned planning policy in this setting.
Suitability-reward work remains blocked until an
independent non-leakage label registry passes Phase 40 and a leakage-aware
GeoFM prior passes Phase 41.
## Claim-Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| The repository implements a reproducible GeoFM-enhanced farmland planning workflow. | Phase 1-25 pipeline, real Bishan DLTB adapter, tiled contracts, padded held-out policy runner, smoke tests. | Supported |
| Raw GeoFM B1 improves learned-policy planning decisions. | Phase 26 B1-B0 mean delta `-0.1318712688`, `3 / 9` positive tile-seed pairs. | Not supported |
| Raw B1 carries a stable representation advantage over controls. | Phase 28 reports `compression_matches_raw`; D4P8 and D4P16 exceed B1 at 4096 steps. | Not supported |
| Compressed GeoFM state routes improve learned-policy planning decisions under the base reward. | Phase 48 reports `compressed_geofm_route_supported`; pooled compressed-control delta `0.4673011499`, `48 / 72` positive comparisons. Phase 52 expands the six-variant run to five held-out tiles and three seeds with pooled delta `0.2921767818` and `74 / 120` positive comparisons. | Supported |
| The compressed GeoFM state-route effect is row-level statistically robust within the current Bishan protocol. | Phase 49 reports `compressed_route_statistically_robust`; sign-test p `0.0031549137`; bootstrap CI95 `[0.2827829983, 0.6639974489]`; leave-one tile/seed means remain positive. Phase 52 repeats row-level robustness with sign-test p `0.0066881634` and bootstrap CI95 `[0.1623326461, 0.4323997354]`. | Supported |
| The compressed GeoFM route is supported after tile-seed cluster aggregation when magnitude and mean support are considered. | Phase 50 reports sign-only `cluster_directional_support`; Phase 51 reports `cluster_magnitude_support`, positive rank sum `40 / 45`, signed-rank p `0.01953125`. Phase 52 keeps sign-only cluster support directional (`10 / 15`, p `0.1508789062`) but supports cluster magnitude with positive rank sum `96 / 120`, signed-rank p `0.0206298828`. Phase 53 reports `cluster_mean_support`, exact sign-flip p `0.0196838379`, bootstrap CI95 `[0.0570820445, 0.5823557658]`, and positive leave-one cluster/tile/seed means. | Supported with magnitude-sensitive and cluster-mean checks |
| The formal Phase 52/53 artifact chain is internally reproducible. | Phase 54 reports `artifact_lineage_consistent`: Phase 50 cluster means recomputed from the authoritative Phase 48 delta table, and Phase 51/53 statistics recomputed from the authoritative cluster CSV, match the formal manuscript values. | Supported |
| Normalization or more budget resolves the raw-B1 representation problem. | Phase 30 partially improves B1; Phase 33 full bounded aggregate reports `budget_not_explanatory`. | Not supported |
| Suitability reward is ready for B2/B3. | Phase 36 `proxy_signal_not_supported`; Phase 38 `proxy_rebuild_diagnostic_only`; Phase 40 `independent_label_inputs_missing`; Phase 41 `phase41_independent_label_inputs_missing`. | Not supported |
| Local files already contain a usable independent Paper11 suitability label. | Phase 42 finds only diagnostic DLTB/slope weak labels and unrelated Paper10/Paper58 labels. | Not supported |
| The evidence supports a broad GeoFM-performance or transfer conclusion. | Suitability, B2/B3, and transfer claims remain unsupported. | Not supported |
| The evidence supports a bounded positive compressed-GeoFM representation conclusion. | D4P8/D4P16 exceed B0, raw B1, D2, and D3 on mean held-out base-reward policy reward; Phase 53 supports the expanded cluster mean and influence checks. | Supported |
## Data Availability

The repository includes lightweight Bishan AlphaEarth sample arrays, code,
tests, reproduction guides, and file manifests for reviewer smoke checks. Full
real Bishan DLTB-with-slope inputs are external local data and are not
redistributed in ordinary Git. Large derived arrays, full generated outputs,
and trained artifacts should be deposited in an external archive before any
submission that relies on them.

## Code Availability

All code for the reviewer-facing workflow is maintained in the Paper11
repository:

```text
https://github.com/zhouning/paper11-geofm-farmland-suitability-rl
```

The final submitted version should cite a release tag or immutable commit hash.

## Submission Metadata Still Required

- Final author list, affiliations, and corresponding-author details.
- Funding statement.
- Author contributions.
- Final data access statement for the external DLTB-with-slope GeoPackage.
- Reference list and citation formatting for the selected journal.
- Final decision on target journal format for a bounded compressed-GeoFM representation article.
