---
title: "Compressed geospatial foundation-model representations improve held-out farmland layout optimization under evidence gates"
---

**Short title:** Compressed GeoFM planning

**Keywords:** Geospatial foundation model; AlphaEarth embeddings; farmland layout optimization; reinforcement learning; representation control; evidence gate; land-use planning; Bishan

# Abstract

Farmland layout optimization increasingly depends on spatial decisions for many heterogeneous planning units, yet direct measurements of soil quality, irrigation, productivity, and long-term suitability are often incomplete at operational planning scale. Geospatial foundation-model (GeoFM) embeddings provide a possible source of latent land-surface information, but their value for reinforcement-learning-based planning depends on how the embeddings enter the policy state and whether stronger suitability claims are protected from label leakage. We built an evidence-gated workflow that aggregates AlphaEarth embeddings and explicit planning attributes to real Bishan DLTB land-use blocks, constructs tiled and padded policy interfaces, and evaluates GeoFM-enhanced state representations under a deterministic base planning reward. Raw 64-dimensional GeoFM injection did not improve held-out learned-policy reward: B1 had a B1-B0 mean reward delta of `-0.1318712688`, with only `3 / 9` held-out tile-seed pairs favoring B1. In contrast, PCA-compressed GeoFM routes were consistently stronger. In the original three-tile audit, D4P8 and D4P16 exceeded B0, raw B1, random D2, and shuffled D3 on mean reward, with pooled compressed-control delta `0.4673011499` and `48 / 72` positive comparisons. In an expanded five-tile, three-seed replication, all eight compressed-versus-control mean deltas remained positive; the pooled delta was `0.2921767818`, with `74 / 120` positive row-level comparisons, one-sided sign-test p `0.0066881634`, 95% bootstrap CI `[0.1623326461, 0.4323997354]`, cluster signed-rank p `0.0206298828`, and exact cluster-mean sign-flip p `0.0196838379`. A representation-geometry audit aligned all `64,984` blocks and showed that D4P8 and D4P16 retained `85.87823898%` and `94.96006154%` of raw GeoFM variance while reducing effective rank from `9.4947211626` to `5.1322783588` and `7.3009059917`, respectively, and reducing the covariance condition number from `6658.9542931381` to `16.2704676982` and `53.6978527088`. Artifact-lineage checks reproduced the formal expanded-replication values from one authoritative delta-to-cluster-to-statistics chain. Suitability-reward claims remain unsupported because available weak labels are DLTB/slope-derived leakage risks and no independent Bishan suitability label passed the label gate. These results support a bounded conclusion: GeoFM information improved farmland layout optimization when represented as controlled compressed state features under the Bishan base-reward protocol, but raw injection, B2/B3 suitability reward, and transfer claims remain unproven.

# 1. Introduction

Farmland layout optimization is a spatial decision problem with direct consequences for land consolidation, fragmentation control, and agricultural protection. Planning units differ in area, slope, land-use class, adjacency, and geometric context, but they may also differ in latent environmental properties that are not fully recorded in operational planning data. This creates a recurring modelling gap: explicit GIS-derived attributes can define a feasible planning task, yet they may not capture all surface conditions relevant to the quality of a layout decision.

Geospatial foundation models offer a plausible route for filling part of this gap. Products such as AlphaEarth embeddings summarize multi-sensor land-surface information in dense feature vectors and can be joined to planning units after spatial aggregation. However, adding such embeddings to a planning policy is not automatically beneficial. Dense remote-sensing embeddings may be redundant with explicit variables, poorly scaled for policy optimization, vulnerable to shuffled or random controls, or incorrectly interpreted as agronomic suitability when no independent suitability label is available.

This distinction is central to the present study. We do not ask only whether GeoFM features can be appended to a reinforcement-learning planning state. We ask which representation route is supported by held-out policy reward, which controls rule out simpler explanations, and where the evidence must stop. The design therefore separates three claim levels: first, whether a reproducible Bishan planning workflow can operate on real planning units; second, whether GeoFM state information improves learned-policy reward under the same deterministic base reward; and third, whether the evidence is strong enough to introduce a suitability reward.

We show that the answer is conditional rather than uniformly positive or negative. Raw 64-dimensional GeoFM injection was not supported. PCA-compressed GeoFM representations, however, improved held-out learned-policy reward over explicit-only, raw GeoFM, random-control, and shuffled-control state variants. The positive result survived an expanded five-tile replication, row-level robustness tests, magnitude-sensitive cluster testing, direct cluster-mean support, and artifact-lineage verification. At the same time, suitability-reward variants remain blocked because the available weak labels are leakage-prone and no independent Bishan suitability label has been identified. The contribution is therefore a bounded positive representation result, not a broad claim that GeoFM directly validates suitability or transfers beyond the current setting.

# 2. Materials and Methods

## 2.1 Study units and feature sources

The workflow used real Bishan DLTB land-use polygons as planning units. The real-data adapter exported `64,984` DLTB polygons into analysis-ready block feature tables. Explicit planning attributes included available slope- and land-use-derived variables, while GeoFM features were obtained by aggregating AlphaEarth annual embeddings to block-level representations.

The repository separates lightweight reproducibility from full local reconstruction. Lightweight sample AlphaEarth arrays, tests, file manifests, and manuscript-facing summaries are included for reviewer smoke checks. The full Bishan DLTB-with-slope GeoPackage is treated as an external local input and is not redistributed through ordinary Git. Large derived arrays, full generated experiment outputs, and trained artifacts are excluded from Git and should be archived externally for submission-grade data release.

## 2.2 State representations

We evaluated explicit-only, raw GeoFM, control, compressed, and diagnostic state variants under the same planning interface. `B0` used explicit planning features with the deterministic base planning reward. `B1` appended the raw 64-dimensional AlphaEarth/GeoFM embedding to the explicit features. `D2` appended random 64-dimensional controls, and `D3` appended shuffled AlphaEarth embeddings. `D4P8` and `D4P16` appended PCA-compressed AlphaEarth embeddings with 8 and 16 components, respectively. `N1Z` and `N1ZR` were normalized-B1 diagnostic variants used to test whether raw-B1 behavior was explained by feature scale. `B2` and `B3` were reserved for suitability-reward experiments and were not enabled because the suitability evidence gates did not pass.

This representation ladder was designed to distinguish information content from representation form. If raw B1 failed while random and shuffled controls performed similarly, there would be no supported GeoFM state claim. If compressed GeoFM variants exceeded B0, B1, D2, and D3 under the same reward and evaluation protocol, the supported claim would be narrower: GeoFM information is useful only after controlled compression.

## 2.3 Planning reward and policy interface

All supported representation experiments used a deterministic base planning reward. This reward encoded explicit planning logic, including slope, contiguity, baimu-fang-style consolidation, action validity, and related spatial constraints. It deliberately excluded any suitability reward term because the available suitability proxies had not passed independent-label validation.

The learned-policy interface used tiled planning contracts and padded variable-size observations/actions so that policies could be evaluated across held-out Bishan tiles with different numbers of planning units. The original B0/B1 and representation-control experiments used three held-out tiles and seeds `0`, `1`, and `2`; the expanded replication used five held-out tiles and the same three seeds. Reported learned-policy comparisons used `4096` training steps and evaluation horizon `8` unless otherwise stated.

## 2.4 Evidence gates and statistical checks

The manuscript-facing evidence gates were organized around increasingly strong claim thresholds. Raw B1 was first compared with B0 under held-out learned-policy reward. Representation controls then compared raw B1 and compressed GeoFM routes against explicit-only, random-control, and shuffled-control variants. Normalized-B1 and bounded higher-budget checks tested whether the raw-B1 result was primarily an optimization-scale problem.

For the compressed route, row-level support was assessed through compressed-minus-control deltas, positive comparison counts, one-sided sign tests, 95% bootstrap CI intervals, and leave-one tile/seed sensitivity. Because row-level comparisons share tile and seed contexts, cluster-level checks aggregated deltas to tile-seed clusters. The cluster evidence included a conservative sign-only test, an exact signed-rank test over cluster magnitudes, and a direct cluster-mean sign-flip and bootstrap audit. A final artifact-lineage audit recomputed the formal cluster values from the authoritative delta and cluster CSV files to prevent mixed generated directories from supporting the manuscript claim.

To test why the compressed route worked while raw B1 did not, a read-only representation-geometry audit aligned the B1, D4P8, and D4P16 feature tables by `block_id`. The audit computed total centered variance, covariance eigenvalues, effective rank, participation ratio, positive-eigenvalue condition number, and feature-standard-deviation spread for the raw and compressed GeoFM coordinates. It then linked these geometry summaries to the expanded compressed-minus-control reward deltas and to held-out tile memberships. This audit was diagnostic: it could support a plausible mechanism for the compressed representation result, but it did not add a new policy-training result or prove that PCA was the optimal compression method.

Suitability-reward evidence was handled by a separate hard gate. Weak labels derived from DLTB, slope, or source metadata were treated as diagnostic leakage-risk labels rather than independent agronomic suitability labels. A B2/B3 suitability reward could proceed only after a registered independent non-leakage label passed the label gate and a leakage-aware GeoFM prior passed explicit-baseline, shuffled-control, random-control, fold-stability, and calibration checks.

# 3. Results

## 3.1 The real Bishan workflow produced held-out policy comparisons but did not support raw GeoFM injection

The workflow successfully connected real Bishan DLTB planning units, block-level AlphaEarth embeddings, tiled planning contracts, and padded held-out policy evaluation. This established the technical basis for comparing GeoFM-enhanced state representations on real planning units rather than only on synthetic smoke tests.

Raw GeoFM injection did not improve learned-policy reward. In the main B0/B1 held-out analysis, the model trained on `tile_r003_c003` and was evaluated on held-out tiles `tile_r002_c003`, `tile_r005_c004`, and `tile_r005_c003` across seeds `0`, `1`, and `2`. At `4096` training steps and evaluation horizon `8`, the learned-policy B1-B0 mean reward delta was `-0.1318712688`, and only `3 / 9` tile-seed pairs favored B1. This result ruled out a simple claim that appending raw 64-dimensional GeoFM embeddings improves the planning policy.

Representation controls confirmed that the raw-B1 route was not the strongest way to use GeoFM information. At `4096` steps, B1 trailed B0, shuffled D3, and both compressed GeoFM variants on mean reward. The B1-minus-comparator results were `-0.1318712688` versus B0, `0.0744591656` versus random D2, `-0.1094750135` versus shuffled D3, `-0.3768518347` versus D4P8, and `-0.6411940236` versus D4P16. The supported interpretation was not that GeoFM information was useless, but that raw direct injection was not an effective state representation.

## 3.2 Compressed GeoFM states improved reward over explicit, raw, random, and shuffled controls

PCA-compressed GeoFM routes changed the result. In the original three-tile audit, both D4P8 and D4P16 were compared against B0, raw B1, random D2, and shuffled D3 on the same tile-seed pairs. All eight compressed-minus-control mean deltas were positive. D4P8 exceeded B0 by `0.2449805659`, B1 by `0.3768518347`, D2 by `0.4513110003`, and D3 by `0.2673768211`. D4P16 exceeded B0 by `0.5093227548`, B1 by `0.6411940236`, D2 by `0.7156531892`, and D3 by `0.5317190100`.

The pooled compressed-control delta in this audit was `0.4673011499`, with `48 / 72` positive row-level comparisons. A row-level robustness audit reported a one-sided sign-test p of `0.0031549137`, 95% bootstrap CI `[0.2827829983, 0.6639974489]`, and positive leave-one-tile and leave-one-seed means. These checks supported the compressed representation route under the current Bishan base-reward protocol while keeping the raw-B1 claim rejected.

Cluster aggregation made the inference more conservative. When the original deltas were aggregated to `9` tile-seed clusters, the sign-only result was directional rather than conventionally significant: `7 / 9` positive clusters, p `0.08984375`. Because the sign-only test ignores magnitude, an exact signed-rank test over cluster mean deltas was also used; it supported the magnitude-sensitive cluster effect with positive rank sum `40 / 45` and p `0.01953125`. The cluster evidence therefore supported the compressed route when magnitude was considered, while preserving the narrower wording required by the conservative sign-only test.

## 3.3 Expanded replication preserved the compressed-route conclusion

The expanded replication tested whether the compressed-route result survived a larger held-out tile set. The same six variants were evaluated over five held-out tiles and three seeds at `4096` training steps and evaluation horizon `8`. Mean learned-policy rewards were `0.1793245179` for B0, `0.2639655302` for B1, `0.2183220949` for D2, `0.2716306377` for D3, `0.4690087215` for D4P8, and `0.5819662325` for D4P16.

All eight expanded compressed-versus-control mean deltas remained positive. D4P8 exceeded B0 by `0.2896842037`, B1 by `0.2050431914`, D2 by `0.2506866266`, and D3 by `0.1973780839`. D4P16 exceeded B0 by `0.4026417146`, B1 by `0.3180007023`, D2 by `0.3636441375`, and D3 by `0.3103355948`. The pooled expanded compressed-control delta was `0.2921767818`, with `74 / 120` positive row-level comparisons.

The expanded row-level robustness checks again supported the compressed route. The one-sided sign-test p was `0.0066881634`, the 95% bootstrap CI was `[0.1623326461, 0.4323997354]`, and leave-one-tile and leave-one-seed means remained positive. Cluster sign-only support was again directional (`10 / 15` positive clusters, p `0.1508789062`), but the exact signed-rank test over cluster magnitudes remained significant, with positive rank sum `96 / 120` and p `0.0206298828`.

A direct cluster-mean audit addressed the possibility that the expanded effect was driven by one favorable tile or seed. The mean cluster delta remained `0.2921767818`. The exact one-sided sign-flip mean p was `0.0196838379`, the 95% bootstrap CI was `[0.0570820445, 0.5823557658]`, and the minimum leave-one-cluster, leave-one-tile, and leave-one-seed means were `0.2060081575`, `0.0954244478`, and `0.2083797951`, respectively. These results supported a bounded positive compressed-GeoFM state claim after expanded replication and influence checking.

## 3.4 Compressed routes retained signal while reducing representation rank and conditioning burden

The representation-geometry audit provided a mechanism-level explanation for why the compressed route was more effective than raw injection. The B1, D4P8, and D4P16 feature tables aligned exactly over `64,984` Bishan blocks. Raw B1 contained `64` GeoFM dimensions with total centered variance `0.0981484274`, effective rank `9.4947211626`, and positive-eigenvalue condition number `6658.9542931381`. D4P8 reduced the GeoFM coordinates to `8` components while retaining total centered variance `0.0842881410`, or `85.87823898%` of raw GeoFM variance. Its effective rank was `5.1322783588`, and its condition number was `16.2704676982`. D4P16 retained total centered variance `0.0932018070`, or `94.96006154%` of raw GeoFM variance, with effective rank `7.3009059917` and condition number `53.6978527088`.

These geometry values supported the compressed-route interpretation. The compressed states preserved most of the raw GeoFM variance but presented it to the policy through lower-rank and substantially better-conditioned coordinates. The same audit linked these representations to the expanded replication deltas: D4P8 had mean compressed-minus-control reward gain `0.2356980264` with `38 / 60` positive comparisons, and D4P16 had mean gain `0.3486555373` with `36 / 60` positive comparisons. Tile-level retention-gain correlations were near zero or weakly negative (`-0.0207226322` for D4P8, `-0.2059768413` for D4P16, and `0.0257762396` pooled), so the result should not be interpreted as a simple monotonic per-tile variance-retention rule. The stronger conclusion is that compressed GeoFM states retained the useful shared signal while avoiding the high-dimensional and ill-conditioned raw representation presented by B1.

## 3.5 Normalization and bounded budget checks did not explain away the compressed route

The normalized-B1 branch tested whether raw-B1 underperformance was primarily a scaling artifact. At `4096` steps, normalized variants improved raw B1 and recovered the B0 mean gap: N1Z reached mean learned-policy reward `0.6515323140`, compared with `0.4825072170` for B0 and `0.3506359482` for raw B1. However, the normalized variants did not consistently exceed the compressed controls.

A bounded `5120`-step matched pilot further tested whether modestly higher training budget resolved the raw or normalized route. The full aggregate found that the budget did not explain the compressed-route advantage. At `5120` steps, tracked focal gaps remained negative on mean, including `N1Z - D4P16 = -0.7597285327`, `N1Z - D4P8 = -0.4953863438`, `N1ZR - D4P16 = -0.6658915329`, and `N1ZR - D4P8 = -0.4015493440`. These diagnostics did not displace the compressed-route interpretation.

## 3.6 Suitability reward remained blocked by the independent-label gate

The suitability branch did not provide evidence for B2/B3 reward integration. The available labels were current-farmland, farmland-or-orchard, and low-slope-farmland labels, each with `64,984` valid labels. All three were DLTB/slope-derived and were therefore flagged as explicit-label leakage risks rather than independent suitability labels.

The diagnostic classifier results confirmed the leakage concern. Explicit-only models reached ROC AUC, average precision, and balanced accuracy of `1.0` for all three labels, which indicates that the labels are encoded by explicit planning features rather than serving as independent suitability outcomes. GeoFM-only signals were weaker and not sufficient to justify a reward term. For example, raw GeoFM-only features reached ROC AUC `0.6490064144` for the low-slope-farmland label, while the scalar suitability proxy was near random for the same label with ROC AUC `0.4979564572`.

The leakage-aware proxy rebuild remained diagnostic-only, and the independent-label gate found no registered non-leakage label for the real Bishan run. The no-registry run read `64,984` feature rows, found `0` registered label rows, and reported missing independent label inputs. A separate suitability-prior route also found `0` independent-label-gate-passed labels and produced no block-level GeoFM suitability-prior file. A local label-source audit found only diagnostic DLTB/slope weak labels and unrelated Paper10/Paper58 labels, not an external soil, irrigation, yield, productivity, high-standard-farmland, field-survey, or policy-outcome label that could validate Bishan suitability reward. The correct conclusion is therefore that suitability reward remains blocked, not merely untested.

# 4. Discussion

The main finding is that GeoFM information was useful only after representation control. Raw 64-dimensional embedding injection did not improve held-out learned-policy reward, but compressed GeoFM states outperformed explicit-only, raw GeoFM, random-control, and shuffled-control variants under the same base reward. This pattern points to representation geometry as a practical bottleneck: the planning policy benefited when GeoFM information entered through compact PCA-compressed features, not when the raw embedding vector was appended directly to explicit planning attributes.

The mechanism audit strengthens this interpretation. D4P8 and D4P16 retained most of the raw GeoFM variance while sharply reducing effective rank and covariance conditioning burden. This is consistent with a policy-optimization explanation: the compressed features preserved shared land-surface signal but removed much of the redundant or ill-conditioned coordinate structure that made raw B1 a weak state augmentation. The weak tile-level retention-gain correlations also keep the interpretation bounded. Compression did not help because every tile with more retained variance gained more reward; rather, it changed the global state geometry in a way that was more usable under the current policy interface.

The evidence is stronger than a single favorable comparison. The positive compressed-route result appeared in the original three-tile audit, survived row-level sign-test and bootstrap checks, remained positive under leave-one sensitivity, and was reproduced in an expanded five-tile, three-seed evaluation. Cluster analysis imposed a more conservative independence structure. Although sign-only cluster counts were directional rather than conventionally significant, magnitude-sensitive signed-rank and direct cluster-mean tests supported the effect. The artifact-lineage audit further reduced a reproducibility risk by showing that the formal values were derived from one authoritative evidence chain.

The result should not be over-read. The supported claim concerns Bishan held-out planning under a deterministic base planning reward. It does not prove that PCA is the optimal compression method, that the same effect transfers to other regions, or that the policy learned agronomic suitability semantics. The present evidence also does not validate B2/B3 suitability rewards. Those claims require independent non-leakage labels and calibrated GeoFM priors that are not currently available in the local Paper11 inputs.

The negative raw-B1 result remains scientifically useful. Without the compressed controls, it would have been easy to conclude that GeoFM information adds no value to this planning task. The controlled ladder instead shows a more actionable result: GeoFM can help, but the state representation must be engineered and validated. For applied land-use planning workflows, this distinction matters because adding high-dimensional foundation-model embeddings directly to a policy state may be less effective than introducing low-dimensional, controlled representations whose behavior can be compared against random and shuffled alternatives.

The practical path forward is therefore specific. The current manuscript can support a compressed-state representation claim for Bishan base-reward planning. A future suitability-reward study would require external labels that are not derived from the same DLTB/slope features used by the reward and planning state. Once such labels exist, the suitability-prior route should be tested against explicit-feature, random, shuffled, fold-stability, and calibration controls before any reward term is introduced.

# 5. Conclusion

This study establishes a reproducible evidence-gated workflow for evaluating GeoFM-enhanced farmland layout optimization on real Bishan planning units. The evidence rejects raw 64-dimensional GeoFM direct injection as a supported planning-state improvement, but supports PCA-compressed GeoFM routes D4P8 and D4P16 under the Bishan base-reward held-out protocol. The strongest current performance evidence comes from the expanded five-tile, three-seed replication: pooled compressed-control delta `0.2921767818`, `74 / 120` positive row-level comparisons, row-level sign-test p `0.0066881634`, cluster signed-rank p `0.0206298828`, exact cluster-mean sign-flip p `0.0196838379`, and 95% bootstrap CI `[0.0570820445, 0.5823557658]`. The mechanism evidence shows that these compressed routes retained `85.87823898%` and `94.96006154%` of raw GeoFM variance while reducing effective rank and condition number relative to B1. The appropriate conclusion is therefore bounded and positive: GeoFM improved the learned planning policy when represented through controlled compressed state features. Raw B1 superiority, suitability reward, B2/B3 readiness, cross-region transfer, and independently validated agronomic suitability remain unsupported.

# Data Availability

The repository includes lightweight Bishan AlphaEarth sample arrays, code, tests, reproduction guides, and file manifests for reviewer smoke checks. Full real Bishan DLTB-with-slope inputs are external local data and are not redistributed in ordinary Git. Large derived arrays, full generated outputs, and trained artifacts should be deposited in an external archive before any submission that relies on them.

# Code Availability

All reviewer-facing code is maintained in the Paper11 repository:

```text
https://github.com/zhouning/paper11-geofm-farmland-suitability-rl
```

The final submitted version should cite a release tag or immutable commit hash.

# Submission Metadata Still Required

- Final author list, affiliations, and corresponding-author details.
- Funding statement.
- Author contributions.
- Final data access statement for the external DLTB-with-slope GeoPackage.
- Reference list and citation formatting for the selected journal.
- Final decision on target journal format for a bounded compressed-GeoFM representation article.