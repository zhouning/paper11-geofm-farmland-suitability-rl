# IJAEOG Submission Readiness Audit

Checked on 2026-06-19 and updated on 2026-07-06 after Phase 42. Target journal source:

- International Journal of Applied Earth Observation and Geoinformation,
  Guide for Authors:
  <https://www.elsevier.com/journals/international-journal-of-applied-earth-observation-and-geoinformation/1569-8432/guide-for-authors>
- ScienceDirect journal guide mirror:
  <https://www.sciencedirect.com/journal/international-journal-of-applied-earth-observation-and-geoinformation/publish/guide-for-authors>

This audit uses the guide as a submission-structure reference. Requirements can
change, so re-check the live guide before final upload.

Phase 42 update: Paper11 now has a defensible conclusion-type formal manuscript
route in `04_formal_conclusion_manuscript.md`. This route is not a positive
GeoFM-performance submission. It argues that raw GeoFM state injection and the
current suitability-reward route are unsupported under the completed Bishan
protocol, while the positive B2/B3 route remains blocked by Phase 40/41 and the
Phase 42 local label-source audit.

## One-Sentence Manuscript Argument

In farmland spatial layout optimization, we show that raw GeoFM state injection
and the current suitability-reward route are unsupported under the completed
real Bishan evidence gates, using held-out B0/B1 policy comparisons,
representation controls, budget checks, independent-label gates, and a local
label-source audit.

## Terminology Ledger

| Canonical term | First-use definition | Variants to avoid | Decision |
|---|---|---|---|
| GeoFM | geospatial foundation model (GeoFM) | geospatial FM, foundation remote sensing model | Spell out once, then use GeoFM. |
| AlphaEarth embeddings | AlphaEarth 64-dimensional annual satellite embeddings | AlphaEarth vector, satellite embedding vector | Use AlphaEarth embeddings when source is AlphaEarth. |
| base planning reward | deterministic base planning reward | base reward, planning score | Use `base planning reward` in prose and `base_planning_reward` for code fields. |
| suitability proxy | weakly supervised latent suitability proxy | suitability score, suitability measurement | Use proxy language unless externally validated. |
| DLTB | real Bishan DLTB land-use polygons | parcel data, land block data | Define the data source in Methods and avoid parcel-accuracy overclaims. |
| tiled MaskablePPO readiness | tiled MaskablePPO API readiness smoke check | trained MaskablePPO result | State that this is API readiness, not policy performance. |
| bounded same-tile B0/B1 pilot | Phase 20 bounded same-tile B0/B1 MaskablePPO pilot | held-out evaluation, transfer result | Use only for pilot execution evidence, not policy superiority. |
| cross-tile per-block scorer pilot | Phase 21 bounded cross-tile per-block scorer pilot | cross-region transfer result, final policy result | Use only for variable-block-count interface evidence. |
| multi-tile multi-seed scorer pilot | Phase 22 bounded multi-tile, multi-seed per-block scorer evaluation pilot | transfer result, final multi-seed policy result | Use only for broadened interface-pilot evidence. |
| multi-seed B0/B1 training pilot | Phase 23 bounded multi-seed same-tile B0/B1 MaskablePPO training pilot | final performance result, transfer result | Use only for same-tile learned-policy pilot evidence. |
| IJAEOG evidence package | Phase 24 synthesis and claim-readiness package | new performance result, new transfer result | Use only as an evidence ledger and claim-boundary artifact. |
| padded held-out B0/B1 policy pilot | Phase 25 bounded padded variable-size held-out Bishan tile MaskablePPO pilot | cross-region transfer result, suitability-reward result, final policy-performance result | Use only for held-out Bishan tile learned-policy pilot evidence under deterministic base planning reward. |
| main B0/B1 held-out analysis package | Phase 26 B0/B1 padded held-out multi-seed, multi-tile analysis package | final B2/B3 result, suitability-reward result, cross-region transfer result | Use only for B1-B0 learned-policy deltas across held-out Bishan tiles and seeds under deterministic base planning reward. |
| B0/B1 stability diagnosis | Phase 27 read-only comparison of 1024-step and 4096-step Phase 26 artifacts | new training result, convergence proof, positive performance result | Use only to explain budget and tile-seed instability in the current negative evidence. |
| representation-control diagnosis | Phase 28 B0/B1/D2/D3/D4 padded held-out diagnostic package | positive raw-B1 superiority, suitability reward, B2/B3, transfer | Use only to show whether B1 is distinguishable from random, shuffled, and PCA-compressed controls under the current base-reward protocol. |

## Readiness Summary

| Item | Status | Evidence or blocker |
|---|---|---|
| Repository and code package | Ready | The repository includes the Phase 26 main empirical analysis package, Phase 27 stability diagnosis, and Phase 28 representation-control CLI; final verification must be rerun before upload. |
| Lightweight reproducibility | Ready | `python scripts\smoke_check.py` passes with included Bishan AlphaEarth sample arrays. |
| Real-data adapter | Ready for local reproduction | Phase 11 exports 64,984 Bishan DLTB polygons into Phase 2-compatible artifacts, using a local external GeoPackage. |
| Tiled contract | Ready | Phase 13 creates 54 non-empty tiles; largest tile has 2,234 blocks. |
| Base planning reward | Ready as first implementation | Phase 19 implements a bounded explicit-feature reward; Phase 14 largest-tile B1 one-step reward is `-0.197259`. |
| MaskablePPO API path | Ready as smoke evidence | Phase 17 passes the tiled MaskablePPO readiness smoke check. |
| Bounded B0/B1 training pilot | Ready as same-tile pilot evidence | Phase 20 trains and evaluates B0/B1 on `tile_r003_c003`, writes six summary rows, and records `blocked_variable_observation_shape` for cross-tile learned-policy evaluation. |
| Cross-tile learned-policy interface | Ready as per-block scorer pilot evidence | Phase 21 trains a standardized ridge-linear block scorer on `tile_r003_c003`, evaluates on distinct tile `tile_r002_c003`, writes six summary rows, and reports `executed_distinct_tile`. |
| Multi-tile scorer evaluation | Ready as broadened interface-pilot evidence | Phase 22 trains the same scorer once per B0/B1 variant on `tile_r003_c003`, evaluates `tile_r002_c003` and `tile_r005_c004` across seeds `0` and `1`, and writes 24 summary rows. |
| Multi-seed learned-policy training | Ready as same-tile pilot evidence | Phase 23 repeats bounded B0/B1 MaskablePPO training on `tile_r003_c003` across seeds `0`, `1`, and `2`, writes 18 summary rows, and reports B1-B0 learned-policy mean reward delta `0.4273019432` under the short pilot budget. |
| IJAEOG evidence package | Ready as claim-readiness synthesis | Phase 24 consolidates Phase 22/23 outputs into CSV, JSON, and Markdown evidence artifacts and keeps `submission_ready: not_ready`. |
| Padded held-out learned-policy pilot | Ready as bounded held-out Bishan tile pilot evidence | Phase 25 trains B0/B1 MaskablePPO policies on `tile_r003_c003` and evaluates on distinct held-out tile `tile_r002_c003` under deterministic `base_planning_reward`; the verified smoke writes six summary rows and reports B1-B0 held-out learned-policy mean reward delta `1.3314600457` with `pilot_result_status: B1_improves_B0`. |
| Main empirical analysis package | Ready as current negative evidence | Phase 26 ingests Phase 25 outputs and now includes macOS 1024-step and 4096-step result sets. The 4096-step learned-policy B1-B0 mean reward delta is `-0.1318712688`, with only `3 / 9` positive tile-seed pairs and claim status `not_supported`. |
| B0/B1 stability diagnosis | Ready as current diagnostic evidence | Phase 27 compares the 1024-step and 4096-step Phase 26 result sets and reports `budget_not_explanatory`: mean delta improves by `0.3010310174`, but positive tile-seed count falls by `1`, with stability counts `1` stable-positive, `3` stable-negative, `2` flip-to-positive, and `3` flip-to-negative. |
| Representation-control diagnosis | Ready as current negative diagnostic evidence | Phase 28 compares B1 against B0, D2 random controls, D3 shuffled controls, and D4 PCA-compressed controls at 1024 and 4096 steps. Both runs report `compression_matches_raw`; at 4096 steps B1-B0 is `-0.1318712688`, B1-D2 is `0.0744591656`, B1-D3 is `-0.1094750135`, B1-D4P8 is `-0.3768518347`, and B1-D4P16 is `-0.6411940236`. |
| Independent-label gate | Ready as current no-go evidence | Phase 40 reads the real Bishan Phase 2 table and requires a registered independent non-leakage label before any Phase 38 rerun or B2/B3 reward smoke. The current no-registry run reports `independent_label_inputs_missing` over `64,984` feature rows and `0` registry rows, so the suitability-reward route remains stopped. |
| GeoFM suitability-prior gate | Ready as current no-go evidence | Phase 41 implements the revised GeoFM route: an independent-label-calibrated low-dimensional prior instead of raw 64-dimensional state injection. The current real no-registry run reports `phase41_independent_label_inputs_missing`, so no calibrated prior exists and B2/B3 remains blocked. |
| Suitability reward | Not ready | Phase 10/12 keep suitability reward disabled, Phase 36 reports `proxy_signal_not_supported`, and Phase 40 keeps the route stopped until an external independent label registry passes the gate. |
| Planning-performance experiments | Not ready | Phase 18 still reports `performance_experiment_ready: false`. |
| Conclusion manuscript route | Ready as formal text | Use `paper/submission/04_formal_conclusion_manuscript.md` for a bounded negative/evidence-gated manuscript. Positive performance, suitability-reward, B2/B3, and transfer claims remain blocked. |

## Submission Material Checklist

| Material | Current action |
|---|---|
| Title page | Use a guarded title from `02_draft_titles_highlights_declarations.md`; add author affiliations manually. |
| Abstract | Ready for conclusion route | Use the Phase 42-synchronized abstract in `04_formal_conclusion_manuscript.md` or the scaffold in `02_draft_titles_highlights_declarations.md`; do not convert it into a positive performance abstract. |
| Highlights | Ready for conclusion route | Use the Phase 42 guarded highlights in `02_draft_titles_highlights_declarations.md`; do not revise into positive performance wording. |
| Keywords | Use GeoFM, farmland spatial optimization, reinforcement learning, suitability proxy, AlphaEarth, tiled planning. |
| Main manuscript | Ready only for conclusion route | `04_formal_conclusion_manuscript.md` is the current formal text. A positive IJAEOG-style performance manuscript remains blocked. |
| Figures | Pending for final upload | The formal text is available, but final journal upload still needs figure decisions. For the conclusion route, prioritize workflow, evidence-gate, representation-control, and label-gate figures rather than positive transfer/performance figures. |
| Data availability | Draft available; final version needs repository URL, large-data archive, and external DLTB access constraints. |
| Code availability | Ready once GitHub URL and final release commit or tag are cited. Do not use an in-file self-reference as the final commit hash. |
| Declaration of interests | Draft available; authors must confirm. |
| Funding statement | Missing. Must be supplied by authors. |
| Author contributions | Missing. Must be supplied by authors. |
| Ethics statement | Likely not applicable for remote-sensing/planning data, but confirm no human/animal/private-person data. |
| Cover letter | Phase 42 conclusion skeleton available | Use `02_draft_titles_highlights_declarations.md`; final author and submission metadata must still be supplied. |

## Claim-Evidence Gate Before Submission

Do not use these claims until the evidence exists:

| Manuscript claim | Required evidence |
|---|---|
| GeoFM improves planning decisions | B1 outperforms B0 under the same reward and evaluation protocol. |
| Suitability reward improves spatial realism | Phase 40 passes with an independent non-leakage label, Phase 38 proxy rebuild clears control gates, and B2/B3 improve suitability-weighted metrics without unacceptable slope, contiguity, or validity loss. |
| GeoFM improves transfer | B1/B3 reduce held-out-region performance drop relative to B0. |
| Full Paper11 model is best | B3 beats or matches B1/B2 across main and transfer metrics, with uncertainty or multi-seed support. |
| Suitability proxy is meaningful | Independent label registry passes Phase 40, proxy rebuild passes leakage-aware validation, and distributional diagnostics show decision-relevant signal beyond DLTB/slope-derived labels. |
| GeoFM suitability prior is admissible | Phase 40 passes with an independent non-leakage label, and Phase 41 reports `geofm_suitability_prior_supported` after explicit baseline, shuffled-control, random-control, fold-stability, and calibration checks. |

Phase 40 now makes the suitability branch conditional: no Phase 38 rerun, B2/B3
reward smoke, or positive suitability-reward claim should proceed until an
external independent label registry passes the gate. The current no-registry
run remains `independent_label_inputs_missing`.

Phase 41 changes the proposed GeoFM route from raw embedding concatenation to a
strict prior gate. The current real status remains
`phase41_independent_label_inputs_missing`, so no calibrated GeoFM suitability
prior exists for the manuscript and B2/B3 remains blocked.

Safe current claim:

> The repository implements a reproducible GeoFM-enhanced farmland-planning
> input pipeline, real Bishan DLTB adaptation, tiled environment contracts, and
> a first deterministic base planning reward. It also executes a bounded
> same-tile B0/B1 training pilot and a cross-tile per-block scorer pilot, but
> it does not yet provide final planning-performance, suitability-reward, or
> transfer evidence for learned policy superiority. Phase 22 broadens the
> per-block scorer pilot across multiple evaluation tiles and seeds, but remains
> interface-pilot evidence rather than final policy-performance evidence.
> Phase 23 adds a multi-seed same-tile B0/B1 MaskablePPO pilot and reports a
> positive B1-B0 learned-policy mean reward delta under a short training budget,
> but this remains pilot evidence pending longer training, ablations,
> suitability-reward validation, and held-out-region transfer. Phase 24 records
> these boundaries in a reviewer-facing evidence package and keeps full
> submission readiness at `not_ready`. Phase 25 adds a padded variable-size
> held-out Bishan tile B0/B1 learned-policy pilot under deterministic
> `base_planning_reward`; the verified smoke result is positive for B1-B0 on
> one held-out Bishan tile, but remains bounded pilot evidence and does not
> resolve suitability-reward readiness, B2/B3 claims, cross-region transfer,
> long-budget robustness, or final submission readiness. Phase 26 adds the
> main empirical analysis package for B0/B1 padded held-out outputs, and the
> current macOS result sets do not support a positive B1-over-B0 learned-policy
> claim: the 4096-step mean delta is `-0.1318712688` and only `3 / 9`
> tile-seed pairs favor B1. Phase 27 adds a budget and tile-seed stability
> diagnosis and reports `budget_not_explanatory`. Phase 28 then adds
> B0/B1/D2/D3/D4 representation controls at 1024 and 4096 steps and reports
> `compression_matches_raw` in both runs, so the current evidence still does
> not support a positive raw-B1 learned-policy or representation-superiority
> claim.

Unsafe current claim:

> The GeoFM-enhanced DRL policy outperforms existing farmland-layout optimization
> methods.

## Phase 20/21/22/23/24/25/26/27/28 Status and Recommended Next Experimental Phase

Phase 20 now provides a bounded same-tile B0/B1 training and evaluation pilot.
It avoids suitability reward, uses fixed tile selection and seed, runs a short
MaskablePPO budget, writes deterministic baseline comparisons, and records
selected-block diagnostics. It does not provide held-out tile or transfer
evidence because the current flat observation and action spaces are
tile-size-specific.

Phase 21 addresses that shape blocker at the interface level. It trains a
standardized ridge-linear per-block scorer on `tile_r003_c003` and evaluates on
the distinct tile `tile_r002_c003`. This is cross-tile learned-scorer evidence,
not final DRL performance evidence.

Phase 22 broadens that interface pilot across multiple distinct evaluation
tiles and seeds. It still uses a ridge-linear per-block scorer and deterministic
baseline policies rather than PPO-compatible variable-size policy training, so
it is not final policy-performance or transfer evidence.

Phase 23 responds to the reviewer concern about single-seed learned-policy
evidence by repeating the Phase 20 same-tile B0/B1 MaskablePPO pilot across
three seeds. It reports a positive B1-B0 mean learned-policy reward delta under
the short pilot budget, but the same-tile design and short training budget mean
this still cannot support final GeoFM-superiority or transfer claims.

The next experimental phase should move beyond Phase 23 into a true variable
tile-size policy or held-out-region experiment before any transfer claim.
Phase 24 can be cited as the current evidence ledger, but not as a new
performance experiment.

Phase 25 introduces a padded variable-size learned-policy held-out Bishan tile
pilot. The verified Windows smoke trains on `tile_r003_c003`, evaluates on the
distinct tile `tile_r002_c003`, writes six B0/B1 learned-policy and baseline
summary rows, reports all evaluations completed, and records a B1-B0 held-out
learned-policy mean reward delta of `1.3314600457` with
`pilot_result_status: B1_improves_B0`. This can support the limited statement
that a B0/B1 learned policy can be trained on one Bishan tile and evaluated on
a distinct held-out Bishan tile under the deterministic base planning reward.
It does not resolve suitability-reward readiness, B2/B3 claims, cross-region
transfer, long-budget robustness, or final submission readiness.

Phase 26 is the current main empirical analysis package. It does not create a
new reward or representation family; it consumes Phase 25 outputs and writes
the main B0/B1 held-out summary, tile-seed delta table, comparison JSON, and
claim-readiness Markdown. The macOS main result artifacts now show that the
current B1 learned policy does not stably outperform B0. The 1024-step result
reports a B1-B0 mean delta of `-0.4329022862` with `4 / 9` positive tile-seed
pairs, and the 4096-step result reports `-0.1318712688` with `3 / 9` positive
tile-seed pairs. The next experimental action is diagnostic rather than
claim-extending: compare budget sensitivity, run representation controls, and
repair suitability-proxy readiness before any positive performance claim.

Phase 27 performs that diagnostic comparison without rerunning training. It
finds that the higher budget improves the mean B1-B0 delta by `0.3010310174`,
but the higher-budget result remains negative and the positive tile-seed count
falls by `1`. Its stability counts are `1` stable-positive, `3`
stable-negative, `2` flip-to-positive, and `3` flip-to-negative. The diagnosis
therefore remains conservative: budget alone is not a sufficient explanation.

Phase 28 runs the representation-control package that Phase 27 required. It
evaluates B0, B1, D2, D3, D4P8, and D4P16 under the same padded held-out
Bishan base-reward protocol at 1024 and 4096 training steps. Both runs report
`compression_matches_raw` rather than `representation_signal_supported`. At
4096 steps, B1 is slightly above D2 but remains below B0, D3, D4P8, and
D4P16. This means the current raw 64-dimensional GeoFM B1 arm is not yet a
stable positive representation signal, and compressed controls must be treated
as a serious rival explanation before any manuscript-level claim.

Minimum requirements are:

- renewed suitability-proxy validation before any reward integration;
- spatial case maps for representative held-out tiles and selected blocks;
- diagnosis of why PCA-compressed controls exceed raw B1 in the 4096-step
  Phase 28 run;
- a redesigned representation or reward experiment before any B2/B3
  escalation;
- explicit claim boundary that current B1 evidence remains negative until
  ablation, suitability-reward, and held-out-region tests become supportive.

## Final Upload Risks

1. The manuscript will be rejected if it claims DRL performance or raw-B1
   representation superiority despite the current Phase 26/27/28 negative
   evidence.
2. The suitability contribution is not yet ready because Phase 10 blocks
   suitability reward.
3. The external DLTB GeoPackage is not committed, so final data availability
   needs an access path or a clearly documented restriction.
4. The code package is strong, but a journal article still needs figures,
   quantitative results, and calibrated claims.
