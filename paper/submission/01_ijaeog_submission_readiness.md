# IJAEOG Submission Readiness Audit

Checked on 2026-06-12. Target journal source:

- International Journal of Applied Earth Observation and Geoinformation,
  Guide for Authors:
  <https://www.elsevier.com/journals/international-journal-of-applied-earth-observation-and-geoinformation/1569-8432/guide-for-authors>
- ScienceDirect journal guide mirror:
  <https://www.sciencedirect.com/journal/international-journal-of-applied-earth-observation-and-geoinformation/publish/guide-for-authors>

This audit uses the guide as a submission-structure reference. Requirements can
change, so re-check the live guide before final upload.

## One-Sentence Manuscript Argument

In farmland spatial layout optimization, we test whether frozen GeoFM
embeddings and weakly supervised suitability proxies improve block-level DRL
state representation and transfer, using real Bishan DLTB planning units and a
tiled reproducibility workflow, while treating remote-sensing embeddings as
latent proxies rather than direct soil, irrigation, or fertility measurements.

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

## Readiness Summary

| Item | Status | Evidence or blocker |
|---|---|---|
| Repository and code package | Ready | `origin/main` includes Phase 22 after this update; `python -m pytest tests -q` reports `133 passed`. |
| Lightweight reproducibility | Ready | `python scripts\smoke_check.py` passes with included Bishan AlphaEarth sample arrays. |
| Real-data adapter | Ready for local reproduction | Phase 11 exports 64,984 Bishan DLTB polygons into Phase 2-compatible artifacts, using a local external GeoPackage. |
| Tiled contract | Ready | Phase 13 creates 54 non-empty tiles; largest tile has 2,234 blocks. |
| Base planning reward | Ready as first implementation | Phase 19 implements a bounded explicit-feature reward; Phase 14 largest-tile B1 one-step reward is `-0.197259`. |
| MaskablePPO API path | Ready as smoke evidence | Phase 17 passes the tiled MaskablePPO readiness smoke check. |
| Bounded B0/B1 training pilot | Ready as same-tile pilot evidence | Phase 20 trains and evaluates B0/B1 on `tile_r003_c003`, writes six summary rows, and records `blocked_variable_observation_shape` for cross-tile learned-policy evaluation. |
| Cross-tile learned-policy interface | Ready as per-block scorer pilot evidence | Phase 21 trains a standardized ridge-linear block scorer on `tile_r003_c003`, evaluates on distinct tile `tile_r002_c003`, writes six summary rows, and reports `executed_distinct_tile`. |
| Multi-tile scorer evaluation | Ready as broadened interface-pilot evidence | Phase 22 trains the same scorer once per B0/B1 variant on `tile_r003_c003`, evaluates `tile_r002_c003` and `tile_r005_c004` across seeds `0` and `1`, and writes 24 summary rows. |
| Suitability reward | Not ready | Phase 10/12 keep suitability reward disabled because weak-label evidence is incomplete. |
| Planning-performance experiments | Not ready | Phase 18 still reports `performance_experiment_ready: false`. |
| Full manuscript claims | Not ready | No B0/B1/B2/B3 policy-training comparison, ablation, transfer test, or final figures yet. |

## Submission Material Checklist

| Material | Current action |
|---|---|
| Title page | Use a guarded title from `02_draft_titles_highlights_declarations.md`; add author affiliations manually. |
| Abstract | Draft only after real training/evaluation results exist; current abstract can only describe design/readiness. |
| Highlights | Use the guarded draft now; revise after quantitative results. |
| Keywords | Use GeoFM, farmland spatial optimization, reinforcement learning, suitability proxy, AlphaEarth, tiled planning. |
| Main manuscript | Not ready. Results section needs policy-performance evidence before submission. |
| Figures | Not ready. Need at least method diagram, study area/data flow, main performance, ablation, transfer, and spatial case maps. |
| Data availability | Draft available; final version needs repository URL, large-data archive, and external DLTB access constraints. |
| Code availability | Ready once GitHub URL and final release commit or tag are cited. Do not use an in-file self-reference as the final commit hash. |
| Declaration of interests | Draft available; authors must confirm. |
| Funding statement | Missing. Must be supplied by authors. |
| Author contributions | Missing. Must be supplied by authors. |
| Ethics statement | Likely not applicable for remote-sensing/planning data, but confirm no human/animal/private-person data. |
| Cover letter | Skeleton available; final letter must name the completed evidence, not planned experiments. |

## Claim-Evidence Gate Before Submission

Do not use these claims until the evidence exists:

| Manuscript claim | Required evidence |
|---|---|
| GeoFM improves planning decisions | B1 outperforms B0 under the same reward and evaluation protocol. |
| Suitability reward improves spatial realism | B2/B3 improve suitability-weighted metrics without unacceptable slope, contiguity, or validity loss. |
| GeoFM improves transfer | B1/B3 reduce held-out-region performance drop relative to B0. |
| Full Paper11 model is best | B3 beats or matches B1/B2 across main and transfer metrics, with uncertainty or multi-seed support. |
| Suitability proxy is meaningful | Weak-label validation, distributional diagnostics, and, if available, external proxy checks. |

Safe current claim:

> The repository implements a reproducible GeoFM-enhanced farmland-planning
> input pipeline, real Bishan DLTB adaptation, tiled environment contracts, and
> a first deterministic base planning reward. It also executes a bounded
> same-tile B0/B1 training pilot and a cross-tile per-block scorer pilot, but
> it does not yet provide final planning-performance, suitability-reward, or
> transfer evidence for learned policy superiority. Phase 22 broadens the
> per-block scorer pilot across multiple evaluation tiles and seeds, but remains
> interface-pilot evidence rather than final policy-performance evidence.

Unsafe current claim:

> The GeoFM-enhanced DRL policy outperforms existing farmland-layout optimization
> methods.

## Phase 20/21/22 Status and Recommended Next Experimental Phase

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

The next experimental phase should move beyond the Phase 22 scorer protocol
into a true policy experiment before any transfer claim. Minimum requirements
are:

- a variable-size, padded, or per-block PPO-compatible policy architecture, or
  a clearly justified non-PPO policy protocol;
- multi-tile and multi-seed B0/B1 evaluation under the same base planning reward;
- deterministic non-learning baselines from Phase 16 for each evaluation tile;
- total base reward, action validity, selected-block diagnostics, and runtime;
- explicit claim boundary that this remains pilot evidence until multi-seed,
  ablation, suitability-reward, and held-out-region tests are complete.

## Final Upload Risks

1. The manuscript will be rejected if it claims DRL performance before B0/B1/B2/B3
   policy comparisons exist.
2. The suitability contribution is not yet ready because Phase 10 blocks
   suitability reward.
3. The external DLTB GeoPackage is not committed, so final data availability
   needs an access path or a clearly documented restriction.
4. The code package is strong, but a journal article still needs figures,
   quantitative results, and calibrated claims.
