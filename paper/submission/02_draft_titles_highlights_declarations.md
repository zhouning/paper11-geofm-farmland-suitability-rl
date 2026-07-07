# Draft Submission Text

> Phase 48 supersession note: this Phase 42 scaffold is retained as provenance only. Use `04_formal_conclusion_manuscript.md` and `final/Paper11_cover_letter_and_declarations.md` for the current bounded positive compressed-GeoFM representation package.

Use this file as guarded starting text for the Phase 42-synchronized conclusion
submission route. It supports a bounded negative/evidence-gated manuscript, not
a positive GeoFM-superiority, B2/B3, suitability-reward, or transfer claim.

## Title Options

Recommended:

1. Evidence-gated rejection of unsupported GeoFM superiority in reinforcement-learning farmland layout optimization

More method focused:

2. Evidence gates for testing GeoFM representations in farmland layout optimization

More negative-results focused:

3. Representation controls reject unsupported raw GeoFM gains in farmland layout reinforcement learning

Do not use a performance-led title unless future independent-label and B2/B3
experiments overturn the current evidence.

## Keywords

GeoFM; AlphaEarth embeddings; farmland spatial optimization; reinforcement
learning; suitability proxy; representation control; independent-label gate;
land-use planning

## Highlights

Current guarded version:

- A reproducible workflow links AlphaEarth embeddings to real Bishan DLTB planning blocks.
- Tiled and padded policy interfaces make large real planning instances testable.
- Multi-tile B0/B1 learned-policy evidence does not support raw GeoFM superiority.
- Representation controls show that compressed GeoFM controls can exceed raw B1.
- Suitability-reward work is blocked until an independent non-leakage label gate passes.
- A local Phase 42 label-source audit finds no usable external suitability label for the real Bishan run.
- The evidence supports a bounded negative conclusion rather than a positive GeoFM-superiority claim.

## Abstract Scaffold

Farmland spatial layout optimization requires decisions over many heterogeneous
planning units, but direct variables for soil quality, irrigation, productivity,
and long-term suitability are often unavailable at operational planning scale.
This study tests whether geospatial foundation-model (GeoFM) embeddings provide
a reliable latent representation for reinforcement-learning farmland layout
optimization when evaluated with representation controls and leakage-aware
suitability gates.

The workflow aggregates AlphaEarth embeddings and explicit planning features to
`64,984` real Bishan DLTB land-use blocks, constructs tiled and padded planning
interfaces, and evaluates GeoFM-enhanced B1 against explicit-feature B0 and
representation controls under a deterministic base planning reward. The current
held-out B0/B1 result does not support raw GeoFM superiority: at 4096 training
steps, the B1-B0 mean reward delta is `-0.1318712688`, with only `3 / 9`
held-out tile-seed pairs favoring B1. Phase 28 reports
`compression_matches_raw`, and PCA-compressed controls exceed raw B1 in the
4096-step run. Phase 33 further reports `budget_not_explanatory` for bounded
5120-step matched normalized-B1 checks.

The suitability branch is also stopped by evidence gates. Phase 36 reports
`proxy_signal_not_supported`; Phase 38 remains
`proxy_rebuild_diagnostic_only`; Phase 40 reports
`independent_label_inputs_missing` in the current no-registry run; and Phase 41
reports `phase41_independent_label_inputs_missing` for the calibrated GeoFM
suitability-prior route. Phase 42 audits local candidate label sources and
finds no local external soil, irrigation, yield, productivity,
high-standard-farmland, field-survey, or policy-outcome label that can pass
Phase 40.

The manuscript should therefore conclude that raw GeoFM state injection and the
current suitability-reward route are unsupported under the completed Paper11
evidence gates. It should not claim that GeoFM universally fails, nor that
GeoFM improves learned farmland planning decisions in the current evidence.

## Cover Letter Scaffold

Dear Editor,

We submit the manuscript entitled "Evidence-gated rejection of unsupported GeoFM
superiority in reinforcement-learning farmland layout optimization" for
consideration in International Journal of Applied Earth Observation and
Geoinformation. The paper addresses a practical question for Earth-observation
and land-use planning research: whether satellite foundation-model embeddings
can be treated as decision-ready representations for farmland layout
optimization.

The manuscript reports a reproducible Bishan case-study workflow that links
AlphaEarth embeddings to real DLTB planning blocks and evaluates GeoFM-enhanced
representations under held-out reinforcement-learning planning protocols,
representation controls, suitability-proxy diagnostics, and independent-label
gates. The central finding is negative and evidence-bounded. Raw GeoFM B1 does
not stably outperform explicit-feature B0, compressed representation controls
exceed raw B1, normalized-B1 and budget checks do not rescue the claim, and the
suitability-reward route is stopped by independent-label and calibrated-prior
gates.

This work should be of interest to IJAEOG readers because it provides a
reproducible cautionary test for using Earth-observation foundation-model
embeddings in operational land-use optimization. Rather than assuming that a
semantically rich remote-sensing embedding is a validated suitability variable,
the manuscript shows how representation controls and independent-label gates can
prevent unsupported planning claims.

The manuscript is original, is not under consideration elsewhere, and all
authors have approved the submission. [Confirm or edit before use.]

Sincerely,

[Corresponding author name]

## Data Availability Draft

The lightweight Bishan AlphaEarth sample arrays, code, reproduction scripts,
and file manifests needed for reviewer smoke tests are available in the project
repository:

```text
https://github.com/zhouning/paper11-geofm-farmland-suitability-rl
```

The repository commit used for submission should be recorded here:

```text
[final release commit or DOI/tag]
```

Large derived arrays, trained weights, and any full external data products not
included in ordinary Git should be deposited in [Zenodo/OSF/institutional
repository] before final submission, with checksums and access instructions.
The real Bishan DLTB-with-slope GeoPackage used for local Phase 11 reproduction
is an external source and is not redistributed in this repository; final
availability wording must state whether it can be shared, requested, or only
reproduced by authorized users.

## Code Availability Draft

All code required for the reviewer-facing smoke tests, reproduction guide, and
Phase 1-42 workflow is available at:

```text
https://github.com/zhouning/paper11-geofm-farmland-suitability-rl
```

The repository includes a reproducibility guide, file manifest, test suite, and
lightweight sample data. Before final submission, rerun `python
scripts\smoke_check.py` and the focused Phase 40/41 tests and record the final
release commit or DOI.

## Declaration Drafts

Declaration of competing interest:

> The authors declare that they have no known competing financial interests or
> personal relationships that could have appeared to influence the work reported
> in this paper.

Funding:

> [Funding statement required from authors.]

Author contributions:

> [CRediT author contribution statement required from authors.]

Ethics:

> This study uses remote-sensing and land-use planning data and does not involve
> human participants, human tissue, animal experiments, or private personal
> data. [Confirm before use.]

Use of AI-assisted tools:

> AI-assisted tools were used for language editing, code assistance, and
> submission-preparation drafting. All scientific claims, code outputs,
> references, data interpretations, and manuscript text were checked and edited
> by the authors. [Revise to match the final journal policy and actual use.]

## Claim Boundary

Current safe boundary:

> The repository implements a reproducible GeoFM-enhanced farmland-planning
> workflow and shows that, under the completed Bishan evidence gates, raw GeoFM
> B1 is not a stable positive learned-policy signal and the suitability-reward
> route remains blocked by independent-label and calibrated-prior gates.

Current blocked boundary:

> Do not claim GeoFM superiority, B2/B3 readiness, suitability-reward
> improvement, cross-region transfer, or independent agronomic suitability
> validation. Phase 40 reports `independent_label_inputs_missing`; Phase 41
> reports `phase41_independent_label_inputs_missing`; and Phase 42 finds no
> local external label source that can pass Phase 40.
