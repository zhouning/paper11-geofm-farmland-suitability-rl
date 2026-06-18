# Draft Submission Text

Use this file as guarded starting text. Replace bracketed placeholders only
after the corresponding evidence or author information exists.

## Title Options

Recommended:

1. GeoFM-enhanced farmland suitability representation for reinforcement-learning-based spatial layout optimization

More remote-sensing focused:

2. Geospatial foundation-model embeddings for farmland suitability representation in spatial layout optimization

More planning focused:

3. Latent remote-sensing suitability proxies for block-level farmland consolidation planning

Do not use a performance-led title until the experiments support it.

## Keywords

GeoFM; AlphaEarth embeddings; farmland spatial optimization; reinforcement
learning; suitability proxy; tiled planning; land-use planning

## Highlights

Current guarded version:

- A GeoFM-enhanced pipeline links satellite embeddings to farmland planning units.
- Real Bishan DLTB polygons are converted into reproducible block-level inputs.
- Tiled environment contracts make large real planning instances tractable.
- A deterministic base planning reward supports bounded B0/B1 pilot protocols.
- Multi-seed B0/B1 learned-policy pilots execute with evidence-gated claims.
- A padded variable-size policy contract enables a bounded held-out Bishan tile learned-policy pilot for B0/B1 under a deterministic planning reward.
- A main empirical analysis package reports that current B1 learned-policy results do not stably outperform B0 across held-out Bishan tiles and random seeds.
- Claim-readiness artifacts separate pilot evidence from unsupported claims.

Before submission, revise the last two bullets after diagnostic, ablation,
suitability-reward, and transfer evidence exists. Do not convert the current
Phase 26 result into a positive performance claim.

## Abstract Scaffold

Farmland spatial layout optimization requires information about both spatial
configuration and environmental suitability, but explicit soil, irrigation, and
productivity variables are often unavailable at planning scale. This study
examines whether geospatial foundation-model (GeoFM) embeddings can provide
latent remote-sensing proxies for farmland suitability in a block-level
reinforcement-learning planning workflow.

We construct a reproducible Paper11 pipeline that aggregates AlphaEarth
embeddings and explicit planning features to real Bishan DLTB polygons, defines
B0/B1/B2/B3 representation contracts, and builds tiled environment interfaces
for large real planning instances. The current repository implements the
feature pipeline, weak-label diagnostics, real-data tiling, MaskablePPO API
readiness checks, and a deterministic base planning reward for explicit-feature
and GeoFM-enhanced base-reward variants. It also runs a bounded same-tile B0/B1
training pilot that records cross-tile learned-policy evaluation as blocked by
the current tile-size-specific flat observation design. A subsequent per-block
scorer pilot trains on one tile and evaluates on a distinct tile, demonstrating
a variable-block-count interface without claiming final policy performance.
The current multi-tile, multi-seed scorer pilot broadens this interface check
across distinct evaluation tiles and seeds while remaining pilot evidence only.
A multi-seed same-tile B0/B1 MaskablePPO pilot reports a positive B1-B0
learned-policy mean reward delta under a short training budget, but this remains
preliminary until longer training, ablations, suitability-reward validation, and
held-out-region transfer tests are complete. The current IJAEOG evidence
package summarizes these pilot outputs and remaining gaps without creating new
policy-performance, transfer, or suitability-reward evidence. A padded
variable-size held-out Bishan tile B0/B1 learned-policy pilot further shows that
the learned policy can be trained on one Bishan tile and evaluated on a distinct
held-out Bishan tile under the deterministic base planning reward, while still
excluding suitability reward, B2/B3, cross-region transfer, and submission-level
performance claims. The Phase 26 analysis package converts Phase 25 outputs
into main empirical tables. The current macOS result sets do not support a
positive B1-over-B0 learned-policy claim: the 1024-step mean delta is
`-0.4329022862`, and the 4096-step mean delta is `-0.1318712688` with only
`3 / 9` positive tile-seed pairs.

[Evidence needed: B0/B1 stability diagnosis, representation ablation results,
suitability-reward validation, B2/B3 training and evaluation results,
cross-region transfer results, and final spatial diagnostics.]

The final manuscript should conclude only from completed comparisons. Until
those results exist, the contribution should be framed as a reproducible
GeoFM-enhanced representation and experiment platform, not as evidence that a
learned policy improves farmland planning.

## Cover Letter Scaffold

Dear Editor,

We submit the manuscript entitled "[final title]" for consideration in
International Journal of Applied Earth Observation and Geoinformation. The
paper addresses the need for environmental suitability information in farmland
spatial layout optimization, where direct soil, irrigation, and productivity
data are often unavailable.

The manuscript introduces a GeoFM-enhanced block-level planning workflow that
aggregates AlphaEarth embeddings to farmland planning units and evaluates
whether these latent remote-sensing representations improve reinforcement
learning-based spatial optimization. The study is designed to retain explicit
planning constraints while testing whether GeoFM-derived information improves
representation, suitability-aware decision making, and cross-region transfer.

[Evidence needed: one paragraph summarizing completed quantitative results and
their main implications.]

The work should be of interest to readers of IJAEOG because it connects
Earth-observation foundation-model representations with a practical land-use
planning and optimization problem, while keeping clear boundaries around proxy
suitability and direct agronomic measurement.

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
Phase 1-26 workflow is available at:

```text
https://github.com/zhouning/paper11-geofm-farmland-suitability-rl
```

The repository includes a reproducibility guide, file manifest, test suite, and
lightweight sample data. At the time of this draft, `python scripts\smoke_check.py`
and `python -m pytest tests -q` pass on the local submission-preparation
environment.

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

> AlphaEarth and other GeoFM embeddings are treated as latent remote-sensing
> proxies for environmental and land-surface conditions related to farmland
> suitability. They are not described as direct measurements of soil quality,
> fertility, irrigation access, or productivity.

Current blocked boundary:

> Do not claim learned policy superiority, suitability-reward improvement, or
> transfer improvement until bounded training, evaluation, ablation, and
> held-out-region tests have been completed. The current Phase 20 pilot is
> same-tile only, and Phase 21 is a cross-tile interface pilot rather than
> final transfer or policy-performance evidence. Phase 22 broadens that scorer
> interface pilot across multiple evaluation tiles and seeds, but still does
> not provide final transfer or policy-performance evidence. Phase 23 adds
> multi-seed same-tile learned-policy evidence, but still does not provide
> final transfer, suitability-reward, ablation, or submission-level performance
> evidence. Phase 24 is a synthesis and claim-readiness package only; it
> records current pilot evidence and remaining gaps, but does not create new
> policy-performance, transfer, or suitability-reward evidence. Phase 25 adds
> a padded variable-size held-out Bishan tile B0/B1 learned-policy pilot under
> deterministic base planning reward, but it still does not enable suitability
> reward, test B2/B3, demonstrate cross-region transfer, or support
> submission-level performance claims. Phase 26 adds the main B0/B1 held-out
> analysis package and the current macOS artifacts do not support positive
> multi-tile B1-over-B0 results. The 4096-step learned-policy mean delta is
> negative, so the next step is diagnosis rather than manuscript performance
> claiming.
