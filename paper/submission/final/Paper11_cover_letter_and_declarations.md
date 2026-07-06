# Paper11 Cover Letter and Declarations

## Cover Letter Draft

Dear Editor,

We submit the manuscript entitled "Evidence-gated rejection of unsupported GeoFM
superiority in reinforcement-learning farmland layout optimization" for
consideration in International Journal of Applied Earth Observation and
Geoinformation. The manuscript examines whether geospatial foundation-model
(GeoFM) embeddings can be treated as decision-ready representations for
reinforcement-learning farmland layout optimization.

The study reports a reproducible Bishan case-study workflow that links
AlphaEarth embeddings to real DLTB planning blocks and evaluates GeoFM-enhanced
representations under held-out reinforcement-learning planning protocols,
representation controls, suitability-proxy diagnostics, independent-label gates,
and a calibrated-prior gate. The central finding is negative and evidence
bounded. Raw GeoFM B1 does not stably outperform explicit-feature B0, compressed
representation controls exceed raw B1, normalized-B1 and budget checks do not
rescue the claim, and the suitability-reward route is stopped by independent
label and calibrated-prior gates.

This manuscript should be relevant to readers of IJAEOG because it provides a
reproducible cautionary test for using Earth-observation foundation-model
embeddings in operational land-use optimization. Rather than assuming that a
semantically rich remote-sensing embedding is a validated suitability variable,
the study shows how representation controls and independent-label gates can
prevent unsupported planning claims.

The manuscript is original and is not under consideration elsewhere. All authors
must confirm authorship, approval of the final manuscript, funding statements,
competing interests, data availability wording, and journal-specific disclosure
requirements before upload.

Sincerely,

[corresponding author name and contact information required]

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or
personal relationships that could have appeared to influence the work reported
in this paper.

Author confirmation is required before submission.

## Funding

[author-supplied funding statement required]

## Author Contributions

[author-supplied CRediT author contribution statement required]

## Ethics Statement

This study uses remote-sensing and land-use planning data and does not involve
human participants, human tissue, animal experiments, or private personal data.
Author confirmation is required before submission.

## Data Availability

The lightweight Bishan AlphaEarth sample arrays, code, reproduction scripts,
and file manifests needed for reviewer smoke tests are available in the project
repository:

```text
https://github.com/zhouning/paper11-geofm-farmland-suitability-rl
```

The final submitted version should cite a release tag, immutable commit hash,
or archive DOI. Large derived arrays, trained weights, and full external data
products not included in ordinary Git should be deposited in an external archive
before final submission, with checksums and access instructions. The real Bishan
DLTB-with-slope GeoPackage used for local Phase 11 reproduction is an external
source and is not redistributed in this repository. The final availability
wording must state whether that source can be shared, requested, or reproduced
only by authorized users.

## Code Availability

All reviewer-facing code is maintained in the Paper11 repository:

```text
https://github.com/zhouning/paper11-geofm-farmland-suitability-rl
```

The final submitted version should cite a release tag, immutable commit hash,
or archive DOI.

## AI-Assisted Tools Statement

AI-assisted tools were used for language editing, code assistance, and
submission-preparation drafting. All scientific claims, code outputs,
references, data interpretations, and manuscript text must be checked and
approved by the authors before submission. The final wording should be revised
to match the selected journal policy and the authors' actual use.

## Claim Boundary for Upload

This submission package supports only a bounded negative/evidence-gated
manuscript. It does not support claims of GeoFM superiority, B2/B3 readiness,
suitability-reward improvement, cross-region transfer, or independently
validated agronomic suitability. The current defensible conclusion is that raw
GeoFM state injection and the current suitability-reward route are unsupported
under the completed Paper11 evidence gates.