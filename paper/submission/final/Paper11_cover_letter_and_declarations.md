# Paper11 Cover Letter and Declarations

## Cover Letter Draft

Dear Editor,

We submit the manuscript entitled "Compressed GeoFM representations improve
held-out farmland layout optimization under evidence gates" for consideration
in International Journal of Applied Earth Observation and Geoinformation. The
manuscript examines whether geospatial foundation-model (GeoFM) embeddings can
serve as controlled state representations for reinforcement-learning farmland
layout optimization.

The study reports a reproducible Bishan case-study workflow that links
AlphaEarth embeddings to real DLTB planning blocks and evaluates GeoFM-enhanced
representations under held-out reinforcement-learning planning protocols,
representation controls, suitability-proxy diagnostics, independent-label gates,
and a calibrated-prior gate. The central finding is positive but bounded. Raw
64-dimensional GeoFM direct injection does not stably outperform
explicit-feature B0, but Phase 48 shows that compressed GeoFM state routes
D4P8 and D4P16 outperform B0, raw B1, random D2, and shuffled D3 on mean
held-out base-reward policy reward. Phase 49 further shows that this compressed
route is statistically robust in the current Bishan protocol, with pooled
sign-test p `0.0031549137` and bootstrap CI95 `[0.2827829983, 0.6639974489]`.
Phase 50 provides a conservative cluster-level boundary: `7 / 9` tile-seed
clusters are positive, the cluster sign-test p is `0.08984375`, and Phase 51 exact signed-rank testing supports the cluster magnitude effect with p `0.01953125`.
Phase 52 expands the same six-variant protocol to five held-out tiles and three
seeds, again supporting the compressed route with pooled delta `0.2921767818`,
`74 / 120` positive row-level comparisons, row-level sign-test p
`0.0066881634`, and cluster signed-rank p `0.0206298828`. Phase 53 supports the
expanded cluster mean with exact sign-flip p `0.0196838379`, bootstrap CI95
`[0.0570820445, 0.5823557658]`, and positive leave-one cluster, tile, and seed
means. Suitability-reward
and B2/B3 claims remain blocked by independent-label and calibrated-prior gates.

This manuscript should be relevant to readers of IJAEOG because it provides a
reproducible test for using Earth-observation foundation-model embeddings in
operational land-use optimization. Rather than assuming that a semantically
rich remote-sensing embedding is automatically a validated suitability variable,
the study shows that representation design is decisive: compressed GeoFM state
inputs support the base-reward planning policy, while raw injection and
suitability-reward routes require stricter evidence.

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

This submission package supports a bounded positive compressed-GeoFM
representation manuscript. It does not support claims of raw GeoFM B1
superiority, B2/B3 readiness, suitability-reward improvement, cross-region
transfer, or independently validated agronomic suitability. The current
defensible conclusion is that compressed GeoFM state routes improve mean
held-out base-reward policy reward under the Bishan protocol and remain
positive under Phase 49 robustness checks and Phase 52 expanded replication,
with Phase 50 and Phase 52 cluster-level evidence remaining directional by sign
test but supported by signed-rank magnitude testing and Phase 53 cluster-mean
influence checks, while raw GeoFM state injection and the current
suitability-reward route remain unsupported under the completed Paper11
evidence gates.