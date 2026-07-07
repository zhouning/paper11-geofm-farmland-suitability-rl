# Paper11 Formal Submission Files

This folder contains generated delivery files for the Phase 54 conclusion-type
Paper11 submission package, plus the current LaTeX/PDF formal export.

## Core Files

- `Paper11_formal_conclusion_manuscript.docx`: word-processing manuscript file
  generated from `paper/submission/04_formal_conclusion_manuscript.md` with
  Pandoc.
- `Paper11_formal_conclusion_manuscript.md`: editable copy of the formal
  conclusion manuscript source included for transfer convenience.
- `Paper11_formal_conclusion_manuscript.tex`: standalone LaTeX export of
  the Phase 54 formal conclusion manuscript for journal upload.
- `Paper11_formal_conclusion_manuscript.pdf`: 12-page PDF generated from
  the LaTeX file with `pdflatex`; this is the current formal PDF submission file.
- `Paper11_cover_letter_and_declarations.docx`: word-processing cover-letter
  and declarations file generated with Pandoc.
- `Paper11_cover_letter_and_declarations.md`: editable cover-letter,
  declaration, data availability, code availability, AI-assisted tools, and
  claim-boundary text.

## Bundle Files

- `Paper11_phase46_submission_contents_sha256.txt`: SHA256 checksums for the
  files included in the bundle.
- `Paper11_phase46_submission_bundle.zip`: transfer archive containing the core
  files and the content-checksum file.
- `Paper11_phase46_submission_bundle_sha256.txt`: SHA256 checksum for the zip
  archive itself.

## Claim Boundary

This is a bounded positive compressed-GeoFM representation manuscript package.
It does not claim raw GeoFM B1 superiority, B2/B3 readiness,
suitability-reward improvement, cross-region transfer, or independently
validated agronomic suitability.

Current conclusion: raw GeoFM state injection remains unsupported, but Phase 48
supports compressed GeoFM state routes (`D4P8` and `D4P16`) under the current
Bishan base-reward held-out protocol, Phase 49 reports this compressed route as
row-level statistically robust, Phase 50 reports directional sign-only cluster
support, and Phase 51 reports magnitude-sensitive cluster support
(`cluster_magnitude_support`). Phase 52 expands the same six-variant protocol
to five held-out tiles and three seeds, again supporting the compressed route
with row-level sign-test p `0.0066881634` and cluster signed-rank p
`0.0206298828`. Phase 53 reports `cluster_mean_support`, exact sign-flip p
`0.0196838379`, bootstrap CI95 `[0.0570820445, 0.5823557658]`, and positive
leave-one cluster, tile, and seed means. Phase 54 reports
`artifact_lineage_consistent`, verifying that the formal Phase 52/53 values are
reproducible from one authoritative artifact chain. The current suitability-reward route
remains blocked until an external independent-label registry passes Phase 40
and a calibrated low-dimensional prior passes Phase 41.

## Before Journal Upload

The authors still need to supply final author metadata, funding, CRediT roles,
reference formatting, final data-access wording for the external DLTB input,
any required journal figure files, and a release tag, immutable commit hash, or
archive DOI.