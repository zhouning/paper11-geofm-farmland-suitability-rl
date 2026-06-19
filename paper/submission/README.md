# Paper11 Submission Package

This folder tracks manuscript-submission preparation for Paper11. It is separate
from the executable reproducibility workflow because the repository is now a
reviewer-code package, while the manuscript is still evidence-gated.

## Files

- `01_ijaeog_submission_readiness.md`: target-journal readiness audit for
  International Journal of Applied Earth Observation and Geoinformation
  (IJAEOG).
- `02_draft_titles_highlights_declarations.md`: guarded title, highlight,
  keyword, cover-letter, and declaration text that can be reused after the
  experimental evidence is complete.

## Current Status

As of 2026-06-19, the code/reproducibility repository is ready to cite as a
reviewer-facing artifact, but the Paper11 research manuscript is not yet ready
for positive performance claims. Phase 28 now adds B0/B1/D2/D3/D4
representation-control diagnostics at 1024 and 4096 training steps, and both
runs report `compression_matches_raw` rather than a supported raw-B1
representation signal.

Do not submit a main research article claiming model superiority until the
current negative representation-control evidence is resolved by validated
suitability-reward evidence, stronger B2/B3 experiments, transfer tests, and
spatial case diagnostics.
