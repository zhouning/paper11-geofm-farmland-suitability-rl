# Phase 24 IJAEOG Evidence Package Design

## Goal

Create a reviewer-facing IJAEOG evidence package that consolidates Phase 22 and
Phase 23 pilot outputs into manuscript-safe tables, JSON, and Markdown claim
readiness summaries.

## Problem

Phase 22 and Phase 23 now provide useful pilot evidence, but their conclusions
are easy to overstate:

- Phase 22 is a multi-tile, multi-seed per-block scorer interface pilot, not
  PPO training or transfer evidence.
- Phase 23 is a multi-seed same-tile B0/B1 MaskablePPO pilot, not cross-tile or
  held-out-region evidence.

IJAEOG review needs clear separation between supported pilot observations and
unsupported final manuscript claims. A small executable evidence package should
read the generated Phase 22/23 artifacts and produce a compact claim-readiness
record that can be cited in the submission readiness audit and manuscript
drafting notes.

## Inputs

Phase 24 consumes:

- Phase 22 summary CSV;
- Phase 23 summary CSV;
- Phase 23 comparison JSON;
- output directory.

## Protocol

1. Read Phase 22 summary rows and aggregate mean reward by variant and policy.
2. Read Phase 23 summary rows and comparison JSON.
3. Build a claim-readiness table for:
   - B0/B1 same-tile learned-policy pilot;
   - multi-tile scorer interface pilot;
   - suitability reward;
   - transfer;
   - full IJAEOG submission.
4. Write a compact CSV evidence table, JSON summary, and Markdown claim
   readiness note.

## Outputs

Phase 24 writes:

- `phase24_ijaeog_evidence_table.csv`;
- `phase24_ijaeog_evidence_summary.json`;
- `phase24_ijaeog_claim_readiness.md`.

The outputs must make `submission_ready` false until longer-budget/full
B0/B1/B2/B3 experiments, ablations, suitability reward validation, transfer,
and final figures are complete.

## Claim Boundary

Phase 24 is a synthesis and claim-readiness package. It summarizes current
pilot evidence and remaining gaps; it does not create new policy-performance,
transfer, or suitability-reward evidence.

## Real Bishan Expected Result

For current real Bishan artifacts, Phase 24 should report:

- Phase 22 summary rows: 24;
- Phase 23 summary rows: 18;
- Phase 23 B1-B0 learned-policy mean reward delta: `0.4273019432`;
- same-tile B0/B1 pilot readiness: `pilot_supported`;
- multi-tile scorer interface readiness: `pilot_supported`;
- suitability reward readiness: `not_ready`;
- transfer readiness: `not_ready`;
- submission readiness: `not_ready`.

## Implementation Units

- `src/paper11_geofm/ijaeog_evidence_package.py`: input parsing, aggregation,
  claim-readiness table construction, artifact writing.
- `experiments/phase24_ijaeog_evidence_package/run_phase24_ijaeog_evidence_package.py`:
  CLI runner.
- `tests/test_phase24_ijaeog_evidence_package.py`: tests for aggregation,
  claim readiness, writer, and CLI output.
- Documentation updates in `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`,
  `reproducibility/FILE_MANIFEST.tsv`, and submission-readiness materials.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Consistency check: Phase 24 only summarizes Phase 22/23 outputs and does not
  claim new experimental evidence.
- Scope check: this phase is limited to evidence packaging and claim-readiness
  auditing.
- Ambiguity check: inputs, outputs, readiness states, and claim boundary are
  explicit.
