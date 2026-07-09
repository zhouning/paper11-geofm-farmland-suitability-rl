# Phase 69 Label-Free Evidence Synthesis Gate Design

## Purpose

Phase 69 follows the Phase 68 external independent-label package. Phase 68 made
the independent-label route concrete, but the real Bishan run still has no
external label CSV and no registry. Phase 67 also showed that current artifacts
cannot support reward redesign from existing labels or diagnostic targets.

The next step should not wait passively for external data, train another policy
variant, rewrite rewards, or revise the formal manuscript. Phase 69 should
build a read-only synthesis gate that asks what Paper11 can still defend from
label-free evidence alone.

Phase 69 is algorithm and experiment-evidence work. It does not revise formal
submission files and it does not make new manuscript-level claims.

## Scientific Question

What is the strongest defensible algorithm claim that remains after combining
the compressed-route evidence, mechanism audits, matched-control limits,
reward-target audits, and external-label readiness state?

The expected output is a conservative cross-phase decision. It should identify
whether Paper11 currently supports only a bounded low-dimensional compressed
state route, whether even that label-free claim is insufficient, or whether any
claim requires external independent labels before it can proceed.

## Why Phase 69 Is Needed

The Paper11 evidence chain is now broad but fragmented:

- Phase 48/52 and Phase 53 support a positive compressed-state route under the
  Bishan base-reward protocol.
- Phase 57 supports a mechanism-consistent geometry interpretation: D4P8 and
  D4P16 preserve most raw GeoFM variance while reducing effective rank and
  covariance conditioning burden.
- Phase 59 and Phase 62 prevent stronger matched-dimension, GeoFM-specific, or
  PCA-optimality claims.
- Phase 66 shows that the current base reward is almost fully explained by
  explicit planning features and can mask GeoFM representation signal.
- Phase 67 finds zero candidate reward/label targets that can justify reward
  redesign without independent labels.
- Phase 68 provides an external-label package, but no external independent
  label has been supplied.

These results are scientifically useful only if their boundaries are made
machine-checkable. Phase 69 should synthesize them into a reproducible gate that
separates defensible algorithm evidence from blocked suitability, reward, and
agronomic claims.

## Scope

Phase 69 should be read-only and reproducible from existing artifacts. It should
not rerun PPO, generate new feature tables, alter rewards, create B2/B3
variants, or edit `paper/submission/final/*`.

Included:

- load existing result JSON and/or paper-facing result notes for the relevant
  phases;
- reduce each phase into a normalized evidence-axis row;
- check whether required evidence axes are present and internally consistent;
- assign a conservative synthesis status;
- write JSON, CSV, and Markdown artifacts;
- record a paper-facing Phase 69 result note under `paper/phase28_results`.

Excluded:

- training or fine-tuning policies;
- changing reward definitions;
- calibrating a suitability prior;
- using DLTB/slope/source metadata as independent labels;
- treating Phase 68 templates as supplied external labels;
- revising formal submission files;
- claiming suitability reward, B2/B3, transfer, agronomic validity, PCA
  optimality, or GeoFM-specific matched-dimension superiority.

## Evidence Axes

Phase 69 should summarize five axes.

### Route Support

Inputs should include the compressed-route performance evidence from Phase 48,
Phase 52, and/or Phase 53. The axis should pass when the existing artifacts show
positive low-dimensional compressed route evidence under the expanded Bishan
base-reward protocol.

The axis should not claim suitability or agronomic validity. It supports only a
bounded base-reward protocol result.

### Mechanism Support

Inputs should include Phase 57. The axis should pass when compressed GeoFM
features preserve most raw GeoFM variance while lowering effective rank and
conditioning burden.

This axis supports a geometry-consistency interpretation, not PCA optimality.

### Mechanism Limits

Inputs should include Phase 59 and Phase 62. The axis should pass as a limiting
axis when matched-dimension and D6 projection controls block stronger claims.
This is a successful boundary check, not a failure of the whole paper.

This axis should explicitly prevent:

- GeoFM-specific same-dimension advantage claims;
- PCA optimality claims;
- claims that D4P8/D4P16 uniquely outperform all matched low-dimensional
  controls.

### Reward And Target Limits

Inputs should include Phase 66 and Phase 67. The axis should pass as a limiting
axis when it records that reward redesign and candidate-target routes remain
blocked.

This axis should explicitly prevent:

- reward-redesign claims;
- suitability-reward readiness;
- claims that existing weak labels can support an independent reward route.

### External Label State

Inputs should include Phase 68. The axis should pass as a readiness-but-blocked
axis when the external-label package is ready but no external label CSV or
registry has been supplied.

This axis should explicitly prevent treating a template-only package as a
Phase 40-passed label.

## Status Model

Phase 69 should return one top-level status:

- `bounded_label_free_algorithm_claim_supported`: route support and mechanism
  support pass, limiting axes are present, and no blocked claim is promoted.
- `claim_must_be_narrowed_to_low_dimensional_route`: current evidence supports
  only a bounded low-dimensional compressed state route, while mechanism,
  reward, target, and label limits block stronger claims.
- `external_label_required_for_suitability_or_reward_claims`: label-free
  evidence is not enough for suitability, reward redesign, or agronomic claims.
- `label_free_evidence_insufficient`: required route-support or mechanism
  evidence is missing or contradictory.

The current expected real-run status is:

`claim_must_be_narrowed_to_low_dimensional_route`

This means Paper11 can continue to defend a bounded algorithm/evidence
contribution about low-dimensional compressed state routes under the Bishan
base-reward protocol, but cannot defend suitability reward, GeoFM-specific
matched-dimension superiority, PCA optimality, or independent agronomic
validity.

## Architecture

Add one focused module:

`src/paper11_geofm/phase69_label_free_evidence_synthesis_gate.py`

Responsibilities:

- own the Phase 69 claim boundary;
- load required JSON or Markdown result artifacts;
- normalize each phase into an evidence-axis row;
- classify each axis as `support`, `limit`, `blocked`, `missing`, or
  `contradictory`;
- compute the top-level synthesis status;
- write CSV, JSON, and Markdown artifacts.

Add one thin runner:

`experiments/phase69_label_free_evidence_synthesis_gate/run_phase69_label_free_evidence_synthesis_gate.py`

Responsibilities:

- parse paths to the relevant Phase 48/52/53/57/59/60/62/66/67/68 artifacts;
- call the module;
- print status, artifact paths, recommended next step, and claim boundary.

Add focused tests:

`tests/test_phase69_label_free_evidence_synthesis_gate.py`

Responsibilities:

- support rows produce bounded route support;
- missing support evidence returns `label_free_evidence_insufficient`;
- matched-control negative evidence narrows rather than invalidates the bounded
  route;
- Phase 66/67 blocks reward-redesign claims;
- Phase 68 template-only status blocks suitability/reward claims;
- artifact writer produces stable CSV, JSON, and Markdown files;
- CLI runner succeeds on fixtures.

## Data Flow

1. The runner receives paths to existing phase artifacts and an output
   directory.
2. The module loads each artifact and extracts only the status and core metrics
   needed for the synthesis gate.
3. Each phase contributes one or more normalized evidence-axis rows.
4. The synthesis gate checks required support axes, required limiting axes, and
   any blocked-claim violations.
5. The module writes:
   - `phase69_evidence_axes.csv`;
   - `phase69_claim_boundary_matrix.csv`;
   - `phase69_label_free_evidence_synthesis_gate.json`;
   - `phase69_label_free_evidence_synthesis_gate.md`.
6. A real-run result note records the conservative status, key axis decisions,
   reproduction command, and boundary.

## Claim Boundary Matrix

Phase 69 should explicitly map claims to allowed or blocked status:

- bounded low-dimensional compressed state route: allowed if support and
  mechanism axes pass;
- raw B1 superiority: blocked;
- PCA optimality: blocked;
- GeoFM-specific matched-dimension superiority: blocked;
- suitability reward readiness: blocked;
- B2/B3 reward integration: blocked;
- external independent-label readiness: package-ready only, not label-passed;
- independent agronomic suitability: blocked;
- cross-region transfer: blocked;
- formal submission readiness: out of scope.

## Error Handling

Hard errors should be reserved for unreadable or malformed required artifacts:

- missing required JSON or Markdown path;
- invalid JSON;
- expected status field absent from a JSON artifact when JSON is supplied;
- output directory cannot be written.

Recoverable evidence problems should be represented in output rows:

- an optional artifact is missing;
- an axis has incomplete metrics but an interpretable status;
- a phase reports a status that narrows or blocks claims;
- a phase has a status that is not recognized by Phase 69.

## Testing And Verification

Targeted tests:

- current-style fixture evidence yields
  `claim_must_be_narrowed_to_low_dimensional_route`;
- missing compressed-route support yields `label_free_evidence_insufficient`;
- Phase 59/62 matched-control limits block GeoFM-specific and PCA-optimality
  claims;
- Phase 66/67 block reward redesign;
- Phase 68 template-only package blocks suitability/reward claims;
- artifact writers produce stable CSV, JSON, and Markdown;
- CLI runner works on fixture artifacts.

Regression checks:

- Phase 69 tests;
- Phase 68 external independent-label package tests;
- Phase 67 candidate reward/label target audit tests;
- Phase 66 reward-label representation audit tests;
- smoke check;
- `git diff --check`;
- formal manuscript diff check under `paper/submission/final`.

## Expected Result Note

The real run should create:

`paper/phase28_results/35_phase69_label_free_evidence_synthesis_gate.md`

The note should report:

- top-level Phase 69 status;
- evidence-axis table summary;
- allowed bounded claim;
- blocked stronger claims;
- reproduction command;
- statement that no formal manuscript files changed.

## Claim Boundary

Phase 69 may claim that Paper11 has a reproducible label-free synthesis gate and
may state the strongest defensible algorithm claim under existing evidence.

Phase 69 must not claim that suitability reward is ready, that B2/B3 can run,
that GeoFM has a matched-dimension-specific advantage, that PCA is optimal, that
external independent labels are available, that agronomic suitability is
validated, or that the formal manuscript is ready for submission.
