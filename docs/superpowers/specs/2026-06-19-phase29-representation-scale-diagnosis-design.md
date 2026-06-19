# Phase 29 Representation-Scale Diagnosis Design

## Goal

Add a read-only Phase 29 diagnosis that tests whether the current raw B1
64-dimensional GeoFM representation has scale, norm, and redundancy properties
that could make the current PPO setup harder to optimize than the
PCA-compressed D4P8/D4P16 controls observed in Phase 28.

The diagnostic question is:

```text
Do the existing B1, D4P8, and D4P16 feature tables show representation scale,
row-norm, or redundancy patterns that plausibly explain why compressed
controls can exceed raw B1 under the current Phase 28 protocol?
```

## Motivation

Phase 28 reported `compression_matches_raw` at both 1024 and 4096 training
steps. The 4096-step run showed that D4P8 and D4P16 exceeded raw B1 while
selecting almost disjoint block sets. The Phase 28 compression diagnosis also
showed high PCA concentration in the raw B1 embedding matrix.

The next step should not be another PPO rerun. A new training run would mix
representation-scale questions with fresh stochastic policy optimization.
Phase 29 should therefore be read-only and should inspect the existing feature
tables and tile index before proposing any normalization or rerun.

## Inputs

Phase 29 consumes:

```text
experiments/phase11_bishan_dltb_real/outputs/phase2_real/variant_B1_features.csv
experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P8_features.csv
experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P16_features.csv
experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv
```

It may also consume a Phase 28 summary CSV to identify train/evaluation tiles,
but the diagnosis must remain valid without that optional summary.

## Analysis

The module should compute:

- global scale summaries for B1 raw embeddings, D4P8, and D4P16;
- B1 row-norm summaries under raw, column z-score, row L2, and z-score plus
  row L2 transformations;
- B1 tile-level scale summaries using the Phase 13 tile index;
- raw B1 PCA concentration, numerical rank, and effective-rank diagnostics;
- a conservative status and interpretation that describe possible
  optimization difficulty without claiming causality.

## Outputs

Write artifacts under a requested output directory:

```text
phase29_variant_scale_summary.csv
phase29_tile_scale_summary.csv
phase29_b1_normalization_profiles.csv
phase29_representation_scale_diagnosis.json
phase29_representation_scale_diagnosis.md
```

## Claim Boundary

Phase 29 is a read-only feature-table diagnosis. It does not run new policy
training, does not alter rewards, does not test B2/B3, does not prove that PCA
is intrinsically superior, and does not prove that normalization would improve
PPO performance.

## Public API

Create:

```text
src/paper11_geofm/phase29_representation_scale_diagnosis.py
```

with:

```python
PHASE29_REPRESENTATION_SCALE_CLAIM_BOUNDARY = "..."

def build_phase29_representation_scale_diagnosis(...):
    ...

def write_phase29_representation_scale_diagnosis_artifacts(...):
    ...
```

Create:

```text
experiments/phase29_representation_scale_diagnosis/run_phase29_representation_scale_diagnosis.py
```

The CLI should accept the four required CSV paths, an optional Phase 28 summary
CSV, an output directory, and a rank threshold.

## Success Criteria

Phase 29 is successful when:

1. tests verify the analysis, writer, and CLI behavior on small fixtures;
2. the real Bishan feature tables produce all expected artifacts;
3. the Markdown interpretation keeps the causal boundary explicit;
4. README, reproduction guide, result package, and file manifest reference the
   new diagnostic;
5. regression checks pass.

## Spec Self-Review

- Placeholder scan: no placeholder requirements remain.
- Consistency check: the design is read-only and does not introduce new PPO
  training, suitability reward, B2/B3, or transfer claims.
- Scope check: the work is one focused implementation package.
- Ambiguity check: inputs, outputs, diagnostics, API, and boundaries are
  explicit.
