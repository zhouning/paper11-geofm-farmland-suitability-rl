# Phase 30 Normalized-B1 Ablation Design

## Goal

Add a bounded Phase 30 follow-up experiment that tests the Phase 29
optimization-difficulty hypothesis under the same padded held-out Bishan
base-reward protocol used in Phase 28.

The diagnostic question is:

```text
If raw B1 is replaced by column-standardized B1 variants, does the learned
policy result improve relative to raw B1 strongly enough to support the narrow
hypothesis that representation scaling, rather than GeoFM semantics alone,
contributed to the Phase 28 compressed-control advantage?
```

## Motivation

Phase 28 reported `compression_matches_raw` at both 1024 and 4096 training
steps. Phase 29 then showed that raw B1 is highly redundant and has smaller
per-dimension scale than D4P8 and D4P16, while raw B1 row norms are already
stable.

The next rigorous step should not expand to B2/B3 or reward redesign. The
cleanest follow-up is a bounded normalization ablation that changes only the
representation transform while keeping:

- the same explicit features;
- the same Phase 13 tile index;
- the same deterministic base planning reward;
- the same padded held-out MaskablePPO evaluation structure;
- the same D2, D3, D4P8, and D4P16 controls for context.

## Inputs

Phase 30 consumes:

```text
experiments/phase11_bishan_dltb_real/outputs/phase2_real/
experiments/phase8_ablation_controls/outputs/real_bishan_controls/
experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv
```

The Phase 2 directory must contain ready `B0` and `B1` variant tables. The
Phase 8 directory must contain ready `D2`, `D3`, `D4P8`, and `D4P16` tables.

## Variant Design

Phase 30 adds two normalized B1 diagnostic variants:

| Variant | Definition | Purpose |
|---|---|---|
| `N1Z` | explicit features plus column-centered, standard-deviation-scaled B1 embeddings | tests whether exposing per-coordinate variation helps optimization |
| `N1ZR` | explicit features plus column-centered, standard-deviation-scaled, then row-L2-normalized B1 embeddings | tests whether removing post-z-score row-norm variation changes the result |

Both variants keep the original `embedding_mean_00` to `embedding_mean_63`
column names so they remain compatible with the existing tiled-input and
Phase 25/28 policy code.

## Experimental Scope

Phase 30 should evaluate:

```text
B0, B1, N1Z, N1ZR, D2, D3, D4P8, D4P16
```

under the same bounded padded held-out structure already used in Phase 28:

- one training tile;
- up to three distinct held-out evaluation tiles;
- explicit seeds;
- deterministic base reward only;
- no suitability reward;
- no B2/B3;
- no transfer-region expansion.

## Analysis

The Phase 30 analysis should compute:

- mean learned-policy reward by variant;
- tile-seed delta rows for `N1Z` minus `B1` and `N1ZR` minus `B1`;
- tile-seed delta rows for normalized variants against `B0`, `D4P8`, and
  `D4P16`;
- a conservative status that distinguishes:
  - no normalization support;
  - normalized B1 improves raw B1 but still does not recover the B0 gap;
  - normalized B1 improves raw B1 and recovers the B0 gap;
  - normalized B1 improves raw B1 and also matches or exceeds the compressed
    controls.

The interpretation must stay descriptive. Even a positive result would support
only a narrow optimization explanation under the current Bishan protocol.

## Outputs

Write artifacts under a requested Phase 30 output directory:

```text
derived_normalized_controls/
phase30_normalized_b1_summary.csv
phase30_normalized_b1_traces.json
phase30_normalized_b1_comparison.json
phase30_normalized_b1_delta_table.csv
phase30_normalized_b1_readiness.md
```

The `derived_normalized_controls/` directory should contain a small manifest and
ready-only feature tables for `N1Z` and `N1ZR`.

## Claim Boundary

Phase 30 is a bounded representation-only ablation under the existing Bishan
base-reward held-out protocol. It does not validate suitability reward, does
not test B2/B3, does not test cross-region transfer, does not prove that
normalization is generally beneficial, and does not support submission-level
planning-performance claims.

## Public API

Create:

```text
src/paper11_geofm/phase30_normalized_b1_ablation.py
```

with a focused API that includes:

```python
PHASE30_CLAIM_BOUNDARY = "..."

def build_phase30_normalized_b1_controls(...):
    ...

def write_phase30_normalized_b1_controls(...):
    ...

def run_phase30_normalized_b1_ablation(...):
    ...

def build_phase30_normalized_b1_analysis(...):
    ...

def write_phase30_normalized_b1_artifacts(...):
    ...
```

Create:

```text
experiments/phase30_normalized_b1_ablation/run_phase30_normalized_b1_ablation.py
```

The CLI should support:

- `run-and-analyze`;
- `analyze-only`;
- explicit Phase 2, Phase 8, tile-index, and output paths;
- explicit variants, seeds, timesteps, and eval-step settings;
- an optional normalized-control output directory override;
- an optional existing Phase 28 control summary CSV so the experiment can train
  only `N1Z` and `N1ZR` while reusing verified `B0`, `B1`, `D2`, `D3`,
  `D4P8`, and `D4P16` control rows.

## Success Criteria

Phase 30 is successful when:

1. tests cover normalized-control generation, summary analysis, artifact
   writing, and CLI behavior;
2. the derived normalized control tables remain compatible with
   `load_variant_input()` and tiled Phase 25/28 policy code;
3. real-data execution writes the expected artifacts without modifying the
   existing Phase 2 or Phase 8 outputs in place;
4. the optional incremental path avoids redundant retraining of already-frozen
   Phase 28 control variants;
5. the Markdown interpretation keeps the claim boundary explicit;
6. regression checks pass.

## Spec Self-Review

- Placeholder scan: no TODO or TBD markers remain.
- Consistency check: the design stays within the Phase 28/29 representation
  branch and does not introduce reward or transfer changes.
- Scope check: this is one bounded experiment package plus one derived-input
  helper path.
- Ambiguity check: the normalized variants, comparison targets, and artifact
  expectations are explicit.
