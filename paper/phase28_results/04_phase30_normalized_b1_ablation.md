# Phase 30 Normalized-B1 Ablation

## One-Sentence Argument

Phase 30 provides partial support for the Phase 29 optimization hypothesis:
bounded normalization improves raw B1 and lifts the mean learned-policy reward
above B0 under the current 4096-step held-out Bishan protocol, but the
normalized variants still do not match the compressed controls.

## Terminology Ledger

| Canonical term | Definition | Boundary |
|---|---|---|
| Phase 30 | bounded held-out normalized-B1 follow-up under the existing base-reward protocol | no suitability reward, no transfer expansion |
| `N1Z` | explicit features plus column-centered and standard-deviation-scaled B1 embeddings | representation transform only |
| `N1ZR` | explicit features plus column-centered, standard-deviation-scaled, then row-L2-normalized B1 embeddings | representation transform only |
| incremental control reuse | reuse existing Phase 28 `B0,B1,D2,D3,D4P8,D4P16` summary rows while training only `N1Z/N1ZR` | runtime optimization, not a protocol change |
| recovery of the B0 gap | normalized B1 mean reward exceeds B0 mean reward in the bounded protocol | not a final manuscript claim |

## What Was Used

This follow-up used:

```text
experiments/phase11_bishan_dltb_real/outputs/phase2_real/
experiments/phase8_ablation_controls/outputs/real_bishan_controls/
experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv
experiments/phase28_representation_controls/outputs/real_bishan_4096/phase28_representation_control_summary.csv
```

Reproduce the incremental Phase 30 run with:

```text
python experiments/phase30_normalized_b1_ablation/run_phase30_normalized_b1_ablation.py --mode run-and-analyze --phase2-output-dir experiments/phase11_bishan_dltb_real/outputs/phase2_real --phase8-output-dir experiments/phase8_ablation_controls/outputs/real_bishan_controls --tile-index-csv experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv --existing-control-summary-csv experiments/phase28_representation_controls/outputs/real_bishan_4096/phase28_representation_control_summary.csv --variants B0,B1,N1Z,N1ZR,D2,D3,D4P8,D4P16 --total-timesteps 4096 --eval-max-steps 8 --seeds 0,1,2 --max-eval-tiles 3 --output-dir experiments/phase30_normalized_b1_ablation/outputs/real_bishan_4096_incremental
```

The generated artifacts are:

```text
derived_normalized_controls/
phase30_normalized_b1_summary.csv
phase30_normalized_b1_traces.json
phase30_normalized_b1_comparison.json
phase30_normalized_b1_delta_table.csv
phase30_normalized_b1_readiness.md
```

## Mean Learned-Policy Result

| Variant | Mean learned-policy reward |
|---|---:|
| B0 | `0.4825072170` |
| B1 | `0.3506359482` |
| D2 | `0.2761767826` |
| D3 | `0.4601109618` |
| D4P8 | `0.7274877829` |
| D4P16 | `0.9918299718` |
| N1Z | `0.6515323140` |
| N1ZR | `0.5772465716` |

Both normalized variants improve the raw B1 mean. `N1Z` and `N1ZR` also exceed
the explicit-only B0 mean under this bounded protocol. This is the first
representation-branch result in the current held-out package where the mean is
consistently above both raw B1 and B0.

## Focal Delta Result

| Comparison | Mean delta | Positive tile-seed count |
|---|---:|---:|
| N1Z - B1 | `0.3008963657` | `6 / 9` |
| N1ZR - B1 | `0.2266106233` | `4 / 9` |
| N1Z - B0 | `0.1690250969` | `4 / 9` |
| N1ZR - B0 | `0.0947393545` | `4 / 9` |
| N1Z - D4P8 | `-0.0759554690` | `4 / 9` |
| N1ZR - D4P8 | `-0.1502412114` | `2 / 9` |
| N1Z - D4P16 | `-0.3402976578` | `4 / 9` |
| N1ZR - D4P16 | `-0.4145834002` | `3 / 9` |

This pattern is important. The normalization transforms improve raw B1, and
`N1Z` improves raw B1 in `6 / 9` tile-seed pairs, but the recovery is still not
stable enough to dominate B0 or the compressed controls across tile-seed pairs.

## Status

The current Phase 30 status is:

```text
normalized_b1_recovers_b0_gap
```

This means the normalization branch recovered the mean B0 gap, but it did not
close the compressed-control gap.

## Interpretation

Phase 30 narrows the representation diagnosis substantially. The Phase 29
optimization-difficulty hypothesis now has direct experimental support:
representation scaling matters under the current padded MaskablePPO setup.
Normalizing raw B1 is not a cosmetic transform; it changes the learned-policy
result enough to move the mean above B0.

However, the result is still incomplete evidence. If normalization fully
explained the Phase 28 pattern, the normalized variants would be expected to
match or exceed D4P8/D4P16 more clearly. They do not. The current evidence
therefore supports a partial explanation:

1. raw-B1 scaling contributed to underperformance;
2. scaling alone does not explain the full compressed-control advantage;
3. the remaining gap may involve redundancy, optimization geometry, or
   selection behavior that is not removed by z-scoring alone.

## What This Supports

This follow-up supports three conservative claims:

1. representation normalization can improve raw B1 under the current bounded
   held-out Bishan protocol;
2. the Phase 29 scale diagnosis identified a real optimization lever rather
   than a purely descriptive artifact;
3. raw B1 should not be treated as the only fair representation form for the
   GeoFM branch in later bounded diagnostics.

## What This Does Not Support Yet

This follow-up does not support:

- a positive raw-B1 learned-policy claim;
- the claim that normalization is generally beneficial outside this protocol;
- the claim that normalized B1 is better than the compressed controls;
- suitability reward readiness, B2/B3 superiority, or transfer conclusions;
- submission-level planning-performance claims.

## Recommended Next Step

The representation branch should now pivot from “does normalization help?” to
“why do compressed controls still lead after normalization?” A rigorous next
step would be case-map and action-selection diagnostics for representative
held-out tiles, or another bounded robustness check that compares `N1Z`,
`N1ZR`, `D4P8`, and `D4P16` without reopening the full raw-B1 control package.

In parallel, the project should resume the blocked suitability-proxy validation
and spatial case-map work, because Phase 30 still does not justify a direct
move into B2/B3 manuscript claims.

## Claim Boundary

Phase 30 is a bounded representation-only ablation under the existing Bishan
base-reward held-out protocol. It does not validate suitability reward, does
not test B2/B3, does not test cross-region transfer, does not prove that
normalization is generally beneficial, and does not support submission-level
planning-performance claims.
