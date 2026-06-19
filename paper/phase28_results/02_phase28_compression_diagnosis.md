# Phase 28 Compression Diagnosis

## One-Sentence Argument

In the current 4096-step Phase 28 held-out Bishan diagnostic, we observe that
the PCA-compressed controls outperform raw B1 not because they simply reproduce
the same selected blocks, but because they drive substantially different block
selection patterns that carry better explicit base-reward components under the
current protocol, with the boundary that this remains a read-only diagnosis
rather than a causal explanation.

## Terminology Ledger

| Canonical term | Definition | Boundary |
|---|---|---|
| compression diagnosis | read-only follow-up analysis over existing Phase 28 outputs | no new training, no new reward family |
| raw B1 | explicit planning features plus raw 64-dimensional AlphaEarth embeddings | current B1 state design only |
| D4P8 / D4P16 | explicit planning features plus PCA-compressed GeoFM embeddings with 8 or 16 components | compression controls only |
| selection overlap | Jaccard overlap between the sets of selected block IDs for two variants on the same tile-seed pair | action-level diagnostic only |
| reward component mean | average contribution of each explicit base-reward term across selected blocks | not a causal representation metric |

## What Was Used

This read-only diagnosis used:

```text
experiments/phase28_representation_controls/outputs/real_bishan_4096/phase28_representation_control_summary.csv
experiments/phase11_bishan_dltb_real/outputs/phase2_real/variant_B1_features.csv
experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P8_features.csv
experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P16_features.csv
```

The numbers below are now reproducible through:

```text
python experiments/phase28_compression_diagnosis/run_phase28_compression_diagnosis.py --summary-csv experiments/phase28_representation_controls/outputs/real_bishan_4096/phase28_representation_control_summary.csv --phase2-b1-features-csv experiments/phase11_bishan_dltb_real/outputs/phase2_real/variant_B1_features.csv --d4p8-features-csv experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P8_features.csv --d4p16-features-csv experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P16_features.csv --output-dir experiments/phase28_compression_diagnosis/outputs/real_bishan_4096
```

The read-only artifacts are:

```text
experiments/phase28_compression_diagnosis/outputs/real_bishan_4096/phase28_compression_overlap.csv
experiments/phase28_compression_diagnosis/outputs/real_bishan_4096/phase28_compression_reward_components.csv
experiments/phase28_compression_diagnosis/outputs/real_bishan_4096/phase28_compression_diagnosis.json
experiments/phase28_compression_diagnosis/outputs/real_bishan_4096/phase28_compression_diagnosis.md
```

The scope matches the main 4096-step Phase 28 run:

- train tile: `tile_r003_c003`
- held-out tiles: `tile_r002_c003`, `tile_r005_c004`, `tile_r005_c003`
- seeds: `0, 1, 2`
- action horizon: `8`

## Selection-Overlap Result

Mean Jaccard overlap between raw B1 and each comparator:

| Comparator | Mean Jaccard overlap | Mean shared selected blocks | Mean B1 minus comparator reward |
|---|---:|---:|---:|
| B0 | `0.0000` | `0.000` | `-0.1318712688` |
| D2 | `0.4907` | `4.889` | `0.0744591656` |
| D3 | `0.4923` | `5.222` | `-0.1094750135` |
| D4P8 | `0.0148` | `0.222` | `-0.3768518347` |
| D4P16 | `0.0074` | `0.111` | `-0.6411940236` |

This pattern is sharp. Raw B1 shares roughly half of its selected blocks with
the random and shuffled controls, but it shares almost none of its selected
blocks with D4P8 or D4P16. The compressed controls therefore are not merely a
slightly denoised version of the same B1 policy trajectory. Under the current
policy fit, they induce a substantially different action sequence.

## Tile-Seed Examples

The low-overlap pattern appears on most tile-seed pairs:

- `tile_r002_c003`, seed `0`: B1 versus D4P8 overlap `0.0000`, B1-D4P8 delta
  `-1.0976874771`; B1 versus D4P16 overlap `0.0000`, B1-D4P16 delta
  `-1.4080247960`.
- `tile_r005_c003`, seed `1`: B1 versus D4P8 overlap `0.0000`, B1-D4P8 delta
  `-0.7649909116`; B1 versus D4P16 overlap `0.0000`, B1-D4P16 delta
  `-2.0293605320`.
- `tile_r005_c004`, seed `2`: B1 versus D4P8 overlap `0.0667`, B1-D4P8 delta
  `0.0635699712`; B1 versus D4P16 overlap `0.0000`, B1-D4P16 delta
  `-0.5268667928`.

These examples show that the compressed-control advantage is usually attached
to a distinct block-ranking behavior, not to minor perturbations around the B1
selection order.

## Explicit Reward-Component Result

Mean explicit reward-component contributions across selected blocks:

| Variant | low-slope farmland / orchard | current farmland / orchard | low slope | area score | mean-slope penalty | max-slope penalty | built-up penalty | water penalty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | `0.024306` | `0.105556` | `0.019444` | `0.013968` | `-0.059202` | `-0.019965` | `-0.033333` | `-0.006944` |
| D4P8 | `0.048611` | `0.111111` | `0.027778` | `0.013370` | `-0.056474` | `-0.018738` | `-0.031944` | `-0.002778` |
| D4P16 | `0.058333` | `0.116667` | `0.030556` | `0.010280` | `-0.050853` | `-0.016003` | `-0.020833` | `-0.004167` |

Relative to raw B1, both PCA-compressed controls select blocks with:

1. higher low-slope farmland-or-orchard contribution;
2. slightly higher current-farmland contribution;
3. weaker mean-slope and max-slope penalties;
4. weaker built-up and water penalties.

Under the current deterministic reward, those differences are directionally
consistent with the higher total rewards observed for D4P8 and D4P16.

## Compression-Scale Result

The underlying B1 embedding matrix is strongly compressible:

| Diagnostic | Value |
|---|---:|
| numerical rank above `1e-12` | `64` |
| top-8 PCA variance ratio | `0.858782` |
| top-16 PCA variance ratio | `0.949601` |
| mean raw-embedding column std | `0.037792` |
| mean D4P8 component std | `0.091246` |
| mean D4P16 component std | `0.062077` |

The first 8 PCA components retain about `85.9%` of the observed embedding
variance, and the first 16 retain about `95.0%`. The compressed controls also
present larger per-dimension standard deviations than the raw embedding
columns. This is consistent with a representation that is highly redundant in
its original 64-dimensional form, although redundancy alone does not prove why
the policy optimizer prefers the compressed controls.

## Interpretation

The current evidence supports a narrower explanation than the main Phase 28
diagnosis alone. Raw B1 does not simply lose because D4P8 or D4P16 are acting
as near-identical surrogates. Instead, the compressed controls produce almost
disjoint selected-block sets while landing on blocks with better explicit
base-reward profiles.

This suggests two plausible, but still unproven, interpretations:

1. the raw 64-dimensional embedding arm may be harder for the current PPO
   setup to optimize than a lower-dimensional compressed arm;
2. the dominant planning signal available to the current policy may live in a
   low-dimensional subspace of the GeoFM representation.

Both interpretations remain hypotheses. The present diagnosis does not isolate
optimization difficulty from semantic signal quality.

## What This Supports

This diagnosis supports three conservative claims:

1. the compressed-control advantage at 4096 steps is associated with different
   selected blocks, not with near-identical trajectories to raw B1;
2. those selected blocks have better explicit base-reward components on
   average than the raw-B1 selections;
3. the raw 64-dimensional embedding matrix is strongly compressible, so a
   lower-dimensional representation is a technically plausible next design.

## What This Does Not Support Yet

This diagnosis does not support:

- the claim that PCA is intrinsically superior to raw GeoFM semantics;
- the claim that low-dimensional compression is the sole cause of the
  performance difference;
- the claim that Paper11 is ready to replace B1 with D4P8 or D4P16 as a final
  manuscript model;
- any suitability-reward, B2/B3, or transfer conclusion.

## Recommended Next Step

The next rigorous step should be a dedicated read-only or lightweight
experiment that separates representation compression from optimization effects.
Examples include:

1. intermediate-budget Phase 28 checkpoints for B1, D4P8, and D4P16;
2. representation-scale diagnostics or normalization ablations before PPO
   training;
3. spatial case maps comparing B1 and D4 selected blocks on the same held-out
   tiles.

## Claim Boundary

This document is a read-only follow-up analysis over existing Phase 28 outputs.
It does not run new training, does not alter the reward, does not enable
suitability reward, does not test B2/B3, and does not support final
submission-level planning-performance claims.
