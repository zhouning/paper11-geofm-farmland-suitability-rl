# Phase 28 Representation-Control Diagnosis

## One-Sentence Argument

Phase 28 shows that the current B1 raw GeoFM representation is not yet a
stable positive learned-policy signal under the padded held-out Bishan
base-reward protocol, because both the 1024-step and 4096-step
representation-control diagnostics report `compression_matches_raw` rather
than `representation_signal_supported`.

## Terminology Ledger

| Canonical term | Definition | Boundary |
|---|---|---|
| Phase 28 | B0/B1/D2/D3/D4 representation-control diagnostic under the padded held-out policy protocol | diagnostic only, not final policy-performance evidence |
| B0 | explicit planning features with deterministic `base_planning_reward` | explicit-feature baseline |
| B1 | explicit planning features plus raw AlphaEarth/GeoFM embeddings with deterministic `base_planning_reward` | raw GeoFM representation arm |
| D2 | explicit planning features plus random 64-dimensional controls | dimensionality control, not semantic GeoFM evidence |
| D3 | explicit planning features plus shuffled AlphaEarth embeddings | spatial-alignment control |
| D4P8/D4P16 | explicit planning features plus PCA-compressed AlphaEarth embeddings | compression controls |
| `compression_matches_raw` | Phase 28 status when at least one compressed control matches or exceeds B1 while full control coverage is present | not support for raw B1 superiority |

## What Was Used

Phase 28 consumed:

```text
experiments/phase11_bishan_dltb_real/outputs/phase2_real/
experiments/phase8_ablation_controls/outputs/real_bishan_controls/
experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv
```

The two current diagnostic runs used the same evaluation scope:

- variants: `B0,B1,D2,D3,D4P8,D4P16`
- train tile: `tile_r003_c003`
- held-out tiles: `tile_r002_c003`, `tile_r005_c004`, `tile_r005_c003`
- seeds: `0, 1, 2`
- evaluation max steps: `8`
- tile-seed pairs per comparison: `9`

## Budget-Level Result

| Budget | Phase 28 status | B1-B0 mean delta | Positive tile-seed count |
|---|---|---:|---:|
| 1024 steps | `compression_matches_raw` | `-0.4329022862` | `4 / 9` |
| 4096 steps | `compression_matches_raw` | `-0.1318712688` | `3 / 9` |

The 4096-step run improves the mean B1-B0 delta by `0.3010310174`, matching
the Phase 27 budget diagnosis, but the learned-policy comparison remains
negative and the positive tile-seed count falls from `4 / 9` to `3 / 9`.

## Representation-Control Result

| Budget | Comparator | B1 minus comparator mean delta | Positive tile-seed count |
|---|---|---:|---:|
| 1024 steps | B0 | `-0.4329022862` | `4 / 9` |
| 1024 steps | D2 | `-0.5470593171` | `0 / 9` |
| 1024 steps | D3 | `-0.0883601687` | `4 / 9` |
| 1024 steps | D4P8 | `-0.0211358403` | `4 / 9` |
| 1024 steps | D4P16 | `-0.4680716340` | `2 / 9` |
| 4096 steps | B0 | `-0.1318712688` | `3 / 9` |
| 4096 steps | D2 | `0.0744591656` | `4 / 9` |
| 4096 steps | D3 | `-0.1094750135` | `3 / 9` |
| 4096 steps | D4P8 | `-0.3768518347` | `2 / 9` |
| 4096 steps | D4P16 | `-0.6411940236` | `2 / 9` |

At 1024 steps, B1 is below every primary comparator. At 4096 steps, B1 is
slightly above D2 on average but remains below B0, D3, D4P8, and D4P16. This
does not satisfy the Phase 28 gate for a supported representation signal,
which requires B1 to beat B0, D2, and D3 with adequate positive fractions.

## Mean Reward by Variant

| Budget | B0 | B1 | D2 | D3 | D4P8 | D4P16 |
|---|---:|---:|---:|---:|---:|---:|
| 1024 steps | `0.5159329661` | `0.0830306799` | `0.6300899970` | `0.1713908486` | `0.1041665202` | `0.5511023139` |
| 4096 steps | `0.4825072170` | `0.3506359482` | `0.2761767826` | `0.4601109618` | `0.7274877829` | `0.9918299718` |

The compressed controls are especially important. D4P8 and D4P16 exceed B1 at
4096 steps, so the current evidence is more consistent with compression or
optimization effects than with a robust advantage for the raw 64-dimensional
GeoFM representation.

## Interpretation

Phase 28 resolves the immediate ablation gap left after Phase 27. The current
negative B1 result is not explained only by the absence of representation
controls: after adding random, shuffled, and PCA-compressed controls, B1 still
does not become a stable positive learned-policy signal.

The result narrows the next scientific question. Paper11 should not continue
by claiming that raw GeoFM features improve planning decisions under the
current B1 design. The more defensible next step is to diagnose why compressed
controls can match or exceed raw B1, then revisit suitability-proxy validation,
reward design, and spatial case maps before any B2/B3 or transfer claim.

## What This Supports

Phase 28 supports three conservative claims:

1. The repository can run full B0/B1/D2/D3/D4 representation-control
   diagnostics under the same padded held-out Bishan policy protocol.
2. The current raw B1 representation is not distinguishable as a stable
   positive learned-policy signal at either 1024 or 4096 training steps.
3. Compression controls are competitive with or stronger than raw B1 in the
   current setup, so future work should test representation compression,
   normalization, reward alignment, and spatial diagnostics before extending
   claims.

## What This Does Not Support Yet

Phase 28 does not support:

- a positive GeoFM-over-explicit-feature planning-performance claim;
- a claim that raw 64-dimensional AlphaEarth embeddings are better than
  random, shuffled, or compressed controls in the current policy setup;
- suitability reward readiness;
- B2/B3 superiority;
- cross-region transfer;
- final manuscript submission readiness.

## Manuscript Use

Safe wording:

```text
Representation-control diagnostics over B0, B1, random, shuffled, and
PCA-compressed controls showed that the current raw GeoFM feature arm did not
produce a stable positive learned-policy signal under the padded held-out
Bishan base-reward protocol. The 4096-step run improved the B1-B0 mean delta
relative to 1024 steps, but both runs remained classified as
compression_matches_raw, and compressed controls exceeded B1 at 4096 steps.
```

Unsafe wording:

```text
GeoFM features improve farmland planning decisions.
```

## Claim Boundary

Phase 28 is a representation-control diagnostic over Bishan held-out tiles
under deterministic `base_planning_reward`. It does not enable suitability
reward, test B2/B3, evaluate cross-region transfer, provide spatial case maps,
or justify final submission-level planning-performance claims.
