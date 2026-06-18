# Phase 27 Stability Diagnosis

## One-Sentence Argument

Phase 27 shows that increasing the current Phase 26 training budget from 1024
to 4096 steps improves the mean B1-B0 learned-policy delta but does not explain
or resolve the negative evidence, because the higher-budget result remains
negative and tile-seed signs are unstable.

## Terminology Ledger

| Canonical term | Definition | Boundary |
|---|---|---|
| Phase 27 | read-only stability diagnosis over existing Phase 26 artifacts | no new training or reward family |
| lower budget | current 1024-step macOS Phase 26 result set | not an independent replicate |
| higher budget | current 4096-step macOS Phase 26 result set | not convergence proof |
| stability class | sign transition of a paired tile-seed B1-B0 delta | diagnostic only |
| budget_not_explanatory | Phase 27 status when higher budget remains unsupported | not a paper-performance claim |

## What Was Used

Phase 27 consumed the existing Phase 26 comparison JSON files:

```text
experiments/phase26_main_experiment/outputs/macos_main/phase26_analysis/phase26_main_comparison.json
experiments/phase26_main_experiment/outputs/macos_main_4096/phase26_analysis/phase26_main_comparison.json
```

It wrote diagnostic artifacts under:

```text
experiments/phase27_stability_diagnosis/outputs/macos_1024_vs_4096/
```

## Budget-Level Result

| Budget | B1-B0 learned-policy mean delta | Positive tile-seed count | Claim status |
|---|---:|---:|---|
| 1024 steps | `-0.4329022862` | `4 / 9` | `not_supported` |
| 4096 steps | `-0.1318712688` | `3 / 9` | `not_supported` |

The mean delta improves by `0.3010310174`, but the positive tile-seed count
falls by `1`. Phase 27 therefore assigns:

```text
phase27_diagnostic_status: budget_not_explanatory
```

## Tile-Seed Stability

| Stability class | Count |
|---|---:|
| stable_positive | `1` |
| stable_negative | `3` |
| flip_to_positive | `2` |
| flip_to_negative | `3` |
| incomplete | `0` |

The result is not a simple longer-budget improvement story. Three tile-seed
pairs flip from positive at 1024 steps to non-positive at 4096 steps, while
only two flip to positive. The strongest tile-level divergence is also
important: `tile_r002_c003` worsens by a mean `-0.9207331214`, while
`tile_r005_c003` improves by `0.9854193915`.

## Interpretation

Budget sensitivity exists, but budget alone does not explain the current B1
failure. The higher-budget run remains negative on average and less stable by
positive-count criterion. The current learned-policy evidence should therefore
remain a negative boundary result.

The next work should not extend the B1 superiority claim. It should prioritize:

1. representation controls against random, shuffled, and PCA-compressed GeoFM
   features;
2. repeated or intermediate budget stability checks if more training is run;
3. renewed suitability-proxy validation before any reward integration.

## Manuscript Use

Safe wording:

```text
A follow-on stability diagnosis compared the current 1024-step and 4096-step
B0/B1 held-out results. Although the higher budget improved the mean B1-B0
delta, the result remained negative and tile-seed signs were unstable, so the
current evidence does not support a positive B1-over-B0 learned-policy claim.
```

Unsafe wording:

```text
Longer training shows that GeoFM improves farmland planning.
```

## Claim Boundary

Phase 27 is a read-only diagnosis of existing Phase 26 B0/B1 padded held-out
Bishan learned-policy artifacts. It does not run new training, enable
suitability reward, test B2/B3, demonstrate cross-region transfer, or support
final submission-level claims.
