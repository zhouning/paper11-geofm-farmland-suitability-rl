# Phase 26 Result Interpretation

## One-Sentence Argument

In the Paper11 repository, Phase 26 shows that the padded held-out Bishan
B0/B1 learned-policy outputs can be turned into manuscript-facing empirical
tables, but the current 4096-step result does not support a stable claim that
GeoFM-enhanced B1 outperforms explicit-feature B0 across held-out tiles and
seeds.

## Terminology Ledger

| Canonical term | Definition | Boundary |
|---|---|---|
| Paper11 | GeoFM-enhanced current-state farmland suitability representation and DRL layout optimization. | Does not include future-aware prediction-optimization. |
| Phase 25 | Padded variable-size held-out Bishan tile B0/B1 learned-policy pilot. | Pilot evidence only, not submission-level performance evidence. |
| Phase 26 | Main empirical analysis package built from Phase 25 outputs. | Analysis package only; no new training or reward family. |
| B0 | explicit planning features with deterministic `base_planning_reward`. | Baseline representation. |
| B1 | explicit planning features plus raw AlphaEarth/GeoFM embeddings with deterministic `base_planning_reward`. | GeoFM-enhanced representation under the same reward. |
| `base_planning_reward` | deterministic planning reward used for the bounded B0/B1 experiments. | Does not enable suitability reward. |
| held-out Bishan tile | an evaluation tile distinct from the train tile. | Not a cross-region transfer split. |
| claim status | conservative support label for the current evidence. | Not a manuscript-ready performance claim. |

## What Was Used

The current Phase 26 interpretation is based on the macOS 4096-step result set:

```text
experiments/phase26_main_experiment/outputs/macos_main_4096/phase25_run/
experiments/phase26_main_experiment/outputs/macos_main_4096/phase26_analysis/
```

The analysis package reports:

- training timesteps: `4096`
- evaluation max steps: `8`
- train tile: `tile_r003_c003`
- held-out tiles: `tile_r002_c003`, `tile_r005_c004`, `tile_r005_c003`
- seeds: `0, 1, 2`
- learned-policy B1-B0 mean reward delta: `-0.1318712688`
- positive tile-seed count: `3 / 9`
- claim status: `not_supported`

## Result Interpretation

Phase 26 is the first Paper11 package that summarizes multi-tile, multi-seed
held-out B0/B1 learned-policy behavior in a manuscript-facing form. That is a
meaningful progress step, because it turns Phase 25 smoke evidence into a
structured comparison table and a conservative claim-readiness note.

The observed 4096-step learned-policy delta is negative on average. More
importantly, only three of the nine tile-seed pairs favor B1 over B0. That is
not stable enough to support a claim that the GeoFM-enhanced representation is
currently better than the explicit-feature baseline under the same reward and
evaluation protocol.

The baselines are not equivalent evidence. `first_valid` is exactly tied for
B0 and B1, while `seeded_random` shows a positive B1-B0 mean delta. Those
baselines are useful diagnostics, but they do not rescue the learned-policy
result because the paper's central question is about the learned policy.

## What This Supports

Phase 26 supports three repository and manuscript-development claims:

1. The Phase 25 held-out B0/B1 pilot can be summarized into manuscript-facing
   empirical tables.
2. The current multi-tile, multi-seed learned-policy evidence is not stable
   enough to support a positive B1-over-B0 claim.
3. The evidence boundary remains explicit: suitability reward, B2/B3, and
   cross-region transfer are still out of scope.

## What This Does Not Support Yet

Phase 26 does not support the main Paper11 performance claims:

- It does not show stable GeoFM superiority over the explicit baseline.
- It does not show that the current B1 representation generalizes better than
  B0 across held-out tiles.
- It does not validate suitability reward readiness.
- It does not compare B2 or B3.
- It does not test cross-region transfer.
- It does not provide spatial case maps, uncertainty analysis, or a full
  ablation package.

## Manuscript Use

In the manuscript workflow, this result should be used as a results boundary
checkpoint. It can support a statement such as:

```text
We converted the padded held-out Bishan B0/B1 pilot into a multi-tile,
multi-seed empirical analysis package and found that the current learned-policy
evidence is not yet stable enough to support a positive B1-over-B0 claim under
the deterministic base planning reward.
```

It should not be used as:

```text
GeoFM improves farmland layout optimization.
```

That stronger claim still requires stable learned-policy evidence, suitability
validation, ablations, transfer testing, and spatial diagnostics.
