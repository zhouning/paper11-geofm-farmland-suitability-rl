# Phase 36 Suitability-Proxy Validation

## One-Sentence Argument

Phase 36 tests whether the current GeoFM-derived feature families add
weak-label suitability signal beyond explicit planning features before Paper11
enables suitability reward or runs B2/B3 planning-performance experiments.

## Current Experiment Snapshot

The real Bishan Phase 36 diagnostic used the existing Phase 11/Phase 2 real
feature tables, Phase 8 random/shuffled/PCA controls, and Phase 30 normalized
controls:

- input rows: `64,984` Bishan DLTB blocks
- split: existing `split` column
- train rows: `45,460`
- evaluation rows: `19,524`
- feature families evaluated: `11`
- label columns evaluated:
  - `current_farmland_label`
  - `farmland_or_orchard_label`
  - `low_slope_farmland_label`

The local ignored artifacts are under:

```text
experiments/phase36_suitability_proxy_validation/outputs/real_bishan
```

Generated artifacts:

```text
phase36_label_summary.csv
phase36_model_summary.csv
phase36_suitability_proxy_validation.json
phase36_suitability_proxy_validation.md
```

## Main Result

The current Phase 36 status is:

```text
proxy_signal_not_supported
```

All three available weak labels are usable in the train/evaluation split, but
all three are DLTB/slope-derived and are flagged as
`explicit_label_leakage_risk`.

| Label | Valid labels | Positives | Train / eval | Leakage risk |
|---|---:|---:|---:|---|
| `current_farmland_label` | `64,984` | `25,359` | `45,460 / 19,524` | `explicit_label_leakage_risk` |
| `farmland_or_orchard_label` | `64,984` | `28,279` | `45,460 / 19,524` | `explicit_label_leakage_risk` |
| `low_slope_farmland_label` | `64,984` | `7,443` | `45,460 / 19,524` | `explicit_label_leakage_risk` |

Because the labels are directly or indirectly encoded by explicit planning
features, `explicit_only` reaches ROC AUC, average precision, and balanced
accuracy of `1.0` for all three labels. Adding raw, normalized, random,
shuffled, or PCA GeoFM controls also reaches `1.0` whenever explicit features
are included, which means those combined-feature scores are not evidence of
GeoFM-specific suitability signal.

The GeoFM-only and scalar-proxy-only checks are more informative:

| Label | `raw_geofm_only` ROC AUC | `raw_geofm_only` AP | `suitability_proxy_only` ROC AUC | `suitability_proxy_only` AP |
|---|---:|---:|---:|---:|
| `current_farmland_label` | `0.5482365921` | `0.4205359849` | `0.5081982029` | `0.3934952625` |
| `farmland_or_orchard_label` | `0.5293767230` | `0.4559993536` | `0.5124973908` | `0.4430566614` |
| `low_slope_farmland_label` | `0.6490064144` | `0.1695914498` | `0.4979564572` | `0.1130506500` |

## Interpretation

Phase 36 does not support enabling the current B2/B3 suitability reward. The
available weak labels are usable computationally, but they are not independent
validation labels: explicit planning features already encode the label logic.
The current scalar `suitability_proxy` is also not aligned strongly enough with
the available weak labels to justify reward use.

The one narrow positive signal is that raw GeoFM-only features show weak
held-out association with `low_slope_farmland_label` (`ROC AUC =
0.6490064144`). This is not enough to override the leakage boundary, but it is
useful for the next step: rebuild the suitability proxy from supervised or
semi-supervised GeoFM features rather than using the current centroid-style
scalar as a reward term.

## Claim Boundary

Phase 36 is a read-only weak-label validation diagnostic. It does not run
policy training, alter rewards, enable suitability reward, test B2/B3, prove
GeoFM agronomic validity, or support planning-performance claims.

## Next Step

The next experiment should not be another PPO budget increase. The strongest
next step is a Phase 37 decision-alignment or proxy-rebuild branch:

1. obtain or construct a more independent suitability label, such as
   high-standard farmland, retention/productivity, irrigation/water-proximity
   proxy, or external soil/yield proxy;
2. train a supervised suitability proxy under spatial held-out validation;
3. check whether selected Phase 33/34/35 block sets are ordered by the rebuilt
   proxy;
4. only then run a bounded B2/B3 reward smoke.
