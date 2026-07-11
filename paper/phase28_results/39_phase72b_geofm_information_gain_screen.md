# Phase 72B GeoFM Information-Gain Screen

Archived pre-integrity-repair confirmation status:
`geofm_information_not_supported`

Official integrity-verified status: pending completion of the clean refit and
receipt-bound confirmation started on 2026-07-11. The numerical evidence below
describes the archived pre-repair confirmation and must be rechecked against
the new official receipt before this note is finalized.

## Purpose

Phase 72B tested whether temporal AlphaEarth features improve one-year
farmland-conversion prediction beyond explicit public-GIS and land-cover
history features. The locked screen used independent annual product labels,
strict representation controls, temporal confirmation, buffered spatial
folds, and bidirectional zero-shot transfer. It did not train GeoFM-STaR,
alter a planning reward, or run a planner.

## Inputs and Frozen State

- Terrain source: Copernicus DEM GLO-30, Earth Engine collection
  `COPERNICUS/DEM/GLO30`, aggregated at `500 m`.
- Bishan terrain: `67 x 70`, SHA256
  `dc85a6ae9939f251144f1e0372fea957dfc65f383b343015b10c46ab5369a90e`.
- Dongxing terrain: `91 x 99`, SHA256
  `3c1a68a56bcc4e4133daeb79895d2f56a4c59a62737a3ada776a3b9ab51ba2ce`.
- Development rows, origins 2017-2022: `28,586`.
- Confirmation rows, origin 2023: `3,041`, including `630` conversions.
- Frozen protocol SHA256:
  `b51a8b45050579a7741d43d2244571815ef752304483184de30cb18a9cc1f864`.
- Frozen selected-model SHA256:
  `0476dc525d302f1c08d6b1469b158fc186b054a184255c70e8c9a1b2eab5ade0`.
- Confirmation coverage: `155,091` prediction rows, `153` metric rows,
  `1,530` calibration rows, `16` bootstrap rows, 10 valid spatial axes,
  and zero invalid spatial axes.

The hashes reported by confirmation exactly match the values written before
the held-out outcomes were evaluated.

## Frozen Pooled Models

| Variant | Family | Hyperparameters | Calibration | Candidate |
| --- | --- | --- | --- | --- |
| Explicit history | Logistic regression | `C=0.01`, balanced class weight | Isotonic | `efdc4c12c72fbba4` |
| Explicit history + full temporal GeoFM | Histogram gradient boosting | learning rate `0.03`, 31 leaves, minimum leaf size `20`, 200 iterations, L2 `1.0` | Isotonic | `eb74812a4f18dedf` |

## Pooled Confirmation Evidence

| Variant | AP | Brier | ECE | ROC AUC | Balanced accuracy | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Explicit history | 0.446926399333 | 0.159201942852 | 0.135765732022 | 0.768664454583 | 0.709785506903 | 0.500545256270 |
| Explicit history + full temporal GeoFM | 0.479935630218 | 0.135867713995 | 0.043965456329 | 0.785087528721 | 0.706482523882 | 0.512654120701 |

Favorable primary deltas were AP `+0.033009230885`, Brier
`+0.023334228857`, and ECE `+0.091800275693`. The paired block bootstrap used
all `2,000 / 2,000` valid replicates across 214 clusters. Its mean AP delta was
`+0.032544250173` with 95% interval `[0.002717295271, 0.062074205092]`; its
mean favorable Brier delta was `+0.023332824458` with 95% interval
`[0.014780303261, 0.032000531884]`.

The pooled practical and statistical comparisons therefore passed. They are
not sufficient for a positive Phase 72B result because the frozen protocol
also requires the primary model to clear every strict representation control.

## Representation Controls

Each control family used five frozen seeds. Deltas below are primary temporal
GeoFM minus the selected strongest control for AP, and control minus primary
for Brier and ECE, so positive values favor the primary model.

| Control | Selected seed | AP delta | AP range across seeds | Brier delta | Brier range across seeds | ECE delta | ECE range across seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Temporal-order shuffle | 76 | 0.000529157576 | [-0.002149496118, 0.010686837909] | 0.000987420911 | [0.000966900674, 0.001955914907] | 0.000115241698 | [0.000115241698, 0.016859553374] |
| Spatial shuffle | 74 | 0.008391128471 | [0.007150682124, 0.038574628539] | 0.012669512479 | [0.003610920032, 0.014320575958] | 0.068184174684 | [-0.006732825573, 0.068184174684] |
| Same-dimension random projection | 74 | 0.014150381739 | [-0.004586192049, 0.046399915991] | 0.039950202911 | [0.018246672641, 0.046626428778] | 0.157703888077 | [0.081770584327, 0.157703888077] |

The frozen control gate requires AP delta at least `0.005` and Brier delta at
least `0.002` for every control. The temporal-order-shuffle comparison missed
both thresholds. This is the direct reason for
`geofm_information_not_supported`: the measured gain cannot be attributed to
the ordered temporal GeoFM representation under the predeclared controls.

## Transfer and Buffered Spatial Evidence

| Transfer axis | Rows | AP delta | Brier delta | ECE delta |
| --- | ---: | ---: | ---: | ---: |
| Bishan to Dongxing | 2,353 | -0.016801606373 | 0.026115464037 | 0.096944527306 |
| Dongxing to Bishan | 688 | 0.000851525829 | -0.001755600472 | -0.019835919679 |

Neither direction met the complete transfer gate. Bishan-to-Dongxing improved
Brier and ECE but exceeded the allowed AP harm. Dongxing-to-Bishan did not
reach either required gain threshold.

| Spatial axis | Rows | AP delta | Brier delta | ECE delta |
| --- | ---: | ---: | ---: | ---: |
| Bishan fold 0 | 144 | -0.033467536200 | 0.017125427321 | 0.017839127381 |
| Bishan fold 1 | 236 | -0.189975963504 | -0.402026254272 | -0.488578403546 |
| Bishan fold 2 | 97 | -0.064175704570 | -0.000375813662 | 0.049833213042 |
| Bishan fold 3 | 86 | -0.092345035014 | -0.009961377789 | -0.009703598966 |
| Bishan fold 4 | 125 | -0.000497973451 | 0.052945628322 | 0.037384936266 |
| Dongxing fold 0 | 429 | 0.005868697499 | 0.005782788378 | -0.001085980478 |
| Dongxing fold 1 | 499 | -0.024969169524 | -0.002611492960 | -0.010058954896 |
| Dongxing fold 2 | 470 | -0.048419961144 | 0.013588661930 | 0.080752454345 |
| Dongxing fold 3 | 528 | -0.045260610746 | 0.005914935260 | 0.078841979088 |
| Dongxing fold 4 | 427 | -0.090247705682 | -0.004299528419 | 0.006999889613 |

All spatial folds were evaluable, but the direction was heterogeneous. Only
Dongxing fold 0 improved both AP and Brier; several folds harmed both metrics,
including a large Bishan fold 1 degradation.

## Archived Gate Decision

The archived Phase 72B confirmation status was
`geofm_information_not_supported`, with no measured input blockers. Pooled
evidence showed predictive signal, but the
predeclared strict-control gate failed and transfer/spatial evidence did not
support a consistent multi-region advantage. No post hoc threshold, metric,
region, seed, or fold change is permitted.

Phase 72C must not begin while the integrity-verified confirmation is pending.
If the clean confirmation reproduces the archived negative gate, the GeoFM-STaR
route stops and the next allowed work is the approved Phase 72 exhaustion
analysis. That analysis must retain the broader design's independent-product,
endpoint, model, transfer, spatial, noise-sensitivity, and planning criteria.

## Generated Artifacts

Ignored real-run artifacts are under:

```text
experiments/phase72b_geofm_information_gain_screen/outputs/terrain
experiments/phase72b_geofm_information_gain_screen/outputs/prepared
experiments/phase72b_geofm_information_gain_screen/outputs/frozen
experiments/phase72b_geofm_information_gain_screen/outputs/confirmation
```

The confirmation directory contains the metric, prediction, calibration,
bootstrap, control, transfer, JSON, and Markdown outputs required by the
frozen protocol.

## Reproduction

Fetch terrain:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase72b_geofm_information_gain_screen\fetch_phase72b_terrain.py --phase72a-region-config experiments\phase72a_temporal_label_package\phase72a_regions.json --phase72b-protocol experiments\phase72b_geofm_information_gain_screen\phase72b_protocol.json --output-dir experiments\phase72b_geofm_information_gain_screen\outputs\terrain
```

Prepare the label-separated feature package:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase72b_geofm_information_gain_screen\run_phase72b_information_gain_screen.py --mode prepare --protocol experiments\phase72b_geofm_information_gain_screen\phase72b_protocol.json --phase72a-region-config experiments\phase72a_temporal_label_package\phase72a_regions.json --phase72a-package-dir experiments\phase72a_temporal_label_package\outputs\bishan_dongxing_esri_2017_2024 --embedding-dir bishan=data\bishan_alphaearth_sample --label-dir bishan=D:\test\paper58-geofm-world-model-rl\data\independent_change_labels\labels --embedding-dir dongxing=D:\test\dongxing_alphaearth --label-dir dongxing=experiments\phase72a_temporal_label_package\outputs\esri_labels --terrain-dir experiments\phase72b_geofm_information_gain_screen\outputs\terrain --output-dir experiments\phase72b_geofm_information_gain_screen\outputs\prepared
```

Freeze development-selected models, then confirm once:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase72b_geofm_information_gain_screen\run_phase72b_information_gain_screen.py --mode fit-freeze --prepared-dir experiments\phase72b_geofm_information_gain_screen\outputs\prepared --output-dir experiments\phase72b_geofm_information_gain_screen\outputs\frozen
D:\adk\.venv\Scripts\python.exe experiments\phase72b_geofm_information_gain_screen\run_phase72b_information_gain_screen.py --mode confirm --prepared-dir experiments\phase72b_geofm_information_gain_screen\outputs\prepared --frozen-dir experiments\phase72b_geofm_information_gain_screen\outputs\frozen --output-dir experiments\phase72b_geofm_information_gain_screen\outputs\confirmation
```

## Claim Boundary

Phase 72B is a leakage-free, low-cost information-gain screen using
independent annual product labels. It shows a pooled predictive improvement but
does not establish representation-specific, spatially stable, or transferable
GeoFM information. It does not implement GeoFM-STaR, alter planning rewards,
run planning, or revise the formal manuscript.
