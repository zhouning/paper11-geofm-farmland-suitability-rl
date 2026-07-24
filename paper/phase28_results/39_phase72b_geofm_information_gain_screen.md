# Phase 72B GeoFM Information-Gain Screen

Official integrity-verified status: `geofm_information_not_supported`

## Conclusion

Phase 72B detected a pooled predictive gain from temporal GeoFM features beyond
the explicit-history baseline, but the gain was not representation-specific,
spatially stable, or transferable. The clean receipt-bound confirmation failed
the frozen temporal-order and spatial-shuffle control gates, both zero-shot
transfer directions, and the regional spatial-consistency gate. The Phase 72C
GeoFM-STaR route must therefore stop.

## Purpose

Phase 72B tested whether temporal AlphaEarth features improve one-year
farmland-conversion prediction beyond explicit public-GIS and land-cover
history features. The locked screen used independent annual product labels,
strict representation controls, temporal confirmation, buffered spatial folds,
and bidirectional zero-shot transfer. It did not train GeoFM-STaR, alter a
planning reward, or run a planner.

## Integrity-Bound Execution

The official run regenerated the terrain provenance manifest and prepared
package under the final contract, then refit all models from an empty frozen
directory. Terrain values did not change during regeneration: the old and new
NPZ hashes were identical for both regions.

- Terrain source: Copernicus DEM GLO-30, Earth Engine collection
  `COPERNICUS/DEM/GLO30`, aggregated at `500 m`.
- Bishan terrain: `67 x 70`, SHA256
  `dc85a6ae9939f251144f1e0372fea957dfc65f383b343015b10c46ab5369a90e`.
- Dongxing terrain: `91 x 99`, SHA256
  `3c1a68a56bcc4e4133daeb79895d2f56a4c59a62737a3ada776a3b9ab51ba2ce`.
- Development rows, origins 2017-2022: `28,586`.
- Confirmation rows, origin 2023: `3,041`, including `630` conversions.
- Frozen protocol SHA256:
  `d7275d5264649d0215e784e800961aa205cf4986cf788123d5de7307016866bb`.
- Prepared-artifact manifest SHA256:
  `4843dfda860e0f87c276e62efad05b0604e9e3d95ff812d8f0000ce0619c9357`.
- Selected-model SHA256:
  `79c00435de9c537ab25cf36c19e91cafd4654ed9077fd7680e366ef524be70e0`.
- Fit-control manifest SHA256:
  `53ab9f106eff53c0ae04aa5bc21e13790d9c7c24b3e84af56666a67ed3feb449`.
- Confirmation-control manifest SHA256:
  `d8666d0e6290eaa203894e1f6e5f46ef980eba58f11468a6ca4f0f00f2139e71`.
- Confirmation receipt SHA256:
  `2de7750a82562178a25731c1250c7bbdb45502b29e903dbab5c905791ffe5988`.

The fit completed `153 / 153` bundles across 13 axes and wrote 4,806
validation metric rows. Its 150 control-manifest rows used only declared train
or validation partitions and all had `cross_partition_count=0`. Confirmation
wrote 75 control-manifest rows across the same 13 axes; every row used a
declared test partition and had `cross_partition_count=0`.

The first post-fit confirmation attempt stopped before opening confirmation
targets because BLAS thread count changed the byte hash of mathematically
equivalent random-projection products. It returned
`phase72b_inputs_not_ready` with zero confirmation, prediction, and metric rows.
Commit `e46c1fb` made random-projection multiplication thread-count invariant.
The repaired implementation reproduced every recorded fit-control matrix hash,
and all 142 Phase 72B tests passed before the target-opening confirmation. The
pre-target blocker receipt remains archived locally and was not treated as a
scientific result.

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
all `2,000 / 2,000` replicates across 214 clusters. Its mean AP delta was
`+0.032544250173` with a 95% interval of
`[0.002717295271, 0.062074205092]`; its mean favorable Brier delta was
`+0.023332824458` with a 95% interval of
`[0.014780303261, 0.032000531884]`.

The pooled practical and statistical checks passed. These checks were not
sufficient for a positive Phase 72B result because the frozen protocol also
required the primary model to clear every representation control, both
zero-shot transfer directions, and the buffered spatial gate.

## Representation Controls

Deltas below are primary temporal GeoFM minus the selected strongest control
for AP, and control minus primary for Brier and ECE. Positive values favor the
primary model.

| Control | Selected seed | AP delta | AP range across seeds | Brier delta | Brier range across seeds | ECE delta | ECE range across seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Temporal-order shuffle | 74 | -0.005872817466 | [-0.005872817466, 0.027670012406] | 0.001402165769 | [-0.000420302017, 0.003465212065] | 0.009099422172 | [-0.004855919342, 0.017256921482] |
| Spatial shuffle | 72 | -0.003434131465 | [-0.003434131465, 0.030937810238] | 0.011750438090 | [0.001901893161, 0.019188902304] | 0.066173321790 | [-0.002802631987, 0.076672185281] |
| Same-dimension random projection | 74 | 0.006494142840 | [-0.003157034090, 0.046279797734] | 0.034024283598 | [0.022082329628, 0.046840640247] | 0.144007406457 | [0.101202342165, 0.153922235581] |

The frozen control gate required AP delta at least `0.005` and Brier delta at
least `0.002` for every control. Temporal-order shuffle failed both margins.
Spatial shuffle passed Brier but failed AP. Random projection passed both. The
pooled improvement therefore could not be attributed to ordered temporal GeoFM
information under the predeclared controls.

## Transfer and Buffered Spatial Evidence

| Transfer axis | Rows | AP delta | Brier delta | ECE delta |
| --- | ---: | ---: | ---: | ---: |
| Bishan to Dongxing | 2,353 | -0.016801606373 | 0.026115464037 | 0.096944527306 |
| Dongxing to Bishan | 688 | 0.000851525829 | -0.001755600472 | -0.019835919679 |

Neither transfer direction passed. Bishan-to-Dongxing improved Brier but
exceeded the allowed AP harm. Dongxing-to-Bishan reached neither required gain
threshold.

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

All 10 spatial axes were evaluable, but the direction was heterogeneous.
Bishan degraded on both region-level AP and Brier. Dongxing improved Brier at
the region level but degraded AP, and several individual folds harmed both
metrics.

## Comparison with the Pre-Repair Archive

The official and pre-repair confirmations both contained 153 metric rows. All
78 non-control rows were identical, including pooled explicit and primary
metrics, both transfer directions, all spatial explicit-primary deltas, and all
13 primary-versus-explicit bootstrap rows. The transfer CSV was byte-identical.

Only the 75 control metric rows and three control bootstrap comparisons
changed. The clean refit used the complete candidate-grid and calibration
contract and selected temporal-order seed 74 instead of 76 and spatial-shuffle
seed 72 instead of 74. Random-projection seed 74 was unchanged. These changes
strengthened the negative control result and did not alter the pooled,
transfer, or spatial primary evidence.

## Official Gate Decision

The final status is `geofm_information_not_supported`, with zero input
blockers. No post hoc threshold, metric, region, seed, or fold change is
permitted.

```text
Do not begin Phase 72C.
Stop the GeoFM-STaR route at the Phase 72B gate.
Proceed only with the approved Phase 72 exhaustion analysis.
Do not alter the planning reward or formal manuscript on this evidence.
```

## Generated Artifacts

Ignored real-run artifacts are under:

```text
experiments/phase72b_geofm_information_gain_screen/outputs/terrain
experiments/phase72b_geofm_information_gain_screen/outputs/prepared
experiments/phase72b_geofm_information_gain_screen/outputs/frozen
experiments/phase72b_geofm_information_gain_screen/outputs/confirmation
```

The confirmation receipt binds nine stable artifacts: metrics, predictions,
calibration, bootstrap deltas, control comparison, confirmation-control
manifest, transfer summary, JSON, and Markdown. All nine byte hashes and the
receipt sidecar were independently verified.

## Reproduction

Run from the repository root:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase72b_geofm_information_gain_screen\fetch_phase72b_terrain.py --phase72a-region-config experiments\phase72a_temporal_label_package\phase72a_regions.json --phase72b-protocol experiments\phase72b_geofm_information_gain_screen\phase72b_protocol.json --output-dir experiments\phase72b_geofm_information_gain_screen\outputs\terrain

D:\adk\.venv\Scripts\python.exe experiments\phase72b_geofm_information_gain_screen\run_phase72b_information_gain_screen.py --mode prepare --protocol experiments\phase72b_geofm_information_gain_screen\phase72b_protocol.json --phase72a-region-config experiments\phase72a_temporal_label_package\phase72a_regions.json --phase72a-package-dir experiments\phase72a_temporal_label_package\outputs\bishan_dongxing_esri_2017_2024 --embedding-dir bishan=data\bishan_alphaearth_sample --label-dir bishan=D:\test\paper58-geofm-world-model-rl\data\independent_change_labels\labels --embedding-dir dongxing=D:\test\dongxing_alphaearth --label-dir dongxing=experiments\phase72a_temporal_label_package\outputs\esri_labels --terrain-dir experiments\phase72b_geofm_information_gain_screen\outputs\terrain --output-dir experiments\phase72b_geofm_information_gain_screen\outputs\prepared

D:\adk\.venv\Scripts\python.exe experiments\phase72b_geofm_information_gain_screen\run_phase72b_information_gain_screen.py --mode fit-freeze --prepared-dir experiments\phase72b_geofm_information_gain_screen\outputs\prepared --output-dir experiments\phase72b_geofm_information_gain_screen\outputs\frozen

D:\adk\.venv\Scripts\python.exe experiments\phase72b_geofm_information_gain_screen\run_phase72b_information_gain_screen.py --mode confirm --prepared-dir experiments\phase72b_geofm_information_gain_screen\outputs\prepared --frozen-dir experiments\phase72b_geofm_information_gain_screen\outputs\frozen --output-dir experiments\phase72b_geofm_information_gain_screen\outputs\confirmation
```

Each output directory must be absent before its stage writes. Do not reuse
legacy progress, bundles, prepared artifacts, or confirmation receipts.

## Claim Boundary

Phase 72B is a leakage-free, low-cost information-gain screen using independent
annual product labels. It shows a pooled predictive improvement but does not
establish representation-specific, spatially stable, or transferable GeoFM
information. It does not implement GeoFM-STaR, validate agronomic suitability,
alter planning rewards, run planning, or revise the formal manuscript.
