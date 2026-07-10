# Phase 72A Independent Temporal Label Package

Status: `phase72a_label_inputs_ready`

## Purpose

Phase 72A replaces the circular base-reward target with an independently
sourced future land-cover endpoint. It aligns annual ESRI Global LULC product
labels with AlphaEarth histories truncated at each prediction origin year for
Bishan and Dongxing. This phase prepares evidence for a later GeoFM information
test; it does not train GeoFM-STaR or establish a GeoFM advantage.

## Source and Intake Audit

- Product: ESRI Global LULC 10 m Time Series.
- Earth Engine collection:
  `projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS`.
- Extraction scale: `500 m`, matching the current AlphaEarth grids.
- Years: `2017-2024` for both regions.
- Crop class code: `5`.
- Bishan grid: `67 x 70`; AlphaEarth tensors: `67 x 70 x 64`.
- Dongxing grid: `91 x 99`; AlphaEarth tensors: `91 x 99 x 64`.
- Annual asset manifest: `32` rows, comprising 16 embeddings and 16 labels.
- Both regional audits passed all year, shape, source-role, and SHA256 checks.
- The Dongxing static DLTB proxy labels were not used.

The ESRI arrays are independent annual product labels. They are not described
as manual ground truth, field-observed policy outcomes, agronomic suitability,
or causal evidence.

## Temporal Sample Evidence

| Region | 1-year rows | Persistent | Converted | Persistence rate | 2-year rows | 2-year persistent | Continuous 2-year persistent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bishan | 8,535 | 6,444 | 2,091 | 0.75500879 | 7,847 | 5,031 | 4,569 |
| Dongxing | 23,092 | 16,889 | 6,203 | 0.73137883 | 20,739 | 13,796 | 11,891 |
| Total | 31,627 | 23,333 | 8,294 | 0.73775572 | 28,586 | 18,827 | 16,460 |

All 14 region-origin-year cohorts from 2017 through 2023 contain both one-year
outcome classes. The observed one-year persistence rates range from
`0.59989485` to `0.91633729`; therefore no region-year fold is blocked by a
single-class outcome at this stage.

The tensor package has shape `31,627 x 8 x 64`. Independent verification found
that every history mask length equals `origin_year - 2017 + 1`, all later
history slots are false and zero-filled, sample indexes are contiguous, and
NPZ one-year labels reproduce the CSV labels exactly. Unavailable two-year
outcomes use blank CSV values and tensor sentinel `-1`.

## Manual Review Frame

The deterministic review frame contains `560` rows: 20 rows for each of 28
region-year-transition strata. `review_label`, `review_source`, `review_date`,
and `review_confidence` are blank. The frame is an intake package for later
high-resolution or manual review; it does not contain fabricated validation
decisions.

## Generated Artifacts

The ignored real-run artifacts are under:

```text
experiments/phase72a_temporal_label_package/outputs/esri_labels
experiments/phase72a_temporal_label_package/outputs/bishan_dongxing_esri_2017_2024
```

The dual-region directory contains the stable manifest CSV, regional audit
CSV, temporal sample index CSV, compressed tensor NPZ, blank review frame,
class-support summary, JSON package, and Markdown audit report.

## Reproduction

Fetch Dongxing labels from the repository root:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase72a_temporal_label_package\fetch_phase72a_esri_lulc.py --region-config experiments\phase72a_temporal_label_package\phase72a_regions.json --regions dongxing --years 2017-2024 --output-dir experiments\phase72a_temporal_label_package\outputs\esri_labels
```

Build the dual-region package:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase72a_temporal_label_package\run_phase72a_temporal_label_package.py --region-config experiments\phase72a_temporal_label_package\phase72a_regions.json --embedding-dir bishan=data\bishan_alphaearth_sample --label-dir bishan=D:\test\paper58-geofm-world-model-rl\data\independent_change_labels\labels --embedding-dir dongxing=D:\test\dongxing_alphaearth --label-dir dongxing=experiments\phase72a_temporal_label_package\outputs\esri_labels --manual-review-per-stratum 20 --spatial-block-size 8 --output-dir experiments\phase72a_temporal_label_package\outputs\bishan_dongxing_esri_2017_2024
```

## Gate Decision

Phase 72A passes the transition gate because both regions passed, one- and
two-year cohorts are nonempty, both one-year outcome classes occur in every
region-year cohort, and no independence, year, shape, or hash blocker remains.
Phase 72B may now implement the leakage-free low-cost information-gain screen
against explicit GIS/current-LULC baselines, temporal and spatial shuffles, and
same-dimension random controls.

## Claim Boundary

Phase 72A validates and aligns independent annual product labels with
temporally truncated AlphaEarth histories. It does not train a prediction
model, alter rewards, run planning, prove GeoFM value, or revise the formal
manuscript.
