# Phase 1 Result Interpretation

## One-Sentence Argument

In the Paper11 repository, Phase 1 shows that the included Bishan AlphaEarth sample can be converted into deterministic region-level GeoFM representations and a bounded latent suitability proxy, supported by an executable baseline and JSON/CSV artifacts, with the boundary that no DRL planning performance or agronomic measurement claim is made.

## Terminology Ledger

| Canonical term | Definition | Boundary |
|---|---|---|
| Paper11 | GeoFM-enhanced current-state farmland suitability representation and DRL layout optimization. | Does not include future-aware prediction-optimization. |
| AlphaEarth | Annual remote-sensing foundation-model embedding source recorded as `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`. | Treated as latent remote-sensing representation, not direct soil or irrigation measurement. |
| GeoFM | Geospatial foundation model representation used as a latent environmental channel. | Enriches explicit planning constraints; does not replace GIS, DEM, slope, or contiguity. |
| Phase 1 Bishan baseline | Deterministic representation smoke experiment using the included Bishan sample. | Not a full DRL experiment. |
| region-level GeoFM feature | Mean 64-dimensional embedding aggregated over a deterministic grid region. | Region is planning-like, not an official farmland block polygon. |
| suitability_proxy | Bounded score derived from embedding similarity, temporal stability, and dispersion. | Not a soil quality, fertility, or irrigation score. |

## What Was Run

The Phase 1 baseline was executed with the default command:

```powershell
python experiments\phase1_bishan_baseline\run_phase1.py
```

The executable path is:

```text
experiments/phase1_bishan_baseline/run_phase1.py
```

The input sample is:

```text
data/bishan_alphaearth_sample/
```

The script used the 2020 embedding as the base year, read annual embeddings from 2017 through 2024, partitioned the 67 by 70 grid into 5 by 5 deterministic grid regions, aggregated 64-dimensional region-level embedding means, and wrote the generated artifacts:

```text
experiments/phase1_bishan_baseline/outputs/region_features.csv
experiments/phase1_bishan_baseline/outputs/summary.json
```

## Observed Phase 1 Summary

The current `summary.json` reports:

| Field | Value |
|---|---|
| Metadata source | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` |
| Base year | 2020 |
| Years | 2017-2024 |
| Grid shape | 67 by 70 |
| Embedding dimension | 64 |
| Row bins | 5 |
| Column bins | 5 |
| Number of regions | 25 |
| `suitability_proxy` minimum | 0.0 |
| `suitability_proxy` maximum | 1.0 |
| `suitability_proxy` mean | 0.6863009610963511 |

The summary also records the claim boundary:

```text
The suitability_proxy is derived from latent remote-sensing embeddings and does not directly measure soil quality, fertility, or irrigation access.
```

## Result Interpretation

Phase 1 validates the representation pipeline rather than the planning model. The result shows that the repository can read the included Bishan AlphaEarth sample, verify the expected years and embedding dimension, aggregate the embedding grid into deterministic regions, and export a stable table of region-level GeoFM features.

The `suitability_proxy` range is intentionally bounded to `[0, 1]`. This makes it usable as a controlled diagnostic feature or future reward component, but the value should be interpreted only as a latent remote-sensing proxy. The observed mean of approximately `0.6863` describes the current default scoring scale over 25 grid regions; it is not an accuracy score, an agronomic suitability validation result, or a planning performance metric.

The 25 regions are deterministic grid regions. They are useful for repository-level smoke testing and for illustrating how GeoFM features can be aggregated into planning-like units. They should not be described as cadastral parcels, official farmland blocks, or administrative planning units until real parcel or block polygons are connected.

## What This Supports

Phase 1 supports three repository and manuscript-development claims:

1. The standalone Paper11 repository contains an executable GeoFM representation baseline.
2. The included Bishan sample can produce region-level 64-dimensional GeoFM features and a bounded proxy score without internet, GPU, Earth Engine, or full DRL training.
3. The claim boundary is explicit in both the code output and the result interpretation.

## What This Does Not Support Yet

Phase 1 does not support the main Paper11 performance claims:

- It does not show that GeoFM-enhanced states improve DRL action selection.
- It does not show that the suitability proxy improves final farmland layout realism.
- It does not compare B0, B1, B2, or B3 conditions.
- It does not test transfer across regions.
- It does not validate the proxy against stable farmland, slope, high-standard farmland, soil, irrigation, or productivity labels.

These claims require Phase 2 and Phase 3 experiments using real planning blocks, explicit GIS features, and DRL evaluation.

## Manuscript Use

In the manuscript workflow, this result should be used as a methods and reproducibility checkpoint. It can support a statement such as:

```text
We first implemented a repository-level representation baseline that converts annual AlphaEarth embeddings into deterministic region-level features and a bounded latent suitability proxy for the Bishan sample.
```

It should not be used as:

```text
GeoFM improves farmland layout optimization.
```

That stronger claim requires baseline comparison and transfer evidence.
