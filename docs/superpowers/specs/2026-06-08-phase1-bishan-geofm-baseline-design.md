# Phase 1 Bishan GeoFM Baseline Design

Saved: 2026-06-08

## Objective

Implement a lightweight, executable Phase 1 experiment that turns the included Bishan AlphaEarth sample into region-level GeoFM representations and a conservative suitability proxy. This phase moves Paper11 beyond a reviewer skeleton while preserving the current-state suitability scope.

Phase 1 is not a full DRL training run. It is a representation smoke experiment that proves the repository can read the included sample data, aggregate embeddings into planning-like regions, compute bounded proxy scores, and export inspectable artifacts.

## Scope

In scope:

- read `data/bishan_alphaearth_sample/metadata.json`;
- load annual `bishan_emb_2017.npy` through `bishan_emb_2024.npy`;
- optionally load `bishan_context.npy` when shape-compatible;
- build deterministic grid or KMeans regions from a selected annual embedding;
- aggregate each region into mean 64-dimensional GeoFM features;
- compute simple embedding dispersion features;
- compute a bounded suitability proxy from embedding similarity and stability cues;
- export JSON and CSV summaries under `experiments/phase1_bishan_baseline/outputs/`;
- add tests for input loading, aggregation shape, suitability score range, and artifact schema.

Out of scope:

- DRL policy training;
- changing `src/legacy_runtime/county_env.py`;
- future land-state prediction;
- scenario actions;
- Google Earth Engine extraction;
- direct claims about soil quality, fertility, or irrigation access;
- large artifact distribution.

## Architecture

Add a small package:

```text
src/paper11_geofm/
  __init__.py
  sample_data.py
  regions.py
  features.py
  suitability.py
  artifacts.py
```

Responsibilities:

- `sample_data.py`: load metadata and memory-map annual embeddings.
- `regions.py`: create deterministic region labels. The default should be a fixed spatial grid because it is stable and has no stochastic dependency. A KMeans option can be added behind an explicit argument if needed.
- `features.py`: compute region-level mean embedding, standard deviation, pixel count, and optional context statistics.
- `suitability.py`: compute a bounded proxy score in `[0, 1]` using only latent remote-sensing representation properties. The first version should use distance to a high-stability embedding centroid and a dispersion penalty. The name must remain `suitability_proxy`, not `soil_quality`, `fertility`, or `irrigation`.
- `artifacts.py`: write machine-readable JSON summary and CSV region table.

Add an experiment entry point:

```text
experiments/phase1_bishan_baseline/run_phase1.py
```

Default command:

```powershell
python experiments\phase1_bishan_baseline\run_phase1.py
```

## Data Flow

```text
Bishan annual embeddings + metadata
  -> validate years, grid shape, embedding dimension
  -> select base year, default 2020
  -> create deterministic region labels
  -> aggregate region-level GeoFM features
  -> estimate temporal stability centroid from all years
  -> compute suitability_proxy per region
  -> export region_features.csv and summary.json
```

## Suitability Proxy

The proxy must be explicitly bounded and conservative:

```text
suitability_proxy =
  sigmoid_or_minmax(
    similarity_to_stable_embedding_centroid
    - dispersion_penalty
  )
```

Interpretation:

- high score means the region is close to stable latent remote-sensing states in the included sample;
- low score means the region is distant from that centroid or internally heterogeneous;
- the score is a planning representation proxy, not an agronomic measurement.

## Artifact Schema

`region_features.csv` should include at least:

```text
region_id
pixel_count
row_min
row_max
col_min
col_max
embedding_mean_00 ... embedding_mean_63
embedding_std_mean
temporal_stability
suitability_proxy
```

`summary.json` should include:

```text
metadata_source
base_year
years
grid_shape
embedding_dim
n_regions
region_table
suitability_min
suitability_max
suitability_mean
claim_boundary
```

The `claim_boundary` field should state that the proxy is derived from latent remote-sensing embeddings and does not directly measure soil, fertility, or irrigation.

## Testing

Add tests before implementation:

- sample loader returns metadata years 2017-2024 and embedding dimension 64;
- deterministic region labels cover the full grid and have expected shape;
- region feature table has one row per region and 64 embedding mean columns;
- suitability proxy values are finite and within `[0, 1]`;
- experiment runner writes both JSON and CSV artifacts to a caller-provided output directory.

The tests should use the included Bishan sample and small temporary output directories. They must not require internet, GPU, Earth Engine, or full DRL dependencies.

## Acceptance Criteria

Phase 1 is complete when:

- `python experiments\phase1_bishan_baseline\run_phase1.py` exits with code 0;
- output artifacts are created under `experiments/phase1_bishan_baseline/outputs/`;
- `python scripts\smoke_check.py` passes;
- `python -m pytest tests` passes;
- README or reproduction docs describe the Phase 1 command;
- `reproducibility/FILE_MANIFEST.tsv` lists the new package, experiment, tests, and docs.

## Risks

- The Bishan sample is a grid-level embedding cache, not a block polygon dataset. Phase 1 regions are planning-like analysis regions, not official farmland blocks.
- Suitability proxy validation remains weak until external farmland, slope, irrigation, soil, or high-standard farmland labels are linked.
- KMeans can introduce stochastic differences. The default grid partition should be deterministic for reviewer reproducibility.

## Recommended Next Step

After this design is approved, write an implementation plan and then implement the smallest end-to-end slice: deterministic grid partition, region aggregation, bounded proxy score, JSON/CSV export, and tests.
