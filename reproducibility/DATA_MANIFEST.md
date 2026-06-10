# Data Manifest

## Included Lightweight Sample

Directory:

```text
data/bishan_alphaearth_sample/
```

Files:

| File | Role |
|---|---|
| `metadata.json` | Bounding box, years, scale, grid shape, embedding dimension, and source collection. |
| `bishan_context.npy` | Lightweight terrain/context array used with the Bishan embeddings. |
| `bishan_emb_2017.npy` through `bishan_emb_2024.npy` | Annual AlphaEarth embedding grids for Bishan, final dimension 64. |

Source collection recorded in metadata:

```text
GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
```

The included sample is intended for repository smoke tests and reviewer inspection. It is not a complete experiment dataset.

## Included Phase 2 CSV Fixture

Directory:

```text
data/bishan_phase2_csv_sample/
```

Files:

| File | Role |
|---|---|
| `block_pixel_mapping.csv` | Tiny block-to-pixel mapping table with four sample blocks and eight mapped Bishan grid pixels. |
| `block_attributes.csv` | Tiny block attribute table with 17 explicit feature columns, weak labels, and split labels. |

This fixture is for exercising the Phase 2 `--mapping-csv` and `--attributes-csv` code path. It is not a real parcel/block planning dataset and should not be interpreted as real block-level evidence.

## External Local Real-Data Source Used by Phase 11

Path on the current workstation:

```text
D:\test\dem_slope_analysis\output\DLTB_with_slope.gpkg
```

Role:

- real Bishan DLTB polygon source with slope attributes from previous paper work;
- read by `experiments/phase11_bishan_dltb_real/run_phase11_bishan_dltb_adapter.py`;
- converted into Phase 2-compatible `block_pixel_mapping.csv` and `block_attributes.csv` artifacts;
- not committed to ordinary Git because it is a large external geospatial artifact with separate data provenance.

Observed local properties:

| Property | Value |
|---|---|
| Layer | `DLTB` |
| CRS | `EPSG:4326` |
| Polygon count | 101,657 |
| File size | approximately 160 MB |
| Key fields | `BSM`, `DLBM`, `DLMC`, `TBMJ`, `category`, `slope_mean`, `slope_max`, `slope_pixel_count` |
| Bishan AlphaEarth bbox rows read | 65,146 |
| Adapter rows exported | 64,984 |

This source enables real Bishan DLTB feature-table experiments. It does not by itself provide parcel-accurate GeoFM overlap, high-standard farmland labels, stable farmland labels, or DRL policy-performance evidence.

## Deliberately Excluded Large Artifacts

The source workspace also contained larger arrays and trained artifacts, including:

```text
experiments/paper8/data/village/*.npy
experiments/paper8/data/heping/*.npy
experiments/paper8/data/intervention_transitions.npz
experiments/paper8/data/block_feature_encoder.pt
experiments/paper8/data/intervention_dynamics.pt
experiments/paper8/weights/
experiments/paper8/results/
```

These are excluded from ordinary Git because several arrays are tens of megabytes each and model or result artifacts should be archived deliberately with checksums.

## Recommended Full-Release Handling

For a full reproducibility release:

1. Store large `.npy`, `.npz`, `.pt`, and result artifacts in Git LFS, Zenodo, OSF, or an institutional data repository.
2. Add SHA256 checksums and byte sizes for each archived artifact.
3. Record the exact extraction date, Earth Engine collection identifier, spatial extent, scale, and projection assumptions.
4. Keep the Paper11 claim boundary explicit: AlphaEarth embeddings are latent remote-sensing proxies, not direct soil, fertility, or irrigation measurements.
