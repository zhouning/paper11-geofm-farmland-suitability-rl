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
