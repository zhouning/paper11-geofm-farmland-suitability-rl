# Phase 8 Ablation-Control Feature Tables Design

## Goal

Add a lightweight Phase 8 pipeline that derives diagnostic ablation/control
feature tables from ready Phase 2 B-variant outputs. The phase should create
table-ready controls for dimensionality, spatial alignment, and compressed
GeoFM representation checks without training, tuning, evaluating, or reporting a
DRL policy.

## Rationale

Paper11's experiment plan identifies a central risk: raw 64-dimensional GeoFM
features may appear useful simply because they add input capacity rather than
semantic remote-sensing information. Phase 7 verified that the masked-policy
stack can consume the Phase 4 environment, but it still does not address this
ablation risk. Phase 8 fills that gap by making the D-control tables executable
and reproducible before any real policy training.

## Scope

Create a focused ablation-control table builder:

```text
Phase 2 ready B0/B1 feature tables
  -> deterministic random-64 dimensionality control
  -> deterministic shuffled-GeoFM spatial-alignment control
  -> deterministic PCA-compressed GeoFM controls
  -> Phase 3-compatible experiment_variants.json
  -> JSON summary artifact
```

Default source inputs are the Phase 2 fixture outputs with ready B0 and B1
tables. Phase 8 must not mutate Phase 2 outputs. It writes all generated
controls into a requested output directory.

## Diagnostic Variants

Generate these variants:

| ID | State | Reward | Purpose |
|---|---|---|---|
| D2 | explicit features + deterministic random 64d | base reward | controls for extra dimensionality without GeoFM semantics |
| D3 | explicit features + block-shuffled GeoFM 64d | base reward | controls for spatial semantic alignment |
| D4P8 | explicit features + PCA-8 GeoFM projection | base reward | tests compact GeoFM representation wiring |
| D4P16 | explicit features + PCA-16 GeoFM projection | base reward | tests wider compact GeoFM representation wiring |

D2 and D3 should keep the canonical `embedding_mean_00` through
`embedding_mean_63` column names so existing downstream code sees the same
feature count as B1. Their manifest `state_groups` distinguish
`random_embedding_control` and `shuffled_geofm_embedding`.

D4P8 and D4P16 should use `embedding_pca_00`-style column names. For tiny
fixtures where the available matrix rank is lower than 8 or 16, the projection
must still emit the requested number of columns by zero-padding trailing
components. This keeps the smoke contract deterministic while avoiding a false
requirement for large sample counts.

## Artifacts

Write these files under the requested output directory:

```text
experiment_variants.json
phase8_ablation_control_summary.json
variant_D2_features.csv
variant_D3_features.csv
variant_D4P8_features.csv
variant_D4P16_features.csv
```

`experiment_variants.json` should follow the existing Phase 3 loader contract:

- top-level `claim_boundary`;
- top-level `variants` object;
- per-variant `description`, `state_groups`, `reward`, `required_columns`,
  `ready`, `missing`, `feature_table`, and `row_count`.

The summary records:

- source Phase 2 output directory;
- source B0/B1 row counts and feature counts;
- seed;
- generated variant IDs;
- per-variant row count and feature count;
- shuffle permutation for D3;
- PCA dimensions requested and emitted;
- artifact filenames;
- claim boundary.

## Claim Boundary

Use an explicit Phase 8 claim boundary:

```text
Phase 8 builds diagnostic ablation-control feature tables; it does not train,
tune, evaluate, compare, or report a useful DRL policy.
```

The generated controls are table-readiness artifacts only. They do not prove
that GeoFM features improve planning, that PCA preserves useful information, or
that random/shuffled controls underperform a real model.

## Non-Goals

Do not:

- run MaskablePPO learning or any other policy training;
- run rollout comparisons or report rewards;
- compute planning, transfer, slope, contiguity, baimu-fang, compactness, or
  policy-quality metrics;
- modify Phase 2 outputs in place;
- modify the legacy county or parcel environments;
- claim that AlphaEarth embeddings directly measure soil quality, fertility, or
  irrigation;
- frame random, shuffled, or PCA controls as scientific results before real
  evaluation exists.

## Determinism

D2 random values should be deterministic for a given seed. To keep random
columns on a comparable numeric scale, generate a standard-normal matrix and
rescale each column to the source B1 embedding column mean and standard
deviation, using a safe unit standard deviation when a source column is
constant.

D3 should use a deterministic block permutation for a given seed. If the
sample has more than one block and the generated permutation is the identity,
rotate it by one position so the control actually breaks block-to-embedding
alignment in smoke fixtures.

D4P8 and D4P16 should compute PCA via deterministic NumPy SVD on centered B1
embedding columns. The implementation should avoid adding a new dependency
because NumPy is already required.

## Public API

Create `src/paper11_geofm/ablation_controls.py` with:

```python
PHASE8_CLAIM_BOUNDARY = (
    "Phase 8 builds diagnostic ablation-control feature tables; it does not "
    "train, tune, evaluate, compare, or report a useful DRL policy."
)


def build_phase8_ablation_controls(
    phase2_output_dir: Path | str,
    seed: int = 0,
    pca_dimensions: Sequence[int] = (8, 16),
) -> dict[str, object]:
    ...


def write_phase8_ablation_artifacts(
    protocol: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    ...
```

The build function should load Phase 2 B0 and B1 variant inputs using the
existing `load_variant_input()` API, verify that block IDs align, derive the D
variant rows and manifest metadata, and return an in-memory protocol. The writer
should create CSV files, `experiment_variants.json`, and
`phase8_ablation_control_summary.json`.

## CLI

Create:

```text
experiments/phase8_ablation_controls/run_phase8_ablation_controls.py
```

The command accepts:

- `--phase2-output-dir`;
- `--output-dir`;
- `--seed`, defaulting to `0`;
- `--pca-dimensions`, defaulting to `8,16`.

It prints source row count, generated variant IDs, per-variant feature counts,
artifact paths, and the claim boundary.

## Documentation

Update:

- `README.md`: add a Phase 8 command after Phase 7;
- `reproducibility/REPRODUCTION_GUIDE.md`: add expected Phase 8 controls,
  artifact files, deterministic seed behavior, and no-policy-performance
  boundary;
- `reproducibility/FILE_MANIFEST.tsv`: add the design, plan, module, CLI, and
  test file.

Use `experiments/phase8_ablation_controls/outputs/` for reviewer command
examples because generated experiment outputs are ignored by Git.

## Test Strategy

Use TDD. Add tests that:

- build ready Phase 2 fixture outputs with 4 blocks;
- verify D2, D3, D4P8, and D4P16 are generated and ready;
- verify feature counts are D2 = 81, D3 = 81, D4P8 = 25, and D4P16 = 33;
- verify D2 is deterministic for the same seed and seed-sensitive for a
  changed seed;
- verify D3 uses a non-identity permutation when more than one block exists;
- verify PCA outputs keep the requested component counts through zero-padding;
- verify `load_variant_input()` can load generated D variants from the Phase 8
  output directory;
- verify JSON and CSV artifacts are written;
- verify the CLI prints a concise artifact summary.

All tests remain offline and CPU-only and make no policy-performance claims.
