# Phase 11 Bishan DLTB Real-Data Adapter Design

## Goal

Add a Phase 11 adapter that uses the real Bishan DLTB land-use dataset with
slope attributes as Paper11 input data. The adapter should convert
`DLTB_with_slope.gpkg` into Phase 2-compatible `block_pixel_mapping.csv` and
`block_attributes.csv`, then let the existing Phase 2, Phase 9, and Phase 10
commands run on real Bishan DLTB-derived blocks.

## Rationale

The standalone Paper11 repository already has smoke fixtures and an AlphaEarth
sample for Bishan, but the previous Phase 2 CSV fixture is tiny and synthetic.
The older paper workspace contains a real Bishan DLTB source:

```text
D:\test\dem_slope_analysis\output\DLTB_with_slope.gpkg
```

This file has 101,657 DLTB polygons and includes `DLBM`, `DLMC`, `TBMJ`,
`category`, `slope_mean`, `slope_max`, and `slope_pixel_count`. Within the
Bishan AlphaEarth sample bbox, roughly 65k polygons are available, including
about 25k farmland polygons. This is enough to replace the tiny Phase 2 fixture
with a real-data input path.

## Scope

Create a focused adapter:

```text
real Bishan DLTB_with_slope.gpkg
  + Paper11 Bishan AlphaEarth metadata bbox/grid
  -> centroid-to-AlphaEarth-grid assignment
  -> block_pixel_mapping.csv
  -> block_attributes.csv
  -> phase11_bishan_dltb_adapter_summary.json
```

The adapter must not copy the 160 MB DLTB file into the Git repository. It
should read it from a user-supplied local path and write generated CSV/JSON
artifacts under ignored `experiments/phase11_bishan_dltb_real/outputs/`.

## Data Alignment

Use the metadata in:

```text
data/bishan_alphaearth_sample/metadata.json
```

Required fields:

- `bbox`: `[min_lon, min_lat, max_lon, max_lat]`;
- `grid_shape`: `[n_rows, n_cols]`;
- `scale_m`;
- `embedding_dim`.

For this first real adapter, assign each DLTB polygon to one AlphaEarth pixel by
its centroid:

```text
col = floor((lon - min_lon) / (max_lon - min_lon) * n_cols)
row = floor((max_lat - lat) / (max_lat - min_lat) * n_rows)
```

Rows and columns are clipped to the valid grid. Only polygons whose centroid is
inside the AlphaEarth bbox are exported. This is a deterministic real-data
adapter, not a final sub-pixel area-overlap alignment method.

## Output CSVs

### block_pixel_mapping.csv

Columns:

```text
block_id,row,col,weight
```

Use one DLTB polygon as one `block_id`, with:

- `block_id = dltb_<BSM>`;
- `row` and `col` from centroid-to-grid assignment;
- `weight = 1.0`.

### block_attributes.csv

Columns:

```text
block_id,
explicit_feature_00 through explicit_feature_16,
current_farmland_label,
low_slope_farmland_label,
farmland_or_orchard_label,
split,
source_bsm,
source_dlbm,
source_dlmc,
source_category,
area_m2,
slope_mean,
slope_max,
slope_pixel_count
```

The 17 explicit features should be deterministic DLTB-derived planning
features:

| Feature | Meaning |
|---|---|
| `explicit_feature_00` | area in hectares |
| `explicit_feature_01` | `slope_mean` |
| `explicit_feature_02` | `slope_max` |
| `explicit_feature_03` | `slope_pixel_count` |
| `explicit_feature_04` | farmland indicator |
| `explicit_feature_05` | paddy indicator (`DLBM == 011`) |
| `explicit_feature_06` | dryland indicator (`DLBM == 013`) |
| `explicit_feature_07` | orchard indicator |
| `explicit_feature_08` | forest indicator |
| `explicit_feature_09` | built-up indicator |
| `explicit_feature_10` | water indicator |
| `explicit_feature_11` | facility-agriculture indicator |
| `explicit_feature_12` | grass or bare-land indicator |
| `explicit_feature_13` | low-slope indicator (`slope_mean <= 6`) |
| `explicit_feature_14` | moderate-slope indicator (`6 < slope_mean <= 15`) |
| `explicit_feature_15` | high-slope indicator (`slope_mean > 15`) |
| `explicit_feature_16` | low-slope farmland-or-orchard candidate indicator |

Weak labels:

- `current_farmland_label`: `1` for DLTB farmland classes, else `0`;
- `low_slope_farmland_label`: `1` for current farmland with `slope_mean <= 6`,
  else `0`;
- `farmland_or_orchard_label`: `1` for farmland or orchard, else `0`.

Do not call these labels stable farmland or high-standard farmland. They are
real DLTB-derived current-state weak labels only.

## Summary Artifact

Write `phase11_bishan_dltb_adapter_summary.json` with:

- DLTB source path;
- metadata source path;
- bbox and grid shape;
- rows read in bbox;
- rows exported after centroid filtering;
- DLTB CRS;
- category counts;
- label positive counts;
- slope summary;
- output filenames;
- claim boundary.

## Claim Boundary

Use an explicit Phase 11 claim boundary:

```text
Phase 11 builds real Bishan DLTB-derived Phase 2 inputs; centroid-to-grid
assignment is an alignment adapter, not final parcel-accurate GeoFM evidence,
and this phase does not train or evaluate a DRL policy.
```

## Public API

Create `src/paper11_geofm/dltb_adapter.py` with:

```python
PHASE11_CLAIM_BOUNDARY = (
    "Phase 11 builds real Bishan DLTB-derived Phase 2 inputs; "
    "centroid-to-grid assignment is an alignment adapter, not final "
    "parcel-accurate GeoFM evidence, and this phase does not train or "
    "evaluate a DRL policy."
)


def build_bishan_dltb_phase2_inputs(
    dltb_path: Path | str,
    metadata_path: Path | str,
    max_blocks: int | None = None,
) -> dict[str, object]:
    """Return mapping rows, attribute rows, and summary for real Bishan DLTB."""


def write_bishan_dltb_phase2_inputs(
    payload: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    """Write mapping CSV, attributes CSV, and summary JSON."""
```

The builder may import `geopandas` inside the function so the base package can
still be imported in environments without geospatial extras. Missing geospatial
dependencies should raise a clear `ImportError`.

## CLI

Create:

```text
experiments/phase11_bishan_dltb_real/run_phase11_bishan_dltb_adapter.py
```

Arguments:

- `--dltb-path`;
- `--metadata-path`, defaulting to
  `data/bishan_alphaearth_sample/metadata.json`;
- `--output-dir`;
- `--max-blocks`, optional smoke cap.

The CLI prints rows exported, category counts, label positive counts, output
paths, and the claim boundary.

## Reviewer Workflow

Local real-data workflow:

```powershell
python experiments\phase11_bishan_dltb_real\run_phase11_bishan_dltb_adapter.py --dltb-path D:\test\dem_slope_analysis\output\DLTB_with_slope.gpkg --output-dir experiments\phase11_bishan_dltb_real\outputs\adapter
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv experiments\phase11_bishan_dltb_real\outputs\adapter\block_pixel_mapping.csv --attributes-csv experiments\phase11_bishan_dltb_real\outputs\adapter\block_attributes.csv --output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real
python experiments\phase9_proxy_validation\run_phase9_proxy_validation.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase11_bishan_dltb_real\outputs\phase9_real --label-columns current_farmland_label,low_slope_farmland_label,farmland_or_orchard_label
python experiments\phase10_reward_readiness\run_phase10_reward_readiness.py --phase9-report experiments\phase11_bishan_dltb_real\outputs\phase9_real\phase9_proxy_validation_report.json --output-dir experiments\phase11_bishan_dltb_real\outputs\phase10_real --required-labels current_farmland_label,low_slope_farmland_label,farmland_or_orchard_label
```

The full real DLTB file is local and large, so these generated outputs remain
ignored by Git. The commands should also work with `--max-blocks` for quick
local smoke checks.

## Non-Goals

Do not:

- copy the full DLTB dataset into Git;
- claim parcel-accurate embedding overlap from centroid assignment;
- treat `current_farmland_label` as stable or high-standard farmland;
- train, tune, evaluate, or report a DRL policy;
- change legacy runtime environments;
- make claims about AlphaEarth directly measuring soil, fertility, or
  irrigation.

## Test Strategy

Use TDD. Add tests with a tiny synthetic GeoDataFrame written to a temporary
GeoPackage:

- verify centroid-to-grid row/column assignment;
- verify all 17 explicit features are written;
- verify weak labels are derived from DLTB fields;
- verify `max_blocks` caps rows deterministically;
- verify writer creates mapping CSV, attributes CSV, and summary JSON;
- verify CLI prints concise adapter output.

The full `D:\test` DLTB file is exercised by the reviewer workflow, not by unit
tests, so the test suite stays portable.
