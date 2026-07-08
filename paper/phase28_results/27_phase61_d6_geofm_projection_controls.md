# Phase 61 D6 GeoFM Projection Controls

Phase 61 builds and audits D6 GeoFM-derived projection controls after Phase 60
narrowed the mechanism claim. It prepares trainable same-dimension controls for
a later matched PPO experiment, but does not train policies in this phase.

## Status

`d6_projection_controls_ready_for_training`

## Generated Controls

| Variant | Projection type | Dimension | Rows | Raw-B1 variance retention | Effective rank | D4 similarity |
|---|---|---:|---:|---:|---:|---:|
| D6R8 | random orthonormal raw-B1 projection | 8 | 64984 | 0.1257182217 | 4.2966518122 | 0.13402939 |
| D6P8 | PCA raw-B1 projection | 8 | 64984 | 0.8587823898 | 5.1322783588 | 1.0 |
| D6R16 | random orthonormal raw-B1 projection | 16 | 64984 | 0.2492633812 | 6.9432558615 | 0.1827507258 |
| D6P16 | PCA raw-B1 projection | 16 | 64984 | 0.9496006154 | 7.3009059917 | 1.0 |

## Interpretation

All D6 feature tables preserve the `64,984` aligned B0/B1/D4 block rows and copy
all `17` explicit planning columns. D6R8, D6P8, D6R16, and D6P16 all have the
expected projection dimensions and nonzero centered projection variance, so the
D6 projection controls are ready for a later matched training run under the
Phase 52 five-tile, three-seed base-reward protocol.

The D6P diagnostics are also a lineage check for the existing D4 route. D6P8
matches D4P8 with mean absolute column correlation `1.0`, and D6P16 matches
D4P16 with mean absolute column correlation `1.0`. This confirms that the
current D4P8/D4P16 tables are reproducible raw-B1 PCA projections in the Phase
61 implementation.

The D6R controls provide the new experimental contrast. D6R8 and D6R16 are
GeoFM-derived random orthonormal projections from raw B1, not row-shuffled or
moment-matched noise controls. Their D4 similarity is low (`0.13402939` and
`0.1827507258`), so they are suitable next controls for separating
GeoFM-derived low-dimensional information from PCA-specific variance ordering.

## Reproduction

Run Phase 61 from the repository root after the B0/B1 and D4 feature tables
exist:

```powershell
python experiments\phase61_d6_geofm_projection_controls\run_phase61_d6_geofm_projection_controls.py --b0-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\variant_B0_features.csv --b1-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\variant_B1_features.csv --d4p8-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P8_features.csv --d4p16-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P16_features.csv --output-dir experiments\phase61_d6_geofm_projection_controls\outputs\phase52_full5_seed3 --dimensions 8,16 --seed 61
```

Expected status: `d6_projection_controls_ready_for_training`.

## Boundary

Phase 61 does not train PPO policies, does not compare learned rewards, does
not enable suitability reward, does not test B2/B3, does not test transfer, does
not prove PCA optimality, and does not validate independent agronomic
suitability. No formal manuscript files were changed in this phase.

## Artifacts

- `experiments/phase61_d6_geofm_projection_controls/outputs/phase52_full5_seed3/variant_D6R8_features.csv`
- `experiments/phase61_d6_geofm_projection_controls/outputs/phase52_full5_seed3/variant_D6P8_features.csv`
- `experiments/phase61_d6_geofm_projection_controls/outputs/phase52_full5_seed3/variant_D6R16_features.csv`
- `experiments/phase61_d6_geofm_projection_controls/outputs/phase52_full5_seed3/variant_D6P16_features.csv`
- `experiments/phase61_d6_geofm_projection_controls/outputs/phase52_full5_seed3/experiment_variants.json`
- `experiments/phase61_d6_geofm_projection_controls/outputs/phase52_full5_seed3/phase61_d6_projection_feature_summary.json`
- `experiments/phase61_d6_geofm_projection_controls/outputs/phase52_full5_seed3/phase61_d6_projection_geometry.json`
- `experiments/phase61_d6_geofm_projection_controls/outputs/phase52_full5_seed3/phase61_d6_projection_geometry.csv`
- `experiments/phase61_d6_geofm_projection_controls/outputs/phase52_full5_seed3/phase61_d6_projection_similarity.csv`
- `experiments/phase61_d6_geofm_projection_controls/outputs/phase52_full5_seed3/phase61_d6_projection_controls.md`
