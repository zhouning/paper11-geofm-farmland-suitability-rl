# Phase 59 Matched-Dimension Controls

Phase 59 tests whether the D4P8/D4P16 compressed GeoFM gains exceed
same-dimension random and shuffled controls under the same Bishan base-reward
held-out protocol used for the expanded Phase 52 evidence.

## Status

`matched_dimension_geofm_not_supported`

## Matched Comparisons

| Comparison | Mean delta | Positive rows |
|---|---:|---:|
| D4P8 - D5R8 | -0.0107871307 | 5 / 15 |
| D4P8 - D5S8 | 0.0003232239 | 7 / 15 |
| D4P16 - D5R16 | -0.1193811247 | 2 / 15 |
| D4P16 - D5S16 | 0.060921975 | 8 / 15 |

## Pooled and Cluster Evidence

- Pooled mean delta: -0.0172307641
- Pooled positive rows: 22 / 60
- Pooled bootstrap CI95: [-0.1081223337, 0.0751760409]
- Cluster mean delta: -0.0172307641
- Positive tile-seed clusters: 8 / 15
- Cluster sign-test p: 0.5
- Signed-rank positive rank sum: 55 / 120
- Signed-rank p: 0.6192321777

## Interpretation

Phase 59 does not support a GeoFM-specific matched-dimension advantage for the
compressed D4P8/D4P16 route. D4P8 and D4P16 remain part of the earlier positive
compressed-route evidence against B0, raw B1, random D2, and shuffled D3, but
the Phase 59 audit shows that their advantage does not clearly exceed
same-dimension random or shuffled controls in the current five-tile, three-seed
Bishan base-reward protocol.

The mechanism wording should therefore be narrowed. The defensible claim is
that low-dimensional compressed state routes can outperform the earlier
explicit/raw/control variants under the current protocol; the stronger claim
that the current PCA-compressed GeoFM coordinates provide a unique advantage
over matched low-dimensional controls is not supported by this audit.

## Boundary

Phase 59 does not enable suitability reward, B2/B3, cross-region transfer, PCA
optimality, or independent agronomic suitability claims. It is a matched-control
mechanism audit over base-reward Bishan held-out policy runs, not a final
submission-level planning-performance claim.

## Artifacts

- `experiments/phase59_matched_dimension_controls/outputs/phase52_full5_seed3/phase59_matched_dimension_controls.json`
- `experiments/phase59_matched_dimension_controls/outputs/phase52_full5_seed3/phase59_matched_dimension_control_summary.csv`
- `experiments/phase59_matched_dimension_controls/outputs/phase52_full5_seed3/phase59_matched_dimension_delta_table.csv`
- `experiments/phase59_matched_dimension_controls/outputs/phase52_full5_seed3/phase59_matched_dimension_controls.md`
