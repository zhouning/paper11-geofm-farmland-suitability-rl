# Phase 60 Information-vs-Optimization Attribution

Phase 60 reconciles the compressed-route evidence chain with the
matched-dimension control result. It is a read-only attribution audit over the
existing Phase 48/52, Phase 53, Phase 57, and Phase 59 artifacts; it does not
retrain policies or revise the formal manuscript.

## Status

`mechanism_claim_narrowed`

## Attribution Axes

| Axis | Source | Status | Primary metric | Primary value |
|---|---|---|---|---:|
| Compressed route performance | Phase 48/52 | supported | pooled_mean_delta | 0.2921767818 |
| Cluster-level robustness | Phase 53 | supported | cluster_mean_delta | 0.2921767818 |
| Compressed geometry consistency | Phase 57 | supported | max_compressed_effective_rank | 7.3009059917 |
| GeoFM-specific matched dimension | Phase 59 | not_supported | pooled_matched_control_mean_delta | -0.0172307641 |

## Interpretation

D4P8/D4P16 remain supported against B0, raw B1, random D2, and shuffled D3
under the expanded five-tile, three-seed Bishan base-reward protocol. Phase 53
supports the expanded cluster mean, and Phase 57 shows that D4P8/D4P16 preserve
most raw GeoFM variance while reducing effective rank and conditioning burden.

Phase 59 prevents a stronger attribution claim. D4P8/D4P16 do not clearly
outperform same-dimension random or shuffled controls in the current run, so
the evidence supports a bounded low-dimensional compressed state route rather
than a proven GeoFM-specific same-dimension advantage.

The Phase 60 recommendation is `narrow_to_low_dimensional_route`. Before a
stronger mechanism claim, the next optional experiment should add D6-style
GeoFM projection controls that separate GeoFM-derived information from generic
low-dimensional optimization effects.

## Reproduction

Run Phase 60 from the repository root after the Phase 48/52, Phase 53, Phase
57, and Phase 59 artifacts exist:

```powershell
python experiments\phase60_information_optimization_attribution\run_phase60_information_optimization_attribution.py --phase48-json experiments\phase52_expanded_cluster_replication\outputs\phase52_full5_seed3_phase48_compressed_rescue\phase48_compressed_geofm_rescue_comparison.json --phase53-json experiments\phase53_cluster_mean_support\outputs\phase52_full5_seed3\phase53_cluster_mean_support.json --phase57-json experiments\phase57_compressed_representation_mechanism\outputs\phase52_full5_seed3\phase57_compressed_representation_mechanism.json --phase59-json experiments\phase59_matched_dimension_controls\outputs\phase52_full5_seed3\phase59_matched_dimension_controls.json --output-dir experiments\phase60_information_optimization_attribution\outputs\phase52_full5_seed3
```

Expected status: `mechanism_claim_narrowed`.

## Boundary

Phase 60 does not enable suitability reward, B2/B3, cross-region transfer, PCA
optimality, independent agronomic suitability claims, or submission-level
planning-performance claims. No formal manuscript files were changed in this
phase.

## Artifacts

- `experiments/phase60_information_optimization_attribution/outputs/phase52_full5_seed3/phase60_information_optimization_attribution.json`
- `experiments/phase60_information_optimization_attribution/outputs/phase52_full5_seed3/phase60_attribution_axes.csv`
- `experiments/phase60_information_optimization_attribution/outputs/phase52_full5_seed3/phase60_information_optimization_attribution.md`