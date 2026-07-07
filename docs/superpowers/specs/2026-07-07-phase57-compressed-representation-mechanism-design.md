# Phase 57 Compressed Representation Mechanism Design

## Purpose

Phase 57 adds a read-only mechanism audit for the current Paper11 conclusion. The audit does not retrain policies and does not introduce a new reward. It explains why the supported result is a compressed GeoFM state route rather than raw 64-dimensional embedding injection.

## Scientific Question

Does the feature geometry of the current real Bishan state tables support the manuscript interpretation that PCA-compressed GeoFM routes improve planning because they provide a lower-dimensional and better-conditioned representation of the raw embedding signal?

## Inputs

- `experiments/phase11_bishan_dltb_real/outputs/phase2_real/variant_B1_features.csv`
- `experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P8_features.csv`
- `experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P16_features.csv`
- `experiments/phase52_expanded_cluster_replication/outputs/phase52_full5_seed3_phase48_compressed_rescue/phase48_compressed_geofm_rescue_delta_table.csv`
- `experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv`

## Method

The audit aligns feature rows by `block_id` and computes representation geometry for the raw GeoFM dimensions and both compressed PCA state routes:

- row count and feature count;
- total centered variance;
- explained raw-variance retention for D4P8 and D4P16, using compressed score variance divided by raw embedding variance;
- effective rank and participation ratio from covariance eigenvalues;
- condition number from positive covariance eigenvalues;
- feature standard-deviation spread;
- expanded-replication reward-gain summaries by compressed variant;
- diagnostic tile-level association between tile mean reward gain and tile-level compressed geometry.

## Claim Boundary

Phase 57 is a read-only geometry and mechanism audit. It can support the plausibility of the compressed-route interpretation, but it does not prove that PCA is optimal, does not add a new RL training result, does not enable B2/B3 suitability reward, does not test transfer, and does not validate agronomic suitability.

## Status Rule

The audit reports `compressed_geometry_consistent` when all rows align, both compressed variants retain nonzero raw embedding variance, compressed effective ranks are lower than the raw effective rank, and the expanded-replication reward-gain summaries remain positive for D4P8 and D4P16.

## Outputs

- `src/paper11_geofm/phase57_compressed_representation_mechanism.py`
- `experiments/phase57_compressed_representation_mechanism/run_phase57_compressed_representation_mechanism.py`
- `tests/test_phase57_compressed_representation_mechanism.py`
- `experiments/phase57_compressed_representation_mechanism/outputs/phase52_full5_seed3/phase57_compressed_representation_mechanism.json`
- `experiments/phase57_compressed_representation_mechanism/outputs/phase52_full5_seed3/phase57_representation_geometry.csv`
- `experiments/phase57_compressed_representation_mechanism/outputs/phase52_full5_seed3/phase57_tile_geometry_gain.csv`
- `experiments/phase57_compressed_representation_mechanism/outputs/phase52_full5_seed3/phase57_compressed_representation_mechanism.md`
- manuscript updates in `paper/submission/final/Paper11_formal_conclusion_manuscript.md`, `.tex`, and `.pdf`