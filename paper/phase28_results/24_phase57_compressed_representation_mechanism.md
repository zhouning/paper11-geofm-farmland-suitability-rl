# Phase 57 Compressed Representation Mechanism

## Purpose

This read-only audit tests the mechanism behind the current Paper11 conclusion:
compressed GeoFM state routes work under the Bishan base-reward protocol, while
raw 64-dimensional B1 injection does not. It does not retrain RL policies and
does not introduce a suitability reward.

## Inputs

- `experiments/phase11_bishan_dltb_real/outputs/phase2_real/variant_B1_features.csv`
- `experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P8_features.csv`
- `experiments/phase8_ablation_controls/outputs/real_bishan_controls/variant_D4P16_features.csv`
- `experiments/phase52_expanded_cluster_replication/outputs/phase52_full5_seed3_phase48_compressed_rescue/phase48_compressed_geofm_rescue_delta_table.csv`
- `experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv`

## Result

Status: `compressed_geometry_consistent`.

All three feature tables aligned over `64,984` Bishan blocks. Raw B1 had `64`
GeoFM dimensions, total centered variance `0.0981484274`, effective rank
`9.4947211626`, and positive-eigenvalue condition number `6658.9542931381`.
D4P8 retained `0.0842881410` variance, or `85.87823898%` of raw GeoFM
variance, while reducing effective rank to `5.1322783588` and condition number
to `16.2704676982`. D4P16 retained `0.0932018070` variance, or
`94.96006154%` of raw GeoFM variance, while reducing effective rank to
`7.3009059917` and condition number to `53.6978527088`.

The expanded-replication reward gains remained positive when summarized by
compressed variant. D4P8 had mean compressed-minus-control reward gain
`0.2356980264` with `38 / 60` positive comparisons. D4P16 had mean gain
`0.3486555373` with `36 / 60` positive comparisons.

Tile-level retention-gain correlations were not a strong positive pattern:
`-0.0207226322` for D4P8, `-0.2059768413` for D4P16, and `0.0257762396`
pooled. This keeps the mechanism claim bounded. The audit supports a global
representation-geometry interpretation, not a monotonic per-tile
variance-retention rule.

## Manuscript Interpretation

The current evidence supports the conclusion that GeoFM can help the Bishan
planning policy when represented through controlled compressed state features.
The mechanism evidence is that compression retained most of the raw GeoFM
variance while reducing effective rank and covariance conditioning burden.
Raw B1 superiority, PCA optimality, B2/B3 suitability reward, transfer, and
independent agronomic suitability remain unsupported.

## Outputs

- `experiments/phase57_compressed_representation_mechanism/outputs/phase52_full5_seed3/phase57_compressed_representation_mechanism.json`
- `experiments/phase57_compressed_representation_mechanism/outputs/phase52_full5_seed3/phase57_representation_geometry.csv`
- `experiments/phase57_compressed_representation_mechanism/outputs/phase52_full5_seed3/phase57_reward_gain_summary.csv`
- `experiments/phase57_compressed_representation_mechanism/outputs/phase52_full5_seed3/phase57_tile_geometry_gain.csv`
- `experiments/phase57_compressed_representation_mechanism/outputs/phase52_full5_seed3/phase57_compressed_representation_mechanism.md`
