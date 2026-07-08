# Phase 62 D4/D6 Matched PPO Evaluation

Phase 62 is the bounded learned-policy follow-up to Phase 61. It compares the
existing D4 PCA-compressed controls against D6 raw-B1 random orthonormal
projection controls under the same Phase 52 five-tile, three-seed Bishan
base-reward held-out PPO protocol.

## Status

`d6_random_projection_advantage`

## Primary Comparisons

| Comparison | Mean delta | Positive rows |
|---|---:|---:|
| D4P8 - D6R8 | -0.0279096981 | 5 / 15 |
| D4P16 - D6R16 | -0.1004704046 | 1 / 15 |

## Pooled and Cluster Evidence

- Pooled primary mean delta: -0.0641900514
- Pooled positive rows: 6 / 30
- Pooled bootstrap CI95: [-0.1459486743, 0.01836613]
- Pooled one-sided sign-test p: 0.9998375429
- Cluster mean delta: -0.0641900514
- Positive tile-seed clusters: 5 / 15
- Cluster sign-test p: 0.9407653809
- Signed-rank positive rank sum: 34 / 105
- Signed-rank p: 0.8793945312
- Coverage issues: missing 0, duplicate 0, unexpected 0

## Interpretation

Phase 62 does not support a PCA-specific advantage for D4P8/D4P16 over raw-B1
random orthonormal projections. Both primary mean deltas are negative, the
pooled primary mean is negative, and the cluster summaries are also mean-negative
under complete five-tile, three-seed coverage.

This result is consistent with the narrowed Phase 60 mechanism boundary rather
than a reversal of it. D4P8/D4P16 remain part of the earlier positive compressed
route evidence against B0, raw B1, random D2, and shuffled D3, but Phase 59 and
Phase 62 now jointly prevent a stronger claim that the current PCA-compressed
GeoFM coordinates uniquely outperform matched low-dimensional controls. The
current defensible algorithm claim remains a bounded low-dimensional compressed
state route under the Bishan base-reward protocol, not PCA optimality and not a
submission-level planning-performance result.

## Reproduction

Run Phase 62 from the repository root after the Phase 8 D4 features, Phase 61
D6 projection-control features, and Phase 13 tile index exist:

```powershell
python experiments/phase62_d4_d6_matched_ppo/run_phase62_d4_d6_matched_ppo.py --mode run-and-analyze --phase8-output-dir experiments/phase8_ablation_controls/outputs/real_bishan_controls --phase61-output-dir experiments/phase61_d6_geofm_projection_controls/outputs/phase52_full5_seed3 --tile-index-csv experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv --variants D4P8,D4P16,D6R8,D6R16 --eval-tile-ids tile_r002_c003,tile_r005_c004,tile_r005_c003,tile_r000_c004,tile_r001_c004 --total-timesteps 4096 --eval-max-steps 8 --seeds 0,1,2 --bootstrap-iterations 5000 --seed 62 --output-dir experiments/phase62_d4_d6_matched_ppo/outputs/phase52_full5_seed3
```

Expected status for the current run: `d6_random_projection_advantage`.

## Boundary

Phase 62 tests base-reward learned-policy differences between D4 PCA compressed
states and D6 raw-B1 random projection controls. It does not enable suitability
reward, does not test B2/B3, does not test cross-region transfer, does not prove
independent agronomic suitability, and does not justify formal submission-level
performance claims. No formal manuscript files were changed in this phase.

## Artifacts

- `experiments/phase62_d4_d6_matched_ppo/outputs/phase52_full5_seed3/phase62_d4_d6_matched_ppo.json`
- `experiments/phase62_d4_d6_matched_ppo/outputs/phase52_full5_seed3/phase62_d4_d6_matched_ppo_summary.csv`
- `experiments/phase62_d4_d6_matched_ppo/outputs/phase52_full5_seed3/phase62_d4_d6_matched_ppo_traces.json`
- `experiments/phase62_d4_d6_matched_ppo/outputs/phase52_full5_seed3/phase62_d4_d6_delta_table.csv`
- `experiments/phase62_d4_d6_matched_ppo/outputs/phase52_full5_seed3/phase62_d4_d6_cluster_summary.csv`
- `experiments/phase62_d4_d6_matched_ppo/outputs/phase52_full5_seed3/phase62_d4_d6_matched_ppo.md`
