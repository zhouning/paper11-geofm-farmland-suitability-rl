# Phase 51 Cluster Magnitude Support Audit

## Purpose

Phase 51 follows the conservative Phase 50 tile-seed cluster audit. Phase 50
showed `7 / 9` positive clusters but did not clear the sign-test threshold
because the sign test ignores effect magnitude. Phase 51 applies an exact
one-sided signed-rank test to the same cluster mean deltas.

## Real Bishan Result

Command:

```powershell
python experiments\phase51_cluster_magnitude_support\run_phase51_cluster_magnitude_support.py --phase50-cluster-csv experiments\phase50_cluster_level_robustness\outputs\real_bishan_4096\phase50_cluster_delta_summary.csv --output-dir experiments\phase51_cluster_magnitude_support\outputs\real_bishan_4096
```

Status: `cluster_magnitude_support`

| Metric | Value |
| --- | ---: |
| Cluster count | `9` |
| Positive rank sum | `40` |
| Total rank sum | `45` |
| Exact one-sided signed-rank p | `0.01953125` |

## Conclusion

Phase 51 strengthens the cluster-level interpretation. Phase 50 remains useful
as a conservative sign-only boundary, but the magnitude-sensitive exact
signed-rank test supports the compressed GeoFM route at the tile-seed cluster
level.

The correct manuscript wording is: the compressed GeoFM route is supported on
mean reward, row-level robustness checks, and exact cluster-level signed-rank
evidence, while the sign-only cluster test is directional but underpowered.

## Artifacts

- `experiments/phase51_cluster_magnitude_support/outputs/real_bishan_4096/phase51_cluster_magnitude_support.json`
- `experiments/phase51_cluster_magnitude_support/outputs/real_bishan_4096/phase51_cluster_signed_rank.csv`
- `experiments/phase51_cluster_magnitude_support/outputs/real_bishan_4096/phase51_cluster_magnitude_support.md`
