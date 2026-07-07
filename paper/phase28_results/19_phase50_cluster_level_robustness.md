# Phase 50 Cluster-Level Robustness Audit

## Purpose

Phase 50 addresses a statistical independence risk in Phase 49. The Phase 49
pooled audit treats 72 compressed-versus-control deltas as comparison rows, but
those rows share the same held-out tile and seed contexts. Phase 50 aggregates
the deltas to tile-seed clusters before applying the sign test.

## Real Bishan Result

Command:

```powershell
python experiments\phase50_cluster_level_robustness\run_phase50_cluster_level_robustness.py --phase48-delta-csv experiments\phase48_compressed_geofm_rescue\outputs\real_bishan_4096\phase48_compressed_geofm_rescue_delta_table.csv --output-dir experiments\phase50_cluster_level_robustness\outputs\real_bishan_4096
```

Status: `cluster_directional_support`

Cluster summary:

| Metric | Value |
| --- | ---: |
| Tile-seed clusters | `9` |
| Mean cluster delta | `0.4673011499` |
| Positive clusters | `7 / 9` |
| One-sided cluster sign-test p | `0.0898437500` |

## Conclusion

Phase 50 does not overturn Phase 48/49. The cluster-level mean remains positive
and `7 / 9` tile-seed clusters favor the compressed route. However, after
aggregating to the more conservative tile-seed unit, the one-sided sign test
does not clear alpha `0.05`.

The correct manuscript wording is therefore:

- compressed GeoFM state routes are supported on mean reward and row-level
  robustness checks;
- tile-seed cluster aggregation remains directionally positive but
  underpowered at `n = 9`;
- no claim should state that the cluster-level sign test is significant at
  alpha `0.05`;
- raw B1 and suitability reward remain unsupported.

## Artifacts

- `experiments/phase50_cluster_level_robustness/outputs/real_bishan_4096/phase50_cluster_level_robustness.json`
- `experiments/phase50_cluster_level_robustness/outputs/real_bishan_4096/phase50_cluster_delta_summary.csv`
- `experiments/phase50_cluster_level_robustness/outputs/real_bishan_4096/phase50_cluster_level_robustness.md`
