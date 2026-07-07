# Phase 54 Artifact Lineage Consistency

Phase 54 audits the formal Phase 52/53 compressed GeoFM evidence chain. It was
added because multiple generated Phase 52 output directories exist locally, and
the formal manuscript must be traceable to one authoritative artifact lineage.

The audit is read-only. It does not run policy training, alter rewards, enable
suitability reward, test `B2/B3`, test cross-region transfer, or validate
independent agronomic suitability.

## Authoritative Artifact Chain

Phase 54 treats the following files as the formal evidence chain:

```text
experiments/phase52_expanded_cluster_replication/outputs/phase52_full5_seed3_phase48_compressed_rescue/phase48_compressed_geofm_rescue_delta_table.csv
experiments/phase52_expanded_cluster_replication/outputs/phase52_full5_seed3_phase50_cluster/phase50_cluster_delta_summary.csv
experiments/phase52_expanded_cluster_replication/outputs/phase52_full5_seed3_phase51_magnitude/phase51_cluster_magnitude_support.json
experiments/phase53_cluster_mean_support/outputs/phase52_full5_seed3/phase53_cluster_mean_support.json
```

Historical `real_bishan_4096_5tiles*` generated directories are not used as the
formal manuscript evidence source for the Phase 52/53 conclusion.

## Real Audit

The runner was executed as:

```powershell
python experiments\phase54_artifact_lineage_consistency\run_phase54_artifact_lineage_consistency.py --phase48-delta-csv experiments\phase52_expanded_cluster_replication\outputs\phase52_full5_seed3_phase48_compressed_rescue\phase48_compressed_geofm_rescue_delta_table.csv --phase50-cluster-csv experiments\phase52_expanded_cluster_replication\outputs\phase52_full5_seed3_phase50_cluster\phase50_cluster_delta_summary.csv --phase51-json experiments\phase52_expanded_cluster_replication\outputs\phase52_full5_seed3_phase51_magnitude\phase51_cluster_magnitude_support.json --phase53-json experiments\phase53_cluster_mean_support\outputs\phase52_full5_seed3\phase53_cluster_mean_support.json --output-dir experiments\phase54_artifact_lineage_consistency\outputs\phase52_full5_seed3
```

Status:

```text
artifact_lineage_consistent
```

Core recomputed values:

| Quantity | Value |
|---|---:|
| Cluster count | `15` |
| Mean cluster delta | `0.2921767818` |
| Phase 51 signed-rank p | `0.0206298828` |
| Phase 53 sign-flip mean p | `0.0196838379` |

All artifact-lineage checks passed. Phase 54 recomputed the Phase 50 cluster
means from the authoritative Phase 48 delta table, recomputed the Phase 51
signed-rank result from the authoritative cluster CSV, and recomputed the Phase
53 cluster-mean support values from the same cluster CSV.

## Interpretation

Phase 54 strengthens the formal submission package by closing an artifact-
lineage risk. It shows that the manuscript's Phase 52/53 compressed-route values
are internally reproducible from one authoritative chain: delta rows to cluster
means to signed-rank testing to cluster-mean support.

The scientific conclusion remains bounded. The evidence supports compressed
GeoFM state representations under the current Bishan base-reward held-out
protocol. It does not support raw 64-dimensional `B1` direct injection,
suitability reward, `B2/B3`, cross-region transfer, or independently validated
agronomic suitability.
