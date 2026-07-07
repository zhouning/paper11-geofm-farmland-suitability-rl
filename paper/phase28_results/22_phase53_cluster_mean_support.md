# Phase 53 Cluster Mean Support

Phase 53 audits the expanded Phase 52 compressed-route cluster deltas to answer
one reviewer-facing question: whether the positive cluster-magnitude result is
driven only by one or two large positive tile-seed clusters.

The audit is read-only. It consumes the Phase 52/Phase 50 tile-seed cluster
summary:

```text
experiments/phase52_expanded_cluster_replication/outputs/phase52_full5_seed3_phase50_cluster/phase50_cluster_delta_summary.csv
```

It does not run policy training, alter rewards, enable suitability reward, test
`B2/B3`, test cross-region transfer, or validate independent agronomic
suitability.

## Cluster Mean Audit

The Phase 53 runner was executed as:

```powershell
python experiments\phase53_cluster_mean_support\run_phase53_cluster_mean_support.py --phase50-cluster-csv experiments\phase52_expanded_cluster_replication\outputs\phase52_full5_seed3_phase50_cluster\phase50_cluster_delta_summary.csv --output-dir experiments\phase53_cluster_mean_support\outputs\phase52_full5_seed3 --bootstrap-iterations 20000 --random-seed 53
```

Status:

```text
cluster_mean_support
```

Main values:

| Quantity | Value |
|---|---:|
| Tile-seed clusters | `15` |
| Mean cluster delta | `0.2921767818` |
| Exact one-sided sign-flip mean p | `0.0196838379` |
| Bootstrap CI95 | `[0.0570820445, 0.5823557658]` |

The exact sign-flip test uses the observed cluster-delta magnitudes and tests
whether random signs would produce a mean at least as large as the observed
positive mean. The bootstrap interval is computed over cluster means with
`20000` resamples and random seed `53`.

## Influence Checks

All leave-one summaries remain positive:

| Influence check | Minimum retained mean |
|---|---:|
| Leave-one-cluster | `0.2060081575` |
| Leave-one-tile | `0.0954244478` |
| Leave-one-seed | `0.2083797951` |

The weakest leave-one-tile case is still positive after removing all three
clusters from `tile_r005_c003`. This matters because it reduces the risk that
the Phase 52 cluster-magnitude result is an artifact of a single favorable tile,
seed, or tile-seed cluster.

## Interpretation

Phase 53 strengthens the compressed GeoFM route. The expanded Phase 52 evidence
already showed positive compressed-versus-control mean deltas, row-level
sign-test support, and exact signed-rank cluster-magnitude support. Phase 53 now
adds direct cluster-mean support: the cluster mean is positive, the exact
sign-flip mean test clears alpha `0.05`, the bootstrap lower bound remains
above zero, and all leave-one cluster/tile/seed means remain positive.

The correct conclusion remains bounded. The evidence supports compressed GeoFM
state representations under the current Bishan base-reward held-out protocol.
It does not support raw 64-dimensional `B1` direct injection, suitability
reward, `B2/B3`, cross-region transfer, or independently validated agronomic
suitability.
