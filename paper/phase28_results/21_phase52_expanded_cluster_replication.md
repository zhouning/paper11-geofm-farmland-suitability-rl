# Phase 52 Expanded Cluster Replication

Phase 52 records the expanded compressed-route replication after the complete
Phase 28-style six-variant run over five held-out Bishan tiles and three seeds.
The run evaluates `B0`, raw `B1`, random `D2`, shuffled `D3`, compressed
`D4P8`, and compressed `D4P16` at `4096` training steps and evaluation horizon
`8`.

The expanded run is treated as replication evidence for the compressed state
route. It does not enable suitability reward, does not test `B2` or `B3`, does
not test transfer, and does not validate independent agronomic suitability.

## Expanded Phase 48-Style Result

The existing Phase 48 read-only analyzer was run on:

```text
experiments/phase52_expanded_cluster_replication/outputs/real_bishan_4096_5tiles/phase28_representation_control_summary.csv
```

Status:

```text
compressed_geofm_route_supported
```

Mean learned-policy reward by variant:

| Variant | Mean reward |
|---|---:|
| `B0` | `0.1793245179` |
| `B1` | `0.2639655302` |
| `D2` | `0.2183220949` |
| `D3` | `0.2716306377` |
| `D4P8` | `0.4690087215` |
| `D4P16` | `0.5819662325` |

Compressed candidate deltas:

| Comparison | Compressed minus comparator mean delta | Positive tile-seed count |
|---|---:|---:|
| `D4P8 - B0` | `0.2896842037` | `9 / 15` |
| `D4P8 - B1` | `0.2050431914` | `9 / 15` |
| `D4P8 - D2` | `0.2506866266` | `10 / 15` |
| `D4P8 - D3` | `0.1973780839` | `10 / 15` |
| `D4P16 - B0` | `0.4026417146` | `8 / 15` |
| `D4P16 - B1` | `0.3180007023` | `10 / 15` |
| `D4P16 - D2` | `0.3636441375` | `7 / 15` |
| `D4P16 - D3` | `0.3103355948` | `11 / 15` |

All eight compressed-versus-control mean deltas are positive. The pooled
compressed-control delta is `0.2921767818`, with `74 / 120` positive row-level
comparisons.

## Robustness

The Phase 49 robustness analyzer reports:

```text
compressed_route_statistically_robust
```

Pooled robustness summary:

```text
mean delta: 0.2921767818
positive comparisons: 74 / 120
one-sided sign-test p: 0.0066881634
bootstrap CI95: [0.1623326461, 0.4323997354]
```

Leave-one sensitivity remains positive for every held-out tile and every held-
out seed. The smallest leave-one-tile mean is `0.0954244478`, and the smallest
leave-one-seed mean is `0.2083797951`.

## Cluster Boundary

The Phase 50 cluster sign-only analyzer aggregates the same `120` row-level
comparisons into `15` tile-seed clusters:

```text
cluster_directional_support
mean cluster delta: 0.2921767818
positive clusters: 10 / 15
one-sided sign-test p: 0.1508789062
```

This is a wording boundary. The expanded cluster sign-only test remains
directional rather than alpha-0.05 significant.

The Phase 51 magnitude-sensitive exact signed-rank test over the same `15`
cluster mean deltas reports:

```text
cluster_magnitude_support
positive rank sum: 96
total rank sum: 120
one-sided signed-rank p: 0.0206298828
```

## Interpretation

Phase 52 strengthens the Phase 48-51 conclusion. The compressed GeoFM route is
not a one-off result from the original three held-out tiles: after expanding
the same six-variant protocol to five held-out tiles and three seeds, `D4P8`
and `D4P16` still outperform `B0`, raw `B1`, random `D2`, and shuffled `D3` on
mean reward.

The correct conclusion remains bounded. The evidence supports compressed GeoFM
state representations under the current Bishan base-reward held-out protocol.
It does not support raw 64-dimensional `B1` direct injection, suitability
reward, `B2/B3`, cross-region transfer, or independently validated agronomic
suitability.
