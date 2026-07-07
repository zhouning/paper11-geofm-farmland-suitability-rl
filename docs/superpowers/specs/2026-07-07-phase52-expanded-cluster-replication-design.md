# Phase 52 Expanded Cluster Replication Design

Phase 52 reuses the completed expanded Phase 28-style six-variant run over
five held-out Bishan tiles and three seeds. It does not add new reward terms or
new policy logic. Its purpose is to test whether the Phase 48 compressed GeoFM
route remains positive when the held-out coverage is expanded from the original
three-tile, three-seed summary to a five-tile, three-seed summary.

The decision route is read-only:

- confirm the expanded output has complete B0, B1, D2, D3, D4P8, and D4P16
  rows for five held-out tiles and seeds 0, 1, and 2;
- run the existing Phase 48 compressed-route analyzer on the expanded summary;
- run the existing Phase 49 row-level robustness analyzer on the expanded
  Phase 48 delta table;
- run the existing Phase 50 tile-seed cluster sign test;
- run the existing Phase 51 exact signed-rank cluster magnitude test.

The supported conclusion is bounded to compressed GeoFM state representation
under the deterministic Bishan base-reward protocol. Phase 52 does not enable
suitability reward, B2/B3 experiments, cross-region transfer claims, or
independent agronomic suitability claims.
