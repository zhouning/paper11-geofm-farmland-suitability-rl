# Phase 53 Cluster Mean Support Design

Phase 53 tests whether the expanded Phase 52 compressed-route cluster effect is
driven by a stable positive cluster mean rather than by the sign-only cluster
count. It consumes the Phase 50 tile-seed cluster CSV produced from the Phase
52 expanded run.

The audit is read-only and computes:

- exact one-sided sign-flip p value over cluster mean magnitudes;
- nonparametric bootstrap CI for the cluster mean;
- leave-one-cluster, leave-one-tile, and leave-one-seed influence summaries.

Status `cluster_mean_support` requires a positive cluster mean, exact sign-flip
p below alpha, positive bootstrap lower bound, and positive leave-one means.

The claim remains bounded to compressed GeoFM state representation under the
Bishan base-reward held-out protocol. It does not enable suitability reward,
B2/B3, transfer, or independent agronomic suitability claims.
