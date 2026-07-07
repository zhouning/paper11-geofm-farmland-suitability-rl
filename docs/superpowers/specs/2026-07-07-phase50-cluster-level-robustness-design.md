# Phase 50 Cluster-Level Robustness Design

## Decision

Phase 50 aggregates Phase 48 compressed-route deltas to tile-seed clusters
before computing sign-test support. This addresses the reviewer risk that the
72 row-level deltas are not independent observations.

## Status Rules

- `cluster_statistical_support`: cluster mean is positive, more than half of
  clusters are positive, and the cluster-level one-sided sign-test p is below
  alpha.
- `cluster_directional_support`: cluster mean is positive and more than half of
  clusters are positive, but p does not clear alpha.
- `cluster_not_supported`: cluster mean is non-positive or no majority of
  clusters is positive.

## Boundary

Phase 50 is a claim-boundary audit. It may narrow wording even when it does not
overturn the Phase 48/49 representation result.
