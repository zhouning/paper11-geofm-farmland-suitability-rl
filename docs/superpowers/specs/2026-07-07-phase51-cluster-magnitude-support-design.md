# Phase 51 Cluster Magnitude Support Design

Phase 51 applies an exact one-sided signed-rank test to Phase 50 tile-seed
cluster mean deltas. It addresses the limitation that the Phase 50 sign test
uses only signs and ignores effect magnitude.

Status `cluster_magnitude_support` requires positive mean cluster delta and
signed-rank p below alpha.
