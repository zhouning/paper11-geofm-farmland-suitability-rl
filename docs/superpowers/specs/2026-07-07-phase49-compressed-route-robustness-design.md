# Phase 49 Compressed Route Robustness Design

## Decision

Phase 49 tests whether the Phase 48 compressed GeoFM route remains supported
under simple statistical robustness checks. It is read-only over the Phase 48
delta table and performs no additional policy training.

## Checks

- Pooled one-sided sign test over compressed-versus-control deltas.
- Deterministic bootstrap confidence interval for the pooled mean delta.
- Leave-one evaluation tile sensitivity.
- Leave-one seed sensitivity.
- Per-comparison mean-delta summaries for D4P8/D4P16 against B0/B1/D2/D3.

## Status Rules

- `compressed_route_statistically_robust`: all per-comparison means are
  positive, pooled mean is positive, pooled sign-test p is below alpha,
  bootstrap CI95 low is positive, and all leave-one tile/seed means are
  positive.
- `compressed_route_fragile`: the compressed route has some positive evidence
  but fails at least one robustness check.

## Boundary

Phase 49 strengthens only the base-reward compressed representation claim. It
does not enable suitability reward, B2/B3, transfer, or independent agronomic
suitability claims.
