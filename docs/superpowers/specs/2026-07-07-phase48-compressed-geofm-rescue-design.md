# Phase 48 Compressed GeoFM Rescue Design

## Decision

Phase 48 reclassifies `D4P8` and `D4P16` from secondary representation controls into explicit compressed GeoFM candidate routes. The audit is read-only over the existing Phase 28 held-out Bishan summary rows.

## Scope

The audit answers one question: whether compressed GeoFM candidates outperform `B0`, raw `B1`, random control `D2`, and shuffled control `D3` under the same base-reward held-out policy protocol.

The audit does not enable suitability reward, does not run `B2`/`B3`, does not add independent suitability labels, and does not support cross-region or submission-level planning-performance claims.

## Status Rules

- `compressed_geofm_route_supported`: both `D4P8` and `D4P16` have positive mean reward deltas against all four comparators, and the pooled compressed-control delta has positive mean and at least half of tile-seed comparisons positive.
- `compressed_geofm_route_partial`: at least one compressed candidate recovers the `B0` and raw `B1` gap, but random or shuffled controls are not cleared.
- `compressed_geofm_route_not_supported`: compressed candidates do not recover the `B0`/raw-`B1` gap.
- `insufficient`: required `B0`, `B1`, `D2`, `D3`, `D4P8`, or `D4P16` tile-seed rows are missing or duplicated.

## Expected Output

Phase 48 writes a summary CSV copy, a comparison JSON, a compressed delta CSV, and a reviewer-facing Markdown readiness file. The manuscript conclusion should change only if the real audit status is supported or partial.
