# Phase 49 Compressed Route Robustness Audit

## Purpose

Phase 49 tests whether the Phase 48 compressed GeoFM route is only a positive
mean effect or whether it survives basic statistical robustness checks. It is a
read-only audit over the Phase 48 delta table and does not retrain policies.

The audit checks pooled sign-test support, bootstrap confidence intervals for
the pooled mean delta, and leave-one-tile/leave-one-seed sensitivity.

## Real Bishan Result

Command:

```powershell
python experiments\phase49_compressed_route_robustness\run_phase49_compressed_route_robustness.py --phase48-delta-csv experiments\phase48_compressed_geofm_rescue\outputs\real_bishan_4096\phase48_compressed_geofm_rescue_delta_table.csv --output-dir experiments\phase49_compressed_route_robustness\outputs\real_bishan_4096 --bootstrap-iterations 10000 --random-seed 49
```

Status: `compressed_route_statistically_robust`

Pooled compressed-control result:

| Metric | Value |
| --- | ---: |
| Mean delta | `0.4673011499` |
| Positive comparisons | `48 / 72` |
| Positive fraction | `0.6666666667` |
| One-sided sign-test p | `0.0031549137` |
| Bootstrap CI95 low | `0.2827829983` |
| Bootstrap CI95 high | `0.6639974489` |

Leave-one sensitivity:

| Check | Group count | Minimum leave-one mean delta | All positive |
| --- | ---: | ---: | --- |
| Leave-one tile | `3` | `0.1613586660` | true |
| Leave-one seed | `3` | `0.3644002401` | true |

## Conclusion

Phase 49 strengthens the Phase 48 conclusion. The compressed GeoFM route is not
only positive on average; it remains positive under pooled sign-test,
bootstrap, and leave-one tile/seed sensitivity checks.

The manuscript can now state that the compressed GeoFM base-reward route is
statistically robust within the current Bishan held-out protocol. The boundary
does not change: Phase 49 does not enable suitability reward, does not test
`B2`/`B3`, does not test transfer, and does not validate independent agronomic
suitability.

## Artifacts

- `experiments/phase49_compressed_route_robustness/outputs/real_bishan_4096/phase49_compressed_route_robustness.json`
- `experiments/phase49_compressed_route_robustness/outputs/real_bishan_4096/phase49_per_comparison_robustness.csv`
- `experiments/phase49_compressed_route_robustness/outputs/real_bishan_4096/phase49_leave_one_sensitivity.csv`
- `experiments/phase49_compressed_route_robustness/outputs/real_bishan_4096/phase49_compressed_route_robustness.md`
