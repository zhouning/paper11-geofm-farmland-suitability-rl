# Phase 48 Compressed GeoFM Rescue Audit

## Purpose

Phase 48 tests whether the earlier negative raw-B1 result came from the GeoFM information itself or from the way the information entered the policy state. It re-evaluates `D4P8` and `D4P16` as compressed GeoFM candidate routes rather than treating them only as representation controls.

The audit is read-only over the existing Phase 28 `4096`-step held-out Bishan summary rows. It does not retrain policies, does not enable suitability reward, does not test `B2`/`B3`, and does not create cross-region evidence.

## Real Bishan Result

Command:

```powershell
python experiments\phase48_compressed_geofm_rescue\run_phase48_compressed_geofm_rescue.py --existing-summary-csv experiments\phase28_representation_controls\outputs\real_bishan_4096\phase28_representation_control_summary.csv --output-dir experiments\phase48_compressed_geofm_rescue\outputs\real_bishan_4096
```

Status: `compressed_geofm_route_supported`

Mean learned-policy reward by variant:

| Variant | Mean reward |
| --- | ---: |
| `B0` | `0.4825072170` |
| `B1` | `0.3506359482` |
| `D2` | `0.2761767826` |
| `D3` | `0.4601109618` |
| `D4P8` | `0.7274877829` |
| `D4P16` | `0.9918299718` |

Compressed-candidate deltas:

| Comparison | Mean delta | Positive tile-seed pairs |
| --- | ---: | ---: |
| `D4P8 - B0` | `0.2449805659` | `4 / 9` |
| `D4P8 - B1` | `0.3768518347` | `7 / 9` |
| `D4P8 - D2` | `0.4513110003` | `7 / 9` |
| `D4P8 - D3` | `0.2673768211` | `6 / 9` |
| `D4P16 - B0` | `0.5093227548` | `5 / 9` |
| `D4P16 - B1` | `0.6411940236` | `7 / 9` |
| `D4P16 - D2` | `0.7156531892` | `5 / 9` |
| `D4P16 - D3` | `0.5317190100` | `7 / 9` |

Pooled compressed-control delta: mean `0.4673011499`, `48 / 72` positive comparisons, positive fraction `0.6666666667`.

## Conclusion

Phase 48 changes the Paper11 conclusion. The evidence no longer supports a broad statement that GeoFM fails in Paper11. It supports a narrower, more useful conclusion:

- Raw 64-dimensional GeoFM direct state injection (`B1`) remains unsupported.
- Compressed GeoFM state routes (`D4P8`, `D4P16`) are supported as candidate base-reward representation routes under the current held-out Bishan protocol.
- The suitability-reward route still remains blocked because Phase 40/41 have not found or validated independent labels.

The current defensible manuscript framing is therefore positive but bounded: GeoFM can help when represented through compressed state inputs under the existing base-reward planning protocol, while raw injection and suitability-reward claims remain unsupported.

## Artifacts

- `experiments/phase48_compressed_geofm_rescue/outputs/real_bishan_4096/phase48_compressed_geofm_rescue_summary.csv`
- `experiments/phase48_compressed_geofm_rescue/outputs/real_bishan_4096/phase48_compressed_geofm_rescue_comparison.json`
- `experiments/phase48_compressed_geofm_rescue/outputs/real_bishan_4096/phase48_compressed_geofm_rescue_delta_table.csv`
- `experiments/phase48_compressed_geofm_rescue/outputs/real_bishan_4096/phase48_compressed_geofm_rescue_readiness.md`
