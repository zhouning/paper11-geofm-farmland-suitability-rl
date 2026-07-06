# Phase 41 GeoFM Suitability Prior Gate

Phase 41 tests whether GeoFM can be used more safely as a low-dimensional
independent-label-calibrated suitability prior rather than as raw 64-dimensional
policy state.

## Current Real Bishan Run

The current real run used:

```text
experiments/phase11_bishan_dltb_real/outputs/phase2_real
```

No independent label registry was supplied. The run read `64,984` feature rows,
`0` Phase 40 label-gate rows, and `0` Phase 40-passed labels. The current status
is:

```text
phase41_independent_label_inputs_missing
```

## Interpretation

This status does not mean GeoFM is permanently useless. It means Paper11 still
does not have the independent label evidence required to calibrate GeoFM into a
defensible suitability prior. Therefore Phase 41 cannot generate
`block_geofm_suitability_prior.csv` for the current real run.

## Claim Boundary

Phase 41 does not run PPO, alter rewards, enable B2/B3, prove suitability, or
support planning-policy improvement. It only tests whether a Phase 40-passed
independent label allows GeoFM to clear baseline, control, fold-stability, and
calibration checks.

## Reproduction Command

```powershell
python experiments\phase41_geofm_suitability_prior\run_phase41_geofm_suitability_prior.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase41_geofm_suitability_prior\outputs\real_bishan_no_registry
```

## Next Step

If the authors supply an external independent label registry, rerun Phase 40
and then Phase 41. Only a real `geofm_suitability_prior_supported` Phase 41
result should authorize a later bounded low-dimensional prior experiment.
