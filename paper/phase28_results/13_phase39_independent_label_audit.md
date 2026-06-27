# Phase 39 Independent-Label Audit

Phase 39 audits whether the current real Bishan Phase 2 feature table contains
or points to any independent, non-leakage labels that could support a later
Phase 38 proxy-rebuild rerun. It is a reviewer-facing label-readiness check,
not a reward or policy-training experiment.

## Labels Audited

The audited default labels are:

- `current_farmland_label`
- `farmland_or_orchard_label`
- `low_slope_farmland_label`
- `source_bsm`
- `source_category`
- `source_dlbm`
- `source_dlmc`

## Status And Counts

Status: `independent_label_inputs_missing`

Row counts from `phase39_independent_label_audit.json`:

- block rows: `64984`
- label inventory rows: `7`
- label readiness rows: `7`
- registry rows: `0`

Interpretation: the available defaults remain DLTB/slope/source-code fields or
source metadata rather than independent agronomic validation labels. Phase 38
cannot yet be rerun with a stronger non-leakage label and B2/B3 remains
blocked.

## Reproduction

Run from the repository root after the Phase 2 real Bishan output exists:

```powershell
python experiments\phase39_independent_label_audit\run_phase39_independent_label_audit.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase39_independent_label_audit\outputs\real_bishan
```

## Artifacts

Expected local artifacts:

- `phase39_label_inventory.csv`
- `phase39_label_readiness.csv`
- `phase39_label_registry_template.csv`
- `phase39_independent_label_audit.json`
- `phase39_independent_label_audit.md`

## Claim Boundary

Phase 39 does not run PPO, alter rewards, enable B2/B3, prove agronomic
validity, or support planning-performance claims. It only records that the
current repository inputs are missing independent label evidence needed for a
stronger non-leakage Phase 38 proxy-rebuild rerun.
