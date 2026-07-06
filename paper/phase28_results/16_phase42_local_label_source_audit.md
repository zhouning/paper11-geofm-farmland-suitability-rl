# Phase 42 Local Label Source Audit

Phase 42 records a local search for candidate independent labels after Phase 41
introduced the calibrated GeoFM suitability-prior route. The purpose is to
separate three cases that otherwise look similar in file searches:

1. true external independent suitability labels;
2. local DLTB/slope-derived weak labels that are useful only for diagnostics;
3. unrelated labels from other projects or regions.

## Search Scope

The audit inspected the Paper11 repository and relevant local `D:\test` data
locations for possible soil, irrigation, yield, high-standard-farmland,
productivity, farmland-quality, suitability, label, and registry files.

Important local candidates included:

- `D:\test\dem_slope_analysis\output\DLTB_with_slope.gpkg`
- `D:\test\paper11_macos_transfer\DLTB_with_slope.gpkg`
- `D:\test\dem_slope_analysis\output\slope_statistics_summary.csv`
- `D:\test\dem_slope_analysis\intermediate\parcels_attributes.csv`
- `D:\test\现状用地数据\GDB.gdb`
- Paper58 `data\independent_change_labels\*`
- Paper10 value-label outputs

## Candidate Assessment

The DLTB/slope files contain columns such as `BSM`, `DLBM`, `DLMC`, `category`,
`slope_mean`, `slope_max`, and `slope_pixel_count`. These are already part of,
or directly derived from, the explicit planning-feature path used by Paper11.
They are therefore not independent suitability labels.

The Paper58 files are independent change/LULC benchmark labels for other areas
and another paper's world-model task. They are not Bishan block-level farmland
suitability labels and cannot be registered for Paper11 Phase 40.

The Paper10 value-label files are model-generated or experiment-derived value
labels for another project. They are not external agronomic, soil, irrigation,
yield, high-standard-farmland, or field-survey labels for Paper11 Bishan blocks.

## Diagnostic Registry Check

To confirm that the available Paper11 weak labels are not merely missing from a
registry, Phase 42 created a diagnostic registry containing:

- `current_farmland_label`
- `farmland_or_orchard_label`
- `low_slope_farmland_label`

The registry explicitly marked the labels as `dltb_derived` or `slope_derived`
with `leakage_risk` independence.

Phase 40 result with this diagnostic registry:

```text
independent_label_gate_diagnostic_only
```

Phase 41 result with the same diagnostic registry:

```text
phase41_independent_label_inputs_missing
```

## Interpretation

The local files do not contain a Phase 40-passing independent non-leakage label.
The DLTB/slope labels are computable, but they remain diagnostic-only because
they are source-derived and overlap with explicit planning features. Phase 41
therefore still cannot generate `block_geofm_suitability_prior.csv`.

## Claim Boundary

Phase 42 does not run PPO, alter rewards, enable B2/B3, validate suitability,
or support planning-policy improvement. It only records that the local file
system search and diagnostic registry check did not uncover a usable independent
label source.

## Required External Label Source

The next valid input must be an external label joined to Paper11 blocks or
spatially joinable to the real Bishan DLTB units, such as:

- field-survey farmland quality;
- soil fertility or soil-quality class;
- irrigation or water-access class not derived from the current explicit path;
- yield or productivity observation;
- high-standard-farmland designation from an external source;
- retention or policy outcome label not derived from DLTB/slope/source metadata.

Only after such a label passes Phase 40 should Phase 41 be rerun as a real
GeoFM suitability-prior test.
