# Phase 54 Artifact Lineage Consistency Design

## Purpose

Phase 54 adds a read-only audit for the Phase 52/53 compressed GeoFM evidence
chain. The goal is to prove that the formal manuscript values come from one
consistent artifact lineage rather than from mixed Phase 52 output directories.

## Authoritative Inputs

Phase 54 treats these files as the authoritative formal-evidence chain:

- `experiments/phase52_expanded_cluster_replication/outputs/phase52_full5_seed3_phase48_compressed_rescue/phase48_compressed_geofm_rescue_delta_table.csv`
- `experiments/phase52_expanded_cluster_replication/outputs/phase52_full5_seed3_phase50_cluster/phase50_cluster_delta_summary.csv`
- `experiments/phase52_expanded_cluster_replication/outputs/phase52_full5_seed3_phase51_magnitude/phase51_cluster_magnitude_support.json`
- `experiments/phase53_cluster_mean_support/outputs/phase52_full5_seed3/phase53_cluster_mean_support.json`

The older `real_bishan_4096_5tiles*` directories are not used as formal
manuscript evidence sources in Phase 54. They may remain as historical generated
outputs, but manuscript claims should point to the authoritative chain above.

## Audit Requirements

1. Recompute Phase 50 tile-seed cluster means from the authoritative Phase 48
   delta table and compare them against the authoritative Phase 50 cluster CSV.
2. Recompute Phase 51 signed-rank values from the authoritative cluster CSV and
   compare them against the authoritative Phase 51 JSON.
3. Recompute Phase 53 cluster-mean support values from the authoritative cluster
   CSV and compare them against the authoritative Phase 53 JSON.
4. Report a lineage status:
   - `artifact_lineage_consistent` only when all three checks match.
   - `artifact_lineage_inconsistent` when any check differs beyond tolerance.
5. Keep the claim boundary unchanged: this audit does not enable suitability
   reward, does not test `B2/B3`, does not test transfer, and does not validate
   independent agronomic suitability.

## Outputs

The runner writes:

- `phase54_artifact_lineage_consistency.json`
- `phase54_artifact_lineage_checks.csv`
- `phase54_artifact_lineage_consistency.md`

## Manuscript Use

If Phase 54 reports `artifact_lineage_consistent`, the formal manuscript can
state that the Phase 52/53 compressed-route evidence chain is internally
reproducible from the authoritative delta table through cluster aggregation,
signed-rank testing, and cluster-mean support. It should not claim broader tile
generalization, suitability reward, or transfer.
