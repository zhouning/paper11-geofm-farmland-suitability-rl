# Phase 9 Proxy-Validation Report Design

## Goal

Add a dedicated Phase 9 pipeline that turns Phase 2 block-level outputs into a
reviewer-facing weak-label validation report for `suitability_proxy`. The phase
should make the proxy evidence easier to inspect before any suitability reward,
policy training, or planning-performance claim is introduced.

## Rationale

Paper11's central methodological risk is over-interpreting AlphaEarth-derived
latent features. Phase 2 already writes a small `weak_label_validation.json`
when weak labels are available, but that artifact is attached to feature
assembly and is too narrow for manuscript review. The experiment plan requires
minimum suitability-model checks before the proxy can be used in a reward.

Phase 9 promotes the existing diagnostic into an explicit reproducibility step:
it reports label availability, suitability distribution, rank alignment, mean
separation, and quantile summaries while preserving a conservative claim
boundary.

## Approach Options

Recommended approach: build a standalone proxy-validation module and CLI that
read `block_geofm_features.csv` from any Phase 2 output directory. This keeps
the report independent of training code, works for fixture data and later real
regions, and avoids mutating Phase 2 artifacts.

Alternative 1: extend `write_phase2_artifacts()` with richer validation. This
would reduce files, but it would keep reviewer-facing validation hidden inside
feature assembly and make it harder to rerun validation against an existing
Phase 2 directory.

Alternative 2: make Phase 9 validate all Phase 8 ablation-control variants.
This would test downstream wiring, but it would not address the main evidence
gap: whether `suitability_proxy` is directionally aligned with available weak
labels.

## Scope

Create a focused validation-report builder:

```text
Phase 2 block_geofm_features.csv
  -> parse requested weak-label columns
  -> compute suitability distribution
  -> compute per-label alignment diagnostics
  -> write phase9_proxy_validation_report.json
```

Default labels are:

```text
stable_farmland_label
high_standard_farmland_label
```

The implementation should accept additional labels via CLI for later datasets,
but the included fixture path should work without extra arguments.

## Report Contents

Write `phase9_proxy_validation_report.json` under the requested output
directory. The report should contain:

- `phase`: `phase9_proxy_validation_report`;
- `phase2_output_dir`;
- `block_table`;
- `label_columns_requested`;
- `label_columns_available`;
- `label_columns_missing`;
- `n_blocks`;
- `suitability_summary` with min, max, mean, standard deviation, and quartiles;
- `labels`, keyed by label column;
- `claim_boundary`.

Each label entry should include:

- `validation_available`;
- `valid_label_count`;
- `missing_label_count`;
- `positive_count`;
- `negative_count`;
- `positive_suitability_mean`;
- `negative_suitability_mean`;
- `mean_difference`;
- `rank_auc`;
- `suitability_quantiles_by_label`;
- `interpretation`.

`interpretation` should be a constrained diagnostic category, not a scientific
claim:

- `positive_alignment`: positives have higher mean suitability and rank AUC is
  at least 0.5;
- `negative_or_no_alignment`: positives do not have higher mean suitability or
  rank AUC is below 0.5;
- `insufficient_label_variation`: only one class is present after parsing;
- `label_unavailable`: the requested column is absent or has no parseable
  binary labels.

## Claim Boundary

Use an explicit Phase 9 claim boundary:

```text
Phase 9 is a weak-label proxy-validation report for suitability_proxy; it does
not prove agronomic validity, train a policy, evaluate a policy, or report
planning performance.
```

The report may support the conservative statement that the proxy is
directionally aligned with available weak labels in the sample. It must not
claim that AlphaEarth directly measures soil quality, fertility, irrigation, or
legal farmland quality.

## Public API

Create `src/paper11_geofm/proxy_validation.py` with:

```python
PHASE9_CLAIM_BOUNDARY = (
    "Phase 9 is a weak-label proxy-validation report for suitability_proxy; "
    "it does not prove agronomic validity, train a policy, evaluate a policy, "
    "or report planning performance."
)


def build_phase9_proxy_validation_report(
    phase2_output_dir: Path | str,
    label_columns: Sequence[str] = (
        "stable_farmland_label",
        "high_standard_farmland_label",
    ),
) -> dict[str, object]:
    """Return the in-memory Phase 9 report dictionary."""


def write_phase9_proxy_validation_report(
    report: Mapping[str, object],
    output_dir: Path | str,
) -> Path:
    """Write phase9_proxy_validation_report.json and return its path."""
```

The builder should load `block_geofm_features.csv`, validate that
`suitability_proxy` exists and is numeric for at least one row, and compute all
statistics from parsed binary labels. Missing labels should be represented in
the report instead of causing the whole phase to fail.

## CLI

Create:

```text
experiments/phase9_proxy_validation/run_phase9_proxy_validation.py
```

The command accepts:

- `--phase2-output-dir`;
- `--output-dir`;
- `--label-columns`, defaulting to
  `stable_farmland_label,high_standard_farmland_label`.

It prints the report path, block count, available labels, per-label rank AUC
and mean difference, and the claim boundary. It returns exit code `1` for a
missing block table or unusable suitability values.

## Documentation

Update:

- `README.md`: add a Phase 9 command after Phase 8;
- `reproducibility/REPRODUCTION_GUIDE.md`: describe the report, expected
  artifact, label limitations, and no-policy-performance boundary;
- `reproducibility/FILE_MANIFEST.tsv`: add the design, plan, module, CLI, and
  test file.

Use `experiments/phase9_proxy_validation/outputs/` for reviewer command
examples because generated experiment outputs are ignored by Git.

## Test Strategy

Use TDD. Add tests that:

- build Phase 2 fixture outputs with the included mapping and attributes CSVs;
- verify the report lists requested, available, and missing labels;
- verify suitability summary includes min, max, mean, standard deviation, and
  quartiles;
- verify per-label positive/negative counts, mean difference, rank AUC, and
  interpretation are computed deterministically;
- verify missing label columns produce `label_unavailable` entries without
  failing the whole report;
- verify a missing block table raises `FileNotFoundError`;
- verify an unusable `suitability_proxy` column raises `ValueError`;
- verify JSON writing creates `phase9_proxy_validation_report.json`;
- verify the CLI prints a concise artifact summary.

All tests remain offline and CPU-only and make no policy-training or
planning-performance claims.
