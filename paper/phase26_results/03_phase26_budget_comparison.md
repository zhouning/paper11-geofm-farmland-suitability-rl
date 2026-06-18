# Phase 26 Budget Comparison

## One-Sentence Argument

Comparing the current 1024-step and 4096-step Phase 26 result sets shows that
longer training moves the learned-policy B1-B0 mean delta upward, but it still
does not stabilize the learned policy enough to support a positive
B1-over-B0 claim.

## Terminology Ledger

| Canonical term | Definition | Boundary |
|---|---|---|
| 1024-step result set | the shorter current macOS Phase 26 result set | learned-policy diagnosis only |
| 4096-step result set | the longer current macOS Phase 26 result set | learned-policy diagnosis only |
| learned-policy mean delta | B1 minus B0 mean reward over paired tile-seed rows | not a transfer or suitability metric |
| positive tile-seed count | count of tile-seed pairs where B1 > B0 | stability diagnostic only |
| claim status | conservative support label for the current evidence | not a manuscript-ready claim |

## What Was Compared

| Budget | Learned-policy mean delta | Positive tile-seed count | Claim status |
|---|---|---|---|
| 1024 steps | `-0.4329022862` | `4 / 9` | `not_supported` |
| 4096 steps | `-0.1318712688` | `3 / 9` | `not_supported` |

The 4096-step result is less negative by `0.3010310174`, but the positive
tile-seed count falls from `4 / 9` to `3 / 9`. That combination matters more
than the mean alone: the budget increase improved the average gap, but it did
not make the B1 advantage more stable.

## Result Interpretation

The current evidence suggests budget sensitivity, not convergence. A larger
budget moved some tile-seed outcomes toward B1, but the overall learned-policy
signal remained negative and the positive cases did not dominate.

That means the current Paper11 conclusion stays bounded:

- B1 is not yet reliably better than B0 under the deterministic base planning
  reward.
- The current evidence does not justify a positive learned-policy claim.
- Budget alone is not enough to resolve the representation question.

## What This Supports

This comparison supports a practical next step:

1. run one more stability sweep with an intermediate or repeated budget;
2. separate budget effects from tile-seed noise;
3. only then decide whether to continue extending the learned-policy route or
   shift emphasis to proxy validation and ablations.

## What This Does Not Support

This comparison does not support:

- a positive B1-over-B0 claim;
- suitability reward readiness;
- B2/B3 superiority;
- cross-region transfer;
- manuscript-level performance claims.

## Manuscript Use

This should be cited as a diagnosis note only. It is useful for explaining why
the current Paper11 evidence remains incomplete, but it is not a results
section figure or a submission-ready performance summary.
