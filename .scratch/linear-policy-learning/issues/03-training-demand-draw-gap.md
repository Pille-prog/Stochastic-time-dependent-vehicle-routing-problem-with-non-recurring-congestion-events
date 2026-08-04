# 03 — The training demand draw is further apart than the evaluation one

**Status:** open, unclaimed

## What was noticed

Ticket 01 captured 200 training episodes per side and recorded the demand each
drew:

| | mean Clients | mean vehicles |
| --- | --- | --- |
| legacy, seeds 1000-1199 | 146.28 | 5.70 |
| repo, seeds 1000-1199 | 153.38 | 5.96 |

Both generators are `N(150, 30)` floored at 60, with vehicles
`ceil(clients / 28)`. Ticket 13 (ADR-0001 phase 2) replaced the legacy's shared
global Mersenne streams with per-Episode `SeedSequence` spawning, so seed 1000
draws a different realisation on each side *by design* — different values are
expected, and this is not on its own a bug.

What makes it worth a look is the size. The standard error of a 200-sample mean
at σ = 30 is ≈ 2.1, so a 7.1-Client gap is ≈ 3.4σ. `spec.md` records the
evaluation-seed comparison as much closer (148.94 vs 147.18 over 50 seeds), so
the two measurements disagree about how well the generators line up.

## Why it might matter

Ticket 01's finding is that our episodes cost 1.68x the legacy's during training.
5% more Clients per episode with the same vehicle ratio is a real difficulty
difference and would account for part of that, which changes how much of the
1.68x needs another explanation. It is worth pricing before ticket 02 attributes
the whole gap to conditioning.

It also cuts the other way as a check on the port: if the two generators are not
drawing from the same distribution, `spec.md`'s "the demand matches" claim —
which several of its comparisons lean on — needs restating.

## Done when

- [ ] Both generators sampled over the *same* large seed range (thousands, not
      200) and their Client-count distributions compared — mean, σ, and the
      floor-at-60 tail, not just the mean
- [ ] A verdict: same distribution sampled differently (in which case ticket 01's
      gap is sampling noise and the note comes out of `spec.md`), or a real
      difference in the generator, with the divergence named
- [ ] If real: whether the episode-cost gap survives conditioning on Client count
