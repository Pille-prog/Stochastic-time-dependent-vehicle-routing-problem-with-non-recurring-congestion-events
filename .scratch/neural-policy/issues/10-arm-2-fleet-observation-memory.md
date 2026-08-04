# 10 — Rung 3: the fleet's shared observation memory (conditional)

**What to build:** The one pre-committed escalation. Give the network what its
own fleet has already *seen*, pooled across vehicles — and measure whether
that, rather than the approximator or the estimator, is what the problem was
missing.

**Blocked by:** 09 — **runs only if neither encoder arm passes Gate B.** It
runs on whichever arm did better, and on nothing else.

**Status:** open (conditional)

## Why this rung matters more after ticket 08's diagnosis, not less

`docs/research/rl-methodology-for-stdvrp.md`'s bottom line is that **the
observation set, not the algorithm, is the binding constraint**, and its ranked
#1 — congestion-aware and time-dependent features — is the one it says
"strictly dominates every algorithmic change below".

This effort has now spent itself on the algorithmic changes. Tickets 13-17 do
recommendation **#3** (least-squares estimator) and **#5** (neural VFA over
enriched features) properly, and ticket 14 restores the candidate heuristic.
If Gate B still loses after all of that, the remaining explanation is the one
the research note put first — and this rung is how the claim gets *tested*
rather than inferred.

Rungs 1 and 2 hold the observation constant on purpose, so that any win is
attributable. That makes this the cleanest test of the note's central claim the
lab can run.

## The decomposition, now in three terms

```
delta(frozen)                     value of the cost features + the estimator
delta(trained) − delta(frozen)    value of representation learning
delta(rung 3)  − delta(best)      value of the observation
```

That decomposition is the scientific point of running staged rungs at all, and
it is the deliverable of this ticket even if the rung loses.

## What is admissible, and what is not

Rung 3 does **not** lift the observability rule (ADR-0006). It extends the
observation to the fleet's own measurements:

> A dispatcher may aggregate the reports of its own vehicles. It may not read
> the world's velocity field.

**Admissible:** velocities this Episode's vehicles actually measured on arcs
they actually traversed, pooled across the fleet and carried forward — a shared
memory rather than each vehicle's private 3-slot window. Aggregates derived
from that memory (e.g. observed-vs-historical speed ratio on visited regions,
how stale each observation is).

**Not admissible, and this is the whole discipline of the rung:**
`congested_arcs`, `TravelTimeModel` evaluated at `tau`, the velocity of any arc
no vehicle has driven, or the existence/expiry of any congestion event the
fleet has not experienced.

## Design notes

- `State.observed_velocity` is a per-vehicle window of the last
  `n_observed_velocities` **decision epochs**, not distinct arcs (B18,
  `docs/simulator-review.md`) — a vehicle resampled repeatedly on one arc fills
  several slots with that arc. `feature_extraction.py`'s own docstring argues
  this is *arguably the better congestion proxy of the two*, since an event
  lasts tens of minutes. Whatever this rung builds, that observation is the raw
  material; do not silently redefine it.
- The shared memory must be keyed by *where and when* an observation was taken,
  or it is just an average with no locality.
- **The myopic base is congestion-blind by construction** — `c(s, a)` is built
  from `EpisodeGeometry.average_minutes`, a static historical prior. So there
  are two distinct places this observation could enter: the residual's features
  `φ` (the network learns to correct the base where the fleet has seen the road
  is slow) or the base itself (the projection uses observed rather than
  historical times). **They are different experiments.** The first keeps ticket
  15's guarantee that `W = 0` reproduces the frozen null; the second moves the
  null and needs it re-frozen and re-reported. Decide and record which, and do
  not do both in one measurement.
- Same everything else: same architecture, same estimator, same action set,
  same budget, same frozen protocol, same 3 init seeds. **Only the observation
  changes** — otherwise the decomposition above is meaningless.

## Outcomes

1. **Wins**: the effort's objective is met, and the research note's central
   claim is confirmed with a number attached. Ticket 11 records both.
2. **Loses**: the effort closes with an honest negative result. Ticket 11
   writes **ADR-0009** recording it; the Policy ships available-but-not-default;
   and the finding — *a transformer over raw State, with a corrected estimator,
   the baseline's own action set and its own fleet's observations, did not beat
   a 19-feature linear VFA on this problem* — is a real contribution to the
   lab, not a failure to hide. There is precedent:
   `simulation-performance` ticket 10 measured a candidate fix, found it worse,
   and the negative result **was** the decision.

**No rung 4.** If this loses, the next idea is a new effort with a new spec,
decided with these numbers in hand — not another lap of this one. The research
note already names the successor: multiagent rollout (#2, Bertsekas Prop. 2.1),
which comes with a *guaranteed* improvement over its base policy and which the
one-agent-at-a-time structure this effort preserved is exactly built for.

## Acceptance

- [ ] The three-term decomposition reported explicitly, each as a number with a
      sign: value of the estimator, value of the representation, value of the
      observation.
- [ ] Which of the two entry points was used, and why, recorded before the
      measurement rather than after.
- [ ] Predicted self-golden diff: **zero.**

## Comments
