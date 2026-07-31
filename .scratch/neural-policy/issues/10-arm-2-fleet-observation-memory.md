# 10 — Arm 2: the fleet's shared observation memory (conditional)

**What to build:** The one pre-committed escalation. Give the network what its
own fleet has already *seen*, pooled across vehicles — and measure whether that,
rather than the approximator, is what the problem was missing.

**Blocked by:** 09 — **runs only if arm 1 loses Gate B.**

**Status:** open (conditional)

## Why this arm exists and what it tests

`docs/research/rl-methodology-for-stdvrp.md`'s bottom line is that **the
observation set, not the algorithm, is the binding constraint**. Arm 1 holds the
observation constant on purpose, so that a win is attributable to the
approximator. If arm 1 loses, that is evidence *for* the research note — and
this arm is how the claim gets tested rather than merely inferred.

`delta(arm 2) − delta(arm 1)` is the value of the observation.
`delta(arm 1)` was the value of the approximator. That decomposition is the
scientific point of running two arms at all.

## What is admissible, and what is not

Arm 2 does **not** lift the observability rule (ADR-0006). It extends the
observation to the fleet's own measurements:

> A dispatcher may aggregate the reports of its own vehicles. It may not read
> the world's velocity field.

**Admissible:** velocities this Episode's vehicles actually measured on arcs
they actually traversed, pooled across the fleet and carried forward — a shared
memory rather than each vehicle's private 3-slot window. Aggregates derived from
that memory (e.g. observed-vs-historical speed ratio on visited regions, how
stale each observation is).

**Not admissible, and this is the whole discipline of the arm:**
`congested_arcs`, `TravelTimeModel` evaluated at `tau`, the velocity of any arc
no vehicle has driven, or the existence/expiry of any congestion event the fleet
has not experienced.

## Design notes

- `State.observed_velocity` is a per-vehicle window of the last
  `n_observed_velocities` **decision epochs**, not distinct arcs (B18,
  `docs/simulator-review.md`) — a vehicle resampled repeatedly on one arc fills
  several slots with that arc. `feature_extraction.py`'s own docstring argues
  this is *arguably the better congestion proxy of the two*, since an event
  lasts tens of minutes. Whatever this arm builds, that observation is the raw
  material; do not silently redefine it.
- The shared memory must be keyed by *where and when* an observation was taken,
  or it is just an average with no locality.
- Same everything else: same architecture, same learning rule, same budget, same
  frozen protocol, same 3 init seeds. **Only the tokens change** — otherwise the
  decomposition above is meaningless.

## Outcomes

1. **Wins**: the effort's objective is met, and the research note's claim is
   confirmed with a number attached. Ticket 11 records both.
2. **Loses**: the effort closes with an honest negative result. Ticket 11 writes
   **ADR-0008** recording it; the Policy ships available-but-not-default; and
   the finding — *a transformer over raw State, with and without fleet
   observation memory, did not beat a 19-feature linear VFA on this problem* —
   is a real contribution to the lab, not a failure to hide. There is precedent:
   `simulation-performance` ticket 10 measured a candidate fix, found it worse,
   and the negative result **was** the decision.

**No arm 3.** Two attempts, declared in advance. If both lose, the next idea is
a new effort with a new spec, decided with these numbers in hand — not another
lap of this one.

## Acceptance

- [ ] The two-arm decomposition reported explicitly: value of the approximator,
      value of the observation.
- [ ] Predicted self-golden diff: **zero.**

## Comments
