# 05 — Endgame: no crash, no double booking

**What to build:** Two independent defects in the endgame branches of
`_select_vehicle_possible_actions`. Closes B5 and B11.

**Blocked by:** 01

**Status:** open

## B5 — `min()` on an empty sequence

`policies/monte_carlo.py:425-437`. Two uncoordinated thresholds leave a 40-minute
window uncovered: `_select_vehicle_possible_actions` diverts to the depot at
`tau > 350` (line 298) while `_classify_shortest_distance_clients` discards
vehicles at `tau > 310` (line 429). With exactly one Client pending, every
vehicle reading `position == depot`, and `310 < tau <= 350`, `distances` is
empty and `min()` raises `ValueError`. Reproduced by the review at tau 320 and
340.

The condition is easier to hit than it looks, because `position == depot` is
also true for vehicles that merely crossed the depot — this is ticket 04's root
cause showing up again.

- [ ] **Fallback only**: when `distances` is empty, fall back to the depot,
      exactly as the two-Client branch already does via `heapq.nsmallest`.
- [ ] **Do not unify 310 and 350.** The review proposes it; this effort's
      criterion excludes it — unifying thresholds is re-tuning, not fixing a
      miswritten predicate. Document the disagreement where the literals live,
      alongside the existing quirk note at `monte_carlo.py:62-64`.
- [ ] Invariant: the Policy returns a legal action for every vehicle, at every
      tau, under every valid config. Property-based, sweeping tau across the
      310/350 window with small pending-Client counts and small
      `min_number_clients` — the regime the review flags as most exposed.

## B11 — the same Client assigned to two vehicles

`policies/monte_carlo.py:302-308`. The endgame branch
(`len(clients_not_visited) < 3`) does not filter `forbidden_actions`, unlike the
normal branch at line 311. Reproduced: `action=[7, 7]` with two vehicles,
`action=[7, 0, 7]` with three; **734 transitions over 60 episodes**.

- [ ] Filter `forbidden_actions` in the endgame branch, matching the normal
      branch.
- [ ] Invariant: two vehicles never receive the same non-depot Client in one
      decision.
- [ ] Note the latent second-order effect and **leave it latent**: the losing
      vehicle arriving at an already-served Client is the only transition path
      that skips `commit_transition()` (`model.py:505-514`), which would put its
      cost inside `distance_cost` but outside `total_cost`. The review
      instrumented 84 episodes and reached it **zero times** — the loser always
      gets rerouted first. Fixing B11 removes the entry condition; do not also
      restructure the commit path on the strength of an unreached branch.

## Predicted self-golden diff

**B5 contributes zero.** The crash requires exactly one pending Client with all
vehicles reading `position == depot` in a 40-minute window; the 15 capture
episodes complete without raising, so the branch is not being taken. If ticket
01's 60-seed bench *did* trip it, this fix moved forward and that is recorded
there instead.

**B11 contributes a real diff.** At 734 duplicate transitions per 60 episodes it
fires often enough to expect hits among the 15 capture seeds. On affected seeds
the losing vehicle now receives a different action, so the trajectory changes:
expect `state_count`, `tau` and `distance_cost` to move, direction unpredictable
per seed. Aggregate direction over the 60-seed bench should be **cost
non-increasing** — the fix recovers fleet capacity that was being wasted on a
Client another vehicle was already serving — but a single seed may go either way
and that alone is not a contradiction.

**Seeds with zero duplicate assignments must be bit-identical in all three
blocks.** Ticket 01 reports the per-seed counts; list those seeds here before
running. That is this ticket's falsifiable claim.

## Evidence required

Duplicate-assignment counter at zero over the 60-seed bench (was 734 per 60
episodes). The crash invariant green across the swept tau window. The
untouched-seed bit-identity list.

## Comments
