# 07 — Vectorized candidate-action selection

**What to build:** `_select_vehicle_possible_actions` and
`_classify_shortest_distance_clients` over the ticket-04 geometry matrices:
travel times from a vehicle to all remaining clients come from one row slice;
the top-k selection replaces `heapq.nsmallest` over Python tuples.

**Blocked by:** 04.

**Status:** resolved

- [x] Top-k with **identical ordering semantics** to
      `heapq.nsmallest(k, [(time, client), ...])` — lexicographic on
      (time, client id), so float ties break exactly as today (np.argpartition
      alone does not guarantee this; a stable sort on the k-slice or structured
      sort is required).
- [x] The `list(set(...))` dedup quirk's iteration-order effect on the
      candidate list is preserved (or proven order-irrelevant downstream and
      the proof recorded in Comments).
- [x] Forbidden-action filtering, the depot-append rules (the 350 cutoff, the
      end-of-horizon reachability check) and delayed-client injection keep
      their exact branch semantics.
- [x] Tier 1 gate: bit-exact self-golden pass (selection is order/compare
      logic, no float arithmetic created); if any reduction reorders, drop to
      Tier 2 and justify.
- [x] Benchmark note in Comments.

## Comments

### Resolution (2026-07-24)

`MonteCarloPolicy._closest_allowed_clients` (new) is the whole vectorization:
one `EpisodeGeometry.average_minutes_row(position)` slice plus one
`np.lexsort` replaces the per-Client `average_minutes` lookups and the
`heapq.nsmallest` over `(travel time, Client)` Python tuples. Every other
branch of `_select_vehicle_possible_actions` — the depot cutoffs, the
`list(set(...))` dedup, the end-of-horizon depot append, the delayed-client
injection — is untouched, line for line.

**Ordering is identical, not just equivalent.** `np.lexsort((ids, times))` is
a stable multi-key sort with `times` primary and Client id secondary, which is
exactly what tuple comparison did, and `nsmallest(k, xs) == sorted(xs)[:k]`.
`np.argpartition` was rejected as the ticket anticipated: it makes no ordering
promise inside a partition, so a float tie straddling the k-th position would
resolve arbitrarily (confirmed against numpy's own docs — partition only
accepts a secondary key via structured `order=`, and still leaves intra-
partition order undefined). A full lexsort over the remaining Clients costs
microseconds at every size this simulation reaches, so the O(n log n) vs
O(n log k) difference is not worth the tie-break risk.

**Forbidden Clients are dropped after the sort, not before.** Taking the
`k + len(forbidden)` nearest guarantees at least `k` allowed survivors whenever
the filtered set has that many — at most `len(forbidden)` of them can be
forbidden — so the first `k` survivors are exactly `nsmallest(k, filtered)`.
The alternative (a boolean mask over every remaining Client before sorting)
was implemented and measured *slower than the code it replaces* at fixture
scale — 19.4µs vs 9.4µs at 21 Clients — because it costs one vectorized pass
per forbidden vehicle over the full array to save a Python scan of ~7 ids.

**The `list(set(...))` quirk is preserved by construction**, not by argument:
the line is unchanged and still fed the top-k ids in nearest-first order, as
the same Python `int` objects the State holds. Order genuinely matters
downstream — `_select_best_q_action_for_vehicle` keeps the *first* argmin on a
tie and `rng.choice(self.possible_actions)` indexes the list — so
`_closest_allowed_clients` returns State ints rather than numpy scalars
(`test_selected_ids_are_the_states_own_python_ints`); a leaked `np.int64` would
compare equal in every test while hashing into `clients_arrival` differently.

**One rebuild per decision pass.** `_remaining_clients` memoizes
`clients_not_visited` as the id array and the geometry column indices
(`EpisodeGeometry.column_positions`, new — the one API this ticket adds), keyed
on *content* equality: the Model mutates that list in place, so an identity
check would go stale silently. All ~6–12 selection calls of a decision pass
share one build. A stale-cache mutant fails
`test_selection_follows_clients_being_served`.

**`_classify_shortest_distance_clients` stays scalar — measured, not assumed.**
It only runs with one or two Clients left, over at most 8 vehicles, so its
arrays are 2 elements wide; the vectorized form (one `np.ix_` gather plus a
row sum per vehicle) measured **17.0µs against the scalar 8.5µs** — numpy's
per-call dispatch overhead dominates at that width, the same effect ticket 04
recorded for single-cell reads. It also does not appear at all in the cProfile
top-22 (<0.4% of episode time). Vectorizing it would have cost measured
performance for symmetry, which the spec explicitly forbids. Its branch
semantics are therefore trivially preserved, and the differential test below
still covers the endgame path.

**Tests.** `tests/unit/test_monte_carlo_policy.py` gained
`TestCandidateSelection` (6 cases) and `reference_possible_actions` — the
pre-ticket implementation kept verbatim as a differential oracle. The main case
runs 200 randomized states × 3 vehicles through *every* branch (empty /
endgame / normal, random vehicle positions, random taus across both the 350 and
end-of-horizon cutoffs, random taken actions, k from 1 to 4) with travel times
quantized to half-minutes so Client ties are frequent, mutating
`clients_not_visited` in place exactly as `Model.vehicle_reaches_client` does.
Two mutants were run against it to prove it bites: replacing the lexsort with a
stable sort on times alone, and never invalidating the memo — each fails.
Full suite: **249 passed** (243 + 6 new), run in a clean worktree at HEAD with
only this ticket's diff applied, because the shared working tree currently
carries other sessions' in-flight ticket 05/08 work.

**Tier-1 gate: bit-exact, live on this machine** (not skipped —
`tests/test_self_golden.py`, 5/5). Expected: this ticket creates no float
arithmetic, only compare-and-order logic over values copied verbatim from the
same matrices.

**Benchmark (this machine, mini fixture).** Whole-episode wall clock is
useless here — other sessions were running episodes on this machine throughout,
and world load alone varied 26–28s across runs — so the site was measured
directly, both implementations alternating inside one process against the same
world and seeds (min over 8 rounds of 5 training + 5 evaluation episodes):

```
time inside _select_vehicle_possible_actions, per episode
  before   7.09 ms/ep        after   5.84 ms/ep      +21% at the site
  (5236 selection calls per round, identical both ways)
```

That is ~5% of episode time on the fixture (cProfile: the site fell from 6.6%
to 4.9% of profiled time, and 29,038 of the ~744k per-run `average_minutes`
calls disappeared), so it is invisible in whole-episode timing — the fixture's
21 Clients are the least favorable case for this change. Per-call, with the
real geometry and the real method bodies on both sides:

```
clients  vehicles     legacy   vectorized   (arrays rebuilt every call)
     21         6    18.3 µs      7.4 µs         14.1 µs
     60         5    37.0 µs      8.6 µs         21.1 µs
    150         6   112.8 µs     15.7 µs         41.2 µs
```

150 Clients is the real Chengdu configuration (`mean_number_clients: 150`),
where the legacy path is linear in remaining Clients and this one is nearly
flat: **7.2× per call**. The third column is the pathological case where the
memo never hits (rebuilding the id/column arrays on every single call instead
of once per decision pass) — still faster than the code it replaces at every
size, so the memo is an optimization, not a correctness crutch.

Re-measure after ticket 05 lands: `_extract_state_action_features` is still
1.9s of the 2.6s profile and owns the other 96% of the geometry lookups, so the
effort's "no site above 10% of episode time" rule is decided there, not here.
