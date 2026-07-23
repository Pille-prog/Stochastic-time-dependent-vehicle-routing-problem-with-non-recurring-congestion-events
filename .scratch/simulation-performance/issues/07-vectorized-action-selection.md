# 07 — Vectorized candidate-action selection

**What to build:** `_select_vehicle_possible_actions` and
`_classify_shortest_distance_clients` over the ticket-04 geometry matrices:
travel times from a vehicle to all remaining clients come from one row slice;
the top-k selection replaces `heapq.nsmallest` over Python tuples.

**Blocked by:** 04.

**Status:** open

- [ ] Top-k with **identical ordering semantics** to
      `heapq.nsmallest(k, [(time, client), ...])` — lexicographic on
      (time, client id), so float ties break exactly as today (np.argpartition
      alone does not guarantee this; a stable sort on the k-slice or structured
      sort is required).
- [ ] The `list(set(...))` dedup quirk's iteration-order effect on the
      candidate list is preserved (or proven order-irrelevant downstream and
      the proof recorded in Comments).
- [ ] Forbidden-action filtering, the depot-append rules (the 350 cutoff, the
      end-of-horizon reachability check) and delayed-client injection keep
      their exact branch semantics.
- [ ] Tier 1 gate: bit-exact self-golden pass (selection is order/compare
      logic, no float arithmetic created); if any reduction reorders, drop to
      Tier 2 and justify.
- [ ] Benchmark note in Comments.
