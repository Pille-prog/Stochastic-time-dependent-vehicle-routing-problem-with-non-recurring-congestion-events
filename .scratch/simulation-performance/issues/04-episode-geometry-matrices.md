# 04 — Per-Episode geometry matrices replace path-cache dict lookups

**What to build:** The core data structure of ADR-0003: dense numpy matrices of
`average_minutes` and `length`, built once per Episode from the
ShortestPathCache, indexed `[node, client-column]` — replacing the 1.13M
per-episode `path_between` tuple-dict lookups with array indexing, and
enabling every later vectorization (tickets 05, 07).

**Blocked by:** 01.

**Status:** resolved

- [x] A concrete facade (working name `EpisodeGeometry`; final name decided in
      ticket, domain vocabulary only if it earns a CONTEXT.md entry) built per
      Episode: columns are that Episode's clients + depot, rows cover every
      node id that can appear as a vehicle position; values copied from the
      ShortestPathCache so floats are bit-identical to today's lookups.
- [x] Scalar accessors mirror `path_between(...).average_minutes/.length`
      semantics (including KeyError-equivalent behavior for absent pairs) so
      call sites migrate mechanically; vectorized row/column accessors exist
      for tickets 05/07.
- [x] Path *node sequences* (`.nodes`, used by the Model for routing) stay on
      the ShortestPathCache — this ticket only migrates time/length reads on
      the Policy hot path.
- [x] Tier 1 gate: bit-exact self-golden pass (pure representation change; no
      float arithmetic reordered).
- [x] Benchmark note in Comments: episode throughput before/after.

## Comments

### Resolution (2026-07-23)

`EpisodeGeometry` (`src/stdvrp/network/episode_geometry.py`) built as the
working name settled: pure implementation vocabulary (array representation of
an existing lookup), not a domain concept, so no CONTEXT.md entry.

**Shape.** Columns are `depot` + this Episode's Clients, in that order
(`_ordered_unique`, dedup-preserving). Rows are every node id the
ShortestPathCache ever priced *toward some client* — in practice the road
network's full node universe, since the cache is dense over (every graph
node) x (the client-universe nodes); this is provably the exact set that can
appear as `state.vehicle_position` (depot at start, a served Client, or an
intermediate path node from `ShortestPathCache.path_between(...).nodes`,
which is itself drawn from that same node universe).

**Two representations, one build.** `EpisodeGeometry.build()` slices this
Episode's columns out of a *memoized whole-cache index* — one dense
`[all-nodes, all-clients]` pair of matrices built once per `ShortestPathCache`
instance (`WeakKeyDictionary`, safe since the cache is never mutated after
construction) and reused for every Episode of a run; only the column-select
(numpy fancy indexing) repeats per Episode. A first implementation rebuilt
from scratch every Episode via one filtered pass over
`ShortestPathCache.items()` — correct, but on the mini fixture it cost about
as much as the dict lookups it replaced (measured, see below), so it was
memoized. `ShortestPathCache` gained a non-copying `items()` view (`as_dict()`
stays test-only per its docstring) for exactly this one full-cache pass.

Scalar reads (`average_minutes`, `length`) additionally keep a `.tolist()`
nested-list copy of each per-Episode matrix. Measured on this machine: a
single `ndarray[i, j]` scalar read costs ~2x a nested-list `list[i][j]` read
(numpy's per-call C-API dispatch overhead) — worse than the dict lookup it
replaces once paid one call at a time across the ~150K scalar reads per
Episode the Policy still makes today. The numpy arrays stay canonical for the
vectorized `average_minutes_row/column` / `length_row/column` accessors
tickets 05/07 build on.

**Call-site migration.** `MonteCarloPolicy` no longer holds a
`ShortestPathCache` at all: its constructor takes `geometry: EpisodeGeometry`,
and every `path_between(...).average_minutes/.length` call site became
`self.geometry.average_minutes(...)` / `.length(...)` — mechanical,
one-for-one. `Model` is unchanged (still holds `ShortestPathCache` for
`.nodes` routing). `episode.py`'s two Episode runners build one
`EpisodeGeometry.build(shortest_path_cache, clients, depot)` per Episode,
right after `State`, and inject it into `MonteCarloPolicy`.

**Tests.** `tests/unit/test_episode_geometry.py` (new, 17 cases): shape and
column ordering, scalar/vector accessor parity with `path_between`,
KeyError-equivalence for absent pairs (missing cache pairs, nodes outside the
cache, and Clients outside *this Episode's* columns even when the underlying
cache has them for a different Episode), and a cross-check against the real
45x45 mini-fixture cache. `tests/unit/test_shortest_path_cache.py` gained one
case for `items()`. `tests/unit/test_monte_carlo_policy.py` migrated its
`MonteCarloPolicy` construction to build a geometry from its hand-built
caches. Full suite: 243 passed. Tier-1 self-golden gate: bit-exact, live on
this machine (numpy 2.4.6, Windows/AMD64 — matches the ticket-01 capture
fingerprint).

**Benchmark note (mini fixture, `scripts/benchmark_episodes.py --train 30
--eval 30`, this machine).**

```
                  before (ticket 01)   after (ticket 04)
training  s/ep          0.092                0.109
evaluation s/ep          0.060                0.096
```

A **regression**, not a win, in isolation — expected and documented rather
than hidden. `path_between` was 17% of episode time (ticket 01's profile);
replacing one dict lookup with a row/col dict-index pair plus a nested-list
read is inherently more per-call work than the single tuple-key dict lookup
it replaces, *when called one scalar at a time*. This ticket's payoff is not
its own throughput — it is that `average_minutes_row`/`_column` now exist as
true numpy views, which is what tickets 05 and 07 need to turn "one scalar
lookup per candidate, per feature" into "one batched array operation per
vehicle." Re-profiled (5 training episodes, mini fixture): `average_minutes`
+ its lookup logic is now ~45% of tottime (was `path_between` at 17%) — the
expected shape for an unvectorized caller sitting on top of a vectorization-
ready representation. Re-measure after 05/07 land; the effort's stopping rule
(no site > 10% of episode time) is evaluated across that arc, not this ticket
alone.
