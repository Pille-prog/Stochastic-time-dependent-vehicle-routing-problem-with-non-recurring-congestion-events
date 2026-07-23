# 04 — Per-Episode geometry matrices replace path-cache dict lookups

**What to build:** The core data structure of ADR-0003: dense numpy matrices of
`average_minutes` and `length`, built once per Episode from the
ShortestPathCache, indexed `[node, client-column]` — replacing the 1.13M
per-episode `path_between` tuple-dict lookups with array indexing, and
enabling every later vectorization (tickets 05, 07).

**Blocked by:** 01.

**Status:** open

- [ ] A concrete facade (working name `EpisodeGeometry`; final name decided in
      ticket, domain vocabulary only if it earns a CONTEXT.md entry) built per
      Episode: columns are that Episode's clients + depot, rows cover every
      node id that can appear as a vehicle position; values copied from the
      ShortestPathCache so floats are bit-identical to today's lookups.
- [ ] Scalar accessors mirror `path_between(...).average_minutes/.length`
      semantics (including KeyError-equivalent behavior for absent pairs) so
      call sites migrate mechanically; vectorized row/column accessors exist
      for tickets 05/07.
- [ ] Path *node sequences* (`.nodes`, used by the Model for routing) stay on
      the ShortestPathCache — this ticket only migrates time/length reads on
      the Policy hot path.
- [ ] Tier 1 gate: bit-exact self-golden pass (pure representation change; no
      float arithmetic reordered).
- [ ] Benchmark note in Comments: episode throughput before/after.
