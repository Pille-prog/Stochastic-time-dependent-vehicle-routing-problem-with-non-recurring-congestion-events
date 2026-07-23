# 09 — Model decomposition into concrete collaborators

**What to build:** The elegance ticket. `Model` is a god class mixing the
event loop, cost accounting, velocity sampling and episode running; `State`
carries parallel per-vehicle lists; `end_transition_function = 1/2` is an int
flag. Decompose into cohesive *concrete* classes (ADR-0002: no new seams, no
interfaces) without changing behavior. This runs last on the hot path so it
refactors the already-optimized code, not code tickets 05–07 are about to
replace.

**Blocked by:** 05, 06, 07.

**Status:** open

- [ ] Cost accounting (the `total_*` accumulators and per-transition charging)
      extracted into a concrete ledger collaborator; the transition function
      reads as event dispatch, not bookkeeping.
- [ ] Velocity sampling (`create_random_velocity`, memoization,
      `generate_normal_velocity`, congestion interaction) extracted as a
      concrete collaborator owning `velocity_rng` and the memo dict.
- [ ] The `end_transition_function` int flag becomes an explicit control-flow
      construct (bool/enum/loop restructure — pick what reads best).
- [ ] Per-vehicle parallel lists (`node_time_arrival`, `departure_tau`,
      `vehicles_shortest_path`, …) regrouped coherently — struct-of-arrays is
      fine (ADR-0003), scattered instance attributes are not.
- [ ] Preserved-quirk documentation (the 1150/1198 constants, the epoch-gate
      arithmetic) moves with the code it describes; no quirk is fixed here.
- [ ] Tier 1 gate: bit-exact self-golden pass. Benchmark note in Comments
      proving no measurable regression (>2% episode throughput).
