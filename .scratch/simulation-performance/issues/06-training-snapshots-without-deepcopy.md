# 06 — Training snapshots without deepcopy

**What to build:** Replace the per-step `copy.deepcopy(self.state)` /
`copy.deepcopy(action)` in `Model.run_training_episode` with a purpose-built
snapshot capturing exactly what `MonteCarloPolicy.update_W` replays.

**Blocked by:** 01.

**Status:** open

- [ ] Identify (and pin with a test) the exact State surface `update_W` reads
      via `_calculate_already_acquired_cost`, `_extract_general_state_features`
      and `_extract_state_action_features`: `tau_episode`,
      `clients_not_visited`, `vehicle_position`, `observed_velocity`, plus any
      field the ticket's audit finds. The snapshot copies only that, shallowly
      where aliasing is safe, and is immutable (frozen dataclass or equivalent).
- [ ] `update_W` accepts the snapshot type; the `self.state` rebinding to
      historical snapshots keeps working (or is redesigned away — the rebind is
      hidden temporal coupling; dissolving it counts as the elegance goal, but
      stays behavior-identical).
- [ ] Tier 1 gate: W trajectories bit-identical to self-golden on the fixture.
- [ ] Benchmark note in Comments: training-episode throughput before/after.
