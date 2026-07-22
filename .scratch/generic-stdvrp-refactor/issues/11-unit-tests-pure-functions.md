# 11 — Unit tests for pure functions

**What to build:** Fast, focused unit coverage of the pure computational core, catching regressions the episode-level tests would only report indirectly.

**Blocked by:** 08.

**Status:** ready-for-agent

- [ ] Haversine distance against known real-world reference distances
- [ ] Speed interpolation edge cases: interval boundaries, first/last interval, missing observations
- [ ] W update: dimensions, a hand-computable single step
- [ ] Epsilon-greedy selection at epsilon = 0 (pure greedy) and epsilon = 1 (pure random with seeded RNG)
- [ ] Time-unit conversions
