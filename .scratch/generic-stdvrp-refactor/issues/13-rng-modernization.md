# 13 — RNG modernization with statistical re-baseline

**What to build:** Replace global `random`/`np.random` seeding with injected numpy Generators throughout the package, and prove equivalence statistically — exact golden-master equality cannot survive RNG reordering (ADR-0001 phase 2).

**Blocked by:** 12.

**Status:** claimed

- [ ] No global RNG seeding anywhere in the package; Generators are injected via config/constructors, one stream per stochastic concern (demand, congestion, velocities, policy exploration)
- [ ] Statistical regression test: mean episode cost over N seeds within a pre-registered tolerance of the pre-migration baseline
- [ ] The exact golden-master test is retired to legacy-only documentation status, superseded by the statistical baseline; the decision is logged in ADR-0001
