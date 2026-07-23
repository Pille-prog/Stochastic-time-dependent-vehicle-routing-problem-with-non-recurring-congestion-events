# 11 — Unit tests for pure functions

**What to build:** Fast, focused unit coverage of the pure computational core, catching regressions the episode-level tests would only report indirectly.

**Blocked by:** 08.

**Status:** resolved

- [x] ~~Haversine distance against known real-world reference distances~~ — n/a, dead code; see comment
- [x] Speed interpolation edge cases: interval boundaries, first/last interval, missing observations
- [x] W update: dimensions, a hand-computable single step
- [x] Epsilon-greedy selection at epsilon = 0 (pure greedy) and epsilon = 1 (pure random with seeded RNG)
- [x] Time-unit conversions

## Comments

**2026-07-23 — resolved.** Delivered three unit files (43 tests): `tests/unit/test_periods.py` (`period_start_minute`, the minutes-since-03:00 conversion), `tests/unit/test_speed_interpolation.py` (`_interpolated_speed`: exclusive window boundaries 420/540/660/840/960/1080, first/last minute inside each window, the preserved missing-endpoint → `None` quirk), and `tests/unit/test_monte_carlo_policy.py` (W/feature-vector dimensions 19 = 12 + 7, a hand-derived single SGD step, newest-first return accumulation with the `rewards[t+1]` indexing, ε=0 pure greedy touching no global RNG, ε=1 pure random replayed exactly from the seeded global stream).

**Haversine dropped as not applicable.** Neither haversine definition reaches the live path: the `DataCalculations` static shadow (legacy line 1133, over 901 — the shadowing was already flagged in ticket 07's notes) is called only from dead feature extractors (`calculate_clients_dispersion`, `calculate_nodes_caracteristics`) and the radius congestion variant; the `environment` copy (line 138) only from the commented-out shortest-path precomputation; the `policy` copy (line 3955) only from the Rutas/Multiarmed selector. All live distances come from `all_shortest_paths.csv` and `link.csv`. Porting a haversine nothing calls would violate the spec's port scope (dead variants stay in the `legacy-monolith` tag), so there is nothing to unit test.

**Drive-by fix (import graph, no behavior change):** `stdvrp/policies/monte_carlo.py` imported `State` at runtime, so importing `stdvrp.policies` before `stdvrp.simulation` raised a circular-import `ImportError` (every prior test entered via `stdvrp.simulation`, which hid it). Moved the import under `TYPE_CHECKING`, the same pattern `policies/base.py` already used for exactly this cycle.

Verification: 43 new tests green; full suite 211 passed, 5 deselected (golden opt-in); `ruff check`/`format` and `mypy src` clean.
