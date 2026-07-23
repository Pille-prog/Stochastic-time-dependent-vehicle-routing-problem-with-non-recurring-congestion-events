# 07 — Tracer bullet: Monte Carlo evaluation Episode end-to-end

**What to build:** A full evaluation Episode through the new package: State, Model transition function, the live CongestionGenerator, and MonteCarloPolicy in evaluation mode with a given W — reproducing the golden-master episode cost exactly. This is the slice that proves the whole architecture.

**Blocked by:** 04, 05, 06.

**Status:** resolved

- [x] Policy interface plus MonteCarloPolicy evaluation path: the feature extraction, action-set selection and epsilon-greedy variants the legacy main() actually executes (the LAST definitions of shadowed methods — nothing else)
- [x] CongestionGenerator interface plus the single live implementation
- [x] Model.transition_function and its callees ported, preserving global-RNG consumption order (ADR-0001)
- [x] For every golden-master seed: episode total cost and all four components match exactly

## Comments

**2026-07-22 — resolved.** Delivered: `Policy` ABC (`src/stdvrp/policies/base.py`) + `MonteCarloPolicy` evaluation path (`monte_carlo.py`); `CongestionGenerator` ABC + live `ArcProbabilityCongestionGenerator` (`src/stdvrp/congestion/generator.py`); `Model.transition_function` and callees (`src/stdvrp/simulation/model.py`) with `State` (`state.py`) and the `run_evaluation_episode` orchestrator + `EpisodeResult` (`episode.py`). `ShortestPathCache` signatures widened `int` → `float` for the legacy int/float node-id quirk.

Facts worth knowing for later tickets:

- **Shadowed-method resolution needs AST, not grep.** Most apparent duplicate defs in the monolith (`select_vehicle_possible_actions` at 2023/2085/2155/4757/4990, `generate_best_approximate_action` at 4168, `monte_carlo_policy` at 5011) are dead code inside triple-quoted string blocks. The surviving defs on the eval path: `select_vehicle_possible_actions` line 4450 (3-arg), `generate_best_approximate_action` 4088, and the one genuine last-wins shadow in the file: `haversine_distance` 1133 (no `self`, swapped arg order) over 901.
- **Eval-path RNG order** (preserved bit-exact): per seed, `ClientGenerator.generate(seed)` consumes global `random` (seed/gauss/sample/randint), then `np.random.seed(seed)`; per horizon step inside `transition_function`, all congestion `np.random.uniform` draws (one per arc in `event_probability` insertion order) happen before per-vehicle `random.gauss` velocity draws (memoized per `(node_start, node_end, minute)`; congested arcs skip the draw). The greedy test policy consumes no randomness.
- **Verification:** `tests/unit/test_congestion_generator.py` 6 passed; `tests/test_evaluation_episode_vs_legacy.py` 4 seeds bit-identical vs the in-process legacy monolith incl. end-of-episode RNG stream positions (~65 s, committed mini fixtures); full suite 159 passed; `uv run mypy src` clean; `uv run pytest -m golden` (`tests/test_new_package_vs_golden_master.py`) **passed in 22 min 27 s** against the full local Chengdu dataset — every golden-master test seed's total cost and all four components match exactly.
