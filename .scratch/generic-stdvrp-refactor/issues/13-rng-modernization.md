# 13 — RNG modernization with statistical re-baseline

**What to build:** Replace global `random`/`np.random` seeding with injected numpy Generators throughout the package, and prove equivalence statistically — exact golden-master equality cannot survive RNG reordering (ADR-0001 phase 2).

**Blocked by:** 12.

**Status:** resolved

- [x] No global RNG seeding anywhere in the package; Generators are injected via config/constructors, one stream per stochastic concern (demand, congestion, velocities, policy exploration)
- [x] Statistical regression test: mean episode cost over N seeds within a pre-registered tolerance of the pre-migration baseline
- [x] The exact golden-master test is retired to legacy-only documentation status, superseded by the statistical baseline; the decision is logged in ADR-0001

## Comments

**2026-07-23 — resolved.** `src/stdvrp` no longer touches global `random`/`np.random` state anywhere; every stochastic concern gets an injected `np.random.Generator` (ADR-0001 phase-2 addendum has the full design). Summary:

- **Demand** (`ClientGenerator.generate(seed)`): builds `np.random.default_rng(seed)` locally per call — demand draws once per Episode, so no cross-call state is needed. Signature unchanged.
- **Congestion, velocities, policy exploration**: `stdvrp.simulation.episode._spawn_episode_rngs(seed)` derives three independent children via `np.random.SeedSequence(seed).spawn(3)` and injects them into the fresh `Model` (`velocity_rng`, `congestion_rng` — new required constructor params) and `MonteCarloPolicy` (`exploration_rng` — new required keyword-only param) each Episode runner call constructs. `ArcProbabilityCongestionGenerator.generate` gained a required `rng` parameter (it's Trainer-wide and long-lived, so the stream can't live on it as a field).
- **Policy exploration collapsed from three legacy draw sites to one**: the legacy split it across `local_rng` (gate), `local_rng_2` (repair) and the *global* stream (the exploratory pick). `MonteCarloPolicy` now uses a single `exploration_rng` for all of it, which retired `train_exploration_seed_offset`/`train_repair_seed_offset` from `ExperimentConfig` and `exploration_seed`/`repair_seed` from `run_training_episode` — that offset convention existed only to make the legacy's *unseeded* RNGs reproducible from the driver side; a Generator spawned from the Episode seed is deterministic by construction, so the workaround is gone, not deprecated.
- **Retired bit-exact suites** (cannot pass once streams are independent and PCG64-backed instead of a shared Mersenne-Twister stream): `tests/test_evaluation_episode_vs_legacy.py` and `tests/test_training_episode_vs_legacy.py` deleted outright; the demand half of `tests/test_demand_and_paths_vs_legacy.py` removed (path-cache half stays, no RNG involved); `characterization_world.py`'s `LegacySpreadCongestionGenerator`/`count_horizon_terminations`/`build_legacy_calc`/`build_legacy_spm`/`build_ported_world` removed as dead code once their only callers were gone. `tests/unit/test_congestion_generator.py` and the epsilon-greedy half of `tests/unit/test_monte_carlo_policy.py` were rewritten (not deleted) around a tiny scripted-RNG test double instead of numpy-seed-hunting — a better test shape, only possible once the code accepted an injected Generator.
- **`tests/test_golden_master.py` untouched** — it never imported `stdvrp`; it was already legacy-only documentation before this ticket.
- **Statistical regression, verified on the full local dataset**: `tests/test_new_package_vs_golden_master.py` replaced its three exact-equality tests with three mean-cost-over-N-seeds tolerance checks (Trainer.run() end to end, the captured eval seeds, and each final-test action-count's seeds), reusing `chengdu_full_phase2.json` (ticket 12's re-baseline, captured the moment before this ticket touched any RNG call site) as the pre-migration baseline. Pre-registered tolerance: 40% relative to the baseline mean, sized from that baseline's own ~40% coefficient of variation across its eval/test seed samples. `uv run pytest -m golden` on the full local dataset: **6 passed** (3 legacy-vs-frozen-capture in `test_golden_master.py`, unaffected; 3 new statistical checks, all within tolerance) in 27m25s.

Verification: `uv run pytest` (non-golden) 199 passed; `uv run pytest -m golden` 6 passed on the full dataset; ruff + ruff format + mypy clean.

**Effort-2 note**: the frontier is now empty for agent work. Tickets 01–14 are all resolved; ticket 15 (repo rename) is a user-only action with nothing blocking on it. The `generic-stdvrp-refactor` effort's Phase 0–2 scope (spec.md) is complete.
