# Spec: Generic STDVRP package, instantiable for Chengdu

Status: approved (grilling session 2026-07-21)

## Goal

Transform the single-file research script (`Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py`, ~6,580 lines) into a professionally structured research laboratory: an installable `stdvrp` package that models the generic domain (stochastic time-dependent VRP with non-recurring congestion events), with Chengdu as its first instance, guarded by a test pyramid that pins current behavior before anything moves.

**Purpose of the repo**: active research lab. Optimize for safe, reproducible experimentation — not for a stable public API.

## Decisions (from the grilling session)

| Decision | Choice |
|---|---|
| Fidelity to current behavior | Characterize then evolve (ADR-0001): golden master against `legacy-monolith` tag, then deliberate documented changes |
| Vocabulary | Powell sequential decision analytics, English identifiers (ADR-0002, `CONTEXT.md`) |
| Abstraction seams | Exactly three: `Policy`, `CongestionGenerator`, `DataSource` (ADR-0002) |
| Port scope | Only the path `main()` exercises today: Monte Carlo episodes (train/test) with linear function approximation and the 4 cost components (distance, delay, earliness, overtime). Everything else stays in the git tag |
| Repo layout | src-layout package + `experiments/chengdu/` outside the package |
| Data | `data/` gitignored; small deterministic sub-network fixture committed for tests; database `DataSource` planned later |
| Tests | Golden master + Hypothesis property-based invariants + unit tests for pure functions + statistical regression for RNG migration |
| Tooling | uv, ruff (lint+format), mypy (gradual; strict on new core interfaces), pytest + hypothesis, GitHub Actions CI |
| Experiment config | Frozen, validated `ExperimentConfig` dataclass loaded from per-experiment `config.yaml` (versioned) |

## Target structure

```
stdvrp_orquestator/
├── pyproject.toml
├── CONTEXT.md                  # glossary (done)
├── docs/adr/                   # 0001, 0002 (done)
├── src/stdvrp/
│   ├── network/                # RoadNetwork, ShortestPathCache
│   ├── traffic/                # TrafficHistory, TravelTimeModel, DataSource (CSV impl)
│   ├── congestion/             # CongestionGenerator interface + the live implementation
│   ├── demand/                 # Client, ClientGenerator
│   ├── simulation/             # State, Model (transition function), Episode runners
│   ├── policies/               # Policy interface, MonteCarloPolicy (linear approx, W)
│   └── training/               # Trainer (train loop, evaluation, plots)
├── experiments/chengdu/
│   ├── config.yaml
│   └── run.py
├── data/                       # gitignored (link.csv, speed[601]_[0].csv, speed[601]_[1].csv)
└── tests/
    ├── fixtures/               # committed sub-network + captured golden-master values
    ├── test_golden_master.py
    ├── test_invariants.py      # Hypothesis
    └── unit/
```

## Live code path to port (traced from `main()`)

- `environment` data loading → `RoadNetwork` + `TrafficHistory` via CSV `DataSource`
- `DataCalculations`: speed interpolation, stochastic velocities, the congestion generation actually invoked by `model.transition_function`, travel times → `TravelTimeModel` + live `CongestionGenerator`
- `shortest_path_memory` → `ShortestPathCache`
- `ClientGenerator.client_generator_function` → `demand`
- `state` → `State`
- `policy`: constructor, `monte_carlo_policy_train/test`, the LAST definitions of shadowed methods (e.g. `select_vehicle_possible_actions` at line 4990), the feature extractors the MC path actually calls, `create_W`, `actualize_W` → `MonteCarloPolicy`
- `model`: `transition_function` and all its callees, `create_monte_carlo_episode_train/test` → `Model` + episode runners
- `training_and_testing.training_model` / `test_model` → `Trainer` (note: hardcoded horizon 300–780, `n_arcs=3`, warm-up lr `1e-6`, eval seeds 100000–100049, and the `mean_static_policy` plot lookup table all move to `ExperimentConfig`)

Exact call-graph confirmation of which shadowed variants are live happens during Phase 1 porting, method by method.

## Phases

### Phase 0 — Safety net (blocks everything else)
1. Tag current `main` state as `legacy-monolith`.
2. Add `data/` + `.gitignore`; document data acquisition in README.
3. Build the committed fixture: a small deterministic sub-network (tens of nodes) extracted from real data, with matching speed files.
4. Minimal shim so the ORIGINAL script runs on the fixture (parameterize the 3 hardcoded file paths — nothing else).
5. Capture golden master: episode costs / trajectories for fixed seeds on the fixture.

**Acceptance** (amended 2026-07-21 by the ADR-0001 addendum, discovery in ticket 03): committed mini fixture + structural tests in CI; the original script runs reproducibly on the **full local dataset** instead of the fixture — the legacy's hardcoded 1,900-node client universe, 60-client floor and 88-file aggregation make a sub-megabyte legacy-runnable fixture impossible. Golden-master values stored under `tests/fixtures/`; the exact-equality test skips when local data is absent.

### Phase 1 — Structural refactor (golden master stays green)
1. Scaffold: pyproject (uv), ruff, mypy, pytest+hypothesis, GitHub Actions.
2. Port the live path module by module, preserving global-RNG consumption order (ADR-0001).
3. `experiments/chengdu/run.py` + `config.yaml` replace `sys.argv` comma string.

**Acceptance**: `uv run experiments/chengdu/run.py` reproduces golden-master values exactly; CI green; ruff/mypy clean on new code.

### Phase 2 — Deliberate evolution
1. Hypothesis invariant suite: clock never decreases; every client ends served or penalized exactly once; travel times > 0; congestion factor within `[lower_bound, upper_bound]`; total cost = distance + delay + earliness + overtime; episode terminates within horizon rules.
2. Unit tests: haversine (known distances), interpolation, time conversion, W update dimensions, ε-greedy at ε=0/1.
3. Documented bug fixes (each with its own test + note).
4. RNG modernization: injected `np.random.Generator`, re-baseline via statistical regression (mean cost over N seeds within tolerance).

**Acceptance**: invariant + unit suites green; ADR-0001 phase-2 notes updated with each behavior change.

## Out of scope (deliberately)

- Porting the ~15 dead/shadowed experiment variants (Slack / K-Means / Multiarmed action selectors, fourier/routes/oficial feature extractors, `test_model_2`, unused congestion generators). They live in the `legacy-monolith` tag.
- Abstracting network/travel-time/state/transition — no second implementation exists (ADR-0002).
- Database `DataSource` implementation (the seam exists; the implementation waits for the real DB).
- Performance scalability work.

## Open questions (blockers marked)

1. ~~Data location~~ — resolved 2026-07-21: the CSVs live in the `Mega city` folder one level above the repo. `link.csv` 0.4 MB; speed files ~23 MB each (days 601–715, morning/afternoon); `all_shortest_paths.csv` 907 MB (precomputed path cache, relevant to ticket 06). Ticket 03 is unblocked.
2. ~~Package name~~ — resolved 2026-07-21: `stdvrp`.
3. ~~Repo directory typo~~ — resolved 2026-07-21: rename to `stdvrp_orchestrator`, executed as part of ticket 02.

## Tickets

Published to `issues/01`–`issues/14` (2026-07-21). Critical path: 03 → 04 → 07 → 08 → 09. Initial frontier: 01 and 02.
