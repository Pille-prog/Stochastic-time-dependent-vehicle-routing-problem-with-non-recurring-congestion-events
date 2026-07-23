# Stochastic time-dependent vehicle routing with non-recurring congestion events

Research laboratory for the STDVRP: simulation and policy optimization on time-dependent stochastic road networks. The `stdvrp` package models the generic domain — vehicles serving time-windowed clients while congestion events perturb historical travel times — and a concrete problem instantiates it with data and a config; Chengdu is the first instance. Vocabulary follows Powell's sequential decision analytics framework (`State`, `Policy`, `Model` with a transition function — see [`CONTEXT.md`](CONTEXT.md)).

The pre-refactor single-file script that backs the original research results lives at the git tag `legacy-monolith` (`git show legacy-monolith:Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py`) — it is no longer in the working tree, but the characterization tests still read it from the tag to prove the port exact (ADR-0001).

## Repository structure

```
src/stdvrp/            The installable package (generic domain)
├── network/           RoadNetwork, ShortestPathCache
├── traffic/           TrafficHistory, TravelTimeModel, DataSource (CSV impl)
├── congestion/        CongestionGenerator interface + implementations
├── demand/            Client, ClientGenerator, EpisodeDemand
├── simulation/        State, Model (transition function), Episode runners
├── policies/          Policy interface, MonteCarloPolicy (linear approximation)
├── training/          Trainer (train / evaluate / test loops, plots)
└── config.py          ExperimentConfig: one frozen, validated dataclass per run
experiments/chengdu/   The Chengdu instance: config.yaml + run.py (outputs in runs/)
data/                  Instance data, gitignored (see data/README.md)
tests/                 Test pyramid; fixtures/ holds the committed mini network
scripts/               capture_golden_master.py, make_fixture.py
docs/adr/              Architectural decision records
CONTEXT.md             Domain glossary (the ubiquitous language of code and issues)
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/); it provisions Python and the environment on first use.

```
uv sync                      # install the package and dev tools
uv run pytest                # full suite on the committed fixture — no data needed
uv run ruff check . && uv run mypy
```

To run the full Chengdu experiment — training with periodic evaluation and best-W tracking, then the final test, all driven by `experiments/chengdu/config.yaml`:

```
uv run python experiments/chengdu/run.py
```

Results (`results.json`) and the training plot land in a per-run directory under `experiments/chengdu/runs/` (gitignored); `--config` and `--output-dir` override the defaults. It needs the full dataset in place (below) and spends ~15 minutes loading it before training starts. CI exercises a smoke-sized version of the same path on the committed fixture (`tests/test_trainer_smoke.py`).

## Data

Instance data is not committed. The Chengdu road network (`link.csv`), the historical speed archive (`speed[601..715]_[0/1].csv`) and the precomputed path cache (`all_shortest_paths.csv`) are kept locally; the default `config.yaml` reads them from the `Mega city` folder one level above the repository, and [`data/README.md`](data/README.md) documents each file and how to place copies under `data/` instead. Tests never need any of this — they run on the small deterministic fixture committed under `tests/fixtures/chengdu_mini/`.

## Extending the lab: the two research axes

Interfaces exist at exactly three seams, each backed by a real second implementation — `Policy`, `CongestionGenerator`, `DataSource` (ADR-0002). The first two are the research axes:

**A new Policy** (decision rule): subclass `Policy` in `src/stdvrp/policies/base.py` — one method, `decide(state) -> list[int]`, the next node per vehicle. `MonteCarloPolicy` (`policies/monte_carlo.py`) is the reference implementation. Wire it into a `Trainer` (or a plain evaluation episode in `simulation/episode.py`), give it unit tests plus an invariant run (`tests/test_invariants.py` shows the harness), and keep its parameters in `ExperimentConfig` rather than hardcoded.

**A new CongestionGenerator** (event model): subclass `CongestionGenerator` in `src/stdvrp/congestion/generator.py` — one method, `generate(minute_start, congested_arcs)`, adding this epoch's events in place. `ArcProbabilityCongestionGenerator` is the live implementation; the invariant to preserve is that every velocity multiplier stays within the configured `[lower_bound, upper_bound]`. The legacy tag holds several unported variants (by radius, bounded); resurrect them deliberately, with tests, rather than copying blindly (ADR-0001).

Everything else — network, travel-time model, state, transition, trainer — is deliberately concrete: do not add seams without a concrete second implementation (ADR-0002).

## Documentation

- [`CONTEXT.md`](CONTEXT.md) — domain glossary; code, tests and issues use exactly this vocabulary
- [`docs/adr/0001`](docs/adr/0001-characterize-then-evolve-refactor.md) — characterize-then-evolve: golden master against the `legacy-monolith` tag, then deliberate documented changes
- [`docs/adr/0002`](docs/adr/0002-powell-vocabulary-and-three-variation-points.md) — Powell vocabulary; abstraction only at the three variation points
- [`tests/fixtures/golden_master/README.md`](tests/fixtures/golden_master/README.md) — what the golden master pins and how to re-verify it (`uv run pytest -m golden`)
- `.scratch/generic-stdvrp-refactor/` — spec and issue tracker of the refactor effort
