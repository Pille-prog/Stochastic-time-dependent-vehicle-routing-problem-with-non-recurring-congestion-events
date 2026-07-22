# Stochastic time-dependent vehicle routing with non-recurring congestion events

Research laboratory for the STDVRP: simulation and policy optimization on time-dependent stochastic road networks, following Powell's sequential decision analytics vocabulary (see `CONTEXT.md`). Chengdu is the first instance.

> **Status**: mid-refactor from the legacy single-file script to the `stdvrp` package. The pre-refactor reference lives at the git tag `legacy-monolith` (ADR-0001). Work is tracked in `.scratch/generic-stdvrp-refactor/`.

## Data

Instance data is not committed. See [`data/README.md`](data/README.md) for the expected files and how to place them. Tests run on a small committed fixture and never need the full dataset.

## Documentation

- `CONTEXT.md` — domain glossary (the vocabulary used everywhere in code, tests and issues)
- `docs/adr/` — architectural decision records
