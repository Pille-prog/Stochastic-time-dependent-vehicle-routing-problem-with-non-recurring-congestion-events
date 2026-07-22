---
status: accepted
---

# Powell vocabulary; abstraction only at three variation points

The domain language follows Powell's sequential decision analytics framework (`State`, `Policy`, `Model` owning the transition function) rather than Gym-style RL (`Environment.step()`/`Agent`), because the research lineage and training loops already follow Powell — see `CONTEXT.md` for the glossary. The word *environment* is banned for data containers: the legacy class named `environment` held historical data, which reads backwards to anyone fluent in either vocabulary.

Interfaces exist at exactly three seams, each justified by a real existing-or-planned second implementation: **`Policy`** (multiple decision rules under active research), **`CongestionGenerator`** (multiple non-recurring event models already written), and **`DataSource`** (CSV today, database planned). Everything else — road network, travel-time model, state, transition, trainer — is deliberately concrete.

## Consequences

- ML engineers expecting `env.step()` will not find it; the mapping is `Model.transition_function`.
- Do not add interfaces/ABCs elsewhere "for consistency" or "for testability" — a new seam requires a concrete second implementation, per the rule that drove this design.
