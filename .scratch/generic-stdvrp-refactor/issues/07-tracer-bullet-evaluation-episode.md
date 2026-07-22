# 07 — Tracer bullet: Monte Carlo evaluation Episode end-to-end

**What to build:** A full evaluation Episode through the new package: State, Model transition function, the live CongestionGenerator, and MonteCarloPolicy in evaluation mode with a given W — reproducing the golden-master episode cost exactly. This is the slice that proves the whole architecture.

**Blocked by:** 04, 05, 06.

**Status:** ready-for-agent

- [ ] Policy interface plus MonteCarloPolicy evaluation path: the feature extraction, action-set selection and epsilon-greedy variants the legacy main() actually executes (the LAST definitions of shadowed methods — nothing else)
- [ ] CongestionGenerator interface plus the single live implementation
- [ ] Model.transition_function and its callees ported, preserving global-RNG consumption order (ADR-0001)
- [ ] For every golden-master seed: episode total cost and all four components match exactly
