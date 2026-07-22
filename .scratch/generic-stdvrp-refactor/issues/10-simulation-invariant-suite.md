# 10 — Simulation invariant suite (property-based)

**What to build:** The Hypothesis suite that makes simulations trustworthy: physical and accounting invariants asserted across randomly generated configs and seeds on the fixture, regardless of policy quality.

**Blocked by:** 07.

**Status:** ready-for-agent

- [ ] Simulated clock never decreases; the Episode terminates according to the Horizon rules
- [ ] Every Client ends the Episode served or penalized exactly once
- [ ] Travel times are strictly positive; congestion factors stay within the configured bounds
- [ ] Episode total cost equals distance + delay + earliness + overtime exactly
- [ ] Suite runs in CI within acceptable time (bounded examples/deadlines)
