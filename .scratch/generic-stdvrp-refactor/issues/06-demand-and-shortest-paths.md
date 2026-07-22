# 06 — Demand and shortest paths

**What to build:** ClientGenerator and ShortestPathCache ported: with a fixed seed, the generated Clients (locations, time windows) and the vehicle count are byte-identical to the legacy script on the fixture.

**Blocked by:** 05.

**Status:** ready-for-agent

- [ ] Client generation reproduces the legacy exactly for fixed seeds — same global-RNG consumption order (ADR-0001)
- [ ] ShortestPathCache returns paths and costs equal to the legacy cache
- [ ] Comparison tests against captured legacy outputs cover both
