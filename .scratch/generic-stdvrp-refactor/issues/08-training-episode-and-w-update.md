# 08 — Training Episode and W update

**What to build:** The training side of the tracer bullet: exploration-mode policy path and W creation/update, producing a W sequence identical to the legacy for fixed seeds.

**Blocked by:** 07.

**Status:** ready-for-agent

- [ ] Training-mode policy path and the W update rule ported
- [ ] W after N fixture training episodes matches the golden master exactly
- [ ] The legacy warm-up quirk (first iteration runs with the hardcoded tiny learning rate before the configured one) is preserved and explicitly documented as legacy behavior pending ticket 12 triage
