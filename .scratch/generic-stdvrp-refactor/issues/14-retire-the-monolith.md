# 14 — Retire the monolith

**What to build:** The legacy 6,580-line script leaves the working tree (it lives forever in the `legacy-monolith` tag), and the README becomes the front door of the new repository.

**Blocked by:** 09.

**Status:** ready-for-agent

- [ ] Legacy script deleted from the working tree; the tag is referenced from README and ADR-0001
- [ ] README covers: what the project is (STDVRP research lab), repository structure, quickstart with uv, how to obtain and place the Chengdu data, and how to add a new Policy or CongestionGenerator (the two research axes)
- [ ] CONTEXT.md and the ADRs are linked from the README
