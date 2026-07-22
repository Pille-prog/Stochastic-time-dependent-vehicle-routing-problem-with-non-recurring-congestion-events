# 02 — Package scaffold, toolchain and CI

**What to build:** An installable, empty `stdvrp` package with the full modern toolchain and green CI, so every later ticket lands on rails. Includes the approved repo rename.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] src-layout package `stdvrp`, installable with uv; lockfile committed
- [ ] Module layout mirrors the domain glossary (network, traffic, congestion, demand, simulation, policies, training) per ADR-0002
- [ ] ruff configured for lint + format; mypy configured gradual (strict on new core interfaces)
- [ ] pytest + hypothesis wired up with a trivial passing test
- [ ] GitHub Actions runs lint, typecheck and tests on every push/PR — green
- [ ] Repo renamed `stdvrp_orquestator` → `stdvrp_orchestrator` locally and on GitHub (user approved 2026-07-21)
