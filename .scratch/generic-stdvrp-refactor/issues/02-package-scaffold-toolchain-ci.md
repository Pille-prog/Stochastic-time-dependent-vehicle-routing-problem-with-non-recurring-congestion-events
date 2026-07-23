# 02 — Package scaffold, toolchain and CI

**What to build:** An installable, empty `stdvrp` package with the full modern toolchain and green CI, so every later ticket lands on rails. Includes the approved repo rename.

**Blocked by:** None — can start immediately.

**Status:** resolved

- [x] src-layout package `stdvrp`, installable with uv; lockfile committed
- [x] Module layout mirrors the domain glossary (network, traffic, congestion, demand, simulation, policies, training) per ADR-0002
- [x] ruff configured for lint + format; mypy configured gradual (strict on new core interfaces)
- [x] pytest + hypothesis wired up with a trivial passing test
- [x] GitHub Actions runs lint, typecheck and tests on every push/PR — green
- [x] ~~Repo renamed~~ — split out to ticket 15 (manual user action; see comment)

## Comments

Resolved 2026-07-21. Full local chain green (ruff check, ruff format --check, mypy strict on package, pytest with hypothesis). Legacy monolith excluded from ruff until ticket 14. The rename criterion moved to ticket 15: it cannot be executed from an agent session because the GitHub CLI is not installed and the local folder is locked as the session's working directory. Discovery: the GitHub repo's real name is the long descriptive one (`Stochastic-time-dependent-...-congestion-events`), not `stdvrp_orquestator` — ticket 15 records both rename halves.
