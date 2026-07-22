---
description: Work the next unblocked ticket of the generic-stdvrp-refactor effort (frontier worker)
---

Implement the next ticket on the frontier of the `generic-stdvrp-refactor` effort. If $ARGUMENTS names a ticket number, work that one instead.

## Read first, in this order

1. `CONTEXT.md` — the domain glossary. Use exactly this vocabulary in code, tests, commits and comments; never the `_Avoid_` terms.
2. `docs/adr/0001-characterize-then-evolve-refactor.md` and `docs/adr/0002-powell-vocabulary-and-three-variation-points.md` — the hard rules: golden-master fidelity, global-RNG order preservation, and abstraction at exactly three seams (Policy, CongestionGenerator, DataSource) — never add a fourth "for consistency".
3. `.scratch/generic-stdvrp-refactor/spec.md` — decisions, phase plan, critical path.
4. `.scratch/generic-stdvrp-refactor/issues/` — the frontier is the lowest-numbered ticket whose `Blocked by` tickets are all `Status: resolved` and whose own status is `ready-for-agent`. Read the resolved tickets' `## Comments` — they carry findings recorded for you.

## Rules of engagement

- **One ticket per session.** When it is resolved, stop and summarize; do not start the next one.
- **Claim before working**: set `Status: claimed` in the ticket file. **Resolve when done**: check the acceptance boxes, set `Status: resolved`, and append a dated `## Comments` entry with the key findings the next ticket needs.
- The legacy monolith (`Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py`, tag `legacy-monolith`) is **read-only reference**. Sole exception: the minimal path shim explicitly allowed by ticket 04.
- Ports must reproduce legacy behavior exactly, including consuming the global `random`/`np.random` streams in the same order (ADR-0001). Found a bug? **Preserve it**, record it in the ticket comment for ticket 12's triage. Never fix silently.
- Remember Python keeps only the LAST of duplicate method definitions — when porting from the monolith, verify which definition is live before reading any shadowed one.
- Quality gate before every commit: `uv run ruff format .` && `uv run ruff check .` && `uv run mypy` && `uv run pytest`.
- Commit with focused messages referencing the ticket; push the feature branch; verify CI via `https://api.github.com/repos/Pille-prog/Stochastic-time-dependent-vehicle-routing-problem-with-non-recurring-congestion-events/actions/runs?per_page=1` (no `gh` CLI on this machine).
- Chengdu data: see `data/README.md` (source files live in the parent folder `Mega city`). Tests must never require the full dataset — only the committed fixture.

## Deliverable

The ticket's acceptance criteria all green with CI passing, the ticket file resolved with findings recorded, and a final summary stating what was built, what was verified, and what the next frontier ticket is. If you keep a memory directory, update the refactor-status note there.
