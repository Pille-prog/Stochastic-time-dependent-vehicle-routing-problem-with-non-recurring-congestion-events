# 14 — Retire the monolith

**What to build:** The legacy 6,580-line script leaves the working tree (it lives forever in the `legacy-monolith` tag), and the README becomes the front door of the new repository.

**Blocked by:** 09.

**Status:** resolved

- [x] Legacy script deleted from the working tree; the tag is referenced from README and ADR-0001
- [x] README covers: what the project is (STDVRP research lab), repository structure, quickstart with uv, how to obtain and place the Chengdu data, and how to add a new Policy or CongestionGenerator (the two research axes)
- [x] CONTEXT.md and the ADRs are linked from the README

## Answer

The monolith is deleted from the working tree; its only home is now the `legacy-monolith` tag, pinned by the golden master's `legacy_sha256` (tag blob verified byte-identical to the deleted file). The characterization venue survives: `tests/characterization_world.py` and `scripts/capture_golden_master.py` gained `read_legacy_source()`/`legacy_script_path()`, which extract the script from the tag at run time (`git show legacy-monolith:<file>`, cached to a temp file per process) — the vs-legacy suites (still needed by tickets 12/13) and `pytest -m golden` keep working unchanged, and fail with a `git fetch --tags` hint rather than skipping silently when the tag is unreachable. The duplicate per-file loader fixtures in `test_demand_and_paths_vs_legacy.py` / `test_travel_time_model_vs_legacy.py` now delegate to the shared loader; `test_golden_master.py` compares `capture.legacy_sha256()` (tag) against the stored capture. CI checkout sets `fetch-tags: true` so the tag exists in its shallow clone; the monolith's ruff `extend-exclude` is gone from pyproject.

README rewritten as the front door: project purpose, repository structure, uv quickstart (`uv sync` / `uv run pytest` needs no data), the one-command Chengdu experiment, data acquisition (pointer to `data/README.md`), how to add a `Policy` or `CongestionGenerator` (the two research axes, with the ADR-0002 no-new-seams rule), and links to `CONTEXT.md`, both ADRs, the golden-master README and the issue tracker. ADR-0001 gained a ticket-14 addendum recording the deletion and the tag-extraction mechanism.

Full suite after deletion: 222 passed (6 golden deselected); ruff, ruff format and mypy clean. Note: the working tree carries unrelated ticket-12 WIP (`travel_time_model.py` NaN-std fix + its test) — left uncommitted, only its dead `import math` was removed to keep `ruff check .` green.
