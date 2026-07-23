# 04 — Golden master capture from the legacy script

**What to build:** The frozen behavioral reference that guards the whole refactor: exact episode outcomes of the original script running on the fixture with fixed seeds, checked by an automated test.

**Blocked by:** 03.

**Status:** resolved

- [x] A minimal shim parameterizes ONLY the three hardcoded data-file paths of the legacy script — nothing else changes (ADR-0001) — implemented as cwd-redirection (`contextlib.chdir` onto the full-data folder) plus importing the monolith unmodified; zero legacy edits
- [x] Golden master captured on the fixture: per-seed episode total cost and the four components (distance, delay, earliness, overtime), plus the W trajectory for a small training run — captured on the **full local dataset** per the ADR-0001 addendum (4 train episodes with W after each, 10 eval seeds, 10 test seeds at actions=2 with total cost + 4 components + tau/state/client counters)
- [x] A pytest test re-runs the legacy script on the fixture and matches the stored values exactly — `tests/test_golden_master.py` (marker `golden`, skips without local data), verified green: bit-exact float equality on re-run in a fresh process
- [x] Captured values and the capture procedure committed alongside the fixtures — `tests/fixtures/golden_master/chengdu_full.json` embeds the full protocol + meta (legacy sha256, data-file signature, library versions); procedure in `scripts/capture_golden_master.py` and the fixtures README

## Comments

Resolved 2026-07-22. Delivered: `scripts/capture_golden_master.py` (shim + capture driver + world cache), `tests/fixtures/golden_master/` (chengdu_full.json 9.7 KB + README), `tests/test_golden_master.py` (3 golden-marked tests: data signature, legacy sha256, bit-exact re-run), `tests/unit/test_golden_capture_helpers.py` (17 unit tests, run in CI), pytest `golden` marker deselected by default via addopts.

**Findings the next tickets need:**

1. **Legacy training is nondeterministic by construction**: `policy.__init__` (monolith line 1677-78) creates two **unseeded** `random.Random()` instances (`local_rng`, `local_rng_2`) used only by `select_epsilon_greedy_action_train` (exploration gate + infeasible-action repair). A plain legacy run trains differently every time. The capture seeds them per train episode at driver level (`protocol["train_exploration_seed_offset"|"train_repair_seed_offset"] + train_seed`, offsets 10M/20M) right after constructing the policy — the legacy file is untouched. **Tickets 08/09: the ported Trainer must reproduce this exact seeding convention to match the stored W trajectory.** Eval/test episodes are unaffected (`select_epsilon_greedy_action_test` uses no RNG), so they are exact functions of (seed, W, data).
2. **Protocol equivalence**: the capture equals a legacy run with `total_train_iterations=4, test_frequency=4, num_iteraciones_test=1`; eval/test seed lists are prefixes of the legacy's hardcoded tables, exact because every episode begins with `random.seed(seed)` (inside `client_generator_function`) + `np.random.seed(seed)`.
3. **The legacy load costs ~25 min CPU** (88 speed files re-read by `DataCalculations.read_all_data`, plus a pure-Python row parse of the 907 MB `all_shortest_paths.csv`, 3.6M rows). The shim caches the pristine post-`__init__` world (`DataCalculations`, `shortest_path_memory`) as a pickle at `%LOCALAPPDATA%/stdvrp/golden_world_cache.pkl` (override `STDVRP_GOLDEN_CACHE`; `--no-cache` to bypass), keyed by legacy sha256 + data-file sizes + horizon/congestion params + python/numpy/pandas versions. Warm golden verification ≈ 3 min (32 s unpickle + ~140 s episodes); cold ≈ 28 min. Cache must be written before any episode runs — episodes mutate `DataCalculations` (`total_congestions`, `unexpected_event_velocity`, …).
4. **Import quirk**: when the monolith is imported (rather than run as `__main__`), `__builtins__` is a dict, which breaks the literal `__builtins__.min(...)` at line 5717; the shim restores the builtins module into the legacy's namespace.
5. W has 19 features; first train episode (warm-up lr 1e-6) cost ≈ 36 k, subsequent ≈ 1–1.5 k; eval/test costs ≈ 0.8–2.5 k — useful orders of magnitude for sanity checks in tickets 07-09.
6. **CI was red since the ticket-05 push, unrelated to this ticket**: without a pinned Python, CI's `uv sync` provisioned Python 3.14, whose numpy wheel ships PEP 695 (`type X = ...`) stubs that mypy under `python_version = "3.11"` rejects with a fatal exit 2. Fixed by committing `.python-version` (3.11); full CI sequence verified green on Linux (WSL) before pushing.
