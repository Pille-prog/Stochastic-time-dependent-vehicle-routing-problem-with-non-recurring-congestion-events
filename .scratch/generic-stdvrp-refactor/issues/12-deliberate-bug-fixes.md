# 12 — Deliberate, documented bug fixes

**What to build:** Phase 2 begins: triage the defect list accumulated while porting (tickets 05–09 record every oddity they preserve), and fix each one as a deliberate behavior change with its own test.

**Blocked by:** 09, 10, 11.

**Status:** resolved

- [x] Every fix follows: failing test first → fix → entry in the ADR-0001 phase-2 change log
- [x] Golden master re-baselined only where a fix intentionally changes outcomes, with the justification recorded
- [x] Shadowed/dead legacy variants are NOT resurrected — they stay in the `legacy-monolith` tag (ADR-0001)
- [x] Candidate list triaged with the user before fixing (some "bugs" may be intended research behavior)

## Comments

**2026-07-23 — resolved.** The user triaged the full candidate list (every documented oddity from tickets 05–10 plus the ADR ticket-09 addendum) and approved all seven candidates; each landed as its own commit with a failing test first and an ADR-0001 phase-2 change-log entry (fixes 1–7 there, in order):

1. Single-observation speed std 0.0 instead of NaN (`_aggregate_speed_statistics` fillna; the legacy `dropna()` no-op).
2. The 420–540 std window stores the blend like the other two windows (the legacy discarded it).
3. Training episodes report their real distance cost (the legacy zeroed the accumulator per step; reporting only, W/rewards unchanged).
4. Warm-up learning rate made optional (`warmup_learning_rate: null` disables the episode-1 1e-6 quirk; existing configs keep it).
5. `terminate_state_if_all_vehicles_come_back` charges `(tau - due)` instead of the hardcoded `(1150 - due)`.
6. The terminating transition's cost counts once in `total_cost` (the legacy fell through the epoch-end gate after horizon termination and double-added it) — `total_cost` now equals the component sum up to float accumulation order.
7. Congestion spread walks the full `max_depth` (the legacy passed `max_depth - 1`; depth-3/0.73 damping was dead) and spread multipliers saturate at `congestion_upper_bound` (they reached `upper/0.78`).

**Re-baseline mechanism** (fixes 2, 6, 7 change outcomes; 1 and 5 only off the exercised paths): `scripts/rebaseline_golden_master.py` runs the legacy capture's protocol through the ported Trainer on the full local dataset and writes `tests/fixtures/golden_master/chengdu_full_phase2.json`; `tests/test_new_package_vs_golden_master.py` now compares against that file. The legacy capture `chengdu_full.json` stays frozen and `tests/test_golden_master.py` still re-verifies the monolith against it — both baselines are permanent evidence. 167 of the stored values differ between the two baselines; the congestion-depth fix dominates (episodes see more congested arcs, so costs/taus shift substantially — e.g. test seed 100 total 2342.5 → 4016.6). That outcome shift is the deliberately accepted research-behavior change.

**Vs-legacy CI gates survive until ticket 14 retires them**: the double-add is compensated exactly (`ported total + legacy final transition cost == legacy total`, bit-for-bit — `count_horizon_terminations` in `characterization_world.py`); the congestion change is not exactly compensable, so the episode comparisons inject the test-local `LegacySpreadCongestionGenerator` (pre-fix spread) and the fixed generator is covered by `tests/unit/test_congestion_generator.py::TestPhase2Fixes` plus the invariant suite. The 44-copy world's all-zero stds keep fixes 1–2 invisible there.

**For ticket 13 (RNG modernization):** the only remaining preserved randomness quirks are the global `random`/`np.random` streams and the config-nullable exploration seeding offsets; the deliberate-bug backlog is empty. `test_invariants.py` now asserts the clean identities (total == components via isclose — per-transition vs per-increment float accumulation order still differs; bit-exactness would need accumulating the total from the components, a candidate for the RNG-era rewrite). Note fixes shifted the derandomized Hypothesis examples: the pinned invariant examples no longer horizon-terminate, so a dedicated probed test (`test_horizon_terminated_episode_counts_the_terminating_transition_once`, seed 0, single vehicle) pins the double-add fix.

Verification: 248 passed non-golden (6 deselected), ruff + ruff format + mypy clean; `uv run pytest -m golden` green on the full dataset (legacy tests against the frozen capture, new-package tests against the phase-2 baseline).
