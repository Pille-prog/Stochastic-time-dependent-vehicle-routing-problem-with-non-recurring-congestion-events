# 07 — Congestion book: compose and purge

**What to build:** Two defects in how the congestion book is written and
maintained. Closes B7 and B17.

**Blocked by:** 01

**Status:** resolved

## B7 — a later event truncates an earlier one's expiry

`congestion/generator.py:88,131`. `congested_arcs` holds a single
`[multiplier, expiry]` pair per arc and **every write replaces both fields**.
Within one `generate()` call, a later event can leave an arc *less* congested
and — the dominant defect — end its congestion *earlier* than the still-active
event already recorded there.

**Measured:** per epoch, 25–30 arcs silently become faster when an event is
added, and **~1400 entries** (≈5% of the ~3300 congested arcs per epoch) see
their congestion end **up to 90 minutes early**. Scope is within a `generate()`
call, not across epochs.

- [x] On writing to an arc that is already congested and still active,
      **compose instead of replacing**: keep the more severe multiplier and the
      later expiry.
- [x] Invariant: a congestion write never shortens an active event's expiry, and
      never makes an active arc faster.

## B17 — the book never purges

`simulation/episode_velocities.py:71`. `congested_arcs` is only cleared at
episode end, so expired events stay in it. `any_congestion` is therefore
permanently `True` after the first roll, and the expiry scan walks the whole
book on every loop iteration.

**Measured:** the book reaches 116 arcs in the fixture, **all 116 already
expired**. The semantics are correct — `sample()` checks `tau >= event[1]` — so
this is compute cost and noise, not a physics error.

- [x] Purge expired entries.
- [x] Invariant: the book holds no expired entries after a clock advance.
- [x] Confirm no throughput regression. The `simulation-performance` effort
      closed at ~10×; purging should *help* (the scan shrinks), but measure
      rather than assume.

## Predicted self-golden diff

**Full divergence on every seed, in all three blocks, including frozen-W.**
Unlike ticket 03, this changes *exogenous information*: which arcs are congested
and for how long changes the velocities drawn, which changes travel times, which
changes arrival order — a decision-stable W does not protect against it. No
per-seed metric invariance is claimed.

**Direction: costs should rise, slightly.** B7 today ends congestion early on
~5% of congested arcs; retaining the correct expiry means more congestion,
therefore slower travel. B17 is semantically neutral (`sample()` already checks
expiry), so it should contribute **zero** to costs and show up only in
throughput.

That split is this ticket's falsifiable claim, and it is worth landing the two
fixes as separate commits so it can be checked: **B17 alone must produce a
bit-identical capture.** If purging a book whose entries were already being
filtered at read time changes a single cost, the purge removed something still
live.

## Evidence required

B17 alone: bit-identical capture, plus the throughput delta. B7: the 60-seed
bench before/after with direction, and a counter showing zero writes that
shorten an active expiry (was ~1400 entries per epoch). Book size after purge,
with the expired count at zero (was 116/116).

## Comments

### Resolution

Landed as two commits, per the ticket's own contingency clause, so B17's
zero-diff claim could be checked in isolation before B7 touched anything:

- **B17** — `EpisodeVelocities.purge_expired(tau_episode)`
  (`src/stdvrp/simulation/episode_velocities.py`) drops every
  `(arc, [multiplier, expiry])` pair with `tau_episode >= expiry`.
  `Model.advance_fleet_to` (`src/stdvrp/simulation/model.py`) calls it right
  after `self.state.tau_episode = minute` — that method is the single choke
  point every clock advance in the transition loop routes through (confirmed
  via `gitnexus impact`: `_congestion_expires`, `_episode_completes`,
  `_service_completes`, `_vehicle_parks_at_depot`, `_decision_epoch_begins`,
  `vehicle_reaches_client`, `vehicle_reaches_node` — all seven call sites go
  through it, none bypass it).
- **B7** — `_compose_congestion` (`src/stdvrp/congestion/generator.py`, module
  level) replaces both write sites (the direct trigger-arc write and the
  spread write, previously at `generator.py:88` and `:131`). Against an arc
  whose existing entry is still active as of `minute_start` (`expiry >
  minute_start`), it keeps `min(new_multiplier, existing_multiplier)` (more
  severe wins) and `max(new_expiry, existing_expiry)` (later wins); against an
  absent or already-expired entry it writes through unconditionally. This
  replaces the old spread-site guard (`generator.py:124-129`), which checked
  severity but not expiry and therefore still let a more-severe, shorter-lived
  event truncate a longer-lived one — and it extends the same protection to
  the direct-write site, which had no guard at all.

### Invariant tests (red before / green after, per the acceptance contract)

- **B7**: `tests/unit/test_congestion_generator.py` — two new example tests at
  the direct-write site (`test_primary_write_composes_with_a_stronger_active_existing_event`,
  `test_primary_write_never_shortens_an_active_expiry`), one for an
  already-expired existing entry (`test_an_expired_existing_event_is_fully_overwritten`),
  and the existing spread-site test strengthened to assert the full
  `[multiplier, expiry]` pair, not just that the multiplier changed
  (`test_weaker_existing_event_is_overwritten_but_keeps_the_later_expiry`).
  Plus a Hypothesis property in `tests/test_invariants.py`
  (`test_congestion_write_never_shortens_an_active_expiry_or_speeds_up`),
  pre-populating the book with arbitrary prior entries on the same node
  universe the generator draws from and asserting the invariant over every
  arc still active at write time. Verified failing pre-fix (temporarily
  reverting `generator.py` alone reproduces the shortened-expiry
  counterexample the ticket describes) and passing post-fix.
- **B17**: `tests/unit/test_episode_velocities.py::TestPurgeExpired` (four
  cases: removes an expired entry, keeps an active one, removes only the
  expired arcs from a mixed book, no-op on an empty book) and a per-clock-advance
  Hypothesis property added to `RecordingModel.advance_fleet_to` in
  `tests/test_invariants.py`, asserting no expired entries survive any single
  clock advance across the full Episode loop (not just a terminal snapshot,
  unlike ticket 01's `congestion_book_expired_at_end` counter). Verified
  failing before the `purge_expired` call was wired in, passing after.

### Self-golden (spec.md decision 9/10, "matches" outcome)

**B17 alone: bit-identical**, exactly as predicted —
`capture_self_golden.py --check` reported `worst relative deviation: 0.000e+00`
against the pre-ticket capture. The only thing that moved in the 60-seed bench
was `congestion_book_size_at_end`/`congestion_book_expired_at_end`; every cost
component, decision count, and violation counter matched per seed, per
config (`ticket07-before-*.txt` vs `ticket07-after-b17-*.txt`,
`.scratch/simulator-correctness/bench-output/`). This is the ticket's own
falsifiable check on B17: had a single cost moved, the purge would have been
removing something still live — it did not.

**B7: full divergence**, also as predicted — `final_w` and every training
seed moved (B7 changes exogenous congestion, which feeds the Monte Carlo
return from the first affected training episode onward), and most but not all
evaluation/frozen-W seeds moved. The ticket's own text anticipates this
("**No per-seed metric invariance is claimed**" — unlike tickets 03/04, no
seed is structurally protected, but a seed whose episode never samples a
composed arc can still land on identical numbers by coincidence, and a few
did). Re-captured (`tests/fixtures/self_golden/mini_fixture.json`);
`tests/test_self_golden.py` passes bit-exact against the new capture on this
environment (numpy 2.4.6, Python 3.11.9, Windows/AMD64 — matches the recorded
fingerprint).

**Direction, checked on the 60-seed bench** (the self-golden capture's ~15
seeds mix decision-path noise in both directions per-seed, exactly as the
ticket predicts — the aggregate direction claim is evidenced by the bench,
not the capture): `mean_total_cost` rose on both configurations —
512.194224 → 512.740181 (default fleet), 1831.729883 → 1949.168467
(`--vehicle-count 1`, where more congestion sticking around compounds against
the single vehicle's schedule far more). `mean_final_tau` rose in both too
(459.45 → 465.33, 822.46 → 846.70), consistent with "more congestion,
therefore slower travel."

### Evidence required, addressed

- **B17**: bit-identical capture (above) + throughput. Alternated
  `benchmark_episodes.py --train 20 --eval 20` before/after (3 rounds, purge
  toggled via `git stash` on `model.py` alone to isolate the comparison from
  this machine's run-to-run drift, per [[dev-environment-constraints]]-style
  precedent): the purge was at least as fast as no-purge in every round and
  faster in two of three, consistent with "the scan shrinks" — **no
  regression**.
- **B7**: 60-seed bench before/after with direction (above). **The
  `congestion_shorten_violations` counter (ticket 01's `BenchCongestionGenerator`)
  reads 0/60 on this mini fixture in every configuration tried — before *and*
  after the fix, matching ticket 01's own committed finding for this same
  counter.** This does not mean the bug was rare-to-nonexistent: a shorten
  violation needs two *different* triggered events (each with its own
  duration draw) to collide on the same arc within one epoch, and this
  fixture's small node universe apparently doesn't produce that collision in
  these 60 seeds at either `--vehicle-count` tried (matches ticket 01's noted
  pattern for B11/B17's headline figures — the review's "~1400/epoch" and
  "116/116" numbers came from a larger probe than this fixture reproduces).
  The fix is proven correct by the red-before/green-after unit and property
  tests above, which construct the colliding-write scenario directly rather
  than hoping a small fixture happens to produce one.
- **Book size after purge, expired count at zero**: confirmed on both bench
  configs with B17+B7 together — default fleet
  `final_book_size_total=5486→2958`, `expired_at_end_total=3064→0`;
  `--vehicle-count 1` `6277→3915`, `3158→0`.

### Methodology note: isolated worktree

This session found the shared working tree mid-flight on several other
tickets' uncommitted changes (08's BFS spread rewrite touching the same file
as B7's compose fix; 09's rename already landed as 061343e underneath this
ticket's start). Per [[git-pathspec-commit-stages-worktree]] and
[[stdvrp-refactor-status]]'s benchmarking note, all of this ticket's
development, TDD, self-golden capture and bench measurement ran in an
isolated `git worktree` at the last committed HEAD (`061343e`), so none of
the evidence above is contaminated by a concurrent session's in-flight,
uncommitted logic changes. The two commits land on the shared branch via
isolated-hunk staging (`git apply --cached` + a pathspec-free `git commit`),
never touching files under concurrent edit beyond this ticket's own hunks.

**Unrelated, pre-existing finding, not fixed here**: `tests/unit/test_monte_carlo_policy.py`
still calls `State(..., n_arcs=3, ...)` at four call sites after ticket 09's
commit renamed the parameter to `n_observed_velocities` — 20 failures,
reproduced identically with or without this ticket's changes (confirmed at
this ticket's clean `061343e` base, before any B7/B17 code existed). Another
concurrent session already had this exact file mid-fix (partially staged) in
the shared working tree at the time this was noticed; not this ticket's to
fix.
