# 02 — Name the two clocks

**What to build:** Split the one config field that carries two concepts into two
named, validated, configurable fields — **without moving a single value**.
Closes B12, B15 and B16.

`horizon_end_minute` does three jobs (overtime threshold, latest window opening,
the Policy's homecoming guard) and does *not* do the one its name promises:
bound the episode. That is `EMERGENCY_HORIZON = 1150`, hardcoded in
`model.py:76` and repeated as a literal in the feature normalizer
`time_left = (1150 - tau) / 850` (`feature_extraction.py:197`).

These are two legitimately different things — a shift that ends (and after which
overtime accrues) and a hard stop — and the code already calls the first one
`shift_end_minute` internally (`model.py:141`). Only the config name lies.

**Blocked by:** 01

**Status:** resolved

- [x] `horizon_end_minute` → **`shift_end_minute`** in `ExperimentConfig` and
      every committed YAML. Same value (780), same three uses.
- [x] New **`episode_end_minute`**, default 1150. `EMERGENCY_HORIZON` and the
      feature normalizer's literal read it instead of hardcoding.
- [x] Validation `shift_end_minute <= episode_end_minute`. **This is what kills
      B15**: negative overtime becomes unconfigurable rather than merely
      unreached. Add the `tau > shift_end` guard in
      `terminate_state_passing_horizon` anyway, for symmetry with
      `_vehicle_parks_at_depot:306` — it is free.
- [x] **B16**: `_congestion_epoch_due` drops the `/60` division that produced
      the non-representable quotient. **Verify before asserting**: `tau_episode`
      is a float, so integer arithmetic only applies if decision epochs land on
      integer minutes. Establish that first; if they do not, the fix is
      "eliminate the division", not "use integers".
- [x] Invariant: no cost component is negative **for any valid config** — a
      parametrized run over several `shift_end_minute` / `episode_end_minute`
      pairs. The existing suite asserts `component_total >= 0` only at 780,
      which is exactly why B15 survived.
- [x] Invariant: congestion cadence fires
      `⌊(episode_end − horizon_start) / max_congestion_duration⌋` times, for
      every integer duration in a swept range — the review's 50 and 70 included,
      which today collapse to one roll.

## Predicted self-golden diff

**Exactly zero.** Renaming a field and exposing an identical default cannot move
a float. Every committed config keeps 780 and gets 1150, which is what
`EMERGENCY_HORIZON` already is.

**The one risky part is B16.** `(tau + 180 - 2)` is written as two operations
because it rounds twice and the Tier-1 gate is bit-exact (the existing docstring
says so). Any reassociation can move the firing set by a last-bit rounding. The
prediction stays **zero** — under `max_congestion_duration: 120` the review
measured the gate firing at the intended cadence — but if it is non-zero, the
mechanism is a float-rounding edge at a specific tau and that is outcome 2:
explain it to the tau, record it, do **not** edit this prediction.

## Evidence required

Zero diff on all three capture blocks. Bench output identical to ticket 01's
baseline. The B16 integrality finding stated explicitly.

## Comments

### Resolution (2026-07-30)

Landed on an isolated worktree — this same working tree had a concurrent
session mid-edit on nearly every file this ticket touches (tickets 07/08/09
in flight simultaneously) — so ticket 02 branched off ticket 09's commit
(`061343e`) rather than share the dirty tree; see
[[git-pathspec-commit-stages-worktree]].

`horizon_end_minute` → `shift_end_minute` + new `episode_end_minute` threaded
through `ExperimentConfig` → `ClientGenerator` / `episode_pool.episode_kwargs`
/ `episode.py`'s two runners → `MonteCarloPolicy` → `FeatureExtractor` /
`Model`. Every committed YAML keeps `shift_end_minute: 780` and gains
`episode_end_minute: 1150`. `Model.EMERGENCY_HORIZON` stays as a module
constant (measurement_bench.py and the invariant suite still read it as "the
value every shipped config uses") but is no longer read by `Model` itself —
`_decision_epoch_begins` now compares against `self.episode_end_minute`.
`FeatureExtractor._general_features`'s `time_left` normalizer reads
`self._episode_end_minute` instead of the literal `1150`.

**B15**: `ExperimentConfig.__post_init__` rejects `shift_end_minute >
episode_end_minute` outright. `terminate_state_passing_horizon` additionally
guards its overtime charge with `tau > shift_end_minute`, mirroring
`_vehicle_parks_at_depot`'s existing one — confirmed **not** merely
redundant-by-construction: with the guard removed, `tests/unit/
test_model_termination.py::TestPassingHorizonOvertimeGuard::
test_no_overtime_when_shift_end_outlives_the_actual_termination_clock`
reproduces the review's own number exactly, `overtime_cost ==
-43.333333333333336` at `tau=1148, shift_end_minute=1200` (docs/
simulator-review.md's B15 reproduction table: −43.33). `tests/
test_invariants.py`'s `episode_configs()` strategy now draws
`shift_end_minute` from `[HORIZON_START + 1, EMERGENCY_HORIZON]` instead of
fixing it at 780, so `test_episode_invariants`'s existing
`component_total >= 0` assertion is the parametrized-sweep invariant the
ticket asks for, exercising the previously-unseen 780–1150 range on every
Hypothesis run.

**B16 — the integrality finding, stated explicitly.** `next_decision_tau`
starts at `horizon_start_minute + DECISION_EPOCH_MINUTES` (both ints) and is
incremented only by the int `DECISION_EPOCH_MINUTES`, so it is always
integer-valued — established in ticket 01's `measurement_bench.py` docstring,
reused rather than re-derived. That makes the fix "eliminate the `/ 60`
division", not "switch to integer types" (there was never a float-vs-int
question, only a needless division). `_congestion_epoch_due` keeps
`(self.state.tau_episode + 180 - 2)` as the exact same two-operation
sub-expression (bit-for-bit unchanged, per the existing docstring's rounding
note) and compares it against the raw `self.max_congestion_duration` with
plain `%`, instead of dividing both sides by 60 and comparing against
`self.hours_max_duration` (removed; nothing else read it). New
`tests/unit/test_congestion_epoch_cadence.py` pins this against an
independent integer-arithmetic oracle over the review's tested/shipped
durations (30/45/60/90/120/150/180/200/240) plus the two it flags (50, 70).
**Correction to the review's own headline**: it names 50 and 70 as the
breaking durations, but 200 breaks too (measured: shipped fires 1, intended
17 at 50; shipped 2, intended 12 at 70; shipped 1, intended 4 at 200) — the
new test asserts all three. Confirmed red before the fix (temporarily
restored the pre-fix formula, reran: `50`/`70`/`200` failed with exactly
those shipped counts, everything else passed) and green after.

**Self-golden: exactly zero**, as predicted. `scripts/capture_self_golden.py
--check --rtol 1e-9` → `worst relative deviation: 0.000e+00 at (identical)`,
`per-seed per-metric diff: (identical - nothing moved)` on all three blocks
(training, evaluation, frozen-W). No re-capture needed — there is nothing to
overwrite when the diff is genuinely zero.

**Bench output identical to ticket 01's baseline**, all four saved runs
(`default-fleet`, `default-fleet-frozenw`, `vc1`, `capture-seeds`) — byte-
identical once compared modulo line endings (the committed baselines carry
CRLF from whatever wrote them originally; this session's fresh runs write LF,
per `measurement_bench.py`'s own `write_text(..., newline="\n")`; diffed with
`tr -d '\r'` on both sides to confirm, since a raw `diff` reports every line
as changed otherwise — a false alarm worth recording so nobody re-chases it).
Also verified this is not a ticket-02 artifact: a fresh checkout of ticket
01's own commit (`55f32aa`) reproduces the same CRLF-only "diff" against its
own committed file, and ticket 09's commit (`061343e`, the base this ticket
branched from) reproduces bench output identical to ticket 01's too — so the
whole `55f32aa..HEAD` range is behavior-preserving on this bench, not just
this ticket's slice of it.

`CONTEXT.md`'s **Horizon** definition corrected (it claimed all decisions and
events happen within `[horizon_start_minute, horizon_end_minute]`, which the
spec's own point-of-departure section already flagged as false — episodes
demonstrably run to 1148 with `shift_end_minute` at 780): now names both
clocks and states the gap explicitly.

Full non-golden suite: 3042 passed (golden-marked tests need the real
Chengdu dataset and skip without it, unaffected either way).
`ruff check` / `ruff format --check` / `mypy src/stdvrp` all clean.

One incidental fix, unrelated to this ticket's own scope but needed to run
the suite at all: `tests/unit/test_monte_carlo_policy.py` had four leftover
`State(..., n_arcs=3, ...)` call sites ticket 09's `n_arcs` →
`n_observed_velocities` rename missed (that file wasn't in ticket 09's diff).
Fixed as the same mechanical rename, verified via `git diff` against ticket
09's commit that this was pre-existing and not something this ticket's edits
caused.
