# 07 — Congestion book: compose and purge

**What to build:** Two defects in how the congestion book is written and
maintained. Closes B7 and B17.

**Blocked by:** 01

**Status:** open

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

- [ ] On writing to an arc that is already congested and still active,
      **compose instead of replacing**: keep the more severe multiplier and the
      later expiry.
- [ ] Invariant: a congestion write never shortens an active event's expiry, and
      never makes an active arc faster.

## B17 — the book never purges

`simulation/episode_velocities.py:71`. `congested_arcs` is only cleared at
episode end, so expired events stay in it. `any_congestion` is therefore
permanently `True` after the first roll, and the expiry scan walks the whole
book on every loop iteration.

**Measured:** the book reaches 116 arcs in the fixture, **all 116 already
expired**. The semantics are correct — `sample()` checks `tau >= event[1]` — so
this is compute cost and noise, not a physics error.

- [ ] Purge expired entries.
- [ ] Invariant: the book holds no expired entries after a clock advance.
- [ ] Confirm no throughput regression. The `simulation-performance` effort
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
