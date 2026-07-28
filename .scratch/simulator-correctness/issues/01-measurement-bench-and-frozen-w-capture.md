# 01 — Measurement bench and the frozen-W capture block

**What to build:** The instrumentation every other ticket depends on, plus a
reproduction of the review's headline numbers at HEAD. No production behavior
changes.

The review's measurements exist today as ad-hoc `Model` subclasses pasted into a
report. This ticket turns them into a command, and adds the one capture block
that makes per-ticket diff attribution possible at all.

**Blocked by:** —

**Status:** open

- [ ] **Measurement bench** in `scripts/` — runs a fixed 60-seed set on
      `tests/fixtures/chengdu_mini` and reports, per seed and aggregated: mean
      total cost and its 4 components, km driven, final tau, decision count,
      **unserved-Client count split by in-window / overdue**, and a violation
      counter per invariant in the spec's catalogue. Reusable: tickets 02–09
      call it for their before/after evidence, so its output must be diffable
      text, not prose.
- [ ] **Frozen-W evaluation block** added to `scripts/capture_self_golden.py`:
      the same `EVAL_SEEDS`, run with a fixed literal W committed in the
      protocol (use the current `final_w`), written to the capture under a new
      key. The two existing blocks are **not touched**.
      `tests/test_self_golden.py` asserts the new block float-for-float like the
      others.
- [ ] **Diff reporter**: extend `--check` so it reports *what* changed per seed
      per metric, not only the worst relative deviation. This is the tool the
      three-outcome rule is enforced with — a bare "worst rtol 3e-2" cannot be
      checked against a prediction.
- [ ] **Reproduce the review at HEAD.** Confirm or refute, with numbers:
      - B1b: ~0.4 mid-arc teleports per episode; 24 of 60 episodes affected
      - B3: 15 of 60 single-vehicle episodes leave ≥1 in-window Client unpriced
        (116 Clients total)
      - B11: 734 duplicate-assignment transitions over 60 episodes
      - B17: the book reaches 116 arcs, all 116 expired
      Any finding that does not reproduce gets its ticket re-scoped or dropped
      **before** it costs work.
- [ ] **Size the blast radius of tickets 03 and 04 on the capture seeds.** All
      15 capture episodes terminate all-back between tau 396 and 502 — inside
      both B3's and B1a's regime. Report per capture seed how many Clients are
      left unserved and how many teleports fire, so tickets 03 and 04 can write
      honest predictions instead of guesses.

## Predicted self-golden diff

**Exactly zero on the two existing blocks.** This ticket adds a key to the
capture and changes no production code. The new frozen-W block has no prior
value to diff against.

If the existing blocks move, something in the bench leaked into production code
— revert and find it.

## Evidence required

The four reproduction numbers above, plus the per-capture-seed unserved and
teleport counts. Recorded in Comments.

## Contingency

If the bench trips B5's `min()` crash on any of the 60 seeds, pull B5's
two-line fallback forward into this ticket and note it here; ticket 05 then
carries only B11.

## Comments
