# 08 — Breadth-first spread

**What to build:** `_reachable_nodes` (`congestion/generator.py:136-164`) calls
itself "BFS-by-recursion" in its docstring, but the recursion is **depth-first
with a single `visited` set**. A node first discovered down a deep branch keeps
that depth, receives the wrong damping factor, and — if its recorded depth
reaches `max_depth` — stops expanding. Closes B9.

**Measured:** **~15% of the nodes genuinely within `max_depth`** of an epicentre
are never congested, and *which* arcs the spread reaches depends partly on the
**row order of the arc table** in `successors` rather than on network topology.
It runs twice per triggered event (once per endpoint), ~64 events per epoch, ~8
epochs per episode.

**Blocked by:** 01

**Status:** open

- [ ] Real breadth-first traversal: every node is recorded at its **minimum**
      depth from the epicentre.
- [ ] Invariant: every node within `max_depth` is congested, at its true minimum
      depth, and **the result is independent of arc-table order** — shuffle
      `successors` and assert the same congested set. That second half is what
      pins the actual defect; depth correctness alone would not catch an
      order-dependent traversal.

## Predicted self-golden diff

**Full divergence on every seed, in all three blocks, including frozen-W** —
same reason as ticket 07: this changes which arcs are congested, hence the
velocities drawn.

**Direction: costs rise.** ~15% more nodes fall inside each event's blast radius,
so more of the network is slow, so travel takes longer.

**One thing that must *not* show up.** The review's caveat: under the shipped
`0.3`/`0.4` bounds the damping factor is invisible because multipliers saturate
(B8), so this fix changes **which** arcs are congested, not **by how much**. The
distribution of multiplier *values* across the book should therefore be
essentially unchanged — still ~3/4 of congested arcs sitting at exactly 0.4
regardless of hop distance. If the multiplier distribution shifts, this fix
touched damping, which is B8's territory and out of scope for this effort.

## Evidence required

The order-independence invariant green. Node coverage before/after (was ~85% of
in-range nodes). The multiplier-value distribution before/after, showing it did
**not** move. The 60-seed bench before/after with direction.

## Comments
