# 04 — The State says where the vehicle is

**What to build:** Fix the review's common root cause — `State.vehicle_position`
does not mean "where the vehicle is" — in the domain model rather than at the
call sites. Closes B1a and B1b. Writes **ADR-0005**.

`vehicle_position` holds **the last node the vehicle reached**.
`vehicle_reaches_node` (`model.py:517`) writes it for a node the vehicle merely
*passes through*, and `begin_arc` immediately launches it onto the next arc.
Nothing distinguishes "standing at the node" from "crossing it". `state.py:32`
already half-admits it in a comment: *"Node each vehicle last departed from (or
is at)"* — that "(or is at)" is the ambiguity, and it is why the same wrong
predicate got written at three sites.

It matters because the depot is not only the starting point: 138 of 2025 cached
shortest paths (6.8%) use it as an interior node, and the review measured 75
depot crossings in 60 episodes.

**Two layers, two fixes, and fixing one does not cure the other:**

| | Where | What breaks | If only this one is fixed |
|---|---|---|---|
| B1a | `policies/monte_carlo.py:297` | The fleet loses capacity — up to half the fleet idle with Clients pending | The vehicle keeps working and layer 2 stops being reachable by this route |
| B1b | `simulation/model.py:376` | Physics and accounting — the vehicle is teleported to the depot, remaining arc and return trip never charged | The vehicle finishes the arc, drives back and parks — it pays the trip, but **still leaves service** |

**Blocked by:** 03 (so the impact is measured under the repaired objective)

**Status:** open

- [ ] **Rename** `State.vehicle_position` → an honest name (`last_node_reached`
      or similar). This is half the fix: it makes
      `if vehicle_position == depot` un-writable by anyone who means "is at the
      depot". Six read sites — audit each for which of the two meanings it
      wants.
- [ ] **Add the missing fact to `State`**, maintained by the `Model` wherever it
      already writes the position: whether the vehicle is standing at a node
      rather than travelling. The Policy reads it.
      **Do not hand `FleetRoutes` to the Policy** — `fleet_routes.py:11-15`
      deliberately keeps the route/progress half invisible to the Policy
      (simulation-performance ticket 09). The boundary is right; the State was
      simply missing a fact on the correct side of it.
- [ ] **The two dead fields.** `State.vehicle_next_node` (written
      `model.py:281`) and `State.vehicles_direction` (written `model.py:406`)
      are written and **read by nobody** — vestigial slots for exactly this kind
      of information. Either the new fact lands in one of them or both go.
      Do not leave a third dead field behind.
- [ ] **B1a**: the depot-idle rule applies only to a vehicle genuinely standing
      at the depot. **The literal 350 is not touched** — it stays the documented
      legacy quirk (`monte_carlo.py:62-64`). What changes is what it applies to.
- [ ] **B1b**: the same distinction guards `model.py:376`. The branch is correct
      and frequent in its normal case (a vehicle actually parked at the depot
      told to stay); it is only a bug for a vehicle mid-arc.
- [ ] Invariant: **no vehicle becomes `PARKED` while
      `departure_tau < tau < arrival_tau`** (the review's own proposal).
- [ ] Invariant: a vehicle's recorded node never changes without charged
      distance — no teleporting.
- [ ] **`TrainingSnapshot` audit**: it copies four State fields for `update_W`'s
      replay. Establish whether the replay path reads the new fact; add it if so.
- [ ] **ADR-0005** — the State says where the vehicle is. Records the rejected
      alternative (passing `FleetRoutes` to the Policy: breaks the ticket-09
      boundary and leaves the ambiguous predicate alive at four other sites) and
      the naming decision. **`CONTEXT.md`** gains the distinction: "vehicle
      position" today means two different things depending on who reads it.

## Predicted self-golden diff

**Not surgical, and that is expected** — unlike ticket 03, this changes
*decisions*: a vehicle that used to be retired now keeps working, so its whole
trajectory changes. The falsifiable claims are these:

- **Capture seeds with zero teleports and zero spurious depot-idle firings must
  be bit-identical, in all three blocks.** Ticket 01 reports those per-seed
  counts; list the untouched seeds here before running.
- On affected seeds, in the frozen-W block: `distance_cost` **rises** (the
  remaining arc and the return trip are now driven and charged), `tau`
  **extends**, `state_count` **rises**. A fall in any of the three contradicts
  the mechanism.
- Aggregate direction over the 60-seed bench: the review measured **+1.24% mean
  cost and +0.19 km/episode recovered** over 180 episodes (1060.27 → 1073.39,
  34.14 km). Reproduce that magnitude or explain the difference — note it was
  measured under the *old* objective, so ticket 03 shifts the denominator.

**Do not expect the fleet-capacity win to show up as a cost reduction.** Fixing
B1b alone makes the vehicle pay for its trip home without returning it to
service; only B1a puts it back to work. Both land here, so the net is a fleet
that works longer *and* pays for its distance — the review's honest reading is
that the important thing is the physics and the accounting, not the number.

## Evidence required

The untouched-seed bit-identity list. Per-seed direction of `distance_cost`,
`tau`, `state_count` on affected seeds. The 60-seed bench before/after with km
driven. Teleport counter at zero after the fix (it was ~0.4/episode, 24 of 60
episodes).

## Comments
