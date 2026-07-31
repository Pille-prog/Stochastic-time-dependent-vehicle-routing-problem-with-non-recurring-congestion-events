---
status: accepted
---

# The episode clock and the price of abandoned demand

The simulator-review's B3 (`docs/simulator-review.md`): `_charge_unserved_delays`
priced every Client an Episode ended without serving at `tau_episode - due`,
the exact same formula `charge_lateness` uses for a Client the fleet is still
trying to reach. That formula, `cost(t) = max(0, t - due)`, is correct while
the Episode runs — a pending Client's window might still be met, so pricing it
at the live clock reflects an outcome that is genuinely still open. Termination
is a different event: the outcome is closed, and the Client will never be
served. Reusing the live formula unchanged made the price of abandonment a
function of *when the fleet happened to stop*, not of the Client it abandoned.
Ticket 01's measurement bench sized the defect: the same 19 abandoned Clients
in the review's seed 14 cost 0.00 or 11 245.00 depending only on which
termination path fired and at which clock — four orders of magnitude apart for
an identical demand set. No other correctness fix in the simulator-correctness
effort can be measured against a cost function that unstable, which is why B3
is in scope despite being closer to a modeling decision than a code defect
(spec.md, decision 2): it is the yardstick, not a measurement.

**Decision.** Every unserved Client is priced at Episode termination as:

```
cost = max(0, max(episode_end_minute, tau_episode) - due)
```

against a fixed **reference clock**, `max(episode_end_minute, tau_episode)`,
rather than `tau_episode` alone. Both termination call sites
(`terminate_state_passing_horizon`, `terminate_state_if_all_vehicles_come_back`)
only ever fire with `tau_episode <= episode_end_minute` — the Model
force-terminates there (ticket 02, simulator-correctness, B12) — so in every
reachable case the `max` collapses to the constant `episode_end_minute`: the
termination charge becomes a pure function of `due` and config, never of
`tau_episode`. That is deliberate and is the whole point — it is what makes
the charge comparable across episodes that happen to stop at different clocks,
which the live formula could never be.

The live in-episode formula is untouched. `charge_lateness` still prices a
Client actually served late at the clock it was served at; only
`_charge_unserved_delays` — the termination event — moves to the fixed clock.

## Considered options

- **`shift_end_minute` (780) as the reference clock.** Rejected: under it,
  serving a Client at 900 costs 120 while abandoning one due at 700 costs 80 —
  abandoning stays *cheaper* than serving, and `best_w` is selected by minimum
  mean cost, so a policy that abandons demand would look better than one that
  serves it. Any reference clock at or before a Client's actual achievable
  service time reopens this hole; `shift_end_minute` is simply the shipped
  value where the review measured it.
- **A fixed no-service charge**, independent of `due`. Conceptually cleaner —
  lost demand is not the same kind of event as a late arrival — but it
  introduces a free parameter with nothing in this simulator to calibrate it
  against, and moving that number moves the optimum it trains toward. Recorded
  here as a deliberate extension for the separate modeling effort
  (spec.md, decision 1's split between "does what it says" and "should it say
  that"), not adopted now.
- **`episode_end_minute` (chosen).** By construction the maximum clock any
  Episode can reach, so abandoning is never cheaper than serving however late;
  introduces no new parameter (`episode_end_minute` already exists, ticket 02);
  and leaves horizon-terminated episodes essentially unmoved — the predicted
  self-golden diff turns `1148 - due` into `1150 - due`, a couple of minutes'
  difference — concentrating the entire behavior change on the all-back
  termination path, which is where the defect actually lived (ticket 01: every
  self-golden episode in the capture terminates all-back between tau 396 and
  502, none reaching `shift_end_minute` let alone `episode_end_minute`).

## Consequences

- **B14 follows from the same fix.** `CostLedger.charge_unserved_delays` and
  `charge_fleet_overtime` now count exactly the Clients/vehicles they charge a
  strictly positive amount to, retiring the preserved legacy quirk of counting
  neither. `unserved_clients` is its own counter, never folded into
  `late_clients`: under the new clock every abandoned Client generates a
  charge, so lumping the two together would conflate "served late" with
  "never served" — two different outcomes this ADR exists to distinguish (see
  `CONTEXT.md`, "Unserved Client"). `EpisodeResult` and the self-golden capture
  gain `unserved_clients` as a new metric key.
- **The statistical gate retires** (spec.md, decision 8). The three ±40%
  tolerance tests in `tests/test_new_package_vs_golden_master.py` compared the
  new package's mean cost against `chengdu_full_phase2.json`, a baseline
  captured under the old (buggy) termination formula. A baseline computed
  under a different cost function cannot gate a run under this one — that
  comparison is now noise wearing the costume of a guarantee, so the file is
  deleted outright. `chengdu_full_phase2.json` stays in the repo as historical
  evidence of the new package's pre-ticket-13 behavior; no test reads it.
  `tests/test_golden_master.py` (the legacy monolith against its own frozen
  capture) is unaffected — it never imports `stdvrp`.
- **The self-golden diff, measured, turned out to be zero — for a reason
  outside this fix.** The predicted diff (frozen-W block: `delay_cost` and
  `total_cost` rise on every seed with unserved Clients; training/evaluation
  diverge fully) assumed ticket 01's point-of-departure measurement, taken
  before ticket 08 (breadth-first congestion spread, B9) landed: every one of
  the 15 self-golden episodes served *every* Client by the time this ticket
  ran — `unserved_clients` is 0 everywhere in the recapture — so
  `_charge_unserved_delays` iterates an empty collection on all 15 and the fix
  is numerically inert on this specific fixture. Confirmed by an isolated
  before/after comparison (both freshly captured on identical code, differing
  only in this ticket's changes): zero leaves moved. This is spec.md decision
  10's "explained to a mechanism" outcome, not "matches" or "unexplained" —
  the fix still lands, verified instead on the 60-seed bench with
  `--vehicle-count 1` (ticket 01's own probe for this exact demand), where 42
  of 60 episodes do have unserved Clients: `mean_delay_cost` moves 1713.3 ->
  3072.7, `mean_total_cost` by the identical amount, and the other three
  components, `mean_final_tau` and `mean_decisions` stay bit-identical, per
  the frozen-W invariant. That run also measured B14 directly:
  `money_without_counter` violations 33 (31/60 episodes) before this fix, 0
  after.
- **The recapture also lands ticket 08's own dangling gap.** Ticket 08's
  commit (`c900423`) changed simulator behavior (which arcs congest) without
  committing a matching self-golden recapture — a pre-existing, already
  memory-documented gap this ticket did not create. Adding the
  `unserved_clients` key requires a recapture regardless, and any recapture
  from this commit onward necessarily reflects ticket 08's already-merged code
  too. The 177 leaves that move against the previously-committed fixture are
  therefore ticket 08's effect, not this ticket's — confirmed by the same
  before/after comparison above, which isolates this ticket's own contribution
  as exactly zero.
