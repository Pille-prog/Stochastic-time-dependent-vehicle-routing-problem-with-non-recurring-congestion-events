# 13 — The action set becomes shared code

**What to build:** Extract `MonteCarloPolicy._select_vehicle_possible_actions`
and the two collaborators it reaches into a stateless module both Policies
call. **No behaviour change on either side.** This is the only ticket in the
effort that edits `monte_carlo.py`, and the exactly-zero self-golden diff is
the proof, not a formality.

**Blocked by:** — (pure extraction; nothing depends on it landing first)

**Status:** resolved

## Why the baseline's file is touched at all

Ticket 14 gives this Policy the **identical** action set, reversing ADR-0007
(the evidence is in 14, not here). "Identical" is only true if there is one
definition. The alternatives were weighed and rejected:

- **Duplicate it in `transformer_policy.py`,** as `_already_acquired_cost` was.
  That precedent is ten lines of arithmetic over two constants. This is ~65
  lines across three branches, a delayed-Client classifier, a lexsort helper
  and the `StateFeatures` plumbing. Two copies of that drift — and not
  drifting is the entire value of ticket 14's decision.
- **Compose:** hold a `MonteCarloPolicy` instance purely to ask it for
  candidates. Drags `W`, the exploration RNG and a rebound `self.state` along
  for one method call, and needs its `self.action` kept in sync every epoch. It
  converts a visible coupling into a hidden one.

spec.md's "Out of scope" lists **changing** the linear baseline. An extraction
with a zero diff does not change it, and the landing gate already anticipates
exactly this case: *"if a protocol extraction moves a single float, it was not
an extraction."*

## What moves

Into `src/stdvrp/policies/action_set.py` — no torch, no state, arguments in and
a list out:

```
select_vehicle_possible_actions(
    number_of_actions, vehicle, features, state, current_action,
    geometry, depot, number_vehicles, shift_end_minute) -> list[int]
```

with `_closest_allowed_clients` and `_classify_shortest_distance_clients`
beside it. `self.state` becomes the `state` argument, `self.action` becomes
`current_action`, `self.end_of_horizon` becomes `shift_end_minute`.

`MonteCarloPolicy._select_vehicle_possible_actions` stays as a thin method that
delegates — the name is referenced by its own module docstring, by ADR-0001's
change log and by tickets 06/08/09, and orphaning it would cost more than it
saves.

**Nothing is cleaned up on the way through.** The `350` and the `310` that
disagree by 40 minutes, the `list(set(...))` dedup, the duplicate-append quirk
in `clasify_delayed_clients`, the `< 3 clients` branch — all preserved exactly.
ADR-0001's rule holds: fix what crashes, never re-tune what is tuned. A tidy-up
here would silently move the opponent this whole effort is measured against.

## The one real hazard

`possible_actions = list(set(possible_actions))` depends on CPython's set
iteration order for these int node ids — deterministic in-process, and
`monte_carlo.py`'s own docstring already records it as a preserved quirk. It
stays deterministic **only if the same ints are inserted in the same order**,
so the extraction must not reorder the appends before the dedup.

- [x] A test that pins the returned list **element by element, in order**, for
      a fixture state with a duplicate-producing candidate set — not just set
      equality, which is exactly the assertion that would let this regress.

## Work

- [x] Extract, delegate, run `test_self_golden.py`.
- [x] Unit tests for the new module covering all three branches: the
      parked-at-depot/`tau > 350` branch, the `< 3 clients` classifier branch,
      and the normal branch with its depot append and its `delayed_clients`
      append.
- [x] `mypy` clean, `ruff` clean.

## Acceptance

- [x] Predicted self-golden diff: **exactly zero.** This ticket is the one
      where that number carries real information — every other ticket in the
      effort predicts zero because it never touches the path at all.
- [x] The ordering test above exists and fails if the appends are reordered.

## Comments

Landed as planned: `select_vehicle_possible_actions` plus `_closest_allowed_clients`
and `_classify_shortest_distance_clients` moved into
`src/stdvrp/policies/action_set.py`, verbatim, with the exact signature this
ticket specified. `MonteCarloPolicy._select_vehicle_possible_actions` stays as
a thin delegate.

`tests/test_self_golden.py` passes; `mypy`/`ruff` clean; full suite
(`uv run pytest`, i.e. `pytest` with `testpaths=["tests"]`) — 4273 passed, 3
`golden`-marked deselected as configured, 0 failed. New coverage:
`tests/unit/test_action_set.py` (three branches of `select_vehicle_possible_actions`,
plus the dedup-ordering hazard test, pinned `result == [10, 1, 2, 3]` — not
set equality); `tests/unit/test_monte_carlo_policy.py`'s pre-vectorization
oracle (`reference_possible_actions`) repointed at
`action_set._classify_shortest_distance_clients` since the instance method it
called no longer exists.

One trade-off surfaced by review and now documented in `action_set.py`'s
module docstring rather than left silent: simulation-performance ticket 07's
cross-vehicle cache (`_RemainingClients`/`_remaining_clients_cache`, which
memoized `clients_not_visited` as sortable arrays across one decision pass's
vehicles) has nowhere to live in a stateless module and is not preserved —
`_closest_allowed_clients` rebuilds `column_positions` every call now. Output
is bit-identical (confirmed by the zero self-golden diff); this is a
performance cost, not a behaviour change, and the ticket's own "Compose"
alternative was rejected for exactly the kind of hidden coupling reintroducing
that cache here would require.
