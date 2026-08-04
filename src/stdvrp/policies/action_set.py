"""select_vehicle_possible_actions: the candidate action set, shared code.

Ticket 13 (neural-policy, ADR-0011): extracted verbatim from
``MonteCarloPolicy._select_vehicle_possible_actions`` and the two collaborators
it reaches (``_closest_allowed_clients``, ``_classify_shortest_distance_clients``),
so both the linear baseline and the transformer Policy (ticket 14) call one
definition instead of drifting copies. Stateless: every argument the legacy read
off ``self`` — ``self.state``, ``self.action``, ``self.geometry``, ``self.depot``,
``self.number_vehicles``, ``self.end_of_horizon`` — is now a parameter, and
nothing here is cached across calls.

**No behaviour change.** Nothing is cleaned up on the way through (spec.md
decision 4's amendment, ADR-0001): the ``350``/``310`` depot-idle literals that
disagree by 40 minutes, the ``list(set(...))`` dedup, the duplicate-append
quirk in the delayed-Client classifier and the ``< 3`` clients branch are all
preserved exactly, quirks and all — see ``monte_carlo.py``'s module docstring
for the full catalogue, which still applies verbatim to the code that moved
here.

**The one real hazard.** ``possible_actions = list(set(possible_actions))``
depends on CPython's set iteration order for these int node ids — deterministic
in-process, but only if the same ints are inserted in the same order. The
extraction must not, and does not, reorder the appends before the dedup.

**What is not preserved: simulation-performance ticket 07's cross-vehicle
cache.** ``MonteCarloPolicy`` used to memoize ``clients_not_visited`` as sortable
arrays (``_RemainingClients``, keyed by content so it survived unchanged across
every vehicle of one decision pass) purely to avoid rebuilding
``geometry.column_positions(...)`` once per vehicle. A stateless module has
nowhere to hold that cache without reintroducing the hidden coupling this
ticket's own "Compose" alternative was rejected for (see the ticket's "Why the
baseline's file is touched at all"), so :func:`_closest_allowed_clients` now
rebuilds it on every call. Output is bit-identical — this is a performance
cost, not a behaviour change — but it is a real one, undocumented here until
now: one more array build and column lookup per vehicle per decision epoch
instead of one per decision epoch total.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from stdvrp.simulation.state import is_parked_at_depot

if TYPE_CHECKING:
    from stdvrp.network.episode_geometry import EpisodeGeometry
    from stdvrp.policies.feature_extraction import StateFeatures
    from stdvrp.simulation.state import State

__all__ = ["select_vehicle_possible_actions"]


def select_vehicle_possible_actions(
    number_of_actions: int,
    vehicle: int,
    features: StateFeatures,
    state: State,
    current_action: list[int],
    geometry: EpisodeGeometry,
    depot: int,
    number_vehicles: int,
    shift_end_minute: int,
) -> list[int]:
    """Ports the live ``select_vehicle_possible_actions`` (per-vehicle) definition."""
    possible_actions: list[int] = []
    forbidden_actions = []

    for v in range(number_vehicles):
        if v == vehicle:
            continue
        else:
            forbidden_actions.append(current_action[v])

    # This 350 and the 310 in _classify_shortest_distance_clients below disagree
    # by 40 minutes (monte_carlo.py's module docstring, "Depot-idle cutoffs");
    # both literals stay as-is (spec.md decision 3: fix what crashes or
    # misclassifies, never re-tune what is tuned).
    if (
        is_parked_at_depot(state.last_node_reached[vehicle], state.vehicle_standing[vehicle], depot)
        and state.tau_episode > 350
    ) or len(state.clients_not_visited) == 0:
        possible_actions.append(depot)

    elif len(state.clients_not_visited) < 3:
        # B11: filter forbidden_actions here too, matching the normal branch
        # below — otherwise two vehicles can both be offered (and both pick)
        # the same Client this classifier assigns to more than one of them.
        shortest_distance_clients = _classify_shortest_distance_clients(state, geometry, depot)
        for _distance, client in shortest_distance_clients[vehicle]:
            if client not in forbidden_actions:
                possible_actions.append(client)
        if not possible_actions:
            possible_actions.append(depot)

    else:
        possible_actions = _closest_allowed_clients(
            state.last_node_reached[vehicle], number_of_actions, forbidden_actions, state, geometry
        )

        possible_actions = list(set(possible_actions))

        if (
            geometry.average_minutes(state.last_node_reached[vehicle], depot) + state.tau_episode
            > shift_end_minute
        ):
            possible_actions.append(depot)

        for delayed_client in features.delayed_clients[vehicle]:
            if delayed_client not in possible_actions and delayed_client not in forbidden_actions:
                possible_actions.append(delayed_client)

        if len(possible_actions) == 0:
            possible_actions.append(depot)

    return possible_actions


def _closest_allowed_clients(
    position: float,
    number_of_actions: int,
    forbidden_actions: list[int],
    state: State,
    geometry: EpisodeGeometry,
) -> list[int]:
    """The unvisited Clients closest to ``position``, nearest first, at most k of them.

    One geometry row slice plus one ``np.lexsort`` (simulation-performance
    ticket 07, ADR-0003): the ordering is identical to the pre-vectorization
    ``heapq.nsmallest`` over ``(travel time, Client)`` tuples this ports —
    lexsort's primary key is the travel time and its secondary the Client id,
    exactly how tuple comparison broke float ties.

    Forbidden Clients (the other vehicles' current actions) are dropped
    *after* the sort: the ``k + len(forbidden)`` nearest contain at least ``k``
    allowed ones whenever the filtered set has that many, so the first ``k``
    survivors are the same list filtering first would give.

    The returned ids are the State's own Python ints, never numpy scalars:
    they flow into the caller's ``action`` and from there into the Model.
    """
    clients = list(state.clients_not_visited)
    client_ids = np.asarray(clients)
    column_positions = geometry.column_positions(clients)
    travel_times = geometry.average_minutes_at(position, column_positions)
    nearest = np.lexsort((client_ids, travel_times))[: number_of_actions + len(forbidden_actions)]

    closest: list[int] = []
    for index in nearest:
        if len(closest) == number_of_actions:
            break
        client = clients[index]
        if client not in forbidden_actions:
            closest.append(client)
    return closest


def _classify_shortest_distance_clients(
    state: State, geometry: EpisodeGeometry, depot: int
) -> defaultdict[int, list[tuple[float, int]]]:
    """Ports ``clasify_shortest_distance_clients`` (endgame with < 3 Clients left).

    Stays scalar on purpose (simulation-performance ticket 07): with one or two
    Clients left its arrays are two elements wide, where numpy's per-call
    overhead measured 2x slower than these loops.
    """
    shortest_distance_clients: defaultdict[int, list[tuple[float, int]]] = defaultdict(list)

    clients_remaining = len(state.clients_not_visited)
    last_node_reached = state.last_node_reached
    vehicle_standing = state.vehicle_standing

    if clients_remaining == 2:
        # 310 here, 350 in select_vehicle_possible_actions above — the
        # disagreement documented there. heapq.nsmallest(2, []) == [] below if
        # this filters out every vehicle, so this branch never hits B5's
        # empty-min() crash (that's the one-Client branch just below).
        # ``vehicle_standing`` guards the same depot-idle predicate as the
        # 350 branch above — a vehicle mid-arc past the depot must stay
        # eligible.
        vehicle_distances = []
        for vehicle_idx, node in enumerate(last_node_reached):
            if (
                is_parked_at_depot(node, vehicle_standing[vehicle_idx], depot)
                and state.tau_episode > 310
            ):
                continue

            total_distance = sum(
                geometry.average_minutes(node, client) for client in state.clients_not_visited
            )
            vehicle_distances.append((total_distance, vehicle_idx))

        closest_two_vehicles = heapq.nsmallest(2, vehicle_distances)

        for _, vehicle_idx in closest_two_vehicles:
            for client in state.clients_not_visited:
                travel_time = geometry.average_minutes(last_node_reached[vehicle_idx], client)
                shortest_distance_clients[vehicle_idx].append((travel_time, client))

    elif clients_remaining == 1:
        client = next(iter(state.clients_not_visited))
        distances = []
        for vehicle_idx, node in enumerate(last_node_reached):
            if (
                is_parked_at_depot(node, vehicle_standing[vehicle_idx], depot)
                and state.tau_episode > 310
            ):
                continue

            travel_time = geometry.average_minutes(node, client)
            distances.append((travel_time, vehicle_idx))

        # B5: every vehicle can read position == depot with tau in (310, 350]
        # (see the disagreement noted above), leaving `distances` empty. Fall
        # back to the depot, exactly as the two-Client branch above already
        # does via heapq.nsmallest(2, []) == [].
        if distances:
            closest_vehicle = min(distances)
            assigned_vehicle_idx = closest_vehicle[1]
            shortest_distance_clients[assigned_vehicle_idx].append((closest_vehicle[0], client))

    return shortest_distance_clients
