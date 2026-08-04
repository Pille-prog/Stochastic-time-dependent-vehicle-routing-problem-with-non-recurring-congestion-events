"""Unit tests for ``stdvrp.policies.action_set`` (ticket 13, neural-policy).

``MonteCarloPolicy``'s existing tests (``test_monte_carlo_policy.py``) already
pin this code's behaviour end to end, through the Policy's delegating method —
they keep passing unmodified after the extraction, which is the "no behaviour
change" half of this ticket's proof. These tests instead exercise the shared
module directly, one branch of ``select_vehicle_possible_actions`` at a time,
plus the one hazard the extraction singles out: the ``list(set(...))`` dedup's
dependence on insertion order.
"""

import numpy as np

from stdvrp.network.episode_geometry import EpisodeGeometry
from stdvrp.network.shortest_path_cache import ShortestPath, ShortestPathCache
from stdvrp.policies import action_set
from stdvrp.policies.feature_extraction import StateFeatures
from stdvrp.simulation.state import State

DEPOT = 0
SHIFT_END = 780.0


def make_cache(arcs: dict) -> ShortestPathCache:
    """arcs: (node, node) -> (average_minutes, length_km)."""
    return ShortestPathCache(
        {
            key: ShortestPath([float(key[0]), float(key[1])], minutes, length)
            for key, (minutes, length) in arcs.items()
        }
    )


def make_state(
    *, clients: list[int], tau: float, position: float, standing: bool, vehicles: int = 1
) -> State:
    state = State(
        number_vehicles=vehicles,
        clients=list(clients),
        n_observed_velocities=3,
        horizon_start_minute=300,
        depot=DEPOT,
    )
    state.tau_episode = tau
    state.last_node_reached[:] = [position] * vehicles
    state.vehicle_standing[:] = [standing] * vehicles
    return state


def make_features(delayed_clients: tuple[tuple[int, ...], ...]) -> StateFeatures:
    """A ``StateFeatures`` carrying only ``delayed_clients`` — the sole field
    ``select_vehicle_possible_actions`` reads off it; the rest are unused filler.
    """
    vehicles = len(delayed_clients)
    return StateFeatures(
        general=np.zeros(12),
        delayed_clients=delayed_clients,
        mean_velocities=tuple(0.0 for _ in range(vehicles)),
        tau=0.0,
        active=np.zeros(1, dtype=np.bool_),
        late=np.zeros(1, dtype=np.bool_),
        vehicle_minutes=np.zeros((vehicles, 1)),
        vehicle_length=np.zeros((vehicles, 1)),
        closest_client_counts=np.zeros((vehicles, 1), dtype=np.int64),
    )


def no_delayed_clients(vehicles: int = 1) -> StateFeatures:
    return make_features(tuple(() for _ in range(vehicles)))


# --- Branch 1: parked at the depot past tau=350, or nothing left to serve ------


class TestParkedAtDepotBranch:
    def test_parked_past_350_returns_only_the_depot(self):
        cache = make_cache({(0, 0): (0.0, 0.0), (0, 1): (10.0, 5.0), (0, 2): (20.0, 8.0)})
        geometry = EpisodeGeometry.build(cache, clients=[1, 2], depot=DEPOT)
        state = make_state(clients=[1, 2], tau=360.0, position=DEPOT, standing=True)

        result = action_set.select_vehicle_possible_actions(
            2, 0, no_delayed_clients(), state, [DEPOT], geometry, DEPOT, 1, SHIFT_END
        )

        assert result == [DEPOT]

    def test_no_clients_remaining_returns_only_the_depot_regardless_of_tau_or_position(self):
        cache = make_cache({(0, 0): (0.0, 0.0)})
        geometry = EpisodeGeometry.build(cache, clients=[], depot=DEPOT)
        # Not parked at the depot and tau below the 350 cutoff: only the
        # zero-remaining-Clients half of branch 1's ``or`` can be firing here.
        state = make_state(clients=[], tau=300.0, position=DEPOT, standing=False)

        result = action_set.select_vehicle_possible_actions(
            2, 0, no_delayed_clients(), state, [DEPOT], geometry, DEPOT, 1, SHIFT_END
        )

        assert result == [DEPOT]


# --- Branch 2: fewer than three Clients left, the endgame classifier ----------


class TestFewerThanThreeClientsBranch:
    def test_uses_the_shortest_distance_classifier(self):
        cache = make_cache({(0, 0): (0.0, 0.0), (0, 5): (10.0, 4.0), (0, 6): (20.0, 8.0)})
        geometry = EpisodeGeometry.build(cache, clients=[5, 6], depot=DEPOT)
        state = make_state(clients=[5, 6], tau=300.0, position=DEPOT, standing=True)

        result = action_set.select_vehicle_possible_actions(
            2, 0, no_delayed_clients(), state, [DEPOT], geometry, DEPOT, 1, SHIFT_END
        )

        # _classify_shortest_distance_clients orders this vehicle's two
        # remaining Clients nearest-first; alone and unforbidden, both survive.
        assert result == [5, 6]


# --- Branch 3: the normal k-nearest sweep, its depot append and its ------------
# --- delayed-Client append -----------------------------------------------------


class TestNormalBranch:
    def test_appends_the_depot_when_the_return_leg_breaches_the_horizon(self):
        hub = 9  # a node the vehicle stands on, distinct from the depot and every Client
        cache = make_cache(
            {
                (0, 0): (0.0, 0.0),
                (hub, 0): (100.0, 40.0),
                (hub, 1): (1.0, 1.0),
                (hub, 2): (2.0, 2.0),
                (hub, 3): (3.0, 3.0),
            }
        )
        geometry = EpisodeGeometry.build(cache, clients=[1, 2, 3], depot=DEPOT)
        state = make_state(clients=[1, 2, 3], tau=700.0, position=hub, standing=False)

        result = action_set.select_vehicle_possible_actions(
            3, 0, no_delayed_clients(), state, [DEPOT], geometry, DEPOT, 1, shift_end_minute=750.0
        )

        # tau(700) + minutes back to the depot(100) breaches shift_end(750): the
        # depot is appended after the k-nearest dedup, alongside the three
        # Clients, not instead of them.
        assert result == [1, 2, 3, DEPOT]

    def test_appends_delayed_clients_not_already_present_or_forbidden(self):
        cache = make_cache(
            {
                (0, 0): (0.0, 0.0),
                (0, 1): (1.0, 1.0),
                (0, 2): (2.0, 2.0),
                (0, 3): (3.0, 3.0),
                (0, 4): (4.0, 4.0),
            }
        )
        geometry = EpisodeGeometry.build(cache, clients=[1, 2, 3, 4], depot=DEPOT)
        state = make_state(clients=[1, 2, 3, 4], tau=300.0, position=DEPOT, standing=True)
        # Client 5 is outside this fixture's geometry entirely: the
        # delayed-Client append never looks it up, only compares ids.
        features = make_features(((5,),))

        result = action_set.select_vehicle_possible_actions(
            2, 0, features, state, [DEPOT], geometry, DEPOT, 1, SHIFT_END
        )

        assert result == [1, 2, 5]

    def test_a_delayed_client_already_selected_is_not_appended_twice(self):
        cache = make_cache(
            {
                (0, 0): (0.0, 0.0),
                (0, 1): (1.0, 1.0),
                (0, 2): (2.0, 2.0),
                (0, 3): (3.0, 3.0),
            }
        )
        geometry = EpisodeGeometry.build(cache, clients=[1, 2, 3], depot=DEPOT)
        state = make_state(clients=[1, 2, 3], tau=300.0, position=DEPOT, standing=True)
        features = make_features(((2,),))  # already among the k-nearest below

        result = action_set.select_vehicle_possible_actions(
            2, 0, features, state, [DEPOT], geometry, DEPOT, 1, SHIFT_END
        )

        assert result == [1, 2]


# --- The dedup's dependence on CPython's set iteration order ------------------


class TestDedupOrderingHazard:
    """``possible_actions = list(set(possible_actions))`` — module docstring's
    "one real hazard": deterministic in-process, but only if the same ints are
    inserted in the same order. Pinned element by element, in order — set
    equality (``set(result) == {...}``) is exactly the assertion a reordering of
    the appends before the dedup would still pass.
    """

    def test_the_returned_order_is_the_hash_table_order_not_the_nearest_first_one(self):
        cache = make_cache(
            {
                (0, 0): (0.0, 0.0),
                (0, 1): (1.0, 1.0),
                (0, 2): (2.0, 2.0),
                (0, 3): (3.0, 3.0),
                (0, 10): (4.0, 4.0),
            }
        )
        geometry = EpisodeGeometry.build(cache, clients=[1, 2, 3, 10], depot=DEPOT)
        state = make_state(clients=[1, 2, 3, 10], tau=300.0, position=DEPOT, standing=True)

        result = action_set.select_vehicle_possible_actions(
            4, 0, no_delayed_clients(), state, [DEPOT], geometry, DEPOT, 1, SHIFT_END
        )

        # The k-nearest search hands the dedup Clients in ascending-travel-time
        # order [1, 2, 3, 10]. CPython's set iteration for exactly these four
        # ints comes out [10, 1, 2, 3] — Client 10, the *farthest* one, first.
        # Precondition, so a future change to this fixture cannot silently
        # collapse the two orderings into looking the same by coincidence:
        assert [1, 2, 3, 10] != [10, 1, 2, 3]
        assert result == [10, 1, 2, 3]
