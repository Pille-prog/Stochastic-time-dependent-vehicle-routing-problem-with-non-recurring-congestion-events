"""Unit tests for ``Model._reroute_for``'s depot-park-forever branch (ticket 04, B1b).

The review's finding: a vehicle mid-arc past the depot — the depot used as an
*interior* node of its route, 6.8% of cached shortest paths do this — leaves
``last_node_reached == depot`` even though the vehicle never stopped there. If
the next decision then sends that vehicle to the depot (e.g. because no
Clients are left), the branch used to read that stale node as "already
parked" and set ``fleet.arrival_tau = PARKED`` on the spot: the remaining arc
and the drive home are never charged, and the vehicle can never revive
(``is_travelling`` permanently ``False``). ``State.vehicle_standing``
(ADR-0005) is the fix — the branch now fires only for a vehicle genuinely
standing at the depot, exactly its intended, frequent, harmless case.

The Model here is built via ``__new__`` with only the collaborators
``_reroute_for`` touches, mirroring ``test_model_termination.py``'s
``make_terminating_model``.
"""

from stdvrp.network.shortest_path_cache import ShortestPath, ShortestPathCache
from stdvrp.simulation.fleet_routes import PARKED, FleetRoutes
from stdvrp.simulation.model import Model
from stdvrp.simulation.state import State

DEPOT = 0
NEXT_NODE = 7.0  # the node the vehicle is travelling towards when mid-arc


def make_reroute_model(*, tau: float) -> Model:
    state = State(1, [1, 2], 3, 300, DEPOT)
    state.tau_episode = tau

    fleet = FleetRoutes(1, 300)
    # Mid-arc: departed the depot (as an interior waypoint) at 340, arrives at
    # NEXT_NODE at 400 — genuinely travelling at any tau in (340, 400).
    fleet.departure_tau[0] = 340.0
    fleet.arrival_tau[0] = 400.0
    fleet.route[0] = [DEPOT, NEXT_NODE]
    fleet.destination[0] = 5  # the target this vehicle was already headed towards

    model = Model.__new__(Model)
    model.state = state
    model.fleet = fleet
    model.depot = DEPOT
    model.shortest_path_cache = ShortestPathCache(
        {(NEXT_NODE, DEPOT): ShortestPath([NEXT_NODE, DEPOT], 10.0, 5.0)}
    )
    return model


class TestDepotParkForeverGuard:
    def test_a_vehicle_genuinely_parked_at_the_depot_is_left_parked(self):
        """The branch's normal, frequent, harmless case: unaffected by the fix.

        ``arrival_tau`` starts at a value that is deliberately *not* ``PARKED``
        (500.0, an arbitrary future clock) so the assertion below proves the
        branch itself fired and parked the vehicle — asserting ``== PARKED``
        against a fixture that was already ``PARKED`` beforehand would pass
        vacuously even if this branch never ran at all.
        """
        model = make_reroute_model(tau=360.0)
        model.state.last_node_reached[0] = DEPOT
        model.state.vehicle_standing[0] = True
        model.fleet.arrival_tau[0] = 500.0
        model.fleet.departure_tau[0] = 0

        model._reroute_for([DEPOT])

        assert model.fleet.arrival_tau[0] == PARKED
        assert model.fleet.horizon_change_tau[0] == 360.0

    def test_a_vehicle_mid_arc_past_the_depot_is_not_teleported_home(self):
        """B1b: crossing the depot must not read as "standing" there."""
        model = make_reroute_model(tau=360.0)
        model.state.last_node_reached[0] = DEPOT  # crossed it, per vehicle_reaches_node
        model.state.vehicle_standing[0] = False  # ...but never stopped

        model._reroute_for([DEPOT])

        assert model.fleet.arrival_tau[0] != PARKED, (
            "a mid-arc vehicle must not be retired to the depot on a stale node read"
        )
        # It falls to the mid-arc reroute branch instead: still travelling,
        # now routed to finish this arc and come home.
        assert model.fleet.destination[0] == DEPOT
        assert model.fleet.route[0][0] == DEPOT
