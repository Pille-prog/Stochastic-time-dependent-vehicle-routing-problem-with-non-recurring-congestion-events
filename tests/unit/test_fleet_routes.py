"""Unit tests for FleetRoutes (ticket 09, simulation-performance).

The Model's six per-vehicle lists, regrouped. Most of it is plain state; what is
worth pinning is the initial values (the transition function's first pass reads
them), the tie-break in "who arrives next" and the asymmetry of parking.
"""

from stdvrp.simulation.fleet_routes import PARKED, FleetRoutes

HORIZON_START = 300


def make_fleet(vehicles: int = 3) -> FleetRoutes:
    return FleetRoutes(vehicles, HORIZON_START)


class TestInitialState:
    def test_every_clock_starts_at_the_horizon(self):
        fleet = make_fleet()
        assert fleet.arrival_tau == [HORIZON_START] * 3
        assert fleet.departure_tau == [HORIZON_START] * 3
        assert fleet.horizon_change_tau == [HORIZON_START] * 3

    def test_every_vehicle_starts_at_the_depot_with_nothing_travelled(self):
        fleet = make_fleet()
        assert fleet.destination == [0, 0, 0]
        assert fleet.route == [[0, 0], [0, 0], [0, 0]]
        assert fleet.arc_distance_travelled == [0, 0, 0]

    def test_routes_are_not_shared_between_vehicles(self):
        fleet = make_fleet()
        fleet.route[0].append(9.0)
        assert fleet.route[1] == [0, 0]


class TestEarliestArrival:
    def test_the_soonest_arrival_wins(self):
        fleet = make_fleet()
        fleet.arrival_tau = [340.0, 310.0, 900.0]
        assert fleet.earliest_arrival() == (1, 310.0)

    def test_a_tie_goes_to_the_lowest_vehicle_index(self):
        fleet = make_fleet()
        fleet.arrival_tau = [310.0, 310.0, 900.0]
        assert fleet.earliest_arrival() == (0, 310.0)

    def test_parked_vehicles_lose_to_anyone_still_travelling(self):
        fleet = make_fleet()
        fleet.arrival_tau = [PARKED, 900.0, PARKED]
        assert fleet.earliest_arrival() == (1, 900.0)

    def test_an_all_parked_fleet_still_answers(self):
        fleet = make_fleet()
        fleet.arrival_tau = [PARKED, PARKED, PARKED]
        assert fleet.earliest_arrival() == (0, PARKED)


class TestParking:
    def test_a_fresh_fleet_is_not_parked(self):
        assert not make_fleet().all_parked()
        assert make_fleet().is_travelling(0)

    def test_parking_retires_the_vehicle_and_zeroes_its_departure(self):
        fleet = make_fleet()
        fleet.park(1)

        assert fleet.arrival_tau[1] == PARKED
        assert fleet.departure_tau[1] == 0
        assert not fleet.is_travelling(1)

    def test_the_fleet_is_parked_only_once_every_vehicle_is(self):
        fleet = make_fleet()
        fleet.park(0)
        fleet.park(1)
        assert not fleet.all_parked()

        fleet.park(2)
        assert fleet.all_parked()


class TestRouteProgress:
    def test_the_current_arc_is_the_first_two_nodes(self):
        fleet = make_fleet()
        fleet.route[0] = [5.0, 6.0, 7.0]

        assert fleet.current_arc(0) == (5.0, 6.0)
        assert fleet.next_node(0) == 6.0
        assert fleet.nodes_left(0) == 3

    def test_advancing_drops_the_node_just_left(self):
        fleet = make_fleet()
        fleet.route[0] = [5.0, 6.0, 7.0]

        fleet.advance(0)

        assert fleet.route[0] == [6.0, 7.0]
        assert fleet.current_arc(0) == (6.0, 7.0)
        assert fleet.nodes_left(0) == 2


class TestIsAtNode:
    """Ticket 11 (simulator-correctness, B20, ADR-0008): the missing fact.

    Positional presence -- zero progress into the next arc -- not
    ``vehicle_standing``, which can already be ``False`` at this exact instant
    (``begin_arc`` flips it the moment it launches the vehicle, before any
    distance is covered).
    """

    def test_true_the_instant_departure_equals_tau(self):
        fleet = make_fleet()
        fleet.departure_tau[0] = 360.0
        assert fleet.is_at_node(0, 360.0)

    def test_true_while_still_awaiting_departure(self):
        """A future ``departure_tau`` (mid-service) is at-a-node too."""
        fleet = make_fleet()
        fleet.departure_tau[0] = 365.0
        assert fleet.is_at_node(0, 360.0)

    def test_false_once_arc_progress_is_nonzero(self):
        fleet = make_fleet()
        fleet.departure_tau[0] = 340.0
        assert not fleet.is_at_node(0, 360.0)
