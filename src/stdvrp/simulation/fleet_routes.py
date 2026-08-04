"""FleetRoutes: where every vehicle is going, and how far along it is.

Simulation-performance ticket 09 (the decomposition ticket): six parallel
per-vehicle lists were scattered across the Model's instance attributes under
ported legacy names. They are one struct-of-arrays value here (ADR-0003:
array-oriented inside, domain-named outside), indexed by vehicle throughout —
plain Python lists, because the transition function reads them one vehicle at a
time and a list index beats a numpy scalar index on that path (the same
measurement that shaped ``EpisodeGeometry``).

This is the *Model's* half of the per-vehicle picture: the route a vehicle is
executing and its progress along it, which the Policy never sees. The half the
Policy does see — positions, directions, observed velocities — stays on
:class:`State`, where Powell's vocabulary puts it. Concrete by design
(ADR-0002): no seam.

Legacy names, for anyone reading the ported code against the original:
:attr:`destination` was ``model.action``, :attr:`route` was
``vehicles_shortest_path``, :attr:`arrival_tau` was ``node_time_arrival``,
:attr:`departure_tau` was ``tau_salida``, :attr:`horizon_change_tau` was
``tau_vehicle_horizon_change``, :attr:`arc_distance_travelled` was
``distance_arc_distance_travelled``.
"""

from __future__ import annotations

from operator import itemgetter

_second = itemgetter(1)

#: A vehicle that has parked at the depot for good: it will not arrive anywhere
#: again, so it drops out of every "who moves next?" comparison.
PARKED = float("inf")


class FleetRoutes:
    """Per-vehicle routing state for one Episode, one entry per vehicle."""

    def __init__(self, number_vehicles: int, horizon_start_minute: int) -> None:
        # The node this vehicle was last routed towards. Preserved legacy quirk
        # (ADR-0001): initialized to the literal 0 rather than to the configured
        # depot — identical only because the depot is 0, exactly like
        # ``State.vehicle_completing_service``. ``Model`` compares the new
        # decision against this to decide whether to reroute, so the literal is
        # load-bearing on the first transition.
        self.destination: list[int] = [0 for _ in range(number_vehicles)]

        # Remaining node sequence of the shortest path being executed: the
        # vehicle sits at ``route[v][0]`` and is travelling towards
        # ``route[v][1]``, so a two-node route means "one arc left".
        self.route: list[list[float]] = [[0, 0] for _ in range(number_vehicles)]

        # When the vehicle reaches the node it is travelling to, or PARKED.
        self.arrival_tau: list[float] = [horizon_start_minute for _ in range(number_vehicles)]

        # When the vehicle leaves the node it is at, once service there is done.
        self.departure_tau: list[float] = [horizon_start_minute for _ in range(number_vehicles)]

        # Clock at the vehicle's last velocity re-sample: the base the distance
        # covered on the current arc since then is measured from.
        self.horizon_change_tau: list[float] = [
            horizon_start_minute for _ in range(number_vehicles)
        ]

        # Distance covered so far on the arc currently being traversed.
        self.arc_distance_travelled: list[float] = [0 for _ in range(number_vehicles)]

    def earliest_arrival(self) -> tuple[int, float]:
        """The vehicle that arrives next, and when — ties go to the lowest index."""
        return min(enumerate(self.arrival_tau), key=_second)

    def all_parked(self) -> bool:
        """Has every vehicle finished at the depot?"""
        return all(tau == PARKED for tau in self.arrival_tau)

    def is_travelling(self, vehicle: int) -> bool:
        """Is this vehicle still on the road (rather than parked for good)?"""
        return self.arrival_tau[vehicle] != PARKED

    def park(self, vehicle: int) -> None:
        """Retire this vehicle at the depot: no further arrivals, no departure."""
        self.arrival_tau[vehicle] = PARKED
        self.departure_tau[vehicle] = 0

    def settle_at_node(self, vehicle: int, tau: float) -> None:
        """The vehicle is standing at a node: re-base its clock, clear its arc progress.

        Whatever happens next — a new route, a service time, another decision —
        starts from here, so the distance covered on the arc just finished is
        spent and the base the next arc's progress is measured from is now.
        """
        self.horizon_change_tau[vehicle] = tau
        self.arc_distance_travelled[vehicle] = 0

    def is_at_node(self, vehicle: int, tau: float) -> bool:
        """Zero progress into the next arc: on a node, not merely near one.

        Ticket 11 (simulator-correctness, B20, ADR-0008): positional presence,
        not ``State.vehicle_standing`` — the two can disagree for one instant.
        ``begin_arc`` flips ``vehicle_standing`` to ``False`` the moment it
        launches a vehicle, at the same ``departure_tau == tau`` this checks;
        a decision landing at exactly that instant needs "is this vehicle
        still at the node" answered from progress, not from the standing flag.
        """
        return self.departure_tau[vehicle] >= tau

    def current_arc(self, vehicle: int) -> tuple[float, float]:
        """The arc being traversed: (node departed from, node travelling to)."""
        route = self.route[vehicle]
        return route[0], route[1]

    def next_node(self, vehicle: int) -> float:
        """The node this vehicle is travelling to."""
        return self.route[vehicle][1]

    def nodes_left(self, vehicle: int) -> int:
        """Nodes remaining on the route, current position included."""
        return len(self.route[vehicle])

    def advance(self, vehicle: int) -> None:
        """Drop the node just left, so the next arc becomes the current one."""
        self.route[vehicle].pop(0)
