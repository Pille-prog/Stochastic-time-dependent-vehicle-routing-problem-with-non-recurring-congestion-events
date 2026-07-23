"""State: the information available to make a decision at a point in simulated time.

Phase-1 structural port of the legacy ``state`` class (ADR-0001). Deliberately
concrete — no interface (ADR-0002). Node ids start as ints (the depot and Client
nodes) and become floats as vehicles traverse cached paths, whose node ids the
legacy parsed as floats; float and int ids hash and compare equal, so lookups work.

Preserved legacy quirk: ``vehicle_completing_service`` is initialized with the
depot id but used as a 0/1 service flag — identical only because the depot is 0.
"""

from __future__ import annotations


class State:
    """Mutable per-Episode state: the Model's transition function advances it."""

    def __init__(
        self,
        number_vehicles: int,
        clients: list[int],
        n_arcs: int,
        horizon_start_minute: int,
        depot: int,
    ) -> None:
        # Simulated time in minutes since 03:00.
        self.tau_episode: float = horizon_start_minute
        self.horizon_start_minute = horizon_start_minute

        # Node each vehicle last departed from (or is at).
        self.vehicle_position: list[float] = [depot for _ in range(number_vehicles)]

        # The very list handed in — the legacy aliased and mutated it in place.
        self.clients_not_visited = clients

        # Velocities observed on the last n_arcs arcs, per vehicle (km/min).
        self.observed_velocity: list[list[float]] = [
            [0 for _ in range(n_arcs)] for _ in range(number_vehicles)
        ]
        self.n_arcs = n_arcs

        self.terminal = False
        self.number_vehicles = number_vehicles

        # Client each vehicle is currently heading to.
        self.vehicles_direction: list[float] = [depot for _ in range(number_vehicles)]

        # client -> [arrival_minute, vehicle] once served.
        self.clients_arrival: dict[float, list[float]] = {}

        self.total_vehicle_distance_travelled: dict[int, float] = {
            vehicle: 0 for vehicle in range(number_vehicles)
        }

        self.vehicle_next_node: list[float] = [depot for _ in range(number_vehicles)]

        # 0/1 flag: vehicle is inside a Client's service time (see module docstring).
        self.vehicle_completing_service: list[float] = [depot for _ in range(number_vehicles)]
