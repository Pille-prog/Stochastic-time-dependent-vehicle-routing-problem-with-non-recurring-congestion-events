"""State: the information available to make a decision at a point in simulated time.

Phase-1 structural port of the legacy ``state`` class (ADR-0001). Deliberately
concrete — no interface (ADR-0002). Node ids start as ints (the depot and Client
nodes) and become floats as vehicles traverse cached paths, whose node ids the
legacy parsed as floats; float and int ids hash and compare equal, so lookups work.

Preserved legacy quirk: ``vehicle_completing_service`` is initialized with the
depot id but used as a 0/1 service flag — identical only because the depot is 0.
"""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass(frozen=True, slots=True)
class TrainingSnapshot:
    """Immutable capture of the State surface ``MonteCarloPolicy.update_W`` replays.

    ``Model.run_training_episode`` snapshots the State before every transition;
    ``update_W`` walks the snapshots backward, handing each in turn to
    ``MonteCarloPolicy._already_acquired_cost`` and to the ``FeatureExtractor``
    (``state_features`` / ``action_features``). That replay path reads exactly
    these four fields and never
    mutates them — narrower and cheaper to copy than ``copy.deepcopy(state)``,
    which also duplicated fields the replay never touches (``clients_arrival``,
    ``total_vehicle_distance_travelled``, ...). ``State`` mutates
    ``clients_not_visited``, ``vehicle_position`` and ``observed_velocity`` in
    place across the Episode, so ``capture`` must copy them, not alias them; the
    tuples below make that copy and the resulting snapshot immutable in one step.
    """

    tau_episode: float
    clients_not_visited: tuple[int, ...]
    vehicle_position: tuple[float, ...]
    observed_velocity: tuple[tuple[float, ...], ...]

    @classmethod
    def capture(cls, state: State) -> TrainingSnapshot:
        """Copy the four fields ``update_W``'s replay path reads off ``state``."""
        return cls(
            tau_episode=state.tau_episode,
            clients_not_visited=tuple(state.clients_not_visited),
            vehicle_position=tuple(state.vehicle_position),
            observed_velocity=tuple(tuple(v) for v in state.observed_velocity),
        )
