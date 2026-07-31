"""State: the information available to make a decision at a point in simulated time.

Phase-1 structural port of the legacy ``state`` class (ADR-0001). Deliberately
concrete — no interface (ADR-0002). Node ids start as ints (the depot and Client
nodes) and become floats as vehicles traverse cached paths, whose node ids the
legacy parsed as floats; float and int ids hash and compare equal, so lookups work.

Preserved legacy quirk: ``vehicle_completing_service`` is initialized with the
depot id but used as a 0/1 service flag — identical only because the depot is 0.

Ticket 04 (simulator-correctness, B1a/B1b, ADR-0005): ``last_node_reached``
(formerly ``vehicle_position``) is **only** the last node a vehicle reached —
never "where it is". A vehicle can be strictly mid-arc, having merely passed
through that node, with ``last_node_reached`` unchanged. Whether it is actually
standing there is the separate ``vehicle_standing`` fact, maintained by the
:class:`~stdvrp.simulation.model.Model` at every transition that changes either
one. Two now-honestly-named-away fields used to blur exactly this: ``vehicles_direction``
was write-only (removed) and ``vehicle_next_node`` was write-only too — its slot
is reused for ``vehicle_standing`` rather than adding a third list.
"""

from __future__ import annotations

from dataclasses import dataclass


def is_parked_at_depot(node: float, standing: bool, depot: int) -> bool:
    """Is a vehicle genuinely parked at the depot — not merely last seen there?

    The one predicate every "is this vehicle idle/home?" read site needs
    (ticket 04, ADR-0005): reconstructing ``node == depot and standing``
    inline at each site is exactly the failure mode that produced B1a/B1b —
    three sites had already gotten a weaker version of it wrong. Takes plain
    values, not a ``State``, so it reads off a live ``State``, a
    ``TrainingSnapshot``, or an already-extracted array row alike.
    """
    return node == depot and standing


class State:
    """Mutable per-Episode state: the Model's transition function advances it."""

    def __init__(
        self,
        number_vehicles: int,
        clients: list[int],
        n_observed_velocities: int,
        horizon_start_minute: int,
        depot: int,
    ) -> None:
        # Simulated time in minutes since 03:00.
        self.tau_episode: float = horizon_start_minute
        self.horizon_start_minute = horizon_start_minute

        # Node each vehicle last reached — a waypoint it may since have driven
        # straight through. Whether it is actually standing there is
        # ``vehicle_standing`` below, not this field (ticket 04, ADR-0005).
        self.last_node_reached: list[float] = [depot for _ in range(number_vehicles)]

        # The very list handed in — the legacy aliased and mutated it in place.
        self.clients_not_visited = clients

        # Sliding window of the last n_observed_velocities velocity observations,
        # per vehicle (km/min) — one entry per decision epoch, not per distinct
        # arc: a vehicle resampled several times on the same arc fills several
        # slots with that one arc's velocities (B18, docs/simulator-review.md).
        self.observed_velocity: list[list[float]] = [
            [0 for _ in range(n_observed_velocities)] for _ in range(number_vehicles)
        ]
        self.n_observed_velocities = n_observed_velocities

        self.terminal = False
        self.number_vehicles = number_vehicles

        # client -> [arrival_minute, vehicle] once served.
        self.clients_arrival: dict[float, list[float]] = {}

        self.total_vehicle_distance_travelled: dict[int, float] = {
            vehicle: 0 for vehicle in range(number_vehicles)
        }

        # Is the vehicle genuinely standing at ``last_node_reached`` (parked,
        # serving, or holding), rather than mid-arc past it? The Model flips
        # this at every arrival and at every real departure (``begin_arc``'s
        # travel branch); it starts ``True`` (every vehicle begins parked at
        # the depot). Reuses the slot the write-only ``vehicle_next_node``
        # used to occupy (ticket 04, ADR-0005) rather than adding a third list.
        self.vehicle_standing: list[bool] = [True for _ in range(number_vehicles)]

        # 0/1 flag: vehicle is inside a Client's service time (see module docstring).
        self.vehicle_completing_service: list[float] = [depot for _ in range(number_vehicles)]


@dataclass(frozen=True, slots=True)
class TrainingSnapshot:
    """Immutable capture of the State surface training replay reads.

    ``Model.run_training_episode`` snapshots the State before every transition;
    ``MonteCarloPolicy.learn`` (aliased ``update_W``) walks the snapshots
    backward, handing each in turn to ``MonteCarloPolicy._already_acquired_cost``
    and to the ``FeatureExtractor`` (``state_features`` / ``action_features``).
    That replay path reads the first five fields below and never mutates
    them — narrower and cheaper to copy than ``copy.deepcopy(state)``,
    which also duplicated fields the replay never touches (``clients_arrival``,
    ``total_vehicle_distance_travelled``, ...). ``State`` mutates
    ``clients_not_visited``, ``last_node_reached``, ``vehicle_standing`` and
    ``observed_velocity`` in
    place across the Episode, so ``capture`` must copy them, not alias them; the
    tuples below make that copy and the resulting snapshot immutable in one step.
    ``vehicle_standing`` joined the other four in ticket 04 (ADR-0005):
    ``_already_acquired_cost``'s overtime term reads it alongside
    ``last_node_reached`` to tell a vehicle genuinely parked at the depot from
    one merely last seen there mid-arc.

    ``vehicle_completing_service`` joined the other five in ticket 02
    (neural-policy): no read site inside ``update_W``/``learn``'s replay path
    reads it (hence the previous five being "exactly right"), but ticket 04's
    tokenizer will, replaying off the same snapshots. Copied, like the rest —
    ``State`` mutates this list in place too.
    """

    tau_episode: float
    clients_not_visited: tuple[int, ...]
    last_node_reached: tuple[float, ...]
    vehicle_standing: tuple[bool, ...]
    observed_velocity: tuple[tuple[float, ...], ...]
    vehicle_completing_service: tuple[float, ...]

    @classmethod
    def capture(cls, state: State) -> TrainingSnapshot:
        """Copy the fields the replay path (``learn``) and the tokenizer read off ``state``."""
        return cls(
            tau_episode=state.tau_episode,
            clients_not_visited=tuple(state.clients_not_visited),
            last_node_reached=tuple(state.last_node_reached),
            vehicle_standing=tuple(state.vehicle_standing),
            observed_velocity=tuple(tuple(v) for v in state.observed_velocity),
            vehicle_completing_service=tuple(state.vehicle_completing_service),
        )
