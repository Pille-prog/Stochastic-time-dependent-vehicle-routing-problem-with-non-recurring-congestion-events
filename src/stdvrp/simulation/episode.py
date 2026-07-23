"""Evaluation Episode runner: one full greedy episode from a seed.

Ports the per-episode block of the legacy ``training_and_testing.test_model`` /
evaluation loops (ADR-0001), in the exact seed order the legacy uses: client
generation reseeds the global ``random`` stream, then ``np.random.seed(seed)`` is
called (ticket 06 finding: that call belongs to the episode runner, not the
ClientGenerator), then State, Policy and Model are built — the Policy constructor
consumes one ``random.choice`` per vehicle — and the episode runs to termination.

Test episodes override the fleet size from the legacy's per-seed vehicle table and
widen the action pool (``vehicles + actions``); evaluation episodes use the
generated fleet size and ``vehicles + 2``. Both are expressed through
``vehicle_count`` / ``number_actions_test``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stdvrp.congestion import CongestionGenerator
from stdvrp.demand.client_generator import ClientGenerator
from stdvrp.network.shortest_path_cache import ShortestPathCache
from stdvrp.policies.monte_carlo import MonteCarloPolicy
from stdvrp.simulation.model import Model
from stdvrp.simulation.state import State
from stdvrp.traffic.travel_time_model import TravelTimeModel


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """The cost outcome of one Episode, component by component."""

    total_cost: float
    distance_cost: float
    delay_cost: float
    earliness_cost: float
    overtime_cost: float
    tau: float
    state_count: int
    delay_clients: int
    earliness_clients: int


def run_evaluation_episode(
    *,
    seed: int,
    client_generator: ClientGenerator,
    travel_time_model: TravelTimeModel,
    shortest_path_cache: ShortestPathCache,
    congestion_generator: CongestionGenerator,
    W: NDArray[np.float64] | None,
    epsilon: float,
    max_congestion_duration: int,
    horizon_start_minute: int,
    horizon_end_minute: int,
    n_observed_arcs: int,
    depot: int = 0,
    vehicle_count: int | None = None,
    number_actions_test: int | None = None,
) -> EpisodeResult:
    """Run one greedy evaluation Episode and return its costs.

    ``W`` is used as-is (never mutated); ``None`` makes the Policy lazily create a
    zero vector, as the legacy does on the very first episode.
    """
    demand = client_generator.generate(seed)
    np.random.seed(seed)

    number_vehicles = vehicle_count if vehicle_count is not None else demand.vehicle_count
    if number_actions_test is None:
        number_actions_test = number_vehicles + 2

    clients = [client.node for client in demand.clients]
    time_windows = {
        client.node: (client.time_window_start, client.time_window_end) for client in demand.clients
    }

    state = State(number_vehicles, clients, n_observed_arcs, horizon_start_minute, depot)
    policy = MonteCarloPolicy(
        number_vehicles,
        shortest_path_cache,
        time_windows,
        state,
        len(clients),
        epsilon,
        depot,
        number_actions_test,
        horizon_end_minute,
        W,
    )
    model = Model(
        state,
        policy,
        travel_time_model,
        shortest_path_cache,
        time_windows,
        number_vehicles,
        horizon_start_minute,
        horizon_end_minute,
        depot,
        congestion_generator,
        max_congestion_duration,
    )
    model.run_evaluation_episode()

    return EpisodeResult(
        total_cost=model.total_cost,
        distance_cost=model.total_distance_cost,
        delay_cost=model.total_delay_cost,
        earliness_cost=model.total_earliness_cost,
        overtime_cost=model.total_overtime_cost,
        tau=model.state.tau_episode,
        state_count=model.total_state_counter,
        delay_clients=model.total_delay_clients,
        earliness_clients=model.total_earliness_clients,
    )
