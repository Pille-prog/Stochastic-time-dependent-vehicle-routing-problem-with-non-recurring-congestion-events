"""Episode runners: one full evaluation or training Episode from a seed.

Ports the per-episode blocks of the legacy ``training_and_testing`` loops
(ADR-0001), in the exact seed order the legacy uses: client generation reseeds the
global ``random`` stream, then ``np.random.seed(seed)`` is called (ticket 06
finding: that call belongs to the episode runner, not the ClientGenerator), then
State, Policy and Model are built — the Policy constructor consumes one
``random.choice`` per vehicle — and the episode runs to termination.

Test episodes override the fleet size from the legacy's per-seed vehicle table and
widen the action pool (``vehicles + actions``); evaluation episodes use the
generated fleet size and ``vehicles + 2``. Both are expressed through
``vehicle_count`` / ``number_actions_test``.

Training episodes (ticket 08) mirror the per-seed block of
``training_and_testing.training_model``: the fleet size is always the generated
one and both action pools are ``vehicles + 2``. Two legacy behaviors are the
caller's responsibility, exactly as in the legacy loop:

* **Warm-up learning-rate quirk** (pending ticket 12 triage): the legacy sets
  ``lr = 0.000001`` before its training loop and only assigns the configured
  learning rate after constructing the first Episode's policy — so the FIRST
  training Episode always updates W with the hardcoded tiny warm-up rate and
  every later Episode uses the configured one. Callers (the ticket 09 Trainer,
  the golden-master tests) must pass ``learning_rate`` per Episode accordingly.
* **Exploration seeding** (ticket 04 finding 1): the legacy policy's two
  exploration RNGs are unseeded, making training nondeterministic by
  construction. ``exploration_seed`` / ``repair_seed`` reproduce the golden
  capture's convention (offset + train seed, applied right after the policy is
  built); leave them ``None`` for legacy-faithful nondeterminism.
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


@dataclass(frozen=True, slots=True)
class TrainingEpisodeResult:
    """The W produced by one training Episode plus its cost outcome.

    ``episode.distance_cost`` is always 0 by the preserved legacy quirk: the
    training loop zeroes the distance accumulator after every step.
    """

    w: NDArray[np.float64]
    episode: EpisodeResult


def run_training_episode(
    *,
    seed: int,
    client_generator: ClientGenerator,
    travel_time_model: TravelTimeModel,
    shortest_path_cache: ShortestPathCache,
    congestion_generator: CongestionGenerator,
    W: NDArray[np.float64] | None,
    learning_rate: float,
    epsilon: float,
    max_congestion_duration: int,
    horizon_start_minute: int,
    horizon_end_minute: int,
    n_observed_arcs: int,
    depot: int = 0,
    exploration_seed: int | None = None,
    repair_seed: int | None = None,
) -> TrainingEpisodeResult:
    """Run one ε-greedy training Episode and return the updated W with its costs.

    ``W`` is carried over from the previous training Episode (``None`` on the
    first one — the Policy constructor's greedy pass lazily creates the zero
    vector, exactly like the legacy). See the module docstring for the warm-up
    learning-rate quirk and the exploration-seeding convention, both of which
    live at the caller.
    """
    demand = client_generator.generate(seed)
    np.random.seed(seed)

    number_vehicles = demand.vehicle_count
    number_actions = number_vehicles + 2

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
        number_actions,
        horizon_end_minute,
        W,
        number_actions_train=number_actions,
        learning_rate=learning_rate,
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
    # Golden-capture convention: seed the otherwise-unseeded exploration RNGs
    # right after construction, before any training decision runs.
    if exploration_seed is not None:
        policy.local_rng.seed(exploration_seed)
    if repair_seed is not None:
        policy.local_rng_2.seed(repair_seed)

    model.run_training_episode()

    assert policy.W is not None
    return TrainingEpisodeResult(
        w=policy.W,
        episode=EpisodeResult(
            total_cost=model.total_cost,
            distance_cost=model.total_distance_cost,
            delay_cost=model.total_delay_cost,
            earliness_cost=model.total_earliness_cost,
            overtime_cost=model.total_overtime_cost,
            tau=model.state.tau_episode,
            state_count=model.total_state_counter,
            delay_clients=model.total_delay_clients,
            earliness_clients=model.total_earliness_clients,
        ),
    )
