"""Episode runners: one full evaluation or training Episode from a seed.

Ports the per-episode blocks of the legacy ``training_and_testing`` loops
(ADR-0001): demand is drawn first, then State, Policy and Model are built, then
the episode runs to termination.

Ticket 13 (RNG modernization, ADR-0001 phase 2): each Episode's ``seed`` spawns
three independent ``np.random.Generator`` streams via ``np.random.SeedSequence``
— one each for congestion, velocities and policy exploration — injected into the
fresh Model/Policy this call constructs. Demand draws its own Generator directly
from ``seed`` inside ``ClientGenerator.generate`` (see that module). No global
``random``/``np.random`` state is touched, and spawning keeps the four concerns'
streams independent of each other and of every other Episode's. This replaces the
legacy's single shared global stream reseeded to ``seed`` at Episode start; exact
draw-order equality with the legacy is retired (ADR-0001).

Test episodes override the fleet size from the legacy's per-seed vehicle table and
widen the action pool (``vehicles + actions``); evaluation episodes use the
generated fleet size and ``vehicles + 2``. Both are expressed through
``vehicle_count`` / ``number_actions_test``.

Training episodes (ticket 08) mirror the per-seed block of
``training_and_testing.training_model``: the fleet size is always the generated
one and both action pools are ``vehicles + 2``. One legacy behavior is the
caller's responsibility, exactly as in the legacy loop:

* **Warm-up learning-rate quirk** (ticket 12): the legacy sets ``lr = 0.000001``
  before its training loop and only assigns the configured learning rate after
  constructing the first Episode's policy — so the FIRST training Episode always
  updates W with the hardcoded tiny warm-up rate and every later Episode uses the
  configured one. Callers (the ticket 09 Trainer) must pass ``learning_rate`` per
  Episode accordingly.
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

EpisodeRngs = tuple[np.random.Generator, np.random.Generator, np.random.Generator]


def _spawn_episode_rngs(seed: int) -> EpisodeRngs:
    """Three independent per-Episode streams: congestion, velocities, exploration.

    Ticket 13 (ADR-0001 phase 2): ``SeedSequence.spawn`` derives children that are
    (with overwhelming probability) statistically independent of each other and of
    ``np.random.default_rng(seed)`` itself, so this stays decorrelated from the
    demand Generator ``ClientGenerator.generate`` builds straight from ``seed``.
    """
    congestion_seed, velocity_seed, exploration_seed = np.random.SeedSequence(seed).spawn(3)
    return (
        np.random.default_rng(congestion_seed),
        np.random.default_rng(velocity_seed),
        np.random.default_rng(exploration_seed),
    )


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
    congestion_rng, velocity_rng, exploration_rng = _spawn_episode_rngs(seed)

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
        exploration_rng=exploration_rng,
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
        velocity_rng=velocity_rng,
        congestion_rng=congestion_rng,
    )
    model.run_evaluation_episode()

    return _episode_result(model)


def _episode_result(model: Model) -> EpisodeResult:
    """Read the Episode outcome off a finished Model."""
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

    ``episode.distance_cost`` is the Episode's real distance component — the
    legacy zeroed the accumulator after every training step and always reported
    0 (reporting only, rewards were unaffected; fixed in ticket 12, ADR-0001
    phase-2 change log).
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
) -> TrainingEpisodeResult:
    """Run one ε-greedy training Episode and return the updated W with its costs.

    ``W`` is carried over from the previous training Episode (``None`` on the
    first one — the Policy constructor's greedy pass lazily creates the zero
    vector, exactly like the legacy). See the module docstring for the warm-up
    learning-rate quirk, which lives at the caller.
    """
    demand = client_generator.generate(seed)
    congestion_rng, velocity_rng, exploration_rng = _spawn_episode_rngs(seed)

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
        exploration_rng=exploration_rng,
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
        velocity_rng=velocity_rng,
        congestion_rng=congestion_rng,
    )
    model.run_training_episode()

    assert policy.W is not None
    return TrainingEpisodeResult(w=policy.W, episode=_episode_result(model))
