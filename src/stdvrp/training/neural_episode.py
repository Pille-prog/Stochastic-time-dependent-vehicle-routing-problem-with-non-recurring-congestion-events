"""Episode runners for the transformer Policy (ticket 07, neural-policy).

Parallel to :mod:`stdvrp.simulation.episode`'s ``run_training_episode`` /
``run_evaluation_episode`` — deliberately not a change to that module. Those
two functions build a fresh ``MonteCarloPolicy(W=...)`` and hand ``W`` back
functionally; that file, and the linear baseline's path through it, is the
frozen opponent every ticket in this effort predicts a zero self-golden diff
against. ``TransformerMonteCarloPolicy`` instead flows its trainable state
(:class:`NeuralPolicyState`) in as mutable, shared objects that :meth:`
~stdvrp.policies.transformer_policy.TransformerMonteCarloPolicy.learn`
updates in place — the caller (the Trainer's training loop) keeps the same
``NeuralPolicyState`` alive across every Episode of a run, constructing a new,
lightweight ``TransformerMonteCarloPolicy`` wrapper around it each time.

**RNG state, and why the checkpoint does not need to persist it.** Every
stochastic stream this module uses — congestion, velocity, exploration, and
now the minibatch shuffle ``learn`` needs — is spawned fresh from the
Episode's own ``seed`` (:func:`spawn_neural_episode_rngs`, extending ticket
13's ``_spawn_episode_rngs`` with a fourth stream), never carried over from
the previous Episode. Resuming a run therefore needs only the next episode
*index* (which determines every subsequent Episode's seed and hence its
streams) — there is no separate generator state to serialize.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import numpy as np
import torch

from stdvrp.config import ExperimentConfig
from stdvrp.congestion import CongestionGenerator
from stdvrp.demand.client_generator import ClientGenerator
from stdvrp.network.episode_geometry import EpisodeGeometry
from stdvrp.network.shortest_path_cache import ShortestPathCache
from stdvrp.policies.network import QHead, TokenEncoder
from stdvrp.policies.transformer_policy import CalibrationPair, TransformerMonteCarloPolicy
from stdvrp.simulation.episode import EpisodeResult
from stdvrp.simulation.model import Model
from stdvrp.simulation.state import State, TrainingSnapshot
from stdvrp.traffic.travel_time_model import TravelTimeModel

__all__ = [
    "NeuralPolicyState",
    "build_neural_policy_state",
    "run_neural_calibration_episode",
    "run_neural_evaluation_episode",
    "run_neural_training_episode",
    "spawn_neural_episode_rngs",
]

NeuralEpisodeRngs = tuple[
    np.random.Generator, np.random.Generator, np.random.Generator, np.random.Generator
]


def spawn_neural_episode_rngs(seed: int) -> NeuralEpisodeRngs:
    """Four independent per-Episode streams: congestion, velocity, exploration, learn.

    Extends :func:`stdvrp.simulation.episode._spawn_episode_rngs` (ticket 13,
    ADR-0001 phase 2) with a fourth child for ``learn``'s minibatch shuffle —
    a stochastic concern that function's three linear-baseline streams have no
    equivalent of. Kept as this module's own function, not a shared helper,
    for the same reason ``episode.py`` itself stays untouched: this file
    predicts a zero self-golden diff by never editing the pinned baseline.
    """
    congestion_seed, velocity_seed, exploration_seed, learn_seed = np.random.SeedSequence(
        seed
    ).spawn(4)
    return (
        np.random.default_rng(congestion_seed),
        np.random.default_rng(velocity_seed),
        np.random.default_rng(exploration_seed),
        np.random.default_rng(learn_seed),
    )


@dataclass(slots=True)
class NeuralPolicyState:
    """The transformer Policy's trainable state, long-lived across Episodes.

    The neural analogue of ``W: NDArray`` — except mutated in place by
    ``learn`` (an ordinary PyTorch optimizer step) rather than reassigned.
    Callers keep one instance alive for a whole training run and construct a
    fresh, lightweight ``TransformerMonteCarloPolicy`` around it every
    Episode.
    """

    encoder: TokenEncoder
    head: QHead
    optimizer: torch.optim.Optimizer

    @property
    def current_lr(self) -> float:
        lr: float = self.optimizer.param_groups[0]["lr"]
        return lr

    @current_lr.setter
    def current_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = value


def build_neural_policy_state(
    config: ExperimentConfig, init_rng: np.random.Generator
) -> NeuralPolicyState:
    """Fresh network + optimizer from ``config``'s architecture fields.

    ``config.device`` other than ``"cpu"`` is rejected outright rather than
    silently accepted: ``TokenEncoder.forward`` builds its input tensors via
    ``torch.from_numpy`` with no ``.to(device)`` anywhere (see
    ``network.py``), so a non-CPU device is not wired end-to-end yet — the
    same gap ``torch_support.py`` and spec.md's compute-budget section
    already document as untested, not silently claimed here either.
    """
    if config.device != "cpu":
        raise NotImplementedError(
            f"device={config.device!r} is not supported yet: TokenEncoder/QHead's forward "
            "pass always builds CPU tensors (see network.py's module docstring). "
            "Use device: cpu."
        )

    encoder = TokenEncoder(
        d_model=config.neural_d_model,
        n_layers=config.neural_n_layers,
        n_heads=config.neural_n_heads,
        n_observed_velocities=config.n_observed_velocities,
        init_rng=init_rng,
    )
    head = QHead(d_model=config.neural_d_model, init_rng=init_rng)
    optimizer = torch.optim.Adam(
        chain(encoder.parameters(), head.parameters()), lr=config.neural_learning_rate
    )
    return NeuralPolicyState(encoder=encoder, head=head, optimizer=optimizer)


def _build_episode(
    *,
    seed: int,
    client_generator: ClientGenerator,
    travel_time_model: TravelTimeModel,
    shortest_path_cache: ShortestPathCache,
    congestion_generator: CongestionGenerator,
    policy_state: NeuralPolicyState,
    config: ExperimentConfig,
    depot: int,
) -> tuple[Model, TransformerMonteCarloPolicy]:
    """Everything the three episode runners below share: demand -> Model, ready to run.

    ``depot`` and ``config.epsilon`` reach the built ``TransformerMonteCarloPolicy``
    unconditionally (harmless for the two read-only runners: greedy ``decide``
    never consults ``epsilon``, only ``decide_train`` does).
    """
    demand = client_generator.generate(seed)
    congestion_rng, velocity_rng, exploration_rng, learn_rng = spawn_neural_episode_rngs(seed)

    number_vehicles = demand.vehicle_count
    clients = [client.node for client in demand.clients]
    time_windows = {
        client.node: (client.time_window_start, client.time_window_end) for client in demand.clients
    }

    state = State(
        number_vehicles, clients, config.n_observed_velocities, config.horizon_start_minute, depot
    )
    geometry = EpisodeGeometry.build(shortest_path_cache, clients, depot)
    policy = TransformerMonteCarloPolicy(
        number_vehicles,
        geometry,
        time_windows,
        len(clients),
        config.epsilon,
        depot,
        config.shift_end_minute,
        config.episode_end_minute,
        config.horizon_start_minute,
        policy_state.encoder,
        policy_state.head,
        policy_state.optimizer,
        exploration_rng=exploration_rng,
        learn_rng=learn_rng,
        learn_passes=config.neural_learn_passes,
        batch_size=config.neural_batch_size,
    )
    model = Model(
        state,
        policy,
        travel_time_model,
        shortest_path_cache,
        time_windows,
        number_vehicles,
        config.horizon_start_minute,
        config.shift_end_minute,
        config.episode_end_minute,
        depot,
        congestion_generator,
        config.max_congestion_duration,
        velocity_rng=velocity_rng,
        congestion_rng=congestion_rng,
    )
    return model, policy


def _episode_result(model: Model) -> EpisodeResult:
    """Read the Episode outcome off a finished Model (mirrors ``episode.py``'s own)."""
    costs = model.costs
    return EpisodeResult(
        total_cost=costs.total_cost,
        distance_cost=costs.distance_cost,
        delay_cost=costs.delay_cost,
        earliness_cost=costs.earliness_cost,
        overtime_cost=costs.overtime_cost,
        tau=model.state.tau_episode,
        state_count=model.total_state_counter,
        delay_clients=costs.late_clients,
        earliness_clients=costs.early_clients,
        unserved_clients=costs.unserved_clients,
    )


def run_neural_training_episode(
    *,
    seed: int,
    client_generator: ClientGenerator,
    travel_time_model: TravelTimeModel,
    shortest_path_cache: ShortestPathCache,
    congestion_generator: CongestionGenerator,
    policy_state: NeuralPolicyState,
    config: ExperimentConfig,
    depot: int = 0,
) -> tuple[EpisodeResult, float]:
    """Run one epsilon-greedy training Episode; mutate ``policy_state`` in place.

    Returns the Episode's cost outcome and the mean training loss ``learn``
    computed (``TransformerMonteCarloPolicy.last_loss`` — not part of the
    ``TrainablePolicy`` protocol, read here for the live per-episode report).
    """
    model, policy = _build_episode(
        seed=seed,
        client_generator=client_generator,
        travel_time_model=travel_time_model,
        shortest_path_cache=shortest_path_cache,
        congestion_generator=congestion_generator,
        policy_state=policy_state,
        config=config,
        depot=depot,
    )
    model.run_training_episode()

    return _episode_result(model), policy.last_loss


def run_neural_evaluation_episode(
    *,
    seed: int,
    client_generator: ClientGenerator,
    travel_time_model: TravelTimeModel,
    shortest_path_cache: ShortestPathCache,
    congestion_generator: CongestionGenerator,
    policy_state: NeuralPolicyState,
    config: ExperimentConfig,
    depot: int = 0,
) -> EpisodeResult:
    """Run one greedy evaluation Episode; reads ``policy_state``, never mutates it."""
    model, policy = _build_episode(
        seed=seed,
        client_generator=client_generator,
        travel_time_model=travel_time_model,
        shortest_path_cache=shortest_path_cache,
        congestion_generator=congestion_generator,
        policy_state=policy_state,
        config=config,
        depot=depot,
    )
    model.run_evaluation_episode()

    return _episode_result(model)


def run_neural_calibration_episode(
    *,
    seed: int,
    client_generator: ClientGenerator,
    travel_time_model: TravelTimeModel,
    shortest_path_cache: ShortestPathCache,
    congestion_generator: CongestionGenerator,
    policy_state: NeuralPolicyState,
    config: ExperimentConfig,
    depot: int = 0,
) -> tuple[EpisodeResult, list[CalibrationPair]]:
    """Run one greedy Episode, capturing calibration pairs; never mutates ``policy_state``.

    Gate A's (ticket 08) source of ``(Q_predicted, U_t)`` pairs (spec.md's
    calibration check): mirrors :func:`run_neural_training_episode`'s
    snapshot/action/reward capture around :meth:`Model.transition_function`,
    but decides greedily (``policy.decide``, not ``decide_train`` --
    epsilon-exploration has no place in a held-out measurement) and calls
    :meth:`~stdvrp.policies.transformer_policy.TransformerMonteCarloPolicy.calibration_pairs`
    instead of ``learn`` at the end, so no gradient is built and no parameter
    moves. Its cost outcome is bit-identical to :func:`run_neural_evaluation_episode`
    for the same ``seed``/``policy_state`` -- both decide the same way -- it
    additionally captures the trace :func:`run_neural_evaluation_episode`
    (via ``Model.run_evaluation_episode``) never builds.
    """
    model, policy = _build_episode(
        seed=seed,
        client_generator=client_generator,
        travel_time_model=travel_time_model,
        shortest_path_cache=shortest_path_cache,
        congestion_generator=congestion_generator,
        policy_state=policy_state,
        config=config,
        depot=depot,
    )

    snapshots: list[TrainingSnapshot] = []
    actions: list[list[int]] = []
    rewards: list[float] = [0.0]
    while not model.state.terminal:
        action = policy.decide(model.state)
        snapshots.append(TrainingSnapshot.capture(model.state))
        actions.append(list(action))
        reward = model.transition_function(action)
        rewards.append(reward)
        model.total_state_counter += 1
    model.velocities.release()

    pairs = policy.calibration_pairs(snapshots, actions, rewards)
    return _episode_result(model), pairs
