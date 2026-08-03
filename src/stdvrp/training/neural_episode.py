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
lightweight ``TransformerMonteCarloPolicy`` wrapper around it each time. Since
ticket 16, that includes ``NeuralPolicyState.ridge`` — the accumulated ridge
estimator, which must persist across Episodes for the same reason the
network's weights do: an accumulator rebuilt fresh every Episode would never
accumulate anything *across* them, which is the whole point of ticket 16.

**RNG state, and why the checkpoint does not need to persist it.** Every
stochastic stream this module uses — congestion, velocity, exploration and
(since ticket 17) the trained-encoder arm's minibatch shuffle — is spawned
fresh from the Episode's own ``seed`` (:func:`spawn_neural_episode_rngs`,
extending ticket 13's ``_spawn_episode_rngs``), never carried over from the
previous Episode. Resuming a run therefore needs only the next episode
*index* (which determines every subsequent Episode's seed and hence its
streams) — there is no separate generator state to serialize. Ticket 16
retired the fourth stream this module used to spawn (``learn_rng``, the
per-episode minibatch shuffle) — the ridge estimator shuffles nothing; ticket
17 brings it back for the trained-encoder arm's own SGD minibatch shuffle,
which the frozen arm still has no use for (built regardless, at negligible
cost, so both arms share the same per-episode stream layout).
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
from stdvrp.policies.ridge_estimator import RidgeAccumulator
from stdvrp.policies.torch_support import resolve_device
from stdvrp.policies.transformer_policy import (
    CalibrationPair,
    ResidualCalibrationPair,
    TransformerMonteCarloPolicy,
)
from stdvrp.simulation.episode import EpisodeResult
from stdvrp.simulation.model import Model
from stdvrp.simulation.state import State, TrainingSnapshot
from stdvrp.traffic.travel_time_model import TravelTimeModel

__all__ = [
    "NeuralPolicyState",
    "build_neural_policy_state",
    "run_neural_calibration_episode",
    "run_neural_evaluation_episode",
    "run_neural_residual_calibration_episode",
    "run_neural_training_episode",
    "spawn_neural_episode_rngs",
]

#: candidate-spread pairs (ticket 17): ``(sd_candidates(W.phi), sd_candidates(c))``
#: per (decision epoch, vehicle) -- the r diagnostic's raw material
#: (``TransformerMonteCarloPolicy.spread_samples``).
SpreadSamples = tuple[tuple[float, float], ...]

NeuralEpisodeRngs = tuple[
    np.random.Generator, np.random.Generator, np.random.Generator, np.random.Generator
]


def spawn_neural_episode_rngs(seed: int) -> NeuralEpisodeRngs:
    """Four independent per-Episode streams: congestion, velocity, exploration, learn.

    Extends :func:`stdvrp.simulation.episode._spawn_episode_rngs` (ticket 13,
    ADR-0001 phase 2) with a third child for epsilon-greedy exploration — a
    stochastic concern that function's two linear-baseline streams have no
    equivalent of. Kept as this module's own function, not a shared helper,
    for the same reason ``episode.py`` itself stays untouched: this file
    predicts a zero self-golden diff by never editing the pinned baseline.

    A fourth stream (``learn_rng``, the per-episode minibatch shuffle) existed
    here through ticket 15; ticket 16 retired it along with the SGD loop it
    fed (the ridge estimator has nothing to shuffle) and confirmed that
    dropping it left the first three streams unaffected (``spawn(3)``'s
    children are bit-identical to ``spawn(4)``'s first three). Ticket 17
    brings it back unconditionally for the trained-encoder arm's own SGD
    minibatch shuffle (``transformer_policy.py``, "The two arms") — the
    frozen arm simply never reads it, at the cost of one unused
    ``default_rng`` construction per Episode.
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

    The neural analogue of ``W: NDArray`` — mutated in place by ``learn``
    rather than reassigned. Callers keep one instance alive for a whole
    training run and construct a fresh, lightweight
    ``TransformerMonteCarloPolicy`` around it every Episode.

    ``optimizer``/``current_lr`` predate ticket 16 and are no longer stepped
    by anything on the frozen-encoder arm (``learn`` never calls ``.backward()``
    or ``.step()`` any more — the ridge solve in ``self.ridge`` is the only way
    ``head.linear``/``head.layer2`` move, and ``encoder``/``head.layer1`` are
    not moved by this class at all). Kept, rather than removed, for two
    reasons: spec.md decision 12 (patience -> lr cut -> converged) is
    unamended by ticket 16, so ``Trainer.train_neural``'s convergence loop
    still needs a ``current_lr`` to cut on a patience trigger — purely as a
    stopping heuristic now, decoupled from what is actually being fit; and
    (ticket 17) the *trained*-encoder arm resumes training ``encoder``/
    ``head.layer1`` by SGD on the same residual with exactly this optimizer,
    scoped to only those two parameter groups (never ``head.linear``/
    ``head.layer2``, which stay exclusively the ridge's to move) --
    see :func:`build_neural_policy_state`.
    """

    encoder: TokenEncoder
    head: QHead
    optimizer: torch.optim.Optimizer
    ridge: RidgeAccumulator
    device: torch.device
    # Ticket 17: which arm this state belongs to (transformer_policy.py, "The
    # two arms"). False (the default) is the frozen-encoder arm ticket 16
    # shipped -- every existing call site keeps building that arm unchanged.
    train_encoder: bool = False

    @property
    def current_lr(self) -> float:
        lr: float = self.optimizer.param_groups[0]["lr"]
        return lr

    @current_lr.setter
    def current_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = value


def build_neural_policy_state(
    config: ExperimentConfig, init_rng: np.random.Generator, *, train_encoder: bool = False
) -> NeuralPolicyState:
    """Fresh network + ridge accumulator from ``config``'s architecture fields.

    ``config.device`` (``"cpu"``, ``"cuda"``, or ``"auto"``) is resolved
    **once** here, via :func:`~stdvrp.policies.torch_support.resolve_device`
    (ticket 12) — never re-resolved later, since ``"auto"`` is not guaranteed
    to agree between calls. The resolved device is carried on the returned
    :class:`NeuralPolicyState` for every later caller (the Trainer's log line,
    the checkpoint, the results record) to read back rather than re-derive.

    The ridge accumulator (ticket 16) is sized from ``head.feature_dim`` --
    it must match ``QHead``'s own ``linear``/``layer2`` shapes exactly, so it
    is built here, right after ``head``, rather than independently from the
    architecture config fields.

    ``train_encoder`` (ticket 17) scopes ``optimizer`` to ``encoder.parameters()``
    plus ``head.layer1.parameters()`` **only** — never ``head.linear``/
    ``head.layer2``, which the ridge solve exclusively owns on both arms
    (``head.load_w_vector``). Building the same restricted optimizer
    regardless of ``train_encoder`` (rather than only when the trained arm
    asks for it) keeps this function's return shape uniform and costs
    nothing: the frozen arm never calls ``.step()`` on it, exactly as before.
    """
    device = resolve_device(config.device)
    encoder = TokenEncoder(
        d_model=config.neural_d_model,
        n_layers=config.neural_n_layers,
        n_heads=config.neural_n_heads,
        n_observed_velocities=config.n_observed_velocities,
        init_rng=init_rng,
        device=device,
    )
    head = QHead(
        d_model=config.neural_d_model,
        init_rng=init_rng,
        device=device,
        level_gain=config.neural_level_gain,
    )
    optimizer = torch.optim.Adam(
        chain(encoder.parameters(), head.layer1.parameters()), lr=config.neural_learning_rate
    )
    ridge = RidgeAccumulator.zeros(
        head.feature_dim,
        forgetting=config.neural_ridge_gamma,
        ridge=config.neural_ridge_lambda,
    )
    return NeuralPolicyState(
        encoder=encoder,
        head=head,
        optimizer=optimizer,
        ridge=ridge,
        device=device,
        train_encoder=train_encoder,
    )


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
    never consults ``epsilon``, only ``decide_train`` does). ``learn_rng`` is
    always spawned and always passed through (ticket 17): the frozen arm's
    ``TransformerMonteCarloPolicy`` simply never reads it (``train_encoder=False``
    there), matching how ``config.epsilon`` reaches the read-only runners
    unconditionally too.
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
        policy_state.ridge,
        exploration_rng=exploration_rng,
        solve_cadence=config.neural_solve_cadence,
        device=policy_state.device,
        train_encoder=policy_state.train_encoder,
        encoder_optimizer=policy_state.optimizer if policy_state.train_encoder else None,
        learn_rng=learn_rng if policy_state.train_encoder else None,
        learn_passes=config.neural_learn_passes,
        batch_size=config.neural_batch_size,
        grad_clip_norm=config.neural_grad_clip_norm,
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

    Returns the Episode's cost outcome and ``learn``'s residual diagnostic
    (``TransformerMonteCarloPolicy.last_loss`` — not part of the
    ``TrainablePolicy`` protocol, read here for the live per-episode report;
    ticket 16 repurposes it from a training loss to the mean squared residual
    against the entering ``W`` — see that class's module docstring).
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
) -> tuple[EpisodeResult, SpreadSamples]:
    """Run one greedy evaluation Episode; reads ``policy_state``, never mutates it.

    Also returns this Episode's candidate-spread samples (ticket 17:
    ``TransformerMonteCarloPolicy.spread_samples``, accumulated by every
    ``decide()`` call this Episode made) -- Gate A''s ``r`` diagnostic's raw
    material, and spec.md's live per-block report (``neural_report.py``,
    ``EvaluationReport.candidate_spread_ratio``). Purely additive
    instrumentation over ticket 16's return shape: it does not change which
    decision this Episode's ``policy`` makes, only what gets read back off it
    afterwards, so every existing caller needs updating for the new element,
    not for a changed cost.
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
    model.run_evaluation_episode()

    return _episode_result(model), tuple(policy.spread_samples)


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


def run_neural_residual_calibration_episode(
    *,
    seed: int,
    client_generator: ClientGenerator,
    travel_time_model: TravelTimeModel,
    shortest_path_cache: ShortestPathCache,
    congestion_generator: CongestionGenerator,
    policy_state: NeuralPolicyState,
    config: ExperimentConfig,
    depot: int = 0,
) -> tuple[EpisodeResult, list[ResidualCalibrationPair], SpreadSamples]:
    """Run one greedy Episode for Gate A' (ticket 17): cost, residual
    calibration pairs and candidate-spread samples, in one pass.

    Gate A' redefines the calibration check onto the residual the network is
    actually regressed onto (``W . phi`` against ``y~``), because
    ``rho(Q, U_t)`` -- ticket 08's original pairing, :func:`run_neural_calibration_episode`
    below -- passes at ``W = 0`` with no parameter having moved (spec.md,
    "Part 3 is redefined"). A sibling of that function, not a replacement:
    ``gate_a.py`` and :func:`run_neural_calibration_episode` stay untouched
    for ticket 08's own history, and this duplicates their capture loop
    (module docstring precedent: ``transformer_policy.py``'s
    ``_already_acquired_cost``) rather than coupling the historical gate's
    code path to this one's.

    Never mutates ``policy_state``: greedy ``decide`` (not ``decide_train``),
    and ``TransformerMonteCarloPolicy.residual_calibration_pairs`` builds no
    gradient and moves no parameter, exactly like ``calibration_pairs``.
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

    pairs = policy.residual_calibration_pairs(snapshots, actions, rewards)
    return _episode_result(model), pairs, tuple(policy.spread_samples)
