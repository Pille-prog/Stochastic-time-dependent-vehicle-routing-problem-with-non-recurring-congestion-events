"""Unit tests for :mod:`stdvrp.training.neural_episode` (ticket 07, neural-policy).

Runs a handful of **real** episodes against the committed mini fixture (a
tiny architecture keeps this fast) rather than stubbing the simulator —
``transformer_policy.py`` and the tokenizer/network already have their own
focused unit tests; what this module adds is the *wiring*, best checked by
actually running it once.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import stdvrp.simulation  # noqa: E402, F401 -- circular-import landmine, see test_tokenizer.py
from stdvrp.config import ExperimentConfig  # noqa: E402
from stdvrp.training.episode_pool import EpisodeWorld  # noqa: E402
from stdvrp.training.neural_episode import (  # noqa: E402
    build_neural_policy_state,
    run_neural_calibration_episode,
    run_neural_evaluation_episode,
    run_neural_training_episode,
    spawn_neural_episode_rngs,
)

pytestmark = pytest.mark.neural

FIXTURE_CONFIG = Path(__file__).resolve().parents[1] / "fixtures" / "chengdu_mini" / "config.yaml"


def make_config(**overrides: object) -> ExperimentConfig:
    config = ExperimentConfig.from_yaml(FIXTURE_CONFIG)
    values: dict[str, object] = {"neural_d_model": 8, "neural_n_layers": 1, "neural_n_heads": 2}
    values.update(overrides)
    return dataclasses.replace(config, **values)


class TestSpawnNeuralEpisodeRngs:
    def test_same_seed_gives_bit_identical_streams(self) -> None:
        a = spawn_neural_episode_rngs(1000)
        b = spawn_neural_episode_rngs(1000)
        for rng_a, rng_b in zip(a, b, strict=True):
            np.testing.assert_array_equal(rng_a.random(10), rng_b.random(10))

    def test_different_seeds_give_different_streams(self) -> None:
        a = spawn_neural_episode_rngs(1000)
        b = spawn_neural_episode_rngs(1001)
        for rng_a, rng_b in zip(a, b, strict=True):
            assert not np.array_equal(rng_a.random(10), rng_b.random(10))

    def test_the_four_streams_are_mutually_independent(self) -> None:
        streams = spawn_neural_episode_rngs(42)
        draws = [rng.random(20) for rng in streams]
        for i in range(len(draws)):
            for j in range(i + 1, len(draws)):
                assert not np.array_equal(draws[i], draws[j])

    def test_returns_four_streams(self) -> None:
        assert len(spawn_neural_episode_rngs(0)) == 4


class TestBuildNeuralPolicyState:
    def test_rejects_a_non_cpu_device(self) -> None:
        config = make_config(device="cuda")
        with pytest.raises(NotImplementedError, match="cuda"):
            build_neural_policy_state(config, np.random.default_rng(0))

    def test_architecture_matches_config(self) -> None:
        config = make_config(neural_d_model=16, neural_n_layers=2, neural_n_heads=4)
        state = build_neural_policy_state(config, np.random.default_rng(0))
        assert state.encoder.d_model == 16

    def test_same_init_rng_gives_bit_identical_weights(self) -> None:
        config = make_config()
        a = build_neural_policy_state(config, np.random.default_rng(7))
        b = build_neural_policy_state(config, np.random.default_rng(7))
        for pa, pb in zip(a.encoder.parameters(), b.encoder.parameters(), strict=True):
            torch.testing.assert_close(pa, pb, atol=0.0, rtol=0.0)

    def test_optimizer_lr_matches_config(self) -> None:
        config = make_config(neural_learning_rate=1.5e-3)
        state = build_neural_policy_state(config, np.random.default_rng(0))
        assert state.current_lr == pytest.approx(1.5e-3)


class TestEpisodeRunners:
    """Real episodes against the mini fixture -- the wiring, not the Policy's own logic."""

    def _world(self, config: ExperimentConfig) -> EpisodeWorld:
        return EpisodeWorld.load(config)

    def test_training_episode_mutates_the_policy_state_in_place(self) -> None:
        config = make_config()
        world = self._world(config)
        state = build_neural_policy_state(config, np.random.default_rng(0))
        before = [p.clone() for p in state.encoder.parameters()]

        result, loss = run_neural_training_episode(
            seed=1000,
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
            policy_state=state,
            config=config,
        )

        assert result.total_cost >= 0
        assert loss >= 0
        assert any(
            not torch.equal(b, a) for b, a in zip(before, state.encoder.parameters(), strict=True)
        )

    def test_evaluation_episode_does_not_mutate_the_policy_state(self) -> None:
        config = make_config()
        world = self._world(config)
        state = build_neural_policy_state(config, np.random.default_rng(0))
        before = [p.clone() for p in state.encoder.parameters()]

        result = run_neural_evaluation_episode(
            seed=100000,
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
            policy_state=state,
            config=config,
        )

        assert result.total_cost >= 0
        assert all(
            torch.equal(b, a) for b, a in zip(before, state.encoder.parameters(), strict=True)
        )

    def test_same_seed_gives_bit_identical_evaluation_cost(self) -> None:
        config = make_config()
        world = self._world(config)
        state = build_neural_policy_state(config, np.random.default_rng(3))

        def run() -> float:
            return run_neural_evaluation_episode(
                seed=100001,
                client_generator=world.client_generator,
                travel_time_model=world.travel_time_model,
                shortest_path_cache=world.shortest_path_cache,
                congestion_generator=world.congestion_generator,
                policy_state=state,
                config=config,
            ).total_cost

        assert run() == run()


class TestSeed1131Regression:
    """Ticket 11 (simulator-correctness, B20, ADR-0008): the real reproduction.

    Untrained network, epsilon-greedy training (the fixture's ``epsilon:
    0.1``), seed 1131 against the committed mini fixture: two vehicles
    launched in lockstep on prefix-sharing shortest paths arrive at the depot
    at the exact instant the decision names it -- ``departure_tau == tau``
    (zero arc progress) and ``vehicle_standing == False`` (``begin_arc`` had
    already launched the vehicle). Before the fix, ``_reroute_for``'s
    at-a-node branch routed depot -> depot and crashed in
    ``FleetRoutes.current_arc`` at tau=308.7466. This is the only test in the
    suite that reproduces the real failure -- it needs the *correlated* fleet
    behaviour a real Policy produces, which uniform-random and Hypothesis-drawn
    Policies do not (measured in the ticket: 0 crashes in 300 uniform-random
    episodes and 120 adversarial lockstep-flip-flop episodes).
    """

    def test_training_episode_does_not_crash(self) -> None:
        config = make_config()
        world = EpisodeWorld.load(config)
        state = build_neural_policy_state(config, np.random.default_rng(0))

        result, loss = run_neural_training_episode(
            seed=1131,
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
            policy_state=state,
            config=config,
        )

        assert result.total_cost >= 0
        assert loss >= 0


class TestRunNeuralCalibrationEpisode:
    """Greedy, capturing, read-only -- Gate A's (Q_predicted, U_t) source (ticket 08)."""

    def _world(self, config: ExperimentConfig) -> EpisodeWorld:
        return EpisodeWorld.load(config)

    def test_does_not_mutate_the_policy_state(self) -> None:
        config = make_config()
        world = self._world(config)
        state = build_neural_policy_state(config, np.random.default_rng(0))
        before = [p.clone() for p in state.encoder.parameters()]

        result, pairs = run_neural_calibration_episode(
            seed=100000,
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
            policy_state=state,
            config=config,
        )

        assert result.total_cost >= 0
        assert len(pairs) > 0
        assert all(
            torch.equal(b, a) for b, a in zip(before, state.encoder.parameters(), strict=True)
        )

    def test_cost_matches_the_plain_evaluation_runner(self) -> None:
        """``decide`` is greedy and deterministic, so both runners must agree."""
        config = make_config()
        world = self._world(config)
        state = build_neural_policy_state(config, np.random.default_rng(4))

        evaluation_cost = run_neural_evaluation_episode(
            seed=100002,
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
            policy_state=state,
            config=config,
        ).total_cost
        calibration_cost, _ = run_neural_calibration_episode(
            seed=100002,
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
            policy_state=state,
            config=config,
        )

        assert calibration_cost.total_cost == evaluation_cost

    def test_same_seed_gives_bit_identical_pairs(self) -> None:
        config = make_config()
        world = self._world(config)
        state = build_neural_policy_state(config, np.random.default_rng(5))

        def run() -> list[tuple[float, float]]:
            _, pairs = run_neural_calibration_episode(
                seed=100003,
                client_generator=world.client_generator,
                travel_time_model=world.travel_time_model,
                shortest_path_cache=world.shortest_path_cache,
                congestion_generator=world.congestion_generator,
                policy_state=state,
                config=config,
            )
            return pairs

        assert run() == run()
