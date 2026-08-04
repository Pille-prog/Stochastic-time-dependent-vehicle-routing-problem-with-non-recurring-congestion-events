"""Unit tests for :mod:`stdvrp.training.neural_episode` (ticket 07, neural-policy).

Runs a handful of **real** episodes against the committed mini fixture (a
tiny architecture keeps this fast) rather than stubbing the simulator —
``transformer_policy.py`` and the tokenizer/network already have their own
focused unit tests; what this module adds is the *wiring*, best checked by
actually running it once.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
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
    run_neural_residual_calibration_episode,
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


@pytest.fixture(autouse=True)
def _restore_deterministic_algorithms() -> Iterator[None]:
    """Ticket 12: resolving "cuda" (explicitly or via "auto" on this machine's real
    GPU) flips a process-wide torch setting as a side effect -- never leak it into
    later tests in the same session, the same discipline test_torch_support.py uses."""
    was_enabled = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(was_enabled)


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

    def test_matches_spawn_four_directly(self) -> None:
        """Ticket 17 brings the fourth (``learn_rng``) stream back for the
        trained-encoder arm's own SGD minibatch shuffle (ticket 16 had
        retired it, since the ridge estimator shuffles nothing) -- every one
        of the four streams this function returns must match
        ``SeedSequence(seed).spawn(4)``'s children positionally."""
        congestion, velocity, exploration, learn = spawn_neural_episode_rngs(7)
        (
            expected_congestion_seed,
            expected_velocity_seed,
            expected_exploration_seed,
            expected_learn_seed,
        ) = np.random.SeedSequence(7).spawn(4)
        np.testing.assert_array_equal(
            congestion.random(10), np.random.default_rng(expected_congestion_seed).random(10)
        )
        np.testing.assert_array_equal(
            velocity.random(10), np.random.default_rng(expected_velocity_seed).random(10)
        )
        np.testing.assert_array_equal(
            exploration.random(10), np.random.default_rng(expected_exploration_seed).random(10)
        )
        np.testing.assert_array_equal(
            learn.random(10), np.random.default_rng(expected_learn_seed).random(10)
        )


class TestBuildNeuralPolicyState:
    def test_cpu_device_places_parameters_on_cpu(self) -> None:
        config = make_config(device="cpu")
        state = build_neural_policy_state(config, np.random.default_rng(0))
        assert state.device == torch.device("cpu")
        assert next(state.encoder.parameters()).device == torch.device("cpu")
        assert next(state.head.parameters()).device == torch.device("cpu")

    def test_explicit_cuda_without_a_gpu_fails_loudly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ticket 12: no NotImplementedError anymore -- an explicit "cuda" with no
        GPU available now raises the same loud RuntimeError resolve_device does."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        config = make_config(device="cuda")
        with pytest.raises(RuntimeError, match="cuda"):
            build_neural_policy_state(config, np.random.default_rng(0))

    def test_auto_matches_whatever_this_machine_has(self) -> None:
        config = make_config(device="auto")
        state = build_neural_policy_state(config, np.random.default_rng(0))
        expected = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        assert state.device == expected

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU on this machine")
    def test_cuda_device_places_parameters_on_cuda(self) -> None:
        config = make_config(device="cuda")
        state = build_neural_policy_state(config, np.random.default_rng(0))
        assert state.device == torch.device("cuda")
        assert next(state.encoder.parameters()).device.type == "cuda"
        assert next(state.head.parameters()).device.type == "cuda"

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

    def test_train_encoder_defaults_to_false(self) -> None:
        config = make_config()
        state = build_neural_policy_state(config, np.random.default_rng(0))
        assert state.train_encoder is False

    def test_train_encoder_flag_is_carried_on_the_state(self) -> None:
        config = make_config()
        state = build_neural_policy_state(config, np.random.default_rng(0), train_encoder=True)
        assert state.train_encoder is True

    def test_optimizer_covers_only_encoder_and_layer1_never_linear_or_layer2(self) -> None:
        """Ticket 17: head.linear/head.layer2 are exclusively the ridge
        solve's to move, on both arms -- the optimizer must never be able to
        step them, regardless of train_encoder."""
        config = make_config()
        state = build_neural_policy_state(config, np.random.default_rng(0), train_encoder=True)
        optimized = {id(p) for group in state.optimizer.param_groups for p in group["params"]}

        for p in state.encoder.parameters():
            assert id(p) in optimized
        for p in state.head.layer1.parameters():
            assert id(p) in optimized
        for p in [*state.head.linear.parameters(), *state.head.layer2.parameters()]:
            assert id(p) not in optimized


class TestEpisodeRunners:
    """Real episodes against the mini fixture -- the wiring, not the Policy's own logic."""

    def _world(self, config: ExperimentConfig) -> EpisodeWorld:
        return EpisodeWorld.load(config)

    def test_training_episode_mutates_the_policy_state_in_place(self) -> None:
        """Ticket 16: the frozen-encoder arm never moves ``encoder`` -- what
        must mutate in place is the ridge accumulator and, once it has solved
        (the fixture's default ``neural_solve_cadence`` of 1 solves after
        every Episode), ``head.linear``/``head.layer2``."""
        config = make_config()
        world = self._world(config)
        state = build_neural_policy_state(config, np.random.default_rng(0))
        encoder_before = [p.clone() for p in state.encoder.parameters()]
        w_before = state.head.w_vector().clone()

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
        assert all(
            torch.equal(b, a)
            for b, a in zip(encoder_before, state.encoder.parameters(), strict=True)
        ), "the frozen-encoder arm must never move the encoder"
        assert state.ridge.episodes_included + state.ridge.episodes_excluded == 1
        assert not torch.equal(state.head.w_vector(), w_before)

    def test_evaluation_episode_does_not_mutate_the_policy_state(self) -> None:
        config = make_config()
        world = self._world(config)
        state = build_neural_policy_state(config, np.random.default_rng(0))
        before = [p.clone() for p in state.encoder.parameters()]

        result, spread_samples = run_neural_evaluation_episode(
            seed=100000,
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
            policy_state=state,
            config=config,
        )

        assert result.total_cost >= 0
        assert isinstance(spread_samples, tuple)
        assert all(
            torch.equal(b, a) for b, a in zip(before, state.encoder.parameters(), strict=True)
        )

    def test_same_seed_gives_bit_identical_evaluation_cost(self) -> None:
        config = make_config()
        world = self._world(config)
        state = build_neural_policy_state(config, np.random.default_rng(3))

        def run() -> float:
            result, _spread_samples = run_neural_evaluation_episode(
                seed=100001,
                client_generator=world.client_generator,
                travel_time_model=world.travel_time_model,
                shortest_path_cache=world.shortest_path_cache,
                congestion_generator=world.congestion_generator,
                policy_state=state,
                config=config,
            )
            return result.total_cost

        assert run() == run()

    def test_trained_encoder_arm_moves_the_encoder_and_layer1(self) -> None:
        """Ticket 17: with ``train_encoder=True``, ``learn`` additionally runs
        SGD over ``encoder``/``head.layer1`` after the ridge fold/solve --
        the opposite of the frozen arm's own guarantee just above."""
        config = make_config()
        world = self._world(config)
        state = build_neural_policy_state(config, np.random.default_rng(0), train_encoder=True)
        encoder_before = [p.clone() for p in state.encoder.parameters()]
        layer1_before = [p.clone() for p in state.head.layer1.parameters()]

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
            not torch.equal(b, a)
            for b, a in zip(encoder_before, state.encoder.parameters(), strict=True)
        ), "the trained-encoder arm must move the encoder"
        assert any(
            not torch.equal(b, a)
            for b, a in zip(layer1_before, state.head.layer1.parameters(), strict=True)
        ), "the trained-encoder arm must move layer1"


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

        evaluation_result, _spread_samples = run_neural_evaluation_episode(
            seed=100002,
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
            policy_state=state,
            config=config,
        )
        evaluation_cost = evaluation_result.total_cost
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


class TestRunNeuralResidualCalibrationEpisode:
    """Ticket 17 (Gate A'): (W.phi, y~) pairs, a sibling of ticket 08's
    (Q_predicted, U_t) source above -- neither replaces the other."""

    def _world(self, config: ExperimentConfig) -> EpisodeWorld:
        return EpisodeWorld.load(config)

    def test_does_not_mutate_the_policy_state(self) -> None:
        config = make_config()
        world = self._world(config)
        state = build_neural_policy_state(config, np.random.default_rng(0))
        before = [p.clone() for p in state.encoder.parameters()]

        result, pairs, spread_samples = run_neural_residual_calibration_episode(
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
        assert isinstance(spread_samples, tuple)
        assert all(
            torch.equal(b, a) for b, a in zip(before, state.encoder.parameters(), strict=True)
        )

    def test_cost_matches_the_plain_evaluation_runner(self) -> None:
        """``decide`` is greedy and deterministic, so both runners must agree."""
        config = make_config()
        world = self._world(config)
        state = build_neural_policy_state(config, np.random.default_rng(4))

        evaluation_result, _spread_samples = run_neural_evaluation_episode(
            seed=100002,
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
            policy_state=state,
            config=config,
        )
        residual_result, _pairs, _spread = run_neural_residual_calibration_episode(
            seed=100002,
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
            policy_state=state,
            config=config,
        )

        assert residual_result.total_cost == evaluation_result.total_cost

    def test_at_w_zero_the_predicted_residual_is_identically_zero(self) -> None:
        """The guaranteed non-PASS spec.md's redefinition needs: at W = 0
        (an untrained network, before any ridge solve) the predicted half of
        every pair is exactly 0, whatever the residual target holds."""
        config = make_config()
        world = self._world(config)
        state = build_neural_policy_state(config, np.random.default_rng(6))

        _result, pairs, _spread = run_neural_residual_calibration_episode(
            seed=100004,
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
            policy_state=state,
            config=config,
        )

        assert len(pairs) > 0
        assert all(predicted == 0.0 for predicted, _target in pairs)

    def test_same_seed_gives_bit_identical_pairs(self) -> None:
        config = make_config()
        world = self._world(config)
        state = build_neural_policy_state(config, np.random.default_rng(5))

        def run() -> list[tuple[float, float]]:
            _, pairs, _spread = run_neural_residual_calibration_episode(
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
