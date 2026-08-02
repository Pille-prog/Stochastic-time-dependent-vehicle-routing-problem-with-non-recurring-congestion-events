"""Trainer.train_neural's orchestration logic (ticket 07), isolated from real
episodes/networks via stubbed runners -- mirrors test_trainer.py's own pattern
for the linear baseline's train(). Real network construction and checkpoint
I/O stay real (fast: no simulation involved); only the episode *runners* are
stubbed, so these tests exercise the live report, convergence/patience state
machine, checkpointing and resume without running a single real Episode.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import stdvrp.simulation  # noqa: E402, F401 -- circular-import landmine, see test_tokenizer.py
import stdvrp.training.neural_episode as neural_episode_module  # noqa: E402
from stdvrp.config import ExperimentConfig  # noqa: E402
from stdvrp.simulation import EpisodeResult  # noqa: E402
from stdvrp.training import EpisodeWorld, ReferenceCard, Trainer  # noqa: E402

pytestmark = pytest.mark.neural

FIXTURE_CONFIG = Path(__file__).resolve().parents[1] / "fixtures" / "chengdu_mini" / "config.yaml"


def make_config(**overrides: Any) -> ExperimentConfig:
    config = ExperimentConfig.from_yaml(FIXTURE_CONFIG)
    values: dict[str, Any] = {
        "neural_d_model": 8,
        "neural_n_layers": 1,
        "neural_n_heads": 2,
        "evaluation_seed_count": 4,
        "first_train_seed": 1000,
    }
    values.update(overrides)
    return dataclasses.replace(config, **values)


def make_world(config: ExperimentConfig) -> EpisodeWorld:
    return EpisodeWorld(
        config=config,
        client_generator=object(),  # type: ignore[arg-type]
        travel_time_model=object(),  # type: ignore[arg-type]
        shortest_path_cache=object(),  # type: ignore[arg-type]
        congestion_generator=object(),  # type: ignore[arg-type]
    )


def episode_result(total_cost: float) -> EpisodeResult:
    return EpisodeResult(
        total_cost=total_cost,
        distance_cost=total_cost / 2,
        delay_cost=total_cost / 2,
        earliness_cost=0.0,
        overtime_cost=0.0,
        tau=700.0,
        state_count=10,
        delay_clients=0,
        earliness_clients=0,
        unserved_clients=0,
    )


def make_reference_card(config: ExperimentConfig, mean_cost: float = 500.0) -> ReferenceCard:
    seeds = config.evaluation_seeds
    return ReferenceCard(
        winning_budget=100,
        winning_test_action_count=2,
        test_seeds=config.test_seeds,
        test_seed_costs=tuple(mean_cost for _ in config.test_seeds),
        evaluation_seeds=seeds,
        evaluation_seed_costs=tuple(mean_cost + (i % 5) for i in range(len(seeds))),
        best_w=(0.0,) * 19,
        config={},
        wall_clock_seconds={},
    )


class SeedDeterministicTrainingStub:
    """Cost/loss purely a function of ``seed`` -- robust across a fresh instance
    (as a resumed run gets), matching how real per-seed reproducibility works.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> tuple[EpisodeResult, float]:
        self.calls.append(kwargs)
        seed = kwargs["seed"]
        return episode_result(500.0 - (seed % 100)), 0.01 * (seed % 7)


class SeedDeterministicEvaluationStub:
    """Cost purely a function of ``seed`` -- same property, for the resume test."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> EpisodeResult:
        self.calls.append(kwargs)
        seed = kwargs["seed"]
        return episode_result(500.0 + (seed % 5))


class ScriptedEvaluationStub:
    """Returns one scripted mean cost per evaluation block (small per-seed spread)."""

    def __init__(self, n_seeds: int, block_means: list[float]) -> None:
        self.calls: list[dict[str, Any]] = []
        self.n_seeds = n_seeds
        self.block_means = block_means

    def __call__(self, **kwargs: Any) -> EpisodeResult:
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        block = min(index // self.n_seeds, len(self.block_means) - 1)
        within_block = index % self.n_seeds
        return episode_result(self.block_means[block] + (within_block % 3) - 1)


class ExclusionStub:
    """Simulates a fraction of aborted Episodes by incrementing the ridge
    accumulator's exclusion counter directly (ticket 16) -- the real
    ``learn()`` never runs under this stub, so this is the only way to
    exercise ``Trainer.train_neural``'s exclusion-rate stop rule without a
    real Policy actually aborting an Episode."""

    def __init__(self, exclude_every: int) -> None:
        self.calls: list[dict[str, Any]] = []
        self.exclude_every = exclude_every

    def __call__(self, **kwargs: Any) -> tuple[EpisodeResult, float]:
        self.calls.append(kwargs)
        ridge = kwargs["policy_state"].ridge
        if len(self.calls) % self.exclude_every == 0:
            ridge.episodes_excluded += 1
        else:
            ridge.episodes_included += 1
        return episode_result(100.0), 0.01


class ConstantTrainingStub:
    def __init__(self, cost: float = 100.0, loss: float = 0.05) -> None:
        self.calls: list[dict[str, Any]] = []
        self.cost = cost
        self.loss = loss

    def __call__(self, **kwargs: Any) -> tuple[EpisodeResult, float]:
        self.calls.append(kwargs)
        return episode_result(self.cost), self.loss


def patch_runners(monkeypatch: pytest.MonkeyPatch, training: Any, evaluation: Any) -> None:
    monkeypatch.setattr(neural_episode_module, "run_neural_training_episode", training)
    monkeypatch.setattr(neural_episode_module, "run_neural_evaluation_episode", evaluation)


class TestLiveReport:
    def test_episode_and_evaluation_block_lines_are_logged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config()
        patch_runners(
            monkeypatch,
            ConstantTrainingStub(cost=100.0),
            ScriptedEvaluationStub(n_seeds=len(config.evaluation_seeds), block_means=[400.0]),
        )
        log_lines: list[str] = []
        trainer = Trainer(make_world(config), log=log_lines.append)

        trainer.train_neural(
            reference_card=make_reference_card(config, mean_cost=500.0),
            checkpoint_path=tmp_path / "ckpt.pt",
            max_episodes=4,
            evaluation_cadence_minimum=4,
        )

        episode_lines = [line for line in log_lines if line.startswith("[ep")]
        block_lines = [line for line in log_lines if "eval @" in line]
        assert len(episode_lines) == 4
        assert "cost 100.0" in episode_lines[0]
        assert any("eval @4" in line for line in block_lines)
        assert any("wins on" in line for line in log_lines)

    def test_evaluation_seeds_are_paired_in_config_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config(evaluation_seed_count=6)
        evaluation_stub = ScriptedEvaluationStub(
            n_seeds=len(config.evaluation_seeds), block_means=[300.0]
        )
        patch_runners(monkeypatch, ConstantTrainingStub(), evaluation_stub)
        trainer = Trainer(make_world(config), log=lambda _m: None)

        result = trainer.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=tmp_path / "ckpt.pt",
            max_episodes=4,
            evaluation_cadence_minimum=4,
        )

        requested_seeds = [call["seed"] for call in evaluation_stub.calls]
        assert requested_seeds == list(config.evaluation_seeds)
        assert result.evaluations[0].episodes_completed == 4


class TestConvergence:
    def test_a_flat_run_triggers_a_patience_lr_reduction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config()
        n_seeds = len(config.evaluation_seeds)
        # Block 0 sets the best; blocks 1..5 are flat (worse, never improve) --
        # the 5th flat block (6 total) triggers the first reduction.
        means = [300.0] + [400.0] * 6
        patch_runners(monkeypatch, ConstantTrainingStub(), ScriptedEvaluationStub(n_seeds, means))
        trainer = Trainer(make_world(config), log=lambda _m: None)

        result = trainer.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=tmp_path / "ckpt.pt",
            max_episodes=4 * len(means),
            evaluation_cadence_minimum=4,
        )

        assert result.policy_state.current_lr == pytest.approx(config.neural_learning_rate * 0.3)
        assert not result.converged

    def test_three_reductions_with_no_improvement_converges_before_the_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config()
        n_seeds = len(config.evaluation_seeds)
        # One good block, then far more flat blocks than needed to trigger all
        # three reductions plus the convergence-declaring fourth trigger.
        means = [300.0] + [400.0] * 25
        patch_runners(monkeypatch, ConstantTrainingStub(), ScriptedEvaluationStub(n_seeds, means))
        trainer = Trainer(make_world(config), log=lambda _m: None)

        result = trainer.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=tmp_path / "ckpt.pt",
            max_episodes=4 * len(means),  # generous cap the run should never reach
            evaluation_cadence_minimum=4,
        )

        assert result.converged
        assert result.policy_state.current_lr == pytest.approx(config.neural_learning_rate * 0.3**3)
        # Stopped well short of the generous cap because it converged instead.
        assert result.episodes_completed < 4 * len(means)

    def test_hitting_the_hard_cap_reports_not_converged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config()
        n_seeds = len(config.evaluation_seeds)
        # Always improving -- patience never triggers, only the cap can stop it.
        means = [500.0 - i for i in range(50)]
        patch_runners(monkeypatch, ConstantTrainingStub(), ScriptedEvaluationStub(n_seeds, means))
        trainer = Trainer(make_world(config), log=lambda _m: None)

        result = trainer.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=tmp_path / "ckpt.pt",
            max_episodes=8,
            evaluation_cadence_minimum=4,
        )

        assert not result.converged
        assert result.episodes_completed == 8


class TestExclusionRateStopRule:
    """Ticket 16: "a rising exclusion rate is a stop signal" -- the mandatory
    mitigation for the ridge estimator's blind spot to aborted Episodes.
    """

    def test_a_high_exclusion_rate_stops_the_run_uncoverged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config()
        n_seeds = len(config.evaluation_seeds)
        # Every training episode "aborts" -- an exclusion rate of 100%, far
        # past the stop threshold.
        patch_runners(
            monkeypatch,
            ExclusionStub(exclude_every=1),
            ScriptedEvaluationStub(n_seeds, [400.0] * 20),
        )
        trainer = Trainer(make_world(config), log=lambda _m: None)

        result = trainer.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=tmp_path / "ckpt.pt",
            max_episodes=100,
            evaluation_cadence_minimum=4,
        )

        assert result.stopped_for_exclusion_rate
        assert not result.converged
        assert result.episodes_completed < 100

    def test_a_healthy_run_never_stops_for_exclusion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config()
        n_seeds = len(config.evaluation_seeds)
        patch_runners(monkeypatch, ConstantTrainingStub(), ScriptedEvaluationStub(n_seeds, [400.0]))
        trainer = Trainer(make_world(config), log=lambda _m: None)

        result = trainer.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=tmp_path / "ckpt.pt",
            max_episodes=8,
            evaluation_cadence_minimum=4,
        )

        assert not result.stopped_for_exclusion_rate

    def test_the_exclusion_rate_is_reported_per_block_not_cumulatively(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A block's count must be a delta against the previous block, not the
        run's cumulative total -- otherwise a rate that stays flat forever
        would still be misreported as growing block over block. Both blocks
        stay under the stop threshold so the run reaches both without being
        cut short by :class:`TestExclusionRateStopRule`'s own trigger."""
        config = make_config()
        n_seeds = len(config.evaluation_seeds)
        # Block 1 (episodes 1-20): 1 excluded (5%, under the 10% ceiling).
        # Block 2 (episodes 21-40): none excluded.
        calls: list[dict[str, Any]] = []

        def training(**kwargs: Any) -> tuple[EpisodeResult, float]:
            calls.append(kwargs)
            ridge = kwargs["policy_state"].ridge
            if len(calls) == 1:
                ridge.episodes_excluded += 1
            else:
                ridge.episodes_included += 1
            return episode_result(100.0), 0.01

        patch_runners(monkeypatch, training, ScriptedEvaluationStub(n_seeds, [400.0, 390.0]))
        trainer = Trainer(make_world(config), log=lambda _m: None)

        result = trainer.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=tmp_path / "ckpt.pt",
            max_episodes=40,
            evaluation_cadence_minimum=20,
        )

        assert not result.stopped_for_exclusion_rate
        assert result.evaluations[0].training_episodes == 20
        assert result.evaluations[0].excluded_episodes == 1
        assert result.evaluations[0].exclusion_rate == pytest.approx(0.05)
        assert result.evaluations[1].training_episodes == 20
        assert result.evaluations[1].excluded_episodes == 0
        assert result.evaluations[1].exclusion_rate == 0.0


class TestCheckpointAndResume:
    def test_a_checkpoint_is_written_after_every_evaluation_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config()
        n_seeds = len(config.evaluation_seeds)
        patch_runners(
            monkeypatch,
            ConstantTrainingStub(),
            ScriptedEvaluationStub(n_seeds, [400.0, 390.0]),
        )
        checkpoint_path = tmp_path / "ckpt.pt"
        trainer = Trainer(make_world(config), log=lambda _m: None)

        trainer.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=checkpoint_path,
            max_episodes=8,
            evaluation_cadence_minimum=4,
        )

        assert checkpoint_path.exists()
        assert not checkpoint_path.with_name(checkpoint_path.name + ".tmp").exists()

    def test_resume_does_not_rerun_already_completed_episodes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config()
        checkpoint_path = tmp_path / "ckpt.pt"

        training_stub_1 = ConstantTrainingStub()
        patch_runners(
            monkeypatch,
            training_stub_1,
            ScriptedEvaluationStub(len(config.evaluation_seeds), [400.0]),
        )
        trainer_1 = Trainer(make_world(config), log=lambda _m: None)
        first = trainer_1.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=checkpoint_path,
            max_episodes=4,
            evaluation_cadence_minimum=4,
        )
        assert first.episodes_completed == 4

        training_stub_2 = ConstantTrainingStub()
        patch_runners(
            monkeypatch,
            training_stub_2,
            ScriptedEvaluationStub(len(config.evaluation_seeds), [400.0]),
        )
        trainer_2 = Trainer(make_world(config), log=lambda _m: None)
        second = trainer_2.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=checkpoint_path,
            resume=True,
            max_episodes=4,
            evaluation_cadence_minimum=4,
        )

        assert second.episodes_completed == 4
        assert len(training_stub_2.calls) == 0  # nothing left to run before the (unchanged) cap

    def test_interrupting_then_resuming_produces_an_identical_trajectory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config()

        # Uninterrupted reference run.
        patch_runners(
            monkeypatch, SeedDeterministicTrainingStub(), SeedDeterministicEvaluationStub()
        )
        trainer_a = Trainer(make_world(config), log=lambda _m: None)
        result_a = trainer_a.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=tmp_path / "a.pt",
            max_episodes=8,
            evaluation_cadence_minimum=2,
        )

        # Interrupted after the first block, then resumed to the same cap.
        checkpoint_b = tmp_path / "b.pt"
        patch_runners(
            monkeypatch, SeedDeterministicTrainingStub(), SeedDeterministicEvaluationStub()
        )
        trainer_b1 = Trainer(make_world(config), log=lambda _m: None)
        interrupted = trainer_b1.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=checkpoint_b,
            max_episodes=2,
            evaluation_cadence_minimum=2,
        )
        assert interrupted.episodes_completed == 2

        patch_runners(
            monkeypatch, SeedDeterministicTrainingStub(), SeedDeterministicEvaluationStub()
        )
        trainer_b2 = Trainer(make_world(config), log=lambda _m: None)
        result_b = trainer_b2.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=checkpoint_b,
            resume=True,
            max_episodes=8,
            evaluation_cadence_minimum=2,
        )

        assert result_a.episodes_completed == result_b.episodes_completed == 8
        assert len(result_a.evaluations) == len(result_b.evaluations)
        for ra, rb in zip(result_a.evaluations, result_b.evaluations, strict=True):
            assert ra.episodes_completed == rb.episodes_completed
            assert ra.seed_costs == rb.seed_costs

        params_a = list(result_a.policy_state.encoder.parameters())
        params_b = list(result_b.policy_state.encoder.parameters())
        for pa, pb in zip(params_a, params_b, strict=True):
            torch.testing.assert_close(pa, pb, atol=0.0, rtol=0.0)


class TestCudaResume:
    """Ticket 12: ticket 07's own interrupt/resume test (above), re-run with
    device=cuda -- same device throughout, never cross-device
    (neural_checkpoint.py refuses that; test_neural_checkpoint.py::TestDeviceGuard
    covers the refusal itself). Episode *runners* are stubbed here exactly as
    ticket 07's original test stubs them, so this exercises real network
    construction and real checkpoint state-dict I/O on CUDA, and confirms
    init-seed reproducibility holds on real hardware -- it does not exercise a
    real backward pass (see test_transformer_policy.py::TestLearn::
    test_learn_is_deterministic_on_cuda_with_the_same_seed for
    use_deterministic_algorithms(True) proven against a real backward pass on
    real CUDA hardware, tested for the first time; ticket 05 had no GPU
    available to test it against)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU on this machine")
    def test_interrupting_then_resuming_produces_an_identical_trajectory_on_cuda(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config(device="cuda")
        was_enabled = torch.are_deterministic_algorithms_enabled()
        try:
            # Uninterrupted reference run.
            patch_runners(
                monkeypatch, SeedDeterministicTrainingStub(), SeedDeterministicEvaluationStub()
            )
            trainer_a = Trainer(make_world(config), log=lambda _m: None)
            result_a = trainer_a.train_neural(
                reference_card=make_reference_card(config),
                checkpoint_path=tmp_path / "a.pt",
                max_episodes=8,
                evaluation_cadence_minimum=2,
            )

            # Interrupted after the first block, then resumed to the same cap.
            checkpoint_b = tmp_path / "b.pt"
            patch_runners(
                monkeypatch, SeedDeterministicTrainingStub(), SeedDeterministicEvaluationStub()
            )
            trainer_b1 = Trainer(make_world(config), log=lambda _m: None)
            interrupted = trainer_b1.train_neural(
                reference_card=make_reference_card(config),
                checkpoint_path=checkpoint_b,
                max_episodes=2,
                evaluation_cadence_minimum=2,
            )
            assert interrupted.episodes_completed == 2

            patch_runners(
                monkeypatch, SeedDeterministicTrainingStub(), SeedDeterministicEvaluationStub()
            )
            trainer_b2 = Trainer(make_world(config), log=lambda _m: None)
            result_b = trainer_b2.train_neural(
                reference_card=make_reference_card(config),
                checkpoint_path=checkpoint_b,
                resume=True,
                max_episodes=8,
                evaluation_cadence_minimum=2,
            )

            assert result_a.device == result_b.device == "cuda"
            assert result_a.episodes_completed == result_b.episodes_completed == 8
            assert len(result_a.evaluations) == len(result_b.evaluations)
            for ra, rb in zip(result_a.evaluations, result_b.evaluations, strict=True):
                assert ra.episodes_completed == rb.episodes_completed
                assert ra.seed_costs == rb.seed_costs

            params_a = list(result_a.policy_state.encoder.parameters())
            params_b = list(result_b.policy_state.encoder.parameters())
            for pa, pb in zip(params_a, params_b, strict=True):
                torch.testing.assert_close(pa, pb, atol=0.0, rtol=0.0)
        finally:
            torch.use_deterministic_algorithms(was_enabled)


class TestInitSeedOverride:
    """``init_seed`` (ticket 08, Gate A) varies only the network's weight init,
    holding ``first_train_seed``/the training episode sequence fixed -- the one
    real (non-stubbed) network in this file, kept to a single training episode
    for speed against the mini fixture."""

    def test_omitting_init_seed_reports_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config()
        n_seeds = len(config.evaluation_seeds)
        patch_runners(monkeypatch, ConstantTrainingStub(), ScriptedEvaluationStub(n_seeds, [400.0]))
        trainer = Trainer(make_world(config), log=lambda _m: None)

        result = trainer.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=tmp_path / "ckpt.pt",
            max_episodes=4,
            evaluation_cadence_minimum=4,
        )

        assert result.init_seed is None

    def test_an_explicit_init_seed_is_reported_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config()
        n_seeds = len(config.evaluation_seeds)
        patch_runners(monkeypatch, ConstantTrainingStub(), ScriptedEvaluationStub(n_seeds, [400.0]))
        trainer = Trainer(make_world(config), log=lambda _m: None)

        result = trainer.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=tmp_path / "ckpt.pt",
            max_episodes=4,
            evaluation_cadence_minimum=4,
            init_seed=7,
        )

        assert result.init_seed == 7

    def test_different_init_seeds_give_different_weights_after_one_real_episode(
        self, tmp_path: Path
    ) -> None:
        config = dataclasses.replace(
            ExperimentConfig.from_yaml(FIXTURE_CONFIG),
            neural_d_model=8,
            neural_n_layers=1,
            neural_n_heads=2,
            evaluation_seed_count=2,
        )
        world = EpisodeWorld.load(config)

        def train_one_episode(init_seed: int) -> list[torch.Tensor]:
            trainer = Trainer(world, log=lambda _m: None)
            result = trainer.train_neural(
                reference_card=make_reference_card(config),
                checkpoint_path=tmp_path / f"ckpt_{init_seed}.pt",
                max_episodes=1,
                evaluation_cadence_minimum=1,
                init_seed=init_seed,
            )
            return [p.clone() for p in result.policy_state.encoder.parameters()]

        params_a = train_one_episode(0)
        params_b = train_one_episode(1)

        assert any(not torch.equal(a, b) for a, b in zip(params_a, params_b, strict=True))

    def test_same_init_seed_gives_a_bit_identical_run(self, tmp_path: Path) -> None:
        config = dataclasses.replace(
            ExperimentConfig.from_yaml(FIXTURE_CONFIG),
            neural_d_model=8,
            neural_n_layers=1,
            neural_n_heads=2,
            evaluation_seed_count=2,
        )
        world = EpisodeWorld.load(config)

        def train_one_episode(run: int) -> list[torch.Tensor]:
            trainer = Trainer(world, log=lambda _m: None)
            result = trainer.train_neural(
                reference_card=make_reference_card(config),
                checkpoint_path=tmp_path / f"ckpt_{run}.pt",
                max_episodes=1,
                evaluation_cadence_minimum=1,
                init_seed=3,
            )
            return [p.clone() for p in result.policy_state.encoder.parameters()]

        params_a = train_one_episode(1)
        params_b = train_one_episode(2)

        for a, b in zip(params_a, params_b, strict=True):
            torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


class TestInterrupt:
    def test_keyboard_interrupt_leaves_a_valid_checkpoint_and_reraises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config()

        def exploding_training(**kwargs: Any) -> tuple[EpisodeResult, float]:
            raise KeyboardInterrupt

        patch_runners(
            monkeypatch,
            exploding_training,
            ScriptedEvaluationStub(len(config.evaluation_seeds), [400.0]),
        )
        checkpoint_path = tmp_path / "ckpt.pt"
        trainer = Trainer(make_world(config), log=lambda _m: None)

        with pytest.raises(KeyboardInterrupt):
            trainer.train_neural(
                reference_card=make_reference_card(config),
                checkpoint_path=checkpoint_path,
                max_episodes=100,
                evaluation_cadence_minimum=4,
            )

        # No evaluation block ever completed (the very first training episode
        # raised), so no checkpoint should exist -- nothing to resume from.
        assert not checkpoint_path.exists()

    def test_interrupt_after_a_checkpoint_leaves_that_checkpoint_resumable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config()
        checkpoint_path = tmp_path / "ckpt.pt"

        patch_runners(
            monkeypatch,
            ConstantTrainingStub(),
            ScriptedEvaluationStub(len(config.evaluation_seeds), [400.0]),
        )
        trainer_1 = Trainer(make_world(config), log=lambda _m: None)
        trainer_1.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=checkpoint_path,
            max_episodes=4,
            evaluation_cadence_minimum=4,
        )
        assert checkpoint_path.exists()

        calls_before_raise = {"n": 0}

        def flaky_training(**kwargs: Any) -> tuple[EpisodeResult, float]:
            calls_before_raise["n"] += 1
            if calls_before_raise["n"] == 2:
                raise KeyboardInterrupt
            return episode_result(100.0), 0.01

        evaluation_stub = ScriptedEvaluationStub(len(config.evaluation_seeds), [400.0])
        patch_runners(monkeypatch, flaky_training, evaluation_stub)
        trainer_2 = Trainer(make_world(config), log=lambda _m: None)
        with pytest.raises(KeyboardInterrupt):
            trainer_2.train_neural(
                reference_card=make_reference_card(config),
                checkpoint_path=checkpoint_path,
                resume=True,
                max_episodes=100,
                evaluation_cadence_minimum=4,
            )

        # The checkpoint from before the interrupt is still there and loadable.
        patch_runners(
            monkeypatch,
            ConstantTrainingStub(),
            ScriptedEvaluationStub(len(config.evaluation_seeds), [400.0]),
        )
        trainer_3 = Trainer(make_world(config), log=lambda _m: None)
        resumed = trainer_3.train_neural(
            reference_card=make_reference_card(config),
            checkpoint_path=checkpoint_path,
            resume=True,
            max_episodes=8,
            evaluation_cadence_minimum=4,
        )
        assert resumed.episodes_completed == 8
