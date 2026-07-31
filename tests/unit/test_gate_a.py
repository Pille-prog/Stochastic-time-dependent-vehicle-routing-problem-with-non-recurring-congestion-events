"""Unit tests for :mod:`stdvrp.training.gate_a`'s statistics (ticket 08, neural-policy).

Pure and torch-free by construction: ``ArmResult``/``GateAResult`` are plain
dataclasses over already-computed cost/calibration numbers, so every test here
builds them directly rather than running a real simulated Episode --
``test_neural_episode.py`` and ``TestGateARealRun`` (this file, ``neural``
marked) cover the wiring that produces those numbers in the first place.

Threshold constants (5% effect, p<0.05, rho>=0.5, >=3 init seeds) are copied
from spec.md's "Frozen parameters" table and ticket 08's own acceptance table
-- not tuned here.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

# Same circular-import landmine as every other neural test file (see
# test_tokenizer.py) -- harmless and torch-free itself, so importing it
# unconditionally does not affect the torch-free stats tests below.
import stdvrp.simulation  # noqa: F401
from stdvrp.config import ExperimentConfig
from stdvrp.training.gate_a import (
    GATE_A_CALIBRATION_THRESHOLD,
    GATE_A_EFFECT_THRESHOLD_PCT,
    GATE_A_MIN_INIT_SEEDS,
    GATE_A_SIGNIFICANCE_THRESHOLD,
    ArmResult,
    GateAResult,
    format_gate_a_report,
    run_gate_a,
)
from stdvrp.training.reference_card import ReferenceCard

FIXTURE_CONFIG = Path(__file__).resolve().parents[1] / "fixtures" / "chengdu_mini" / "config.yaml"


def make_mini_config(**overrides: object) -> ExperimentConfig:
    config = ExperimentConfig.from_yaml(FIXTURE_CONFIG)
    values: dict[str, object] = {
        "neural_d_model": 8,
        "neural_n_layers": 1,
        "neural_n_heads": 2,
        "evaluation_seed_count": 2,
    }
    values.update(overrides)
    return dataclasses.replace(config, **values)


def make_dummy_reference_card(config: ExperimentConfig) -> ReferenceCard:
    """A reference card shaped for ``train_neural``'s live report -- Gate A's
    own statistics never read it, only its evaluation_seeds pairing."""
    seeds = config.evaluation_seeds
    return ReferenceCard(
        winning_budget=100,
        winning_test_action_count=2,
        test_seeds=config.test_seeds,
        test_seed_costs=tuple(500.0 for _ in config.test_seeds),
        evaluation_seeds=seeds,
        evaluation_seed_costs=tuple(500.0 for _ in seeds),
        best_w=(0.0,) * 19,
        config={},
        wall_clock_seconds={},
    )


def make_arm(
    *,
    init_seed: int = 0,
    trained: tuple[float, ...] = (90.0, 80.0, 95.0, 85.0, 88.0),
    untrained: tuple[float, ...] = (100.0, 100.0, 100.0, 100.0, 100.0),
    trained_calibration: tuple[tuple[float, float], ...] = (),
    untrained_calibration: tuple[tuple[float, float], ...] = (),
    episodes_completed: int = 100,
    converged: bool = True,
) -> ArmResult:
    return ArmResult(
        init_seed=init_seed,
        trained_seed_costs=trained,
        untrained_seed_costs=untrained,
        trained_calibration=trained_calibration,
        untrained_calibration=untrained_calibration,
        episodes_completed=episodes_completed,
        converged=converged,
    )


class TestArmResultValidation:
    def test_mismatched_seed_cost_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="paired"):
            make_arm(trained=(1.0, 2.0), untrained=(1.0,))

    def test_empty_seed_costs_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            make_arm(trained=(), untrained=())


class TestArmResultReduction:
    def test_seed_reductions_pct_is_the_per_seed_percent_improvement(self) -> None:
        arm = make_arm(trained=(90.0, 50.0), untrained=(100.0, 100.0))
        assert arm.seed_reductions_pct == pytest.approx((10.0, 50.0))

    def test_a_worse_trained_cost_is_a_negative_reduction(self) -> None:
        arm = make_arm(trained=(120.0,), untrained=(100.0,))
        assert arm.seed_reductions_pct == pytest.approx((-20.0,))

    def test_mean_and_median_differ_on_a_skewed_distribution(self) -> None:
        # One seed improves hugely, the rest barely -- exactly the "mean and
        # median will differ, and the difference is informative" case (ticket 08).
        arm = make_arm(
            trained=(10.0, 99.0, 99.0, 99.0, 99.0),
            untrained=(100.0, 100.0, 100.0, 100.0, 100.0),
        )
        assert arm.median_reduction_pct == pytest.approx(1.0)
        assert arm.mean_reduction_pct > arm.median_reduction_pct

    def test_zero_effect_gives_zero_reduction(self) -> None:
        arm = make_arm(trained=(100.0, 100.0), untrained=(100.0, 100.0))
        assert arm.mean_reduction_pct == pytest.approx(0.0)
        assert arm.median_reduction_pct == pytest.approx(0.0)


class TestArmResultWilcoxonAndNullModel:
    def test_wilcoxon_p_is_small_for_a_consistent_large_effect(self) -> None:
        trained = tuple(float(i) for i in range(1, 21))
        untrained = tuple(float(i) + 1000.0 for i in range(1, 21))
        arm = make_arm(trained=trained, untrained=untrained)
        assert arm.wilcoxon_p < 0.05

    def test_wilcoxon_p_is_nan_when_every_pair_ties(self) -> None:
        arm = make_arm(trained=(10.0, 20.0, 30.0), untrained=(10.0, 20.0, 30.0))
        assert math.isnan(arm.wilcoxon_p)

    def test_null_model_passes_needs_both_significance_and_effect_size(self) -> None:
        trained = tuple(float(i) for i in range(1, 21))
        # A consistent but tiny (<5%) effect: significant, but too small to pass.
        untrained = tuple(float(i) * 1.02 for i in range(1, 21))
        arm = make_arm(trained=trained, untrained=untrained)
        assert arm.wilcoxon_p < GATE_A_SIGNIFICANCE_THRESHOLD
        assert arm.mean_reduction_pct < GATE_A_EFFECT_THRESHOLD_PCT
        assert not arm.null_model_passes

    def test_null_model_passes_on_a_large_consistent_effect(self) -> None:
        trained = tuple(float(i) for i in range(1, 21))
        untrained = tuple(float(i) * 2.0 for i in range(1, 21))
        arm = make_arm(trained=trained, untrained=untrained)
        assert arm.null_model_passes

    def test_null_model_fails_when_the_effect_is_not_significant(self) -> None:
        # Alternating better/worse: a real ~10% mean effect, but not consistent
        # enough for Wilcoxon to call it significant at n=4.
        arm = make_arm(trained=(80.0, 120.0, 80.0, 120.0), untrained=(100.0, 100.0, 100.0, 100.0))
        assert arm.mean_reduction_pct == pytest.approx(0.0)
        assert not arm.null_model_passes

    def test_a_worse_trained_network_never_passes(self) -> None:
        trained = tuple(float(i) * 2.0 for i in range(1, 21))
        untrained = tuple(float(i) for i in range(1, 21))
        arm = make_arm(trained=trained, untrained=untrained)
        assert arm.mean_reduction_pct < 0
        assert not arm.null_model_passes


class TestArmResultCalibration:
    def test_perfect_agreement_gives_rho_one(self) -> None:
        pairs = tuple((float(i), float(i)) for i in range(10))
        arm = make_arm(trained_calibration=pairs)
        assert arm.trained_calibration_spearman == pytest.approx(1.0)

    def test_perfect_disagreement_gives_rho_minus_one(self) -> None:
        pairs = tuple((float(i), float(-i)) for i in range(10))
        arm = make_arm(trained_calibration=pairs)
        assert arm.trained_calibration_spearman == pytest.approx(-1.0)

    def test_untrained_calibration_is_read_independently_of_trained(self) -> None:
        trained_pairs = tuple((float(i), float(i)) for i in range(10))
        untrained_pairs = tuple((1.0, float(i)) for i in range(10))  # constant predictions
        arm = make_arm(trained_calibration=trained_pairs, untrained_calibration=untrained_pairs)
        assert arm.trained_calibration_spearman == pytest.approx(1.0)
        assert math.isnan(arm.untrained_calibration_spearman)

    def test_fewer_than_two_pairs_is_nan_not_a_crash(self) -> None:
        arm = make_arm(trained_calibration=((1.0, 1.0),))
        assert math.isnan(arm.trained_calibration_spearman)

    def test_no_pairs_is_nan(self) -> None:
        arm = make_arm(trained_calibration=())
        assert math.isnan(arm.trained_calibration_spearman)

    def test_calibration_passes_at_or_above_the_threshold(self) -> None:
        pairs = tuple((float(i), float(i)) for i in range(10))
        arm = make_arm(trained_calibration=pairs)
        assert arm.trained_calibration_spearman >= GATE_A_CALIBRATION_THRESHOLD
        assert arm.calibration_passes

    def test_calibration_fails_below_the_threshold(self) -> None:
        # A weakly-correlated, noisy relationship well under 0.5.
        predicted = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        realised = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
        arm = make_arm(trained_calibration=tuple(zip(predicted, realised, strict=True)))
        assert arm.trained_calibration_spearman < GATE_A_CALIBRATION_THRESHOLD
        assert not arm.calibration_passes


class TestGateAResultAggregation:
    def test_needs_at_least_one_arm(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            GateAResult(arms=())

    def test_mean_and_sd_of_the_per_arm_improvement(self) -> None:
        arms = (
            make_arm(init_seed=0, trained=(80.0,), untrained=(100.0,)),  # +20%
            make_arm(init_seed=1, trained=(90.0,), untrained=(100.0,)),  # +10%
            make_arm(init_seed=2, trained=(70.0,), untrained=(100.0,)),  # +30%
        )
        result = GateAResult(arms=arms)
        assert result.reproducibility_mean_pct == pytest.approx(20.0)
        assert result.reproducibility_sd_pct == pytest.approx(8.16496580927726)

    def test_a_consistent_positive_spread_does_not_straddle_zero(self) -> None:
        arms = (
            make_arm(init_seed=0, trained=(80.0,), untrained=(100.0,)),
            make_arm(init_seed=1, trained=(85.0,), untrained=(100.0,)),
            make_arm(init_seed=2, trained=(90.0,), untrained=(100.0,)),
        )
        result = GateAResult(arms=arms)
        assert not result.reproducibility_spread_straddles_zero

    def test_a_wide_spread_around_a_positive_mean_can_still_straddle_zero(self) -> None:
        # 5%, 5%, 100% -- every individual run clears the bar, but the spread
        # is wide enough that mean - sd crosses zero (ticket 08: "if the spread
        # straddles zero, it did not learn, whatever the best run says").
        arms = (
            make_arm(init_seed=0, trained=(95.0,), untrained=(100.0,)),
            make_arm(init_seed=1, trained=(95.0,), untrained=(100.0,)),
            make_arm(init_seed=2, trained=(0.0,), untrained=(100.0,)),
        )
        result = GateAResult(arms=arms)
        assert result.reproducibility_spread_straddles_zero

    def test_null_model_needs_at_least_three_arms(self) -> None:
        arms = (
            make_arm(
                init_seed=0,
                trained=tuple(float(i) for i in range(1, 21)),
                untrained=tuple(float(i) * 2.0 for i in range(1, 21)),
            ),
            make_arm(
                init_seed=1,
                trained=tuple(float(i) for i in range(1, 21)),
                untrained=tuple(float(i) * 2.0 for i in range(1, 21)),
            ),
        )
        result = GateAResult(arms=arms)
        assert len(arms) < GATE_A_MIN_INIT_SEEDS
        assert not result.null_model_passes
        assert not result.passes

    def test_one_failing_arm_fails_the_whole_null_model_gate(self) -> None:
        strong = make_arm(
            init_seed=0,
            trained=tuple(float(i) for i in range(1, 21)),
            untrained=tuple(float(i) * 2.0 for i in range(1, 21)),
        )
        weak = make_arm(
            init_seed=1, trained=(100.0,) * 20, untrained=(100.0,) * 20
        )  # no effect at all
        result = GateAResult(arms=(strong, strong, weak))
        assert not result.null_model_passes
        assert not result.passes

    def test_passes_requires_null_model_reproducibility_and_calibration(self) -> None:
        good_pairs = tuple((float(i), float(i)) for i in range(20))
        strong = make_arm(
            init_seed=0,
            trained=tuple(float(i) for i in range(1, 21)),
            untrained=tuple(float(i) * 2.0 for i in range(1, 21)),
            trained_calibration=good_pairs,
        )
        result = GateAResult(arms=(strong, strong, strong))
        assert result.null_model_passes
        assert not result.reproducibility_spread_straddles_zero
        assert result.reproducibility_passes
        assert result.calibration_passes
        assert result.passes

    def test_calibration_gate_is_independent_of_the_cost_gates(self) -> None:
        # Costs clear every bar, but calibration does not -- the whole point
        # of the calibration check (spec.md: "cannot be faked by a lucky run").
        bad_pairs = tuple(zip([0.0, 1.0] * 10, [0.0, 0.0, 1.0, 1.0] * 5, strict=True))
        strong = make_arm(
            init_seed=0,
            trained=tuple(float(i) for i in range(1, 21)),
            untrained=tuple(float(i) * 2.0 for i in range(1, 21)),
            trained_calibration=bad_pairs,
        )
        result = GateAResult(arms=(strong, strong, strong))
        assert result.null_model_passes
        assert result.reproducibility_passes
        assert not result.calibration_passes
        assert not result.passes


@pytest.mark.neural
class TestGateARealRun:
    """One real, tiny run of the whole protocol against the mini fixture.

    This is the ticket's own "develop and debug on the mini fixture" step --
    it proves the wiring (real Episodes, real network, real training loop),
    not a real "does it learn" result. The actual Gate A verdict runs against
    the real dataset, separately (ticket 08's Comments record those numbers).
    """

    def test_runs_end_to_end_and_produces_well_shaped_results(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from stdvrp.training.episode_pool import EpisodeWorld

        config = make_mini_config()
        world = EpisodeWorld.load(config)
        reference_card = make_dummy_reference_card(config)

        result = run_gate_a(
            world,
            reference_card=reference_card,
            checkpoint_dir=tmp_path,
            init_seeds=(0, 1),
            max_episodes=2,
            evaluation_cadence_minimum=2,
        )

        assert len(result.arms) == 2
        for arm in result.arms:
            assert len(arm.trained_seed_costs) == len(config.test_seeds)
            assert len(arm.untrained_seed_costs) == len(config.test_seeds)
            assert len(arm.trained_calibration) > 0
            assert all(math.isfinite(value) for pair in arm.trained_calibration for value in pair)
            assert all(math.isfinite(value) for pair in arm.untrained_calibration for value in pair)

        # The untrained (null) network is built once and shared across arms.
        assert result.arms[0].untrained_seed_costs == result.arms[1].untrained_seed_costs

        report = format_gate_a_report(result)
        assert "GATE A:" in report
        assert str(result.arms[0].episodes_completed) in report

    def test_checkpoints_land_one_per_init_seed(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from stdvrp.training.episode_pool import EpisodeWorld

        config = make_mini_config()
        world = EpisodeWorld.load(config)
        reference_card = make_dummy_reference_card(config)

        run_gate_a(
            world,
            reference_card=reference_card,
            checkpoint_dir=tmp_path,
            init_seeds=(0, 1),
            max_episodes=2,
            evaluation_cadence_minimum=2,
        )

        assert (tmp_path / "gate_a_init0.pt").exists()
        assert (tmp_path / "gate_a_init1.pt").exists()
