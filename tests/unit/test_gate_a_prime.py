"""Unit tests for :mod:`stdvrp.training.gate_a_prime`'s statistics (ticket 17, neural-policy).

Pure and torch-free by construction, mirroring ``test_gate_a.py``'s own split:
``ArmResult``/``ArmSummary``/``GateAPrimeResult`` are plain dataclasses over
already-computed cost/calibration/spread numbers, so every test here builds
them directly rather than running a real simulated Episode --
``test_neural_episode.py`` covers the wiring that produces those numbers in
the first place, and ``TestGateAPrimeRealRun`` (this file, ``neural`` marked)
covers the end-to-end orchestration.

Threshold constants (5% effect, p<0.05, rho>=0.5, >=3 init seeds per arm) are
copied from spec.md's "Frozen parameters" table and ticket 17's own
acceptance table -- not tuned here.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

# Same circular-import landmine as every other neural test file (see
# test_tokenizer.py) -- harmless and torch-free itself.
import stdvrp.simulation  # noqa: F401
from stdvrp.config import ExperimentConfig
from stdvrp.training.gate_a_prime import (
    GATE_A_PRIME_CALIBRATION_THRESHOLD,
    GATE_A_PRIME_EFFECT_THRESHOLD_PCT,
    GATE_A_PRIME_MIN_INIT_SEEDS,
    GATE_A_PRIME_SIGNIFICANCE_THRESHOLD,
    ArmResult,
    ArmSummary,
    GateAPrimeResult,
    format_gate_a_prime_report,
    run_gate_a_prime,
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
    """A reference card shaped for ``train_neural``'s live report -- Gate A''s
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


def make_result(
    *,
    arm: str = "frozen",
    init_seed: int = 0,
    trained: tuple[float, ...] = (90.0, 80.0, 95.0, 85.0, 88.0),
    untrained: tuple[float, ...] = (100.0, 100.0, 100.0, 100.0, 100.0),
    residual_calibration: tuple[tuple[float, float], ...] = (),
    spread_samples: tuple[tuple[float, float], ...] = (),
    episodes_completed: int = 100,
    converged: bool = True,
) -> ArmResult:
    return ArmResult(
        arm=arm,
        init_seed=init_seed,
        trained_seed_costs=trained,
        untrained_seed_costs=untrained,
        residual_calibration=residual_calibration,
        spread_samples=spread_samples,
        episodes_completed=episodes_completed,
        converged=converged,
    )


class TestArmResultValidation:
    def test_invalid_arm_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            make_result(arm="blind")

    def test_mismatched_seed_cost_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="paired"):
            make_result(trained=(1.0, 2.0), untrained=(1.0,))

    def test_empty_seed_costs_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            make_result(trained=(), untrained=())


class TestArmResultReduction:
    def test_seed_reductions_pct_is_the_per_seed_percent_improvement(self) -> None:
        result = make_result(trained=(90.0, 50.0), untrained=(100.0, 100.0))
        assert result.seed_reductions_pct == pytest.approx((10.0, 50.0))

    def test_a_worse_trained_cost_is_a_negative_reduction(self) -> None:
        result = make_result(trained=(120.0,), untrained=(100.0,))
        assert result.seed_reductions_pct == pytest.approx((-20.0,))

    def test_mean_and_median_differ_on_a_skewed_distribution(self) -> None:
        result = make_result(
            trained=(10.0, 99.0, 99.0, 99.0, 99.0),
            untrained=(100.0, 100.0, 100.0, 100.0, 100.0),
        )
        assert result.median_reduction_pct == pytest.approx(1.0)
        assert result.mean_reduction_pct > result.median_reduction_pct

    def test_zero_effect_gives_zero_reduction(self) -> None:
        result = make_result(trained=(100.0, 100.0), untrained=(100.0, 100.0))
        assert result.mean_reduction_pct == pytest.approx(0.0)
        assert result.median_reduction_pct == pytest.approx(0.0)


class TestArmResultWilcoxonAndNullModel:
    def test_wilcoxon_p_is_small_for_a_consistent_large_effect(self) -> None:
        trained = tuple(float(i) for i in range(1, 21))
        untrained = tuple(float(i) + 1000.0 for i in range(1, 21))
        result = make_result(trained=trained, untrained=untrained)
        assert result.wilcoxon_p < 0.05

    def test_wilcoxon_p_is_nan_when_every_pair_ties(self) -> None:
        result = make_result(trained=(10.0, 20.0, 30.0), untrained=(10.0, 20.0, 30.0))
        assert math.isnan(result.wilcoxon_p)

    def test_null_model_passes_needs_both_significance_and_effect_size(self) -> None:
        trained = tuple(float(i) for i in range(1, 21))
        untrained = tuple(float(i) * 1.02 for i in range(1, 21))  # significant, <5%
        result = make_result(trained=trained, untrained=untrained)
        assert result.wilcoxon_p < GATE_A_PRIME_SIGNIFICANCE_THRESHOLD
        assert result.mean_reduction_pct < GATE_A_PRIME_EFFECT_THRESHOLD_PCT
        assert not result.null_model_passes

    def test_null_model_passes_on_a_large_consistent_effect(self) -> None:
        trained = tuple(float(i) for i in range(1, 21))
        untrained = tuple(float(i) * 2.0 for i in range(1, 21))
        result = make_result(trained=trained, untrained=untrained)
        assert result.null_model_passes

    def test_a_worse_trained_network_never_passes(self) -> None:
        trained = tuple(float(i) * 2.0 for i in range(1, 21))
        untrained = tuple(float(i) for i in range(1, 21))
        result = make_result(trained=trained, untrained=untrained)
        assert result.mean_reduction_pct < 0
        assert not result.null_model_passes


class TestArmResultCalibration:
    """Ticket 17's redefinition: rho(W.phi, y~), not rho(Q, U_t)."""

    def test_perfect_agreement_gives_rho_one(self) -> None:
        pairs = tuple((float(i), float(i)) for i in range(10))
        result = make_result(residual_calibration=pairs)
        assert result.calibration_spearman == pytest.approx(1.0)

    def test_perfect_disagreement_gives_rho_minus_one(self) -> None:
        pairs = tuple((float(i), float(-i)) for i in range(10))
        result = make_result(residual_calibration=pairs)
        assert result.calibration_spearman == pytest.approx(-1.0)

    def test_w_zero_gives_a_constant_predicted_half_and_therefore_nan(self) -> None:
        """The guaranteed non-PASS spec.md's redefinition needs: at W = 0 the
        predicted residual is identically zero for every sample, so the
        correlation is undefined (NaN) rather than a false PASS -- the exact
        failure ticket 08's ``rho(Q, U_t)`` pairing had (Q still varies with
        c even at W=0, so that pairing could pass with nothing learned)."""
        pairs = tuple((0.0, float(i)) for i in range(10))
        result = make_result(residual_calibration=pairs)
        assert math.isnan(result.calibration_spearman)
        assert not result.calibration_passes

    def test_fewer_than_two_pairs_is_nan_not_a_crash(self) -> None:
        result = make_result(residual_calibration=((1.0, 1.0),))
        assert math.isnan(result.calibration_spearman)

    def test_no_pairs_is_nan(self) -> None:
        result = make_result(residual_calibration=())
        assert math.isnan(result.calibration_spearman)

    def test_calibration_passes_at_or_above_the_threshold(self) -> None:
        pairs = tuple((float(i), float(i)) for i in range(10))
        result = make_result(residual_calibration=pairs)
        assert result.calibration_spearman >= GATE_A_PRIME_CALIBRATION_THRESHOLD
        assert result.calibration_passes

    def test_calibration_fails_below_the_threshold(self) -> None:
        predicted = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        realised = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
        result = make_result(residual_calibration=tuple(zip(predicted, realised, strict=True)))
        assert result.calibration_spearman < GATE_A_PRIME_CALIBRATION_THRESHOLD
        assert not result.calibration_passes


class TestArmResultCandidateSpreadRatio:
    def test_delegates_to_the_shared_helper(self) -> None:
        result = make_result(spread_samples=((1.0, 10.0), (3.0, 10.0)))
        assert result.candidate_spread_ratio == pytest.approx(0.2)

    def test_no_samples_is_nan(self) -> None:
        result = make_result(spread_samples=())
        assert math.isnan(result.candidate_spread_ratio)


class TestArmSummaryAggregation:
    def test_needs_at_least_one_result(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ArmSummary(arm="frozen", results=())

    def test_rejects_an_arm_mismatched_result(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            ArmSummary(arm="frozen", results=(make_result(arm="trained"),))

    def test_mean_and_sd_of_the_per_seed_improvement(self) -> None:
        results = (
            make_result(init_seed=0, trained=(80.0,), untrained=(100.0,)),  # +20%
            make_result(init_seed=1, trained=(90.0,), untrained=(100.0,)),  # +10%
            make_result(init_seed=2, trained=(70.0,), untrained=(100.0,)),  # +30%
        )
        summary = ArmSummary(arm="frozen", results=results)
        assert summary.reproducibility_mean_pct == pytest.approx(20.0)
        assert summary.reproducibility_sd_pct == pytest.approx(8.16496580927726)

    def test_a_consistent_positive_spread_does_not_straddle_zero(self) -> None:
        results = (
            make_result(init_seed=0, trained=(80.0,), untrained=(100.0,)),
            make_result(init_seed=1, trained=(85.0,), untrained=(100.0,)),
            make_result(init_seed=2, trained=(90.0,), untrained=(100.0,)),
        )
        summary = ArmSummary(arm="frozen", results=results)
        assert not summary.reproducibility_spread_straddles_zero

    def test_a_wide_spread_around_a_positive_mean_can_still_straddle_zero(self) -> None:
        results = (
            make_result(init_seed=0, trained=(95.0,), untrained=(100.0,)),
            make_result(init_seed=1, trained=(95.0,), untrained=(100.0,)),
            make_result(init_seed=2, trained=(0.0,), untrained=(100.0,)),
        )
        summary = ArmSummary(arm="frozen", results=results)
        assert summary.reproducibility_spread_straddles_zero

    def test_null_model_needs_at_least_three_seeds(self) -> None:
        results = (
            make_result(
                init_seed=0,
                trained=tuple(float(i) for i in range(1, 21)),
                untrained=tuple(float(i) * 2.0 for i in range(1, 21)),
            ),
            make_result(
                init_seed=1,
                trained=tuple(float(i) for i in range(1, 21)),
                untrained=tuple(float(i) * 2.0 for i in range(1, 21)),
            ),
        )
        summary = ArmSummary(arm="frozen", results=results)
        assert len(results) < GATE_A_PRIME_MIN_INIT_SEEDS
        assert not summary.null_model_passes
        assert not summary.passes

    def test_one_failing_seed_fails_the_whole_null_model_gate(self) -> None:
        strong = make_result(
            init_seed=0,
            trained=tuple(float(i) for i in range(1, 21)),
            untrained=tuple(float(i) * 2.0 for i in range(1, 21)),
        )
        weak = make_result(init_seed=1, trained=(100.0,) * 20, untrained=(100.0,) * 20)
        summary = ArmSummary(arm="frozen", results=(strong, strong, weak))
        assert not summary.null_model_passes
        assert not summary.passes

    def test_passes_requires_null_model_reproducibility_and_calibration(self) -> None:
        good_pairs = tuple((float(i), float(i)) for i in range(20))
        strong = make_result(
            init_seed=0,
            trained=tuple(float(i) for i in range(1, 21)),
            untrained=tuple(float(i) * 2.0 for i in range(1, 21)),
            residual_calibration=good_pairs,
        )
        summary = ArmSummary(arm="frozen", results=(strong, strong, strong))
        assert summary.null_model_passes
        assert not summary.reproducibility_spread_straddles_zero
        assert summary.reproducibility_passes
        assert summary.calibration_passes
        assert summary.passes

    def test_calibration_gate_is_independent_of_the_cost_gates(self) -> None:
        bad_pairs = tuple(zip([0.0, 1.0] * 10, [0.0, 0.0, 1.0, 1.0] * 5, strict=True))
        strong = make_result(
            init_seed=0,
            trained=tuple(float(i) for i in range(1, 21)),
            untrained=tuple(float(i) * 2.0 for i in range(1, 21)),
            residual_calibration=bad_pairs,
        )
        summary = ArmSummary(arm="frozen", results=(strong, strong, strong))
        assert summary.null_model_passes
        assert summary.reproducibility_passes
        assert not summary.calibration_passes
        assert not summary.passes

    def test_mean_candidate_spread_ratio_pools_across_seeds(self) -> None:
        results = (
            make_result(init_seed=0, spread_samples=((1.0, 10.0),)),  # r=0.1
            make_result(init_seed=1, spread_samples=((3.0, 10.0),)),  # r=0.3
            make_result(init_seed=2, spread_samples=()),  # NaN -- dropped
        )
        summary = ArmSummary(arm="frozen", results=results)
        assert summary.mean_candidate_spread_ratio == pytest.approx(0.2)

    def test_mean_candidate_spread_ratio_is_nan_when_every_seed_is_nan(self) -> None:
        results = (
            make_result(init_seed=0, spread_samples=()),
            make_result(init_seed=1, spread_samples=()),
        )
        summary = ArmSummary(arm="frozen", results=results)
        assert math.isnan(summary.mean_candidate_spread_ratio)


class TestGateAPrimeResultDecomposition:
    def _passing_summary(self, arm: str) -> ArmSummary:
        good_pairs = tuple((float(i), float(i)) for i in range(20))
        strong = make_result(
            arm=arm,
            init_seed=0,
            trained=tuple(float(i) for i in range(1, 21)),
            untrained=tuple(float(i) * 2.0 for i in range(1, 21)),
            residual_calibration=good_pairs,
        )
        return ArmSummary(arm=arm, results=(strong, strong, strong))

    def _failing_summary(self, arm: str) -> ArmSummary:
        weak = make_result(arm=arm, trained=(100.0,) * 20, untrained=(100.0,) * 20)
        return ArmSummary(arm=arm, results=(weak, weak, weak))

    def test_rejects_a_mislabeled_frozen_summary(self) -> None:
        with pytest.raises(ValueError, match="frozen"):
            GateAPrimeResult(
                frozen=self._passing_summary("trained"), trained=self._passing_summary("trained")
            )

    def test_rejects_a_mislabeled_trained_summary(self) -> None:
        with pytest.raises(ValueError, match="trained"):
            GateAPrimeResult(
                frozen=self._passing_summary("frozen"), trained=self._passing_summary("frozen")
            )

    def test_representation_learning_delta_is_signed_and_explicit(self) -> None:
        frozen = ArmSummary(
            arm="frozen",
            results=(make_result(arm="frozen", trained=(90.0,), untrained=(100.0,)),) * 3,
        )  # +10%
        trained = ArmSummary(
            arm="trained",
            results=(make_result(arm="trained", trained=(75.0,), untrained=(100.0,)),) * 3,
        )  # +25%
        result = GateAPrimeResult(frozen=frozen, trained=trained)

        assert result.representation_learning_delta_pct == pytest.approx(15.0)

    def test_a_worse_trained_arm_gives_a_negative_delta(self) -> None:
        frozen = ArmSummary(
            arm="frozen",
            results=(make_result(arm="frozen", trained=(75.0,), untrained=(100.0,)),) * 3,
        )  # +25%
        trained = ArmSummary(
            arm="trained",
            results=(make_result(arm="trained", trained=(90.0,), untrained=(100.0,)),) * 3,
        )  # +10%
        result = GateAPrimeResult(frozen=frozen, trained=trained)

        assert result.representation_learning_delta_pct == pytest.approx(-15.0)

    def test_passes_if_either_arm_passes(self) -> None:
        result = GateAPrimeResult(
            frozen=self._passing_summary("frozen"), trained=self._failing_summary("trained")
        )
        assert result.frozen.passes
        assert not result.trained.passes
        assert result.passes

    def test_fails_only_if_neither_arm_passes(self) -> None:
        result = GateAPrimeResult(
            frozen=self._failing_summary("frozen"), trained=self._failing_summary("trained")
        )
        assert not result.passes


class TestFormatGateAPrimeReport:
    def test_report_names_both_arms_and_the_decomposition(self) -> None:
        good_pairs = tuple((float(i), float(i)) for i in range(20))
        strong = make_result(
            arm="frozen",
            trained=tuple(float(i) for i in range(1, 21)),
            untrained=tuple(float(i) * 2.0 for i in range(1, 21)),
            residual_calibration=good_pairs,
        )
        weak = make_result(arm="trained", trained=(100.0,) * 20, untrained=(100.0,) * 20)
        result = GateAPrimeResult(
            frozen=ArmSummary(arm="frozen", results=(strong, strong, strong)),
            trained=ArmSummary(arm="trained", results=(weak, weak, weak)),
        )

        report = format_gate_a_prime_report(result)

        assert "Arm: frozen" in report
        assert "Arm: trained" in report
        assert "delta(trained) - delta(frozen)" in report
        assert "GATE A':" in report
        assert "PASS" in report  # frozen passes, so the overall gate passes

    def test_report_names_the_null_and_that_it_is_not_a_warm_start(self) -> None:
        """spec.md's standing obligation: "report the null alongside the
        trained number, always, and name which warm start produced it" --
        under the residual decomposition there is only ever one null (the
        myopic base), so this names that explicitly rather than a warm start
        choice that no longer exists (ticket 15)."""
        strong = make_result(
            arm="frozen",
            trained=tuple(float(i) for i in range(1, 21)),
            untrained=tuple(float(i) * 2.0 for i in range(1, 21)),
        )
        weak = make_result(arm="trained", trained=(100.0,) * 20, untrained=(100.0,) * 20)
        result = GateAPrimeResult(
            frozen=ArmSummary(arm="frozen", results=(strong, strong, strong)),
            trained=ArmSummary(arm="trained", results=(weak, weak, weak)),
        )

        report = format_gate_a_prime_report(result)

        assert "myopic base" in report
        assert "neural_warm_start" in report
        assert f"{result.null_mean_cost:.4f}" in report


class TestNullMeanCost:
    def test_reads_off_the_frozen_arms_shared_untrained_costs(self) -> None:
        three_costs = (90.0, 80.0, 95.0)
        result = make_result(arm="frozen", trained=three_costs, untrained=(100.0, 200.0, 300.0))
        summary = ArmSummary(arm="frozen", results=(result,))
        trained_summary = ArmSummary(
            arm="trained",
            results=(
                make_result(arm="trained", trained=three_costs, untrained=(100.0, 200.0, 300.0)),
            ),
        )
        gate_result = GateAPrimeResult(frozen=summary, trained=trained_summary)

        assert gate_result.null_mean_cost == pytest.approx(200.0)


@pytest.mark.neural
class TestGateAPrimeRealRun:
    """One real, tiny run of the whole protocol (both arms) against the mini
    fixture -- proves the wiring (real Episodes, real network, real training
    loop, both arms), not a real "does training add value" result. The
    actual Gate A' verdict runs against the real dataset, separately."""

    def test_runs_both_arms_end_to_end_and_produces_well_shaped_results(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("torch")
        from stdvrp.training.episode_pool import EpisodeWorld

        config = make_mini_config()
        world = EpisodeWorld.load(config)
        reference_card = make_dummy_reference_card(config)

        result = run_gate_a_prime(
            world,
            reference_card=reference_card,
            checkpoint_dir=tmp_path,
            init_seeds=(0, 1),
            max_episodes=2,
            evaluation_cadence_minimum=2,
        )

        for summary in (result.frozen, result.trained):
            assert len(summary.results) == 2
            for arm_result in summary.results:
                assert len(arm_result.trained_seed_costs) == len(config.test_seeds)
                assert len(arm_result.untrained_seed_costs) == len(config.test_seeds)
                assert len(arm_result.residual_calibration) > 0
                assert all(
                    math.isfinite(value)
                    for pair in arm_result.residual_calibration
                    for value in pair
                )

        # The untrained (null) network is built once and shared across both
        # arms and every init seed.
        assert (
            result.frozen.results[0].untrained_seed_costs
            == result.trained.results[0].untrained_seed_costs
        )

        report = format_gate_a_prime_report(result)
        assert "GATE A':" in report

    def test_checkpoints_land_one_per_arm_per_init_seed(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from stdvrp.training.episode_pool import EpisodeWorld

        config = make_mini_config()
        world = EpisodeWorld.load(config)
        reference_card = make_dummy_reference_card(config)

        run_gate_a_prime(
            world,
            reference_card=reference_card,
            checkpoint_dir=tmp_path,
            init_seeds=(0, 1),
            max_episodes=2,
            evaluation_cadence_minimum=2,
        )

        for arm in ("frozen", "trained"):
            for init_seed in (0, 1):
                assert (tmp_path / f"gate_a_prime_{arm}_init{init_seed}.pt").exists()
