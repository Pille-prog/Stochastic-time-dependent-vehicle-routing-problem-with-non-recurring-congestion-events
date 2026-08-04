"""Unit tests for :mod:`stdvrp.training.neural_report` (ticket 07, neural-policy).

Pure and torch-free (see the module's own docstring) — no ``neural`` marker,
runs in the default suite regardless of whether torch is installed.
``TestEvaluationReport`` and ``TestConvergence`` are the ticket's real
deliverables: the paired statistics the live report reads, and the
patience -> lr-cut -> converged state machine, both frozen in spec.md's
"Frozen parameters" table.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from stdvrp.training.neural_report import (
    EXCLUSION_RATE_STOP_THRESHOLD,
    LR_REDUCTION_FACTOR,
    MAX_REDUCTIONS,
    PATIENCE_BLOCKS,
    ConvergenceAction,
    ConvergenceState,
    EvaluationReport,
    candidate_spread_ratio,
    evaluation_cadence,
    exclusion_rate_stop_signal,
    format_episode_line,
    format_evaluation_block,
    format_lr,
    hit_safety_cap,
    open_training_log,
    paired_wilcoxon_p,
    update_convergence,
)


class TestPairedWilcoxonP:
    """The standalone helper ``EvaluationReport.wilcoxon_p`` and Gate A both use."""

    def test_small_p_for_a_consistent_large_effect(self) -> None:
        a = tuple(float(i) for i in range(1, 21))
        b = tuple(float(i) + 1000.0 for i in range(1, 21))
        assert paired_wilcoxon_p(a, b) < 0.05

    def test_nan_when_every_pair_ties_exactly(self) -> None:
        assert math.isnan(paired_wilcoxon_p((10.0, 20.0, 30.0), (10.0, 20.0, 30.0)))

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            paired_wilcoxon_p((1.0, 2.0), (1.0,))


def make_report(
    seed_costs: tuple[float, ...],
    reference_seed_costs: tuple[float, ...],
    episodes: int = 100,
    *,
    training_episodes: int = 0,
    excluded_episodes: int = 0,
    spread_samples: tuple[tuple[float, float], ...] = (),
) -> EvaluationReport:
    return EvaluationReport(
        episodes_completed=episodes,
        seed_costs=seed_costs,
        reference_seed_costs=reference_seed_costs,
        training_episodes=training_episodes,
        excluded_episodes=excluded_episodes,
        spread_samples=spread_samples,
    )


class TestCandidateSpreadRatio:
    """Ticket 17's companion diagnostic: r = sd_candidates(W.phi) / sd_candidates(c)."""

    def test_no_samples_is_nan(self) -> None:
        assert math.isnan(candidate_spread_ratio(()))

    def test_pools_the_mean_of_each_half(self) -> None:
        pairs = ((1.0, 10.0), (3.0, 10.0))  # mean residual sd 2.0, mean cost sd 10.0
        assert candidate_spread_ratio(pairs) == pytest.approx(0.2)

    def test_zero_mean_cost_spread_is_nan_not_a_crash(self) -> None:
        assert math.isnan(candidate_spread_ratio(((1.0, 0.0), (2.0, 0.0))))

    def test_ratio_near_zero_means_the_ranking_was_never_touched(self) -> None:
        assert candidate_spread_ratio(((0.001, 5.0),)) == pytest.approx(0.0002)

    def test_ratio_much_greater_than_one_means_overwriting_the_base(self) -> None:
        assert candidate_spread_ratio(((50.0, 5.0),)) == pytest.approx(10.0)


class TestEvaluationReport:
    def test_mean_delta_and_wins(self) -> None:
        report = make_report((100.0, 200.0, 300.0), (150.0, 150.0, 150.0))
        assert report.mean_cost == pytest.approx(200.0)
        assert report.reference_mean_cost == pytest.approx(150.0)
        assert report.delta == pytest.approx(50.0)
        assert report.delta_pct == pytest.approx(50.0 / 150.0 * 100)
        assert report.wins == 1  # only the 100.0 seed beat its 150.0 reference

    def test_improving_block_has_negative_delta_and_more_wins(self) -> None:
        report = make_report((10.0, 20.0, 30.0), (100.0, 100.0, 100.0))
        assert report.delta < 0
        assert report.wins == 3

    def test_hopeless_block_beats_no_seeds(self) -> None:
        report = make_report((500.0, 500.0, 500.0), (10.0, 10.0, 10.0))
        assert report.wins == 0
        assert report.delta_pct > 0

    def test_wilcoxon_p_is_small_for_a_consistent_large_effect(self) -> None:
        seed_costs = tuple(float(i) for i in range(1, 21))
        reference_seed_costs = tuple(float(i) + 1000.0 for i in range(1, 21))
        report = make_report(seed_costs, reference_seed_costs)
        assert report.wilcoxon_p < 0.05

    def test_wilcoxon_p_is_nan_when_every_pair_ties_exactly(self) -> None:
        report = make_report((10.0, 20.0, 30.0), (10.0, 20.0, 30.0))
        assert math.isnan(report.wilcoxon_p)

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="paired length"):
            EvaluationReport(
                episodes_completed=1, seed_costs=(1.0, 2.0), reference_seed_costs=(1.0,)
            )

    def test_empty_block_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            EvaluationReport(episodes_completed=1, seed_costs=(), reference_seed_costs=())

    def test_candidate_spread_ratio_defaults_to_nan_when_omitted(self) -> None:
        report = make_report((1.0,), (2.0,))
        assert report.spread_samples == ()
        assert math.isnan(report.candidate_spread_ratio)

    def test_candidate_spread_ratio_reads_the_pooled_samples(self) -> None:
        report = make_report((1.0,), (2.0,), spread_samples=((1.0, 10.0), (3.0, 10.0)))
        assert report.candidate_spread_ratio == pytest.approx(0.2)


class TestExclusionRate:
    """Ticket 16: the ridge estimator's exclusion count/rate, logged per block."""

    def test_defaults_to_zero_when_omitted(self) -> None:
        report = make_report((1.0,), (2.0,))
        assert report.training_episodes == 0
        assert report.excluded_episodes == 0
        assert report.exclusion_rate == 0.0

    def test_computes_the_rate(self) -> None:
        report = make_report((1.0,), (2.0,), training_episodes=50, excluded_episodes=5)
        assert report.exclusion_rate == pytest.approx(0.1)

    def test_zero_excluded_gives_zero_rate(self) -> None:
        report = make_report((1.0,), (2.0,), training_episodes=50, excluded_episodes=0)
        assert report.exclusion_rate == 0.0

    def test_all_excluded_gives_rate_one(self) -> None:
        report = make_report((1.0,), (2.0,), training_episodes=10, excluded_episodes=10)
        assert report.exclusion_rate == pytest.approx(1.0)

    def test_more_excluded_than_training_episodes_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            make_report((1.0,), (2.0,), training_episodes=5, excluded_episodes=6)


class TestExclusionRateStopSignal:
    def test_below_threshold_does_not_signal(self) -> None:
        report = make_report((1.0,), (2.0,), training_episodes=100, excluded_episodes=5)
        assert not exclusion_rate_stop_signal(report)

    def test_at_or_above_threshold_signals(self) -> None:
        report = make_report((1.0,), (2.0,), training_episodes=100, excluded_episodes=10)
        assert exclusion_rate_stop_signal(report)

    def test_default_threshold_matches_the_module_constant(self) -> None:
        just_under = make_report((1.0,), (2.0,), training_episodes=1000, excluded_episodes=99)
        just_over = make_report((1.0,), (2.0,), training_episodes=1000, excluded_episodes=101)
        assert not exclusion_rate_stop_signal(just_under, threshold=EXCLUSION_RATE_STOP_THRESHOLD)
        assert exclusion_rate_stop_signal(just_over, threshold=EXCLUSION_RATE_STOP_THRESHOLD)

    def test_a_healthy_zero_exclusion_rate_never_signals(self) -> None:
        report = make_report((1.0,), (2.0,), training_episodes=500, excluded_episodes=0)
        assert not exclusion_rate_stop_signal(report)

    def test_custom_threshold_is_respected(self) -> None:
        report = make_report((1.0,), (2.0,), training_episodes=100, excluded_episodes=2)
        assert exclusion_rate_stop_signal(report, threshold=0.01)
        assert not exclusion_rate_stop_signal(report, threshold=0.5)


class TestConvergence:
    """spec.md's frozen stopping rule: patience 5 -> lr*0.3 -> 3 reductions -> converged."""

    def test_an_improving_block_resets_patience_and_updates_best(self) -> None:
        state = ConvergenceState(current_lr=1e-3)
        report = make_report((100.0,) * 5, (200.0,) * 5, episodes=50)

        action = update_convergence(state, report)

        assert action == ConvergenceAction.CONTINUE
        assert state.best_mean_cost == pytest.approx(100.0)
        assert state.best_episode == 50
        assert state.blocks_without_improvement == 0

    def test_patience_blocks_without_improvement_reduce_the_lr(self) -> None:
        state = ConvergenceState(current_lr=1e-3)
        good = make_report((100.0,) * 3, (200.0,) * 3, episodes=50)
        update_convergence(state, good)

        # worse than best, never improves
        flat = make_report((150.0,) * 3, (200.0,) * 3, episodes=100)
        actions = [update_convergence(state, flat) for _ in range(PATIENCE_BLOCKS)]

        assert actions[:-1] == [ConvergenceAction.CONTINUE] * (PATIENCE_BLOCKS - 1)
        assert actions[-1] == ConvergenceAction.REDUCE_LR
        assert state.current_lr == pytest.approx(1e-3 * LR_REDUCTION_FACTOR)
        assert state.reductions_applied == 1
        assert state.blocks_without_improvement == 0  # reset by the trigger itself

    def test_three_reductions_then_a_fourth_patience_trigger_converges(self) -> None:
        state = ConvergenceState(current_lr=1e-3)
        good = make_report((100.0,) * 3, (200.0,) * 3, episodes=1)
        update_convergence(state, good)
        flat = make_report((150.0,) * 3, (200.0,) * 3, episodes=999)

        actions = []
        for _ in range(MAX_REDUCTIONS + 1):
            for _ in range(PATIENCE_BLOCKS):
                actions.append(update_convergence(state, flat))

        reduce_count = actions.count(ConvergenceAction.REDUCE_LR)
        converged_count = actions.count(ConvergenceAction.CONVERGED)
        assert reduce_count == MAX_REDUCTIONS
        assert converged_count == 1
        assert actions[-1] == ConvergenceAction.CONVERGED
        assert state.reductions_applied == MAX_REDUCTIONS

    def test_an_improvement_after_a_reduction_resets_patience_without_reverting_lr(self) -> None:
        state = ConvergenceState(current_lr=1e-3)
        update_convergence(state, make_report((100.0,), (200.0,), episodes=1))
        flat = make_report((150.0,), (200.0,), episodes=2)
        for _ in range(PATIENCE_BLOCKS):
            update_convergence(state, flat)
        assert state.reductions_applied == 1
        reduced_lr = state.current_lr

        better = make_report((50.0,), (200.0,), episodes=3)
        action = update_convergence(state, better)

        assert action == ConvergenceAction.CONTINUE
        assert state.current_lr == pytest.approx(reduced_lr)  # lr cut is not undone
        assert state.blocks_without_improvement == 0


class TestEvaluationCadence:
    def test_floors_at_the_minimum_for_short_runs(self) -> None:
        assert evaluation_cadence(0) == 50
        assert evaluation_cadence(500) == 50

    def test_grows_for_long_runs(self) -> None:
        assert evaluation_cadence(4000) == 100
        assert evaluation_cadence(10_000) == 250

    def test_overrides_are_respected(self) -> None:
        assert evaluation_cadence(1000, minimum=2, target_blocks=10) == 100
        assert evaluation_cadence(5, minimum=2, target_blocks=10) == 2


class TestSafetyCap:
    def test_neither_bound_reached(self) -> None:
        assert not hit_safety_cap(episodes_completed=100, elapsed_seconds=60.0)

    def test_episode_cap(self) -> None:
        assert hit_safety_cap(episodes_completed=10_000, elapsed_seconds=0.0)

    def test_hour_cap(self) -> None:
        assert hit_safety_cap(episodes_completed=0, elapsed_seconds=24.0 * 3600.0)

    def test_overrides_shrink_the_cap_for_tests(self) -> None:
        assert hit_safety_cap(episodes_completed=5, elapsed_seconds=0.0, max_episodes=5)
        assert not hit_safety_cap(episodes_completed=4, elapsed_seconds=0.0, max_episodes=5)


class TestFormatting:
    def test_episode_line_contains_every_field(self) -> None:
        line = format_episode_line(
            episode=348, seed=1348, cost=2691.3, loss=0.387, lr=3.0e-4, wall_clock_seconds=8.9
        )
        assert "348" in line
        assert "1348" in line
        assert "2691.3" in line
        assert "3.9e-01" in line  # scientific: a converging loss rounds to 0.000 at %.3f
        assert "lr 3.0e-4" in line  # not Python's default two-digit-exponent 3.0e-04
        assert "8.9s" in line

    def test_a_converging_loss_stays_readable(self) -> None:
        """The regression ticket 08's log ran into: at ``%.3f`` every loss a
        healthy run produces prints as ``0.000``, so the one live number that
        could show the fit collapsing showed nothing for 1 338 episodes."""
        healthy = format_episode_line(
            episode=1, seed=1000, cost=2500.0, loss=2.1e-4, lr=3.0e-4, wall_clock_seconds=1.0
        )
        collapsed = format_episode_line(
            episode=2, seed=1001, cost=2500.0, loss=2.1e-8, lr=3.0e-4, wall_clock_seconds=1.0
        )

        assert "2.1e-04" in healthy
        assert "2.1e-08" in collapsed

    def test_format_lr_matches_spec_mds_single_digit_exponent(self) -> None:
        assert format_lr(3.0e-4) == "3.0e-4"
        assert format_lr(1e-10) == "1.0e-10"
        assert format_lr(9.0e-5) == "9.0e-5"

    def test_evaluation_block_reports_mean_delta_wins_and_patience(self) -> None:
        report = make_report((10.0,) * 50, (20.0,) * 50, episodes=350)
        state = ConvergenceState(current_lr=3.0e-4)
        update_convergence(state, report)

        text = format_evaluation_block(report, state)

        assert "eval @350" in text
        assert "wins on 50/50 seeds" in text
        assert f"no improvement: 0/{PATIENCE_BLOCKS} blocks" in text

    def test_evaluation_block_handles_a_nan_wilcoxon_p_without_crashing(self) -> None:
        report = make_report((10.0, 20.0), (10.0, 20.0), episodes=50)  # exact tie -> NaN p
        state = ConvergenceState(current_lr=3.0e-4)
        update_convergence(state, report)

        text = format_evaluation_block(report, state)

        assert "Wilcoxon p=nan" in text

    def test_evaluation_block_reports_the_exclusion_rate_when_present(self) -> None:
        report = make_report(
            (10.0,) * 50, (20.0,) * 50, episodes=350, training_episodes=50, excluded_episodes=3
        )
        state = ConvergenceState(current_lr=3.0e-4)
        update_convergence(state, report)

        text = format_evaluation_block(report, state)

        assert "excluded 3/50 training episodes (6.0%)" in text

    def test_evaluation_block_omits_the_exclusion_line_when_absent(self) -> None:
        """Reports built without a training-episode count (every test predating
        ticket 16) must not suddenly print a nonsensical "0/0" line."""
        report = make_report((10.0,) * 50, (20.0,) * 50, episodes=350)
        state = ConvergenceState(current_lr=3.0e-4)
        update_convergence(state, report)

        text = format_evaluation_block(report, state)

        assert "excluded" not in text

    def test_evaluation_block_reports_r_when_spread_samples_are_present(self) -> None:
        report = make_report(
            (10.0,) * 50,
            (20.0,) * 50,
            episodes=350,
            spread_samples=((1.0, 10.0), (3.0, 10.0)),
        )
        state = ConvergenceState(current_lr=3.0e-4)
        update_convergence(state, report)

        text = format_evaluation_block(report, state)

        assert "r = sd(W.phi)/sd(c) = 0.200" in text

    def test_evaluation_block_omits_the_r_line_when_absent(self) -> None:
        """Reports built without any candidate-spread samples (every test
        predating ticket 17) must not print a NaN line every block."""
        report = make_report((10.0,) * 50, (20.0,) * 50, episodes=350)
        state = ConvergenceState(current_lr=3.0e-4)
        update_convergence(state, report)

        text = format_evaluation_block(report, state)

        assert "sd(W.phi)" not in text


class TestOpenTrainingLog:
    def test_writes_every_message_to_the_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        path = tmp_path / "run.log"
        log = open_training_log(path, echo=False)

        log("first message")
        log("second message")

        assert path.read_text(encoding="utf-8") == "first message\nsecond message\n"
        assert capsys.readouterr().out == ""  # echo=False: nothing printed

    def test_echoes_to_stdout_by_default(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        log = open_training_log(tmp_path / "run.log")
        log("visible message")
        assert "visible message" in capsys.readouterr().out

    def test_a_resumed_run_appends_rather_than_truncating(self, tmp_path: Path) -> None:
        path = tmp_path / "run.log"
        open_training_log(path, echo=False)("from the first session")
        open_training_log(path, echo=False)("from the resumed session")

        content = path.read_text(encoding="utf-8")
        assert "from the first session" in content
        assert "from the resumed session" in content

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "run.log"
        open_training_log(path, echo=False)("hello")
        assert path.exists()
