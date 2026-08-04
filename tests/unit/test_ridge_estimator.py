"""Unit tests for :mod:`stdvrp.policies.ridge_estimator` (ticket 16, neural-policy).

Pure numpy, no torch import anywhere in the module under test -- no ``neural``
marker, same discipline as ``test_neural_report.py``.

``TestSolveIsRidgeRegression`` and ``TestStandardizationProtectsSmallScaleColumns``
are the ticket's real deliverables: the accumulator must actually solve the
normal equations it claims to (checked against plain OLS at a near-zero ridge),
and standardizing before regularizing must be what keeps a small-scale column's
recovered coefficient from being crushed relative to a large-scale column's --
the "main technical risk" the ticket calls out by name.
"""

from __future__ import annotations

import numpy as np
import pytest

# Same circular-import landmine as test_tokenizer.py/test_network.py (ticket 03's
# Comments): stdvrp.simulation must finish initializing first.
import stdvrp.simulation  # noqa: F401
from stdvrp.policies.ridge_estimator import RidgeAccumulator


class TestConstruction:
    def test_zeros_starts_at_the_zero_solve(self) -> None:
        accumulator = RidgeAccumulator.zeros(5, forgetting=0.98, ridge=1.0)
        np.testing.assert_array_equal(accumulator.solve(), np.zeros(5))

    def test_zero_solve_needs_no_observations(self) -> None:
        """Ticket 15's null must hold before anything has ever been observed --
        not just before the first *solve* call."""
        accumulator = RidgeAccumulator.zeros(3, forgetting=0.9, ridge=0.5)
        assert accumulator.episodes_included == 0
        np.testing.assert_array_equal(accumulator.solve(), np.zeros(3))

    @pytest.mark.parametrize("dim", [0, -1])
    def test_nonpositive_dim_is_refused(self, dim: int) -> None:
        with pytest.raises(ValueError, match="dim"):
            RidgeAccumulator.zeros(dim, forgetting=0.9, ridge=1.0)

    @pytest.mark.parametrize("forgetting", [0.0, -0.1, 1.1])
    def test_forgetting_out_of_range_is_refused(self, forgetting: float) -> None:
        with pytest.raises(ValueError, match="forgetting"):
            RidgeAccumulator.zeros(3, forgetting=forgetting, ridge=1.0)

    @pytest.mark.parametrize("ridge", [0.0, -1.0])
    def test_nonpositive_ridge_is_refused(self, ridge: float) -> None:
        with pytest.raises(ValueError, match="ridge"):
            RidgeAccumulator.zeros(3, forgetting=0.9, ridge=ridge)

    def test_forgetting_of_exactly_one_is_allowed(self) -> None:
        """No forgetting at all (infinite window) is a legal, if extreme, choice."""
        RidgeAccumulator.zeros(3, forgetting=1.0, ridge=1.0)


def _synthetic_regression(
    rng: np.random.Generator, *, n_samples: int, dim: int, true_w: np.ndarray, noise: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    phi = rng.normal(size=(n_samples, dim))
    y = phi @ true_w + noise * rng.normal(size=n_samples)
    return phi, y


class TestSolveIsRidgeRegression:
    """At a near-zero ridge and a well-conditioned design, the accumulator's
    solve must agree with plain least squares -- standardization changes the
    *regularization*, not the equation being solved in the ridge -> 0 limit.
    """

    def test_matches_ordinary_least_squares_at_a_negligible_ridge(self) -> None:
        rng = np.random.default_rng(0)
        dim = 4
        true_w = np.array([2.0, -1.0, 0.5, 3.0])
        phi, y = _synthetic_regression(rng, n_samples=2000, dim=dim, true_w=true_w)

        accumulator = RidgeAccumulator.zeros(dim, forgetting=1.0, ridge=1e-9)
        accumulator.observe_episode(phi, y, aborted=False)
        solved = accumulator.solve()

        expected, *_ = np.linalg.lstsq(phi, y, rcond=None)
        np.testing.assert_allclose(solved, expected, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(solved, true_w, atol=1e-2, rtol=1e-2)

    def test_splitting_the_same_data_across_episodes_gives_the_same_solve(self) -> None:
        """The accumulator must be equivalent to one batch solve -- accumulating
        in pieces (what actually happens, one Episode at a time) cannot change
        the answer for a fixed total dataset."""
        rng = np.random.default_rng(1)
        dim = 3
        true_w = np.array([1.0, 1.0, 1.0])
        phi, y = _synthetic_regression(rng, n_samples=300, dim=dim, true_w=true_w, noise=0.1)

        whole = RidgeAccumulator.zeros(dim, forgetting=1.0, ridge=0.1)
        whole.observe_episode(phi, y, aborted=False)

        split = RidgeAccumulator.zeros(dim, forgetting=1.0, ridge=0.1)
        for start in range(0, 300, 30):
            split.observe_episode(phi[start : start + 30], y[start : start + 30], aborted=False)

        np.testing.assert_allclose(whole.solve(), split.solve(), atol=1e-8, rtol=1e-8)

    def test_ridge_shrinks_toward_zero_as_lambda_grows(self) -> None:
        rng = np.random.default_rng(2)
        dim = 3
        true_w = np.array([5.0, -5.0, 5.0])
        phi, y = _synthetic_regression(rng, n_samples=500, dim=dim, true_w=true_w)

        light = RidgeAccumulator.zeros(dim, forgetting=1.0, ridge=1e-6)
        light.observe_episode(phi, y, aborted=False)
        heavy = RidgeAccumulator.zeros(dim, forgetting=1.0, ridge=1e6)
        heavy.observe_episode(phi, y, aborted=False)

        assert np.linalg.norm(heavy.solve()) < np.linalg.norm(light.solve())


class TestStandardizationProtectsSmallScaleColumns:
    """The ticket's own "main technical risk": with columns at wildly different
    natural scales, a *raw* (unstandardized) ridge shrinks the small-scale
    column's coefficient far more than the large-scale one's -- "the action
    weights shrink toward zero ... and you have rebuilt the exact failure this
    effort has been measuring for two months". Standardizing first is what
    the accumulator must do to avoid it.
    """

    def _scaled_world(
        self,
        rng: np.random.Generator,
        *,
        n_samples: int,
        small_scale: float,
        large_scale: float,
        noise: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        true_w = np.array([1.0, 1.0])
        small_column = rng.normal(size=n_samples) * small_scale
        large_column = rng.normal(size=n_samples) * large_scale
        phi = np.stack([small_column, large_column], axis=1)
        y = phi @ true_w + noise * rng.normal(size=n_samples)
        return phi, y, true_w

    # Chosen so raw_lambda is comparable to the *small* column's own raw
    # variance (n * small_scale**2 ~ 125) -- enough to crush it in raw units --
    # while staying a small fraction of either column's *standardized*
    # diagonal (~n = 2000). A little target noise avoids the degenerate
    # near-exact-cancellation a perfectly noiseless two-column fit produces
    # (large finite-sample sensitivity to any perturbation, ridge included,
    # unrelated to the standardization question this test asks).
    _N_SAMPLES = 2000
    _SMALL_SCALE = 0.05
    _LARGE_SCALE = 5.0
    _RIDGE = 50.0

    def test_recovers_the_small_scale_coefficient_reasonably_under_a_real_ridge(self) -> None:
        rng = np.random.default_rng(3)
        phi, y, true_w = self._scaled_world(
            rng,
            n_samples=self._N_SAMPLES,
            small_scale=self._SMALL_SCALE,
            large_scale=self._LARGE_SCALE,
            noise=0.01,
        )

        accumulator = RidgeAccumulator.zeros(2, forgetting=1.0, ridge=self._RIDGE)
        accumulator.observe_episode(phi, y, aborted=False)
        solved = accumulator.solve()

        # Both coefficients recovered to within a modest ridge bias despite a
        # 100x scale gap between the columns -- an unstandardized ridge at the
        # same lambda crushes the small-scale one by ~90% (checked below).
        np.testing.assert_allclose(solved, true_w, rtol=0.15)

    def test_an_unstandardized_ridge_would_have_crushed_it_at_the_same_lambda(self) -> None:
        """Not a property of :class:`RidgeAccumulator` -- a control showing the
        failure mode is real at this scale gap and this lambda, so the
        previous test's pass is not vacuous (i.e. not merely "any lambda this
        small passes regardless of standardization")."""
        rng = np.random.default_rng(3)
        phi, y, true_w = self._scaled_world(
            rng,
            n_samples=self._N_SAMPLES,
            small_scale=self._SMALL_SCALE,
            large_scale=self._LARGE_SCALE,
            noise=0.01,
        )

        unstandardized = np.linalg.solve(phi.T @ phi + self._RIDGE * np.eye(2), phi.T @ y)

        assert abs(unstandardized[0] - true_w[0]) > abs(unstandardized[1] - true_w[1]), (
            "the control ridge should shrink the small-scale column's coefficient "
            "far more than the large-scale one's -- if not, this lambda/scale "
            "combination no longer demonstrates the risk"
        )
        # And it should be a much worse relative error than the standardized case.
        assert abs(unstandardized[0] - true_w[0]) / true_w[0] > 0.75

    def test_a_column_that_is_always_zero_does_not_raise_or_produce_nan(self) -> None:
        """``claimed`` is structurally zero for every chosen action
        (``transformer_policy.py``): the frozen scale's floor must keep this
        finite rather than dividing by zero."""
        rng = np.random.default_rng(4)
        n_samples = 50
        real_column = rng.normal(size=n_samples)
        zero_column = np.zeros(n_samples)
        phi = np.stack([real_column, zero_column], axis=1)
        y = real_column * 2.0

        accumulator = RidgeAccumulator.zeros(2, forgetting=1.0, ridge=1.0)
        accumulator.observe_episode(phi, y, aborted=False)
        solved = accumulator.solve()

        assert np.isfinite(solved).all()
        assert solved[1] == pytest.approx(0.0, abs=1e-6)

    def test_scale_freezes_after_the_first_solve_and_never_moves_again(self) -> None:
        rng = np.random.default_rng(5)
        phi_1, y_1, _ = self._scaled_world(rng, n_samples=200, small_scale=1.0, large_scale=1.0)
        accumulator = RidgeAccumulator.zeros(2, forgetting=1.0, ridge=1.0)
        accumulator.observe_episode(phi_1, y_1, aborted=False)
        accumulator.solve()
        frozen = accumulator.scale
        assert frozen is not None

        # A wildly different second episode must not move the frozen scale.
        phi_2, y_2, _ = self._scaled_world(
            rng, n_samples=200, small_scale=1000.0, large_scale=1000.0
        )
        accumulator.observe_episode(phi_2, y_2, aborted=False)
        accumulator.solve()

        np.testing.assert_array_equal(accumulator.scale, frozen)


class TestForgetting:
    def test_gamma_one_never_forgets(self) -> None:
        rng = np.random.default_rng(6)
        dim = 2
        old_w = np.array([10.0, 0.0])
        phi_old, y_old = _synthetic_regression(rng, n_samples=1000, dim=dim, true_w=old_w)

        accumulator = RidgeAccumulator.zeros(dim, forgetting=1.0, ridge=1e-6)
        accumulator.observe_episode(phi_old, y_old, aborted=False)
        before = accumulator.solve()

        # One tiny additional episode should barely move a solve built on 1000
        # samples when nothing is ever forgotten.
        phi_new, y_new = _synthetic_regression(rng, n_samples=5, dim=dim, true_w=-old_w)
        accumulator.observe_episode(phi_new, y_new, aborted=False)
        after = accumulator.solve()

        np.testing.assert_allclose(before, after, atol=0.5)

    def test_a_short_window_forgets_old_data(self) -> None:
        rng = np.random.default_rng(7)
        dim = 2
        old_w = np.array([10.0, 0.0])
        new_w = np.array([-10.0, 0.0])
        accumulator = RidgeAccumulator.zeros(dim, forgetting=0.5, ridge=1e-6)

        phi_old, y_old = _synthetic_regression(rng, n_samples=500, dim=dim, true_w=old_w)
        accumulator.observe_episode(phi_old, y_old, aborted=False)

        # Many episodes at gamma=0.5: the effective window (~2 episodes) means
        # the old data's influence should be negligible well before episode 20.
        for _ in range(20):
            phi_new, y_new = _synthetic_regression(rng, n_samples=500, dim=dim, true_w=new_w)
            accumulator.observe_episode(phi_new, y_new, aborted=False)

        np.testing.assert_allclose(accumulator.solve(), new_w, atol=0.5)

    def test_forgetting_applies_even_on_an_aborted_episode(self) -> None:
        """The window is measured in Episodes elapsed, not Episodes that
        contributed data (module docstring) -- an aborted Episode still ages
        out the accumulator's memory, it just adds nothing new."""
        rng = np.random.default_rng(8)
        dim = 2
        phi, y = _synthetic_regression(rng, n_samples=100, dim=dim, true_w=np.array([1.0, 1.0]))

        accumulator = RidgeAccumulator.zeros(dim, forgetting=0.5, ridge=1.0)
        accumulator.observe_episode(phi, y, aborted=False)
        a_before_abort = accumulator.A.copy()

        accumulator.observe_episode(np.zeros((1, dim)), np.zeros(1), aborted=True)

        np.testing.assert_allclose(accumulator.A, a_before_abort * 0.5)

    def test_raw_sum_sq_decays_exactly_like_effective_n(self) -> None:
        """Regression test: ``raw_sum_sq`` is ``_freeze_scale``'s numerator and
        ``effective_n`` is its denominator (``mean_square = raw_sum_sq /
        effective_n``) -- if one decays with forgetting and the other does
        not, the frozen scale drifts from the accumulator's own effective
        window, silently changing how hard the configured ``lambda`` actually
        shrinks (module docstring, "Standardize, then freeze"). Checked
        directly against the same recurrence :meth:`observe_episode` uses for
        ``effective_n``, not just "does the final answer look plausible."
        """
        rng = np.random.default_rng(11)
        dim = 3
        forgetting = 0.7
        accumulator = RidgeAccumulator.zeros(dim, forgetting=forgetting, ridge=1.0)

        expected_raw_sum_sq = np.zeros(dim)
        expected_effective_n = 0.0
        for _ in range(5):
            phi = rng.normal(size=(4, dim))
            y = rng.normal(size=4)
            accumulator.observe_episode(phi, y, aborted=False)
            expected_raw_sum_sq = expected_raw_sum_sq * forgetting + np.square(phi).sum(axis=0)
            expected_effective_n = expected_effective_n * forgetting + phi.shape[0]

        np.testing.assert_allclose(accumulator.raw_sum_sq, expected_raw_sum_sq)
        assert accumulator.effective_n == pytest.approx(expected_effective_n)
        # The ratio the two feed into _freeze_scale must therefore match a
        # true windowed mean square, not one inflated by an undecayed
        # numerator over a decayed denominator.
        np.testing.assert_allclose(
            accumulator.raw_sum_sq / accumulator.effective_n,
            expected_raw_sum_sq / expected_effective_n,
        )


class TestAbortedEpisodesAreExcluded:
    def test_an_aborted_episode_does_not_change_the_solve_beyond_forgetting(self) -> None:
        rng = np.random.default_rng(9)
        dim = 2
        true_w = np.array([3.0, -2.0])
        phi, y = _synthetic_regression(rng, n_samples=500, dim=dim, true_w=true_w, noise=0.01)

        clean = RidgeAccumulator.zeros(dim, forgetting=1.0, ridge=0.5)
        clean.observe_episode(phi, y, aborted=False)
        clean_solve = clean.solve()

        # A grossly outlying target -- the F10 heavy tail -- must not move the
        # solve at all when marked aborted, forgetting aside (gamma=1.0 here).
        with_abort_noise = RidgeAccumulator.zeros(dim, forgetting=1.0, ridge=0.5)
        with_abort_noise.observe_episode(phi, y, aborted=False)
        outlier_phi = rng.normal(size=(400, dim))
        outlier_y = np.full(400, 1.0e5)
        with_abort_noise.observe_episode(outlier_phi, outlier_y, aborted=True)

        np.testing.assert_allclose(with_abort_noise.solve(), clean_solve, atol=1e-8, rtol=1e-8)

    def test_episodes_included_and_excluded_are_counted_separately(self) -> None:
        dim = 2
        accumulator = RidgeAccumulator.zeros(dim, forgetting=0.99, ridge=1.0)
        phi = np.ones((3, dim))
        y = np.ones(3)

        accumulator.observe_episode(phi, y, aborted=False)
        accumulator.observe_episode(phi, y, aborted=True)
        accumulator.observe_episode(phi, y, aborted=False)
        accumulator.observe_episode(phi, y, aborted=True)
        accumulator.observe_episode(phi, y, aborted=True)

        assert accumulator.episodes_included == 2
        assert accumulator.episodes_excluded == 3


class TestEpisodesSinceSolve:
    def test_increments_on_every_observation_and_resets_on_solve(self) -> None:
        dim = 2
        accumulator = RidgeAccumulator.zeros(dim, forgetting=1.0, ridge=1.0)
        phi = np.ones((2, dim))
        y = np.ones(2)

        accumulator.observe_episode(phi, y, aborted=False)
        accumulator.observe_episode(phi, y, aborted=True)
        assert accumulator.episodes_since_solve == 2

        accumulator.solve()
        assert accumulator.episodes_since_solve == 0


class TestValidation:
    def test_wrong_feature_width_is_rejected(self) -> None:
        accumulator = RidgeAccumulator.zeros(3, forgetting=0.9, ridge=1.0)
        with pytest.raises(ValueError, match=r"\[T, 3\]"):
            accumulator.observe_episode(np.ones((5, 4)), np.ones(5), aborted=False)

    def test_mismatched_row_counts_are_rejected(self) -> None:
        accumulator = RidgeAccumulator.zeros(3, forgetting=0.9, ridge=1.0)
        with pytest.raises(ValueError, match="same length"):
            accumulator.observe_episode(np.ones((5, 3)), np.ones(4), aborted=False)


class TestStateDictRoundTrip:
    def test_round_trip_before_any_solve(self) -> None:
        accumulator = RidgeAccumulator.zeros(4, forgetting=0.95, ridge=2.0)
        accumulator.observe_episode(np.ones((3, 4)), np.ones(3), aborted=False)
        accumulator.observe_episode(np.ones((3, 4)), np.ones(3), aborted=True)

        restored = RidgeAccumulator.from_state_dict(accumulator.state_dict())

        np.testing.assert_array_equal(restored.A, accumulator.A)
        np.testing.assert_array_equal(restored.b, accumulator.b)
        np.testing.assert_array_equal(restored.raw_sum_sq, accumulator.raw_sum_sq)
        assert restored.scale is None
        assert restored.effective_n == accumulator.effective_n
        assert restored.episodes_included == accumulator.episodes_included
        assert restored.episodes_excluded == accumulator.episodes_excluded
        assert restored.episodes_since_solve == accumulator.episodes_since_solve
        np.testing.assert_array_equal(restored.solve(), accumulator.solve())

    def test_round_trip_after_a_solve_preserves_the_frozen_scale(self) -> None:
        rng = np.random.default_rng(10)
        accumulator = RidgeAccumulator.zeros(3, forgetting=0.9, ridge=1.0)
        phi, y = _synthetic_regression(rng, n_samples=50, dim=3, true_w=np.array([1.0, 2.0, 3.0]))
        accumulator.observe_episode(phi, y, aborted=False)
        accumulator.solve()

        restored = RidgeAccumulator.from_state_dict(accumulator.state_dict())

        assert restored.scale is not None
        np.testing.assert_array_equal(restored.scale, accumulator.scale)
        # A further observation on each must land identically.
        more_phi, more_y = _synthetic_regression(
            rng, n_samples=10, dim=3, true_w=np.array([1.0, 2.0, 3.0])
        )
        accumulator.observe_episode(more_phi, more_y, aborted=False)
        restored.observe_episode(more_phi, more_y, aborted=False)
        np.testing.assert_allclose(restored.solve(), accumulator.solve())
