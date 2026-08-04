"""Unit tests for the pure analysis of scripts/compare_w_trajectories.py.

The two capture drivers need the full Chengdu dataset and hours of CPU; the
arithmetic that turns their output into an answer does not, so it lives in pure
functions and is pinned here on hand-built trajectories.
"""

import math

import numpy as np
import pytest


def _trajectory(rows):
    """A W trajectory as the capture scripts write it: one row per episode."""
    return [list(map(float, row)) for row in rows]


def _padded(values, index, width=24):
    """A trajectory whose only non-zero component is ``index``."""
    return _trajectory([[v if i == index else 0.0 for i in range(width)] for v in values])


class TestFeatureNames:
    def test_names_one_per_feature(self, compare_w_module):
        # The 24 names are the whole point of the per-component report: an index
        # nobody can read back to a feature answers nothing.
        assert len(compare_w_module.FEATURE_NAMES) == 24

    def test_names_match_the_extractor_layout(self, compare_w_module):
        from stdvrp.policies.feature_extraction import FEATURE_COUNT

        assert len(compare_w_module.FEATURE_NAMES) == FEATURE_COUNT

    def test_names_are_unique(self, compare_w_module):
        assert len(set(compare_w_module.FEATURE_NAMES)) == len(compare_w_module.FEATURE_NAMES)


class TestStepNorms:
    def test_first_step_is_measured_from_zero(self, compare_w_module):
        # W starts at the zero vector before any episode, so the first episode's
        # step is the whole of W after it — not a dropped entry.
        norms = compare_w_module.step_norms(_padded([3.0, 7.0], index=0))
        assert norms == pytest.approx([3.0, 4.0])

    def test_one_norm_per_episode(self, compare_w_module):
        assert len(compare_w_module.step_norms(_padded([1.0, 2.0, 5.0], index=2))) == 3

    def test_empty_trajectory_yields_no_steps(self, compare_w_module):
        assert compare_w_module.step_norms([]) == []

    def test_a_step_backwards_still_has_positive_norm(self, compare_w_module):
        norms = compare_w_module.step_norms(_padded([5.0, 1.0], index=1))
        assert norms == pytest.approx([5.0, 4.0])


class TestNormTrajectory:
    def test_reports_the_euclidean_norm_per_episode(self, compare_w_module):
        traj = _trajectory([[3.0, 4.0] + [0.0] * 22, [0.0, 5.0] + [0.0] * 22])
        assert compare_w_module.norm_trajectory(traj) == pytest.approx([5.0, 5.0])


class TestComponentStats:
    def test_peaks_split_at_the_halfway_episode(self, compare_w_module):
        stats = compare_w_module.component_stats(_padded([1.0, 2.0, 6.0, 8.0], index=5))[5]
        assert stats.early_peak == pytest.approx(2.0)
        assert stats.late_peak == pytest.approx(8.0)
        assert stats.growth_ratio == pytest.approx(4.0)

    def test_peak_is_magnitude_not_signed_value(self, compare_w_module):
        # A weight that runs away negative is diverging just as hard as one that
        # runs away positive; the report must not cancel them out.
        stats = compare_w_module.component_stats(_padded([-1.0, -9.0], index=3))[3]
        assert stats.peak == pytest.approx(9.0)
        assert stats.final == pytest.approx(-9.0)

    def test_a_component_flat_at_zero_has_unit_growth(self, compare_w_module):
        # W[17] is the permanently-zero padding feature: it can never train, and
        # calling that "infinite growth" would put it at the top of the report.
        stats = compare_w_module.component_stats(_padded([0.0, 0.0, 0.0, 0.0], index=17))[17]
        assert stats.growth_ratio == pytest.approx(1.0)

    def test_a_component_that_starts_at_zero_and_grows_is_infinite(self, compare_w_module):
        traj = _trajectory([[0.0] * 24, [0.0] * 24, [0.0] * 24, [1.0] + [0.0] * 23])
        assert compare_w_module.component_stats(traj)[0].growth_ratio == math.inf

    def test_norm_share_is_of_the_final_weight_vector(self, compare_w_module):
        traj = _trajectory([[3.0, 4.0] + [0.0] * 22])
        stats = compare_w_module.component_stats(traj)
        assert stats[0].norm_share == pytest.approx(9 / 25)
        assert stats[1].norm_share == pytest.approx(16 / 25)

    def test_norm_share_of_an_all_zero_final_w_is_zero(self, compare_w_module):
        stats = compare_w_module.component_stats(_trajectory([[0.0] * 24]))
        assert all(s.norm_share == 0.0 for s in stats)

    def test_one_entry_per_component_carrying_its_name(self, compare_w_module):
        stats = compare_w_module.component_stats(_padded([1.0], index=0))
        assert [s.index for s in stats] == list(range(24))
        assert stats[22].name == compare_w_module.FEATURE_NAMES[22]

    def test_a_single_episode_puts_that_episode_in_both_halves(self, compare_w_module):
        # A one-episode run has no "second half"; reporting late_peak = 0 would
        # read as a component that collapsed rather than one never measured.
        stats = compare_w_module.component_stats(_padded([4.0], index=6))[6]
        assert stats.early_peak == pytest.approx(4.0)
        assert stats.late_peak == pytest.approx(4.0)

    def test_rejects_rows_of_inconsistent_width(self, compare_w_module):
        with pytest.raises(ValueError, match="width"):
            compare_w_module.component_stats([[0.0] * 24, [0.0] * 23])


class TestDivergenceReport:
    def test_ranks_by_how_much_magnitude_the_second_half_adds(self, compare_w_module):
        legacy = _trajectory([[0.0] * 24] * 4)
        repo = _trajectory(
            [
                [0.0] * 24,
                [1.0, 1.0] + [0.0] * 22,
                [1.0, 5.0] + [0.0] * 22,
                [3.0, 50.0] + [0.0] * 22,
            ]
        )
        report = compare_w_module.divergence_report(legacy, repo)
        assert [row.index for row in report[:2]] == [1, 0]

    def test_a_component_that_grows_on_both_sides_is_not_a_divergence(self, compare_w_module):
        # The question is which components misbehave *in ours*; one that the
        # legacy grows just as hard is shared behaviour, not the residual.
        both = _padded([1.0, 1.0, 10.0, 10.0], index=4)
        assert [row.index for row in compare_w_module.divergence_report(both, both)] == []

    def test_keeps_components_bounded_in_the_legacy_and_growing_in_ours(self, compare_w_module):
        legacy = _padded([1.0, 1.0, 1.0, 1.0], index=9)
        repo = _padded([1.0, 1.0, 20.0, 40.0], index=9)
        report = compare_w_module.divergence_report(legacy, repo)
        assert [row.index for row in report] == [9]
        assert report[0].legacy.growth_ratio == pytest.approx(1.0)
        assert report[0].repo.growth_ratio == pytest.approx(40.0)

    def test_threshold_is_configurable(self, compare_w_module):
        legacy = _padded([1.0, 1.0, 1.0, 1.0], index=2)
        repo = _padded([1.0, 1.0, 2.0, 3.0], index=2)
        assert compare_w_module.divergence_report(legacy, repo, growth_threshold=10.0) == []
        assert len(compare_w_module.divergence_report(legacy, repo, growth_threshold=2.0)) == 1


class TestMagnitudeRatios:
    def test_ranks_by_how_much_larger_our_peak_is(self, compare_w_module):
        legacy = _trajectory([[1.0, 1.0] + [0.0] * 22])
        repo = _trajectory([[2.0, 9.0] + [0.0] * 22])
        rows = compare_w_module.magnitude_ratios(legacy, repo)
        assert [row.index for row in rows[:2]] == [1, 0]
        assert rows[0].ratio == pytest.approx(9.0)

    def test_compares_peaks_not_final_values(self, compare_w_module):
        # Both runs oscillate; a component can peak early and end small. The
        # within-run growth measure misses exactly that, which is why this exists.
        legacy = _padded([1.0, 1.0], index=3)
        repo = _padded([50.0, 1.0], index=3)
        assert compare_w_module.magnitude_ratios(legacy, repo)[0].ratio == pytest.approx(50.0)

    def test_a_component_the_legacy_holds_at_zero_sorts_last(self, compare_w_module):
        # earliness_bin3 and zero_pad are structurally zero in the legacy. An
        # infinite ratio off a zero baseline means "absent there", not "worst" —
        # letting it head the table would bury the components that do compare.
        legacy = _trajectory([[0.0, 1.0] + [0.0] * 22])
        repo = _trajectory([[5.0, 3.0] + [0.0] * 22])
        rows = compare_w_module.magnitude_ratios(legacy, repo)
        assert rows[0].index == 1
        assert rows[-1].index == 0
        assert math.isinf(rows[-1].ratio)

    def test_zero_on_both_sides_is_ratio_one_not_infinite(self, compare_w_module):
        same = _padded([1.0], index=0)
        rows = {row.index: row for row in compare_w_module.magnitude_ratios(same, same)}
        assert rows[17].ratio == pytest.approx(1.0)

    def test_summary_line_covers_only_components_the_legacy_trains(self, compare_w_module):
        legacy = _trajectory([[0.0, 2.0] + [0.0] * 22])
        repo = _trajectory([[5.0, 4.0] + [0.0] * 22])
        rows = compare_w_module.magnitude_ratios(legacy, repo)
        text = compare_w_module.format_magnitude_ratios(rows)
        assert "1 components the legacy actually trains" in text
        assert "median 2.00x" in text


class TestLoadTrajectory:
    def test_reads_the_capture_schema(self, compare_w_module, tmp_path):
        import json

        path = tmp_path / "w.json"
        path.write_text(
            json.dumps({"meta": {"side": "repo"}, "w_trajectory": [[1.0] * 24, [2.0] * 24]}),
            encoding="utf-8",
        )
        loaded = compare_w_module.load_trajectory(path)
        assert np.asarray(loaded).shape == (2, 24)

    def test_rejects_a_file_with_no_trajectory(self, compare_w_module, tmp_path):
        import json

        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"meta": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="w_trajectory"):
            compare_w_module.load_trajectory(path)
