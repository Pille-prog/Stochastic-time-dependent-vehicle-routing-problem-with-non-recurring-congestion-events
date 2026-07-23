"""Unit tests for ``_interpolated_speed`` edge cases (ticket 11).

The function blends speeds linearly inside three windows (420-540, 660-840,
960-1080 minutes) and passes every other minute through untouched. The legacy
quirk that a window minute with a missing endpoint observation yields ``None``
(NaN after ``apply``) is preserved behavior (ADR-0001) and pinned here.
"""

import pandas as pd
import pytest

from stdvrp.traffic.travel_time_model import _interpolated_speed

LINK = 7


def row(minute: int, speed: float = 0.5) -> pd.Series:
    return pd.Series({"Link": LINK, "Minute_start": minute, "Speed": speed})


def lookup(speeds_by_minute: dict[int, float]) -> dict:
    """Endpoint observations for LINK, e.g. ``lookup({420: 0.6, 540: 0.9})``."""
    return {(LINK, minute): speed for minute, speed in speeds_by_minute.items()}


class TestOutsideTheWindows:
    @pytest.mark.parametrize("minute", [300, 419, 541, 659, 841, 959, 1081, 1150])
    def test_minutes_outside_all_windows_keep_the_original_speed(self, minute):
        assert _interpolated_speed(row(minute, speed=0.5), {}) == 0.5

    @pytest.mark.parametrize("minute", [420, 540, 660, 840, 960, 1080])
    def test_window_boundaries_are_exclusive(self, minute):
        # The endpoints themselves are never interpolated, even when both
        # endpoint observations exist.
        endpoints = lookup({420: 0.6, 540: 0.9, 660: 0.6, 840: 0.9, 960: 0.6, 1080: 0.9})
        assert _interpolated_speed(row(minute, speed=0.5), endpoints) == 0.5


class TestInsideTheWindows:
    def test_morning_window_midpoint_is_the_average_of_the_endpoints(self):
        assert _interpolated_speed(row(480), lookup({420: 0.6, 540: 0.9})) == pytest.approx(0.75)

    def test_first_minute_inside_the_morning_window(self):
        expected = 0.6 * (119 / 120) + 0.9 * (1 / 120)
        assert _interpolated_speed(row(421), lookup({420: 0.6, 540: 0.9})) == pytest.approx(
            expected
        )

    def test_last_minute_inside_the_morning_window(self):
        expected = 0.6 * (1 / 120) + 0.9 * (119 / 120)
        assert _interpolated_speed(row(539), lookup({420: 0.6, 540: 0.9})) == pytest.approx(
            expected
        )

    def test_midday_window_blends_between_660_and_840(self):
        expected = 0.8 * (2 / 3) + 0.2 * (1 / 3)  # minute 720: 60 of 180 minutes in
        assert _interpolated_speed(row(720), lookup({660: 0.8, 840: 0.2})) == pytest.approx(
            expected
        )

    def test_evening_window_blends_between_960_and_1080(self):
        assert _interpolated_speed(row(1020), lookup({960: 0.4, 1080: 0.6})) == pytest.approx(0.5)

    def test_original_speed_inside_a_window_is_ignored(self):
        interpolated = _interpolated_speed(row(480, speed=99.0), lookup({420: 0.6, 540: 0.9}))
        assert interpolated == pytest.approx(0.75)


class TestMissingObservations:
    def test_missing_left_endpoint_yields_none(self):
        assert _interpolated_speed(row(480), lookup({540: 0.9})) is None

    def test_missing_right_endpoint_yields_none(self):
        assert _interpolated_speed(row(480), lookup({420: 0.6})) is None

    def test_both_endpoints_missing_yields_none(self):
        assert _interpolated_speed(row(720), {}) is None

    def test_endpoints_of_another_link_do_not_count(self):
        other_link = {(LINK + 1, 420): 0.6, (LINK + 1, 540): 0.9}
        assert _interpolated_speed(row(480), other_link) is None
