"""Unit tests for the time-unit conversion ``period_start_minute`` (ticket 11).

The whole pipeline counts minutes since 03:00; this is the single conversion from
the CSV's ``"HH:MM-HH:MM"`` Period strings into that clock.
"""

import pytest

from stdvrp.traffic.periods import period_start_minute


class TestPeriodStartMinute:
    def test_docstring_example(self):
        assert period_start_minute("08:00-08:02") == 300

    def test_clock_origin_is_three_am(self):
        assert period_start_minute("03:00-03:02") == 0

    def test_minutes_are_added_to_the_hour_offset(self):
        assert period_start_minute("03:59-04:01") == 59
        assert period_start_minute("13:58-14:00") == 658

    def test_evening_period(self):
        assert period_start_minute("21:00-21:02") == 1080

    def test_end_of_day(self):
        assert period_start_minute("23:58-00:00") == 1258

    def test_only_the_period_start_matters(self):
        assert period_start_minute("08:00-11:30") == period_start_minute("08:00-08:02")

    @pytest.mark.parametrize(
        ("period", "minute"),
        [("05:00-05:02", 120), ("07:00-07:02", 240), ("09:00-09:02", 360)],
    )
    def test_each_hour_adds_sixty_minutes(self, period, minute):
        assert period_start_minute(period) == minute
