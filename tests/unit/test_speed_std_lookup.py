"""Unit tests for ``_build_speed_std_lookup`` (ticket 12, phase-2 fix 2).

The lookup blends the standard deviation linearly inside the three data-gap
windows (420-540, 660-840, 960-1080), reading the endpoints at the off-by-two
observed minutes (418/542, 658/842, 958/1082 — preserved legacy behavior), and
passes every other minute through raw. The legacy computed the 420-540 blend
into a discarded row copy and stored the raw value instead; that window now
blends like the other two (ADR-0001 phase-2 change log).
"""

import pandas as pd
import pytest

from stdvrp.traffic.travel_time_model import _build_speed_std_lookup

ARC = (1, 2)


def make_table(stds_by_minute: dict[int, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Node_Start": ARC[0],
                "Node_End": ARC[1],
                "Minute_start": minute,
                "speed_std": std,
            }
            for minute, std in stds_by_minute.items()
        ]
    )


class TestOutsideTheWindows:
    @pytest.mark.parametrize("minute", [300, 418, 420, 540, 542, 900, 1150])
    def test_minutes_outside_all_windows_store_the_raw_std(self, minute):
        lookup = _build_speed_std_lookup(make_table({minute: 0.7}))
        assert lookup[(*ARC, minute)] == 0.7


class TestInsideTheWindows:
    def test_morning_window_blends_the_offset_endpoints(self):
        # Minute 480 sits halfway through 420-540; endpoints live at the
        # off-by-two observed minutes 418 and 542.
        table = make_table({418: 0.2, 480: 0.9, 542: 0.6})
        lookup = _build_speed_std_lookup(table)
        assert lookup[(*ARC, 480)] == pytest.approx(0.2 * 0.5 + 0.5 * 0.6)

    def test_morning_window_raw_std_is_ignored(self):
        table = make_table({418: 0.3, 500: 99.0, 542: 0.3})
        lookup = _build_speed_std_lookup(table)
        assert lookup[(*ARC, 500)] == pytest.approx(0.3)

    def test_midday_window_blends_between_658_and_842(self):
        table = make_table({658: 0.9, 720: 0.1, 842: 0.3})
        lookup = _build_speed_std_lookup(table)
        expected = 0.9 * (2 / 3) + (1 / 3) * 0.3  # minute 720: 60 of 180 minutes in
        assert lookup[(*ARC, 720)] == pytest.approx(expected)

    def test_evening_window_blends_between_958_and_1082(self):
        table = make_table({958: 0.4, 1020: 0.1, 1082: 0.6})
        lookup = _build_speed_std_lookup(table)
        assert lookup[(*ARC, 1020)] == pytest.approx(0.5)


class TestMissingEndpoints:
    @pytest.mark.parametrize(
        ("minute", "present_endpoint"),
        [(480, 418), (480, 542), (720, 658), (720, 842), (1020, 958), (1020, 1082)],
    )
    def test_window_minute_with_a_missing_endpoint_gets_no_entry(
        self, minute, present_endpoint
    ):
        lookup = _build_speed_std_lookup(make_table({minute: 0.5, present_endpoint: 0.5}))
        assert (*ARC, minute) not in lookup
