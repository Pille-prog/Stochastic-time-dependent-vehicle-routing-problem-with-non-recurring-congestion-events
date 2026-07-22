"""Fixture-based checks of the data spine: DataSource -> RoadNetwork + TrafficHistory
-> TravelTimeModel.

Expected values are computed here from the raw fixture CSVs with independent,
dict-based logic (no pandas pipeline), mirroring the legacy definitions:

- speeds are km/h in the files; the aggregation divides by 60 (km/min); with the
  single fixture day the aggregated mean equals the observation itself;
- every minute between observations repeats the last observed row;
- minutes strictly inside 420-540, 660-840 and 960-1080 blend the window endpoints.

Exact float equality is asserted throughout: the expressions replicate the legacy
operation order, and ADR-0001 demands bit-for-bit fidelity.

The 44-day exact-equality run against the unmodified legacy classes lives in
tests/test_travel_time_model_vs_legacy.py.
"""

import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

from stdvrp.network import RoadNetwork
from stdvrp.traffic import CsvDataSource, TravelTimeModel, period_start_minute

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "chengdu_mini"
REPO_ROOT = Path(__file__).resolve().parents[1]

# Arc of fixture Link 1 (verified unique: the fixture has no duplicate arcs).
ARC_START, ARC_END, ARC_LINK = 0, 4, 1


@pytest.fixture(scope="module")
def data_source() -> CsvDataSource:
    return CsvDataSource(FIXTURE_DIR, "link.csv", 601, (601,), "all_shortest_paths.csv")


@pytest.fixture(scope="module")
def road_network(data_source: CsvDataSource) -> RoadNetwork:
    return data_source.load_road_network()


@pytest.fixture(scope="module")
def travel_time_model(data_source: CsvDataSource, road_network: RoadNetwork) -> TravelTimeModel:
    return TravelTimeModel(
        road_network, data_source.load_traffic_history(), max_congestion_duration=120
    )


@pytest.fixture(scope="module")
def links() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "link.csv")


@pytest.fixture(scope="module")
def observed_speeds() -> dict[int, float]:
    """Raw km/h observations of ARC_LINK by minute, both instance-day halves."""
    observed: dict[int, float] = {}
    for half in (0, 1):
        speeds = pd.read_csv(FIXTURE_DIR / f"speed[601]_[{half}].csv")
        for row in speeds[speeds["Link"] == ARC_LINK].itertuples():
            observed[period_start_minute(row.Period)] = row.Speed
    return observed


def filled_speed(observed: dict[int, float], minute: int) -> float:
    """km/min speed after minute filling: the last observation at or before `minute`."""
    return observed[max(m for m in observed if m <= minute)] / 60


def expected_speed(observed: dict[int, float], minute: int) -> float:
    """Filled speed, blended inside the three interpolation windows (legacy formulas)."""
    for low, high in ((420, 540), (660, 840), (960, 1080)):
        if low < minute < high:
            ratio_high = (minute - low) / (high - low)
            ratio_low = 1 - ratio_high
            return (filled_speed(observed, low) * ratio_low) + (
                filled_speed(observed, high) * ratio_high
            )
    return filled_speed(observed, minute)


def test_road_network_reads_the_links_table(road_network: RoadNetwork) -> None:
    assert road_network.arc_count == 116
    assert road_network.node_ids[0] == 0
    assert len(road_network.node_ids) == len(set(road_network.node_ids))


def test_road_network_rejects_wrong_schema() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        RoadNetwork(pd.DataFrame({"Link": [1]}))


def test_traffic_history_holds_the_instance_day(data_source: CsvDataSource) -> None:
    history = data_source.load_traffic_history()
    assert history.days == (601,)
    morning = pd.read_csv(FIXTURE_DIR / "speed[601]_[0].csv")
    afternoon = pd.read_csv(FIXTURE_DIR / "speed[601]_[1].csv")
    assert len(history.instance_day_observations) == len(morning) + len(afternoon)


def test_travel_data_covers_every_minute_of_the_data_span(
    travel_time_model: TravelTimeModel,
) -> None:
    minutes = sorted(
        m
        for (start, end, m) in travel_time_model.travel_data
        if (start, end) == (ARC_START, ARC_END)
    )
    assert minutes == list(range(300, 1199))


@pytest.mark.parametrize(
    "minute",
    [
        300,  # first observed minute, outside every window
        301,  # filled from minute 300
        421,  # window 1 (endpoint 420 itself is filled: day 601 has no 10:00 period)
        500,  # window 1, deeper in
        540,  # window boundary: not blended
        599,  # fills the morning/afternoon gap between 598 and 600
        700,  # window 2
        1000,  # window 3
        1198,  # last observed minute
    ],
)
def test_interpolated_speeds_and_travel_times_match_the_legacy_computation(
    travel_time_model: TravelTimeModel,
    links: pd.DataFrame,
    observed_speeds: dict[int, float],
    minute: int,
) -> None:
    length_km = float(links.loc[links["Link"] == ARC_LINK, "Length"].iloc[0]) / 1000
    speed = expected_speed(observed_speeds, minute)
    assert travel_time_model.travel_data[(ARC_START, ARC_END, minute)] == (length_km, speed)


def test_windows_blend_filled_speeds_rather_than_repeat_them(
    travel_time_model: TravelTimeModel, observed_speeds: dict[int, float]
) -> None:
    # The real day-601 data records nothing inside the three windows — they are data
    # gaps, and the legacy interpolation exists to bridge them (its endpoint lookups
    # at 418/542 etc. target the last observed minutes around each gap).
    assert not any(
        420 < minute < 540 or 660 < minute < 840 or 960 < minute < 1080
        for minute in observed_speeds
    )
    # Inside a gap the blend must win over the repeat-last-observation fill.
    candidates = [
        minute
        for minute in range(421, 540)
        if expected_speed(observed_speeds, minute) != filled_speed(observed_speeds, minute)
    ]
    assert candidates, "blend never differs from the fill; the check would be vacuous"
    minute = candidates[0]
    _, speed = travel_time_model.travel_data[(ARC_START, ARC_END, minute)]
    assert speed == expected_speed(observed_speeds, minute)
    assert speed != filled_speed(observed_speeds, minute)


def test_single_day_speed_std_is_nan_everywhere(travel_time_model: TravelTimeModel) -> None:
    # pandas std of one observation is NaN and the legacy dropna() was a silent
    # no-op (ADR-0001): with the one-day fixture every deviation is NaN. Ticket 07
    # must use the multi-day characterization world for stochastic velocities.
    assert set(travel_time_model.speed_std) == set(travel_time_model.travel_data)
    assert all(math.isnan(value) for value in travel_time_model.speed_std.values())


def test_successors_follow_link_order(
    travel_time_model: TravelTimeModel, links: pd.DataFrame
) -> None:
    ordered = links.sort_values("Link")
    expected = {
        start: list(group["Node_End"]) for start, group in ordered.groupby("Node_Start", sort=False)
    }
    assert travel_time_model.successors == expected


def test_node_coordinates_come_from_the_links_table(
    travel_time_model: TravelTimeModel, links: pd.DataFrame
) -> None:
    row = links[links["Link"] == ARC_LINK].iloc[0]
    assert travel_time_model.node_coordinates[ARC_START] == [
        row["Latitude_Start"],
        row["Longitude_Start"],
    ]


def test_event_probabilities_cover_every_arc(
    travel_time_model: TravelTimeModel, links: pd.DataFrame
) -> None:
    arcs = set(zip(links["Node_Start"], links["Node_End"], strict=True))
    assert set(travel_time_model.event_probability) == arcs
    # count * 2 / (days * hours * 3) with hours = 8 / (120 / 60) = 4.
    assert all(value >= 0 for value in travel_time_model.event_probability.values())


def test_mean_arc_data_is_internally_consistent(travel_time_model: TravelTimeModel) -> None:
    table = travel_time_model.mean_arc_data
    assert len(table) == 116
    assert (table["Travel_Time"] == table["Length"] / table["Speed"]).all()


def test_entry_script_builds_the_instance_and_prints_a_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = importlib.util.spec_from_file_location(
        "chengdu_run", REPO_ROOT / "experiments" / "chengdu" / "run.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main(["--config", str(FIXTURE_DIR / "config.yaml")])

    output = capsys.readouterr().out
    assert "road network: 116 arcs" in output
    assert "instance day 601" in output
    assert "event probabilities: 116 arcs" in output
    assert "shortest path cache: 2025 node-client pairs" in output
    assert "demand (seed 1000):" in output


def test_data_source_loads_the_shortest_path_cache(data_source: CsvDataSource) -> None:
    cache = data_source.load_shortest_path_cache()
    assert len(cache) == 45 * 45
