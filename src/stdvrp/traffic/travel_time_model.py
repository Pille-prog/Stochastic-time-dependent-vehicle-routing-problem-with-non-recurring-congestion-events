"""TravelTimeModel: stochastic time-dependent travel times derived from TrafficHistory.

Phase-1 structural port (ADR-0001) of the deterministic data spine of the legacy
``environment`` + ``DataCalculations`` classes. Every pandas operation replicates the
legacy op-for-op — including preserved bugs, flagged inline — so all derived values
are bit-identical to the monolith on the same inputs. Deliberately concrete: no
interface (ADR-0002).

Units, as in the legacy: lengths in km, speeds in km/min, times in minutes since
03:00. Speed CSVs carry km/h; the multi-day aggregation divides by 60.

Known preserved behaviors (do not "fix" before Phase 2, see ADR-0001):

- Speeds between minutes 420-540, 660-840 and 960-1080 are overwritten by a linear
  blend of the interval endpoints, even where real observations exist.
- The standard-deviation lookup stores the *raw* std for minutes 421-539 (the
  interpolated value is computed and discarded), and its endpoint lookups are
  off by two minutes (418/542, 658/842, 958/1082).
- With a single traffic day, every std is NaN: pandas ``std`` of one observation is
  NaN and the legacy ``dropna()`` result was discarded (a silent no-op, preserved
  here by simply not dropping).
- The per-arc event probability divides by the day count (the legacy hardcoded 44,
  its day count); identical whenever all 44 archive days are configured.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from stdvrp.network import RoadNetwork
from stdvrp.traffic.history import TrafficHistory
from stdvrp.traffic.periods import period_start_minute

ArcKey = tuple[int, int]
ArcMinuteKey = tuple[int, int, int]


class TravelTimeModel:
    """Interpolated mean speeds, speed deviations and travel times per arc and minute.

    Exposed lookups (legacy attribute in parentheses):

    - ``travel_data`` (``DataCalculations.travel_data``): (node_start, node_end,
      minute) -> (length_km, mean_speed_km_min).
    - ``speed_std`` (``get_standard_deviation_dict``): same key -> std of the speed.
    - ``successors`` (``travel_arc_information_dictionary``): node -> arc-end nodes,
      in arc-table order (congestion spreading iterates this order; ADR-0001).
    - ``node_coordinates`` (``latitude_and_longitude``): node -> [latitude, longitude].
    - ``event_probability`` (``probability_of_event``): (node_start, node_end) ->
      probability of a non-recurring congestion event, in insertion order (the live
      CongestionGenerator draws one random number per key in this order; ADR-0001).
    - ``mean_arc_data`` (``environment.data_average``): per-arc all-day mean speed,
      coordinates, length and travel time.
    """

    def __init__(
        self,
        road_network: RoadNetwork,
        traffic_history: TrafficHistory,
        max_congestion_duration: int,
        horizon_start_minute: int = 300,
    ) -> None:
        aggregated = _aggregate_speed_statistics(traffic_history)
        speed_table = _build_speed_table(
            road_network, traffic_history, aggregated, horizon_start_minute
        )
        self.mean_arc_data = _build_mean_arc_table(speed_table)
        speed_table = _fill_missing_minutes(speed_table)
        speed_table = _interpolate_speeds(speed_table)
        self._speed_table = speed_table

        self.speed_std = _build_speed_std_lookup(speed_table)
        self.travel_data = _build_travel_data(speed_table)
        self.successors = _build_successors(self.mean_arc_data)
        self.node_coordinates = _build_node_coordinates(self.mean_arc_data)
        self.event_probability = _compute_event_probabilities(
            traffic_history, speed_table, max_congestion_duration
        )


def _aggregate_speed_statistics(history: TrafficHistory) -> pd.DataFrame:
    """Mean and std of the km/min speed per (Period, Link) over all configured days.

    Ports ``environment.process_all_data``. The legacy called ``all_data.dropna()``
    and discarded the result — a silent no-op, preserved by not dropping (ADR-0001).
    """
    all_data = pd.concat(
        [history.observations_by_day[day] for day in history.days], ignore_index=True
    )
    all_data["Speed"] = all_data["Speed"] / 60
    return (
        all_data.groupby(["Period", "Link"])
        .agg(speed=("Speed", "mean"), speed_std=("Speed", "std"))
        .reset_index()
    )


def _build_speed_table(
    road_network: RoadNetwork,
    history: TrafficHistory,
    aggregated: pd.DataFrame,
    horizon_start_minute: int,
) -> pd.DataFrame:
    """One row per (arc, observed instance-day period), speeds replaced by all-day means.

    Ports ``environment.__init__`` up to the ``Travel_Time`` column: merge the links
    table with the instance day's observations, convert lengths to km and periods to
    minutes, drop pre-horizon rows, then swap the single-day speed for the aggregated
    mean (km/min). The legacy hardcoded 300 where this takes ``horizon_start_minute``.
    """
    speed_table = pd.merge(road_network.links, history.instance_day_observations, on="Link")
    speed_table["Length"] = speed_table["Length"] / 1000
    speed_table["Minute_start"] = speed_table["Period"].apply(period_start_minute)
    speed_table = speed_table[speed_table["Minute_start"] >= horizon_start_minute]
    speed_table = pd.merge(
        speed_table, aggregated, on=["Link", "Period"], how="left", suffixes=("", "_df2")
    )
    speed_table = speed_table.drop("Speed", axis=1)
    speed_table.rename(columns={"speed": "Speed"}, inplace=True)
    speed_table["Travel_Time"] = speed_table["Length"] / speed_table["Speed"]
    return speed_table


def _build_mean_arc_table(speed_table: pd.DataFrame) -> pd.DataFrame:
    """Per-arc mean speed over the horizon with coordinates and length.

    Ports the ``environment.data_average`` block. Runs before minute filling and
    interpolation, exactly as in the legacy constructor order.
    """
    average_speed = speed_table.groupby("Link")["Speed"].mean().reset_index()
    coords_start = speed_table.drop_duplicates(subset="Link")[
        ["Link", "Node_Start", "Latitude_Start", "Longitude_Start"]
    ]
    coords_end = speed_table.drop_duplicates(subset="Link")[
        ["Link", "Node_End", "Latitude_End", "Longitude_End", "Length"]
    ]
    mean_arc_data = average_speed.merge(coords_start, on="Link")
    mean_arc_data = mean_arc_data.merge(coords_end, on="Link")
    mean_arc_data["Travel_Time"] = mean_arc_data["Length"] / mean_arc_data["Speed"]
    return mean_arc_data


def _fill_missing_minutes(speed_table: pd.DataFrame) -> pd.DataFrame:
    """Fill every minute gap per link by repeating the last observed row.

    Ports the row-filling half of ``DataCalculations.process_data``: observations are
    two minutes apart with holes, so this materializes one row per (link, minute).
    """
    speed_table = speed_table.sort_values(by=["Link", "Minute_start"])
    new_rows = []
    for link in speed_table["Link"].unique():
        link_rows = speed_table[speed_table["Link"] == link]
        prev_row = None
        for _, row in link_rows.iterrows():
            if prev_row is not None:
                time_diff = row["Minute_start"] - prev_row["Minute_start"]
                if time_diff > 1:
                    for minute in range(prev_row["Minute_start"] + 1, row["Minute_start"], 1):
                        new_row = prev_row.copy()
                        new_row["Minute_start"] = minute
                        new_rows.append(new_row)
            prev_row = row
    new_rows_df = pd.DataFrame(new_rows)
    return pd.concat([speed_table, new_rows_df]).sort_values(by=["Link", "Minute_start"])


def _interpolate_speeds(speed_table: pd.DataFrame) -> pd.DataFrame:
    """Overwrite speeds inside the three blend windows and recompute travel times.

    Ports the interpolation half of ``DataCalculations.process_data``. The lookup is
    built once, before any overwrite, so blends read pre-interpolation values.
    """
    speed_lookup = speed_table.set_index(["Link", "Minute_start"])["Speed"].to_dict()
    speed_table["Speed"] = speed_table.apply(  # type: ignore[call-overload]
        lambda row: _interpolated_speed(row, speed_lookup), axis=1
    )
    speed_table["Travel_Time"] = speed_table["Length"] / speed_table["Speed"]
    return speed_table


def _interpolated_speed(row: pd.Series[Any], speed_lookup: dict[Any, float]) -> float | None:
    """Ports ``DataCalculations.get_interpolated_speed`` verbatim.

    Legacy quirk preserved (ADR-0001): inside a window with a missing endpoint the
    function falls through and returns None (NaN after ``apply``).
    """
    link = row["Link"]
    minute_start = row["Minute_start"]
    original_speed = row["Speed"]

    if 420 < minute_start < 540:
        ratio_540 = (minute_start - 420) / (540 - 420)
        ratio_420 = 1 - ratio_540
        sp_420 = speed_lookup.get((link, 420))
        sp_540 = speed_lookup.get((link, 540))
        if sp_420 is not None and sp_540 is not None:
            return float((sp_420 * ratio_420) + (sp_540 * ratio_540))
    elif 660 < minute_start < 840:
        ratio_840 = (minute_start - 660) / (840 - 660)
        ratio_660 = 1 - ratio_840
        sp_660 = speed_lookup.get((link, 660))
        sp_840 = speed_lookup.get((link, 840))
        if sp_660 is not None and sp_840 is not None:
            return float((sp_660 * ratio_660) + (ratio_840 * sp_840))
    elif 960 < minute_start < 1080:
        ratio_1080 = (minute_start - 960) / (1080 - 960)
        ratio_960 = 1 - ratio_1080
        sp_960 = speed_lookup.get((link, 960))
        sp_1080 = speed_lookup.get((link, 1080))
        if sp_960 is not None and sp_1080 is not None:
            return float((sp_960 * ratio_960) + (sp_1080 * ratio_1080))
    else:
        return float(original_speed)
    return None


def _build_speed_std_lookup(speed_table: pd.DataFrame) -> dict[ArcMinuteKey, float]:
    """(node_start, node_end, minute) -> speed std, ports ``get_standard_deviation``.

    Legacy bugs preserved (ADR-0001): in the 420-540 window the interpolated value is
    discarded and the raw std stored instead; every endpoint lookup is offset by two
    minutes (418/542, 658/842, 958/1082).
    """
    std_lookup = speed_table.set_index(["Node_Start", "Node_End", "Minute_start"])[
        "speed_std"
    ].to_dict()
    result: dict[ArcMinuteKey, float] = {}
    for _, row in speed_table.iterrows():
        key = (row["Node_Start"], row["Node_End"], row["Minute_start"])
        minute_start = row["Minute_start"]

        if 420 < minute_start < 540:
            sp_420 = std_lookup.get((row["Node_Start"], row["Node_End"], 418))
            sp_540 = std_lookup.get((row["Node_Start"], row["Node_End"], 542))
            if sp_420 is not None and sp_540 is not None:
                # The legacy computed the blend into a discarded row copy and stored
                # the raw value; the blend is therefore not computed at all here.
                result[key] = row["speed_std"]
        elif 660 < minute_start < 840:
            ratio_840 = (minute_start - 660) / (840 - 660)
            ratio_660 = 1 - ratio_840
            sp_660 = std_lookup.get((row["Node_Start"], row["Node_End"], 658))
            sp_840 = std_lookup.get((row["Node_Start"], row["Node_End"], 842))
            if sp_660 is not None and sp_840 is not None:
                result[key] = (sp_660 * ratio_660) + (ratio_840 * sp_840)
        elif 960 < minute_start < 1080:
            ratio_1080 = (minute_start - 960) / (1080 - 960)
            ratio_960 = 1 - ratio_1080
            sp_960 = std_lookup.get((row["Node_Start"], row["Node_End"], 958))
            sp_1080 = std_lookup.get((row["Node_Start"], row["Node_End"], 1082))
            if sp_960 is not None and sp_1080 is not None:
                result[key] = (sp_960 * ratio_960) + (sp_1080 * ratio_1080)
        else:
            result[key] = row["speed_std"]
    return result


def _build_travel_data(speed_table: pd.DataFrame) -> dict[ArcMinuteKey, tuple[float, float]]:
    """(node_start, node_end, minute) -> (length_km, speed_km_min).

    Ports the first loop of ``arcs_data_to_dictionary``; on duplicate arcs the last
    row wins, as in the legacy dict fill.
    """
    travel_data: dict[ArcMinuteKey, tuple[float, float]] = {}
    for _, row in speed_table.iterrows():
        key = (row["Node_Start"], row["Node_End"], row["Minute_start"])
        travel_data[key] = (row["Length"], row["Speed"])
    return travel_data


def _build_successors(mean_arc_data: pd.DataFrame) -> dict[int, list[int]]:
    """node -> arc-end nodes in arc-table order (``travel_arc_information_dictionary``)."""
    successors: dict[int, list[int]] = {}
    for _, row in mean_arc_data.iterrows():
        successors.setdefault(row["Node_Start"], []).append(row["Node_End"])
    return successors


def _build_node_coordinates(mean_arc_data: pd.DataFrame) -> dict[int, list[float]]:
    """node -> [latitude, longitude] (``latitude_and_longitude``); last arc row wins."""
    coordinates: dict[int, list[float]] = {}
    for _, row in mean_arc_data.iterrows():
        coordinates[row["Node_Start"]] = [row["Latitude_Start"], row["Longitude_Start"]]
    return coordinates


def _compute_event_probabilities(
    history: TrafficHistory,
    speed_table: pd.DataFrame,
    max_congestion_duration: int,
) -> dict[ArcKey, float]:
    """Per-arc probability of a non-recurring congestion event per decision epoch.

    Ports the live slices of ``read_all_data`` (the 30-minute aggregation; the 60/90/
    120-minute aggregates and a period-label column were computed but never consumed,
    and are not ported), ``get_mean_of_all_intervals`` (``df_mean``; ``df_30`` was
    unused) and ``store_probability_for_event_of_all_arcs``. The legacy divided by a
    hardcoded 44 — its day count; this uses the configured day count (identical
    whenever all 44 archive days are configured).
    """
    per_day_30min = []
    for day in history.days:
        combined = history.observations_by_day[day].copy()
        combined["start_time"] = combined["Period"].str.split("-").str[0]
        combined["start_time"] = pd.to_datetime(combined["start_time"], format="%H:%M")
        combined = combined[combined["start_time"].dt.hour >= 8]
        combined["30_min_interval"] = combined["start_time"].dt.floor("30min")
        grouped = combined.groupby(["Link", "30_min_interval"], as_index=False)["Speed"].mean()
        per_day_30min.append(grouped)
    all_30_min = pd.concat(per_day_30min, ignore_index=True)

    mean_speed = (
        speed_table.groupby(["Link", "Node_Start", "Node_End"]).agg({"Speed": "mean"}).reset_index()
    )
    mean_speed.rename(columns={"Speed": "avg_speed"}, inplace=True)

    unit_of_time_of_congestions = max_congestion_duration / 60
    hours = 8 / unit_of_time_of_congestions
    merged = pd.merge(all_30_min, mean_speed, on=["Link"], how="outer")
    merged["Speed"] = merged["Speed"] / 60

    probability: dict[ArcKey, float] = {}
    for _, row in merged.iterrows():
        key = (row["Node_Start"], row["Node_End"])
        if key not in probability:
            probability[key] = 0
        if row["Speed"] <= 0.4 * row["avg_speed"] and row["Speed"] >= 0.1 * row["avg_speed"]:
            probability[key] += 1

    day_count = len(history.days)
    for key in probability:
        probability[key] = probability[key] * 2 / (day_count * hours * 3)
    return probability
