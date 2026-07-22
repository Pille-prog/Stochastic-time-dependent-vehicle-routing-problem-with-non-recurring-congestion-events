"""Build the Chengdu instance from an experiment config and print a summary.

Tickets 05-06 scope: config -> DataSource -> RoadNetwork + TrafficHistory ->
TravelTimeModel, plus the ShortestPathCache and one ClientGenerator draw. Later
tickets extend this into full training/evaluation runs.

    uv run python experiments/chengdu/run.py [--config path/to/config.yaml]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from stdvrp.config import ExperimentConfig
from stdvrp.demand import ClientGenerator
from stdvrp.traffic import CsvDataSource, TravelTimeModel


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="experiment config YAML (default: config.yaml next to this script)",
    )
    args = parser.parse_args(argv)

    config = ExperimentConfig.from_yaml(args.config)
    source = CsvDataSource.from_config(config)
    road_network = source.load_road_network()
    traffic_history = source.load_traffic_history()
    travel_time_model = TravelTimeModel(
        road_network,
        traffic_history,
        config.max_congestion_duration,
        config.horizon_start_minute,
    )

    minutes = sorted({key[2] for key in travel_time_model.travel_data})
    nan_stds = sum(1 for value in travel_time_model.speed_std.values() if math.isnan(value))
    print(f"config: {args.config}")
    print(f"road network: {road_network.arc_count} arcs, {len(road_network.node_ids)} start nodes")
    print(f"traffic history: days {list(traffic_history.days)}, instance day {config.instance_day}")
    print(
        f"travel data: {len(travel_time_model.travel_data)} (arc, minute) entries, "
        f"minutes {minutes[0]}..{minutes[-1]}"
    )
    print(
        f"speed std entries: {len(travel_time_model.speed_std)} "
        f"({nan_stds} NaN — expected with a single traffic day)"
    )
    print(f"event probabilities: {len(travel_time_model.event_probability)} arcs")

    shortest_path_cache = source.load_shortest_path_cache()
    print(f"shortest path cache: {len(shortest_path_cache)} node-client pairs")

    demand = ClientGenerator.from_config(config).generate(config.first_train_seed)
    print(
        f"demand (seed {config.first_train_seed}): "
        f"{len(demand.clients)} clients, {demand.vehicle_count} vehicles"
    )


if __name__ == "__main__":
    main()
