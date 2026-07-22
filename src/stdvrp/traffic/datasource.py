"""DataSource: the boundary through which RoadNetwork and TrafficHistory are loaded.

One of the three abstraction seams (ADR-0002): CSV files today, a database later.
Only the origin of the data varies — the domain model does not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from stdvrp.config import ExperimentConfig
from stdvrp.network import RoadNetwork, ShortestPathCache
from stdvrp.traffic.history import TrafficHistory


class DataSource(ABC):
    """Loads the problem data of one instance."""

    @abstractmethod
    def load_road_network(self) -> RoadNetwork:
        """The static road network (nodes, arcs, coordinates, lengths)."""

    @abstractmethod
    def load_traffic_history(self) -> TrafficHistory:
        """Historical speed observations for the configured days."""

    @abstractmethod
    def load_shortest_path_cache(self) -> ShortestPathCache:
        """The precomputed shortest paths between network nodes."""


class CsvDataSource(DataSource):
    """Reads the on-disk CSV layout of the Chengdu archive.

    File naming follows the archive convention: ``speed[<day>]_[0].csv`` holds the
    morning half of a day, ``speed[<day>]_[1].csv`` the afternoon half.
    """

    def __init__(
        self,
        data_dir: Path,
        links_file: str,
        instance_day: int,
        traffic_days: Sequence[int],
        shortest_paths_file: str,
    ) -> None:
        self.data_dir = data_dir
        self.links_file = links_file
        self.instance_day = instance_day
        self.traffic_days = tuple(traffic_days)
        self.shortest_paths_file = shortest_paths_file

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> CsvDataSource:
        return cls(
            config.data_dir,
            config.links_file,
            config.instance_day,
            config.traffic_days,
            config.shortest_paths_file,
        )

    def load_road_network(self) -> RoadNetwork:
        return RoadNetwork(pd.read_csv(self.data_dir / self.links_file))

    def load_shortest_path_cache(self) -> ShortestPathCache:
        return ShortestPathCache.from_csv(self.data_dir / self.shortest_paths_file)

    def load_traffic_history(self) -> TrafficHistory:
        observations_by_day = {day: self._read_day(day) for day in self.traffic_days}
        return TrafficHistory(observations_by_day, self.instance_day)

    def _read_day(self, day: int) -> pd.DataFrame:
        halves = [pd.read_csv(self.data_dir / f"speed[{day}]_[{half}].csv") for half in (0, 1)]
        return pd.concat(halves, ignore_index=True)
