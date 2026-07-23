"""ShortestPathCache: precomputed shortest paths from network nodes to Clients.

Phase-1 port of the legacy ``shortest_path_memory`` (ADR-0001): the cache is purely
CSV-loaded — the legacy computed the paths once, saved them, and every run since
has only read them back. The parse is kept verbatim, including the quirk that path
node ids become floats (``float`` node ids hash and compare equal to the int ids
used everywhere else, so lookups still work); the average time and length columns
carry whatever units the capture wrote (minutes and kilometres for Chengdu).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple


class ShortestPath(NamedTuple):
    """One cached route: its node sequence, mean travel time and total length."""

    nodes: list[float]
    average_minutes: float
    length: float


class ShortestPathCache:
    """Lookup of the precomputed shortest path between any node and Client node."""

    def __init__(self, paths: dict[tuple[float, float], ShortestPath]) -> None:
        self._paths = paths

    @classmethod
    def from_csv(cls, path: Path) -> ShortestPathCache:
        """Read the legacy CSV format: Node, Client, ``a->b->c`` path, time, length."""
        paths: dict[tuple[float, float], ShortestPath] = {}
        with open(path, newline="") as file:
            reader = csv.reader(file)
            next(reader)  # header
            for node, client, path_str, average_minutes, length in reader:
                paths[(int(node), int(client))] = ShortestPath(
                    [float(n) for n in path_str.split("->")],
                    float(average_minutes),
                    float(length),
                )
        return cls(paths)

    def path_between(self, node: float, client: float) -> ShortestPath:
        """The cached path from ``node`` to ``client``; KeyError when the pair is absent."""
        return self._paths[(node, client)]

    def __contains__(self, pair: tuple[float, float]) -> bool:
        return pair in self._paths

    def __len__(self) -> int:
        return len(self._paths)

    def as_dict(self) -> dict[tuple[float, float], ShortestPath]:
        """A copy of the full mapping — characterization tests only; it is large."""
        return dict(self._paths)
