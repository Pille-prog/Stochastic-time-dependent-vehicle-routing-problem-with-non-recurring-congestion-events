"""RoadNetwork: the directed graph of nodes and arcs of the instance (CONTEXT.md).

Deliberately concrete — no interface (ADR-0002).
"""

from __future__ import annotations

import pandas as pd

LINK_COLUMNS = (
    "Link",
    "Node_Start",
    "Longitude_Start",
    "Latitude_Start",
    "Node_End",
    "Longitude_End",
    "Latitude_End",
    "Length",
)


class RoadNetwork:
    """Static arcs of the road network: end nodes, coordinates, length in meters.

    Wraps the raw links table (legacy ``link.csv``) untransformed; unit conversions
    (meters to km) happen in :class:`~stdvrp.traffic.TravelTimeModel`, mirroring the
    legacy pipeline so derived values stay bit-identical.
    """

    def __init__(self, links: pd.DataFrame) -> None:
        missing = [column for column in LINK_COLUMNS if column not in links.columns]
        if missing:
            raise ValueError(f"links table is missing columns {missing}")
        self.links = links

    @property
    def arc_count(self) -> int:
        return len(self.links)

    @property
    def node_ids(self) -> list[int]:
        """Unique arc-start nodes in file order (legacy ``environment.node_list``).

        Legacy semantics preserved: a node appearing only as an arc end is excluded.
        """
        return list(self.links.drop_duplicates(subset="Node_Start")["Node_Start"])
