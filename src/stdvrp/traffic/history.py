"""TrafficHistory: historical speed observations per arc and time interval (CONTEXT.md)."""

from __future__ import annotations

import pandas as pd


class TrafficHistory:
    """Raw per-day speed observations (``Period``, ``Link``, ``Speed`` in km/h).

    Each day's frame is the morning and afternoon halves concatenated in that order,
    exactly as the legacy read them. ``days`` preserves the configured order because
    the multi-day aggregation concatenates the frames in that order and float sums
    are order-sensitive (ADR-0001).
    """

    def __init__(self, observations_by_day: dict[int, pd.DataFrame], instance_day: int) -> None:
        if instance_day not in observations_by_day:
            raise ValueError(f"instance day {instance_day} has no observations")
        self.observations_by_day = observations_by_day
        self.instance_day = instance_day

    @property
    def days(self) -> tuple[int, ...]:
        return tuple(self.observations_by_day)

    @property
    def instance_day_observations(self) -> pd.DataFrame:
        """The single day the simulation runs on (legacy ``environment.data_velocity``)."""
        return self.observations_by_day[self.instance_day]
