"""Period-string helpers shared by the package, the fixture script and tests."""

from __future__ import annotations


def period_start_minute(period: str) -> int:
    """Minutes since 03:00 of a Period's start, as the legacy pipeline defines it.

    Ports ``environment.convert_time_to_minutes``'s inner ``time_to_minutes``:
    ``"08:00-08:02"`` -> 300.
    """
    start = period.split("-")[0]
    hour, minute = map(int, start.split(":"))
    return (hour - 3) * 60 + minute
