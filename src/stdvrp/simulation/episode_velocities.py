"""EpisodeVelocities: the arc velocities one Episode actually experiences.

Simulation-performance ticket 09 (the decomposition ticket): the Model used to
own a stochastic-sampling subsystem alongside its event loop — an injected
``Generator``, a memo dict, a congestion dict and three methods threading
between them. All of it lives here now.

In Powell's terms this is the Episode's *exogenous information*: what the
transition function learns about the world between one decision and the next.
It is the reason the class lives under ``simulation`` and not under ``traffic``
or ``congestion`` — it draws from the :class:`TravelTimeModel`'s distributions
and it holds the :class:`CongestionGenerator`'s events, so it belongs to
neither, only to the Episode. Concrete by design (ADR-0002): no seam.

One instance per Episode, constructed by the Model with that Episode's
``velocity_rng`` (ticket 13, ADR-0001 phase 2: one stream per stochastic
concern, never shared with congestion or with the Policy's exploration).

Preserved legacy quirks (ADR-0001):

- **Normal samples are memoized per (arc, even minute); congested ones are
  not.** A congested velocity is a deterministic function of the event and the
  arc's mean speed, so re-deriving it is cheap — but it also means the arc's
  *uncongested* sample for that minute may be drawn later, out of the order a
  memo-everything sampler would draw in.
- **Time is bucketed to even minutes** by flooring and stepping back one minute
  when odd, matching the two-minute decision epochs.
- The 60 km/h cap applies only to arcs whose mean speed is below 60; the
  120 km/h cap applies to all. Speeds are km/min, so those read as 1 and 2.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from stdvrp.congestion import CongestedArcs
from stdvrp.traffic.travel_time_model import TravelTimeModel

# (arc start node, arc end node, even minute) — the memo and the TravelTimeModel
# lookups are both keyed this way.
ArcMinuteKey = tuple[float, float, int]


class ArcVelocity(NamedTuple):
    """One arc traversal as sampled: how long it takes, how fast, how far."""

    travel_time: float
    velocity: float
    length: float


class EpisodeVelocities:
    """Samples (and remembers) the velocity on an arc at a moment in the Episode.

    :attr:`congested_arcs` is the live event book: the
    :class:`CongestionGenerator` writes into it every congestion epoch, this
    class reads it when sampling, and the Model reads it to find the next
    expiry. It is mutated in place and never rebound, so a reference taken once
    stays valid for the Episode.
    """

    def __init__(
        self, travel_time_model: TravelTimeModel, velocity_rng: np.random.Generator
    ) -> None:
        self._travel_time_model = travel_time_model
        self._rng = velocity_rng
        self._sampled: dict[ArcMinuteKey, ArcVelocity] = {}
        self.congested_arcs: CongestedArcs = {}

    def sample(self, node_start: float, node_end: float, tau_episode: float) -> ArcVelocity:
        """The velocity on ``node_start -> node_end`` at ``tau_episode``.

        Ports ``create_random_velocity``. A congested arc yields its event's
        deterministic slowdown; anything else yields the memoized normal draw
        for the arc's even minute (see the module docstring for why only the
        latter is remembered).
        """
        minute_start = math.floor(tau_episode)
        if minute_start % 2 != 0:
            minute_start -= 1
        key = (node_start, node_end, minute_start)

        event = self.congested_arcs.get((node_start, node_end))
        if event is None or tau_episode >= event[1]:
            return self._normal_velocity(key)

        length, speed = self._travel_time_model.travel_data[key]  # type: ignore[index]
        velocity = max(speed * event[0], 0.0001)
        return ArcVelocity(length / velocity, velocity, length)

    @property
    def any_congestion(self) -> bool:
        """Is any arc congested at all?

        The Model's per-vehicle expiry scan asks this first: most epochs of most
        Episodes have an empty event book, and then there is nothing to scan for.
        """
        return bool(self.congested_arcs)

    def congestion_expiry(self, arc: tuple[float, float]) -> float | None:
        """When the congestion on ``arc`` lifts, or ``None`` if it is not congested."""
        event = self.congested_arcs.get(arc)
        return None if event is None else event[1]

    def purge_expired(self, tau_episode: float) -> None:
        """Drop every event that has lifted by ``tau_episode`` (B17).

        Unpurged, an expired event sits in the book until :meth:`release`
        clears it at Episode end: :attr:`any_congestion` stays permanently
        ``True`` after the first roll and every later lookup carries dead
        weight. Semantically inert either way — :meth:`sample` and
        :meth:`congestion_expiry` already treat ``tau_episode >= event[1]`` as
        uncongested — so this only shrinks the book, it never changes what an
        arc samples as.
        """
        if not self.congested_arcs:
            return
        for arc in [arc for arc, event in self.congested_arcs.items() if tau_episode >= event[1]]:
            del self.congested_arcs[arc]

    def release(self) -> None:
        """Drop this Episode's samples and events once it has terminated.

        Both Episode runners call this at termination so a finished Model — one
        a training Episode still holds through its snapshots — stops pinning a
        per-arc-minute memo. The legacy also cleared its event book *before* the
        loop; a Model is built fresh per Episode, so that clear could only ever
        have been a no-op and is gone.
        """
        self._sampled.clear()
        self.congested_arcs.clear()

    def _normal_velocity(self, key: ArcMinuteKey) -> ArcVelocity:
        """The memoized uncongested sample for an arc-minute, drawing one if absent."""
        sampled = self._sampled.get(key)
        if sampled is None:
            return self._draw_normal_velocity(key)
        return sampled

    def _draw_normal_velocity(self, key: ArcMinuteKey) -> ArcVelocity:
        """Ports ``generate_normal_velocity``: one ``velocity_rng`` draw per arc-minute.

        ``key``'s minute must already be the even-floored one; the legacy
        re-floored it internally, but its only caller (:meth:`sample`) does so
        first.
        """
        length, speed = self._travel_time_model.travel_data[key]  # type: ignore[index]
        std = self._travel_time_model.speed_std[key]  # type: ignore[index]

        velocity = float(self._rng.normal(speed, std))
        velocity = max(velocity, 0)

        # Speeds are km/min: 1 is the 60 km/h cap for ordinary streets, 2 the
        # 120 km/h absolute one. The floor catches an exactly-zero draw only —
        # the max above has already ruled out negatives.
        if speed < 1 and velocity > 1:
            velocity = 1
        if velocity > 2:
            velocity = 2
        if velocity <= 0:
            velocity = 0.001

        sampled = ArcVelocity(length / velocity, velocity, length)
        self._sampled[key] = sampled
        return sampled
