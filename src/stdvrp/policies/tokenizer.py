"""tokenize: ``State``/``TrainingSnapshot`` + ``EpisodeGeometry`` -> the neural
Policy's raw-fact tokens (ticket 04, neural-policy, ADR-0006).

A pure function, not a class with per-Episode setup (contrast
:class:`~stdvrp.policies.feature_extraction.FeatureExtractor`): every value here is
a fact the snapshot or the geometry already holds, so there is nothing to cache
between calls. **The observability rule** (spec.md, "The observability rule,
precisely"; ADR-0006) is this module's whole reason to exist: :func:`tokenize`'s
signature admits exactly five things — a ``State`` or ``TrainingSnapshot``, the
``TimeWindows``, the ``EpisodeGeometry``, and the three config clocks
(``horizon_start_minute``, ``shift_end_minute``, ``episode_end_minute``) — and
nothing else. It cannot reach ``EpisodeVelocities``, ``congested_arcs``,
``TravelTimeModel`` evaluated at ``tau``, or ``FleetRoutes``, because none of
those are arguments; ``tests/unit/test_tokenizer.py::TestObservabilityRule``
pins the signature and this module's imports structurally, not as a docstring
promise. ``EpisodeGeometry.average_minutes``/``.length`` are the one exception,
permitted because they are an offline historical prior baked into the CSV at
capture time, not an observation of this Episode (ADR-0006) — and the identical
object the linear baseline reads.

``claimed`` (which Client each vehicle currently targets) is deliberately not a
token field: it enters at the per-vehicle head (ticket 06), not the encoder, so
one encoder pass serves the whole per-vehicle decision sweep (spec.md decision 6).

## The three tokens

Every field below is a raw fact — never a cost, a polynomial or a bin (the
forbidden list: ``earliness_cost``, ``delay_cost``, ``future_delay``,
``overtime_cost``, the 400/500/600 earliness bins, ``clients_left**2 * time``
and its siblings, the ``late_count / 13`` normalizer, the ``350``/``310``
depot-idle literals — none of them appear here, structurally enforced by the
same test as the observability rule).

``client_tokens``: one row per ``snapshot.clients_not_visited`` entry, **in that
list's order** — this is what makes the tokens permutation-equivariant, not an
incidental detail (spec.md: attention is the right tool for this precisely
because reordering the input reorders the output identically). Row width
``3 + 2 * m`` (``m`` = vehicle count):

    [tw_start, tw_end, tw_end - tau,
     minutes_from_vehicle[0..m-1], path_length_from_vehicle[0..m-1]]

``vehicle_tokens``: one row per vehicle, in ``last_node_reached`` order. Row
width ``3 + n`` (``n`` = ``len(snapshot.observed_velocity[0])``):

    [standing, completing_service, minutes_to_depot, observed_velocity[0..n-1]]

``global_token``: one fixed-width vector, width 5:

    [tau, shift_end - tau, episode_end - tau, n_pending, n_vehicles]

## Normalization (fixed, config-derived — not running statistics)

Running statistics would make one Episode's tokens depend on which Episodes ran
before it, which breaks the per-seed reproducibility the paired comparison
depends on (spec.md, "Why the paired comparison is valid"). Every scale below is
instead a constant derived once from this call's own arguments:

- ``horizon_length = shift_end_minute - horizon_start_minute``: divides every
  minute-valued field whose natural range is "within one shift". The three
  absolute-clock reads — ``tw_start``, ``tw_end`` and ``tau`` (in
  ``global_token``) — first subtract ``horizon_start_minute``, so all three
  share one zero point instead of floating unanchored on the simulated clock;
  the duration fields — ``tw_end - tau``, ``minutes_from_vehicle``,
  ``minutes_to_depot`` and ``shift_end - tau`` — have no origin to subtract
  (a difference of two same-origin clock reads is already origin-free) and
  are divided by ``horizon_length`` directly. ``path_length_from_vehicle``
  (kilometres, not minutes) has no minute-based scale to borrow from the two
  config clocks — dividing it by ``horizon_length`` too is not a unit
  conversion, only a fixed, deterministic choice that keeps its magnitude
  comparable to the minute-valued fields it shares a token row with, instead
  of one field dwarfing the rest.
- ``episode_length = episode_end_minute - horizon_start_minute``: divides
  ``episode_end - tau``, whose natural range extends past ``horizon_length``
  (the Episode can and does run past the shift end).
- ``total_clients = len(geometry.columns) - 1`` (the depot column excluded):
  divides ``n_pending``, turning a raw count into "the fraction of this
  Episode's demand still pending" — bounded in ``[0, 1]`` like the scaled time
  fields, and derived from the ``EpisodeGeometry`` already in hand rather than a
  new config scalar.
- Left unscaled, deliberately: ``standing`` and ``completing_service`` are
  already 0/1 flags; ``observed_velocity`` is already O(1) km/min (an urban
  road network's speeds are well under 2 km/min, never the three-digit
  magnitude raw minutes reach); ``n_vehicles`` is a small raw integer (this
  problem's fleet sizes run single digits to the low tens — see
  ``ClientGenerator.generate``) already the same order of magnitude as the
  scaled fields, unlike ``n_pending``, which reaches into the hundreds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from stdvrp.network.episode_geometry import EpisodeGeometry
from stdvrp.policies.feature_extraction import TimeWindows

if TYPE_CHECKING:
    from stdvrp.simulation.state import State, TrainingSnapshot

__all__ = ["Tokens", "tokenize"]

CLIENT_TOKEN_BASE_WIDTH = 3
VEHICLE_TOKEN_BASE_WIDTH = 3
GLOBAL_TOKEN_WIDTH = 5

# EpisodeGeometry.build always places the depot at column 0 (its own contract:
# "columns are depot first, then this Episode's Clients") — relied on already by
# FeatureExtractor._overtime_costs. Reusing that contract here needs no ``depot``
# argument, one fewer thing for the observability test to have to admit.
_DEPOT_COLUMN = 0


@dataclass(frozen=True, slots=True)
class Tokens:
    """One decision epoch's whole observation: three raw-fact tensors, nothing else."""

    client_tokens: NDArray[np.float64]
    """``[n_pending, 3 + 2*m]``, row order == ``snapshot.clients_not_visited`` order."""

    vehicle_tokens: NDArray[np.float64]
    """``[m, 3 + n]``, row order == ``snapshot.last_node_reached`` order."""

    global_token: NDArray[np.float64]
    """``[5]``."""


def tokenize(
    snapshot: State | TrainingSnapshot,
    geometry: EpisodeGeometry,
    time_windows: TimeWindows,
    *,
    horizon_start_minute: int,
    shift_end_minute: int,
    episode_end_minute: int,
) -> Tokens:
    """The Policy's entire observation of one decision epoch, as raw facts.

    Pure: every output value is computed only from the five arguments above, so
    the same snapshot tokenizes bit-identically every time
    (``tests/unit/test_tokenizer.py::TestPureFunction``).
    """
    tau = float(snapshot.tau_episode)
    positions = snapshot.last_node_reached
    pending = snapshot.clients_not_visited

    horizon_length = float(shift_end_minute - horizon_start_minute)
    episode_length = float(episode_end_minute - horizon_start_minute)
    total_clients = float(len(geometry.columns) - 1)

    vehicle_minutes = geometry.average_minutes_rows(positions)
    vehicle_length = geometry.length_rows(positions)

    client_tokens = _client_tokens(
        pending,
        time_windows,
        tau,
        geometry,
        vehicle_minutes,
        vehicle_length,
        horizon_start_minute,
        horizon_length,
    )
    vehicle_tokens = _vehicle_tokens(snapshot, vehicle_minutes, horizon_length)
    global_token = np.array(
        [
            (tau - horizon_start_minute) / horizon_length,
            (shift_end_minute - tau) / horizon_length,
            (episode_end_minute - tau) / episode_length,
            len(pending) / total_clients,
            float(len(positions)),
        ],
        dtype=np.float64,
    )

    return Tokens(
        client_tokens=client_tokens, vehicle_tokens=vehicle_tokens, global_token=global_token
    )


def _client_tokens(
    pending: list[int] | tuple[int, ...],
    time_windows: TimeWindows,
    tau: float,
    geometry: EpisodeGeometry,
    vehicle_minutes: NDArray[np.float64],
    vehicle_length: NDArray[np.float64],
    horizon_start_minute: int,
    horizon_length: float,
) -> NDArray[np.float64]:
    n_pending = len(pending)
    number_vehicles = vehicle_minutes.shape[0]

    if n_pending:
        windows = np.array([time_windows[client] for client in pending], dtype=np.float64)
    else:
        windows = np.empty((0, 2), dtype=np.float64)
    tw_start, tw_end = windows[:, 0], windows[:, 1]

    columns = geometry.column_positions(pending)
    minutes_from_vehicle = vehicle_minutes[:, columns]  # [number_vehicles, n_pending]
    length_from_vehicle = vehicle_length[:, columns]  # [number_vehicles, n_pending]

    tokens = np.empty((n_pending, CLIENT_TOKEN_BASE_WIDTH + 2 * number_vehicles), dtype=np.float64)
    # Same clock origin as tau/global_token below, so tw_start/tw_end share tau's
    # zero point instead of floating unanchored on the absolute simulated clock.
    tokens[:, 0] = (tw_start - horizon_start_minute) / horizon_length
    tokens[:, 1] = (tw_end - horizon_start_minute) / horizon_length
    tokens[:, 2] = (tw_end - tau) / horizon_length
    minutes_end = 3 + number_vehicles
    length_end = minutes_end + number_vehicles
    tokens[:, 3:minutes_end] = minutes_from_vehicle.T / horizon_length
    tokens[:, minutes_end:length_end] = length_from_vehicle.T / horizon_length
    return tokens


def _vehicle_tokens(
    snapshot: State | TrainingSnapshot,
    vehicle_minutes: NDArray[np.float64],
    horizon_length: float,
) -> NDArray[np.float64]:
    number_vehicles = vehicle_minutes.shape[0]
    n_observed_velocities = len(snapshot.observed_velocity[0])

    tokens = np.empty(
        (number_vehicles, VEHICLE_TOKEN_BASE_WIDTH + n_observed_velocities), dtype=np.float64
    )
    tokens[:, 0] = np.asarray(snapshot.vehicle_standing, dtype=np.float64)
    tokens[:, 1] = np.asarray(snapshot.vehicle_completing_service, dtype=np.float64)
    tokens[:, 2] = vehicle_minutes[:, _DEPOT_COLUMN] / horizon_length
    tokens[:, 3:] = np.asarray(snapshot.observed_velocity, dtype=np.float64)
    return tokens
