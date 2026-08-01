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

## The cost fields (2026-08-01 amendment to spec.md decision 1)

Decision 1 originally forbade a token from carrying "a cost formula, a
polynomial or a bin". Ticket 08's Gate A run amended it, on this diagnosis:
``Q(s, a)`` decomposes as ``V(s) + A(s, a)``, the Monte Carlo return's variance
is almost entirely ``V(s)``, so a network fed only raw facts spends its
capacity fitting ``V`` and leaves ``A`` — the only quantity the argmin reads —
as an unlearnable residual rediscovered from noisy returns. The fix is the
research note's own ranked recommendation #5 ("neural VFA over *enriched*
features", Chen/Ulmer/Thomas, up to +22 %): each ``(client, vehicle)`` arc
token now carries the **projected components of the simulator's own cost
function** for that assignment, so ``A`` is nearly linear in the inputs instead
of something the optimizer has to reconstruct.

What was amended and what was not, precisely:

- **The observability rule (decision 3, ADR-0006) is untouched.** Every cost
  below is arithmetic over the five permitted inputs — ``tau``, the time
  windows, ``EpisodeGeometry.average_minutes`` — plus the simulator's own
  hardcoded cost constants. No new observable enters; the signature is the
  same five arguments, still pinned by the same test.
- **The linear baseline's *state* feature engineering stays out**: the
  ``clients_left**2 * time`` polynomials, the 400/500/600 earliness bins, the
  ``late_count / 13`` normalizer, the ``350``/``310`` depot-idle literals, the
  closest-client multiplicity classifier. Those encode a hand-designed *state*
  representation, which is exactly what the transformer exists to learn.
  What the amendment admits is only the **cost function the run is scored
  by** — the four per-candidate projected costs the baseline's state-action
  features price (``FeatureExtractor.candidate_features``), as *per-pair
  marginals* rather than joint-action sums, since the joint action does not
  exist yet at tokenize time.

The cost constants (``SERVICE_MINUTES``, ``EARLINESS_COST_RATE``,
``DELAY_COST_RATE``, ``OVERTIME_COST_RATE``) mirror the simulator's own
hardcoded rates (``stdvrp/simulation/cost_ledger.py``,
``stdvrp/simulation/model.py::SERVICE_MINUTES``) — **duplicated, not
imported**, the same documented precedent as ``transformer_policy.py``'s
``_DELAY_COST_FACTOR``/``_OVERTIME_COST_FACTOR``: importing them would widen
this module's import surface toward the simulator for four constants that are
part of the problem definition, and the observability test pins this module's
imports.

## The five tensors

``client_tokens``: one row per ``snapshot.clients_not_visited`` entry, **in that
list's order** — this is what makes the tokens permutation-equivariant, not an
incidental detail (spec.md: attention is the right tool for this precisely
because reordering the input reorders the output identically). Row width 3:

    [tw_start, tw_end, tw_end - tau]

``arc_tokens``: ``[n_pending, m, 6]`` (``m`` = vehicle count) — one row per
``(client, vehicle)`` pair, the action-conditional pathway ``QHead`` scores.
With ``arrival = tau + minutes_from_vehicle``:

    [minutes_from_vehicle, path_length_from_vehicle,
     earliness_cost, delay_cost, future_delay, overtime_cost]

where, mirroring ``FeatureExtractor``'s per-candidate formulas exactly (the
``elif`` that makes earliness and delay mutually exclusive, ``max(due, tau)``
as the delay/overtime reference so an already-late Client is priced only on
the marginal minutes past ``tau``):

- ``earliness_cost = (tw_start - arrival) * EARLINESS_COST_RATE`` when
  ``arrival < tw_start``, else 0;
- ``delay_cost = (arrival - max(due, tau)) * DELAY_COST_RATE`` when not early
  and ``arrival > due``, else 0;
- ``future_delay = sum over every other pending Client c' of
  (followup - max(due_c', tau)) * DELAY_COST_RATE where followup =
  arrival + SERVICE_MINUTES + minutes(client, c') breaches due_c'`` — the
  baseline's follow-up formula minus its closest-client multiplicity
  classifier (hand engineering ADR-0007 already retired for this Policy);
- ``overtime_cost = (homecoming - max(shift_end, tau)) * OVERTIME_COST_RATE``
  when ``homecoming = arrival + SERVICE_MINUTES + minutes(client, depot)``
  breaches ``shift_end_minute``, else 0.

``depot_arc_tokens``: ``[m, 6]`` — the same six fields for the synthetic depot
candidate ticket 06 appends to every sweep, so the depot's arc half is built
by the same arithmetic as every real candidate's (previously the Policy
hand-built a 2-wide pair from the geometry; now it reads this). Earliness and
delay are 0 (the depot has no time window); ``future_delay`` prices the
follow-up legs out of the depot by the same uniform formula the baseline
applies to its own depot candidate; ``overtime_cost`` prices the **direct**
return leg (``tau + minutes_to_depot``, no service stop — the baseline's own
depot branch in ``_overtime_costs``).

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
  of one field dwarfing the rest. The three single-Client cost fields
  (``earliness_cost``, ``delay_cost``, ``overtime_cost``) are each one
  rate-scaled duration, so they divide by ``horizon_length`` for the same
  reason the durations do.
- ``horizon_length * total_clients`` divides ``future_delay``, whose natural
  range is a *sum* over up to every pending Client — without the extra factor
  it would dwarf every other field at real scale (150 Clients).
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

**``future_delay`` sums in sorted order, not pending order.** Float addition is
not associative, and the sum ranges over the *other* pending Clients — whose
order is exactly what the permutation-equivariance property
(``TestPermutationEquivariance``) permutes. Sorting the summands first makes
the sum a function of the multiset alone, so reordering ``clients_not_visited``
still reorders ``arc_tokens`` bit-identically instead of perturbing last bits.
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

__all__ = [
    "ARC_TOKEN_WIDTH",
    "DELAY_COST_RATE",
    "EARLINESS_COST_RATE",
    "OVERTIME_COST_RATE",
    "SERVICE_MINUTES",
    "Tokens",
    "tokenize",
]

CLIENT_TOKEN_BASE_WIDTH = 3
VEHICLE_TOKEN_BASE_WIDTH = 3
GLOBAL_TOKEN_WIDTH = 5
ARC_TOKEN_WIDTH = 6

# The simulator's own cost function, exactly as it hardcodes it
# (stdvrp/simulation/cost_ledger.py rates, stdvrp/simulation/model.py
# SERVICE_MINUTES) -- duplicated, not imported; see the module docstring,
# "The cost fields".
SERVICE_MINUTES = 5.0
EARLINESS_COST_RATE = 0.1
DELAY_COST_RATE = 1.0
OVERTIME_COST_RATE = 5 / 6

# EpisodeGeometry.build always places the depot at column 0 (its own contract:
# "columns are depot first, then this Episode's Clients") — relied on already by
# FeatureExtractor._overtime_costs. Reusing that contract here needs no ``depot``
# argument, one fewer thing for the observability test to have to admit.
_DEPOT_COLUMN = 0


@dataclass(frozen=True, slots=True)
class Tokens:
    """One decision epoch's whole observation: five raw-fact tensors, nothing else."""

    client_tokens: NDArray[np.float64]
    """``[n_pending, 3]``, row order == ``snapshot.clients_not_visited`` order."""

    vehicle_tokens: NDArray[np.float64]
    """``[m, 3 + n]``, row order == ``snapshot.last_node_reached`` order."""

    global_token: NDArray[np.float64]
    """``[5]``."""

    arc_tokens: NDArray[np.float64]
    """``[n_pending, m, 6]``: the ``(client, vehicle)`` action-conditional facts —
    ``[minutes, length, earliness_cost, delay_cost, future_delay, overtime_cost]``.
    Client axis order == ``client_tokens``, vehicle axis order == ``vehicle_tokens``."""

    depot_arc_tokens: NDArray[np.float64]
    """``[m, 6]``: the same six fields for the synthetic depot candidate, per vehicle."""


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
        pending, time_windows, tau, horizon_start_minute, horizon_length
    )
    arc_tokens, depot_arc_tokens = _arc_tokens(
        pending,
        time_windows,
        tau,
        geometry,
        vehicle_minutes,
        vehicle_length,
        shift_end_minute=shift_end_minute,
        horizon_length=horizon_length,
        total_clients=total_clients,
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
        client_tokens=client_tokens,
        vehicle_tokens=vehicle_tokens,
        global_token=global_token,
        arc_tokens=arc_tokens,
        depot_arc_tokens=depot_arc_tokens,
    )


def _client_windows(
    pending: list[int] | tuple[int, ...], time_windows: TimeWindows
) -> NDArray[np.float64]:
    if len(pending):
        return np.array([time_windows[client] for client in pending], dtype=np.float64)
    return np.empty((0, 2), dtype=np.float64)


def _client_tokens(
    pending: list[int] | tuple[int, ...],
    time_windows: TimeWindows,
    tau: float,
    horizon_start_minute: int,
    horizon_length: float,
) -> NDArray[np.float64]:
    windows = _client_windows(pending, time_windows)
    tw_start, tw_end = windows[:, 0], windows[:, 1]

    tokens = np.empty((len(pending), CLIENT_TOKEN_BASE_WIDTH), dtype=np.float64)
    # Same clock origin as tau/global_token, so tw_start/tw_end share tau's
    # zero point instead of floating unanchored on the absolute simulated clock.
    tokens[:, 0] = (tw_start - horizon_start_minute) / horizon_length
    tokens[:, 1] = (tw_end - horizon_start_minute) / horizon_length
    tokens[:, 2] = (tw_end - tau) / horizon_length
    return tokens


def _arc_tokens(
    pending: list[int] | tuple[int, ...],
    time_windows: TimeWindows,
    tau: float,
    geometry: EpisodeGeometry,
    vehicle_minutes: NDArray[np.float64],
    vehicle_length: NDArray[np.float64],
    *,
    shift_end_minute: int,
    horizon_length: float,
    total_clients: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The ``(client, vehicle)`` arc block and the depot's — see the module docstring."""
    n_pending = len(pending)
    number_vehicles = vehicle_minutes.shape[0]

    windows = _client_windows(pending, time_windows)
    tw_start, due = windows[:, 0], windows[:, 1]
    # An already-late Client is priced on the marginal minutes past tau, not
    # past its long-gone due time — FeatureExtractor's own due_or_tau.
    due_or_tau = np.maximum(due, tau)
    # FeatureExtractor._overtime_costs' reference clock: past the shift end,
    # overtime is priced from tau (the sunk part is already charged).
    overtime_reference = float(shift_end_minute) if tau < shift_end_minute else tau

    columns = geometry.column_positions(pending)
    minutes = vehicle_minutes[:, columns].T  # [n_pending, number_vehicles]
    lengths = vehicle_length[:, columns].T  # [n_pending, number_vehicles]
    arrival = tau + minutes

    early = arrival < tw_start[:, None]
    late = ~early & (arrival > due[:, None])
    earliness_cost = np.where(early, (tw_start[:, None] - arrival) * EARLINESS_COST_RATE, 0.0)
    delay_cost = np.where(late, (arrival - due_or_tau[:, None]) * DELAY_COST_RATE, 0.0)

    # Follow-up legs out of each candidate: minutes from every pending Client
    # to every column (the same geometry read FeatureExtractor's candidate_rows
    # makes), sliced to the pending Clients and the depot.
    client_rows = geometry.average_minutes_rows(pending)  # [n_pending, columns]
    followup_minutes = client_rows[:, columns]  # [n_pending, n_pending]
    minutes_home = client_rows[:, _DEPOT_COLUMN]  # [n_pending]

    followup = (
        arrival[:, :, None] + SERVICE_MINUTES + followup_minutes[:, None, :]
    )  # [n_pending, number_vehicles, n_pending]
    followup_delay = np.where(
        followup > due[None, None, :],
        (followup - due_or_tau[None, None, :]) * DELAY_COST_RATE,
        0.0,
    )
    if n_pending:
        # A candidate's own follow-up to itself is not a follow-up.
        self_index = np.arange(n_pending)
        followup_delay[self_index, :, self_index] = 0.0
    future_delay = _order_free_sum(followup_delay)  # [n_pending, number_vehicles]

    homecoming = arrival + SERVICE_MINUTES + minutes_home[:, None]
    overtime_cost = np.where(
        homecoming > shift_end_minute,
        (homecoming - overtime_reference) * OVERTIME_COST_RATE,
        0.0,
    )

    arc_tokens = np.empty((n_pending, number_vehicles, ARC_TOKEN_WIDTH), dtype=np.float64)
    arc_tokens[:, :, 0] = minutes / horizon_length
    arc_tokens[:, :, 1] = lengths / horizon_length
    arc_tokens[:, :, 2] = earliness_cost / horizon_length
    arc_tokens[:, :, 3] = delay_cost / horizon_length
    arc_tokens[:, :, 4] = future_delay / (horizon_length * total_clients)
    arc_tokens[:, :, 5] = overtime_cost / horizon_length

    # The synthetic depot candidate's block, by the same arithmetic. No time
    # window, so no earliness/delay; overtime prices the *direct* return leg
    # (no service stop), the baseline's own depot branch.
    depot_minutes = vehicle_minutes[:, _DEPOT_COLUMN]  # [number_vehicles]
    depot_length = vehicle_length[:, _DEPOT_COLUMN]
    depot_arrival = tau + depot_minutes
    depot_node = geometry.columns[_DEPOT_COLUMN]
    depot_followup_minutes = geometry.average_minutes_row(depot_node)[columns]  # [n_pending]
    depot_followup = (
        depot_arrival[:, None] + SERVICE_MINUTES + depot_followup_minutes[None, :]
    )  # [number_vehicles, n_pending]
    depot_followup_delay = np.where(
        depot_followup > due[None, :],
        (depot_followup - due_or_tau[None, :]) * DELAY_COST_RATE,
        0.0,
    )
    depot_future_delay = _order_free_sum(depot_followup_delay)  # [number_vehicles]
    depot_overtime = np.where(
        depot_arrival > shift_end_minute,
        (depot_arrival - overtime_reference) * OVERTIME_COST_RATE,
        0.0,
    )

    depot_arc_tokens = np.zeros((number_vehicles, ARC_TOKEN_WIDTH), dtype=np.float64)
    depot_arc_tokens[:, 0] = depot_minutes / horizon_length
    depot_arc_tokens[:, 1] = depot_length / horizon_length
    depot_arc_tokens[:, 4] = depot_future_delay / (horizon_length * total_clients)
    depot_arc_tokens[:, 5] = depot_overtime / horizon_length

    return arc_tokens, depot_arc_tokens


def _order_free_sum(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sum over the last axis as a function of the multiset, not the order.

    Float addition is not associative; the last axis here ranges over the
    pending Clients, whose order the permutation-equivariance property permutes
    (module docstring, last section). Sorting first pins one summation order
    per multiset, so equivariance holds bit-exactly.
    """
    result: NDArray[np.float64] = np.sort(values, axis=-1).sum(axis=-1)
    return result


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
