"""FeatureExtractor: the Policy's 19-component feature vector as array arithmetic.

Simulation-performance ticket 05 (ADR-0003, "arrays inside, OO outside"): the
three feature routines that dominated episode time — ``clasify_delayed_clients``,
``extract_general_state_features`` and ``extract_state_action_features`` — move
out of ``MonteCarloPolicy`` into this one cohesive concrete collaborator (no new
seam; ADR-0002) and are recomputed as vectorized numpy over the ticket-04
``EpisodeGeometry`` matrices. Evaluating a vehicle's candidate actions becomes one
``[candidates, 19]`` feature matrix and a single ``X @ W`` instead of one Python
pass per candidate.

**Column space.** Every array here is indexed by *EpisodeGeometry column*: index
``j`` means ``geometry.columns[j]``, column 0 is the depot and the rest are this
Episode's Clients. Time windows, therefore, are read once per Episode into two
dense arrays rather than looked up per Client per candidate. This holds because an
Episode's Clients are drawn without replacement (``ClientGenerator.generate``), so
its Client node ids are distinct and ``clients_not_visited`` is exactly the set of
columns flagged in :attr:`StateFeatures.active`.

**Preserved legacy quirks** (ADR-0001 — do not fix):

- the duplicate-append quirk of ``clasify_delayed_clients``, reproduced exactly by
  :meth:`FeatureExtractor._classify_closest_clients` — see its docstring.
  Simulation-performance ticket 10 measured the one-append-per-Client fix on the
  full real Chengdu dataset (100 training + 500 evaluation + 300 final-test
  episodes, both variants) and found it makes the learned policy consistently
  *worse* — evaluation mean cost +13-18% across every block, final-test mean
  cost +15-25% across every action count — for a ~4% throughput gain. The user
  rejected adoption on that evidence; the quirk stays as the sole, deliberate
  implementation (see the ticket's ``## Comments`` for the full numbers);
- the permanently-zero state-action feature (``X[:, 13]``), which is what pads
  ``W`` to its legacy 19 components and keeps stored weight vectors valid. It
  is, after simulator-correctness ticket 06 (below), the *only* deliberately
  dead weight: ``final_w[13] == 0.0`` by design. Before that ticket, ``W[10]``
  was dead too, but by accident — it never trained because feature 10 was
  identically zero;
- every normalization literal (150, 850, 300, 13, 60, 100, 180, 2500, the
  400/500/600 earliness bins, the 310 depot-idle cutoff), which are part of the
  feature *definition* and stay literal. The one exception is the ``time_left``
  clock ``1150`` used to hardcode: ticket 02 (simulator-correctness, B12)
  threads it in as ``episode_end_minute``, the same configurable hard stop
  :class:`~stdvrp.simulation.model.Model` now uses instead of its own hardcoded
  constant;
- ``mean_velocities``, computed by the general-state routine and never appended as
  a feature. Nothing reads it; it is carried on :class:`StateFeatures` rather than
  dropped because deleting legacy computation is a Tier-3 decision, not this
  ticket's. **If a future modeling effort connects it (B4):** it averages
  ``State.observed_velocity``, a recency window over the last
  ``n_observed_velocities`` decision epochs, not the last N *distinct* arcs
  (simulator-correctness ticket 09, B18) — arguably the better congestion proxy
  of the two, since a congestion event lasts tens of minutes and a time window
  captures that where a distinct-arc window would not.

**Float reassociation (Tier 2).** The four state-action cost sums are re-associated
as "the other vehicles' terms, then the decided vehicle's", and the future-delay
feature multiplies each Client's contribution by its duplicate-append multiplicity
instead of adding it once per duplicate. (The Policy's ``X @ W`` over the batch is a
third such site — a BLAS ``gemv`` rather than 19-term dot products.) The
general-state features stay bit-exact: their only sum is over time-window starts,
which are integers. See the ticket ``## Comments`` for what this measures out to on
the committed fixture.

**B10 fixed (simulator-correctness ticket 06).** The legacy's fourth earliness
bin (``general[10]``) was never assigned, so it was identically zero and
``W[10]`` never trained. :meth:`FeatureExtractor._general_features` now assigns
it as *whatever the first three bins leave uncounted* — not redefined as "a
Client whose window opens at minute 600 or later", because a bin also drops to
zero once ``tau`` passes its own gate (e.g. bin 0 once ``tau >= 400``) while its
Clients are still pending. Bin 3 absorbs those too. This keeps the invariant
the four bins exist to satisfy: they partition pending demand, at every
``tau`` — the four fractions in ``general[7:11]`` always sum to
``len(clients_not_visited) / number_clients``. The bin *boundaries*
(400/500/600) are unchanged; only the bin that was missing is filled in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from stdvrp.network.episode_geometry import EpisodeGeometry

if TYPE_CHECKING:
    from stdvrp.simulation.state import State, TrainingSnapshot

TimeWindows = dict[int, tuple[int, int]]

GENERAL_STATE_FEATURES = 12
STATE_ACTION_FEATURES = 7
FEATURE_COUNT = GENERAL_STATE_FEATURES + STATE_ACTION_FEATURES

# The legacy's depot-idle cutoff for the closest-vehicle classifier: a vehicle
# sitting at the depot this late is no longer a candidate carrier. Inconsistent
# with the 350 used by candidate selection — preserved, not reconciled (ADR-0001).
_DEPOT_IDLE_CUTOFF = 310


@dataclass(frozen=True, slots=True)
class StateFeatures:
    """One State, arranged for the feature arithmetic of a single decision.

    Its headline is :attr:`general`: the twelve general-state features that open
    every candidate's 19-component vector. The rest are the per-State quantities
    the seven state-action features are computed against — all in the column space
    described in the module docstring — plus :attr:`delayed_clients`, which is not
    a feature at all but the classification the Policy's candidate selection reads.
    """

    general: NDArray[np.float64]
    """The twelve general-state features (``X_general_state`` in the legacy)."""

    delayed_clients: tuple[tuple[int, ...], ...]
    """Per vehicle, at most two Clients whose due time its closest-Client list breaches."""

    mean_velocities: tuple[float, ...]
    """Computed by the legacy's general-state routine and never used (see module docstring)."""

    tau: float
    active: NDArray[np.bool_]
    """Per column: that Client is still unvisited (always ``False`` for the depot)."""

    late: NDArray[np.bool_]
    """Per column: ``tau`` is already past that Client's due time."""

    vehicle_minutes: NDArray[np.float64]
    """``[vehicle, column]`` travel minutes from each vehicle's position."""

    vehicle_length: NDArray[np.float64]
    """``[vehicle, column]`` path length from each vehicle's position."""

    closest_client_counts: NDArray[np.int64]
    """``[vehicle, column]`` duplicate-append multiplicities (see the classifier)."""


class FeatureExtractor:
    """Computes the Policy's feature vectors for one Episode's geometry and demand."""

    def __init__(
        self,
        geometry: EpisodeGeometry,
        time_windows: TimeWindows,
        *,
        number_vehicles: int,
        number_clients: int,
        depot: int,
        shift_end_minute: int,
        episode_end_minute: int,
        service_time: float,
        delay_cost_factor: float,
        earliness_cost_factor: float,
        overtime_cost_factor: float,
    ) -> None:
        self._geometry = geometry
        self._number_vehicles = number_vehicles
        self._number_clients = number_clients
        self._depot = depot
        self._end_of_horizon = shift_end_minute
        self._episode_end_minute = episode_end_minute
        self._service_time = service_time
        self._delay_cost_factor = delay_cost_factor
        self._earliness_cost_factor = earliness_cost_factor
        self._overtime_cost_factor = overtime_cost_factor

        self._column_nodes = geometry.columns
        self._column_count = len(self._column_nodes)
        # Sort key for the closest-Client tie-break; the Python ids above are what
        # actually flows into actions, so those are never replaced by numpy scalars.
        self._column_ids = np.asarray(self._column_nodes, dtype=np.float64)
        self._earliness = np.zeros(self._column_count, dtype=np.float64)
        self._due = np.zeros(self._column_count, dtype=np.float64)
        for column, node in enumerate(self._column_nodes):
            window = time_windows.get(node)  # type: ignore[call-overload]
            if window is not None:
                self._earliness[column], self._due[column] = window

    # --- per-State ------------------------------------------------------------

    def state_features(self, state: State | TrainingSnapshot) -> StateFeatures:
        """Everything the State contributes to the decision about to be made.

        Ports ``extract_general_state_features`` together with the
        ``clasify_delayed_clients`` call it ends on. The legacy ran that
        classification twice per decision (once directly from the selection
        routine, once from here) on an unchanged State; it runs once here.
        """
        tau = float(state.tau_episode)
        remaining = state.clients_not_visited
        active = np.zeros(self._column_count, dtype=np.bool_)
        active[self._geometry.column_positions(remaining)] = True

        positions = state.vehicle_position
        vehicle_minutes = self._geometry.average_minutes_rows(positions)
        counts, delayed = self._classify_closest_clients(tau, active, vehicle_minutes, positions)

        return StateFeatures(
            general=self._general_features(tau, len(remaining), active),
            delayed_clients=delayed,
            mean_velocities=tuple(
                sum(velocities) / len(velocities) for velocities in state.observed_velocity
            ),
            tau=tau,
            active=active,
            late=tau > self._due,
            vehicle_minutes=vehicle_minutes,
            vehicle_length=self._geometry.length_rows(positions),
            closest_client_counts=counts,
        )

    def _general_features(
        self, tau: float, remaining_count: int, active: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """Ports the live ``extract_general_state_features`` (12 features), bit-exactly.

        The only sum here is over time-window starts — integers, exactly
        representable — so vectorizing it reorders nothing observable.
        """
        clients_left = remaining_count / 150

        if clients_left != 0:
            time_left = (self._episode_end_minute - tau) / (850)
            time = (tau - 300) / 850
        else:
            time_left = 0.0
            time = 0.0

        earliness = self._earliness[active]

        # The legacy's elif chain over disjoint earliness bands, with the tau test
        # hoisted out of the per-Client loop: the bands cannot overlap, so an
        # elif and an if select the same Clients.
        counts_earliness = [0, 0, 0, 0]
        if tau < 400:
            counts_earliness[0] = int(np.count_nonzero(earliness < 400))
        if tau < 500:
            counts_earliness[1] = int(np.count_nonzero((earliness >= 400) & (earliness < 500)))
        if tau < 600:
            counts_earliness[2] = int(np.count_nonzero((earliness >= 500) & (earliness < 600)))
        # B10: the fourth bin is whatever the first three leave uncounted — not
        # redefined as "earliness >= 600", because a bin above also goes to zero
        # once tau passes its own gate (e.g. bin 0 at tau >= 400) even though its
        # Clients are still pending. Bin 3 absorbs those too, so the four counts
        # always partition pending demand: they sum to `remaining_count` at every
        # tau, which is exactly the invariant this bin exists to satisfy.
        counts_earliness[3] = remaining_count - sum(counts_earliness[:3])

        mean_earliness_diff = 0.0
        if (580 - tau) / (280) > 0 and remaining_count != 0:
            mean_earliness = float(earliness.sum()) / remaining_count
            if mean_earliness > tau:
                mean_earliness_diff = (mean_earliness - tau) / 120

        general = np.empty(GENERAL_STATE_FEATURES, dtype=np.float64)
        general[0] = np.sqrt(clients_left)
        general[1] = time_left
        general[2] = time_left**2
        general[3] = clients_left**2
        general[4] = (clients_left**2) * time
        general[5] = (time**2) * clients_left
        general[6] = (time**2) * (clients_left**2)
        general[7:11] = [count / self._number_clients for count in counts_earliness]
        general[11] = mean_earliness_diff
        return general

    def _classify_closest_clients(
        self,
        tau: float,
        active: NDArray[np.bool_],
        vehicle_minutes: NDArray[np.float64],
        positions: Sequence[float],
    ) -> tuple[NDArray[np.int64], tuple[tuple[int, ...], ...]]:
        """Ports ``clasify_delayed_clients`` — duplicate-append quirk and all.

        The legacy scans the eligible vehicles for each unvisited Client keeping a
        running nearest, and appends a ``(travel time, Client)`` pair to the
        running nearest vehicle's list on *every* iteration after the first rather
        than once per Client. So each Client lands in the lists once per prefix of
        that scan, carrying the prefix's running minimum and going to the first
        vehicle that attained it (the legacy's strict ``<``).

        Vectorized down the vehicle axis: the running minima are one
        ``minimum.accumulate``, and the prefix argmin one ``maximum.accumulate``
        over the seats where a new minimum appears. What reaches the future-delay
        feature is only each ``(vehicle, Client)`` pair's *multiplicity*, returned
        as ``counts``. ``delayed_clients`` additionally needs the pairs in sorted
        order, which the lexsort reproduces exactly — travel time primary, Client
        id breaking ties, as tuple comparison did — keeping the first two whose
        estimated arrival breaches the due time, as the legacy's ``break`` did.

        Assumes finite travel times (a real ShortestPathCache has them): the
        legacy's first comparison is against infinity, which every real value wins.
        """
        vehicle_count = self._number_vehicles
        column_count = self._column_count
        empty_delayed = tuple(() for _ in range(vehicle_count))  # type: tuple[tuple[int, ...], ...]

        eligible = [
            vehicle
            for vehicle, position in enumerate(positions)
            if not (position == self._depot and tau > _DEPOT_IDLE_CUTOFF)
        ]
        columns = np.flatnonzero(active)
        if not eligible or columns.size == 0:
            return np.zeros((vehicle_count, column_count), dtype=np.int64), empty_delayed

        eligible_index = np.asarray(eligible, dtype=np.intp)
        minutes = vehicle_minutes[eligible_index][:, columns]
        running_minimum = np.minimum.accumulate(minutes, axis=0)
        is_new_minimum = np.empty(minutes.shape, dtype=np.bool_)
        is_new_minimum[0] = True
        is_new_minimum[1:] = minutes[1:] < running_minimum[:-1]
        seat = np.maximum.accumulate(
            np.where(is_new_minimum, np.arange(len(eligible))[:, None], -1), axis=0
        )
        owner = eligible_index[seat]
        entry_column = np.broadcast_to(columns, minutes.shape)

        counts: NDArray[np.int64] = np.bincount(
            (owner * column_count + entry_column).ravel(),
            minlength=vehicle_count * column_count,
        ).reshape(vehicle_count, column_count)

        breaches_due = (running_minimum + tau >= self._due[columns]).ravel()
        order = np.lexsort(
            (self._column_ids[entry_column].ravel(), running_minimum.ravel(), owner.ravel())
        )
        order = order[breaches_due[order]]
        if order.size == 0:
            return counts, empty_delayed

        sorted_owner = owner.ravel()[order]
        sorted_column = entry_column.ravel()[order]
        # Sorted by owner first, so each vehicle's entries form one run: keep the
        # first two of every run, which is where the legacy's ``break`` stopped.
        group_start = np.flatnonzero(
            np.concatenate(([True], sorted_owner[1:] != sorted_owner[:-1]))
        )
        group_size = np.diff(np.concatenate((group_start, [sorted_owner.size])))
        within_group = np.arange(sorted_owner.size) - np.repeat(group_start, group_size)
        kept = within_group < 2

        delayed: list[list[int]] = [[] for _ in range(vehicle_count)]
        for vehicle, column in zip(
            sorted_owner[kept].tolist(), sorted_column[kept].tolist(), strict=True
        ):
            delayed[vehicle].append(self._column_nodes[column])
        return counts, tuple(tuple(bucket) for bucket in delayed)

    # --- per-action -----------------------------------------------------------

    def action_features(
        self, features: StateFeatures, action: Sequence[int]
    ) -> NDArray[np.float64]:
        """The 19-component vector of one concrete action, for the ``update_W`` replay."""
        row: NDArray[np.float64] = self.candidate_features(features, action, 0, [action[0]])[0]
        return row

    def candidate_features(
        self,
        features: StateFeatures,
        action: Sequence[int],
        vehicle: int,
        candidates: Sequence[int],
    ) -> NDArray[np.float64]:
        """One 19-component feature row per candidate action of ``vehicle``.

        Ports the live ``extract_state_action_features`` for the whole candidate
        set at once: ``action`` supplies the other vehicles' fixed targets and each
        candidate replaces ``action[vehicle]``. Every other vehicle's contribution
        to the four cost features is therefore a constant across the batch,
        computed once; only the decided vehicle's term is a vector.
        """
        geometry = self._geometry
        tau = features.tau
        due = self._due
        due_or_tau = np.maximum(due, tau)
        service_time = self._service_time
        candidate_count = len(candidates)
        candidate_columns = geometry.column_positions(candidates)

        others = [other for other in range(self._number_vehicles) if other != vehicle]
        other_index = np.asarray(others, dtype=np.intp)
        other_columns = geometry.column_positions([action[other] for other in others])

        # Which Clients still count for a given candidate: unvisited, and not
        # already some vehicle's target — the legacy's ``c not in action``. The
        # depot never matches, since its column is never active.
        keep = np.broadcast_to(
            features.active & ~_column_mask(self._column_count, other_columns),
            (candidate_count, self._column_count),
        ).copy()
        keep[np.arange(candidate_count), candidate_columns] = False

        late_count = np.count_nonzero(keep & features.late, axis=1)

        total_distance = (
            float(features.vehicle_length[other_index, other_columns].sum())
            + features.vehicle_length[vehicle, candidate_columns]
        )

        earliness_cost, delay_cost = self._arrival_costs(
            features, other_index, other_columns, vehicle, candidate_columns, due_or_tau
        )

        other_rows = geometry.average_minutes_rows([action[other] for other in others])
        candidate_rows = geometry.average_minutes_rows(candidates)

        counts = features.closest_client_counts
        other_followup = (
            tau + features.vehicle_minutes[other_index, other_columns][:, None] + other_rows
        ) + service_time
        candidate_followup = (
            tau + features.vehicle_minutes[vehicle, candidate_columns][:, None] + candidate_rows
        ) + service_time
        pooled_future_delay = (
            counts[other_index]
            * np.where(
                other_followup > due, (other_followup - due_or_tau) * self._delay_cost_factor, 0.0
            )
        ).sum(axis=0)
        future_delay = keep @ pooled_future_delay + (
            keep
            * counts[vehicle]
            * np.where(
                candidate_followup > due,
                (candidate_followup - due_or_tau) * self._delay_cost_factor,
                0.0,
            )
        ).sum(axis=1)

        overtime_cost = self._overtime_costs(
            features,
            other_index,
            other_columns,
            other_rows,
            vehicle,
            candidate_columns,
            candidate_rows,
        )

        X = np.empty((candidate_count, FEATURE_COUNT), dtype=np.float64)
        X[:, :GENERAL_STATE_FEATURES] = features.general
        X[:, 12] = late_count / 13
        # The preserved permanently-zero feature: it is what pads W to 19.
        X[:, 13] = 0.0
        X[:, 14] = total_distance / 100.0
        X[:, 15] = earliness_cost / 60.0
        X[:, 16] = delay_cost / 60.0
        X[:, 17] = future_delay / 2500.0
        X[:, 18] = overtime_cost / 180.0
        return X

    def _arrival_costs(
        self,
        features: StateFeatures,
        other_index: NDArray[np.intp],
        other_columns: NDArray[np.intp],
        vehicle: int,
        candidate_columns: NDArray[np.intp],
        due_or_tau: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Earliness and delay of arriving at each vehicle's own target.

        Only Clients still unvisited are priced (the legacy's ``a in clients_all``,
        which is exactly :attr:`StateFeatures.active`), and the legacy's ``elif``
        makes the two costs mutually exclusive per vehicle.
        """
        tau = features.tau
        earliness_tw = self._earliness
        due = self._due

        def costs(
            index: NDArray[np.intp] | int, columns: NDArray[np.intp]
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            active = features.active[columns]
            arrival = tau + features.vehicle_minutes[index, columns]
            early = active & (arrival < earliness_tw[columns])
            late = active & ~early & (arrival > due[columns])
            return (
                np.where(
                    early,
                    (earliness_tw[columns] - arrival) * self._earliness_cost_factor,
                    0.0,
                ),
                np.where(late, (arrival - due_or_tau[columns]) * self._delay_cost_factor, 0.0),
            )

        other_earliness, other_delay = costs(other_index, other_columns)
        candidate_earliness, candidate_delay = costs(vehicle, candidate_columns)
        return (
            float(other_earliness.sum()) + candidate_earliness,
            float(other_delay.sum()) + candidate_delay,
        )

    def _overtime_costs(
        self,
        features: StateFeatures,
        other_index: NDArray[np.intp],
        other_columns: NDArray[np.intp],
        other_rows: NDArray[np.float64],
        vehicle: int,
        candidate_columns: NDArray[np.intp],
        candidate_rows: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Cost of every vehicle getting home past the horizon end.

        A vehicle already sent to the depot is priced on the direct return leg; any
        other target adds the service time and the leg back from it.
        """
        tau = features.tau
        end_of_horizon = self._end_of_horizon
        baseline = end_of_horizon if tau < end_of_horizon else tau
        depot_column = 0

        def homecoming(
            index: NDArray[np.intp] | int,
            columns: NDArray[np.intp],
            rows: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            minutes = features.vehicle_minutes
            arrival: NDArray[np.float64] = np.where(
                columns == depot_column,
                tau + minutes[index, depot_column],
                tau + minutes[index, columns] + rows[:, depot_column] + self._service_time,
            )
            return arrival

        def overtime(arrival: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.where(
                arrival > end_of_horizon, (arrival - baseline) * self._overtime_cost_factor, 0.0
            )

        others = overtime(homecoming(other_index, other_columns, other_rows))
        candidates = overtime(homecoming(vehicle, candidate_columns, candidate_rows))
        return float(others.sum()) + candidates


def _column_mask(column_count: int, columns: NDArray[np.intp]) -> NDArray[np.bool_]:
    mask = np.zeros(column_count, dtype=np.bool_)
    mask[columns] = True
    return mask
