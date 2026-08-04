"""FeatureExtractor: the Policy's 24-component feature vector as array arithmetic.

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
- ``mean_velocities``, computed by the general-state routine and carried on
  :class:`StateFeatures` for callers other than the general features themselves
  (the realised-congestion feature below is now one).

**Float reassociation (Tier 2).** The four state-action cost sums are re-associated
as "the other vehicles' terms, then the decided vehicle's", and the future-delay
feature multiplies each Client's contribution by its duplicate-append multiplicity
instead of adding it once per duplicate. (The Policy's ``X @ W`` over the batch is a
third such site — a BLAS ``gemv`` rather than 24-term dot products.) The
general-state features stay bit-exact: their only sum is over time-window starts,
which are integers. See the ticket ``## Comments`` for what this measures out to on
the committed fixture.

**B10 fixed (simulator-correctness ticket 06).** The legacy's fourth earliness
bin (``general[13]``) was never assigned, so it was identically zero and
``W[13]`` never trained. :meth:`FeatureExtractor._general_features` now assigns
it as *whatever the first three bins leave uncounted* — not redefined as "a
Client whose window opens at minute 600 or later", because a bin also drops to
zero once ``tau`` passes its own gate (e.g. bin 0 once ``tau >= 400``) while its
Clients are still pending. Bin 3 absorbs those too. This keeps the invariant
the four bins exist to satisfy: they partition pending demand, at every
``tau`` — the four fractions in ``general[10:14]`` always sum to
``len(clients_not_visited) / number_clients``. The bin *boundaries*
(400/500/600) are unchanged; only the bin that was missing is filled in.

**Restored, not preserved (this ticket): three depot-distance features and one
realised-congestion feature.** Comparison against the legacy source
(``Main_Chengdu_Sirve_2_Acciones.py`` — its ``sys.argv`` order, vehicle-ratio
table and congestion formula all match what ``ExperimentConfig`` already
documents as the port's source) found its live ``extract_general_state_features``
carries 16 general features, not 12, and its live ``extract_state_action_features``
carries 8 state-action features, not 7 — this module's 19 was short by exactly
the five below. None are hand-engineering added after the fact: they were
already in the vector the thesis's published numbers were trained against, and
dropping them during the port was an oversight, not a decision.

- **Depot-distance features** (``general[7:10]``): mean/min/max haversine
  distance from the depot to every pending Client, from ``node_coordinates``
  (``TravelTimeModel.node_coordinates`` — already loaded, previously unused
  downstream of the Policy). Ports ``DataCalculations.calculate_distance_metrics_to_depot``,
  including its call-site swap of the ``haversine_distance`` staticmethod's
  argument order — every call passes ``(lat, lon, lat, lon)`` against a
  signature declared ``(lon1, lat1, lon2, lat2)``, so the formula's ``cos(lat)``
  correction term evaluates ``cos(longitude)`` instead. Preserved exactly
  (ADR-0001): this is the arithmetic the historical constants (2100, 12.9, 29)
  were fit against, not a bug to fix here.
- **The realised-congestion feature** (``general[15]``): restores the legacy
  ``vehicles_direction`` field, removed from ``State`` in ticket 04
  (simulator-correctness, ADR-0005) as apparently write-only — see
  ``stdvrp.simulation.state``'s module docstring for where it comes back. The
  feature counts the fraction of the fleet whose recent observed velocity
  (``mean_velocities`` above — an Episode's own vehicles' own readings,
  ADR-0006) falls below a congestion-scaled historical-average threshold for
  the arc each is about to enter next. Both the observability rule and the
  legacy formula are unchanged.
- **The squared future-delay term** (state-action index 6 below): ports the
  legacy's ``norm_future**2``, appended immediately after ``norm_future`` itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from stdvrp.network.episode_geometry import EpisodeGeometry
from stdvrp.simulation.state import is_parked_at_depot

if TYPE_CHECKING:
    from stdvrp.network.shortest_path_cache import ShortestPathCache
    from stdvrp.simulation.state import State, TrainingSnapshot

NodeCoordinates = dict[int, list[float]]
TravelData = dict[tuple[int, int, int], tuple[float, float]]

TimeWindows = dict[int, tuple[int, int]]

GENERAL_STATE_FEATURES = 16
STATE_ACTION_FEATURES = 8
FEATURE_COUNT = GENERAL_STATE_FEATURES + STATE_ACTION_FEATURES

# The legacy's depot-idle cutoff for the closest-vehicle classifier: a vehicle
# sitting at the depot this late is no longer a candidate carrier. Inconsistent
# with the 350 used by candidate selection — preserved, not reconciled (ADR-0001).
_DEPOT_IDLE_CUTOFF = 310

# The realised-congestion general feature (module docstring, "Restored, not
# preserved"): a vehicle is only checked once it has been under way a few
# minutes, and the threshold it is checked against scales the historical mean
# speed by the run's own congestion_upper_bound. Both literal to the legacy.
_CONGESTION_FEATURE_MIN_TAU = 304
_CONGESTION_FEATURE_SATURATION_DEPTH_0_FACTOR = 0.78


@dataclass(frozen=True, slots=True)
class StateFeatures:
    """One State, arranged for the feature arithmetic of a single decision.

    Its headline is :attr:`general`: the sixteen general-state features that open
    every candidate's 24-component vector. The rest are the per-State quantities
    the eight state-action features are computed against — all in the column space
    described in the module docstring — plus :attr:`delayed_clients`, which is not
    a feature at all but the classification the Policy's candidate selection reads.
    """

    general: NDArray[np.float64]
    """The sixteen general-state features (``X_general_state`` in the legacy)."""

    delayed_clients: tuple[tuple[int, ...], ...]
    """Per vehicle, at most two Clients whose due time its closest-Client list breaches."""

    mean_velocities: tuple[float, ...]
    """Computed by the legacy's general-state routine; also feeds ``general[15]``."""

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
        node_coordinates: NodeCoordinates | None = None,
        travel_data: TravelData | None = None,
        shortest_path_cache: ShortestPathCache | None = None,
        congestion_upper_bound: float | None = None,
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
        # Restored features (module docstring, "Restored, not preserved"):
        # depot-distance needs node coordinates, the congestion feature needs
        # the historical per-arc-minute speed table and a next-hop lookup.
        # Optional, not merely defaulted: MonteCarloPolicy (episode.py) always
        # supplies real values -- this Policy's feature vector was always 24
        # wide. TransformerMonteCarloPolicy's own FeatureExtractor never reads
        # ``general`` (only ``delayed_clients``/``active``/``late``, for the
        # shared action set) and has no route to these four sources through
        # its own constructor, so it is not this ticket's job to thread them
        # there; ``None`` here makes both restored features read as zero
        # rather than crash, leaving that Policy's behavior unchanged.
        self._node_coordinates = node_coordinates
        self._travel_data = travel_data
        self._shortest_path_cache = shortest_path_cache
        self._congestion_upper_bound = congestion_upper_bound

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

        positions = state.last_node_reached
        vehicle_minutes = self._geometry.average_minutes_rows(positions)
        counts, delayed = self._classify_closest_clients(
            tau, active, vehicle_minutes, positions, state.vehicle_standing
        )
        mean_velocities = tuple(
            sum(velocities) / len(velocities) for velocities in state.observed_velocity
        )

        return StateFeatures(
            general=self._general_features(state, tau, len(remaining), active, mean_velocities),
            delayed_clients=delayed,
            mean_velocities=mean_velocities,
            tau=tau,
            active=active,
            late=tau > self._due,
            vehicle_minutes=vehicle_minutes,
            vehicle_length=self._geometry.length_rows(positions),
            closest_client_counts=counts,
        )

    def _general_features(
        self,
        state: State | TrainingSnapshot,
        tau: float,
        remaining_count: int,
        active: NDArray[np.bool_],
        mean_velocities: tuple[float, ...],
    ) -> NDArray[np.float64]:
        """Ports the live ``extract_general_state_features`` (16 features), bit-exactly.

        Most of this is a pure function of ``tau``/``remaining_count``/``active``
        and vectorizes cleanly (the only sum is over time-window starts — integers,
        exactly representable, so reordering it is not observable). The restored
        congestion feature (``general[15]``) is inherently a small per-vehicle
        loop reading ``state`` directly, matching ``_classify_shortest_distance_clients``'s
        own precedent for staying scalar at fleet-sized N.
        """
        clients_left = remaining_count / 150

        if clients_left != 0:
            time_left = (self._episode_end_minute - tau) / (850)
            time = (tau - 300) / 850
        else:
            time_left = 0.0
            time = 0.0

        earliness = self._earliness[active]

        depot_distance_total, depot_distance_min, depot_distance_max = (
            self._depot_distance_features(active, remaining_count)
        )

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
        general[7] = depot_distance_total
        general[8] = depot_distance_min
        general[9] = depot_distance_max
        general[10:14] = [count / self._number_clients for count in counts_earliness]
        general[14] = mean_earliness_diff
        general[15] = self._congestion_signal(state, tau, mean_velocities)
        return general

    def _depot_distance_features(
        self, active: NDArray[np.bool_], remaining_count: int
    ) -> tuple[float, float, float]:
        """Ports ``calculate_distance_metrics_to_depot`` (module docstring, "Restored").

        Legacy returns all-zero for 0 or 1 pending Clients rather than a
        degenerate mean/min/max over one point — preserved, not just avoided.
        """
        if remaining_count <= 1 or self._node_coordinates is None:
            return 0.0, 0.0, 0.0
        depot_lat, depot_lon = self._node_coordinates[self._depot]
        distances = [
            _legacy_haversine_km(depot_lat, depot_lon, *self._node_coordinates[node])
            for node in np.asarray(self._column_nodes)[active]
        ]
        mean_distance = sum(distances) / len(distances)
        total_distance_normalized = (mean_distance * remaining_count) / 2100.0
        return total_distance_normalized, min(distances) / 12.9, max(distances) / 29.0

    def _congestion_signal(
        self, state: State | TrainingSnapshot, tau: float, mean_velocities: tuple[float, ...]
    ) -> float:
        """Ports the legacy's realised-congestion general feature (module docstring).

        A vehicle counts as "congested" when its own recent observed velocity
        (``mean_velocities`` — an Episode's own readings, ADR-0006) falls short
        of the historical mean speed for the arc it is about to enter, scaled by
        this run's ``congestion_upper_bound`` — the same scaling
        ``congestion/generator.py`` uses for its own depth-0 propagation factor
        (B8, docs/simulator-review.md), hence dividing by that factor's 0.78
        rather than an independent constant.
        """
        if (
            tau <= _CONGESTION_FEATURE_MIN_TAU
            or self._shortest_path_cache is None
            or self._travel_data is None
            or self._congestion_upper_bound is None
        ):
            return 0.0
        destination = state.vehicle_destination
        positions = state.last_node_reached
        congested = 0
        state_tau = int(tau)
        if state_tau % 2 == 1:
            state_tau -= 1
        for vehicle in range(self._number_vehicles):
            target = destination[vehicle]
            position = positions[vehicle]
            if target == self._depot or target == position or position == self._depot:
                continue
            next_node = self._shortest_path_cache.path_between(position, target).nodes[1]
            # Node ids are floats-that-hold-integer-values once a vehicle has
            # travelled (state.py's own docstring: "float and int ids hash and
            # compare equal, so lookups work") -- same rationale as the
            # ``time_windows.get(node)`` ignore above.
            historical_speed = self._travel_data[(position, next_node, state_tau)][1]  # type: ignore[index]
            threshold = (
                self._congestion_upper_bound / _CONGESTION_FEATURE_SATURATION_DEPTH_0_FACTOR
            ) * historical_speed
            if mean_velocities[vehicle] < threshold:
                congested += 1
        return congested / self._number_vehicles

    def _classify_closest_clients(
        self,
        tau: float,
        active: NDArray[np.bool_],
        vehicle_minutes: NDArray[np.float64],
        positions: Sequence[float],
        standing: Sequence[bool],
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

        # Ticket 04 (ADR-0005): idle-at-depot requires ``standing`` too — a
        # vehicle mid-arc past the depot (an interior waypoint on 6.8% of
        # cached shortest paths) reads ``position == depot`` but is not idle
        # and must stay eligible.
        eligible = [
            vehicle
            for vehicle, (position, is_standing) in enumerate(zip(positions, standing, strict=True))
            if not (is_parked_at_depot(position, is_standing, self._depot) and tau > _DEPOT_IDLE_CUTOFF)
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
        """One 24-component feature row per candidate action of ``vehicle``.

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
        X[:, 16] = late_count / 13
        # The preserved permanently-zero feature: it is what pads W to 24.
        X[:, 17] = 0.0
        X[:, 18] = total_distance / 100.0
        X[:, 19] = earliness_cost / 60.0
        X[:, 20] = delay_cost / 60.0
        future_delay_normalized = future_delay / 2500.0
        X[:, 21] = future_delay_normalized
        # Restored (module docstring, "Restored, not preserved"): the legacy's
        # ``norm_future**2``, dropped during the port.
        X[:, 22] = future_delay_normalized**2
        X[:, 23] = overtime_cost / 180.0
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


def _legacy_haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Ports ``DataCalculations.haversine_distance`` exactly as its one call site uses it.

    Module docstring, "Restored, not preserved": the legacy staticmethod is
    declared ``haversine_distance(lon1, lat1, lon2, lat2)`` but its only caller
    passes ``(lat, lon, lat, lon)`` — every argument one position off from its
    name. The formula below is that call, arithmetically inlined: not the
    textbook haversine (its ``cos`` correction term ends up evaluating
    longitude, not latitude), and deliberately not corrected — the historical
    normalization constants at the call site were fit against this exact
    arithmetic.
    """
    lat_a, lon_a, lat_b, lon_b = (np.radians(value) for value in (lat_a, lon_a, lat_b, lon_b))
    a = np.sin((lon_b - lon_a) / 2.0) ** 2 + np.cos(lon_a) * np.cos(lon_b) * np.sin(
        (lat_b - lat_a) / 2.0
    ) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(max(1.0 - a, 0.0)))
    return float(6371.0 * c)
