"""Model: the sequential decision model owning the transition function (Powell).

Phase-1 structural port (ADR-0001) of the legacy ``model`` class, restricted to the
path ``main()`` executes: ``transition_function`` with all its callees, the
evaluation Episode loop (``create_monte_carlo_episode_test``) and the training
Episode loop (``create_monte_carlo_episode_train``, ticket 08). Deliberately
concrete — no interface (ADR-0002); the CongestionGenerator seam is injected
instead of the legacy's hardcoded event method.

**What lives here, and what does not** (simulation-performance ticket 09). This
class advances simulated time: it decides *which event happens next* and what
that does to the world. Three collaborators own what it used to inline —

- :class:`CostLedger` (``self.costs``) — every charge and every total, so the
  transition function reads as event dispatch rather than bookkeeping;
- :class:`EpisodeVelocities` (``self.velocities``) — the ``velocity_rng``, the
  per-arc-minute memo and the congestion event book;
- :class:`FleetRoutes` (``self.fleet``) — the six per-vehicle lists (route,
  arrival, departure, ...) as one struct-of-arrays value.

The State keeps what the *Policy* sees (positions, directions, observed
velocities); the fleet keeps what only the simulator does.

Stochastic velocities and congestion events each draw from their own injected
``np.random.Generator`` (``velocity_rng``, ``congestion_rng`` — ticket 13, ADR-0001
phase 2): one per-Episode stream per concern, never shared with each other or with
the Policy's exploration stream. The legacy consumed both from shared global
state in a fixed call order (Phase 1, ADR-0001); that exact interleaving is not
reproducible once the streams are independent, which is the deliberate point —
see the ADR-0001 phase-2 addendum. The per-Episode mutable dicts the legacy kept
on ``DataCalculations`` (``all_arc_velocity``, ``congested_arcs``) live on the
EpisodeVelocities this Model constructs, which is fresh per Episode — the legacy
reset both at every episode boundary, so a fresh Model starts from the identical
state.

Preserved legacy quirks (ADR-0001) owned by this file:

- The hardcoded :data:`CLOCK_CEILING`, ~24 simulated minutes past
  :data:`EMERGENCY_HORIZON` and unreachable in practice — belt-and-braces the
  legacy never needed and this port keeps.
- The shifted-clock gates in :meth:`Model._congestion_epoch_due` and
  :meth:`Model._epoch_ends_the_transition`.
- A decision raised *during* rerouting is discarded (see
  :meth:`Model.transition_function`), and one transition end does not commit its
  cost (see :meth:`Model.vehicle_reaches_node`).

The cost-side quirks — what each charge counts, and the abort penalty that
reaches no component total — moved to :mod:`stdvrp.simulation.cost_ledger` with
the code that implements them; the sampling-side ones to
:mod:`stdvrp.simulation.episode_velocities`.

Phase-2 deliberate fixes (ticket 12, ADR-0001 change log):

- Both termination paths charge late Clients from the actual clock (``tau -
  due``); the legacy hardcoded ``1150 - due`` in the all-vehicles-back path.
- Training episodes keep their real ``distance_cost`` (the legacy zeroed the
  accumulator every step; reporting only).

Ticket 02 fixes (simulator-correctness, B12/B15/B16 — spec.md decision 5): the
config field the legacy called ``horizon_end_minute`` actually meant two
different clocks, so it now names both, validated ``shift_end_minute <=
episode_end_minute``:

- ``shift_end_minute`` (``self.shift_end_minute``, formerly the constructor's
  ``horizon_end_minute``) is the vehicles' shift end — overtime accrues past
  it, exactly as before, same shipped value (780).
- ``episode_end_minute`` (``self.episode_end_minute``) is the episode's hard
  stop — what :data:`EMERGENCY_HORIZON` used to hardcode independent of any
  config; :meth:`_decision_epoch_begins` now reads the configured attribute
  instead, same shipped value (1150).
- :meth:`terminate_state_passing_horizon` gained the ``tau > shift_end_minute``
  guard :meth:`_vehicle_parks_at_depot` already had (B15: unguarded, a
  ``shift_end_minute`` outliving the episode priced negative overtime).
- :meth:`_congestion_epoch_due` dropped the ``/ 60`` float division that
  silently degraded the cadence for non-dyadic ``max_congestion_duration``
  values (B16).

Ticket 03 fixes (simulator-correctness, B3/B14 — spec.md decision 6,
ADR-0004): a Client the Episode ends without serving is priced against a
fixed **reference clock**, ``max(episode_end_minute, tau_episode)``, not the
live ``tau_episode`` ticket 12 charged both termination paths from. The two
are the same event handled two different ways: ``tau_episode`` is correct for
the live, in-episode lateness check (a pending Client's window might still be
met), but termination closes the outcome, so the price must not depend on
*when* the fleet happened to stop — a defect ticket 01 measured moving the
same 19 abandoned Clients' cost by four orders of magnitude across otherwise
identical seeds. Because both termination call sites only ever fire with
``tau_episode <= episode_end_minute`` (the Model force-terminates at
``episode_end_minute``, ticket 02, B12), the ``max`` degenerates in practice to
the constant ``episode_end_minute``, which is the whole point: the charge
becomes a pure function of ``due`` and config, comparable across episodes
regardless of which clock each one actually stopped at. See
:meth:`_charge_unserved_delays`.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from stdvrp.congestion import CongestionGenerator
from stdvrp.network.shortest_path_cache import ShortestPathCache
from stdvrp.policies.base import Policy, TrainablePolicy
from stdvrp.policies.monte_carlo import TimeWindows
from stdvrp.simulation.cost_ledger import CostLedger
from stdvrp.simulation.episode_velocities import EpisodeVelocities
from stdvrp.simulation.fleet_routes import PARKED, FleetRoutes
from stdvrp.simulation.state import State, TrainingSnapshot, is_parked_at_depot
from stdvrp.traffic.travel_time_model import TravelTimeModel

#: The clock at which the Episode is force-terminated. Formerly hardcoded here
#: and independent of any config (B12); ticket 02 (simulator-correctness) makes
#: it configurable as :class:`~stdvrp.config.ExperimentConfig`'s
#: ``episode_end_minute``, threaded to the Model as ``self.episode_end_minute``
#: — this module-level constant is no longer read by :class:`Model` and exists
#: only as the value every shipped config still sets it to, for callers (e.g.
#: ``scripts/measurement_bench.py``) that want that default without loading one.
EMERGENCY_HORIZON = 1150

#: Preserved legacy quirk (ADR-0001): the clock past which no vehicle may still
#: be moving. Every Episode is terminated at :data:`EMERGENCY_HORIZON`, ~24
#: simulated minutes earlier, so the checks against this constant are the
#: legacy's belt-and-braces and are unreachable in practice.
CLOCK_CEILING = 1198

#: Decision epochs are two simulated minutes apart (legacy
#: ``tau_multiplicator_difference``).
DECISION_EPOCH_MINUTES = 2

#: How long a Client is serviced for, once a vehicle reaches it.
SERVICE_MINUTES = 5


class Model:
    """Advances the State given a decision and exogenous velocities and congestion.

    Naming convention: the methods that port a named legacy one — the four
    ``vehicle_reaches_*`` / ``terminate_state_*`` events ADR-0001's change log
    refers to, plus the clock and arc primitives they are built from — keep
    public names, because they are the behavior surface the ADRs, the tickets
    and the invariant suite talk about. The transition loop's own scaffolding,
    which is new in ticket 09, is private.
    """

    def __init__(
        self,
        state: State,
        policy: Policy,
        travel_time_model: TravelTimeModel,
        shortest_path_cache: ShortestPathCache,
        time_windows: TimeWindows,
        number_vehicles: int,
        horizon_start_minute: int,
        shift_end_minute: int,
        episode_end_minute: int,
        depot: int,
        congestion_generator: CongestionGenerator,
        max_congestion_duration: int,
        velocity_rng: np.random.Generator,
        congestion_rng: np.random.Generator,
    ) -> None:
        self.state = state
        self.policy = policy
        self.shortest_path_cache = shortest_path_cache
        self.time_windows = time_windows
        self.number_vehicles = number_vehicles
        self.depot = depot
        self.congestion_generator = congestion_generator
        # Per-Episode stochastic stream for congestion (ticket 13, ADR-0001
        # phase 2); the velocity stream is the EpisodeVelocities' own.
        self.congestion_rng = congestion_rng

        self.costs = CostLedger()
        self.velocities = EpisodeVelocities(travel_time_model, velocity_rng)
        self.fleet = FleetRoutes(number_vehicles, horizon_start_minute)

        # Congestion epoch cadence, in minutes (ticket 02, simulator-correctness,
        # B16: kept as the raw int, not the legacy's float ``hours_max_duration``
        # ratio — see :meth:`_congestion_epoch_due`).
        self.max_congestion_duration = max_congestion_duration

        # The clock at which the Policy is next asked for a decision.
        self.next_decision_tau: float = horizon_start_minute + DECISION_EPOCH_MINUTES

        # End of the vehicles' shift; anything past it is overtime (legacy ``work_time``).
        self.shift_end_minute = shift_end_minute

        # The clock at which the Episode is force-terminated (ticket 02,
        # simulator-correctness, B12): configurable, replacing the module-level
        # hardcoded :data:`EMERGENCY_HORIZON` every shipped config still sets
        # this to.
        self.episode_end_minute = episode_end_minute

        self.visited_clients: list[float] = []
        self.total_state_counter = 0

        # False while the clock is still being advanced; True once the Policy is
        # owed another decision. Set from deep inside the callees, which is why
        # it is state rather than a ``break`` — see :meth:`transition_function`.
        self._transition_ended = False

    # --- Episode runner ---------------------------------------------------------

    def run_evaluation_episode(self) -> None:
        """Ports ``create_monte_carlo_episode_test``: greedy decisions until terminal."""
        while not self.state.terminal:
            action = self.policy.decide(self.state)
            self.transition_function(action)
            self.total_state_counter += 1

        self.velocities.release()

    def run_training_episode(self) -> None:
        """Ports ``create_monte_carlo_episode_train``: ε-greedy Episode, then one W update.

        Snapshots the State and decision before every transition, replays them
        through :meth:`~stdvrp.policies.base.TrainablePolicy.learn` when the
        Episode terminates. Calls through the ``TrainablePolicy`` protocol
        (ticket 02) — any Policy that satisfies it works here, not only
        ``MonteCarloPolicy``, which this file no longer names.
        """
        policy = cast(TrainablePolicy, self.policy)

        self.episode_states: list[TrainingSnapshot] = []
        self.episode_actions: list[list[int]] = []
        self.episode_rewards: list[float] = [0]
        while not self.state.terminal:
            action = policy.decide_train(self.state)
            # Snapshot BEFORE the transition — the weight update replays these epochs.
            # ``self.state`` is mutated in place for the rest of the Episode, and
            # ``policy.action`` (aliased by ``action``) is mutated on every later
            # decision, so both need a real copy here, not a reference.
            self.episode_states.append(TrainingSnapshot.capture(self.state))
            self.episode_actions.append(list(action))
            reward = self.transition_function(action)
            self.episode_rewards.append(reward)
            self.total_state_counter += 1

        policy.learn(self.episode_states, self.episode_actions, self.episode_rewards)
        self.velocities.release()

    # --- Transition function: the event dispatch loop ----------------------------

    def transition_function(self, action: list[int]) -> float:
        """Advance simulated time until the next decision is needed; return its cost.

        Each pass around the loop asks what happens soonest — a congestion event
        expiring, a vehicle arriving somewhere, or the next decision epoch
        starting — and dispatches to the handler for it. The handlers advance the
        clock, charge the ledger and, when the Policy is owed a decision, end the
        transition.

        Preserved legacy quirk (ADR-0001): the reset below is *after* the
        rerouting, so a termination raised while rerouting (only reachable past
        :data:`CLOCK_CEILING`, and every Episode ends at
        :data:`EMERGENCY_HORIZON` before that) is discarded and the loop runs
        anyway.
        """
        self.costs.start_transition()

        self._reroute_for(action)
        self._transition_ended = False

        while not self._transition_ended:
            expiry = self._next_congestion_expiry()
            vehicle, arrival = self.fleet.earliest_arrival()

            if min(self.next_decision_tau, arrival, expiry) == expiry:
                self._congestion_expires(expiry)
            elif self._every_vehicle_home_and_no_clients_left():
                self._episode_completes()
            elif self.fleet.all_parked():
                self.terminate_state_if_all_vehicles_come_back()
            elif self.next_decision_tau > arrival:
                self._vehicle_arrives(vehicle, arrival, action)
            else:
                self._decision_epoch_begins()

        return self.costs.transition_cost

    def _next_congestion_expiry(self) -> float:
        """The soonest congestion expiry a still-travelling vehicle would run into.

        Only expiries strictly between now and the vehicle's arrival matter: an
        event that lifts while the vehicle is mid-arc changes the arrival it is
        currently travelling towards, and nothing else does.
        """
        soonest = float("inf")
        if not self.velocities.any_congestion:
            return soonest

        tau_episode = self.state.tau_episode
        fleet = self.fleet
        congestion_expiry = self.velocities.congestion_expiry
        for vehicle in range(self.number_vehicles):
            arrival = fleet.arrival_tau[vehicle]
            if arrival == PARKED:
                continue
            end = congestion_expiry(fleet.current_arc(vehicle))
            if end is not None and tau_episode < end < arrival and end < soonest:
                soonest = end
        return soonest

    def _every_vehicle_home_and_no_clients_left(self) -> bool:
        """Ticket 04 (B1b, ADR-0005): "home" means genuinely parked, not merely
        last seen at the depot — a vehicle mid-arc past the depot as an
        interior waypoint must not end the Episode out from under itself."""
        state = self.state
        return (
            all(
                is_parked_at_depot(node, standing, self.depot)
                for node, standing in zip(
                    state.last_node_reached, state.vehicle_standing, strict=True
                )
            )
            and len(state.clients_not_visited) == 0
        )

    # --- The events the loop dispatches to ---------------------------------------

    def _congestion_expires(self, expiry: float) -> None:
        """A congestion lifts before anything else happens: re-sample every arc."""
        self.advance_fleet_to(expiry)
        for vehicle in range(self.number_vehicles):
            self.resample_arc(vehicle)

    def _episode_completes(self) -> None:
        """Every vehicle is home and every Client is served: the Episode is over."""
        self.state.terminal = True
        self._transition_ended = True
        self.advance_fleet_to(self.state.tau_episode)
        self.costs.commit_transition()

    def _vehicle_arrives(self, vehicle: int, arrival: float, action: list[int]) -> None:
        """The next thing to happen is a vehicle reaching the node it travels to."""
        fleet = self.fleet
        if fleet.departure_tau[vehicle] == arrival and fleet.nodes_left(vehicle) <= 2:
            self._service_completes(vehicle, arrival)
            return

        next_node = fleet.next_node(vehicle)

        if next_node == self.depot and fleet.nodes_left(vehicle) == 2:
            self._vehicle_parks_at_depot(vehicle, arrival)
        elif next_node in self.state.clients_not_visited and action[vehicle] == next_node:
            self.vehicle_reaches_client(vehicle, arrival)
            self._transition_ended = True
            self.costs.commit_transition()
        else:
            self.vehicle_reaches_node(arrival, vehicle)

    def _service_completes(self, vehicle: int, arrival: float) -> None:
        """A vehicle finishes servicing a Client and is free to be sent somewhere."""
        self.state.vehicle_completing_service[vehicle] = 0
        self.advance_fleet_to(arrival)
        self.fleet.settle_at_node(vehicle, self.state.tau_episode)
        self.state.tau_episode = arrival
        self._transition_ended = True
        self.costs.commit_transition()

    def _vehicle_parks_at_depot(self, vehicle: int, arrival: float) -> None:
        """A vehicle reaches the depot for good: charge its overtime and retire it."""
        self.advance_fleet_to(arrival)
        self.state.last_node_reached[vehicle] = self.depot
        self.state.vehicle_standing[vehicle] = True

        if self.state.tau_episode > self.shift_end_minute and self.fleet.is_travelling(vehicle):
            self.costs.charge_vehicle_overtime(self.state.tau_episode - self.shift_end_minute)

        self.fleet.park(vehicle)
        self.costs.commit_transition()
        self._transition_ended = True

    def _decision_epoch_begins(self) -> None:
        """Nothing happens before the next decision epoch: roll the clock to it."""
        if self.next_decision_tau >= CLOCK_CEILING:
            self._abort_at_clock_ceiling()
            return

        self.advance_fleet_to(self.next_decision_tau)

        if self._congestion_epoch_due():
            self.congestion_generator.generate(
                self.state.tau_episode, self.velocities.congested_arcs, self.congestion_rng
            )

        for vehicle in range(self.number_vehicles):
            self.resample_arc(vehicle)

        self.next_decision_tau += DECISION_EPOCH_MINUTES

        # Phase-2 fix (ticket 12, ADR-0001 change log): the epoch-end gate below
        # always fires at the emergency horizon, so the legacy fell through it
        # after terminating and added the same transition cost a second time.
        if self.next_decision_tau >= self.episode_end_minute:
            self.terminate_state_passing_horizon()
        elif self._epoch_ends_the_transition():
            self._transition_ended = True
            self.costs.commit_transition()

    def _abort_at_clock_ceiling(self) -> None:
        """Unreachable in practice: see :data:`CLOCK_CEILING`."""
        self.state.terminal = True
        self._transition_ended = True
        self.costs.charge_abort_penalty(len(self.visited_clients))
        self.costs.commit_transition()

    def _congestion_epoch_due(self) -> bool:
        """Congestion is rolled every ``max_congestion_duration`` minutes.

        Fixed (ticket 02, simulator-correctness, B16): the shipped gate divided
        the shifted clock by 60 and compared it against
        ``max_congestion_duration / 60`` — a float-equality test that silently
        degenerates whenever that quotient is not exactly representable in
        binary (``max_congestion_duration in (50, 70, 200)``, measured), rolling
        congestion far less often than the calibrated event probabilities
        assume. ``next_decision_tau`` starts at ``horizon_start_minute +
        DECISION_EPOCH_MINUTES`` (both ints) and is incremented by the int
        ``DECISION_EPOCH_MINUTES``, so it is always integer-valued (verified,
        ticket 01's ``measurement_bench.py``) — plain integer-safe modulo
        against the raw ``max_congestion_duration`` answers the same question
        without dividing. The shift stays ``+ 180 - 2`` rather than folded to
        ``+ 178``: that sub-expression is unchanged from the legacy (it rounds
        twice, and the Tier-1 gate is bit-exact), only the division it used to
        feed is gone.
        """
        return (self.state.tau_episode + 180 - 2) % self.max_congestion_duration == 0

    def _epoch_ends_the_transition(self) -> bool:
        """Preserved legacy quirk (ADR-0001): most decision epochs decide nothing.

        The transition only ends — handing the Policy a fresh decision — on
        epochs where the same shifted clock as :meth:`_congestion_epoch_due` is a
        multiple of 6, i.e. every third one.
        """
        return (self.state.tau_episode + 180 - 2) % 6 == 0

    # --- Routing ------------------------------------------------------------------

    def _reroute_for(self, action: list[int]) -> None:
        """Ports ``calculate_action_route``: reroute vehicles whose decision changed."""
        fleet = self.fleet
        for vehicle in range(len(action)):
            # Still serving a Client: only refresh the velocity bookkeeping.
            if fleet.departure_tau[vehicle] > self.state.tau_episode:
                self.begin_arc(vehicle)

            elif (
                action[vehicle] == self.depot
                and self.state.last_node_reached[vehicle] == self.depot
                and fleet.is_at_node(vehicle, self.state.tau_episode)
            ):
                # Ticket 04 (B1b, ADR-0005) narrowed this from "last node was
                # the depot" to "genuinely parked" via ``is_parked_at_depot``
                # (last node + ``vehicle_standing``) — without it this branch
                # could not tell a vehicle genuinely parked at the depot from
                # one merely last seen there while mid-arc past it (the depot
                # is an interior node on 6.8% of cached shortest paths).
                #
                # Ticket 11 (B20, ADR-0008) widened it again:
                # ``vehicle_standing`` flips to ``False`` the instant
                # ``begin_arc`` launches a vehicle, at the very same
                # ``departure_tau == tau`` that also means zero arc progress
                # — one instant where a vehicle is both "at the node" and
                # "not standing". ``is_parked_at_depot`` missed it and fell
                # through to the mid-arc branch below, which routed
                # depot -> depot and crashed. Positional presence
                # (``FleetRoutes.is_at_node``) is the fact that actually
                # answers "can this vehicle park here", not the standing flag.
                #
                # Not ``FleetRoutes.park``: the legacy leaves ``departure_tau``
                # alone here, unlike the vehicle that arrives at the depot.
                fleet.arrival_tau[vehicle] = PARKED
                fleet.horizon_change_tau[vehicle] = self.state.tau_episode
                # The widened branch is now reachable with ``standing ==
                # False``; downstream ``is_parked_at_depot`` call sites
                # (overtime accounting, termination) need it ``True`` here,
                # exactly as ``_vehicle_parks_at_depot`` already sets it on
                # arrival.
                self.state.vehicle_standing[vehicle] = True

            elif fleet.destination[vehicle] != action[vehicle] and fleet.is_travelling(vehicle):
                last_node_reached = self.state.last_node_reached[vehicle]
                vehicle_destination = action[vehicle]
                if fleet.departure_tau[vehicle] == self.state.tau_episode:
                    # At a node: route straight from the current position.
                    fleet.route[vehicle] = list(
                        self.shortest_path_cache.path_between(
                            last_node_reached, vehicle_destination
                        ).nodes
                    )
                    self.begin_arc(vehicle)
                else:
                    # Mid-arc: finish the current arc, then follow the new route.
                    shortest_path = list(
                        self.shortest_path_cache.path_between(
                            fleet.next_node(vehicle), vehicle_destination
                        ).nodes
                    )
                    shortest_path.insert(0, last_node_reached)
                    fleet.route[vehicle] = shortest_path

                fleet.destination[vehicle] = action[vehicle]

    def begin_arc(self, vehicle: int) -> None:
        """Ports ``create_and_actualize_state_velocity``: start travelling the next arc."""
        fleet = self.fleet
        if self.state.tau_episode > CLOCK_CEILING:
            self.terminate_state_passing_horizon()

        elif fleet.departure_tau[vehicle] > self.state.tau_episode:
            self._hold_for_service(vehicle)

        else:
            # Ticket 04 (ADR-0005): this is the moment the vehicle actually
            # leaves ``last_node_reached`` — the fact the Policy needs to tell
            # "standing there" from "merely passed through it" flips here.
            self.state.vehicle_standing[vehicle] = False

            sampled = self.velocities.sample(
                *fleet.current_arc(vehicle),
                self.state.tau_episode,
            )

            self.state.observed_velocity[vehicle].pop(0)
            self.state.observed_velocity[vehicle].append(sampled.velocity)
            fleet.arrival_tau[vehicle] = self.state.tau_episode + sampled.travel_time
            fleet.departure_tau[vehicle] = self.state.tau_episode
            fleet.horizon_change_tau[vehicle] = self.state.tau_episode

    def _hold_for_service(self, vehicle: int) -> None:
        """A vehicle inside a Client's service time: it goes nowhere until service ends.

        Both :meth:`begin_arc` and :meth:`resample_arc` reach this state, and both
        record the same thing: no velocity observed this epoch, and the next
        "arrival" is the moment service finishes.
        """
        self.state.vehicle_standing[vehicle] = True
        self.state.observed_velocity[vehicle].pop(0)
        self.state.observed_velocity[vehicle].append(0)
        self.fleet.arrival_tau[vehicle] = self.fleet.departure_tau[vehicle]
        self.fleet.horizon_change_tau[vehicle] = self.state.tau_episode

    def resample_arc(self, vehicle: int) -> None:
        """Ports ``time_horizon_actualization``: re-price the arc being travelled.

        Credits the distance covered since the last re-sample at the velocity
        that was in force, draws a new one for the rest of the arc, and moves the
        arrival accordingly.
        """
        fleet = self.fleet
        if fleet.departure_tau[vehicle] > self.state.tau_episode:
            self._hold_for_service(vehicle)

        elif fleet.arrival_tau[vehicle] != PARKED:
            sampled = self.velocities.sample(
                *fleet.current_arc(vehicle),
                self.state.tau_episode,
            )

            time_in_arc = self.state.tau_episode - fleet.horizon_change_tau[vehicle]
            fleet.horizon_change_tau[vehicle] = self.state.tau_episode

            distance_travelled = (
                self.state.observed_velocity[vehicle][self.state.n_observed_velocities - 1]
                * time_in_arc
            )
            fleet.arc_distance_travelled[vehicle] += distance_travelled

            distance_left_to_travel = sampled.length - fleet.arc_distance_travelled[vehicle]
            travel_time = distance_left_to_travel / sampled.velocity

            self.state.observed_velocity[vehicle].pop(0)
            self.state.observed_velocity[vehicle].append(sampled.velocity)
            fleet.arrival_tau[vehicle] = self.state.tau_episode + travel_time

    # --- Arrivals ------------------------------------------------------------------

    def vehicle_reaches_client(self, vehicle: int, arrival: float) -> None:
        """Ports ``vehicle_reaches_client``: serve, cost the time window, start service."""
        client = self.fleet.next_node(vehicle)
        self.visited_clients.append(client)
        # The float path-node id equals the int Client id (legacy parse quirk).
        self.state.clients_not_visited.remove(client)  # type: ignore[arg-type]

        earliness_time_window = self.time_windows[client][0]  # type: ignore[index]
        lateness_time_window = self.time_windows[client][1]  # type: ignore[index]

        self.state.last_node_reached[vehicle] = client
        self.state.vehicle_standing[vehicle] = True
        self.state.clients_arrival[client] = [arrival, vehicle]

        if arrival < earliness_time_window:
            self.costs.charge_earliness(earliness_time_window - arrival)
        elif arrival > lateness_time_window:
            self.costs.charge_lateness(arrival - lateness_time_window)

        self.advance_fleet_to(arrival)

        self.fleet.departure_tau[vehicle] = self.state.tau_episode + SERVICE_MINUTES
        self.state.vehicle_completing_service[vehicle] = 1
        self.fleet.settle_at_node(vehicle, self.state.tau_episode)

    def vehicle_reaches_node(self, arrival: float, vehicle: int) -> None:
        """Ports ``vehicle_reaches_node``: an arrival that is not the vehicle's Client."""
        fleet = self.fleet
        if arrival > CLOCK_CEILING or self.state.tau_episode > CLOCK_CEILING:
            self.terminate_state_passing_horizon()

        elif fleet.nodes_left(vehicle) == 2 and fleet.next_node(vehicle) in self.visited_clients:
            # Arrived at a Client already served by another vehicle: ask for a new decision.
            self.advance_fleet_to(arrival)
            self.state.last_node_reached[vehicle] = fleet.next_node(vehicle)
            self.state.vehicle_standing[vehicle] = True
            fleet.departure_tau[vehicle] = self.state.tau_episode
            fleet.settle_at_node(vehicle, self.state.tau_episode)
            # Preserved legacy quirk (ADR-0001): the only transition end that does
            # not commit its cost. The Policy still learns from it — the reward is
            # the transition cost — but the Episode total never sees it.
            self._transition_ended = True

        else:
            # Ticket 04 (B1b, ADR-0005): this is the "passes through" write the
            # review names — ``last_node_reached`` becomes this node, but
            # ``begin_arc`` right below immediately launches (or holds) the
            # vehicle again, so ``vehicle_standing`` is never set ``True``
            # here; ``begin_arc`` alone decides its post-call value.
            self.state.last_node_reached[vehicle] = fleet.next_node(vehicle)
            fleet.advance(vehicle)
            self.advance_fleet_to(arrival)
            self.begin_arc(vehicle)
            fleet.arc_distance_travelled[vehicle] = 0

    # --- The clock -------------------------------------------------------------------

    def advance_fleet_to(self, minute: float) -> None:
        """Ports ``vehicle_distance_transition_cost``: charge the fleet, move the clock.

        Every vehicle still travelling covers ``its observed velocity x the
        elapsed time`` and is charged for it.

        Every call is a clock advance, so this is also the one place that
        purges the congestion book (B17, ticket 07): whatever event book
        entries lift as of ``minute`` are gone from here on, not just at
        Episode end.
        """
        elapsed = minute - self.state.tau_episode
        observed_velocity = self.state.observed_velocity
        last_slot = self.state.n_observed_velocities - 1
        distance_by_vehicle = self.state.total_vehicle_distance_travelled
        arrival_tau = self.fleet.arrival_tau
        charge_distance = self.costs.charge_distance

        for vehicle in range(self.number_vehicles):
            if arrival_tau[vehicle] != PARKED:
                distance_travelled = observed_velocity[vehicle][last_slot] * elapsed
                distance_by_vehicle[vehicle] += distance_travelled
                charge_distance(distance_travelled)

        self.state.tau_episode = minute
        self.velocities.purge_expired(minute)

    # --- Termination ------------------------------------------------------------------

    def terminate_state_passing_horizon(self) -> None:
        """Ports ``terminate_state_passing_horizon``: charge unserved delays and overtime.

        Fixed (ticket 02, simulator-correctness, B15): guards the overtime
        charge with ``tau > shift_end_minute``, the same guard
        :meth:`_vehicle_parks_at_depot` already has — the legacy charged
        ``tau - shift_end_minute`` unconditionally here, pricing *negative*
        overtime whenever a misconfigured ``shift_end_minute`` outlived the
        episode's actual termination clock. ``episode_end_minute``'s
        validation (``ExperimentConfig.__post_init__``) already makes that
        unreachable under any valid config; this guard is symmetry, not the
        primary fix.
        """
        self.state.terminal = True
        self._transition_ended = True

        self._charge_unserved_delays()
        if self.state.tau_episode > self.shift_end_minute:
            state = self.state
            # Ticket 04 (ADR-0005): "out" is genuinely not-parked-at-the-depot,
            # not merely "last node reached wasn't the depot" — a vehicle
            # mid-arc past the depot as an interior waypoint is still out.
            vehicles_out = sum(
                not is_parked_at_depot(node, standing, self.depot)
                for node, standing in zip(state.last_node_reached, state.vehicle_standing, strict=True)
            )
            self.costs.charge_fleet_overtime(
                self.state.tau_episode - self.shift_end_minute, vehicles=vehicles_out
            )

        self.costs.commit_transition()

    def terminate_state_if_all_vehicles_come_back(self) -> None:
        """Ports ``terminate_state_if_all_vehicles_come_back``.

        Priced like its passing-horizon sibling, from the reference clock
        (ticket 03, B3, ADR-0004) — see :meth:`_charge_unserved_delays`. Unlike
        that sibling this path charges no overtime: every vehicle is home.
        """
        self.state.terminal = True
        self._transition_ended = True

        self._charge_unserved_delays()

        self.costs.commit_transition()

    def _charge_unserved_delays(self) -> None:
        """Charge every Client this Episode ends without having served.

        Priced against the fixed reference clock ``max(episode_end_minute,
        tau_episode)`` (ticket 03, simulator-correctness, B3, ADR-0004), never
        ``tau_episode`` alone: the live in-episode formula (``cost(t) = max(0,
        t - due)``, unchanged, still used while a decision is pending) is
        correct because a pending Client's window might still be met, but a
        Client that reaches termination unserved never will be — its outcome
        is priced at the worst clock the Episode could have reached, not the
        one it happened to stop at, so the price is comparable across
        episodes regardless of *when* each one terminated. Floored at 0, like
        the live formula: a Client whose window closes at or after the
        reference clock owes nothing. Every unserved Client is priced, not
        only the ones already late by the actual clock (B14: some of those
        prices may still floor to 0; :meth:`CostLedger.charge_unserved_delays`
        counts only the ones that do not).
        """
        reference_clock = max(self.episode_end_minute, self.state.tau_episode)
        time_windows = self.time_windows
        self.costs.charge_unserved_delays(
            max(0.0, reference_clock - time_windows[client][1])
            for client in self.state.clients_not_visited
        )
