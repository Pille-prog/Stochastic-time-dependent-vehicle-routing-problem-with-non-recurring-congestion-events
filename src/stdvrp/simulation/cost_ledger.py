"""CostLedger: every charge the Model makes, and the totals it makes them into.

Simulation-performance ticket 09 (the decomposition ticket): the transition
function used to interleave *what happened* with *what it costs* — eleven
accumulators and four rate constants inlined between the event branches. That
bookkeeping lives here, so the transition function reads as event dispatch and
this file is the one place to look up how an Episode's cost is built.

Concrete by design (ADR-0002): a ledger, not a seam.

**Charging is deliberately not "add one cost at a time".** Two charges sum a
whole collection into a local before touching a total
(:meth:`charge_unserved_delays`, :meth:`charge_fleet_overtime`), because that is
the order the legacy accumulated them in and float addition is not associative:
``total + (a + b)`` and ``(total + a) + b`` differ in the last bits, and the
effort's Tier-1 gate is bit-exact equality (spec,
``.scratch/simulation-performance/``). Both methods say so at their definition;
do not "simplify" them into per-item calls.

Preserved legacy quirk (ADR-0001), documented at the method that owns it: the
emergency-abort penalty (``40000 - 200 x served``) lands in
:attr:`transition_cost` and :attr:`total_cost` but in *no* component total, so
an aborted Episode's components do not sum to its total.

**Retired quirk** (simulator-correctness ticket 03, B14, ADR-0004): the two
collection charges above used to count no Clients and no vehicles, unlike
their per-event siblings — an Episode terminated past the horizon reported 0
late Clients however many it left unserved. Both now count exactly the
Clients/vehicles they charge a strictly positive amount to, which is what
keeps "money charged > 0 iff its counter > 0" true for every component:
:meth:`charge_unserved_delays` into its own :attr:`unserved_clients` (never
folded into :attr:`late_clients` — "served late" and "never served" are
different outcomes) and :meth:`charge_fleet_overtime` into the same
:attr:`overtime_vehicles` :meth:`charge_vehicle_overtime` already uses.
"""

from __future__ import annotations

from collections.abc import Iterable

# Preserved legacy quirk (ADR-0001): the emergency abort's flat penalty and its
# per-served-Client rebate, both hardcoded in the legacy model and unrelated to
# any configured cost rate. See :meth:`CostLedger.charge_abort_penalty`.
ABORT_PENALTY = 40000
ABORT_PENALTY_PER_SERVED_CLIENT = 200


class CostLedger:
    """The Episode's running cost, by component, plus the transition in flight.

    :attr:`transition_cost` accumulates the charges of the transition currently
    being simulated; :meth:`start_transition` zeroes it and
    :meth:`commit_transition` folds it into :attr:`total_cost`. The two are
    separate calls because the Model does not commit every transition it starts
    — see :meth:`commit_transition`.
    """

    def __init__(self) -> None:
        # Cost rates, exactly as the legacy hardcodes them inside the model:
        # currency per minute early / per unit distance / per minute late / per
        # minute past the end of the shift.
        self._earliness_rate = 0.1
        self._distance_rate = 1
        self._delay_rate = 1
        self._overtime_rate = 5 / 6

        self.transition_cost: float = 0
        self.total_cost: float = 0

        self.distance_cost: float = 0
        self.earliness_cost: float = 0
        self.delay_cost: float = 0
        self.overtime_cost: float = 0

        self.early_clients = 0
        self.late_clients = 0
        self.overtime_vehicles = 0
        self.unserved_clients = 0

    # --- The transition in flight -------------------------------------------

    def start_transition(self) -> None:
        """Begin charging a new transition."""
        self.transition_cost = 0

    def commit_transition(self) -> None:
        """Fold the transition in flight into :attr:`total_cost`.

        The Model calls this at every transition end *except one*: the vehicle
        that arrives at a Client another vehicle already served ends its
        transition without committing (preserved legacy quirk, ADR-0001 — see
        ``Model.vehicle_reaches_node``). The reward the Policy learns from is
        :attr:`transition_cost`, so that transition is still priced for
        learning; only the reported Episode total misses it.
        """
        self.total_cost += self.transition_cost

    # --- Per-event charges ---------------------------------------------------

    def charge_distance(self, distance_travelled: float) -> None:
        """Charge one vehicle for the distance it covered."""
        cost = distance_travelled * self._distance_rate
        self.transition_cost += cost
        self.distance_cost += cost

    def charge_earliness(self, minutes_early: float) -> None:
        """Charge one Client served before its time window opened."""
        cost = minutes_early * self._earliness_rate
        self.transition_cost += cost
        self.earliness_cost += cost
        self.early_clients += 1

    def charge_lateness(self, minutes_late: float) -> None:
        """Charge one Client served after its time window closed."""
        cost = minutes_late * self._delay_rate
        self.transition_cost += cost
        self.delay_cost += cost
        self.late_clients += 1

    def charge_vehicle_overtime(self, minutes_over: float) -> None:
        """Charge one vehicle that reached the depot after the end of the shift."""
        cost = self._overtime_rate * minutes_over
        self.transition_cost += cost
        self.overtime_cost += cost
        self.overtime_vehicles += 1

    # --- Termination charges (whole collections at once) ----------------------

    def charge_unserved_delays(self, minutes_late: Iterable[float]) -> None:
        """Charge every Client the Episode ended without serving, in one total.

        Summed into a local and added to the totals once, which is the legacy's
        accumulation order and therefore the bit-exact one (module docstring).
        Ticket 03 (simulator-correctness, B14, ADR-0004): unlike
        :meth:`charge_lateness`, which counts unconditionally because the Model
        never calls it with a non-positive amount, this counts only the
        Clients actually charged a strictly positive amount — the caller
        (:meth:`~stdvrp.simulation.model.Model._charge_unserved_delays`) passes
        every unserved Client through the same reference-clock formula, some of
        which may floor to 0, and those must not inflate :attr:`unserved_clients`
        past what :attr:`delay_cost` actually moved by. Never folded into
        :attr:`late_clients`: a Client that was never served is a different
        outcome from one served late.
        """
        costs: float = 0
        unserved = 0
        for minutes in minutes_late:
            costs += minutes * self._delay_rate
            if minutes > 0:
                unserved += 1

        self.delay_cost += costs
        self.transition_cost += costs
        self.unserved_clients += unserved

    def charge_fleet_overtime(self, minutes_over: float, vehicles: int) -> None:
        """Charge ``vehicles`` that were still out when the Episode terminated.

        Accumulated by repeated addition rather than multiplied by ``vehicles``,
        which is the legacy's order and therefore the bit-exact one (module
        docstring). Ticket 03 (simulator-correctness, B14, ADR-0004): counts
        every vehicle it charges into the same :attr:`overtime_vehicles`
        :meth:`charge_vehicle_overtime` uses, guarded on ``minutes_over > 0`` so
        a ``vehicles`` count passed alongside a zero (or, pre-B15-fix, negative)
        ``minutes_over`` cannot count vehicles :attr:`overtime_cost` charges
        nothing to.
        """
        costs: float = 0
        for _ in range(vehicles):
            costs += minutes_over * self._overtime_rate

        self.overtime_cost += costs
        self.transition_cost += costs
        if minutes_over > 0:
            self.overtime_vehicles += vehicles

    def charge_abort_penalty(self, served_clients: int) -> None:
        """Charge the emergency abort's flat penalty, rebated per served Client.

        Preserved legacy quirk (ADR-0001): the penalty reaches
        :attr:`transition_cost` — and through :meth:`commit_transition`
        :attr:`total_cost` — but no component total, so an aborted Episode's
        components do not sum to its total. It is also unreachable in practice:
        the emergency horizon terminates the Episode ~24 simulated minutes
        before the clock ceiling this fires at (see ``Model``).
        """
        self.transition_cost += ABORT_PENALTY - ABORT_PENALTY_PER_SERVED_CLIENT * served_clients
