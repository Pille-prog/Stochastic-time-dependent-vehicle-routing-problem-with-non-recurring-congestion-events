"""Unit tests for the CostLedger (ticket 09, simulation-performance).

The ledger is where the Model's cost accounting moved. These pin the three
things that are easy to "simplify" back into a bug: the transition/total split
(and the one transition the Model never commits), the preserved quirks in what
each charge counts, and the float accumulation order the Tier-1 bit-exact gate
depends on.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from stdvrp.simulation.cost_ledger import (
    ABORT_PENALTY,
    ABORT_PENALTY_PER_SERVED_CLIENT,
    CostLedger,
)

EARLINESS_RATE = 0.1
OVERTIME_RATE = 5 / 6


@pytest.fixture
def ledger() -> CostLedger:
    ledger = CostLedger()
    ledger.start_transition()
    return ledger


class TestTheTransitionInFlight:
    def test_charges_land_on_the_transition_before_it_is_committed(self, ledger: CostLedger):
        ledger.charge_distance(10.0)
        assert ledger.transition_cost == 10.0
        assert ledger.total_cost == 0

    def test_committing_folds_the_transition_into_the_total(self, ledger: CostLedger):
        ledger.charge_distance(10.0)
        ledger.commit_transition()
        assert ledger.total_cost == 10.0

    def test_starting_a_transition_zeroes_it_without_touching_the_total(self, ledger: CostLedger):
        ledger.charge_distance(10.0)
        ledger.commit_transition()

        ledger.start_transition()
        assert ledger.transition_cost == 0
        assert ledger.total_cost == 10.0

    def test_an_uncommitted_transition_is_priced_but_never_totalled(self, ledger: CostLedger):
        """The Model's one non-committing transition end (a Client already served)."""
        ledger.charge_distance(10.0)
        ledger.start_transition()
        ledger.charge_distance(3.0)
        ledger.commit_transition()

        assert ledger.total_cost == 3.0
        assert ledger.distance_cost == 13.0, "the component total keeps both charges"


class TestPerEventCharges:
    def test_distance_is_charged_at_its_own_rate(self, ledger: CostLedger):
        ledger.charge_distance(7.5)
        assert ledger.distance_cost == 7.5
        assert ledger.transition_cost == 7.5

    def test_earliness_is_charged_at_its_own_rate_and_counts_the_client(self, ledger: CostLedger):
        ledger.charge_earliness(20.0)
        assert ledger.earliness_cost == 20.0 * EARLINESS_RATE
        assert ledger.transition_cost == 20.0 * EARLINESS_RATE
        assert ledger.early_clients == 1

    def test_lateness_counts_the_client(self, ledger: CostLedger):
        ledger.charge_lateness(20.0)
        assert ledger.delay_cost == 20.0
        assert ledger.late_clients == 1

    def test_vehicle_overtime_counts_the_vehicle(self, ledger: CostLedger):
        ledger.charge_vehicle_overtime(12.0)
        assert ledger.overtime_cost == 12.0 * OVERTIME_RATE
        assert ledger.overtime_vehicles == 1


class TestTerminationCharges:
    def test_unserved_delays_are_charged_and_counted_as_unserved_not_late(self, ledger: CostLedger):
        """Ticket 03 (B14): unserved Clients get their own counter.

        Never ``late_clients`` — "served late" and "never served" are
        different outcomes (preserved from before ticket 03).
        """
        ledger.charge_unserved_delays([30.0, 20.0])
        assert ledger.delay_cost == 50.0
        assert ledger.transition_cost == 50.0
        assert ledger.late_clients == 0
        assert ledger.unserved_clients == 2

    def test_a_zero_priced_unserved_client_is_not_counted(self, ledger: CostLedger):
        """The reference-clock formula floors some unserved Clients' price at 0
        (``Model._charge_unserved_delays``); those must not inflate
        ``unserved_clients`` past what ``delay_cost`` actually moved by (B14's
        "money charged > 0 iff its counter > 0" invariant).
        """
        ledger.charge_unserved_delays([0.0, 5.0])
        assert ledger.delay_cost == 5.0
        assert ledger.unserved_clients == 1

    def test_unserved_delays_sum_before_touching_the_total(self, ledger: CostLedger):
        """Bit-exactness: ``total + (a + b)``, never ``(total + a) + b``."""
        ledger.charge_lateness(0.1)
        before = ledger.delay_cost
        late = [0.2, 0.4, 0.8]

        ledger.charge_unserved_delays(late)

        assert ledger.delay_cost == before + (0.2 + 0.4 + 0.8)

    def test_fleet_overtime_is_charged_and_counts_its_vehicles(self, ledger: CostLedger):
        """Ticket 03 (B14): retires the no-count quirk into the shared overtime_vehicles."""
        ledger.charge_fleet_overtime(6.0, vehicles=3)
        assert ledger.overtime_cost == 6.0 * OVERTIME_RATE * 3
        assert ledger.overtime_vehicles == 3

    def test_fleet_overtime_with_zero_minutes_counts_no_vehicle(self, ledger: CostLedger):
        """B14's invariant at the boundary: no money moved, so no vehicle is counted."""
        ledger.charge_fleet_overtime(0.0, vehicles=3)
        assert ledger.overtime_cost == 0
        assert ledger.overtime_vehicles == 0

    def test_fleet_overtime_accumulates_by_repeated_addition(self, ledger: CostLedger):
        """Bit-exactness: the legacy summed per vehicle; it did not multiply."""
        minutes, vehicles = 0.1, 7
        expected = 0.0
        for _ in range(vehicles):
            expected += minutes * OVERTIME_RATE

        ledger.charge_fleet_overtime(minutes, vehicles=vehicles)

        assert ledger.overtime_cost == expected

    def test_no_unserved_clients_charges_nothing(self, ledger: CostLedger):
        ledger.charge_unserved_delays([])
        ledger.charge_fleet_overtime(6.0, vehicles=0)
        assert ledger.transition_cost == 0
        assert ledger.unserved_clients == 0
        assert ledger.overtime_vehicles == 0


class TestMoneyCounterInvariant:
    """B14: money charged > 0 iff its counter > 0, for the two collection charges.

    The per-event siblings (``charge_lateness``, ``charge_earliness``,
    ``charge_vehicle_overtime``) are unconditional-count-by-one and are only
    ever called by the Model with a strictly positive amount, so they hold
    this invariant by construction and are out of this ticket's scope; these
    two are the ones ticket 03 changed to hold it unconditionally, regardless
    of what the caller passes in.
    """

    @settings(max_examples=100, deadline=None)
    @given(minutes_late=st.lists(st.floats(0.0, 500.0, allow_nan=False), max_size=6))
    def test_unserved_delays(self, minutes_late: list[float]) -> None:
        # A fresh CostLedger per example, not the ``ledger`` fixture: Hypothesis
        # re-runs this function body per example but pytest builds a
        # function-scoped fixture only once, which would silently accumulate
        # charges across examples instead of isolating each one.
        ledger = CostLedger()
        ledger.start_transition()
        ledger.charge_unserved_delays(minutes_late)
        assert (ledger.delay_cost > 0) == (ledger.unserved_clients > 0)

    @settings(max_examples=100, deadline=None)
    @given(
        minutes_over=st.floats(0.0, 500.0, allow_nan=False),
        vehicles=st.integers(0, 6),
    )
    def test_fleet_overtime(self, minutes_over: float, vehicles: int) -> None:
        ledger = CostLedger()
        ledger.start_transition()
        ledger.charge_fleet_overtime(minutes_over, vehicles=vehicles)
        assert (ledger.overtime_cost > 0) == (ledger.overtime_vehicles > 0)


class TestAbortPenalty:
    def test_the_penalty_is_rebated_per_served_client(self, ledger: CostLedger):
        ledger.charge_abort_penalty(served_clients=4)
        assert ledger.transition_cost == ABORT_PENALTY - 4 * ABORT_PENALTY_PER_SERVED_CLIENT

    def test_the_penalty_reaches_no_component_total(self, ledger: CostLedger):
        """Preserved quirk: an aborted Episode's components do not sum to its total."""
        ledger.charge_abort_penalty(served_clients=0)
        ledger.commit_transition()

        components = (
            ledger.distance_cost + ledger.delay_cost + ledger.earliness_cost + ledger.overtime_cost
        )
        assert components == 0
        assert ledger.total_cost == ABORT_PENALTY
