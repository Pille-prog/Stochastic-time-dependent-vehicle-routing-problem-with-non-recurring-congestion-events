"""Unit tests for the Model termination charges (ticket 03, simulator-correctness).

Both termination methods price every unserved Client against the fixed
**reference clock** ``max(episode_end_minute, tau)`` (B3, ADR-0004), not the
live ``tau`` ticket 12 charged them from (ADR-0001 phase-2 change log,
superseded here): termination closes the outcome, so the price must not
depend on *when* the fleet happened to stop.

The Model is built via ``__new__`` with only the collaborators and attributes
the termination methods touch: they are pure accounting over the State and the
CostLedger.

``TestPassingHorizonOvertimeGuard`` pins B15 (ticket 02, simulator-correctness):
``terminate_state_passing_horizon`` charged ``tau - shift_end_minute``
unconditionally, so a ``shift_end_minute`` that outlived the episode's actual
termination clock priced *negative* overtime — the review's own reproduction
(docs/simulator-review.md, B15) is this exact scenario (tau=1148,
shift_end_minute=1200, -43.33 per vehicle).
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from stdvrp.simulation.cost_ledger import CostLedger
from stdvrp.simulation.model import Model
from stdvrp.simulation.state import State

DELAY_RATE = 1  # the CostLedger's per-minute delay rate
CLIENT, DUE = 7, 500.0
EPISODE_END = 1150.0  # the shipped EMERGENCY_HORIZON default


def make_terminating_model(tau: float, episode_end_minute: float = EPISODE_END) -> Model:
    state = State(1, [CLIENT], 3, 300, 0)
    state.tau_episode = tau
    model = Model.__new__(Model)
    model.state = state
    model.time_windows = {CLIENT: (400.0, DUE)}
    model.depot = 0
    model.episode_end_minute = episode_end_minute
    model.costs = CostLedger()
    model.costs.start_transition()
    return model


class TestAllVehiclesComeBack:
    def test_unserved_client_is_charged_from_the_reference_clock_not_tau(self):
        """ADR-0004: priced against ``episode_end_minute``, not the live ``tau``."""
        model = make_terminating_model(tau=900.0, episode_end_minute=EPISODE_END)
        model.terminate_state_if_all_vehicles_come_back()
        assert model.costs.delay_cost == (EPISODE_END - DUE) * DELAY_RATE
        assert model.state.terminal

    def test_a_client_whose_window_is_still_open_is_still_charged(self):
        """B3: the fix. The fleet is home and this Client will never be served,
        however its window reads at the actual clock the episode stopped at —
        termination is a different event from the live in-episode check."""
        model = make_terminating_model(tau=450.0, episode_end_minute=EPISODE_END)
        model.terminate_state_if_all_vehicles_come_back()
        assert model.costs.delay_cost == (EPISODE_END - DUE) * DELAY_RATE

    def test_no_charge_when_the_reference_clock_does_not_reach_due(self):
        model = make_terminating_model(tau=450.0, episode_end_minute=DUE)
        model.terminate_state_if_all_vehicles_come_back()
        assert model.costs.delay_cost == 0

    def test_no_overtime_is_charged_when_the_fleet_is_home(self):
        model = make_terminating_model(tau=900.0)
        model.terminate_state_if_all_vehicles_come_back()
        assert model.costs.overtime_cost == 0

    def test_charge_matches_the_passing_horizon_sibling(self):
        all_back = make_terminating_model(tau=900.0)
        all_back.terminate_state_if_all_vehicles_come_back()

        passing = make_terminating_model(tau=900.0)
        passing.shift_end_minute = 780
        passing.terminate_state_passing_horizon()

        assert all_back.costs.delay_cost == passing.costs.delay_cost


class TestPassingHorizonOvertimeGuard:
    def test_no_overtime_when_shift_end_outlives_the_actual_termination_clock(self):
        model = make_terminating_model(tau=1148.0)
        model.state.vehicle_position = [CLIENT]  # still out on the road, not home
        model.shift_end_minute = 1200.0
        model.terminate_state_passing_horizon()
        assert model.costs.overtime_cost == 0

    def test_overtime_is_still_charged_when_shift_end_precedes_the_clock(self):
        model = make_terminating_model(tau=1148.0)
        model.state.vehicle_position = [CLIENT]
        model.shift_end_minute = 780.0
        model.terminate_state_passing_horizon()
        assert model.costs.overtime_cost == (1148.0 - 780.0) * (5 / 6)

    def test_no_overtime_when_the_fleet_is_home_regardless_of_shift_end(self):
        model = make_terminating_model(tau=1148.0)
        model.shift_end_minute = 780.0  # depot, from make_terminating_model's default
        model.terminate_state_passing_horizon()
        assert model.costs.overtime_cost == 0


@st.composite
def _reachable_termination_scenarios(draw: st.DrawFn):
    """``due``, ``episode_end_minute`` and several clocks the Model could plausibly stop at.

    ``tau`` is drawn no higher than ``episode_end_minute``: both termination
    call sites only ever fire with ``tau_episode <= episode_end_minute`` — the
    Model force-terminates there (ticket 02, B12) — so that is this
    invariant's actual domain, not the unreachable ``tau > episode_end_minute``
    the raw ``max()`` formula would also handle defensively.
    """
    due = draw(st.floats(300.0, 780.0, allow_nan=False))
    episode_end_minute = draw(st.floats(780.0, 1150.0, allow_nan=False))
    taus = draw(
        st.lists(st.floats(300.0, episode_end_minute, allow_nan=False), min_size=2, max_size=5)
    )
    return due, episode_end_minute, taus


class TestReferenceClockInvariant:
    """B3 (ADR-0004): the termination charge is a function of ``due`` and
    config only — never of ``tau`` — over the clock's entire reachable range.
    """

    @settings(max_examples=50, deadline=None)
    @given(scenario=_reachable_termination_scenarios())
    def test_same_unserved_set_prices_the_same_at_every_reachable_clock(
        self, scenario: tuple[float, float, list[float]]
    ) -> None:
        due, episode_end_minute, taus = scenario
        prices = set()
        for tau in taus:
            model = make_terminating_model(tau=tau, episode_end_minute=episode_end_minute)
            model.time_windows = {CLIENT: (400.0, due)}
            model.terminate_state_if_all_vehicles_come_back()
            prices.add(model.costs.delay_cost)
        assert len(prices) == 1, (
            f"the same unserved Client priced differently across clocks: {prices}"
        )
