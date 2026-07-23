"""Unit tests for the Model termination charges (ticket 12, phase-2 fix 5).

``terminate_state_if_all_vehicles_come_back`` charged late unserved Clients
``(1150 - due)`` — the legacy's hardcoded emergency horizon — where its sibling
``terminate_state_passing_horizon`` charges ``(tau - due)``. Both now charge
from the actual clock (ADR-0001 phase-2 change log).

The Model is built via ``__new__`` with only the attributes the termination
methods touch: they are pure accounting over the State and cost constants.
"""

from stdvrp.simulation.model import Model
from stdvrp.simulation.state import State

DELAY_COST = 1  # the Model's hardcoded per-minute delay rate
CLIENT, DUE = 7, 500.0


def make_terminating_model(tau: float) -> Model:
    state = State(1, [CLIENT], 3, 300, 0)
    state.tau_episode = tau
    model = Model.__new__(Model)
    model.state = state
    model.time_windows = {CLIENT: (400.0, DUE)}
    model.delay_cost = DELAY_COST
    model.depot = 0
    model.total_delay_cost = 0
    model.transition_cost = 0
    model.total_cost = 0
    return model


class TestAllVehiclesComeBack:
    def test_late_client_is_charged_from_the_actual_clock(self):
        model = make_terminating_model(tau=900.0)
        model.terminate_state_if_all_vehicles_come_back()
        assert model.total_delay_cost == (900.0 - DUE) * DELAY_COST
        assert model.state.terminal

    def test_client_within_its_window_is_not_charged(self):
        model = make_terminating_model(tau=450.0)
        model.terminate_state_if_all_vehicles_come_back()
        assert model.total_delay_cost == 0

    def test_charge_matches_the_passing_horizon_sibling(self):
        all_back = make_terminating_model(tau=900.0)
        all_back.terminate_state_if_all_vehicles_come_back()

        passing = make_terminating_model(tau=900.0)
        passing.work_time = 780
        passing.total_overtime_cost = 0
        passing.terminate_state_passing_horizon()

        assert all_back.total_delay_cost == passing.total_delay_cost
