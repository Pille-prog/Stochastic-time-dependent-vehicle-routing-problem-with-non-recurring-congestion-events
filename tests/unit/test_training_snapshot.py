"""Unit tests for ``TrainingSnapshot`` (ticket 06): the deepcopy replacement.

Pins the exact State surface ``MonteCarloPolicy.update_W`` replays — ``tau_episode``,
``clients_not_visited``, ``vehicle_position``, ``observed_velocity`` — and the two
properties that let it stand in for ``copy.deepcopy(state)``: the capture is immune
to later in-place mutation of the source State (the reason the original code needed
a *deep* copy at all — ``State`` mutates these three list fields in place across an
Episode, see its module docstring), and the snapshot itself cannot be mutated.
"""

import dataclasses

import pytest

from stdvrp.simulation.state import State, TrainingSnapshot

DEPOT = 0


def make_state() -> State:
    state = State(
        number_vehicles=2, clients=[1, 2, 3], n_arcs=3, horizon_start_minute=300, depot=DEPOT
    )
    state.tau_episode = 400.0
    state.vehicle_position = [1.0, 2.0]
    state.observed_velocity = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return state


class TestCapture:
    def test_copies_the_four_replayed_fields(self):
        state = make_state()

        snapshot = TrainingSnapshot.capture(state)

        assert snapshot.tau_episode == 400.0
        assert snapshot.clients_not_visited == (1, 2, 3)
        assert snapshot.vehicle_position == (1.0, 2.0)
        assert snapshot.observed_velocity == ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))

    def test_is_immune_to_later_mutation_of_the_source_state(self):
        """The reason the original code deep-copied: State mutates these fields in place."""
        state = make_state()

        snapshot = TrainingSnapshot.capture(state)

        state.clients_not_visited.remove(2)
        state.vehicle_position[0] = 99.0
        state.observed_velocity[0].pop(0)
        state.observed_velocity[0].append(9.9)
        state.tau_episode = 999.0

        assert snapshot.clients_not_visited == (1, 2, 3)
        assert snapshot.vehicle_position == (1.0, 2.0)
        assert snapshot.observed_velocity == ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
        assert snapshot.tau_episode == 400.0

    def test_snapshot_is_frozen(self):
        snapshot = TrainingSnapshot.capture(make_state())

        with pytest.raises(dataclasses.FrozenInstanceError):
            snapshot.tau_episode = 0.0  # type: ignore[misc]
