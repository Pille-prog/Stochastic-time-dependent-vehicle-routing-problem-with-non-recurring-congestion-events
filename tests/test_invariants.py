"""Simulation invariant suite (Hypothesis): physical and accounting properties.

Asserts, across randomly drawn experiment configs and seeds on the committed mini
fixture and regardless of policy quality:

- the simulated clock never decreases and the Episode terminates by the Horizon
  rules (terminal state, clock inside [horizon start, the legacy 1150 emergency
  horizon]);
- every Client ends the Episode served exactly once or left unserved and
  penalized by exactly one termination charge;
- every sampled travel time and velocity is strictly positive and finite;
- congestion factors stay within the configured bounds (spread arcs divide the
  drawn factor by the depth damping, so the reachable range is
  ``[lower_bound, upper_bound / 0.78]``) and events expire inside
  ``[start + 30, start + max_congestion_duration]``;
- the Episode total cost equals distance + delay + earliness + overtime, up to
  float accumulation order — with one documented legacy exception: when the
  Episode terminates through ``terminate_state_passing_horizon`` inside a
  decision epoch, the epoch-end branch adds the final ``transition_cost`` to
  ``total_cost`` a second time, so the total exceeds the component sum by
  exactly that transition's cost (preserved bug, ADR-0001; ticket 12 triage).

Randomness comes from real gauss draws: the module fixture builds a multi-day
traffic world by deterministically perturbing the fixture day's speeds, giving
every arc a positive speed standard deviation (a single day would leave the
legacy-preserved NaN stds, see ``TravelTimeModel``).

Episodes are the expensive unit (~0.03-0.3 s each), so examples are bounded and
``deadline`` is disabled; ``derandomize=True`` keeps CI deterministic.
"""

from __future__ import annotations

import itertools
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import stdvrp.simulation.episode as episode_module
from stdvrp.congestion import (
    ArcProbabilityCongestionGenerator,
    CongestedArcs,
    CongestionGenerator,
)
from stdvrp.demand import ClientGenerator
from stdvrp.network import ShortestPathCache
from stdvrp.simulation import run_evaluation_episode
from stdvrp.simulation.model import Model
from stdvrp.simulation.state import State
from stdvrp.traffic import CsvDataSource, TravelTimeModel

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "chengdu_mini"

HORIZON_START, HORIZON_END = 300, 780
# The legacy model terminates at hardcoded clock 1150 regardless of the
# configured horizon end (see the Model module docstring).
EMERGENCY_HORIZON = 1150
PERTURBED_DAYS = tuple(range(601, 609))
# Spread arcs divide the drawn factor by the depth damping; reachable depths are
# 0-2 (``_reachable_nodes`` receives ``max_depth - 1``), so 0.78 is the divisor
# of the widest reachable spread.
MAX_SPREAD_DAMPING = 0.78

FIXTURE_DEMAND = dict(
    mean_number_clients=20,
    client_count_stddev=4.0,
    min_number_clients=8,
    client_universe_node_range=(1, 45),
    clients_per_vehicle=4,
    time_window_spread=60,
    horizon_start_minute=HORIZON_START,
    horizon_end_minute=HORIZON_END,
)


# --- Instrumentation: recording subclasses, behavior-identical to the originals ----


class TauRecordingState(State):
    """State that records every assignment of the simulated clock."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.tau_history: list[float] = []
        super().__init__(*args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "tau_episode":
            self.tau_history.append(value)
        super().__setattr__(name, value)


class RecordingModel(Model):
    """Model that records velocity samples and termination calls."""

    last_instance: RecordingModel | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.velocity_observations: list[tuple[float, float, float]] = []
        self.terminate_passing_horizon_calls = 0
        self.terminate_all_back_calls = 0
        super().__init__(*args, **kwargs)
        RecordingModel.last_instance = self

    def create_random_velocity(
        self, node_start: float, node_end: float, tau_episode: float
    ) -> tuple[float, float, float]:
        result = super().create_random_velocity(node_start, node_end, tau_episode)
        self.velocity_observations.append(result)
        return result

    def terminate_state_passing_horizon(self) -> None:
        self.terminate_passing_horizon_calls += 1
        super().terminate_state_passing_horizon()

    def terminate_state_if_all_vehicles_come_back(self) -> None:
        self.terminate_all_back_calls += 1
        super().terminate_state_if_all_vehicles_come_back()


class RecordingCongestionGenerator(CongestionGenerator):
    """Delegates to the live generator, recording every event it writes."""

    def __init__(self, inner: CongestionGenerator) -> None:
        self.inner = inner
        # (minute_start, arc, velocity multiplier, end minute) per written event.
        self.events: list[tuple[float, tuple[float, float], float, float]] = []

    def generate(self, minute_start: float, congested_arcs: CongestedArcs) -> None:
        before = dict(congested_arcs)
        self.inner.generate(minute_start, congested_arcs)
        for arc, value in congested_arcs.items():
            # Values are freshly created lists, so identity detects overwrites.
            if arc not in before or before[arc] is not value:
                self.events.append((minute_start, arc, value[0], value[1]))


# --- Fixtures ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim_world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A fixture-derived world whose perturbed days give positive speed stds."""
    world = tmp_path_factory.mktemp("invariant_world")
    shutil.copyfile(FIXTURE_DIR / "link.csv", world / "link.csv")
    for day in PERTURBED_DAYS:
        rng = np.random.default_rng(day)
        for half in (0, 1):
            speeds = pd.read_csv(FIXTURE_DIR / f"speed[601]_[{half}].csv")
            speeds["Speed"] = speeds["Speed"] * rng.uniform(0.8, 1.2, size=len(speeds))
            speeds.to_csv(world / f"speed[{day}]_[{half}].csv", index=False)

    source = CsvDataSource(world, "link.csv", 601, PERTURBED_DAYS, "all_shortest_paths.csv")
    travel_time_model = TravelTimeModel(
        source.load_road_network(),
        source.load_traffic_history(),
        120,
        horizon_start_minute=HORIZON_START,
    )
    return {
        "travel_time_model": travel_time_model,
        "cache": ShortestPathCache.from_csv(FIXTURE_DIR / "all_shortest_paths.csv"),
        "client_generator": ClientGenerator(**FIXTURE_DEMAND),
        "arcs": list(travel_time_model.event_probability),
    }


@pytest.fixture(scope="module")
def instrumented_episode() -> Any:
    """Swap the recording subclasses into the Episode runner for this module."""
    original_state, original_model = episode_module.State, episode_module.Model
    episode_module.State = TauRecordingState  # type: ignore[misc]
    episode_module.Model = RecordingModel  # type: ignore[misc]
    yield
    episode_module.State = original_state  # type: ignore[misc]
    episode_module.Model = original_model  # type: ignore[misc]


# --- Strategies -------------------------------------------------------------------


@st.composite
def episode_configs(draw: st.DrawFn) -> dict[str, Any]:
    lower = draw(st.floats(0.05, 0.9, allow_nan=False))
    return {
        "seed": draw(st.integers(0, 2**32 - 1)),
        "congestion_lower_bound": lower,
        "congestion_upper_bound": draw(st.floats(lower, 0.95, allow_nan=False)),
        "max_congestion_duration": draw(st.sampled_from([60, 120, 180, 240])),
        "event_probability": draw(st.floats(0.0, 0.35, allow_nan=False)),
        "W": draw(
            st.none()
            | st.lists(
                st.floats(-1, 1, allow_nan=False, allow_infinity=False),
                min_size=19,
                max_size=19,
            ).map(np.array)
        ),
        "vehicle_count": draw(st.none() | st.integers(1, 8)),
    }


# --- The Episode-level invariant property -----------------------------------------


@settings(
    max_examples=25,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(config=episode_configs())
def test_episode_invariants(
    sim_world: dict[str, Any], instrumented_episode: Any, config: dict[str, Any]
) -> None:
    duration = config["max_congestion_duration"]
    generator = RecordingCongestionGenerator(
        ArcProbabilityCongestionGenerator(
            event_probability=dict.fromkeys(sim_world["arcs"], config["event_probability"]),
            successors=sim_world["travel_time_model"].successors,
            congestion_lower_bound=config["congestion_lower_bound"],
            congestion_upper_bound=config["congestion_upper_bound"],
            max_congestion_duration=duration,
        )
    )

    result = run_evaluation_episode(
        seed=config["seed"],
        client_generator=sim_world["client_generator"],
        travel_time_model=sim_world["travel_time_model"],
        shortest_path_cache=sim_world["cache"],
        congestion_generator=generator,
        W=config["W"],
        epsilon=0.05,
        max_congestion_duration=duration,
        horizon_start_minute=HORIZON_START,
        horizon_end_minute=HORIZON_END,
        n_observed_arcs=3,
        vehicle_count=config["vehicle_count"],
    )
    model = RecordingModel.last_instance
    assert isinstance(model, RecordingModel)
    state = model.state
    assert isinstance(state, TauRecordingState)

    # The clock never decreases and the Episode ends by the Horizon rules.
    assert state.terminal
    assert state.tau_history[0] == HORIZON_START
    for earlier, later in itertools.pairwise(state.tau_history):
        assert later >= earlier, f"clock decreased: {earlier} -> {later}"
    assert HORIZON_START <= state.tau_episode <= EMERGENCY_HORIZON

    # Every Client ends served exactly once or unserved and penalized exactly once.
    demand = sim_world["client_generator"].generate(config["seed"])
    original_clients = {client.node for client in demand.clients}
    served = model.visited_clients
    unserved = set(state.clients_not_visited)
    assert len(served) == len(set(served)), "a Client was served more than once"
    assert set(served) | unserved == original_clients
    assert not set(served) & unserved
    assert set(served) == set(state.clients_arrival)
    for arrival_minute, _vehicle in state.clients_arrival.values():
        assert HORIZON_START <= arrival_minute <= state.tau_episode

    termination_charges = model.terminate_passing_horizon_calls + model.terminate_all_back_calls
    assert termination_charges <= 1, "unserved Clients were penalized more than once"
    if unserved:
        assert termination_charges == 1, "unserved Clients were never penalized"

    # Travel times and velocities are strictly positive and finite.
    for travel_time, velocity, length in model.velocity_observations:
        assert velocity > 0
        assert length > 0
        assert travel_time > 0
        assert math.isfinite(travel_time)

    # Congestion factors and durations stay within the configured bounds.
    highest_factor = config["congestion_upper_bound"] / MAX_SPREAD_DAMPING
    for minute_start, _arc, multiplier, end_minute in generator.events:
        assert config["congestion_lower_bound"] <= multiplier <= highest_factor + 1e-9
        assert minute_start + 30 <= end_minute <= minute_start + duration

    # Total cost equals the sum of the four components (float accumulation order),
    # except the documented double-charge when terminating past the horizon.
    component_sum = (
        result.distance_cost + result.delay_cost + result.earliness_cost + result.overtime_cost
    )
    for component_total in (
        result.distance_cost,
        result.delay_cost,
        result.earliness_cost,
        result.overtime_cost,
    ):
        assert component_total >= 0
    if model.terminate_passing_horizon_calls:
        expected_total = component_sum + model.transition_cost
    else:
        expected_total = component_sum
    assert math.isclose(result.total_cost, expected_total, rel_tol=1e-9, abs_tol=1e-9)


# --- The CongestionGenerator bounds property (cheap, no fixture) ------------------


@settings(max_examples=80, deadline=None, derandomize=True)
@given(
    event_probability=st.dictionaries(
        keys=st.tuples(st.integers(0, 5), st.integers(0, 5)),
        values=st.floats(0, 1, allow_nan=False),
        min_size=1,
        max_size=10,
    ),
    successors=st.dictionaries(
        keys=st.integers(0, 5),
        values=st.lists(st.integers(0, 5), unique=True, max_size=6),
        max_size=6,
    ),
    lower=st.floats(0.05, 0.9, allow_nan=False),
    upper_offset=st.floats(0, 0.5, allow_nan=False),
    duration=st.integers(40, 300),
    minute_start=st.floats(HORIZON_START, EMERGENCY_HORIZON, allow_nan=False),
    np_seed=st.integers(0, 2**32 - 1),
)
def test_congestion_generator_stays_in_bounds(
    event_probability: dict[tuple[int, int], float],
    successors: dict[int, list[int]],
    lower: float,
    upper_offset: float,
    duration: int,
    minute_start: float,
    np_seed: int,
) -> None:
    upper = min(lower + upper_offset, 0.95)
    generator = ArcProbabilityCongestionGenerator(
        event_probability=event_probability,
        successors=successors,
        congestion_lower_bound=lower,
        congestion_upper_bound=upper,
        max_congestion_duration=duration,
    )
    np.random.seed(np_seed)

    congested: CongestedArcs = {}
    generator.generate(minute_start, congested)

    for multiplier, end_minute in congested.values():
        assert lower <= multiplier <= upper / MAX_SPREAD_DAMPING + 1e-9
        assert minute_start + 30 <= end_minute <= minute_start + duration
