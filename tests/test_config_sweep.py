"""The robustness gate: a config cross-product sweep (simulator-correctness ticket 10).

Three of the 19 review findings — B5, B15, B16 — were configuration traps: none
fires at the shipped values (``shift_end_minute=780``, ``max_congestion_duration=120``).
Tickets 02-09 each proved their own fix with a targeted invariant; nothing in the
repository swept the *configuration space* itself, and nothing rules out a fourth,
fifth or sixth trap at a value nobody has run yet.

This module runs every ticket 02-09 invariant this file can check without a
real Episode's cooperation, over a cross product of the five knobs the review's
traps hid in: ``shift_end_minute``/``episode_end_minute`` pairs, fleet size
(``vehicle_count``), Client count (``mean_number_clients``/``min_number_clients``,
via two named ``ClientGenerator`` configs) and ``max_congestion_duration``, times
three seeds. On every run it asserts:

- no exception raised;
- no negative cost component (B15);
- no NaN or infinity in sampled velocities, travel times, path lengths or the
  general-state feature vector;
- the congestion cadence (B16) fires the shipped count exactly as many times as
  the intended integer-arithmetic count, for that run's own duration/horizon;
- every directly-countable invariant from tickets 02-09 stays at zero
  (:class:`scripts.measurement_bench.BenchModel`'s counters: B1a/B1b, B7, B10,
  B11, B14, B17).

B9 (breadth-first spread) and the config-validation invariant (B15's other half)
are structural — independent of any Episode — and get their own dedicated tests
below instead of one per grid point.

**Named traps** (ticket 10's own list), each an explicit value in the grid
below rather than left to chance:

- ``max_congestion_duration`` 50 and 70 — the review's own two breaking
  durations (ticket 02 fixed the float-division cadence bug these exposed).
- The 310-350 tau window with ``min_number_clients`` low — ``LOW_DEMAND``
  crossed with ``vehicle_count=1`` makes the endgame branches
  (``clients_not_visited`` under 3, B5/B11's territory) far more likely to
  actually execute during a swept Episode, complementing (not replacing)
  ``tests/unit/test_monte_carlo_policy.py::TestEndgameInvariants``, which
  forces the branch directly rather than hoping an Episode wanders into it.
- ``shift_end_minute`` at and above ``episode_end_minute`` — see
  ``test_shift_end_minute_validation_boundary`` below. **Correction to this
  ticket's own wording**: ``ExperimentConfig`` (ticket 02, B15) validates
  ``shift_end_minute <= episode_end_minute``, so "at" (equal) is the accepted
  boundary by design — a shift that ends exactly when the episode does is a
  legitimate zero-overtime config — and only "above" (strictly greater) is
  rejected. Both halves of that boundary are asserted explicitly rather than
  only the rejection this ticket's checklist names.

Mini-fixture episodes are the expensive unit here (~0.03-0.3 s each per
``tests/test_invariants.py``), so the grid is sized to run in well under a
minute: 4 horizon pairs x 6 durations x 3 fleet sizes x 2 demand configs x 3
seeds = 432 combinations, stated here so "the sweep passed" is not a claim
without a stated denominator (ticket 10's own evidence requirement).
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
import pytest

import stdvrp.simulation.episode as episode_module
from stdvrp.config import ExperimentConfig
from stdvrp.congestion import ArcProbabilityCongestionGenerator
from stdvrp.demand import ClientGenerator
from stdvrp.simulation import run_evaluation_episode
from stdvrp.simulation.episode_velocities import ArcVelocity
from stdvrp.simulation.model import DECISION_EPOCH_MINUTES, Model

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "chengdu_mini"

HORIZON_START = 300

# --- Client count axis: two named ClientGenerator configs --------------------

_SHIPPED_DEMAND_KWARGS = dict(
    mean_number_clients=20,
    client_count_stddev=4.0,
    min_number_clients=8,
    client_universe_node_range=(1, 45),
    clients_per_vehicle=4,
    time_window_spread=60,
    horizon_start_minute=HORIZON_START,
    shift_end_minute=780,
)
# Named trap: low min_number_clients makes the "<3 clients_not_visited" endgame
# branches (B5's empty-min() crash, B11's duplicate booking) far more reachable.
_LOW_DEMAND_KWARGS = dict(
    _SHIPPED_DEMAND_KWARGS, mean_number_clients=3, client_count_stddev=1.0, min_number_clients=1
)

DEMAND_VARIANTS: dict[str, dict[str, Any]] = {
    "shipped": _SHIPPED_DEMAND_KWARGS,
    "low": _LOW_DEMAND_KWARGS,
}

# --- (shift_end_minute, episode_end_minute) axis ------------------------------

HORIZON_PAIRS: tuple[tuple[int, int], ...] = (
    (780, 1150),  # shipped
    (301, 320),  # degenerate: shift ends almost immediately after horizon_start
    (600, 600),  # boundary: shift_end_minute == episode_end_minute (see below)
    (1100, 1150),  # shift end near the top of the emergency horizon
)

# --- max_congestion_duration axis: shipped/tested values plus B16's traps ----

DURATIONS: tuple[int, ...] = (30, 50, 70, 120, 200, 240)

# --- fleet size axis (None = demand-generated fleet) --------------------------

VEHICLE_COUNTS: tuple[int | None, ...] = (1, None, 4)

SEEDS: tuple[int, ...] = (0, 1, 2)


class SweepCase(NamedTuple):
    demand_name: str
    shift_end_minute: int
    episode_end_minute: int
    max_congestion_duration: int
    vehicle_count: int | None
    seed: int


GRID: list[SweepCase] = [
    SweepCase(demand_name, shift_end_minute, episode_end_minute, duration, vehicle_count, seed)
    for demand_name in DEMAND_VARIANTS
    for shift_end_minute, episode_end_minute in HORIZON_PAIRS
    for duration in DURATIONS
    for vehicle_count in VEHICLE_COUNTS
    for seed in SEEDS
]


def _case_id(case: SweepCase) -> str:
    return (
        f"demand={case.demand_name}-shift={case.shift_end_minute}"
        f"-episode_end={case.episode_end_minute}-duration={case.max_congestion_duration}"
        f"-vehicles={case.vehicle_count}-seed={case.seed}"
    )


# --- B16 cadence: the real production oracle, varied over episode_end_minute --
#
# ``scripts/measurement_bench.py``'s own ``check_b16_cadence`` is ticket 01's
# frozen *reproduction* of the pre-fix bug (a standalone float-division
# formula, never updated after ticket 02's fix landed — its own resolution
# notes confirm the bench's saved output is byte-identical before and after).
# It is evidence of what the old code used to do, not a live check of
# ``Model._congestion_epoch_due``. This sweep instead calls the real method
# directly, mirroring ``tests/unit/test_congestion_epoch_cadence.py``'s own
# oracle (``(tau + 178) % duration == 0``) — that file fixes
# ``episode_end_minute`` at 1150; this varies it over the grid's own pairs,
# which is the dimension ticket 10 adds.


def _cadence_fires(horizon_start: int, episode_end: int, duration: int) -> tuple[int, int]:
    """(what the real Model fires, what the integer-arithmetic oracle intends)."""
    taus = []
    tau = float(horizon_start + DECISION_EPOCH_MINUTES)
    while tau < episode_end:
        taus.append(tau)
        tau += DECISION_EPOCH_MINUTES

    model = Model.__new__(Model)
    model.max_congestion_duration = duration
    model.state = SimpleNamespace(tau_episode=0.0)  # type: ignore[assignment]
    fires_model = 0
    for tau in taus:
        model.state.tau_episode = tau
        if model._congestion_epoch_due():
            fires_model += 1
    fires_intended = sum(1 for tau in taus if (tau + 178) % duration == 0)
    return fires_model, fires_intended


# --- Instrumentation: BenchModel (ticket 01) plus the two checks unique here --


def _sweep_model(bench: ModuleType) -> type:
    """Build ``SweepModel`` against the loaded ``measurement_bench`` module.

    A subclass of ``BenchModel`` (not a fresh reimplementation) so B1a/B1b, B7,
    B10, B11, B14 and B17's end-of-episode counters are the exact ones tickets
    02-09 already measured with; this only adds what the robustness gate needs
    beyond them: finite velocities/features, and the per-clock-advance purge
    check (ticket 07's continuous invariant, not just a terminal snapshot).
    """

    class SweepModel(bench.BenchModel):  # type: ignore[misc, name-defined]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.non_finite_violations = 0
            self.congestion_purge_violations = 0
            super().__init__(*args, **kwargs)
            self._real_sample = self.velocities.sample
            self.velocities.sample = self._checked_sample  # type: ignore[method-assign]

        def _checked_sample(
            self, node_start: float, node_end: float, tau_episode: float
        ) -> ArcVelocity:
            sampled = self._real_sample(node_start, node_end, tau_episode)
            if not (
                math.isfinite(sampled.travel_time)
                and math.isfinite(sampled.velocity)
                and math.isfinite(sampled.length)
            ):
                self.non_finite_violations += 1
            return sampled

        def transition_function(self, action: list[int]) -> float:
            extractor = getattr(self.policy, "feature_extractor", None)
            if extractor is not None:
                general = extractor.state_features(self.state).general
                if not np.all(np.isfinite(general)):
                    self.non_finite_violations += 1
            return super().transition_function(action)

        def advance_fleet_to(self, minute: float) -> None:
            super().advance_fleet_to(minute)
            tau = self.state.tau_episode
            expiries = self.velocities.congested_arcs.values()
            if any(tau >= expiry for _multiplier, expiry in expiries):
                self.congestion_purge_violations += 1

    return SweepModel


# --- Fixtures ------------------------------------------------------------------


@pytest.fixture(scope="module")
def sweep_model_class(measurement_bench_module: ModuleType) -> type:
    return _sweep_model(measurement_bench_module)


@pytest.fixture(scope="module")
def base_config() -> ExperimentConfig:
    return ExperimentConfig.from_yaml(FIXTURE_DIR / "config.yaml")


# --- The cross-product sweep ---------------------------------------------------


@pytest.mark.parametrize("case", GRID, ids=_case_id)
def test_config_sweep(
    perturbed_mini_world: dict[str, Any],
    measurement_bench_module: ModuleType,
    sweep_model_class: type,
    case: SweepCase,
) -> None:
    travel_time_model = perturbed_mini_world["travel_time_model"]
    arcs = perturbed_mini_world["arcs"]

    client_generator = ClientGenerator(**DEMAND_VARIANTS[case.demand_name])
    congestion_generator = measurement_bench_module.BenchCongestionGenerator(
        ArcProbabilityCongestionGenerator(
            event_probability=dict.fromkeys(arcs, 0.05),
            successors=travel_time_model.successors,
            congestion_lower_bound=0.3,
            congestion_upper_bound=0.4,
            max_congestion_duration=case.max_congestion_duration,
        )
    )

    original_model = episode_module.Model
    episode_module.Model = sweep_model_class  # type: ignore[misc]
    try:
        # No exception raised (B5's own crash, and any regression like it) is
        # the first, unconditional assertion: an uncaught exception here fails
        # this test node directly.
        result = run_evaluation_episode(
            seed=case.seed,
            client_generator=client_generator,
            travel_time_model=travel_time_model,
            shortest_path_cache=perturbed_mini_world["shortest_path_cache"],
            congestion_generator=congestion_generator,
            W=None,
            epsilon=0.1,
            max_congestion_duration=case.max_congestion_duration,
            horizon_start_minute=HORIZON_START,
            shift_end_minute=case.shift_end_minute,
            episode_end_minute=case.episode_end_minute,
            n_observed_velocities=3,
            vehicle_count=case.vehicle_count,
        )
    finally:
        episode_module.Model = original_model

    model = sweep_model_class.last_instance
    assert isinstance(model, sweep_model_class)
    costs = model.costs

    # No negative cost component (B15).
    components = (costs.distance_cost, costs.delay_cost, costs.earliness_cost, costs.overtime_cost)
    for component in components:
        assert component >= 0, f"negative cost component: {components}"

    # No NaN or infinity anywhere: velocities/features (checked continuously by
    # SweepModel above) plus the final costs.
    assert model.non_finite_violations == 0
    for value in (
        result.total_cost,
        result.distance_cost,
        result.delay_cost,
        result.earliness_cost,
        result.overtime_cost,
    ):
        assert math.isfinite(value)

    # B16: the real cadence gate fires exactly the intended integer-arithmetic
    # count, for this run's own duration and episode_end_minute.
    fires_model, fires_intended = _cadence_fires(
        HORIZON_START, case.episode_end_minute, case.max_congestion_duration
    )
    assert fires_model == fires_intended

    # Ticket 02-09's other directly-countable invariants stay at zero.
    assert model.mid_arc_park_violations == 0  # B1a/B1b
    assert model.duplicate_assignment_transitions == 0  # B11
    assert model.earliness_bin_violations == 0  # B10
    assert congestion_generator.shorten_violations == 0  # B7
    assert model.congestion_book_expired_at_end == 0  # B17, terminal snapshot
    assert model.congestion_purge_violations == 0  # B17, every clock advance
    # B14: money charged > 0 iff its counter > 0, for every component.
    assert (costs.delay_cost > 0) == (costs.late_clients > 0 or costs.unserved_clients > 0)
    assert (costs.overtime_cost > 0) == (costs.overtime_vehicles > 0)
    assert (costs.earliness_cost > 0) == (costs.early_clients > 0)


# --- Structural invariants: independent of any Episode ------------------------


def test_b9_spread_is_order_independent_and_minimal_depth_everywhere(
    perturbed_mini_world: dict[str, Any], measurement_bench_module: ModuleType
) -> None:
    """B9 (ticket 08): re-verified on this fixture's own topology, once (not per grid point)."""
    check = measurement_bench_module.check_b9_spread_depths(
        perturbed_mini_world["travel_time_model"].successors
    )
    assert check.wrong_depth == 0
    assert check.missed_entirely == 0


# --- Named trap: shift_end_minute vs. episode_end_minute validation -----------


def test_shift_end_minute_validation_boundary(base_config: ExperimentConfig) -> None:
    """B15 (ticket 02): shift_end_minute <= episode_end_minute is the validated boundary.

    Ticket 10's own checklist says "at and above episode_end_minute ... must be
    rejected" — but the effort's own boundary decision (spec.md decision 5,
    landed in ticket 02) is ``<=``, not ``<``: a shift that ends exactly when
    the episode does is a legitimate zero-overtime config, not a trap. Both
    halves of the real boundary are asserted here: "at" is accepted, "above" is
    rejected.
    """
    accepted = dataclasses.replace(base_config, shift_end_minute=600, episode_end_minute=600)
    assert accepted.shift_end_minute == accepted.episode_end_minute == 600

    with pytest.raises(ValueError, match="shift_end_minute"):
        dataclasses.replace(base_config, shift_end_minute=601, episode_end_minute=600)
