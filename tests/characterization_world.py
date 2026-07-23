"""Shared venue for the Episode characterization tests (tickets 07/08).

The unmodified legacy classes run on a temporary world of 44 fixture-day copies
(the legacy hardcodes 44 traffic days; single-day speed std is NaN, 44 identical
copies make it exactly 0.0), and the ported world is built from the same files.
Demand comes from the ported ClientGenerator for BOTH sides — ticket 06 proved its
global-stream consumption is bit-identical to the legacy's, so episodes on either
side start from identical RNG state.

Constants and builders live here; the module-scoped fixtures wiring them together
live in ``conftest.py``. Tests with a different world shape (e.g. the ticket 05
travel-time characterization) keep their own fixtures.
"""

import builtins
import importlib.util
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from legacy_source import legacy_script_path
from stdvrp.congestion import ArcProbabilityCongestionGenerator
from stdvrp.demand import ClientGenerator
from stdvrp.network import ShortestPathCache
from stdvrp.traffic import CsvDataSource, TravelTimeModel

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "chengdu_mini"
LEGACY_DAYS = tuple(range(601, 631)) + tuple(range(701, 715))

HORIZON_START, HORIZON_END = 300, 780
N_OBSERVED_ARCS = 3
CONGESTION_LOWER, CONGESTION_UPPER = 0.3, 0.4
MAX_CONGESTION_DURATION = 120

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


def build_legacy_world(world: Path) -> Path:
    """Populate ``world`` with link.csv plus the 44 fixture-day speed copies."""
    shutil.copyfile(FIXTURE_DIR / "link.csv", world / "link.csv")
    for day in LEGACY_DAYS:
        for half in (0, 1):
            shutil.copyfile(
                FIXTURE_DIR / f"speed[601]_[{half}].csv",
                world / f"speed[{day}]_[{half}].csv",
            )
    return world


def load_legacy_module() -> ModuleType:
    """Import the monolith unchanged (read-only reference, ADR-0001)."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    spec = importlib.util.spec_from_file_location("legacy_monolith", legacy_script_path())
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Imported modules get a dict __builtins__; the legacy transition_function
    # calls the literal ``__builtins__.min`` (script-style), so restore the module.
    module.__dict__["__builtins__"] = builtins
    return module


@contextmanager
def count_horizon_terminations(legacy_module: ModuleType) -> Iterator[list[int]]:
    """Count legacy ``terminate_state_passing_horizon`` calls during the block.

    Ticket 12 fix 6 removed the port's double-add of the terminating
    transition's cost past the emergency horizon; the episode comparisons
    compensate the exact legacy delta, which applies whenever this method fired
    (the legacy epoch-end gate always re-added the final transition cost).
    """
    calls: list[int] = []
    original = legacy_module.model.terminate_state_passing_horizon

    def counting(self: Any) -> None:
        calls.append(1)
        original(self)

    legacy_module.model.terminate_state_passing_horizon = counting
    try:
        yield calls
    finally:
        legacy_module.model.terminate_state_passing_horizon = original


def build_legacy_calc(legacy_module: ModuleType, legacy_world: Path) -> Any:
    """The legacy ``DataCalculations`` built inside the 44-copy world."""
    previous_cwd = os.getcwd()
    os.chdir(legacy_world)
    try:
        env = legacy_module.environment(
            "link.csv", "speed[601]_[0].csv", "speed[601]_[1].csv", [0], HORIZON_START, HORIZON_END
        )
        env.preprocess_data_average()
        calc = legacy_module.DataCalculations(env, MAX_CONGESTION_DURATION)
    finally:
        os.chdir(previous_cwd)
    return calc


def build_legacy_spm(legacy_module: ModuleType) -> Any:
    """The legacy path cache — loads "all_shortest_paths.csv" from the CWD."""
    previous_cwd = os.getcwd()
    os.chdir(FIXTURE_DIR)
    try:
        return legacy_module.shortest_path_memory(SimpleNamespace(node_list=[]))
    finally:
        os.chdir(previous_cwd)


def build_ported_world(legacy_world: Path) -> dict[str, Any]:
    """The new-package world over the same files as the legacy side."""
    source = CsvDataSource(legacy_world, "link.csv", 601, LEGACY_DAYS, "all_shortest_paths.csv")
    travel_time_model = TravelTimeModel(
        source.load_road_network(),
        source.load_traffic_history(),
        MAX_CONGESTION_DURATION,
        horizon_start_minute=HORIZON_START,
    )
    return {
        "travel_time_model": travel_time_model,
        "cache": ShortestPathCache.from_csv(FIXTURE_DIR / "all_shortest_paths.csv"),
        "client_generator": ClientGenerator(**FIXTURE_DEMAND),
        "congestion_generator": ArcProbabilityCongestionGenerator(
            event_probability=travel_time_model.event_probability,
            successors=travel_time_model.successors,
            congestion_lower_bound=CONGESTION_LOWER,
            congestion_upper_bound=CONGESTION_UPPER,
            max_congestion_duration=MAX_CONGESTION_DURATION,
        ),
    }
