"""Shared fixtures for the test suite."""

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

import characterization_world
from stdvrp.network import ShortestPathCache
from stdvrp.traffic import CsvDataSource, TravelTimeModel

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "chengdu_mini"
# Real gauss draws around the single committed fixture day, so every arc gets a
# positive speed standard deviation (a single day would leave the
# legacy-preserved NaN stds, see TravelTimeModel) — needed for congestion's
# stochastic velocity draws to mean anything.
PERTURBED_DAYS = tuple(range(601, 609))


def _load_script_module(name: str) -> ModuleType:
    """Load ``scripts/<name>.py`` by file path (scripts/ is not an importable package).

    Registered in ``sys.modules`` before exec so dataclasses under
    ``from __future__ import annotations`` can resolve their own module.
    """
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def capture() -> ModuleType:
    """The golden-master capture module (scripts/ is not an importable package)."""
    return _load_script_module("capture_golden_master")


@pytest.fixture(scope="session")
def benchmark_module() -> ModuleType:
    """scripts/benchmark_episodes.py — the episode benchmark and projection (ticket 01)."""
    return _load_script_module("benchmark_episodes")


@pytest.fixture(scope="session")
def measurement_bench_module() -> ModuleType:
    """scripts/measurement_bench.py — the invariant-violation counters (ticket 01).

    Reused by ``tests/test_config_sweep.py`` (ticket 10) so the robustness gate
    checks the same, already-reviewed counters tickets 02-09 measured
    before/after with, instead of a second implementation of the same checks.
    """
    return _load_script_module("measurement_bench")


# --- Legacy-monolith characterization venue (see characterization_world) ---


@pytest.fixture(scope="module")
def legacy_world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return characterization_world.build_legacy_world(tmp_path_factory.mktemp("legacy_world"))


@pytest.fixture(scope="module")
def legacy_module() -> ModuleType:
    return characterization_world.load_legacy_module()


# --- Perturbed mini-fixture traffic world (shared by test_invariants.py and
# test_config_sweep.py, simulator-correctness tickets 01/10) -----------------


@pytest.fixture(scope="module")
def perturbed_mini_world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A fixture-derived traffic world whose perturbed days give positive speed stds.

    Just the road network and travel-time statistics — no demand, no congestion
    generator — so every consumer builds its own ``ClientGenerator``/
    ``ArcProbabilityCongestionGenerator`` around it for whatever it is sweeping.
    """
    world = tmp_path_factory.mktemp("perturbed_mini_world")
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
        horizon_start_minute=300,
    )
    return {
        "travel_time_model": travel_time_model,
        "shortest_path_cache": ShortestPathCache.from_csv(FIXTURE_DIR / "all_shortest_paths.csv"),
        "arcs": list(travel_time_model.event_probability),
    }
