"""Exact-equality characterization: the ported data spine vs the unmodified legacy.

The legacy ``environment``/``DataCalculations`` hardcode the 44 archive days
(601-630, 701-714), so they cannot run on the one-day fixture directly. Instead we
build a temporary world where every one of those 44 days is a copy of the fixture's
day 601, run the legacy classes byte-for-byte unmodified on it (read-only import,
allowed by ADR-0001 for characterization), and require the new TravelTimeModel to
reproduce every derived structure bit-for-bit, including insertion orders that the
simulation's RNG consumption depends on.

This complements ticket 04's golden master (full local data): here CI itself proves
the port is exact on real-shaped data.
"""

import importlib.util
import os
import shutil
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from stdvrp.traffic import CsvDataSource, TravelTimeModel

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "chengdu_mini"
LEGACY_SCRIPT = REPO_ROOT / "Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py"
LEGACY_DAYS = tuple(range(601, 631)) + tuple(range(701, 715))
MAX_CONGESTION_DURATION = 120


@pytest.fixture(scope="module")
def legacy_world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory satisfying the legacy's hardcoded file expectations."""
    world = tmp_path_factory.mktemp("legacy_world")
    shutil.copyfile(FIXTURE_DIR / "link.csv", world / "link.csv")
    for day in LEGACY_DAYS:
        for half in (0, 1):
            shutil.copyfile(
                FIXTURE_DIR / f"speed[601]_[{half}].csv",
                world / f"speed[{day}]_[{half}].csv",
            )
    return world


@pytest.fixture(scope="module")
def legacy_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("legacy_monolith", LEGACY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def legacy_spine(legacy_module: ModuleType, legacy_world: Path) -> tuple[object, object]:
    """(environment, DataCalculations) built exactly as the legacy main() builds them."""
    previous_cwd = os.getcwd()
    os.chdir(legacy_world)
    try:
        env = legacy_module.environment(
            "link.csv", "speed[601]_[0].csv", "speed[601]_[1].csv", [0], 300, 780
        )
        env.preprocess_data_average()
        calc = legacy_module.DataCalculations(env, MAX_CONGESTION_DURATION)
    finally:
        os.chdir(previous_cwd)
    return env, calc


@pytest.fixture(scope="module")
def ported_model(legacy_world: Path) -> TravelTimeModel:
    source = CsvDataSource(legacy_world, "link.csv", 601, LEGACY_DAYS)
    return TravelTimeModel(
        source.load_road_network(),
        source.load_traffic_history(),
        MAX_CONGESTION_DURATION,
        horizon_start_minute=300,
    )


def test_travel_data_is_bit_identical(
    legacy_spine: tuple[object, object], ported_model: TravelTimeModel
) -> None:
    _, calc = legacy_spine
    assert ported_model.travel_data == calc.travel_data


def test_speed_std_lookup_is_bit_identical(
    legacy_spine: tuple[object, object], ported_model: TravelTimeModel
) -> None:
    _, calc = legacy_spine
    assert ported_model.speed_std == calc.get_standard_deviation_dict


def test_successor_lists_match_in_order(
    legacy_spine: tuple[object, object], ported_model: TravelTimeModel
) -> None:
    # Congestion spreading walks these lists; order is behavior (ADR-0001).
    _, calc = legacy_spine
    assert ported_model.successors == calc.travel_arc_information_dictionary


def test_node_coordinates_match(
    legacy_spine: tuple[object, object], ported_model: TravelTimeModel
) -> None:
    _, calc = legacy_spine
    assert ported_model.node_coordinates == calc.latitude_and_longitude


def test_event_probabilities_match_in_value_and_iteration_order(
    legacy_spine: tuple[object, object], ported_model: TravelTimeModel
) -> None:
    # The live CongestionGenerator draws one random number per key in dict order,
    # so insertion order is part of the behavior being preserved (ADR-0001).
    _, calc = legacy_spine
    assert list(ported_model.event_probability) == list(calc.probability_of_event)
    assert ported_model.event_probability == calc.probability_of_event


def test_mean_arc_table_is_bit_identical(
    legacy_spine: tuple[object, object], ported_model: TravelTimeModel
) -> None:
    env, _ = legacy_spine
    pd.testing.assert_frame_equal(ported_model.mean_arc_data, env.data_average, check_exact=True)
