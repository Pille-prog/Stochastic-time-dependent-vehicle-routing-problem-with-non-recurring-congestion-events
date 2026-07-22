"""Structural guarantees of the committed Chengdu mini fixture.

The fixture under ``tests/fixtures/chengdu_mini/`` is a renumbered sub-network
extracted from the real Chengdu data by ``scripts/make_fixture.py``. Every test
here is a property the rest of the suite may rely on without re-checking.
"""

from itertools import pairwise
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "chengdu_mini"

LINK_COLUMNS = [
    "Link",
    "Node_Start",
    "Longitude_Start",
    "Latitude_Start",
    "Node_End",
    "Longitude_End",
    "Latitude_End",
    "Length",
]
SPEED_COLUMNS = ["Period", "Link", "Speed"]
PATHS_COLUMNS = ["Node", "Client", "ShortestPath", "AverageTime", "Length"]

DATA_FILES = [
    "link.csv",
    "speed[601]_[0].csv",
    "speed[601]_[1].csv",
    "all_shortest_paths.csv",
    "node_map.csv",
]

# The simulated Horizon is [300, 780]; periods are 2-minute intervals.
HORIZON_START_MINUTE = 300
HORIZON_END_MINUTE = 782


def period_start_minute(period: str) -> int:
    """Minutes since 03:00 of a Period's start, as the legacy pipeline defines it."""
    start = period.split("-")[0]
    hour, minute = map(int, start.split(":"))
    return (hour - 3) * 60 + minute


@pytest.fixture(scope="module")
def link() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "link.csv")


@pytest.fixture(scope="module")
def speeds() -> dict[str, pd.DataFrame]:
    return {half: pd.read_csv(FIXTURE_DIR / f"speed[601]_[{half}].csv") for half in ("0", "1")}


@pytest.fixture(scope="module")
def paths() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "all_shortest_paths.csv")


def test_all_files_present() -> None:
    for name in DATA_FILES:
        assert (FIXTURE_DIR / name).is_file(), f"missing {name}"


def test_total_size_well_under_one_megabyte() -> None:
    total = sum((FIXTURE_DIR / name).stat().st_size for name in DATA_FILES)
    assert total < 1_000_000, f"fixture is {total} bytes"


def test_schemas_match_real_data() -> None:
    assert list(pd.read_csv(FIXTURE_DIR / "link.csv", nrows=0).columns) == LINK_COLUMNS
    for half in ("0", "1"):
        speed_file = FIXTURE_DIR / f"speed[601]_[{half}].csv"
        assert list(pd.read_csv(speed_file, nrows=0).columns) == SPEED_COLUMNS
    paths_file = FIXTURE_DIR / "all_shortest_paths.csv"
    assert list(pd.read_csv(paths_file, nrows=0).columns) == PATHS_COLUMNS


def test_nodes_are_contiguous_and_include_depot(link: pd.DataFrame) -> None:
    nodes = set(link["Node_Start"]) | set(link["Node_End"])
    assert 0 in nodes, "depot node 0 missing"
    assert nodes == set(range(len(nodes))), "node ids must be contiguous 0..N-1"
    assert 30 <= len(nodes) <= 100, f"unexpected fixture size: {len(nodes)} nodes"


def test_network_is_strongly_connected(link: pd.DataFrame) -> None:
    graph = nx.DiGraph()
    graph.add_edges_from(zip(link["Node_Start"], link["Node_End"], strict=True))
    assert nx.is_strongly_connected(graph)


def test_speed_files_cover_every_link_and_horizon_period(
    link: pd.DataFrame, speeds: dict[str, pd.DataFrame]
) -> None:
    link_ids = set(link["Link"])
    for half, speed in speeds.items():
        assert set(speed["Link"]) == link_ids, f"half {half}: link coverage mismatch"
        minutes = speed["Period"].map(period_start_minute)
        assert (minutes >= HORIZON_START_MINUTE).all()
        assert (minutes < HORIZON_END_MINUTE).all()
        periods = set(speed["Period"])
        per_link = speed.groupby("Link")["Period"].apply(set)
        assert (per_link == periods).all(), f"half {half}: some link misses periods"


def test_speeds_are_positive(speeds: dict[str, pd.DataFrame]) -> None:
    for speed in speeds.values():
        assert (speed["Speed"] > 0).all()


def test_paths_cover_all_ordered_pairs(link: pd.DataFrame, paths: pd.DataFrame) -> None:
    n = len(set(link["Node_Start"]) | set(link["Node_End"]))
    assert len(paths) == n * n
    assert set(zip(paths["Node"], paths["Client"], strict=True)) == {
        (a, b) for a in range(n) for b in range(n)
    }


def test_paths_walk_real_arcs_only(link: pd.DataFrame, paths: pd.DataFrame) -> None:
    arcs = set(zip(link["Node_Start"], link["Node_End"], strict=True))
    for row in paths.itertuples():
        hops = [int(float(part)) for part in str(row.ShortestPath).split("->")]
        assert hops[0] == row.Node and hops[-1] == row.Client
        for a, b in pairwise(hops):
            assert (a, b) in arcs, f"path {row.Node}->{row.Client} uses missing arc ({a},{b})"


def test_node_map_is_a_bijection_into_real_ids(link: pd.DataFrame) -> None:
    node_map = pd.read_csv(FIXTURE_DIR / "node_map.csv")
    assert list(node_map.columns) == ["fixture_node", "chengdu_node"]
    n = len(set(link["Node_Start"]) | set(link["Node_End"]))
    assert sorted(node_map["fixture_node"]) == list(range(n))
    assert node_map["chengdu_node"].is_unique
    assert (node_map.loc[node_map["fixture_node"] == 0, "chengdu_node"] == 0).all(), (
        "depot must map to the real Chengdu depot (node 0)"
    )
