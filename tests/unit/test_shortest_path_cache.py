"""ShortestPathCache: legacy-format CSV loading and lookup."""

from pathlib import Path

import pytest

from stdvrp.network import ShortestPath, ShortestPathCache

FIXTURE_CSV = (
    Path(__file__).resolve().parents[1] / "fixtures" / "chengdu_mini" / "all_shortest_paths.csv"
)


@pytest.fixture(scope="module")
def cache() -> ShortestPathCache:
    return ShortestPathCache.from_csv(FIXTURE_CSV)


def test_loads_every_node_client_pair(cache: ShortestPathCache) -> None:
    assert len(cache) == 45 * 45


def test_path_between_returns_the_csv_row_exactly(cache: ShortestPathCache) -> None:
    # Second data row of the fixture CSV: 0,1,0->1,1.613732,0.921041
    assert cache.path_between(0, 1) == ShortestPath([0.0, 1.0], 1.613732, 0.921041)


def test_self_path_is_a_single_node_with_zero_cost(cache: ShortestPathCache) -> None:
    path = cache.path_between(0, 0)
    assert path.nodes == [0.0]
    assert path.average_minutes == 0.0
    assert path.length == 0.0


def test_path_nodes_are_floats_like_the_legacy_parse(cache: ShortestPathCache) -> None:
    # The legacy loader maps path nodes through float(); they still hash and compare
    # equal to the int node ids, and Phase 1 preserves the quirk (ADR-0001).
    assert all(isinstance(node, float) for node in cache.path_between(0, 1).nodes)


def test_missing_pair_raises_key_error_like_legacy_indexing(cache: ShortestPathCache) -> None:
    with pytest.raises(KeyError):
        cache.path_between(0, 99999)


def test_contains_and_tuple_indexing(cache: ShortestPathCache) -> None:
    assert (0, 1) in cache
    assert (0, 99999) not in cache
    # The live legacy code indexes tuples positionally; ShortestPath keeps that shape.
    path = cache.path_between(0, 1)
    assert path[0] == path.nodes
    assert path[1] == path.average_minutes
    assert path[2] == path.length
