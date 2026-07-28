"""Unit tests for EpisodeGeometry (ticket 04, simulation-performance).

Pins the facade's contract against ShortestPathCache.path_between: scalar
accessors must be bit-identical to the dict lookups they replace, including
KeyError-equivalent behavior for absent (node, client) pairs, and the matrix
covers exactly the depot plus this Episode's Clients as columns.
"""

from pathlib import Path

import numpy as np
import pytest

from stdvrp.network.episode_geometry import EpisodeGeometry
from stdvrp.network.shortest_path_cache import ShortestPath, ShortestPathCache

FIXTURE_CSV = (
    Path(__file__).resolve().parents[1] / "fixtures" / "chengdu_mini" / "all_shortest_paths.csv"
)

DEPOT = 0


def make_cache(arcs: dict) -> ShortestPathCache:
    """arcs: (node, client) -> (average_minutes, length)."""
    return ShortestPathCache(
        {
            key: ShortestPath([float(key[0]), float(key[1])], minutes, length)
            for key, (minutes, length) in arcs.items()
        }
    )


SPARSE_CACHE = make_cache(
    {
        (0, 0): (0.0, 0.0),
        (0, 1): (10.0, 5.0),
        (1, 0): (10.0, 5.0),
        (0, 2): (20.0, 8.0),
        (2, 0): (20.0, 8.0),
        (1, 2): (15.0, 6.0),
        (2, 1): (15.0, 6.0),
    }
)

# Dense over {0, 1, 2}: every (node, client) pair including self-loops, for
# vector-accessor tests that read a full row/column (row/column accessors
# assume every column is priced, like a real ShortestPathCache).
DENSE_CACHE = make_cache(
    {
        (node, client): (float(node + client), float(node + client) / 2)
        for node in range(3)
        for client in range(3)
    }
)


class TestBuild:
    def test_columns_are_depot_first_then_clients_in_order(self) -> None:
        geometry = EpisodeGeometry.build(SPARSE_CACHE, clients=[2, 1], depot=DEPOT)
        assert geometry.columns == (0, 2, 1)

    def test_depot_is_not_duplicated_when_already_in_clients(self) -> None:
        geometry = EpisodeGeometry.build(SPARSE_CACHE, clients=[0, 1, 2], depot=DEPOT)
        assert geometry.columns == (0, 1, 2)

    def test_duplicate_clients_collapse_to_one_column(self) -> None:
        geometry = EpisodeGeometry.build(SPARSE_CACHE, clients=[1, 1, 2], depot=DEPOT)
        assert geometry.columns == (0, 1, 2)


class TestScalarAccessorsMatchTheCache:
    def test_average_minutes_matches_path_between(self) -> None:
        geometry = EpisodeGeometry.build(SPARSE_CACHE, clients=[1, 2], depot=DEPOT)
        for node, client in [(0, 0), (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]:
            assert (
                geometry.average_minutes(node, client)
                == SPARSE_CACHE.path_between(node, client).average_minutes
            )

    def test_length_matches_path_between(self) -> None:
        geometry = EpisodeGeometry.build(SPARSE_CACHE, clients=[1, 2], depot=DEPOT)
        for node, client in [(0, 0), (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]:
            assert geometry.length(node, client) == SPARSE_CACHE.path_between(node, client).length

    def test_scalar_accessors_return_native_floats(self) -> None:
        geometry = EpisodeGeometry.build(SPARSE_CACHE, clients=[1, 2], depot=DEPOT)
        assert type(geometry.average_minutes(0, 1)) is float
        assert type(geometry.length(0, 1)) is float

    def test_float_and_int_node_ids_hash_equal_like_the_legacy_parse(self) -> None:
        # ShortestPathCache keys are ints (from_csv); vehicle positions become
        # floats once a vehicle reaches an intermediate path node (State docstring).
        geometry = EpisodeGeometry.build(SPARSE_CACHE, clients=[1, 2], depot=DEPOT)
        assert geometry.average_minutes(0.0, 1) == geometry.average_minutes(0, 1)


class TestAbsentPairsRaiseKeyError:
    def test_missing_self_loop_raises_key_error(self) -> None:
        # (1, 1) is absent from SPARSE_CACHE, exactly like the legacy dict lookup.
        geometry = EpisodeGeometry.build(SPARSE_CACHE, clients=[1, 2], depot=DEPOT)
        with pytest.raises(KeyError):
            SPARSE_CACHE.path_between(1, 1)
        with pytest.raises(KeyError):
            geometry.average_minutes(1, 1)
        with pytest.raises(KeyError):
            geometry.length(1, 1)

    def test_node_outside_the_cache_raises_key_error(self) -> None:
        geometry = EpisodeGeometry.build(SPARSE_CACHE, clients=[1, 2], depot=DEPOT)
        with pytest.raises(KeyError):
            geometry.average_minutes(99999, 1)

    def test_client_outside_this_episodes_columns_raises_key_error(self) -> None:
        # Client 2 exists in the cache but this Episode only serves Client 1.
        geometry = EpisodeGeometry.build(SPARSE_CACHE, clients=[1], depot=DEPOT)
        with pytest.raises(KeyError):
            geometry.average_minutes(0, 2)


class TestVectorAccessors:
    def test_average_minutes_row_matches_scalar_reads(self) -> None:
        geometry = EpisodeGeometry.build(DENSE_CACHE, clients=[1, 2], depot=DEPOT)
        row = geometry.average_minutes_row(0)
        expected = [geometry.average_minutes(0, c) for c in geometry.columns]
        np.testing.assert_array_equal(row, expected)

    def test_length_row_matches_scalar_reads(self) -> None:
        geometry = EpisodeGeometry.build(DENSE_CACHE, clients=[1, 2], depot=DEPOT)
        row = geometry.length_row(1)
        expected = [geometry.length(1, c) for c in geometry.columns]
        np.testing.assert_array_equal(row, expected)

    def test_average_minutes_column_matches_scalar_reads(self) -> None:
        geometry = EpisodeGeometry.build(DENSE_CACHE, clients=[1, 2], depot=DEPOT)
        column = geometry.average_minutes_column(2)
        expected = [geometry.average_minutes(n, 2) for n in (0, 1, 2)]
        np.testing.assert_array_equal(column, expected)

    def test_average_minutes_at_matches_scalar_reads_of_those_columns(self) -> None:
        geometry = EpisodeGeometry.build(DENSE_CACHE, clients=[1, 2], depot=DEPOT)
        positions = geometry.column_positions([2, 0])
        expected = [geometry.average_minutes(1, 2), geometry.average_minutes(1, 0)]
        np.testing.assert_array_equal(geometry.average_minutes_at(1, positions), expected)

    def test_column_positions_locate_clients_in_column_order(self) -> None:
        geometry = EpisodeGeometry.build(DENSE_CACHE, clients=[1, 2], depot=DEPOT)
        np.testing.assert_array_equal(geometry.column_positions([2, 1, 0]), [2, 1, 0])
        with pytest.raises(KeyError):
            geometry.column_positions([99999])

    def test_average_minutes_at_raises_for_a_pair_the_cache_never_priced(self) -> None:
        # Unlike the raw row accessor, this one keeps the scalar KeyError contract:
        # (1, 1) is absent from SPARSE_CACHE, so the whole slice must raise.
        geometry = EpisodeGeometry.build(SPARSE_CACHE, clients=[1, 2], depot=DEPOT)
        with pytest.raises(KeyError):
            geometry.average_minutes_at(1, geometry.column_positions([2, 1]))
        np.testing.assert_array_equal(
            geometry.average_minutes_at(1, geometry.column_positions([2, 0])),
            [geometry.average_minutes(1, 2), geometry.average_minutes(1, 0)],
        )

    def test_average_minutes_rows_stacks_one_row_per_node_in_order(self) -> None:
        # The whole fleet's travel times in one gather (ticket 05): rows follow the
        # node order handed in, which is vehicle order, not the matrix's row order.
        geometry = EpisodeGeometry.build(DENSE_CACHE, clients=[1, 2], depot=DEPOT)
        rows = geometry.average_minutes_rows([2, 0, 2])
        expected = [geometry.average_minutes_row(node) for node in (2, 0, 2)]
        np.testing.assert_array_equal(rows, expected)

    def test_length_rows_stacks_one_row_per_node_in_order(self) -> None:
        geometry = EpisodeGeometry.build(DENSE_CACHE, clients=[1, 2], depot=DEPOT)
        rows = geometry.length_rows([1, 0])
        np.testing.assert_array_equal(rows, [geometry.length_row(1), geometry.length_row(0)])

    def test_rows_for_no_nodes_keeps_the_column_width(self) -> None:
        geometry = EpisodeGeometry.build(DENSE_CACHE, clients=[1, 2], depot=DEPOT)
        assert geometry.average_minutes_rows([]).shape == (0, len(geometry.columns))

    def test_rows_for_an_unknown_node_raises_key_error(self) -> None:
        geometry = EpisodeGeometry.build(DENSE_CACHE, clients=[1, 2], depot=DEPOT)
        with pytest.raises(KeyError):
            geometry.average_minutes_rows([0, 99999])

    def test_row_for_unknown_node_raises_key_error(self) -> None:
        geometry = EpisodeGeometry.build(DENSE_CACHE, clients=[1, 2], depot=DEPOT)
        with pytest.raises(KeyError):
            geometry.average_minutes_row(99999)

    def test_column_for_client_outside_episode_raises_key_error(self) -> None:
        geometry = EpisodeGeometry.build(DENSE_CACHE, clients=[1], depot=DEPOT)
        with pytest.raises(KeyError):
            geometry.average_minutes_column(2)


class TestAgainstTheRealFixtureCache:
    """Cross-checks against the committed mini-fixture cache (45x45, dense)."""

    def test_matches_path_between_across_a_sample_of_pairs(self) -> None:
        cache = ShortestPathCache.from_csv(FIXTURE_CSV)
        clients = list(range(1, 22))  # 21 clients, like the committed fixture world
        geometry = EpisodeGeometry.build(cache, clients=clients, depot=DEPOT)

        for node in range(0, 45):
            for client in [DEPOT, *clients[::5]]:
                expected = cache.path_between(node, client)
                assert geometry.average_minutes(node, client) == expected.average_minutes
                assert geometry.length(node, client) == expected.length

    def test_client_not_in_this_episode_is_unreachable_even_if_cached(self) -> None:
        cache = ShortestPathCache.from_csv(FIXTURE_CSV)
        geometry = EpisodeGeometry.build(cache, clients=[1, 2, 3], depot=DEPOT)
        # Client 40 has a cached path in the fixture, but this Episode never serves it.
        assert (0, 40) in cache
        with pytest.raises(KeyError):
            geometry.average_minutes(0, 40)
