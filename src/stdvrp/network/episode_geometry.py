"""EpisodeGeometry: dense per-Episode travel-time/length matrices (ADR-0003).

Simulation-performance ticket 04 (see its ``## Comments`` for the full
rationale and benchmark numbers): replaces ``ShortestPathCache.path_between``
dict lookups on the Policy hot path with array indexing — one dense
``[node, client-or-depot]`` matrix pair per Episode. Rows are every node id
the cache ever priced toward some client (in practice the road network's full
node universe); columns are this Episode's Clients plus the depot. Values are
the cache's own floats, copied verbatim, so scalar reads stay bit-identical to
the lookups they replace (Tier 1, pure representation change).

Two design choices worth knowing before touching this file:

- **Per-cache memoization.** Rescanning the whole ShortestPathCache every
  Episode measurably regressed throughput, so the scan is memoized per
  ShortestPathCache instance (:func:`_full_index`, a ``WeakKeyDictionary`` —
  safe, since the cache is never mutated after construction); each Episode
  then does a cheap vectorized column-select out of that shared index.
- **Scalar reads use nested Python lists, not numpy indexing** — a single
  ``ndarray[i, j]`` read measured slower than a nested-list ``list[i][j]``
  read, and this path is called ~1e5 times per Episode. Each built matrix
  keeps its numpy form (for the vectorized row/column accessors tickets 05/07
  build on) *and* a ``.tolist()`` copy that the scalar accessors read from.

Path *node sequences* (``.nodes``, used by the Model for routing) are not part
of this facade — they stay on the ShortestPathCache (ticket 04 scope: only the
Policy's time/length reads migrate).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple
from weakref import WeakKeyDictionary

import numpy as np
from numpy.typing import NDArray

from stdvrp.network.shortest_path_cache import ShortestPathCache


class _FullIndex(NamedTuple):
    """The whole ShortestPathCache as dense matrices: every node x every client."""

    average_minutes: NDArray[np.float64]
    length: NDArray[np.float64]
    present: NDArray[np.bool_]
    row_index: dict[float, int]
    col_index: dict[float, int]


_full_index_cache: WeakKeyDictionary[ShortestPathCache, _FullIndex] = WeakKeyDictionary()


def _full_index(shortest_path_cache: ShortestPathCache) -> _FullIndex:
    """The memoized whole-cache index, building it once per cache instance."""
    cached = _full_index_cache.get(shortest_path_cache)
    if cached is not None:
        return cached
    built = _build_full_index(shortest_path_cache)
    _full_index_cache[shortest_path_cache] = built
    return built


def _build_full_index(shortest_path_cache: ShortestPathCache) -> _FullIndex:
    row_index: dict[float, int] = {}
    col_index: dict[float, int] = {}
    entries: list[tuple[float, float, float, float]] = []
    for (node, client), path in shortest_path_cache.items():
        entries.append((node, client, path.average_minutes, path.length))
        if node not in row_index:
            row_index[node] = len(row_index)
        if client not in col_index:
            col_index[client] = len(col_index)

    shape = (len(row_index), len(col_index))
    average_minutes = np.zeros(shape, dtype=np.float64)
    length = np.zeros(shape, dtype=np.float64)
    present = np.zeros(shape, dtype=np.bool_)
    for node, client, minutes_value, length_value in entries:
        row, col = row_index[node], col_index[client]
        average_minutes[row, col] = minutes_value
        length[row, col] = length_value
        present[row, col] = True

    return _FullIndex(average_minutes, length, present, row_index, col_index)


class EpisodeGeometry:
    """Dense ``[node, client-or-depot]`` matrices of travel minutes and length.

    Scalar accessors (:meth:`average_minutes`, :meth:`length`) mirror
    ``ShortestPathCache.path_between(...).average_minutes/.length`` exactly,
    including raising ``KeyError`` for a pair the cache never priced (or whose
    client is outside this Episode's columns). Row/column accessors return one
    node's or one client's full slice, aligned to :attr:`columns`, for the
    vectorized feature extraction and candidate selection built on top of this
    facade (tickets 05, 07); they assume every column is priced for that row —
    true for a real ShortestPathCache — and are not the source of truth for
    absent-pair semantics. :meth:`average_minutes_at` is the exception that
    proves that rule: a row restricted to the Client subset at
    :meth:`column_positions`, keeping the scalar accessors' ``KeyError``.
    """

    def __init__(
        self,
        average_minutes: NDArray[np.float64],
        length: NDArray[np.float64],
        present: NDArray[np.bool_],
        row_index: dict[float, int],
        columns: tuple[float, ...],
    ) -> None:
        self._average_minutes = average_minutes
        self._length = length
        self._present = present
        self._row_index = row_index
        self._columns = columns
        self._col_index = {client: index for index, client in enumerate(columns)}

        # Scalar-read copies (see module docstring): nested Python lists are
        # faster than numpy scalar indexing for the Policy's one-cell-at-a-time
        # reads; the numpy arrays above stay canonical for the row/column
        # vectorized accessors.
        self._average_minutes_list: list[list[float]] = average_minutes.tolist()
        self._length_list: list[list[float]] = length.tolist()
        self._present_list: list[list[bool]] = present.tolist()

        # Settled once so :meth:`average_minutes_at` can skip its presence check
        # on a real ShortestPathCache, which prices every (node, client) pair.
        self._every_pair_present = bool(present.all())

    @classmethod
    def build(
        cls, shortest_path_cache: ShortestPathCache, clients: Sequence[float], depot: float
    ) -> EpisodeGeometry:
        """Build this Episode's matrices: columns are ``depot`` then ``clients``.

        Selects columns out of the memoized whole-cache index (:func:`_full_index`)
        — a vectorized slice, not a fresh cache scan, so this is cheap even though
        the underlying ShortestPathCache is large.
        """
        columns = _ordered_unique((depot, *clients))
        full = _full_index(shortest_path_cache)

        row_count = full.average_minutes.shape[0]
        shape = (row_count, len(columns))
        average_minutes = np.zeros(shape, dtype=np.float64)
        length = np.zeros(shape, dtype=np.float64)
        present = np.zeros(shape, dtype=np.bool_)
        for index, client in enumerate(columns):
            full_col = full.col_index.get(client)
            if full_col is None:
                continue  # never cached toward this client: column stays all-absent
            average_minutes[:, index] = full.average_minutes[:, full_col]
            length[:, index] = full.length[:, full_col]
            present[:, index] = full.present[:, full_col]

        return cls(average_minutes, length, present, full.row_index, columns)

    @property
    def columns(self) -> tuple[float, ...]:
        """Column node ids in matrix order: ``depot`` first, then this Episode's Clients."""
        return self._columns

    def average_minutes(self, node: float, client: float) -> float:
        """The cached mean travel minutes from ``node`` to ``client``; ``KeyError`` if absent."""
        row, col = self._locate(node, client)
        return self._average_minutes_list[row][col]

    def length(self, node: float, client: float) -> float:
        """The cached path length from ``node`` to ``client``; ``KeyError`` if absent."""
        row, col = self._locate(node, client)
        return self._length_list[row][col]

    def average_minutes_row(self, node: float) -> NDArray[np.float64]:
        """Travel minutes from ``node`` to every column, in :attr:`columns` order."""
        row: NDArray[np.float64] = self._average_minutes[self._row(node)]
        return row

    def length_row(self, node: float) -> NDArray[np.float64]:
        """Path length from ``node`` to every column, in :attr:`columns` order."""
        row: NDArray[np.float64] = self._length[self._row(node)]
        return row

    def average_minutes_rows(self, nodes: Sequence[float]) -> NDArray[np.float64]:
        """``[node, column]`` travel minutes: one :meth:`average_minutes_row` per node.

        The whole fleet's travel times in one gather, which is how ticket 05's
        feature extraction reads the geometry — every vehicle against every column,
        once per decision, instead of once per (vehicle, Client) pair.
        """
        rows: NDArray[np.float64] = self._average_minutes[self._rows(nodes)]
        return rows

    def length_rows(self, nodes: Sequence[float]) -> NDArray[np.float64]:
        """``[node, column]`` path lengths: one :meth:`length_row` per node."""
        rows: NDArray[np.float64] = self._length[self._rows(nodes)]
        return rows

    def average_minutes_column(self, client: float) -> NDArray[np.float64]:
        """Travel minutes from every row node to ``client``, in row-build order."""
        return self._average_minutes[:, self._col(client)]

    def length_column(self, client: float) -> NDArray[np.float64]:
        """Path length from every row node to ``client``, in row-build order."""
        return self._length[:, self._col(client)]

    def column_positions(self, clients: Sequence[float]) -> NDArray[np.intp]:
        """Positions of ``clients`` in :attr:`columns`; ``KeyError`` for one not held.

        The index array :meth:`average_minutes_at` reads a Client subset with —
        the unvisited ones, for the Policy's candidate selection (ticket 07).
        Callers that slice the same subset repeatedly should hold on to it.
        """
        col_index = self._col_index
        return np.fromiter(
            (col_index[client] for client in clients), dtype=np.intp, count=len(clients)
        )

    def average_minutes_at(self, node: float, positions: NDArray[np.intp]) -> NDArray[np.float64]:
        """Travel minutes from ``node`` to the columns at ``positions``.

        The vectorized counterpart of the *scalar* :meth:`average_minutes`, not of
        :meth:`average_minutes_row`: it keeps the absent-pair contract, raising
        ``KeyError`` for a pair the cache never priced rather than reading the
        zero those cells hold. On a real (dense) ShortestPathCache that check
        costs nothing — density is settled once at build time.
        """
        row = self._row(node)
        if not self._every_pair_present:
            present = self._present[row].take(positions)
            if not present.all():
                absent = int(np.flatnonzero(~present)[0])
                raise KeyError((node, self._columns[positions[absent]]))
        minutes: NDArray[np.float64] = self._average_minutes[row].take(positions)
        return minutes

    def _row(self, node: float) -> int:
        row = self._row_index.get(node)
        if row is None:
            raise KeyError(node)
        return row

    def _rows(self, nodes: Sequence[float]) -> NDArray[np.intp]:
        return np.fromiter((self._row(node) for node in nodes), dtype=np.intp, count=len(nodes))

    def _col(self, client: float) -> int:
        col = self._col_index.get(client)
        if col is None:
            raise KeyError(client)
        return col

    def _locate(self, node: float, client: float) -> tuple[int, int]:
        row = self._row_index.get(node)
        col = self._col_index.get(client)
        if row is None or col is None or not self._present_list[row][col]:
            raise KeyError((node, client))
        return row, col


def _ordered_unique(values: Sequence[float]) -> tuple[float, ...]:
    seen: set[float] = set()
    ordered: list[float] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)
