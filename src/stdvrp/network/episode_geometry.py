"""EpisodeGeometry: dense per-Episode travel-time/length matrices (ADR-0003).

Simulation-performance ticket 04: the Policy hot path drove 1.13M per-episode
``ShortestPathCache.path_between`` dict lookups (ticket 01's profile, ~17% of
episode time). This facade replaces the *time/length* half of those lookups —
``.average_minutes`` and ``.length`` — with array indexing: one dense
``[node, client-or-depot]`` matrix pair built once per Episode from the
ShortestPathCache and reused for the rest of the Episode's decisions.

Rows cover every node id that has a cached path toward some client — in
practice the road network's full node universe, since the cache is dense over
(every graph node) x (the experiment's client-universe nodes). Columns are
this Episode's Clients plus the depot, so the matrix stays small (tens of MB,
ADR-0003) even though the cache itself is not. Values are the cache's own
floats, copied verbatim — no arithmetic — so scalar reads stay bit-identical
to the ``path_between(...).average_minutes/.length`` calls they replace
(Tier 1, pure representation change).

**Per-cache memoization.** A single pass over the whole ShortestPathCache costs
about as much as the dict lookups it replaces (measured on the mini-fixture
benchmark: rebuilding from scratch every Episode regressed throughput). But the
cache is immutable and shared across every Episode of a run, and its *full*
column universe (every client the cache was ever priced toward) never changes
between Episodes — only which subset of it one Episode's Clients select. So the
one full-cache scan is memoized per ShortestPathCache instance (a
``WeakKeyDictionary``, safe because the cache is never mutated after
construction); each Episode's :meth:`EpisodeGeometry.build` then does a cheap
vectorized column-select out of that shared full index, which is what actually
lands the "~ms per Episode" ADR-0003 target.

**Scalar reads use nested Python lists, not numpy indexing.** Measured (this
repo, mini fixture): a single ``ndarray[i, j]`` scalar read costs about 2x a
nested-list ``list[i][j]`` read — numpy's per-call C-API dispatch overhead,
worse than the dict lookup it replaces once paid one call at a time on the
1e5-calls-per-Episode Policy hot path. So each built matrix keeps its numpy
form (for the vectorized row/column accessors tickets 05/07 build on) *and* a
``.tolist()`` copy (cheap: one bulk vectorized conversion) that
:meth:`average_minutes`/:meth:`length` read from instead.

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
    absent-pair semantics.
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
        row = self._row_index.get(node)
        col = self._col_index.get(client)
        if row is None or col is None or not self._present_list[row][col]:
            raise KeyError((node, client))
        return self._average_minutes_list[row][col]

    def length(self, node: float, client: float) -> float:
        """The cached path length from ``node`` to ``client``; ``KeyError`` if absent."""
        row = self._row_index.get(node)
        col = self._col_index.get(client)
        if row is None or col is None or not self._present_list[row][col]:
            raise KeyError((node, client))
        return self._length_list[row][col]

    def average_minutes_row(self, node: float) -> NDArray[np.float64]:
        """Travel minutes from ``node`` to every column, in :attr:`columns` order."""
        row: NDArray[np.float64] = self._average_minutes[self._row(node)]
        return row

    def length_row(self, node: float) -> NDArray[np.float64]:
        """Path length from ``node`` to every column, in :attr:`columns` order."""
        row: NDArray[np.float64] = self._length[self._row(node)]
        return row

    def average_minutes_column(self, client: float) -> NDArray[np.float64]:
        """Travel minutes from every row node to ``client``, in row-build order."""
        return self._average_minutes[:, self._col(client)]

    def length_column(self, client: float) -> NDArray[np.float64]:
        """Path length from every row node to ``client``, in row-build order."""
        return self._length[:, self._col(client)]

    def _row(self, node: float) -> int:
        row = self._row_index.get(node)
        if row is None:
            raise KeyError(node)
        return row

    def _col(self, client: float) -> int:
        col = self._col_index.get(client)
        if col is None:
            raise KeyError(client)
        return col


def _ordered_unique(values: Sequence[float]) -> tuple[float, ...]:
    seen: set[float] = set()
    ordered: list[float] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)
