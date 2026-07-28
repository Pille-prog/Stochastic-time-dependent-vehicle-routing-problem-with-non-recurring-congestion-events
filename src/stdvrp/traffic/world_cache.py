"""WorldCache: a binary snapshot of the loaded world (ticket 03).

Sits at the ``CsvDataSource`` / ``Trainer.from_config`` boundary: the first load
for a given (data files, world-shaping config) parses the 88 speed CSVs through
the ``TravelTimeModel`` pandas pipeline and the 907 MB ``all_shortest_paths.csv``
through :meth:`ShortestPathCache.from_csv`, then pickles the *derived* state
(``TravelTimeModel``'s six public attributes plus the path-cache mapping) to
disk; later loads with a matching cache key read the pickle instead, skipping
CSV parsing and the aggregation pipeline entirely.

**Format: pickle.** ``TravelTimeModel``'s derived state is dicts keyed by arc/
minute tuples plus one DataFrame — not naturally columnar, and ticket 04 is
about to replace this representation with numpy geometry matrices anyway, so
investing in a columnar format (parquet/arrow) now would optimize a
representation this effort is already retiring. Pickle round-trips every value
(including dict insertion order, which the live ``CongestionGenerator``
depends on for ``event_probability``) exactly, needs no new dependency, and the
package already has a working precedent for this exact pattern
(``scripts/capture_golden_master.py``'s legacy world cache).

**Invalidation.** The cache key hashes, per consumed file, size + mtime (never
content: hashing ~1-2 GB of CSVs before every load would itself cost real
seconds, defeating the "seconds not minutes" goal — the same tradeoff the
legacy world cache already made), plus the config fields that shape the world
(``traffic_days``, ``instance_day``, ``max_congestion_duration``,
``horizon_start_minute``) and the python/numpy/pandas versions (pickles are not
guaranteed portable across them). Known limitation, inherited from the same
precedent: a same-size in-place edit of a data CSV within the same mtime second
would go undetected; after any deliberate data edit, clear the cache directory.

**Storage.** One file per cache key, named by the key's digest, under
``cache_dir`` (default: :func:`default_cache_dir`, a local, non-OneDrive-synced
directory — the pickle can reach hundreds of MB). Content-addressed rather than
a single evictable slot (unlike the legacy world cache, which only ever serves
one script's one world) because callers may point multiple distinct worlds
(the committed mini fixture, the real Chengdu archive, ticket 08's parallel
workers) at the same cache directory without evicting each other. Writes are
atomic (temp file + replace) so an interrupted write never leaves a torn cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stdvrp.config import ExperimentConfig
from stdvrp.network import ShortestPathCache
from stdvrp.traffic.datasource import CsvDataSource
from stdvrp.traffic.travel_time_model import TravelTimeModel

CACHE_FORMAT = 1


@dataclass(frozen=True, slots=True)
class LoadedWorld:
    """The two objects a binary-cached world provides."""

    travel_time_model: TravelTimeModel
    shortest_path_cache: ShortestPathCache


def default_cache_dir() -> Path:
    """A local, non-synced location (override with ``STDVRP_WORLD_CACHE_DIR``)."""
    env = os.environ.get("STDVRP_WORLD_CACHE_DIR")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return Path(base) / "stdvrp" / "world_cache"


def load_world(config: ExperimentConfig, *, cache_dir: Path | None = None) -> LoadedWorld:
    """Return the world for ``config``, from ``cache_dir`` on a key hit, else built fresh.

    ``cache_dir=None`` (the default) disables caching entirely: every call parses
    the CSVs, exactly as before this ticket. Passing a directory is the opt-in.
    """
    source = CsvDataSource.from_config(config)
    key = _cache_key(config, source)
    cache_path = None if cache_dir is None else cache_dir / f"{_key_digest(key)}.pkl"

    if cache_path is not None:
        cached = _read_cache(cache_path, key)
        if cached is not None:
            return cached

    travel_time_model = TravelTimeModel(
        source.load_road_network(),
        source.load_traffic_history(),
        config.max_congestion_duration,
        horizon_start_minute=config.horizon_start_minute,
    )
    world = LoadedWorld(
        travel_time_model=travel_time_model,
        shortest_path_cache=source.load_shortest_path_cache(),
    )

    if cache_path is not None:
        _write_cache(cache_path, key, world)
    return world


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _cache_key(config: ExperimentConfig, source: CsvDataSource) -> dict[str, Any]:
    """Everything the built world depends on; any change invalidates the cache."""
    return {
        "cache_format": CACHE_FORMAT,
        "files": {str(path): _file_signature(path) for path in sorted(source.consumed_files())},
        "world_params": {
            "traffic_days": list(config.traffic_days),
            "instance_day": config.instance_day,
            "max_congestion_duration": config.max_congestion_duration,
            "horizon_start_minute": config.horizon_start_minute,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def _key_digest(key: dict[str, Any]) -> str:
    encoded = json.dumps(key, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_cache(path: Path, key: dict[str, Any]) -> LoadedWorld | None:
    """Return the cached world on an exact key match; ``None`` on miss or corruption."""
    try:
        with path.open("rb") as file:
            payload = pickle.load(file)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError):
        return None
    if not isinstance(payload, dict) or payload.get("key") != key:
        return None
    state = payload.get("state")
    if state is None:
        return None
    travel_time_model = TravelTimeModel._from_cached_state(state["travel_time_model"])
    shortest_path_cache = ShortestPathCache(state["shortest_path_cache"])
    return LoadedWorld(travel_time_model, shortest_path_cache)


def _write_cache(path: Path, key: dict[str, Any], world: LoadedWorld) -> None:
    """Atomic write (tmp + replace) so an interrupted run never leaves a torn cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "travel_time_model": world.travel_time_model._cached_state(),
        "shortest_path_cache": world.shortest_path_cache.as_dict(),
    }
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as file:
        pickle.dump({"key": key, "state": state}, file, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
