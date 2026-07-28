# 03 — Binary world cache (TravelTimeModel + ShortestPathCache)

**What to build:** Cache the loaded world in a fast binary form so repeat runs
(and every parallel worker in ticket 08) load in seconds instead of minutes.
Today each process re-parses 88 speed CSVs through pandas aggregation plus the
907 MB `all_shortest_paths.csv` (21.6s even on the mini fixture).

**Blocked by:** 01.

**Status:** resolved

- [x] A cache layer at the `CsvDataSource`/`Trainer.from_config` boundary:
      first load parses CSVs and writes a binary snapshot (format chosen in
      ticket — npz/pickle/parquet/arrow are all allowed, justify in Comments);
      later loads read the snapshot.
- [x] Invalidation by content: the cache key hashes the source files (or
      size+mtime, justified) and the config fields that shape the world
      (`traffic_days`, `instance_day`, `max_congestion_duration`,
      `horizon_start_minute`); a stale cache is rebuilt, never silently reused.
- [x] The cached world is verified equal to the parsed world (exact equality of
      `travel_data`, `speed_std`, `event_probability`, `successors`, and the
      path cache) by a test on the fixture.
- [x] Tier 1 gate: episode outputs bit-identical to self-golden when running
      from cache.
- [x] Benchmark note in Comments: cold vs warm world-load time, fixture and
      real dataset.

## Comments

### Resolution (2026-07-23)

`src/stdvrp/traffic/world_cache.py` — a caching wrapper around exactly the
`CsvDataSource` → `TravelTimeModel` + `ShortestPathCache` construction that
`Trainer.from_config` (and `scripts/benchmark_episodes.py`'s `load_world`)
already did. `Trainer.from_config` and `benchmark_episodes.load_world` both
gained a `cache_dir: Path | None = None` parameter — `None` (the default)
preserves today's behavior exactly (parse fresh, touch no disk), so every
existing caller and test is unaffected; passing a directory opts in.
`experiments/chengdu/run.py` opts in by default (`--cache-dir` /
`--no-cache`, mirroring `scripts/capture_golden_master.py`'s existing
`--cache-path`/`--no-cache` precedent for the legacy world cache).

**Format: pickle**, not npz/parquet/arrow. `TravelTimeModel`'s derived state
(`travel_data`, `speed_std`, `successors`, `node_coordinates`,
`event_probability`) is Python dicts keyed by arc/minute tuples, plus one
`mean_arc_data` DataFrame — not naturally columnar, and ticket 04 (in
progress alongside this one) is already replacing this dict representation
with numpy geometry matrices, so investing in a columnar format now would
optimize a representation this effort is actively retiring. Pickle
round-trips every value exactly, including `event_probability`'s dict
insertion order (the live `CongestionGenerator` draws one random number per
key in that order — ADR-0001) — nothing there needs re-deriving. No new
dependency; the package already has a working precedent for this exact
pattern in `scripts/capture_golden_master.py`'s legacy world cache
(pickle, size+mtime signature, atomic write, environment-fingerprint
invalidation), which this mirrors closely.

To reconstruct a `TravelTimeModel` from the cache without re-running its
pandas pipeline, it gained one additive classmethod,
`TravelTimeModel._from_cached_state(...)`, that sets its six public
attributes directly (`cls.__new__` + assignment, no `__init__` call). It is
for `world_cache` only, not a general public constructor; every other
`TravelTimeModel` construction path (all existing tests, `Trainer.from_config`
on a cache miss) is untouched.

**Invalidation.** The cache key hashes, per consumed file (`links_file`,
`shortest_paths_file`, and the two speed halves for every `traffic_days`
entry), size + mtime — never content: hashing ~1-2 GB of CSVs before every
load would itself cost real seconds, defeating the "seconds not minutes"
goal (the same tradeoff `capture_golden_master.py`'s legacy cache already
made, there justified as "hashing the ~3 GB of CSVs on every run is not
worth it for research data that only ever changes by regeneration"). The key
also includes the four config fields the ticket names
(`traffic_days`, `instance_day`, `max_congestion_duration`,
`horizon_start_minute`) plus the python/numpy/pandas versions (pickles are
not guaranteed portable across them — confirmed via DeepWiki against
pandas-dev/pandas). Storage is **content-addressed** (one file per cache key,
named by the key's SHA-256 digest) rather than a single evictable slot like
the legacy cache: `Trainer.from_config` callers may point the mini fixture,
the real Chengdu archive, and (ticket 08) several parallel workers at the
same `cache_dir` without evicting each other's entries. Known limitation,
inherited from the same precedent: a same-size in-place edit of a data CSV
within the same mtime tick would go undetected — clear the cache directory
after any deliberate data edit.

**Default location**: `default_cache_dir()` — `STDVRP_WORLD_CACHE_DIR` env var,
else `%LOCALAPPDATA%\stdvrp\world_cache` (or the system temp dir), matching
`capture_golden_master.py`'s `default_cache_path()` convention. Deliberately
*not* derived from `data_dir`: the mini fixture's `data_dir` is a committed
repo path (`tests/fixtures/chengdu_mini`) that test runs sometimes point at a
`tmp_path` copy, so a cache written next to the data could pollute the repo
tree or thrash on every test run; a local, non-OneDrive-synced default avoids
both.

**Tests**: `tests/test_world_cache.py` — a cache miss then hit reproduces the
parsed world exactly (`travel_data`, `speed_std`, `successors`,
`node_coordinates`, `event_probability` including insertion order,
`mean_arc_data`, and the path cache, all on the fixture), `cache_dir=None`
never touches disk, a changed world-shaping config field gets its own cache
entry rather than colliding, and touching a source file's mtime forces a
rebuild rather than serving stale content. `tests/test_world_cache_self_golden.py`
— the Tier-1 gate: the self-golden protocol (`scripts/capture_self_golden.py`)
run against a world reconstructed from a cache hit reproduces
`tests/fixtures/self_golden/mini_fixture.json` bit-for-bit (same
environment guard as `tests/test_self_golden.py`, since numpy's Ziggurat
float draws aren't cross-platform bit-identical).

### Benchmark: cold vs warm world-load time

Reproduced via `world_cache.load_world(config, cache_dir=...)` called twice
against an empty cache directory (cold) then the now-populated one (warm).

```
                    cold        warm       speedup
fixture           23.855s      0.099s        ~241x
real (Chengdu)   1258.744s     36.435s        ~34.5x   (0h20m59s -> 0h00m36s)
```

The real-dataset warm load is not sub-second like the fixture's: unpickling
the ~1.3 GB unpacked path-cache dict (907 MB of CSV becomes a larger Python
object graph — one list of floats plus two floats per cached path, times
every node×client pair) is genuine CPU work proportional to data size, not
I/O. Still, cutting world load from ~21 minutes to 36 seconds removes it as
a per-process cost: re-baselining ticket 01's full-run projection
(`world_load` was 1106.0s, 0.5% of the 59h50m total) with the warm number
saves about 17.8 minutes per process — small next to the final test's 58h36m
(ticket 02's target) but the exact win ticket 08's parallel workers need,
since every worker can now share one warm cache instead of each re-paying
the ~21-minute cold parse.
