# 03 — Binary world cache (TravelTimeModel + ShortestPathCache)

**What to build:** Cache the loaded world in a fast binary form so repeat runs
(and every parallel worker in ticket 08) load in seconds instead of minutes.
Today each process re-parses 88 speed CSVs through pandas aggregation plus the
907 MB `all_shortest_paths.csv` (21.6s even on the mini fixture).

**Blocked by:** 01.

**Status:** open

- [ ] A cache layer at the `CsvDataSource`/`Trainer.from_config` boundary:
      first load parses CSVs and writes a binary snapshot (format chosen in
      ticket — npz/pickle/parquet/arrow are all allowed, justify in Comments);
      later loads read the snapshot.
- [ ] Invalidation by content: the cache key hashes the source files (or
      size+mtime, justified) and the config fields that shape the world
      (`traffic_days`, `instance_day`, `max_congestion_duration`,
      `horizon_start_minute`); a stale cache is rebuilt, never silently reused.
- [ ] The cached world is verified equal to the parsed world (exact equality of
      `travel_data`, `speed_std`, `event_probability`, `successors`, and the
      path cache) by a test on the fixture.
- [ ] Tier 1 gate: episode outputs bit-identical to self-golden when running
      from cache.
- [ ] Benchmark note in Comments: cold vs warm world-load time, fixture and
      real dataset.
