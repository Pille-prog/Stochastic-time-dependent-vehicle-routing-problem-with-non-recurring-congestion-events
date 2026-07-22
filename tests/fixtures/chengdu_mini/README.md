# Chengdu mini fixture

A 45-node, 116-arc sub-network extracted deterministically from the real Chengdu dataset (day 601, periods from 08:00 onward — mirroring the legacy `Minute_start >= 300` filter, with no upper cut because trips and interpolation read past the 780 Horizon end), with node ids renumbered to 0–44 (the depot keeps id 0; `node_map.csv` maps back to real Chengdu ids). File names and schemas are byte-compatible with the real dataset; speeds are rounded to 3 decimals.

**Purpose**: fast, committed test data for the `stdvrp` package (which is configuration-driven and happy with any network size). It is *not* sufficient to run the legacy monolith end-to-end — the legacy hardcodes a 1,900-node client universe, a 60-client minimum and a 44-day speed-file aggregation (see ticket 03 comments and ADR-0001).

**Regenerate** (requires the real dataset in the folder above the repo, see `data/README.md`):

```
uv run python scripts/make_fixture.py
```

Larger gitignored variants for local experiments: `--target-nodes 260 --out data/fixture_large --days 601 602`.

`all_shortest_paths.csv` is recomputed on the sub-network (arc weight = length_km / (mean day-601 speed / 60)) so that every path only walks arcs that exist here. **Its values are self-consistent but are not the legacy file's values** — the real `all_shortest_paths.csv` was precomputed from 44-day aggregated speeds; do not assume equivalence when comparing against legacy outputs. `tests/test_fixture.py` asserts every structural property the rest of the suite relies on.
