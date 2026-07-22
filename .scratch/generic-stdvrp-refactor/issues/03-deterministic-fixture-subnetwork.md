# 03 — Deterministic fixture sub-network

**What to build:** A small committed sub-network (tens of nodes) extracted reproducibly from the real Chengdu data, so every test in every later ticket runs in seconds on any machine, including CI, without the full dataset.

**Blocked by:** 01. **Also blocked on the user**: the Chengdu CSVs' location and size are still unknown (asked three times during grilling/spec) — extraction cannot start without them.

**Status:** resolved

- [x] Extraction script produces the fixture deterministically from the full data, with the command documented (`uv run python scripts/make_fixture.py`; byte-identical on regeneration, verified)
- [x] Fixture (links subset + matching morning/afternoon speed subsets) committed under the test fixtures directory, total well under 1 MB (436 KB: 45 nodes, 116 arcs, day 601, horizon periods only)
- [x] Fixture network is strongly connected enough for routing, includes the depot node (node 0), and supports at least a handful of Clients (strong connectivity asserted by test)
- [x] The unmodified legacy script can consume the fixture files — **amended**: format/schema compatibility guaranteed (identical columns, names, Period encoding, plus the discovered 4th file `all_shortest_paths.csv`); an end-to-end legacy *run* on a sub-megabyte fixture is impossible (see comments) and moved to ticket 04 on full local data per the ADR-0001 addendum

## Comments

Resolved 2026-07-21. Delivered: `scripts/make_fixture.py` (deterministic, no RNG anywhere, parameterizable size/days/output), `tests/fixtures/chengdu_mini/` (6 files incl. `node_map.csv` provenance and README), `tests/test_fixture.py` (10 structural property tests the rest of the suite may rely on: schemas, contiguous ids 0..44 with depot 0, strong connectivity, full period×link speed coverage, all-pairs paths that only walk real arcs, size budget, node-map bijection).

**Findings for tickets 04/05 (important):**
1. The legacy reads **~90 files, not 3**: `environment.process_all_data()` aggregates `speed[601..630]_[0/1].csv` + `speed[701..714]_[0/1].csv` (mean/std per Period×Link), and `shortest_path_memory.__init__` hardcodes a 4th path, `all_shortest_paths.csv` (907 MB, all 1902² pairs, format `Node,Client,ShortestPath(a->b->c),AverageTime,Length`). `DataCalculations.read_all_data()` likely re-reads the day files (verify when porting).
2. **All legacy paths are relative** → ticket 04's shim can be `cwd`-redirection with zero code edits, IF run on the full dataset. On any small world the legacy breaks structurally: `ClientGenerator` samples clients from hardcoded `range(1, 1900)`, floors `number_clients` at 60, and its vehicle-ratio table raises `UnboundLocalError` unless `mean_number_clients ∈ {150, 250}`.
3. Real data measured: 1,902 nodes (contiguous 0–1901), 5,943 links, 150 two-minute periods per speed file (03:00–13:00 / 13:00–23:00). Speeds are km/h at source; the legacy divides by 60 (km/min) and works in km/minutes.
4. Golden-master venue changed accordingly — ADR-0001 addendum. `make_fixture.py --target-nodes 260 --out data/fixture_large` can produce a gitignored mid-size world if ticket 04 prefers it over full data (but see the range(1,1900) constraint: full data is the only zero-edit option).
5. Fixture `all_shortest_paths.csv` is recomputed on the sub-network (day-601 mean-speed travel times), guaranteeing path arcs exist in the mini network; duplicate-arc semantics follow legacy `read_network` (last row wins).
