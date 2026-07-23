# 06 — Demand and shortest paths

**What to build:** ClientGenerator and ShortestPathCache ported: with a fixed seed, the generated Clients (locations, time windows) and the vehicle count are byte-identical to the legacy script on the fixture.

**Blocked by:** 05.

**Status:** resolved

- [x] Client generation reproduces the legacy exactly for fixed seeds — same global-RNG consumption order (ADR-0001)
- [x] ShortestPathCache returns paths and costs equal to the legacy cache
- [x] Comparison tests against captured legacy outputs cover both

## Comments

Resolved 2026-07-22. Delivered: `stdvrp.demand.ClientGenerator`/`Client`/`EpisodeDemand` (frozen dataclasses; `generate(seed)` reseeds the global `random` module and consumes it in exact legacy order: one gauss, floor, one sample over the node range, one randint per client), `stdvrp.network.ShortestPathCache`/`ShortestPath` (purely CSV-loaded like the legacy `shortest_path_memory`, float-parsed path node ids preserved), `DataSource.load_shortest_path_cache()` on the ADR-0002 seam (+ `CsvDataSource.from_config`), three new `ExperimentConfig` demand fields (`client_count_stddev`, `min_number_clients`, `clients_per_vehicle`) replacing the legacy hardcodes, fixture config demand values made real (universe 44 nodes, gauss(20, 4), floor 8, ratio 4), `EpisodeDemand` added to the CONTEXT.md glossary, and a demand + cache summary in `experiments/chengdu/run.py`.

**Characterization venue:** like ticket 05, the comparison runs the unmodified legacy classes in-process (`tests/test_demand_and_paths_vs_legacy.py`) rather than against stored captures — client generation needs no data files at all, so 21 seeds (0, 1000-1009, 100000-100009) × both ratio-table rows run in CI, each also asserting the global stream ends at the same position; the cache comparison loads the fixture CSV through both loaders and requires bit-identical mappings including key order. These tests need the monolith in-tree, i.e. they retire with ticket 14. Ticket 04's golden master remains the stored-capture guard.

**Deliberate config-driven generalization** (mirrors ticket 05 finding 5): the vehicle-ratio table `{150 mean: 28, 250 mean: 29}` — with which the legacy crashes (`UnboundLocalError`) for any other mean — became the free `clients_per_vehicle` field, and the gauss stddev/floor/node range became config values. Identical to legacy at the legacy's parameter pairs (proven by the comparison tests); any other pairing is a new, deliberately-allowed experiment, so the mean↔ratio coupling is documented in `experiments/chengdu/config.yaml` rather than validated.

**Findings for tickets 07/08:**
1. `np.random.seed(seed)` is NOT ClientGenerator's job: the legacy training loop calls it right after `client_generator_function` — the episode runner must reproduce that call (and its position in the seed sequence).
2. The live legacy path indexes `spm.shortest_paths[(node, client)]` positionally (`[0]` path, `[1]` avg minutes, `[2]` length) and never iterates the dict; `ShortestPath` is a NamedTuple so ported code may keep either style, and a missing pair raises KeyError exactly like the legacy indexing.
3. Path node ids come back as floats (legacy `map(float, ...)` parse); they hash/compare equal to int node ids, and ticket 05 finding 6 already requires float congested-arc keys — keep both quirks aligned when matching against `travel_data`.
4. Legacy `ClientGenerator` members NOT ported (dead in the live path): `clients_list_2` deepcopy, `write_clients_to_file`, `assign_clients_to_hyper_routes`, the unused `random_depot` field. The depot stays an episode-runner concern (`random_depot = 0` in the training loop).
