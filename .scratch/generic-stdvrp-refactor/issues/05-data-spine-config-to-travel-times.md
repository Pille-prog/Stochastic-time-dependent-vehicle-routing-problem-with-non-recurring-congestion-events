# 05 — Data spine: ExperimentConfig + DataSource → RoadNetwork, TrafficHistory, TravelTimeModel

**What to build:** The first vertical slice of the new package: a config file loads, the CSV DataSource builds RoadNetwork and TrafficHistory, and TravelTimeModel derives interpolated speeds and travel times — verified against legacy-computed values on the fixture.

**Blocked by:** 02, 03.

**Status:** resolved

- [x] Frozen, validated ExperimentConfig loads from YAML, covering every former sys.argv parameter plus the formerly hardcoded values (horizon, arc count, seeds, data paths)
- [x] DataSource interface with the CSV implementation (ADR-0002 seam; database comes later)
- [x] RoadNetwork, TrafficHistory and TravelTimeModel expose what the simulation needs; spot-checked interpolated speeds and travel times equal the legacy computation on the fixture — **and** the whole spine is bit-identical to the unmodified legacy classes on a 44-day characterization world (`tests/test_travel_time_model_vs_legacy.py`)
- [x] Entry script loads the fixture config, builds the instance, prints a summary; mypy strict passes on the new interfaces

## Comments

Resolved 2026-07-21. Delivered: `stdvrp.config.ExperimentConfig` (frozen dataclass, YAML, key/type/range validation, relative `data_dir` resolves against the YAML's folder), `stdvrp.traffic.DataSource`/`CsvDataSource`, `stdvrp.network.RoadNetwork`, `stdvrp.traffic.TrafficHistory`/`TravelTimeModel`, `experiments/chengdu/run.py` + `config.yaml`, `tests/fixtures/chengdu_mini/config.yaml`. `period_start_minute` unified into `stdvrp.traffic.periods` (ticket 03 debt); `make_fixture.py` regeneration verified byte-identical after the change. mypy per-module strict-component flags on `stdvrp.config` and `stdvrp.traffic.datasource` (mypy has no per-module `strict`).

**Characterization pattern (reuse in 06/07):** the legacy hardcodes 44 days, so `test_travel_time_model_vs_legacy.py` builds a tmp world where all 44 day files are copies of the fixture's day 601, `chdir`s there and runs `environment` + `DataCalculations` completely unmodified (read-only import of the monolith). Every derived structure matches bit-for-bit, including the two iteration orders the RNG stream depends on (`successors` lists walked by congestion spreading; `event_probability` key order — one `random.random()` per key per epoch).

**Findings for tickets 06/07:**
1. **Single-day std is NaN**: with `traffic_days: [601]` every `speed_std` value is NaN (pandas std of one observation; the legacy `all_data.dropna()` discarded its result — silent no-op, preserved). On the mini fixture, `random.gauss(speed, nan)` in ticket 07 would produce NaN velocities. Use the 44-copies characterization world (std = 0.0 exactly) or a multi-day gitignored fixture for stochastic-velocity tests.
2. **The three interpolation windows (420-540, 660-840, 960-1080) are data gaps**: real day-601 records nothing inside them; minute filling first repeats the last observed row, then the blend overwrites those filled speeds. The std lookup's "odd" endpoint keys (418/542, 658/842, 958/1082) are precisely the last/first *observed* minutes around each gap. One genuine quirk preserved: in the 420-540 window the interpolated std is computed and discarded, the raw (gap-filled) std is stored instead.
3. **Live congestion path confirmed**: `model.transition_function` calls `create_random_unexpected_event_with_probability_and_2_nodes` (legacy line 5855), which iterates `probability_of_event` in dict order — hence the order assertions.
4. **Not ported (dead in the live path)**: the 60/90/120-minute aggregates and the interval `Period` label column in `read_all_data`; `df_30` in `get_mean_of_all_intervals`; `environment.travel_time_dict` (built by `preprocess_data_average`, which main() calls, but nothing live consumes it); `environment.node_list` (`shortest_path_memory` assigns then immediately deletes it — the cache is purely CSV-loaded, relevant to ticket 06).
5. **Deliberate config-driven generalizations** (identical to legacy at the golden-master settings): event probability divides by `len(traffic_days)` where the legacy hardcodes 44; the `>= 300` data filter takes `horizon_start_minute`.
6. Dict keys are numpy scalars (`np.int64`); they hash/compare equal to Python ints. The legacy floats congested-arc keys (`(float(ns), float(ne))`) — ticket 07 must keep that when matching against `travel_data` keys.
7. Config demand values for the fixture (`client_universe_*`, `mean_number_clients`) are provisional until ticket 06 makes ClientGenerator config-driven; `experiments/chengdu/config.yaml` training values are representative — ticket 04's capture pins the exact argv combination.
