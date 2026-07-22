# 05 — Data spine: ExperimentConfig + DataSource → RoadNetwork, TrafficHistory, TravelTimeModel

**What to build:** The first vertical slice of the new package: a config file loads, the CSV DataSource builds RoadNetwork and TrafficHistory, and TravelTimeModel derives interpolated speeds and travel times — verified against legacy-computed values on the fixture.

**Blocked by:** 02, 03.

**Status:** ready-for-agent

- [ ] Frozen, validated ExperimentConfig loads from YAML, covering every former sys.argv parameter plus the formerly hardcoded values (horizon, arc count, seeds, data paths)
- [ ] DataSource interface with the CSV implementation (ADR-0002 seam; database comes later)
- [ ] RoadNetwork, TrafficHistory and TravelTimeModel expose what the simulation needs; spot-checked interpolated speeds and travel times equal the legacy computation on the fixture
- [ ] Entry script loads the fixture config, builds the instance, prints a summary; mypy strict passes on the new interfaces
